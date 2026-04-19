#!/usr/bin/env python3
"""
Step 03: Comprehensive threshold & strategy sweep on OOF data.
Finds the BEST combination of fusion strategy + thresholds + post-processing.
"""
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from scipy import ndimage
from pathlib import Path
from config import (
    TRAIN_TILES, EKL_OOF, LUIS_V2_OOF, ARTIFACTS, tile_region
)


def load_all_oof():
    """Load OOF predictions + labels for all tiles."""
    tiles_data = {}
    
    for tile in TRAIN_TILES:
        region = tile_region(tile)
        
        # Luis v2 OOF
        luis_path = LUIS_V2_OOF / f"{tile}_oof.npz"
        if not luis_path.exists():
            continue
        luis_data = np.load(luis_path)
        labels = luis_data['labels']
        p_ens_luis = luis_data['p_ens']
        p_lgbm_luis = luis_data['p_lgbm']
        p_unet_luis = luis_data['p_unet']
        
        # Eklavya stack OOF
        ekl_stack_path = EKL_OOF / "stack" / f"{tile}.npy"
        ekl_lgbm_path = EKL_OOF / "lgbm" / f"{tile}.npy"
        
        if not ekl_stack_path.exists():
            continue
        
        ekl_stack = np.load(ekl_stack_path).astype(np.float32).ravel()
        ekl_lgbm = np.load(ekl_lgbm_path).astype(np.float32).ravel() if ekl_lgbm_path.exists() else None
        
        n = min(len(ekl_stack), len(labels))
        
        tiles_data[tile] = {
            'ekl_stack': ekl_stack[:n],
            'ekl_lgbm': ekl_lgbm[:n] if ekl_lgbm is not None else None,
            'luis_ens': p_ens_luis[:n],
            'luis_lgbm': p_lgbm_luis[:n],
            'luis_unet': p_unet_luis[:n],
            'labels': labels[:n],
            'region': region,
        }
    
    return tiles_data


def iou_score(pred, true):
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    return iou, recall, prec, tp, fp, fn


def sweep_all():
    print("Loading OOF data...")
    tiles_data = load_all_oof()
    print(f"Loaded {len(tiles_data)} tiles\n")
    
    # Define fusion strategies to test
    strategies = {}
    
    for w_ekl in np.arange(0.50, 0.96, 0.05):
        w_luis = 1.0 - w_ekl
        strategies[f"wavg_{int(w_ekl*100)}_{int(w_luis*100)}"] = lambda e, l, w=w_ekl: w * e + (1-w) * l
    
    # Geometric mean variants
    strategies["geomean"] = lambda e, l: np.sqrt(np.clip(e * l, 1e-8, 1))
    strategies["geomean_weighted"] = lambda e, l: np.clip(e, 1e-8, 1)**0.7 * np.clip(l, 1e-8, 1)**0.3
    
    # Max / min
    strategies["max"] = lambda e, l: np.maximum(e, l)
    strategies["ekl_only"] = lambda e, l: e
    
    # Agreement boost
    def agreement_boost(e, l):
        avg = 0.7 * e + 0.3 * l
        both_high = (e > 0.5) & (l > 0.3)
        avg[both_high] = np.clip(avg[both_high] * 1.15, 0, 1)
        both_low = (e < 0.3) & (l < 0.3)
        avg[both_low] = avg[both_low] * 0.7
        return avg
    strategies["agreement_boost"] = agreement_boost
    
    # Power mean variants
    for p in [0.5, 2, 3]:
        strategies[f"power_mean_p{p}"] = lambda e, l, pw=p: np.clip(
            ((0.7 * np.clip(e, 1e-8, 1)**pw + 0.3 * np.clip(l, 1e-8, 1)**pw) ** (1/pw)), 0, 1
        )
    
    results = []
    
    # Threshold range
    thresholds = np.arange(0.15, 0.85, 0.02)
    
    print(f"Testing {len(strategies)} strategies × {len(thresholds)} thresholds")
    print(f"{'Strategy':<30s} {'Best_t':>7s} {'IoU':>7s} {'Recall':>7s} {'Prec':>7s}")
    print("-" * 65)
    
    for strat_name, strat_fn in strategies.items():
        # Compute fused probabilities for all tiles
        all_fused = []
        all_labels = []
        all_regions = []
        
        for tile, data in tiles_data.items():
            mask = data['labels'] != -1
            if mask.sum() == 0:
                continue
            
            ekl = data['ekl_stack'][mask]
            luis = data['luis_ens'][mask]
            fused = strat_fn(ekl, luis)
            
            all_fused.append(fused)
            all_labels.append(data['labels'][mask].astype(np.uint8))
            all_regions.append(np.full(mask.sum(), data['region']))
        
        fused_all = np.concatenate(all_fused)
        labels_all = np.concatenate(all_labels)
        regions_all = np.concatenate(all_regions)
        
        # Global sweep
        best_t, best_iou = 0, 0
        best_rec, best_prec = 0, 0
        
        for t in thresholds:
            pred = (fused_all >= t).astype(np.uint8)
            iou, rec, prec, tp, fp, fn = iou_score(pred, labels_all)
            if iou > best_iou:
                best_t, best_iou, best_rec, best_prec = t, iou, rec, prec
        
        # Per-region sweep
        region_best = {}
        for reg in ["amazon", "asia"]:
            rmask = regions_all == reg
            if rmask.sum() == 0:
                continue
            bt, bi = 0, 0
            for t in thresholds:
                pred = (fused_all[rmask] >= t).astype(np.uint8)
                iou = iou_score(pred, labels_all[rmask])[0]
                if iou > bi:
                    bt, bi = t, iou
            region_best[reg] = (bt, bi)
        
        # Combined per-region score
        combined_pred = np.zeros(len(labels_all), dtype=np.uint8)
        for reg, (bt, _) in region_best.items():
            rmask = regions_all == reg
            combined_pred[rmask] = (fused_all[rmask] >= bt).astype(np.uint8)
        
        combined_iou = iou_score(combined_pred, labels_all)[0]
        
        print(f"  {strat_name:<28s} t={best_t:.2f} IoU={best_iou:.4f} R={best_rec:.4f} P={best_prec:.4f} | per-reg IoU={combined_iou:.4f}")
        
        results.append({
            "strategy": strat_name,
            "global_threshold": float(best_t),
            "global_iou": float(best_iou),
            "global_recall": float(best_rec),
            "global_precision": float(best_prec),
            "per_region_thresholds": {k: float(v[0]) for k, v in region_best.items()},
            "per_region_ious": {k: float(v[1]) for k, v in region_best.items()},
            "per_region_combined_iou": float(combined_iou),
        })
    
    # Sort by per-region combined IoU
    results.sort(key=lambda x: -x["per_region_combined_iou"])
    
    print("\n" + "=" * 60)
    print("TOP 10 STRATEGIES (by per-region IoU):")
    print("=" * 60)
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. {r['strategy']:<28s} IoU={r['per_region_combined_iou']:.4f} "
              f"(global={r['global_iou']:.4f}) "
              f"| Amazon t={r['per_region_thresholds'].get('amazon', 0):.2f} "
              f"IoU={r['per_region_ious'].get('amazon', 0):.4f} "
              f"| Asia t={r['per_region_thresholds'].get('asia', 0):.2f} "
              f"IoU={r['per_region_ious'].get('asia', 0):.4f}")
    
    # Save results
    out_dir = ARTIFACTS / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save best strategy config
    best = results[0]
    config = {
        "strategy": best["strategy"],
        "per_region_thresholds": best["per_region_thresholds"],
        "africa_threshold": round(np.mean(list(best["per_region_thresholds"].values())), 2),
        "expected_oof_iou": best["per_region_combined_iou"],
    }
    config["per_region_thresholds"]["africa"] = config["africa_threshold"]
    
    with open(out_dir / "best_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nBest config saved to {out_dir / 'best_config.json'}")
    print(f"Best strategy: {config['strategy']}")
    print(f"Thresholds: {config['per_region_thresholds']}")
    print(f"Expected OOF IoU: {config['expected_oof_iou']:.4f}")


if __name__ == "__main__":
    sweep_all()

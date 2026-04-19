#!/usr/bin/env python3
"""
Step 01: Train a meta-stacker on OOF predictions from Eklavya (5 models) + Luis v2 (LGBM+UNet).

Uses all 7 base model OOF probabilities as features to train a LightGBM meta-learner,
then applies it to test tiles using the saved prob maps.
"""
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
from pathlib import Path
from config import (
    TRAIN_TILES, TEST_TILES, EKL_OOF, LUIS_V2_OOF,
    EKL_SUBMISSION, LUIS_V4_SUBMISSION, SUBMISSION, ARTIFACTS, tile_region
)

# ── Load OOF predictions for all training tiles ──
def load_oof_data():
    """Load OOF probabilities from all models + consensus labels."""
    X_list, y_list, regions_list, tiles_list = [], [], [], []
    
    ekl_models = ["lgbm", "tcn", "unet_eb3", "unet_mit_b1", "unet_r34", "stack"]
    
    for tile in TRAIN_TILES:
        region = tile_region(tile)
        
        # Load Luis v2 OOF
        luis_path = LUIS_V2_OOF / f"{tile}_oof.npz"
        if not luis_path.exists():
            print(f"  [SKIP] {tile}: no Luis v2 OOF")
            continue
        luis_data = np.load(luis_path)
        labels = luis_data['labels']  # -1=unknown, 0=neg, 1=pos
        p_lgbm_luis = luis_data['p_lgbm']
        p_unet_luis = luis_data['p_unet']
        p_ens_luis = luis_data['p_ens']
        n_pixels = len(labels)
        
        # Load Eklavya OOF (per-model)
        ekl_preds = {}
        for model_name in ekl_models:
            ekl_path = EKL_OOF / model_name / f"{tile}.npy"
            if ekl_path.exists():
                arr = np.load(ekl_path).astype(np.float32).ravel()
                # Pad/trim to match
                if len(arr) < n_pixels:
                    arr = np.pad(arr, (0, n_pixels - len(arr)), constant_values=0)
                elif len(arr) > n_pixels:
                    arr = arr[:n_pixels]
                ekl_preds[model_name] = arr
            else:
                ekl_preds[model_name] = np.zeros(n_pixels, dtype=np.float32)
        
        # Only use consensus pixels (label != -1)
        mask = labels != -1
        if mask.sum() == 0:
            continue
        
        # Build feature matrix: [ekl_lgbm, ekl_tcn, ekl_unet_eb3, ekl_unet_mit, ekl_unet_r34, ekl_stack, luis_lgbm, luis_unet, luis_ens]
        features = np.column_stack([
            ekl_preds["lgbm"][mask],
            ekl_preds["tcn"][mask],
            ekl_preds["unet_eb3"][mask],
            ekl_preds["unet_mit_b1"][mask],
            ekl_preds["unet_r34"][mask],
            ekl_preds["stack"][mask],
            p_lgbm_luis[mask],
            p_unet_luis[mask],
            p_ens_luis[mask],
        ])
        
        X_list.append(features)
        y_list.append(labels[mask].astype(np.uint8))
        regions_list.append(np.full(mask.sum(), region, dtype='<U10'))
        tiles_list.append(np.full(mask.sum(), tile, dtype='<U20'))
        
        n_pos = labels[mask].sum()
        print(f"  {tile} ({region}): {mask.sum():>8,} consensus pixels, {n_pos:>7,} positive")
    
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    regions = np.concatenate(regions_list)
    tiles = np.concatenate(tiles_list)
    
    return X, y, regions, tiles

FEATURE_NAMES = [
    "ekl_lgbm", "ekl_tcn", "ekl_unet_eb3", "ekl_unet_mit_b1", "ekl_unet_r34",
    "ekl_stack", "luis_lgbm", "luis_unet", "luis_ens",
]


def train_stacker(X, y, regions):
    """Train LORO meta-stacker using LightGBM."""
    import lightgbm as lgb
    
    amazon_mask = regions == "amazon"
    asia_mask = regions == "asia"
    
    # LORO: train on Amazon, validate on Asia, and vice versa
    oof_probs = np.zeros(len(y), dtype=np.float32)
    models = {}
    
    for fold_name, train_mask, val_mask in [
        ("loro_amazon", asia_mask, amazon_mask),   # train on Asia, predict Amazon
        ("loro_asia", amazon_mask, asia_mask),      # train on Amazon, predict Asia
    ]:
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        # Subsample negative class for training (keep all positives)
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        
        # Keep all positives + 3x negatives (for better calibration)
        n_neg = min(len(neg_idx), 3 * len(pos_idx))
        rng = np.random.RandomState(42)
        neg_sample = rng.choice(neg_idx, n_neg, replace=False)
        sample_idx = np.concatenate([pos_idx, neg_sample])
        rng.shuffle(sample_idx)
        
        print(f"\n  {fold_name}: train={len(sample_idx):,} (pos={len(pos_idx):,}), val={len(X_val):,}")
        
        dtrain = lgb.Dataset(X_train[sample_idx], y_train[sample_idx], feature_name=FEATURE_NAMES)
        dval = lgb.Dataset(X_val, y_val, feature_name=FEATURE_NAMES, reference=dtrain)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": 42,
        }
        
        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)],
        )
        
        oof_probs[val_mask] = model.predict(X_val)
        models[fold_name] = model
        
        # Per-fold metrics
        from sklearn.metrics import f1_score
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            pred = (oof_probs[val_mask] >= t).astype(int)
            tp = ((pred == 1) & (y_val == 1)).sum()
            fp = ((pred == 1) & (y_val == 0)).sum()
            fn = ((pred == 0) & (y_val == 1)).sum()
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0
            print(f"    t={t:.1f}: IoU={iou:.4f} tp={tp:,} fp={fp:,} fn={fn:,}")
    
    # Train full model (all data)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_neg = min(len(neg_idx), 3 * len(pos_idx))
    rng = np.random.RandomState(42)
    neg_sample = rng.choice(neg_idx, n_neg, replace=False)
    sample_idx = np.concatenate([pos_idx, neg_sample])
    rng.shuffle(sample_idx)
    
    dtrain_full = lgb.Dataset(X[sample_idx], y[sample_idx], feature_name=FEATURE_NAMES)
    model_full = lgb.train(params, dtrain_full, num_boost_round=300)
    models["full"] = model_full
    
    return models, oof_probs


def tune_thresholds(oof_probs, y, regions):
    """Find optimal per-region thresholds."""
    thresholds = {}
    
    for region_name in ["amazon", "asia"]:
        mask = regions == region_name
        if mask.sum() == 0:
            continue
        
        probs = oof_probs[mask]
        labels = y[mask]
        
        best_t, best_iou = 0.5, 0
        for t in np.arange(0.10, 0.90, 0.01):
            pred = (probs >= t).astype(np.uint8)
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0
            if iou > best_iou:
                best_t, best_iou = t, iou
        
        thresholds[region_name] = float(best_t)
        print(f"  {region_name}: optimal threshold={best_t:.2f}, IoU={best_iou:.4f}")
    
    # Africa: use lower of the two (more conservative for unknown region)
    thresholds["africa"] = round(np.mean([thresholds.get("amazon", 0.5), thresholds.get("asia", 0.5)]), 2)
    print(f"  africa (median): threshold={thresholds['africa']:.2f}")
    
    return thresholds


def main():
    print("=" * 60)
    print("STEP 1: Load OOF predictions from all models")
    print("=" * 60)
    X, y, regions, tiles = load_oof_data()
    print(f"\nTotal: {len(y):,} pixels, {y.sum():,} positive ({100*y.mean():.1f}%)")
    
    # Also evaluate simple fusion baselines on OOF
    print("\n" + "=" * 60)
    print("Baseline comparisons (OOF):")
    print("=" * 60)
    
    for name, probs in [
        ("Eklavya stack only", X[:, 5]),  # ekl_stack
        ("Luis v2 ensemble", X[:, 8]),     # luis_ens
        ("50/50 avg", 0.5 * X[:, 5] + 0.5 * X[:, 8]),
        ("70/30 ekl/luis", 0.7 * X[:, 5] + 0.3 * X[:, 8]),
    ]:
        best_t, best_iou = 0.5, 0
        for t in np.arange(0.10, 0.95, 0.02):
            pred = (probs >= t).astype(np.uint8)
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0
            if iou > best_iou:
                best_t, best_iou = t, iou
        print(f"  {name:<25s}: best_t={best_t:.2f}, IoU={best_iou:.4f}")
    
    print("\n" + "=" * 60)
    print("STEP 2: Train LightGBM meta-stacker")
    print("=" * 60)
    models, oof_probs = train_stacker(X, y, regions)
    
    # Feature importance
    print("\nFeature importance (full model):")
    imp = models["full"].feature_importance(importance_type='gain')
    for name, score in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1]):
        print(f"  {name:<25s}: {score:.1f}")
    
    print("\n" + "=" * 60)
    print("STEP 3: Evaluate meta-stacker on OOF")
    print("=" * 60)
    best_t_oof, best_iou_oof = 0.5, 0
    for t in np.arange(0.10, 0.90, 0.01):
        pred = (oof_probs >= t).astype(np.uint8)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0
        if iou > best_iou_oof:
            best_t_oof, best_iou_oof = t, iou
    print(f"  Meta-stacker OOF: best_t={best_t_oof:.2f}, IoU={best_iou_oof:.4f}")
    
    print("\n" + "=" * 60)
    print("STEP 4: Tune per-region thresholds")
    print("=" * 60)
    thresholds = tune_thresholds(oof_probs, y, regions)
    
    # Save models and thresholds
    out_dir = ARTIFACTS / "stacker"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        model.save_model(str(out_dir / f"stacker_{name}.txt"))
    
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    
    with open(out_dir / "oof_summary.json", "w") as f:
        json.dump({
            "oof_iou": float(best_iou_oof),
            "oof_threshold": float(best_t_oof),
            "per_region_thresholds": thresholds,
            "n_pixels": int(len(y)),
            "n_positive": int(y.sum()),
        }, f, indent=2)
    
    print(f"\nSaved stacker models to {out_dir}")
    print("Done!")


if __name__ == "__main__":
    main()

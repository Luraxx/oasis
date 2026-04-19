#!/usr/bin/env python3
"""
FULL OOF Union-IoU scorer — replicates the competition evaluator.

Score = Union IoU = sum(TP_m²) / sum(TP_m² + FP_m² + FN_m²) across all tiles.

For each tile:
  1. Load GT consensus raster → vectorize → union geometry
  2. Load OOF prob maps → fuse → threshold → postprocess → vectorize → union geometry
  3. Compute intersection / union area in UTM

This lets us test ANY strategy without submitting.
"""
import sys, json, warnings
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from pathlib import Path
from scipy import ndimage
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd

from config import TRAIN_TILES, EKL_OOF, LUIS_V2_OOF, DATA, tile_region

# ─── GT loading ───────────────────────────────────────────────

def load_gt_binary(tile):
    """Load ground truth consensus binary mask (1002x1002)."""
    # Try oasis bundle.npz first
    bundle = Path(f"/shared-docker/oasis/data/cache/train/{tile}/bundle.npz")
    if bundle.exists():
        d = np.load(bundle)
        if 'consensus_pos' in d:
            return d['consensus_pos'].astype(np.uint8)
        return (d['labels'] > 0).astype(np.uint8)
    
    # Fallback: luis-v2 labels
    lpath = Path(f"/shared-docker/oasis-luis-v2/cache/with2026/{tile}_labels.npz")
    if lpath.exists():
        d = np.load(lpath)
        labels = d['labels'].reshape(1002, 1002)
        return (labels == 1).astype(np.uint8)
    
    raise FileNotFoundError(f"No GT labels for {tile}")


def get_tile_profile(tile):
    """Get rasterio profile (CRS + transform) from S2 data."""
    s2_dir = DATA / "sentinel-2" / "train" / f"{tile}__s2_l2a"
    for year in [2020, 2021, 2022]:
        for month in [1, 6]:
            p = s2_dir / f"{tile}__s2_l2a_{year}_{month}.tif"
            if p.exists():
                with rasterio.open(p) as src:
                    prof = src.profile.copy()
                    prof['count'] = 1
                    prof['dtype'] = 'uint8'
                    return prof
    raise FileNotFoundError(f"No S2 data for {tile}")


# ─── OOF prob loading ────────────────────────────────────────

def load_ekl_oof(tile):
    """Load Eklavya stacked OOF predictions."""
    stack_path = EKL_OOF / "stack" / f"{tile}.npy"
    if stack_path.exists():
        return np.load(stack_path).astype(np.float32)
    
    # Fallback: average individual models
    models = ['lgbm', 'tcn', 'unet_r34', 'unet_eb3', 'unet_mit_b1']
    probs = []
    for m in models:
        p = EKL_OOF / m / f"{tile}.npy"
        if p.exists():
            probs.append(np.load(p).astype(np.float32))
    if probs:
        return np.mean(probs, axis=0)
    raise FileNotFoundError(f"No Eklavya OOF for {tile}")


def load_luis_oof(tile):
    """Load Luis v2 OOF ensemble prediction."""
    lpath = LUIS_V2_OOF / f"{tile}_oof.npz"
    if lpath.exists():
        d = np.load(lpath)
        key = 'p_ens' if 'p_ens' in d else 'p_lgbm'
        arr = d[key].astype(np.float32)
        n = arr.shape[0]
        # Infer 2D shape: find closest square
        side = int(np.sqrt(n))
        # Try common sizes around the square root
        for s1 in range(side - 5, side + 6):
            if s1 <= 0:
                continue
            s2, rem = divmod(n, s1)
            if rem == 0 and abs(s1 - s2) <= 10:
                return arr.reshape(s1, s2)
        # Too small / corrupted → skip
        raise FileNotFoundError(f"Luis OOF for {tile} has unusual size {n}")
    raise FileNotFoundError(f"No Luis OOF for {tile}")


# ─── Fusion + postprocessing ─────────────────────────────────

def fuse_probs(ekl, luis, weights, region):
    """Fuse Eklavya + Luis prob maps."""
    h = min(ekl.shape[0], luis.shape[0])
    w = min(ekl.shape[1], luis.shape[1])
    e, l = ekl[:h, :w], luis[:h, :w]
    
    w_ekl, w_luis = weights
    fused = w_ekl * e + w_luis * l
    return fused


def postprocess(binary, close_iter=2, open_iter=1, min_px=25):
    """Morphological post-processing."""
    result = binary.copy()
    s8 = ndimage.generate_binary_structure(2, 2)
    s4 = ndimage.generate_binary_structure(2, 1)
    result = ndimage.binary_closing(result, s8, iterations=close_iter).astype(np.uint8)
    result = ndimage.binary_opening(result, s4, iterations=open_iter).astype(np.uint8)
    
    labeled, n = ndimage.label(result)
    for cid in range(1, n + 1):
        if (labeled == cid).sum() < min_px:
            result[labeled == cid] = 0
    return result


# ─── Vectorization + Union IoU ────────────────────────────────

def vectorize_mask(binary, profile, min_ha=0.5):
    """Binary raster → GeoDataFrame of polygons in UTM."""
    if binary.sum() == 0:
        return gpd.GeoDataFrame(geometry=[], crs=profile['crs'])
    
    polys = []
    for geom, v in rio_shapes(binary.astype(np.uint8), mask=binary.astype(bool),
                               transform=profile['transform']):
        if v == 1:
            polys.append(shape(geom))
    
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=profile['crs'])
    
    gdf = gpd.GeoDataFrame(geometry=polys, crs=profile['crs'])
    
    # Filter by area in UTM
    if str(gdf.crs).startswith('EPSG:4326'):
        utm = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm)
    else:
        gdf_utm = gdf
    
    areas_ha = gdf_utm.area / 10000.0
    gdf_utm = gdf_utm[areas_ha >= min_ha].reset_index(drop=True)
    return gdf_utm


def compute_union_iou(pred_gdf, gt_gdf):
    """Compute Union IoU between two GeoDataFrames (already in UTM)."""
    if pred_gdf.empty and gt_gdf.empty:
        return 1.0, 0, 0, 0
    if pred_gdf.empty:
        fn = gt_gdf.unary_union.area if not gt_gdf.empty else 0
        return 0.0, 0, 0, fn
    if gt_gdf.empty:
        fp = pred_gdf.unary_union.area if not pred_gdf.empty else 0
        return 0.0, 0, fp, 0
    
    pred_union = unary_union(pred_gdf.geometry)
    gt_union = unary_union(gt_gdf.geometry)
    
    inter = pred_union.intersection(gt_union).area
    tp = inter
    fp = pred_union.area - inter
    fn = gt_union.area - inter
    
    iou = tp / max(tp + fp + fn, 1.0)
    return iou, tp, fp, fn


def poly_recall_fpr(pred_gdf, gt_gdf, match_threshold=0.0):
    """Approximate poly recall and FPR."""
    if gt_gdf.empty:
        return 0.0, 1.0 if not pred_gdf.empty else 0.0
    if pred_gdf.empty:
        return 0.0, 0.0
    
    # Recall: fraction of GT polys matched by any pred poly
    from shapely.strtree import STRtree
    pred_tree = STRtree(pred_gdf.geometry.values)
    matched_gt = 0
    for gt_geom in gt_gdf.geometry:
        candidates = pred_tree.query(gt_geom)
        for idx in candidates:
            if pred_gdf.geometry.iloc[idx].intersection(gt_geom).area > match_threshold:
                matched_gt += 1
                break
    recall = matched_gt / len(gt_gdf)
    
    # FPR: fraction of pred polys NOT matching any GT
    gt_tree = STRtree(gt_gdf.geometry.values)
    unmatched_pred = 0
    for pred_geom in pred_gdf.geometry:
        candidates = gt_tree.query(pred_geom)
        matched = False
        for idx in candidates:
            if gt_gdf.geometry.iloc[idx].intersection(pred_geom).area > match_threshold:
                matched = True
                break
        if not matched:
            unmatched_pred += 1
    fpr = unmatched_pred / len(pred_gdf)
    
    return recall, fpr


# ─── Main sweep ──────────────────────────────────────────────

def evaluate_strategy(tiles, strategy, gt_cache, prof_cache, verbose=False):
    """Evaluate a single strategy across tiles, return Union IoU."""
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_pred_polys = 0
    total_gt_polys = 0
    tile_results = []
    
    for tile in tiles:
        region = tile_region(tile)
        gt_binary = gt_cache[tile]
        prof = prof_cache[tile]
        
        # Load OOF probs
        try:
            ekl = load_ekl_oof(tile)
        except FileNotFoundError:
            continue
        
        try:
            luis = load_luis_oof(tile)
        except FileNotFoundError:
            luis = np.zeros_like(ekl)
        
        h = min(ekl.shape[0], luis.shape[0], gt_binary.shape[0])
        w = min(ekl.shape[1], luis.shape[1], gt_binary.shape[1])
        ekl = ekl[:h, :w]
        luis = luis[:h, :w]
        gt = gt_binary[:h, :w]
        
        # Fuse
        weights = strategy.get('weights', (0.75, 0.25))
        fused = fuse_probs(ekl, luis, weights, region)
        
        # Threshold
        t = strategy.get('thresholds', {}).get(region, strategy.get('threshold', 0.5))
        binary = (fused >= t).astype(np.uint8)
        
        # Postprocess
        pp = strategy.get('postproc', {})
        close_iter = pp.get('close', 2)
        open_iter = pp.get('open', 1)
        min_ha = pp.get('min_ha', {}).get(region, pp.get('min_ha_default', 0.25))
        min_px = int(min_ha / 0.01) if isinstance(min_ha, (int, float)) else 25
        binary = postprocess(binary, close_iter, open_iter, min_px)
        
        # Vectorize
        pred_prof = prof.copy()
        pred_prof['height'] = h
        pred_prof['width'] = w
        
        pred_gdf = vectorize_mask(binary, pred_prof, min_ha=min_ha if isinstance(min_ha, (int, float)) else 0.25)
        gt_gdf = vectorize_mask(gt, pred_prof, min_ha=0.5)
        
        iou, tp, fp, fn = compute_union_iou(pred_gdf, gt_gdf)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_pred_polys += len(pred_gdf)
        total_gt_polys += len(gt_gdf)
        
        tile_results.append({
            'tile': tile, 'region': region, 'iou': iou,
            'pred_polys': len(pred_gdf), 'gt_polys': len(gt_gdf),
            'tp_ha': tp/10000, 'fp_ha': fp/10000, 'fn_ha': fn/10000
        })
        
        if verbose:
            print(f"  {tile} ({region:6s}): IoU={iou:.4f}, pred={len(pred_gdf):>4d}, gt={len(gt_gdf):>4d}, t={t:.2f}")
    
    global_iou = total_tp / max(total_tp + total_fp + total_fn, 1.0)
    return global_iou, total_pred_polys, total_gt_polys, tile_results


def main():
    print("Loading GT labels and tile profiles...")
    gt_cache = {}
    prof_cache = {}
    
    available = []
    for tile in TRAIN_TILES:
        try:
            gt_cache[tile] = load_gt_binary(tile)
            prof_cache[tile] = get_tile_profile(tile)
            # Verify Eklavya OOF exists
            load_ekl_oof(tile)
            available.append(tile)
        except FileNotFoundError as e:
            print(f"  Skip {tile}: {e}")
    
    print(f"  Available tiles: {len(available)}/{len(TRAIN_TILES)}")
    
    # Count GT polygons
    total_gt = 0
    for tile in available:
        gt_gdf = vectorize_mask(gt_cache[tile], prof_cache[tile], min_ha=0.5)
        region = tile_region(tile)
        total_gt += len(gt_gdf)
        print(f"  GT: {tile} ({region:6s}): {len(gt_gdf):>4d} polygons, {gt_cache[tile].sum():>7,} positive px")
    print(f"  Total GT polygons: {total_gt}")
    
    # ─── MEGA SWEEP ──────────────────────────────────────────
    strategies = {}
    
    # 1. Eklavya only at various thresholds
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        strategies[f"ekl_only_t{int(t*100)}"] = {
            'weights': (1.0, 0.0),
            'threshold': t,
            'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
        }
    
    # 2. Per-region thresholds (Eklavya only)
    for ta in [0.35, 0.40, 0.45, 0.50, 0.55]:
        for tas in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            strategies[f"ekl_pr_am{int(ta*100)}_as{int(tas*100)}"] = {
                'weights': (1.0, 0.0),
                'thresholds': {'amazon': ta, 'asia': tas},
                'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
            }
    
    # 3. Fusion at various weights + thresholds
    for w_ekl in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]:
        for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
            strategies[f"fuse_{int(w_ekl*100)}_t{int(t*100)}"] = {
                'weights': (w_ekl, 1.0 - w_ekl),
                'threshold': t,
                'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
            }
    
    # 4. Fusion with per-region thresholds
    for w_ekl in [0.70, 0.75, 0.80]:
        for ta in [0.40, 0.45, 0.50, 0.55]:
            for tas in [0.35, 0.40, 0.45, 0.50, 0.55]:
                strategies[f"fuse_{int(w_ekl*100)}_am{int(ta*100)}_as{int(tas*100)}"] = {
                    'weights': (w_ekl, 1.0 - w_ekl),
                    'thresholds': {'amazon': ta, 'asia': tas},
                    'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
                }
    
    # 5. Min area variants
    for min_ha in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        strategies[f"ekl_minha_{int(min_ha*100)}"] = {
            'weights': (1.0, 0.0),
            'threshold': 0.45,
            'postproc': {'close': 2, 'open': 1, 'min_ha_default': min_ha}
        }
    
    # 6. Morphology variants
    for close in [1, 2, 3]:
        for opn in [0, 1, 2]:
            strategies[f"morph_c{close}_o{opn}"] = {
                'weights': (1.0, 0.0),
                'threshold': 0.45,
                'postproc': {'close': close, 'open': opn, 'min_ha_default': 0.25}
            }
    
    print(f"\n{'='*80}")
    print(f"SWEEPING {len(strategies)} strategies on {len(available)} OOF tiles")
    print(f"{'='*80}")
    
    results = []
    for i, (name, strat) in enumerate(strategies.items()):
        iou, n_pred, n_gt, tiles = evaluate_strategy(available, strat, gt_cache, prof_cache)
        results.append({
            'name': name, 'iou': iou, 'n_pred': n_pred, 'n_gt': n_gt,
            'strategy': strat, 'tiles': tiles
        })
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(strategies)}] Best so far: {max(r['iou'] for r in results):.4f}")
    
    # Sort by IoU
    results.sort(key=lambda r: -r['iou'])
    
    print(f"\n{'='*80}")
    print(f"TOP 30 STRATEGIES by Union IoU")
    print(f"{'='*80}")
    print(f"{'Rank':>4s} {'Name':<40s} {'IoU':>8s} {'Pred':>6s} {'GT':>6s}")
    print('-' * 70)
    for i, r in enumerate(results[:30]):
        print(f"{i+1:>4d} {r['name']:<40s} {r['iou']:>8.4f} {r['n_pred']:>6d} {r['n_gt']:>6d}")
    
    # Region breakdown for top 5
    print(f"\n{'='*80}")
    print("TOP 5 — Per-tile breakdown")
    print(f"{'='*80}")
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['name']} — Union IoU = {r['iou']:.4f}")
        s = r['strategy']
        print(f"       weights={s.get('weights')}, threshold={s.get('threshold', 'per-region')}, thresholds={s.get('thresholds', '-')}")
        for t in r['tiles']:
            print(f"       {t['tile']} ({t['region']:6s}): IoU={t['iou']:.4f}, pred={t['pred_polys']:>4d}, gt={t['gt_polys']:>4d}, FP={t['fp_ha']:.0f}ha, FN={t['fn_ha']:.0f}ha")
    
    # Save full results
    out_path = Path(__file__).resolve().parent.parent / "artifacts" / "oof_sweep"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Serialize results
    save_results = []
    for r in results:
        s = r.copy()
        s['strategy'] = str(s['strategy'])
        s['tiles'] = str(s['tiles'])
        save_results.append(s)
    
    with open(out_path / "full_results.json", "w") as f:
        json.dump(save_results[:100], f, indent=2)
    
    best = results[0]
    with open(out_path / "best_strategy.json", "w") as f:
        json.dump({
            'name': best['name'],
            'iou': best['iou'],
            'n_pred': best['n_pred'],
            'strategy': best['strategy']
        }, f, indent=2)
    
    print(f"\nResults saved to {out_path}/")
    print(f"\nBEST: {best['name']} → OOF Union IoU = {best['iou']:.4f}")
    
    return results


if __name__ == "__main__":
    main()

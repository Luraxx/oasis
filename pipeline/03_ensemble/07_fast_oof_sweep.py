#!/usr/bin/env python3
"""
FAST OOF Optimizer — Two-phase approach:
  Phase 1: Pixel-level IoU (fast, ~1000 strategies in seconds)
  Phase 2: Full polygon Union IoU (slow, only top 20 candidates)

This finds THE best strategy without submitting.
"""
import sys, json, warnings, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))
warnings.filterwarnings('ignore')

import numpy as np
from pathlib import Path
from scipy import ndimage

from config import TRAIN_TILES, EKL_OOF, LUIS_V2_OOF, DATA, tile_region

# ─── Data loading (cached) ──────────────────────────────────

def load_gt_binary(tile):
    bundle = Path(f"/shared-docker/oasis/data/cache/train/{tile}/bundle.npz")
    if bundle.exists():
        d = np.load(bundle)
        if 'consensus_pos' in d:
            return d['consensus_pos'].astype(np.uint8)
        return (d['labels'] > 0).astype(np.uint8)
    lpath = Path(f"/shared-docker/oasis-luis-v2/cache/with2026/{tile}_labels.npz")
    if lpath.exists():
        d = np.load(lpath)
        labels = d['labels'].reshape(1002, 1002)
        return (labels == 1).astype(np.uint8)
    raise FileNotFoundError(f"No GT labels for {tile}")


def load_ekl_oof(tile):
    stack_path = EKL_OOF / "stack" / f"{tile}.npy"
    if stack_path.exists():
        return np.load(stack_path).astype(np.float32)
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
    lpath = LUIS_V2_OOF / f"{tile}_oof.npz"
    if lpath.exists():
        d = np.load(lpath)
        key = 'p_ens' if 'p_ens' in d else 'p_lgbm'
        arr = d[key].astype(np.float32)
        n = arr.shape[0]
        side = int(np.sqrt(n))
        for s1 in range(side - 5, side + 6):
            if s1 <= 0: continue
            s2, rem = divmod(n, s1)
            if rem == 0 and abs(s1 - s2) <= 10:
                return arr.reshape(s1, s2)
        raise FileNotFoundError(f"Luis OOF for {tile} has unusual size {n}")
    raise FileNotFoundError(f"No Luis OOF for {tile}")


def postprocess(binary, close_iter=2, open_iter=1, min_px=25):
    result = binary.copy()
    s8 = ndimage.generate_binary_structure(2, 2)
    s4 = ndimage.generate_binary_structure(2, 1)
    if close_iter > 0:
        result = ndimage.binary_closing(result, s8, iterations=close_iter).astype(np.uint8)
    if open_iter > 0:
        result = ndimage.binary_opening(result, s4, iterations=open_iter).astype(np.uint8)
    if min_px > 1:
        labeled, n = ndimage.label(result)
        sizes = ndimage.sum(result, labeled, range(1, n + 1))
        for cid in range(1, n + 1):
            if sizes[cid - 1] < min_px:
                result[labeled == cid] = 0
    return result


# ─── Phase 1: Pixel IoU (FAST) ──────────────────────────────

def pixel_iou_sweep(tiles_data, strategies):
    """Compute pixel-level IoU for each strategy. ~100x faster than polygon."""
    results = []
    
    for name, strat in strategies.items():
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for tile, (ekl, luis, gt) in tiles_data.items():
            region = tile_region(tile)
            w_ekl, w_luis = strat.get('weights', (1.0, 0.0))
            
            if luis is not None:
                h = min(ekl.shape[0], luis.shape[0], gt.shape[0])
                w = min(ekl.shape[1], luis.shape[1], gt.shape[1])
                fused = w_ekl * ekl[:h, :w] + w_luis * luis[:h, :w]
            else:
                h = min(ekl.shape[0], gt.shape[0])
                w = min(ekl.shape[1], gt.shape[1])
                fused = ekl[:h, :w]
            
            gt_crop = gt[:h, :w]
            
            # Threshold
            t = strat.get('thresholds', {}).get(region, strat.get('threshold', 0.5))
            binary = (fused >= t).astype(np.uint8)
            
            # Postprocess
            pp = strat.get('postproc', {})
            close_i = pp.get('close', 2)
            open_i = pp.get('open', 1)
            min_ha = pp.get('min_ha', {}).get(region, pp.get('min_ha_default', 0.25))
            min_px = int(min_ha / 0.01)
            binary = postprocess(binary, close_i, open_i, min_px)
            
            tp = int((binary & gt_crop).sum())
            fp = int((binary & ~gt_crop).sum())
            fn = int((~binary & gt_crop).sum())
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        iou = total_tp / max(total_tp + total_fp + total_fn, 1)
        results.append((name, iou, total_tp, total_fp, total_fn, strat))
    
    results.sort(key=lambda x: -x[1])
    return results


# ─── Phase 2: Full Polygon Union IoU ─────────────────────────

def full_polygon_iou(tiles_data, prof_cache, strategy, verbose=True):
    """Full polygon-level Union IoU — replicates competition scorer."""
    import rasterio
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union
    import geopandas as gpd
    
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    tile_details = []
    
    for tile, (ekl, luis, gt) in tiles_data.items():
        region = tile_region(tile)
        prof = prof_cache[tile]
        
        w_ekl, w_luis = strategy.get('weights', (1.0, 0.0))
        if luis is not None:
            h = min(ekl.shape[0], luis.shape[0], gt.shape[0])
            w = min(ekl.shape[1], luis.shape[1], gt.shape[1])
            fused = w_ekl * ekl[:h, :w] + w_luis * luis[:h, :w]
        else:
            h = min(ekl.shape[0], gt.shape[0])
            w = min(ekl.shape[1], gt.shape[1])
            fused = ekl[:h, :w]
        
        gt_crop = gt[:h, :w]
        
        t = strategy.get('thresholds', {}).get(region, strategy.get('threshold', 0.5))
        binary = (fused >= t).astype(np.uint8)
        
        pp = strategy.get('postproc', {})
        close_i = pp.get('close', 2)
        open_i = pp.get('open', 1)
        min_ha = pp.get('min_ha', {}).get(region, pp.get('min_ha_default', 0.25))
        min_px = int(min_ha / 0.01)
        binary = postprocess(binary, close_i, open_i, min_px)
        
        # Adjust profile
        p = prof.copy()
        p['height'] = h
        p['width'] = w
        
        def to_polygons_utm(mask, prof_dict, min_ha_filter=0.5):
            if mask.sum() == 0:
                return gpd.GeoDataFrame(geometry=[], crs=prof_dict['crs'])
            polys = [shape(geom) for geom, v in rio_shapes(
                mask.astype(np.uint8), mask=mask.astype(bool),
                transform=prof_dict['transform']) if v == 1]
            if not polys:
                return gpd.GeoDataFrame(geometry=[], crs=prof_dict['crs'])
            gdf = gpd.GeoDataFrame(geometry=polys, crs=prof_dict['crs'])
            utm = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm)
            areas = gdf_utm.area / 10000.0
            return gdf_utm[areas >= min_ha_filter].reset_index(drop=True)
        
        pred_gdf = to_polygons_utm(binary, p, min_ha_filter=min_ha)
        gt_gdf = to_polygons_utm(gt_crop, p, min_ha_filter=0.5)
        
        if pred_gdf.empty and gt_gdf.empty:
            tp, fp, fn = 0, 0, 0
        elif pred_gdf.empty:
            tp, fp, fn = 0, 0, gt_gdf.unary_union.area
        elif gt_gdf.empty:
            tp, fp, fn = 0, pred_gdf.unary_union.area, 0
        else:
            pu = unary_union(pred_gdf.geometry)
            gu = unary_union(gt_gdf.geometry)
            inter = pu.intersection(gu).area
            tp = inter
            fp = pu.area - inter
            fn = gu.area - inter
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        tile_iou = tp / max(tp + fp + fn, 1)
        tile_details.append({
            'tile': tile, 'region': region, 'iou': f"{tile_iou:.4f}",
            'pred_polys': len(pred_gdf), 'gt_polys': len(gt_gdf),
        })
        
        if verbose:
            print(f"  {tile} ({region:6s}): IoU={tile_iou:.4f}, pred={len(pred_gdf):>4d}, gt={len(gt_gdf):>4d}")
    
    global_iou = total_tp / max(total_tp + total_fp + total_fn, 1)
    return global_iou, tile_details


def get_tile_profile(tile):
    s2_dir = DATA / "sentinel-2" / "train" / f"{tile}__s2_l2a"
    for year in [2020, 2021, 2022]:
        for month in [1, 6]:
            p = s2_dir / f"{tile}__s2_l2a_{year}_{month}.tif"
            if p.exists():
                import rasterio
                with rasterio.open(p) as src:
                    prof = src.profile.copy()
                    prof['count'] = 1
                    prof['dtype'] = 'uint8'
                    return prof
    raise FileNotFoundError(f"No S2 data for {tile}")


# ─── Main ────────────────────────────────────────────────────

def main():
    t0 = time.time()
    
    # Load all data into memory
    print("Loading data...")
    tiles_data = {}
    prof_cache = {}
    
    for tile in TRAIN_TILES:
        try:
            gt = load_gt_binary(tile)
            ekl = load_ekl_oof(tile)
            try:
                luis = load_luis_oof(tile)
            except FileNotFoundError:
                luis = None
            tiles_data[tile] = (ekl, luis, gt)
            prof_cache[tile] = get_tile_profile(tile)
        except FileNotFoundError as e:
            print(f"  Skip {tile}: {e}")
    
    print(f"  Loaded {len(tiles_data)} tiles in {time.time()-t0:.1f}s")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Mega pixel-IoU sweep (~1000+ strategies)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PHASE 1: Fast pixel-IoU sweep")
    print(f"{'='*80}")
    
    strategies = {}
    
    # 1. Eklavya only — threshold sweep
    for t in np.arange(0.15, 0.75, 0.05):
        strategies[f"ekl_t{int(t*100):02d}"] = {
            'weights': (1.0, 0.0), 'threshold': round(t, 2),
            'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
        }
    
    # 2. Eklavya — per-region threshold sweep
    for ta in np.arange(0.25, 0.65, 0.05):
        for tas in np.arange(0.25, 0.65, 0.05):
            strategies[f"ekl_am{int(ta*100)}_as{int(tas*100)}"] = {
                'weights': (1.0, 0.0),
                'thresholds': {'amazon': round(ta, 2), 'asia': round(tas, 2)},
                'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
            }
    
    # 3. Fusion — weight × threshold grid
    for w_ekl in np.arange(0.50, 1.01, 0.05):
        for t in np.arange(0.20, 0.65, 0.05):
            strategies[f"f{int(w_ekl*100)}_t{int(t*100):02d}"] = {
                'weights': (round(w_ekl, 2), round(1-w_ekl, 2)),
                'threshold': round(t, 2),
                'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
            }
    
    # 4. Fusion — per-region thresholds
    for w_ekl in [0.65, 0.70, 0.75, 0.80, 0.85]:
        for ta in np.arange(0.30, 0.60, 0.05):
            for tas in np.arange(0.30, 0.60, 0.05):
                strategies[f"f{int(w_ekl*100)}_am{int(ta*100)}_as{int(tas*100)}"] = {
                    'weights': (round(w_ekl, 2), round(1-w_ekl, 2)),
                    'thresholds': {'amazon': round(ta, 2), 'asia': round(tas, 2)},
                    'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
                }
    
    # 5. Morphology variants (on top configs)
    for close in [1, 2, 3]:
        for opn in [0, 1, 2]:
            for min_ha in [0.15, 0.20, 0.25, 0.30, 0.50]:
                strategies[f"morph_c{close}_o{opn}_ha{int(min_ha*100)}"] = {
                    'weights': (1.0, 0.0), 'threshold': 0.45,
                    'postproc': {'close': close, 'open': opn, 'min_ha_default': min_ha}
                }
    
    print(f"  Testing {len(strategies)} strategies...")
    t1 = time.time()
    px_results = pixel_iou_sweep(tiles_data, strategies)
    print(f"  Done in {time.time()-t1:.1f}s")
    
    print(f"\n  TOP 30 by pixel IoU:")
    print(f"  {'Rank':>4s} {'Name':<45s} {'pxIoU':>8s}  {'TP':>10s} {'FP':>10s} {'FN':>10s}")
    print(f"  {'-'*90}")
    for i, (name, iou, tp, fp, fn, strat) in enumerate(px_results[:30]):
        print(f"  {i+1:>4d} {name:<45s} {iou:>8.4f}  {tp:>10,} {fp:>10,} {fn:>10,}")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Full polygon Union IoU on top 20
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PHASE 2: Full polygon Union IoU (top 20 candidates)")
    print(f"{'='*80}")
    
    top_n = 20
    poly_results = []
    
    for i, (name, px_iou, tp, fp, fn, strat) in enumerate(px_results[:top_n]):
        print(f"\n  [{i+1}/{top_n}] {name} (pixel IoU={px_iou:.4f})")
        t2 = time.time()
        poly_iou, details = full_polygon_iou(tiles_data, prof_cache, strat, verbose=True)
        elapsed = time.time() - t2
        poly_results.append((name, poly_iou, px_iou, strat, details))
        print(f"  → Polygon Union IoU = {poly_iou:.4f} ({elapsed:.1f}s)")
    
    poly_results.sort(key=lambda x: -x[1])
    
    print(f"\n{'='*80}")
    print("FINAL RANKING — Polygon Union IoU")
    print(f"{'='*80}")
    print(f"  {'Rank':>4s} {'Name':<45s} {'PolyIoU':>9s} {'PxIoU':>8s}")
    print(f"  {'-'*70}")
    for i, (name, poly_iou, px_iou, strat, details) in enumerate(poly_results):
        print(f"  {i+1:>4d} {name:<45s} {poly_iou:>9.4f} {px_iou:>8.4f}")
    
    # Best strategy details
    best_name, best_iou, best_px, best_strat, best_details = poly_results[0]
    print(f"\n{'='*80}")
    print(f"BEST: {best_name} → OOF Polygon Union IoU = {best_iou:.4f}")
    print(f"{'='*80}")
    print(f"  Strategy: {json.dumps(best_strat, indent=4)}")
    print(f"  Per-tile:")
    for d in best_details:
        print(f"    {d['tile']} ({d['region']:6s}): IoU={d['iou']}, pred={d['pred_polys']:>4d}, gt={d['gt_polys']:>4d}")
    
    # Save
    out_path = Path(__file__).resolve().parent.parent / "artifacts" / "oof_sweep"
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(out_path / "best_strategy.json", "w") as f:
        json.dump({
            'name': best_name,
            'poly_iou': best_iou,
            'px_iou': best_px,
            'strategy': best_strat,
            'details': best_details,
        }, f, indent=2)
    
    # Save top 100 pixel results
    with open(out_path / "pixel_iou_top100.json", "w") as f:
        json.dump([
            {'name': n, 'px_iou': round(iou, 6), 'strategy': str(s)}
            for n, iou, _, _, _, s in px_results[:100]
        ], f, indent=2)
    
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    return poly_results


if __name__ == "__main__":
    main()

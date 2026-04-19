#!/usr/bin/env python3
"""
Phase 3: Fine-tune the best strategy from the OOF sweep.
Explore morphology, min_area, per-region thresholds around the winner.
Then generate THE final submission.
"""
import sys, json, warnings, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from pathlib import Path
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union
import geopandas as gpd

from config import (
    TRAIN_TILES, TEST_TILES, EKL_OOF, LUIS_V2_OOF, DATA, DATA_ADDITIONAL,
    EKL_SUBMISSION, LUIS_V4_SUBMISSION, SUBMISSION, ARTIFACTS, tile_region
)


# ─── Loaders (same as 07) ──────────────────────────────────

def load_gt_binary(tile):
    bundle = Path(f"/shared-docker/oasis/data/cache/train/{tile}/bundle.npz")
    if bundle.exists():
        d = np.load(bundle)
        if 'consensus_pos' in d:
            return d['consensus_pos'].astype(np.uint8)
        return (d['labels'] > 0).astype(np.uint8)
    raise FileNotFoundError(f"No GT for {tile}")


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


def get_tile_profile(tile, split='train'):
    s2_dir = DATA / "sentinel-2" / split / f"{tile}__s2_l2a"
    for year in [2020, 2021, 2022]:
        for month in [1, 6]:
            p = s2_dir / f"{tile}__s2_l2a_{year}_{month}.tif"
            if p.exists():
                with rasterio.open(p) as src:
                    return src.profile.copy()
    raise FileNotFoundError(f"No S2 data for {tile}")


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


def to_polygons_utm(mask, prof, min_ha=0.5):
    if mask.sum() == 0:
        return gpd.GeoDataFrame(geometry=[], crs=prof['crs'])
    polys = [shape(geom) for geom, v in rio_shapes(
        mask.astype(np.uint8), mask=mask.astype(bool),
        transform=prof['transform']) if v == 1]
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=prof['crs'])
    gdf = gpd.GeoDataFrame(geometry=polys, crs=prof['crs'])
    utm = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm)
    return gdf_utm[gdf_utm.area / 10000.0 >= min_ha].reset_index(drop=True)


def evaluate_full(tiles_data, prof_cache, strat, verbose=False):
    total_tp = total_fp = total_fn = 0.0
    details = []
    for tile, (ekl, luis, gt) in tiles_data.items():
        region = tile_region(tile)
        prof = prof_cache[tile]
        w_ekl, w_luis = strat['weights']
        if luis is not None:
            h = min(ekl.shape[0], luis.shape[0], gt.shape[0])
            w = min(ekl.shape[1], luis.shape[1], gt.shape[1])
            fused = w_ekl * ekl[:h, :w] + w_luis * luis[:h, :w]
        else:
            h, w = min(ekl.shape[0], gt.shape[0]), min(ekl.shape[1], gt.shape[1])
            fused = ekl[:h, :w]
        gt_crop = gt[:h, :w]
        
        t = strat.get('thresholds', {}).get(region, strat.get('threshold', 0.5))
        binary = (fused >= t).astype(np.uint8)
        pp = strat.get('postproc', {})
        min_ha = pp.get('min_ha', {}).get(region, pp.get('min_ha_default', 0.25))
        binary = postprocess(binary, pp.get('close', 2), pp.get('open', 1), int(min_ha / 0.01))
        
        p = prof.copy(); p['height'] = h; p['width'] = w
        pred_gdf = to_polygons_utm(binary, p, min_ha)
        gt_gdf = to_polygons_utm(gt_crop, p, 0.5)
        
        if pred_gdf.empty and gt_gdf.empty:
            tp = fp = fn = 0.0
        elif pred_gdf.empty:
            tp = fp = 0.0; fn = gt_gdf.unary_union.area
        elif gt_gdf.empty:
            tp = 0.0; fp = pred_gdf.unary_union.area; fn = 0.0
        else:
            pu, gu = unary_union(pred_gdf.geometry), unary_union(gt_gdf.geometry)
            inter = pu.intersection(gu).area
            tp, fp, fn = inter, pu.area - inter, gu.area - inter
        
        total_tp += tp; total_fp += fp; total_fn += fn
        tile_iou = tp / max(tp + fp + fn, 1)
        details.append({'tile': tile, 'region': region, 'iou': tile_iou,
                        'pred': len(pred_gdf), 'gt': len(gt_gdf)})
        if verbose:
            print(f"  {tile} ({region:6s}): IoU={tile_iou:.4f}, pred={len(pred_gdf):>4d}, gt={len(gt_gdf):>4d}")
    
    iou = total_tp / max(total_tp + total_fp + total_fn, 1)
    return iou, details


# ─── Year estimation (from 05_smart_fusion, fixed) ───────────

def estimate_year(tile, binary, prof, split='train'):
    s2_dir = DATA / "sentinel-2" / split / f"{tile}__s2_l2a"
    h, w = binary.shape
    
    yearly_nbr = {}
    for year in range(2020, 2026):
        nbrs = []
        for month in range(1, 13):
            p = s2_dir / f"{tile}__s2_l2a_{year}_{month}.tif"
            if not p.exists(): continue
            try:
                with rasterio.open(p) as src:
                    b = src.read().astype(np.float32)
                b08, b12 = b[7], b[11]
                d = b08 + b12
                nbr = np.where(d > 0, (b08 - b12) / d, 0)
                nbr[b[0] <= 0] = np.nan
                nbrs.append(nbr[:h, :w])
            except Exception:
                continue
        if nbrs:
            with np.errstate(all='ignore'):
                yearly_nbr[year] = np.nanmedian(nbrs, axis=0)
    
    sorted_yrs = sorted(yr for yr in yearly_nbr.keys() if yr <= 2025)
    max_drop = np.zeros((h, w), dtype=np.float32)
    drop_year = np.full((h, w), 2023, dtype=np.int32)
    
    for i in range(1, len(sorted_yrs)):
        prev, curr = sorted_yrs[i-1], sorted_yrs[i]
        p_arr = yearly_nbr[prev][:h, :w]
        c_arr = yearly_nbr[curr][:h, :w]
        # Ensure same shape
        mh, mw = min(p_arr.shape[0], c_arr.shape[0], h), min(p_arr.shape[1], c_arr.shape[1], w)
        with np.errstate(all='ignore'):
            drop = np.nan_to_num(p_arr[:mh, :mw] - c_arr[:mh, :mw], nan=0)
        drop = np.clip(drop, 0, None)
        
        # Pad to full size if needed
        if mh < h or mw < w:
            full = np.zeros((h, w), dtype=np.float32)
            full[:mh, :mw] = drop
            drop = full
        
        better = drop > max_drop
        max_drop[better] = drop[better]
        drop_year[better] = curr
    
    labeled, n = ndimage.label(binary)
    year_map = np.full((h, w), 2306, dtype=np.int32)
    
    for cid in range(1, n + 1):
        cmask = labeled == cid
        cyears = drop_year[cmask]
        cscores = max_drop[cmask]
        
        if cscores.sum() > 0:
            yt = {}
            for yr, sc in zip(cyears, cscores):
                yt[yr] = yt.get(yr, 0) + sc
            modal_yr = max(yt, key=yt.get)
        else:
            vals, cnts = np.unique(cyears, return_counts=True)
            modal_yr = vals[np.argmax(cnts)]
        
        modal_yr = max(2020, min(2025, modal_yr))
        year_map[cmask] = (modal_yr % 100) * 100 + 6
    
    return year_map


# ─── Generate final submission ───────────────────────────────

def generate_submission(strategy, name="v5_final"):
    """Generate test submission with the optimal strategy."""
    print(f"\n{'='*80}")
    print(f"GENERATING SUBMISSION: {name}")
    print(f"  Strategy: {json.dumps(strategy, indent=2)}")
    print(f"{'='*80}")
    
    w_ekl, w_luis = strategy['weights']
    all_gdfs = []
    
    for tile in TEST_TILES:
        region = tile_region(tile)
        
        # Load test prob maps
        with rasterio.open(EKL_SUBMISSION / f"prob_{tile}.tif") as src:
            ekl = src.read(1).astype(np.float32)
            prof = src.profile.copy()
        if ekl.max() > 10:
            ekl /= 1000.0
        
        with rasterio.open(LUIS_V4_SUBMISSION / f"prob_{tile}.tif") as src:
            luis = src.read(1).astype(np.float32)
        if luis.max() > 10:
            luis /= 1000.0
        
        h = min(ekl.shape[0], luis.shape[0])
        w = min(ekl.shape[1], luis.shape[1])
        ekl, luis = ekl[:h, :w], luis[:h, :w]
        
        fused = w_ekl * ekl + w_luis * luis
        
        t = strategy.get('thresholds', {}).get(region, strategy.get('threshold', 0.5))
        binary = (fused >= t).astype(np.uint8)
        
        pp = strategy.get('postproc', {})
        min_ha = pp.get('min_ha', {}).get(region, pp.get('min_ha_default', 0.25))
        binary = postprocess(binary, pp.get('close', 2), pp.get('open', 1), int(min_ha / 0.01))
        
        # Year estimation
        prof2 = prof.copy()
        prof2['height'] = h; prof2['width'] = w
        year_map = estimate_year(tile, binary, prof2, split='test')
        
        # Vectorize
        if binary.sum() == 0:
            print(f"  {tile} ({region:6s}): 0 polygons (empty)")
            continue
        
        polys = [shape(geom) for geom, v in rio_shapes(
            binary.astype(np.uint8), mask=binary.astype(bool),
            transform=prof2['transform']) if v == 1]
        
        if not polys:
            continue
        
        gdf = gpd.GeoDataFrame(geometry=polys, crs=prof2['crs'])
        gdf = gdf.to_crs("EPSG:4326")
        
        utm = gdf.estimate_utm_crs()
        areas = gdf.to_crs(utm).area / 10000.0
        gdf = gdf[areas >= min_ha].reset_index(drop=True)
        
        if gdf.empty:
            continue
        
        # Assign time_step and confidence
        gdf_px = gdf.to_crs(prof2['crs'])
        transform = prof2['transform']
        
        ts_list, conf_list = [], []
        for _, row in gdf_px.iterrows():
            c = row.geometry.centroid
            col = int((c.x - transform.c) / transform.a)
            rpx = int((c.y - transform.f) / transform.e)
            ts, conf = 2306, 0.5
            if 0 <= rpx < h and 0 <= col < w:
                ts = int(year_map[rpx, col])
                conf = float(fused[rpx, col])
            ts_list.append(ts)
            conf_list.append(round(min(max(conf, 0.01), 1.0), 3))
        
        gdf['time_step'] = ts_list
        gdf['confidence'] = conf_list
        gdf['tile_id'] = tile
        
        print(f"  {tile} ({region:6s}): t={t:.2f}, {len(gdf):>4d} polygons")
        all_gdfs.append(gdf)
        
        # Save prob raster
        out_dir = SUBMISSION / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pp_prof = prof2.copy()
        pp_prof.update(dtype='uint16', count=1, nodata=0)
        with rasterio.open(out_dir / f"prob_{tile}.tif", 'w', **pp_prof) as dst:
            dst.write((np.clip(fused, 0, 1) * 1000).astype(np.uint16), 1)
    
    if all_gdfs:
        combined = gpd.pd.concat(all_gdfs, ignore_index=True)
        combined.insert(0, 'id', range(len(combined)))
        out_path = SUBMISSION / name / "submission.geojson"
        combined[['id', 'time_step', 'confidence', 'tile_id', 'geometry']].to_file(
            out_path, driver='GeoJSON'
        )
        
        # Also copy to main submission dir
        import shutil
        main_sub = SUBMISSION / "submission.geojson"
        shutil.copy2(out_path, main_sub)
        
        print(f"\n  TOTAL: {len(combined)} polygons")
        print(f"  Saved to: {out_path}")
        print(f"  Copied to: {main_sub}")
        
        # Year dist
        ts = combined['time_step'].value_counts().sort_index()
        print(f"  Year distribution: {dict(ts)}")
        
        return combined
    return None


def main():
    t0 = time.time()
    
    # Load OOF data
    print("Loading OOF data...")
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
    
    # ═════════════════════════════════════════════════════════
    # FINE-TUNE around winner: f60_t59 (w=0.60/0.40, t=0.59)
    # ═════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("FINE-TUNING around best strategy (f60_t59)")
    print(f"{'='*80}")
    
    fine_strategies = {}
    
    # A. Weight fine-tuning (0.55-0.65 in 0.01 steps) × threshold (0.55-0.65)
    for w in np.arange(0.55, 0.66, 0.01):
        for t in np.arange(0.55, 0.66, 0.01):
            fine_strategies[f"w{int(w*100)}_t{int(t*100)}"] = {
                'weights': (round(w, 2), round(1-w, 2)),
                'threshold': round(t, 2),
                'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
            }
    
    # B. Per-region thresholds around the optimal
    for w in [0.58, 0.60, 0.62]:
        for ta in np.arange(0.50, 0.70, 0.02):
            for tas in np.arange(0.50, 0.70, 0.02):
                fine_strategies[f"w{int(w*100)}_am{int(ta*100)}_as{int(tas*100)}"] = {
                    'weights': (round(w, 2), round(1-w, 2)),
                    'thresholds': {'amazon': round(ta, 2), 'asia': round(tas, 2)},
                    'postproc': {'close': 2, 'open': 1, 'min_ha_default': 0.25}
                }
    
    # C. Morphology + min_area fine-tuning
    for close in [1, 2, 3]:
        for opn in [0, 1, 2]:
            for min_ha in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                fine_strategies[f"w60_t59_c{close}_o{opn}_ha{int(min_ha*100)}"] = {
                    'weights': (0.60, 0.40),
                    'threshold': 0.59,
                    'postproc': {'close': close, 'open': opn, 'min_ha_default': min_ha}
                }
    
    # Fast pixel-IoU sweep
    print(f"  Pixel-IoU sweep on {len(fine_strategies)} configs...")
    
    results = []
    for name, strat in fine_strategies.items():
        total_tp = total_fp = total_fn = 0
        for tile, (ekl, luis, gt) in tiles_data.items():
            region = tile_region(tile)
            w_ekl, w_luis = strat['weights']
            if luis is not None:
                h = min(ekl.shape[0], luis.shape[0], gt.shape[0])
                w = min(ekl.shape[1], luis.shape[1], gt.shape[1])
                fused = w_ekl * ekl[:h, :w] + w_luis * luis[:h, :w]
            else:
                h, w = min(ekl.shape[0], gt.shape[0]), min(ekl.shape[1], gt.shape[1])
                fused = ekl[:h, :w]
            gt_crop = gt[:h, :w]
            
            t = strat.get('thresholds', {}).get(region, strat.get('threshold', 0.5))
            binary = (fused >= t).astype(np.uint8)
            pp = strat.get('postproc', {})
            min_ha = pp.get('min_ha', {}).get(region, pp.get('min_ha_default', 0.25))
            binary = postprocess(binary, pp.get('close', 2), pp.get('open', 1), int(min_ha / 0.01))
            
            total_tp += int((binary & gt_crop).sum())
            total_fp += int((binary & ~gt_crop).sum())
            total_fn += int((~binary & gt_crop).sum())
        
        iou = total_tp / max(total_tp + total_fp + total_fn, 1)
        results.append((name, iou, strat))
    
    results.sort(key=lambda x: -x[1])
    
    print(f"\n  TOP 20 fine-tuned (pixel IoU):")
    for i, (name, iou, strat) in enumerate(results[:20]):
        print(f"  {i+1:>4d} {name:<45s} {iou:.4f}")
    
    # Full polygon IoU on top 10
    print(f"\n  Full polygon IoU on top 10:")
    poly_results = []
    for i, (name, px_iou, strat) in enumerate(results[:10]):
        poly_iou, details = evaluate_full(tiles_data, prof_cache, strat, verbose=False)
        poly_results.append((name, poly_iou, px_iou, strat, details))
        print(f"  {i+1:>4d} {name:<45s} poly={poly_iou:.4f} (px={px_iou:.4f})")
    
    poly_results.sort(key=lambda x: -x[1])
    
    best_name, best_poly, best_px, best_strat, best_details = poly_results[0]
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL: {best_name} → OOF Poly Union IoU = {best_poly:.4f}")
    print(f"  Strategy: {json.dumps(best_strat, indent=2)}")
    print(f"  Per-tile breakdown:")
    for d in best_details:
        print(f"    {d['tile']} ({d['region']:6s}): IoU={d['iou']:.4f}, pred={d['pred']:>4d}, gt={d['gt']:>4d}")
    
    # Save
    out_path = ARTIFACTS / "oof_sweep"
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "final_best.json", "w") as f:
        json.dump({
            'name': best_name, 'poly_iou': best_poly, 'px_iou': best_px,
            'strategy': best_strat, 'details': [
                {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}
                for d in best_details
            ]
        }, f, indent=2)
    
    # ═════════════════════════════════════════════════════════
    # GENERATE THE FINAL SUBMISSION
    # ═════════════════════════════════════════════════════════
    generate_submission(best_strat, name="v5_final")
    
    # Also generate runner-up for safety
    if len(poly_results) > 1:
        ru_name, _, _, ru_strat, _ = poly_results[1]
        generate_submission(ru_strat, name="v5_runnerup")
    
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

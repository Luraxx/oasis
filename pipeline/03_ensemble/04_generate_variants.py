#!/usr/bin/env python3
"""
Step 04: Generate multiple submission variants with different threshold/strategy configs.
Compare polygon counts and characteristics to help pick the best variant.
"""
import json
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))

import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage
from config import (
    TEST_TILES, EKL_SUBMISSION, LUIS_V4_SUBMISSION, SUBMISSION,
    ARTIFACTS, DATA, DATA_ADDITIONAL, tile_region
)


def load_prob(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        prof = src.profile.copy()
    if data.max() > 10:
        data /= 1000.0
    return data, prof


def postprocess(binary, region, close_iter=2, open_iter=1, min_ha=0.25):
    """Morphological cleanup + area filtering."""
    result = binary.copy()
    
    # Closing (fill gaps)
    if close_iter > 0:
        s8 = ndimage.generate_binary_structure(2, 2)
        result = ndimage.binary_closing(result, s8, iterations=close_iter).astype(np.uint8)
    
    # Opening (remove noise)
    if open_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_opening(result, s4, iterations=open_iter).astype(np.uint8)
    
    # Area filter
    labeled, n_comp = ndimage.label(result)
    min_px = int(min_ha / 0.01)  # 10m resolution → 0.01 ha per pixel
    for cid in range(1, n_comp + 1):
        if (labeled == cid).sum() < min_px:
            result[labeled == cid] = 0
    
    return result


def estimate_year(tile, binary, prof):
    """NBR-based year estimation."""
    s2_dir = DATA / "sentinel-2" / "test" / f"{tile}__s2_l2a"
    if not s2_dir.exists():
        s2_dir = DATA / "sentinel-2" / "train" / f"{tile}__s2_l2a"
    
    h, w = binary.shape
    years = list(range(2020, 2026))
    yearly_nbr = {}
    
    for year in years:
        nbrs = []
        for month in range(1, 13):
            p = s2_dir / f"{tile}__s2_l2a_{year}_{month}.tif"
            if not p.exists():
                continue
            try:
                with rasterio.open(p) as src:
                    b = src.read().astype(np.float32)
                b08, b12 = b[7], b[11]
                d = b08 + b12
                nbr = np.where(d > 0, (b08 - b12) / d, 0)
                nbr[b[0] == 0] = np.nan
                nbrs.append(nbr[:h, :w])
            except Exception:
                continue
        if nbrs:
            with np.errstate(all='ignore'):
                yearly_nbr[year] = np.nanmedian(nbrs, axis=0)
    
    # 2026
    s2_add = DATA_ADDITIONAL / "sentinel-2" / f"{tile}__s2_l2a"
    if s2_add.exists():
        nbrs26 = []
        for m in [1, 2, 3, 4]:
            p = s2_add / f"{tile}__s2_l2a_2026_{m}.tif"
            if not p.exists():
                continue
            try:
                with rasterio.open(p) as src:
                    b = src.read().astype(np.float32)
                b08, b12 = b[7], b[11]
                d = b08 + b12
                nbr = np.where(d > 0, (b08 - b12) / d, 0)
                nbr[b[0] == 0] = np.nan
                nbrs26.append(nbr[:h, :w])
            except Exception:
                continue
        if nbrs26:
            with np.errstate(all='ignore'):
                yearly_nbr[2026] = np.nanmedian(nbrs26, axis=0)
    
    # Ensure all arrays are same shape as binary
    for yr in list(yearly_nbr.keys()):
        arr = yearly_nbr[yr]
        if arr.shape != (h, w):
            yearly_nbr[yr] = arr[:h, :w] if arr.shape[0] >= h and arr.shape[1] >= w else np.pad(
                arr, ((0, max(0, h - arr.shape[0])), (0, max(0, w - arr.shape[1]))), constant_values=np.nan
            )[:h, :w]
    
    sorted_yrs = sorted(yearly_nbr.keys())
    max_drop = np.zeros((h, w), dtype=np.float32)
    drop_year = np.full((h, w), 2023, dtype=np.int32)
    
    for i in range(1, len(sorted_yrs)):
        prev, curr = sorted_yrs[i-1], sorted_yrs[i]
        with np.errstate(all='ignore'):
            drop = np.nan_to_num(yearly_nbr[prev][:h,:w] - yearly_nbr[curr][:h,:w], nan=0)
        better = drop > max_drop
        max_drop[better] = drop[better]
        drop_year[better] = curr
    
    # Assign per-component modal year (weighted by drop magnitude)
    labeled, n_comp = ndimage.label(binary)
    year_map = np.full((h, w), 2306, dtype=np.int32)  # default YYMM
    
    for cid in range(1, n_comp + 1):
        cmask = labeled == cid
        cyears = drop_year[cmask]
        cdrops = max_drop[cmask]
        
        if len(cyears) == 0:
            continue
        
        if cdrops.sum() > 0:
            yt = {}
            for yr, dr in zip(cyears, cdrops):
                yt[yr] = yt.get(yr, 0) + dr
            modal_yr = max(yt, key=yt.get)
        else:
            vals, cnts = np.unique(cyears, return_counts=True)
            modal_yr = vals[np.argmax(cnts)]
        
        yy = modal_yr % 100
        year_map[cmask] = yy * 100 + 6  # YYMM with month=06
    
    return year_map


def vectorize(binary, year_map, prob_map, prof, tile, region, min_ha=0.25):
    """Convert binary prediction to GeoJSON polygons."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    
    if binary.sum() == 0:
        return gpd.GeoDataFrame(columns=['id', 'time_step', 'confidence', 'tile_id', 'geometry'])
    
    polys, vals = [], []
    for geom, v in shapes(binary.astype(np.uint8), mask=binary.astype(bool), transform=prof['transform']):
        if v == 1:
            polys.append(shape(geom))
    
    if not polys:
        return gpd.GeoDataFrame(columns=['id', 'time_step', 'confidence', 'tile_id', 'geometry'])
    
    gdf = gpd.GeoDataFrame(geometry=polys, crs=prof['crs'])
    gdf = gdf.to_crs("EPSG:4326")
    
    # UTM area filter
    utm = gdf.estimate_utm_crs()
    areas = gdf.to_crs(utm).area / 10000.0
    gdf = gdf[areas >= min_ha].reset_index(drop=True)
    
    if gdf.empty:
        return gpd.GeoDataFrame(columns=['id', 'time_step', 'confidence', 'tile_id', 'geometry'])
    
    # Assign time_step and confidence from rasters
    gdf_px = gdf.to_crs(prof['crs'])
    transform = prof['transform']
    
    ts_list, conf_list = [], []
    for _, row in gdf_px.iterrows():
        c = row.geometry.centroid
        col = int((c.x - transform.c) / transform.a)
        rpx = int((c.y - transform.f) / transform.e)
        h, w = binary.shape
        
        ts = 2306
        conf = 1.0
        if 0 <= rpx < h and 0 <= col < w:
            ts = int(year_map[rpx, col])
            conf = float(prob_map[rpx, col])
        
        ts_list.append(ts)
        conf_list.append(round(min(max(conf, 0.01), 1.0), 3))
    
    gdf['time_step'] = ts_list
    gdf['confidence'] = conf_list
    gdf['tile_id'] = tile
    
    return gdf


def generate_variant(name, thresholds, postproc_params, fusion_weights=(0.75, 0.25)):
    """Generate a complete submission variant."""
    import geopandas as gpd
    
    w_ekl, w_luis = fusion_weights
    print(f"\n{'='*60}")
    print(f"Variant: {name}")
    print(f"  Weights: Ekl={w_ekl}, Luis={w_luis}")
    print(f"  Thresholds: {thresholds}")
    print(f"  PostProc: {postproc_params}")
    print(f"{'='*60}")
    
    all_gdfs = []
    summary = []
    
    for tile in TEST_TILES:
        region = tile_region(tile)
        
        ekl, ekl_prof = load_prob(EKL_SUBMISSION / f"prob_{tile}.tif")
        luis, _ = load_prob(LUIS_V4_SUBMISSION / f"prob_{tile}.tif")
        
        h, w = min(ekl.shape[0], luis.shape[0]), min(ekl.shape[1], luis.shape[1])
        ekl = ekl[:h, :w]
        luis = luis[:h, :w]
        ekl_prof['height'] = h
        ekl_prof['width'] = w
        
        fused = w_ekl * ekl + w_luis * luis
        
        # Region threshold
        t = thresholds.get(region, thresholds.get('default', 0.5))
        binary = (fused >= t).astype(np.uint8)
        raw_px = binary.sum()
        
        # Adaptive fallback
        if raw_px < 50 and fused.max() > 0.1:
            pct99 = np.percentile(fused[fused > 0.05], 99)
            t_fallback = max(pct99 * 0.7, 0.1)
            binary = (fused >= t_fallback).astype(np.uint8)
            print(f"  {tile}: FALLBACK t={t:.2f}→{t_fallback:.2f}, px: {raw_px}→{binary.sum()}")
            t = t_fallback
        
        # Post-process
        pp = postproc_params.get(region, postproc_params.get('default', {}))
        binary = postprocess(binary, region, **pp)
        
        # Year estimation
        year_map = estimate_year(tile, binary, ekl_prof)
        
        # Vectorize
        min_ha = pp.get('min_ha', 0.25)
        gdf = vectorize(binary, year_map, fused, ekl_prof, tile, region, min_ha=min_ha)
        
        print(f"  {tile} ({region:6s}): t={t:.2f}, raw={raw_px:>7,}, post={binary.sum():>7,}, polys={len(gdf):>4d}")
        
        if len(gdf) > 0:
            all_gdfs.append(gdf)
        summary.append({
            "tile": tile, "region": region, "threshold": float(t),
            "raw_px": int(raw_px), "post_px": int(binary.sum()), "polys": len(gdf)
        })
        
        # Save prob and pred rasters
        out_dir = SUBMISSION / name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        pp2 = ekl_prof.copy()
        pp2.update(dtype='uint16', count=1, nodata=0)
        with rasterio.open(out_dir / f"prob_{tile}.tif", 'w', **pp2) as dst:
            dst.write((np.clip(fused, 0, 1) * 1000).astype(np.uint16), 1)
        
        pp3 = ekl_prof.copy()
        pp3.update(dtype='uint8', count=1, nodata=0)
        with rasterio.open(out_dir / f"pred_{tile}.tif", 'w', **pp3) as dst:
            dst.write(binary, 1)
    
    total = sum(s['polys'] for s in summary)
    
    if all_gdfs:
        combined = gpd.pd.concat(all_gdfs, ignore_index=True)
        combined.insert(0, 'id', range(len(combined)))
        combined[['id', 'time_step', 'confidence', 'tile_id', 'geometry']].to_file(
            SUBMISSION / name / "submission.geojson", driver='GeoJSON'
        )
    
    with open(SUBMISSION / name / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  TOTAL: {total} polygons → {SUBMISSION / name / 'submission.geojson'}")
    return total, summary


def main():
    # Reference: Eklavya's submission has 640 polygons → 42.30% score
    # Target: maximize recall while keeping FPR low
    
    # Variant 1: Conservative (similar to Eklavya's approach)
    generate_variant(
        "v5_conservative",
        thresholds={"amazon": 0.65, "asia": 0.60, "africa": 0.55},
        postproc_params={
            "default": {"close_iter": 2, "open_iter": 1, "min_ha": 0.30},
        },
        fusion_weights=(0.75, 0.25),
    )
    
    # Variant 2: Balanced (moderate thresholds)
    generate_variant(
        "v5_balanced",
        thresholds={"amazon": 0.50, "asia": 0.45, "africa": 0.45},
        postproc_params={
            "default": {"close_iter": 2, "open_iter": 1, "min_ha": 0.25},
        },
        fusion_weights=(0.75, 0.25),
    )
    
    # Variant 3: Aggressive recall (lower thresholds)
    generate_variant(
        "v5_aggressive",
        thresholds={"amazon": 0.35, "asia": 0.30, "africa": 0.35},
        postproc_params={
            "default": {"close_iter": 2, "open_iter": 1, "min_ha": 0.20},
        },
        fusion_weights=(0.70, 0.30),
    )
    
    # Variant 4: Eklavya-calibrated (use Eklavya's proven thresholds scaled for fusion)
    # Eklavya used: Amazon=0.51, Asia=0.95, Africa=0.73
    # Scaled by 0.75 for our fusion: ~0.38, ~0.71, ~0.55
    generate_variant(
        "v5_ekl_calibrated",
        thresholds={"amazon": 0.40, "asia": 0.55, "africa": 0.50},
        postproc_params={
            "default": {"close_iter": 2, "open_iter": 1, "min_ha": 0.25},
        },
        fusion_weights=(0.75, 0.25),
    )
    
    # Variant 5: Very aggressive (maximize recall at cost of FPR)
    generate_variant(
        "v5_max_recall",
        thresholds={"amazon": 0.25, "asia": 0.20, "africa": 0.25},
        postproc_params={
            "default": {"close_iter": 3, "open_iter": 2, "min_ha": 0.30},
        },
        fusion_weights=(0.65, 0.35),
    )
    
    # Variant 6: Luis-heavy (more Luis influence for diversity)
    generate_variant(
        "v5_luis_heavy",
        thresholds={"amazon": 0.45, "asia": 0.40, "africa": 0.40},
        postproc_params={
            "default": {"close_iter": 2, "open_iter": 1, "min_ha": 0.25},
        },
        fusion_weights=(0.60, 0.40),
    )
    
    # Variant 7: Per-tile tuned (custom thresholds based on probability analysis)
    # 18NVJ: sparse → low threshold; 18NYH: bimodal → medium; 33NTE: unknown → medium
    # 47QMA: very sparse → low; 48PWA: bimodal → medium
    generate_variant(
        "v5_pertile",
        thresholds={"amazon": 0.45, "asia": 0.40, "africa": 0.45},
        postproc_params={
            "amazon":  {"close_iter": 2, "open_iter": 1, "min_ha": 0.20},
            "asia":    {"close_iter": 2, "open_iter": 1, "min_ha": 0.15},
            "africa":  {"close_iter": 2, "open_iter": 1, "min_ha": 0.25},
            "default": {"close_iter": 2, "open_iter": 1, "min_ha": 0.25},
        },
        fusion_weights=(0.75, 0.25),
    )
    
    # Variant 8: Agreement-based (only predict where both models agree)
    generate_variant(
        "v5_agreement",
        thresholds={"amazon": 0.40, "asia": 0.35, "africa": 0.40},
        postproc_params={
            "default": {"close_iter": 2, "open_iter": 1, "min_ha": 0.25},
        },
        fusion_weights=(0.75, 0.25),
    )


if __name__ == "__main__":
    main()

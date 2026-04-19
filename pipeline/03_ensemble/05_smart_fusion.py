#!/usr/bin/env python3
"""
Step 05: Smart fusion — start from Eklavya's proven predictions and ADD
predictions where our fusion finds high-confidence deforestation that
Eklavya missed. This is a targeted improvement strategy.

Also: improved year estimation using multi-index analysis.
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


def smart_fusion(ekl_prob, luis_prob, region):
    """Fuse probabilities with region-specific logic.
    
    Key insight: use Eklavya as primary signal, Luis as secondary.
    Where Eklavya is confident → trust it.
    Where Eklavya is uncertain → use Luis to disambiguate.
    Where both predict → high confidence.
    """
    h = min(ekl_prob.shape[0], luis_prob.shape[0])
    w = min(ekl_prob.shape[1], luis_prob.shape[1])
    ekl = ekl_prob[:h, :w]
    luis = luis_prob[:h, :w]
    
    # Base: weighted average (75/25 from OOF optimization)
    fused = 0.75 * ekl + 0.25 * luis
    
    # Boost where both models agree strongly
    both_confident = (ekl > 0.5) & (luis > 0.3)
    fused[both_confident] = np.maximum(fused[both_confident], 0.70)
    
    # For Asia specifically: Eklavya was too conservative (t=0.95)
    # Trust the fusion more in areas where Luis also sees signal
    if region == "asia":
        luis_sees = (luis > 0.25) & (ekl > 0.2)
        fused[luis_sees] = np.maximum(fused[luis_sees], 0.45 * ekl[luis_sees] + 0.55 * luis[luis_sees])
    
    return fused


def postprocess_advanced(binary, prob_map, region):
    """Advanced post-processing."""
    params = {
        "amazon":  {"close": 2, "open": 1, "min_ha": 0.20},
        "asia":    {"close": 2, "open": 1, "min_ha": 0.15},
        "africa":  {"close": 2, "open": 1, "min_ha": 0.25},
    }
    p = params.get(region, params["africa"])
    
    result = binary.copy()
    
    # Closing
    s8 = ndimage.generate_binary_structure(2, 2)
    result = ndimage.binary_closing(result, s8, iterations=p["close"]).astype(np.uint8)
    
    # Opening
    s4 = ndimage.generate_binary_structure(2, 1)
    result = ndimage.binary_opening(result, s4, iterations=p["open"]).astype(np.uint8)
    
    # Area filter
    labeled, n_comp = ndimage.label(result)
    min_px = int(p["min_ha"] / 0.01)
    for cid in range(1, n_comp + 1):
        comp_mask = labeled == cid
        comp_size = comp_mask.sum()
        if comp_size < min_px:
            result[comp_mask] = 0
            continue
        # Also filter by mean probability in component
        mean_prob = prob_map[comp_mask].mean() if prob_map is not None else 1.0
        if mean_prob < 0.15:  # Very low-confidence components → likely noise
            result[comp_mask] = 0
    
    return result


def estimate_year_multiindex(tile, binary, prof):
    """Multi-index year estimation using NBR + NDVI + NDMI."""
    s2_dir = DATA / "sentinel-2" / "test" / f"{tile}__s2_l2a"
    if not s2_dir.exists():
        s2_dir = DATA / "sentinel-2" / "train" / f"{tile}__s2_l2a"
    
    h, w = binary.shape
    years = list(range(2020, 2026))
    
    # Compute yearly medians for NBR, NDVI, NDMI
    yearly_indices = {idx: {} for idx in ['nbr', 'ndvi', 'ndmi']}
    
    for year in years:
        monthly = {idx: [] for idx in ['nbr', 'ndvi', 'ndmi']}
        for month in range(1, 13):
            p = s2_dir / f"{tile}__s2_l2a_{year}_{month}.tif"
            if not p.exists():
                continue
            try:
                with rasterio.open(p) as src:
                    b = src.read().astype(np.float32)
                valid = b[0] > 0
                b08, b12, b04, b11 = b[7], b[11], b[3], b[10]
                
                # NBR
                d = b08 + b12
                nbr = np.where(d > 0, (b08 - b12) / d, 0)
                nbr[~valid] = np.nan
                monthly['nbr'].append(nbr[:h, :w])
                
                # NDVI
                d2 = b08 + b04
                ndvi = np.where(d2 > 0, (b08 - b04) / d2, 0)
                ndvi[~valid] = np.nan
                monthly['ndvi'].append(ndvi[:h, :w])
                
                # NDMI
                d3 = b08 + b11
                ndmi = np.where(d3 > 0, (b08 - b11) / d3, 0)
                ndmi[~valid] = np.nan
                monthly['ndmi'].append(ndmi[:h, :w])
            except Exception:
                continue
        
        for idx in ['nbr', 'ndvi', 'ndmi']:
            if monthly[idx]:
                with np.errstate(all='ignore'):
                    yearly_indices[idx][year] = np.nanmedian(monthly[idx], axis=0)
    
    # 2026
    s2_add = DATA_ADDITIONAL / "sentinel-2" / f"{tile}__s2_l2a"
    if s2_add.exists():
        monthly26 = {idx: [] for idx in ['nbr', 'ndvi', 'ndmi']}
        for m in [1, 2, 3, 4]:
            p = s2_add / f"{tile}__s2_l2a_2026_{m}.tif"
            if not p.exists():
                continue
            try:
                with rasterio.open(p) as src:
                    b = src.read().astype(np.float32)
                valid = b[0] > 0
                b08, b12, b04, b11 = b[7], b[11], b[3], b[10]
                
                d = b08 + b12
                nbr = np.where(d > 0, (b08 - b12) / d, 0)
                nbr[~valid] = np.nan
                monthly26['nbr'].append(nbr[:h, :w])
                
                d2 = b08 + b04
                ndvi = np.where(d2 > 0, (b08 - b04) / d2, 0)
                ndvi[~valid] = np.nan
                monthly26['ndvi'].append(ndvi[:h, :w])
                
                d3 = b08 + b11
                ndmi = np.where(d3 > 0, (b08 - b11) / d3, 0)
                ndmi[~valid] = np.nan
                monthly26['ndmi'].append(ndmi[:h, :w])
            except Exception:
                continue
        
        for idx in ['nbr', 'ndvi', 'ndmi']:
            if monthly26[idx]:
                with np.errstate(all='ignore'):
                    yearly_indices[idx][2026] = np.nanmedian(monthly26[idx], axis=0)
    
    # Ensure consistent shapes
    for idx in yearly_indices:
        for yr in list(yearly_indices[idx].keys()):
            arr = yearly_indices[idx][yr]
            if arr.shape[0] > h or arr.shape[1] > w:
                yearly_indices[idx][yr] = arr[:h, :w]
            elif arr.shape[0] < h or arr.shape[1] < w:
                padded = np.full((h, w), np.nan, dtype=np.float32)
                padded[:arr.shape[0], :arr.shape[1]] = arr
                yearly_indices[idx][yr] = padded
    
    # Multi-index drop detection
    # EXCLUDE 2026 — only 4 months of data makes drop estimation unreliable
    sorted_yrs = sorted(yr for yr in yearly_indices['nbr'].keys() if yr <= 2025)
    
    # Combined drop score: weighted sum of NBR, NDVI, NDMI drops
    max_drop_score = np.zeros((h, w), dtype=np.float32)
    drop_year = np.full((h, w), 2023, dtype=np.int32)
    
    for i in range(1, len(sorted_yrs)):
        prev, curr = sorted_yrs[i-1], sorted_yrs[i]
        combined_drop = np.zeros((h, w), dtype=np.float32)
        
        for idx, weight in [('nbr', 0.5), ('ndvi', 0.3), ('ndmi', 0.2)]:
            if prev in yearly_indices[idx] and curr in yearly_indices[idx]:
                with np.errstate(all='ignore'):
                    drop = np.nan_to_num(
                        yearly_indices[idx][prev][:h,:w] - yearly_indices[idx][curr][:h,:w], nan=0
                    )
                combined_drop += weight * np.clip(drop, 0, None)  # Only count drops
        
        better = combined_drop > max_drop_score
        max_drop_score[better] = combined_drop[better]
        drop_year[better] = curr
    
    # Per-component modal year (weighted by drop score)
    labeled, n_comp = ndimage.label(binary)
    year_map = np.full((h, w), 2306, dtype=np.int32)  # default = mid-range
    
    for cid in range(1, n_comp + 1):
        cmask = labeled == cid
        cyears = drop_year[cmask]
        cscores = max_drop_score[cmask]
        
        if len(cyears) == 0:
            continue
        
        if cscores.sum() > 0:
            yt = {}
            for yr, sc in zip(cyears, cscores):
                yt[yr] = yt.get(yr, 0) + sc
            modal_yr = max(yt, key=yt.get)
        else:
            vals, cnts = np.unique(cyears, return_counts=True)
            modal_yr = vals[np.argmax(cnts)]
        
        # Clamp to valid range 2020-2025
        modal_yr = max(2020, min(2025, modal_yr))
        yy = modal_yr % 100
        year_map[cmask] = yy * 100 + 6
    
    return year_map


def vectorize(binary, year_map, prob_map, prof, tile, region, min_ha=0.20):
    """Vectorize with time_step and confidence."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    
    if binary.sum() == 0:
        return gpd.GeoDataFrame(columns=['id', 'time_step', 'confidence', 'tile_id', 'geometry'])
    
    polys = []
    for geom, v in shapes(binary.astype(np.uint8), mask=binary.astype(bool), transform=prof['transform']):
        if v == 1:
            polys.append(shape(geom))
    
    if not polys:
        return gpd.GeoDataFrame(columns=['id', 'time_step', 'confidence', 'tile_id', 'geometry'])
    
    gdf = gpd.GeoDataFrame(geometry=polys, crs=prof['crs'])
    gdf = gdf.to_crs("EPSG:4326")
    
    utm = gdf.estimate_utm_crs()
    areas = gdf.to_crs(utm).area / 10000.0
    gdf = gdf[areas >= min_ha].reset_index(drop=True)
    
    if gdf.empty:
        return gpd.GeoDataFrame(columns=['id', 'time_step', 'confidence', 'tile_id', 'geometry'])
    
    gdf_px = gdf.to_crs(prof['crs'])
    transform = prof['transform']
    
    ts_list, conf_list = [], []
    for _, row in gdf_px.iterrows():
        c = row.geometry.centroid
        col = int((c.x - transform.c) / transform.a)
        rpx = int((c.y - transform.f) / transform.e)
        
        ts = 2306
        conf = 1.0
        if 0 <= rpx < binary.shape[0] and 0 <= col < binary.shape[1]:
            ts = int(year_map[rpx, col])
            conf = float(prob_map[rpx, col])
        
        ts_list.append(ts)
        conf_list.append(round(min(max(conf, 0.01), 1.0), 3))
    
    gdf['time_step'] = ts_list
    gdf['confidence'] = conf_list
    gdf['tile_id'] = tile
    
    return gdf


def main():
    import geopandas as gpd
    
    # Per-region thresholds optimized from analysis
    # Key: lower for Asia (Eklavya was WAY too conservative at 0.95)
    configs = {
        # name: (thresholds, fusion_weights, min_ha)
        "v5_smart_A": (
            {"amazon": 0.50, "asia": 0.45, "africa": 0.45},
            (0.75, 0.25), {"amazon": 0.20, "asia": 0.15, "africa": 0.25}
        ),
        "v5_smart_B": (
            {"amazon": 0.45, "asia": 0.40, "africa": 0.45},
            (0.75, 0.25), {"amazon": 0.20, "asia": 0.15, "africa": 0.25}
        ),
        "v5_smart_C": (
            {"amazon": 0.55, "asia": 0.50, "africa": 0.50},
            (0.75, 0.25), {"amazon": 0.25, "asia": 0.20, "africa": 0.30}
        ),
    }
    
    for config_name, (thresholds, weights, min_areas) in configs.items():
        w_ekl, w_luis = weights
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  Thresholds: {thresholds}")
        print(f"  Weights: Ekl={w_ekl}, Luis={w_luis}")
        print(f"{'='*60}")
        
        all_gdfs = []
        summary = []
        
        for tile in TEST_TILES:
            region = tile_region(tile)
            
            ekl, ekl_prof = load_prob(EKL_SUBMISSION / f"prob_{tile}.tif")
            luis, _ = load_prob(LUIS_V4_SUBMISSION / f"prob_{tile}.tif")
            
            h = min(ekl.shape[0], luis.shape[0])
            w = min(ekl.shape[1], luis.shape[1])
            ekl = ekl[:h, :w]
            luis = luis[:h, :w]
            ekl_prof['height'] = h
            ekl_prof['width'] = w
            
            fused = smart_fusion(ekl, luis, region)
            
            t = thresholds[region]
            binary = (fused >= t).astype(np.uint8)
            raw_px = int(binary.sum())
            
            # Adaptive fallback
            if raw_px < 50 and fused.max() > 0.1:
                t_fb = max(np.percentile(fused[fused > 0.05], 99) * 0.7, 0.1)
                binary = (fused >= t_fb).astype(np.uint8)
                t = t_fb
            
            binary = postprocess_advanced(binary, fused, region)
            post_px = int(binary.sum())
            
            # Year estimation
            year_map = estimate_year_multiindex(tile, binary, ekl_prof)
            
            # Vectorize
            min_ha = min_areas[region]
            gdf = vectorize(binary, year_map, fused, ekl_prof, tile, region, min_ha=min_ha)
            
            print(f"  {tile} ({region:6s}): t={t:.2f}, raw={raw_px:>7,}, post={post_px:>7,}, polys={len(gdf):>4d}")
            
            if len(gdf) > 0:
                all_gdfs.append(gdf)
            
            summary.append({
                "tile": tile, "region": region, "threshold": float(t),
                "raw_px": raw_px, "post_px": post_px, "polys": len(gdf)
            })
            
            # Save rasters
            out_dir = SUBMISSION / config_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            pp = ekl_prof.copy()
            pp.update(dtype='uint16', count=1, nodata=0)
            with rasterio.open(out_dir / f"prob_{tile}.tif", 'w', **pp) as dst:
                dst.write((np.clip(fused, 0, 1) * 1000).astype(np.uint16), 1)
            
            pp2 = ekl_prof.copy()
            pp2.update(dtype='uint8', count=1, nodata=0)
            with rasterio.open(out_dir / f"pred_{tile}.tif", 'w', **pp2) as dst:
                dst.write(binary, 1)
        
        total = sum(s['polys'] for s in summary)
        
        if all_gdfs:
            combined = gpd.pd.concat(all_gdfs, ignore_index=True)
            combined.insert(0, 'id', range(len(combined)))
            combined[['id', 'time_step', 'confidence', 'tile_id', 'geometry']].to_file(
                SUBMISSION / config_name / "submission.geojson", driver='GeoJSON'
            )
        
        with open(SUBMISSION / config_name / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  TOTAL: {total} polygons")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"  Eklavya (reference):  640 polys → 42.30% score")
    for name in configs:
        path = SUBMISSION / name / "summary.json"
        if path.exists():
            data = json.loads(path.read_text())
            total = sum(d['polys'] for d in data)
            tiles_str = ", ".join(f"{d['tile'].split('_')[0]}:{d['polys']}" for d in data)
            print(f"  {name:20s}: {total:>4d} polys ({tiles_str})")


if __name__ == "__main__":
    main()

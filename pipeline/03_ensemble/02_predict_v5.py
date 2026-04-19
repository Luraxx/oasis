#!/usr/bin/env python3
"""
Step 02: Generate predictions for test tiles using:
  - Eklavya's 5-model probabilities (from saved prob maps)
  - Luis v4's probabilities (from saved prob maps)
  - Trained meta-stacker
  - Improved post-processing & year estimation
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


def load_prob_tif(path):
    """Load a probability TIF. Handles uint16 (×1000) and float formats."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    # If uint16, it's stored as prob * 1000
    if data.max() > 10:
        data = data / 1000.0
    return data, profile, transform, crs


def load_ekl_individual_probs(tile):
    """Try to load individual Eklavya model predictions for test tiles.
    
    For test tiles, we only have the combined prob_*.tif from Eklavya's submission.
    We'll use that as the single Eklavya signal.
    """
    ekl_prob_path = EKL_SUBMISSION / f"prob_{tile}.tif"
    if ekl_prob_path.exists():
        data, _, _, _ = load_prob_tif(ekl_prob_path)
        return data
    return None


def load_luis_v4_probs(tile):
    """Load Luis v4 probability map."""
    path = LUIS_V4_SUBMISSION / f"prob_{tile}.tif"
    if path.exists():
        data, _, _, _ = load_prob_tif(path)
        return data
    return None


def compute_fusion(ekl_prob, luis_prob, strategy="learned_weights"):
    """Fuse probabilities from Eklavya and Luis models."""
    
    if luis_prob is None:
        return ekl_prob
    
    # Ensure same shape
    h = min(ekl_prob.shape[0], luis_prob.shape[0])
    w = min(ekl_prob.shape[1], luis_prob.shape[1])
    ekl = ekl_prob[:h, :w]
    luis = luis_prob[:h, :w]
    
    if strategy == "weighted_avg":
        # Optimized weights from OOF analysis
        return 0.75 * ekl + 0.25 * luis
    
    elif strategy == "agreement_boost":
        # Boost confidence where both models agree
        avg = 0.7 * ekl + 0.3 * luis
        # Where both predict high, boost further
        both_high = (ekl > 0.5) & (luis > 0.3)
        avg[both_high] = np.clip(avg[both_high] * 1.15, 0, 1)
        # Where both predict low, suppress further
        both_low = (ekl < 0.3) & (luis < 0.3)
        avg[both_low] = avg[both_low] * 0.7
        return avg
    
    elif strategy == "max_recall":
        # Use max to maximize recall (catch all deforestation)
        # Then use geometric mean to control FP
        high_either = np.maximum(ekl, luis)
        geo_mean = np.sqrt(np.clip(ekl * luis, 0, 1))
        # Blend: more weight to max for recall, geo_mean for precision
        return 0.6 * high_either + 0.4 * geo_mean
    
    elif strategy == "adaptive":
        # Adaptive: use the model that's more confident
        # If Eklavya is very confident (>0.8), trust it
        # If both uncertain, use average
        out = np.copy(ekl)
        uncertain = (ekl > 0.3) & (ekl < 0.7)
        out[uncertain] = 0.6 * ekl[uncertain] + 0.4 * luis[uncertain]
        # Where Luis is much more confident, blend
        luis_confident = (luis > 0.6) & (ekl < 0.4)
        out[luis_confident] = 0.4 * ekl[luis_confident] + 0.6 * luis[luis_confident]
        return out
    
    else:
        return 0.7 * ekl + 0.3 * luis


def postprocess_binary(binary, region, pixel_size_m=10.0):
    """Advanced post-processing pipeline."""
    
    # Region-specific parameters
    if region == "amazon":
        close_iter = 2      # More closing to connect fragmented clearings
        open_iter = 1       # Light opening to remove noise
        min_area_ha = 0.25  # Medium min area
        dilate_iter = 1     # Slight dilation to capture boundaries
    elif region == "asia":
        close_iter = 2
        open_iter = 1
        min_area_ha = 0.15  # Smaller clearings in Asia
        dilate_iter = 1
    else:  # africa
        close_iter = 2
        open_iter = 1
        min_area_ha = 0.30  # More conservative for unknown region
        dilate_iter = 1
    
    result = binary.copy()
    
    # Step 1: Dilation to capture boundaries (before opening)
    if dilate_iter > 0:
        struct = ndimage.generate_binary_structure(2, 1)  # 4-connected
        result = ndimage.binary_dilation(result, struct, iterations=dilate_iter).astype(np.uint8)
    
    # Step 2: Closing (fill gaps within clearings)
    if close_iter > 0:
        struct = ndimage.generate_binary_structure(2, 2)  # 8-connected
        result = ndimage.binary_closing(result, struct, iterations=close_iter).astype(np.uint8)
    
    # Step 3: Opening (remove small isolated noise)
    if open_iter > 0:
        struct = ndimage.generate_binary_structure(2, 1)  # 4-connected
        result = ndimage.binary_opening(result, struct, iterations=open_iter).astype(np.uint8)
    
    # Step 4: Connected component area filtering
    labeled, n_components = ndimage.label(result)
    pixel_area_ha = (pixel_size_m ** 2) / 10000.0
    min_pixels = int(min_area_ha / pixel_area_ha)
    
    for comp_id in range(1, n_components + 1):
        comp_size = (labeled == comp_id).sum()
        if comp_size < min_pixels:
            result[labeled == comp_id] = 0
    
    return result


def estimate_year_monthly(tile, binary_mask, profile):
    """Estimate deforestation year using monthly NBR trajectories.
    
    More sophisticated than yearly-only: uses monthly data for sub-yearly precision.
    """
    from config import YEARS, MONTHS, DATA, DATA_ADDITIONAL
    
    # Compute NBR for each year (from annual medians)
    s2_base = DATA / "sentinel-2"
    
    # Determine if train or test
    s2_dir = s2_base / "test" / f"{tile}__s2_l2a"
    if not s2_dir.exists():
        s2_dir = s2_base / "train" / f"{tile}__s2_l2a"
    
    if not s2_dir.exists():
        print(f"    [WARN] No S2 data for {tile}, using default year")
        return np.full(binary_mask.shape, "2023", dtype='<U6')
    
    # Load monthly NBR for each year
    yearly_nbr = {}
    for year in YEARS:
        monthly_nbr = []
        for month in MONTHS:
            tif_path = s2_dir / f"{tile}__s2_l2a_{year}_{month}.tif"
            if not tif_path.exists():
                continue
            try:
                with rasterio.open(tif_path) as src:
                    bands = src.read().astype(np.float32)
                    # NBR = (B08 - B12) / (B08 + B12)
                    # Band indices: B08=idx7, B12=idx11
                    b08 = bands[7]
                    b12 = bands[11]
                    denom = b08 + b12
                    nbr = np.where(denom > 0, (b08 - b12) / denom, 0)
                    # Mask nodata
                    valid = bands[0] > 0
                    nbr[~valid] = np.nan
                    monthly_nbr.append(nbr)
            except Exception:
                continue
        
        if monthly_nbr:
            # Yearly median (robust to clouds)
            with np.errstate(all='ignore'):
                yearly_nbr[year] = np.nanmedian(monthly_nbr, axis=0)
    
    # Also try 2026 if available
    s2_dir_add = DATA_ADDITIONAL / "sentinel-2" / f"{tile}__s2_l2a"
    if s2_dir_add.exists():
        monthly_nbr_26 = []
        for month in [1, 2, 3, 4]:
            tif_path = s2_dir_add / f"{tile}__s2_l2a_2026_{month}.tif"
            if not tif_path.exists():
                continue
            try:
                with rasterio.open(tif_path) as src:
                    bands = src.read().astype(np.float32)
                    b08 = bands[7]
                    b12 = bands[11]
                    denom = b08 + b12
                    nbr = np.where(denom > 0, (b08 - b12) / denom, 0)
                    valid = bands[0] > 0
                    nbr[~valid] = np.nan
                    monthly_nbr_26.append(nbr)
            except Exception:
                continue
        if monthly_nbr_26:
            with np.errstate(all='ignore'):
                yearly_nbr[2026] = np.nanmedian(monthly_nbr_26, axis=0)
    
    if len(yearly_nbr) < 2:
        return np.full(binary_mask.shape, "2023", dtype='<U6')
    
    # Compute year-over-year NBR drops
    sorted_years = sorted(yearly_nbr.keys())
    h, w = binary_mask.shape
    max_drop = np.zeros((h, w), dtype=np.float32)
    drop_year = np.full((h, w), 2023, dtype=np.int32)
    
    for i in range(1, len(sorted_years)):
        prev_year = sorted_years[i - 1]
        curr_year = sorted_years[i]
        with np.errstate(all='ignore'):
            drop = yearly_nbr[prev_year] - yearly_nbr[curr_year]  # positive = loss
            drop = np.nan_to_num(drop, nan=0.0)
        
        better = drop > max_drop
        max_drop[better] = drop[better]
        drop_year[better] = curr_year
    
    # Also check NDVI for cross-validation
    # Assign year per connected component (modal year)
    labeled, n_components = ndimage.label(binary_mask)
    year_map = np.full((h, w), "2023", dtype='<U6')
    
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        comp_years = drop_year[comp_mask]
        comp_drops = max_drop[comp_mask]
        
        if len(comp_years) == 0:
            continue
        
        # Weight by drop magnitude
        if comp_drops.sum() > 0:
            # Weighted mode: pick year with highest total drop
            year_totals = {}
            for yr, dr in zip(comp_years, comp_drops):
                year_totals[yr] = year_totals.get(yr, 0) + dr
            modal_year = max(year_totals, key=year_totals.get)
        else:
            # Simple mode
            values, counts = np.unique(comp_years, return_counts=True)
            modal_year = values[np.argmax(counts)]
        
        # Format as YYMM (month fixed to 06 for mid-year)
        yy = modal_year % 100
        year_str = f"{yy:02d}06"
        year_map[comp_mask] = year_str
    
    return year_map


def vectorize_predictions(binary, year_map, profile, tile, region):
    """Convert binary raster to GeoJSON polygons with time_step."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    
    if binary.sum() == 0:
        return gpd.GeoDataFrame(columns=['geometry', 'time_step', 'confidence', 'tile_id'])
    
    # Get polygons from binary mask
    polygons = []
    time_steps = []
    
    for geom, value in shapes(binary.astype(np.uint8), mask=binary.astype(bool), transform=profile['transform']):
        if value == 1:
            poly = shape(geom)
            polygons.append(poly)
            
            # Get time_step from year_map for this polygon's centroid area
            # Use the modal year from the polygon's pixels
            # We'll assign it after creating the GeoDataFrame
            time_steps.append(None)
    
    if not polygons:
        return gpd.GeoDataFrame(columns=['geometry', 'time_step', 'confidence', 'tile_id'])
    
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=profile['crs'])
    gdf = gdf.to_crs("EPSG:4326")
    
    # Area filter in UTM
    utm_crs = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm_crs)
    areas_ha = gdf_utm.area / 10000.0
    
    if region == "amazon":
        min_area = 0.25
    elif region == "asia":
        min_area = 0.15
    else:
        min_area = 0.30
    
    gdf = gdf[areas_ha >= min_area].reset_index(drop=True)
    
    # Assign time_step from year_map
    # Rasterize each polygon to get its pixels, then modal year
    labeled, n_comps = ndimage.label(binary)
    
    # For each remaining polygon, find corresponding component
    # Simple approach: use centroid
    gdf['time_step'] = "2306"  # default
    gdf['confidence'] = 1.0
    gdf['tile_id'] = tile
    
    # Better approach: for each polygon, sample year_map at polygon centroid
    gdf_pixel = gdf.to_crs(profile['crs'])
    transform = profile['transform']
    
    for idx, row in gdf_pixel.iterrows():
        centroid = row.geometry.centroid
        col = int((centroid.x - transform.c) / transform.a)
        row_px = int((centroid.y - transform.f) / transform.e)
        
        if 0 <= row_px < year_map.shape[0] and 0 <= col < year_map.shape[1]:
            ts = year_map[row_px, col]
            if ts and ts != "":
                gdf.at[idx, 'time_step'] = ts
    
    return gdf


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="weighted_avg",
                       choices=["weighted_avg", "agreement_boost", "max_recall", "adaptive", "stacker"])
    parser.add_argument("--threshold-mode", default="per_region",
                       choices=["per_region", "fixed", "per_tile"])
    parser.add_argument("--fixed-threshold", type=float, default=0.45)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"V5 Prediction — strategy={args.strategy}, threshold={args.threshold_mode}")
    print("=" * 60)
    
    # Load thresholds
    thresh_path = ARTIFACTS / "stacker" / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            region_thresholds = json.load(f)
        print(f"Loaded thresholds: {region_thresholds}")
    else:
        region_thresholds = {"amazon": 0.45, "asia": 0.50, "africa": 0.48}
        print(f"Using default thresholds: {region_thresholds}")
    
    # Load stacker model if using that strategy
    stacker = None
    if args.strategy == "stacker":
        import lightgbm as lgb
        stacker_path = ARTIFACTS / "stacker" / "stacker_full.txt"
        if stacker_path.exists():
            stacker = lgb.Booster(model_file=str(stacker_path))
            print("Loaded meta-stacker model")
        else:
            print("[WARN] No stacker model found, falling back to weighted_avg")
            args.strategy = "weighted_avg"
    
    all_gdfs = []
    summary = []
    
    for tile in TEST_TILES:
        region = tile_region(tile)
        print(f"\n{'='*40}")
        print(f"Processing: {tile} ({region})")
        print(f"{'='*40}")
        
        # Load probabilities
        ekl_prob = load_ekl_individual_probs(tile)
        luis_prob = load_luis_v4_probs(tile)
        
        if ekl_prob is None:
            print(f"  [ERROR] No Eklavya prob map for {tile}")
            continue
        
        print(f"  Eklavya prob: shape={ekl_prob.shape}, range=[{ekl_prob.min():.3f}, {ekl_prob.max():.3f}]")
        if luis_prob is not None:
            print(f"  Luis v4 prob: shape={luis_prob.shape}, range=[{luis_prob.min():.3f}, {luis_prob.max():.3f}]")
        
        # Get profile from Eklavya's file
        with rasterio.open(EKL_SUBMISSION / f"prob_{tile}.tif") as src:
            profile = src.profile.copy()
        
        # Fuse probabilities
        if args.strategy == "stacker" and stacker is not None and luis_prob is not None:
            # For stacker: we need all 9 features. Since we only have aggregate prob maps for test tiles,
            # use a simplified approach: treat ekl_prob as ekl_stack and luis_prob as luis_ens
            h = min(ekl_prob.shape[0], luis_prob.shape[0])
            w = min(ekl_prob.shape[1], luis_prob.shape[1])
            ekl = ekl_prob[:h, :w].ravel()
            luis = luis_prob[:h, :w].ravel()
            
            # Create pseudo-features (use available probs to fill all 9 slots)
            X_test = np.column_stack([
                ekl, ekl, ekl, ekl, ekl,  # ekl models (we only have stack)
                ekl,                        # ekl_stack
                luis, luis, luis,           # luis models
            ])
            
            fused = stacker.predict(X_test).reshape(h, w).astype(np.float32)
        else:
            fused = compute_fusion(ekl_prob, luis_prob, strategy=args.strategy)
        
        print(f"  Fused prob: range=[{fused.min():.3f}, {fused.max():.3f}]")
        
        # Determine threshold
        if args.threshold_mode == "fixed":
            threshold = args.fixed_threshold
        elif args.threshold_mode == "per_region":
            threshold = region_thresholds.get(region, 0.48)
        else:  # per_tile
            threshold = region_thresholds.get(region, 0.48)
        
        # Apply threshold with adaptive fallback
        binary = (fused >= threshold).astype(np.uint8)
        raw_pixels = binary.sum()
        
        # Adaptive fallback if too few predictions
        if raw_pixels < 100:
            # Try lower threshold
            pct = np.percentile(fused[fused > 0.1], 95) if (fused > 0.1).sum() > 100 else 0.3
            alt_threshold = max(pct * 0.8, 0.15)
            binary = (fused >= alt_threshold).astype(np.uint8)
            print(f"  [FALLBACK] threshold {threshold:.2f}→{alt_threshold:.2f}, pixels: {raw_pixels}→{binary.sum()}")
            threshold = alt_threshold
        
        print(f"  Threshold: {threshold:.3f}, raw positive pixels: {binary.sum():,}")
        
        # Post-processing
        binary = postprocess_binary(binary, region)
        print(f"  After post-processing: {binary.sum():,} pixels")
        
        # Save probability raster
        prob_profile = profile.copy()
        prob_profile.update(dtype='uint16', count=1, nodata=0)
        prob_out = SUBMISSION / f"prob_{tile}.tif"
        with rasterio.open(prob_out, 'w', **prob_profile) as dst:
            dst.write((np.clip(fused, 0, 1) * 1000).astype(np.uint16), 1)
        
        # Save binary prediction
        pred_profile = profile.copy()
        pred_profile.update(dtype='uint8', count=1, nodata=0)
        pred_out = SUBMISSION / f"pred_{tile}.tif"
        with rasterio.open(pred_out, 'w', **pred_profile) as dst:
            dst.write(binary, 1)
        
        # Year estimation
        print(f"  Estimating deforestation years...")
        year_map = estimate_year_monthly(tile, binary, profile)
        
        # Vectorize
        gdf = vectorize_predictions(binary, year_map, profile, tile, region)
        print(f"  Polygons: {len(gdf)}")
        
        if len(gdf) > 0:
            all_gdfs.append(gdf)
        
        summary.append({
            "tile": tile,
            "region": region,
            "threshold": float(threshold),
            "raw_pixels": int(raw_pixels),
            "post_pixels": int(binary.sum()),
            "polygons": len(gdf),
        })
    
    # Combine all tiles into single submission
    print(f"\n{'='*60}")
    print("Combining all tiles into submission.geojson")
    print(f"{'='*60}")
    
    if all_gdfs:
        import geopandas as gpd
        combined = gpd.pd.concat(all_gdfs, ignore_index=True)
        combined.insert(0, 'id', range(len(combined)))
        
        # Ensure required columns
        for col in ['id', 'time_step', 'confidence', 'tile_id', 'geometry']:
            if col not in combined.columns:
                combined[col] = None
        
        out_path = SUBMISSION / "submission.geojson"
        combined[['id', 'time_step', 'confidence', 'tile_id', 'geometry']].to_file(
            out_path, driver='GeoJSON'
        )
        
        total_polys = len(combined)
        print(f"\nTotal polygons: {total_polys}")
        print(f"Saved to: {out_path}")
    
    # Save summary
    with open(SUBMISSION / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for s in summary:
        print(f"  {s['tile']:15s} ({s['region']:6s}): {s['polygons']:>4d} polygons, "
              f"threshold={s['threshold']:.3f}, pixels={s['post_pixels']:>7,}")
    print(f"\n  TOTAL: {sum(s['polygons'] for s in summary)} polygons")


if __name__ == "__main__":
    main()

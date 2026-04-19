#!/usr/bin/env python3
"""
V8 SWEEP: Systematically explore threshold, fusion, morphology, and min_area
configurations to maximize Union IoU. Reprocesses existing prob maps — no retraining.

Key insight from leaderboard: We are WAY too conservative.
- Our recall (47.5%) is 25pp below #1 (72.3%)
- Our FPR (11.1%) is lowest in top-5 — over-cautious
- Union IoU uses area² → missing big polygons is catastrophic
- Strategy: aggressively lower thresholds, use MAX fusion, reduce morphological erosion
"""
import sys, json, itertools, warnings
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from scipy import ndimage
from rasterio.features import shapes
from shapely.geometry import shape
from config import TEST_TILES, EKL_SUBMISSION, LUIS_V4_SUBMISSION, DATA, DATA_ADDITIONAL, tile_region

OUT = Path(__file__).resolve().parent.parent / "submission" / "v8_variants"
OUT.mkdir(parents=True, exist_ok=True)


# ─── Data loading ───────────────────────────────────────────────────────────
def load_prob(path):
    with rasterio.open(path) as src:
        d = src.read(1).astype(np.float32)
        p = src.profile.copy()
    if d.max() > 10:
        d /= 1000.0
    return d, p


def load_tile_probs(tile):
    ekl, prof = load_prob(EKL_SUBMISSION / f"prob_{tile}.tif")
    luis_path = LUIS_V4_SUBMISSION / f"prob_{tile}.tif"
    luis, _ = load_prob(luis_path) if luis_path.exists() else (None, None)
    if luis is not None:
        h, w = min(ekl.shape[0], luis.shape[0]), min(ekl.shape[1], luis.shape[1])
        ekl, luis = ekl[:h, :w], luis[:h, :w]
    return ekl, luis, prof


# ─── Fusion strategies ──────────────────────────────────────────────────────
def fuse(ekl, luis, strategy):
    if luis is None:
        return ekl
    if strategy == "wavg_75_25":
        return 0.75 * ekl + 0.25 * luis
    elif strategy == "wavg_60_40":
        return 0.60 * ekl + 0.40 * luis
    elif strategy == "wavg_50_50":
        return 0.50 * ekl + 0.50 * luis
    elif strategy == "max":
        return np.maximum(ekl, luis)
    elif strategy == "softmax":
        # soft-max: emphasize the larger one
        return np.maximum(ekl, luis) * 0.7 + (ekl + luis) / 2 * 0.3
    elif strategy == "geomean":
        return np.sqrt(np.clip(ekl * luis, 0, 1))
    elif strategy == "geomean_w":
        return np.clip(ekl ** 0.7 * luis ** 0.3, 0, 1)
    elif strategy == "max_recall":
        high = np.maximum(ekl, luis)
        geo = np.sqrt(np.clip(ekl * luis, 0, 1))
        return 0.7 * high + 0.3 * geo
    elif strategy == "agreement_boost":
        avg = 0.7 * ekl + 0.3 * luis
        both_high = (ekl > 0.4) & (luis > 0.25)
        avg[both_high] = np.clip(avg[both_high] * 1.2, 0, 1)
        return avg
    elif strategy == "power_mean_05":
        # p=0.5 → generalized mean, more generous than arithmetic
        return np.clip((0.7 * np.sqrt(ekl) + 0.3 * np.sqrt(luis)) ** 2, 0, 1)
    else:
        return 0.75 * ekl + 0.25 * luis


# ─── Post-processing ────────────────────────────────────────────────────────
def postprocess(binary, close_iter, open_iter, dilate_iter, min_area_ha):
    result = binary.copy()
    
    if dilate_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_dilation(result, s4, iterations=dilate_iter).astype(np.uint8)
    
    if close_iter > 0:
        s8 = ndimage.generate_binary_structure(2, 2)
        result = ndimage.binary_closing(result, s8, iterations=close_iter).astype(np.uint8)
    
    if open_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_opening(result, s4, iterations=open_iter).astype(np.uint8)
    
    # Area filter
    labeled, n = ndimage.label(result)
    min_px = int(min_area_ha / 0.01)  # 10m pixels → 0.01 ha each
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_px:
            result[labeled == i] = 0
    
    return result


# ─── Year estimation (reuse v5 logic) ──────────────────────────────────────
def estimate_year(tile, binary, profile):
    s2_dir = DATA / "sentinel-2" / "test" / f"{tile}__s2_l2a"
    if not s2_dir.exists():
        s2_dir = DATA / "sentinel-2" / "train" / f"{tile}__s2_l2a"
    if not s2_dir.exists():
        return np.full(binary.shape, "2306", dtype='<U6')

    yearly_nbr = {}
    for year in range(2020, 2026):
        monthly = []
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
                    nbr[b[0] <= 0] = np.nan
                    monthly.append(nbr)
            except Exception:
                continue
        if monthly:
            mh = min(m.shape[0] for m in monthly)
            mw = min(m.shape[1] for m in monthly)
            monthly = [m[:mh, :mw] for m in monthly]
            with np.errstate(all='ignore'):
                yearly_nbr[year] = np.nanmedian(monthly, axis=0)

    # 2026 partial
    s2_add = DATA_ADDITIONAL / "sentinel-2" / f"{tile}__s2_l2a"
    if s2_add.exists():
        m26 = []
        for month in [1, 2, 3, 4]:
            p = s2_add / f"{tile}__s2_l2a_2026_{month}.tif"
            if not p.exists():
                continue
            try:
                with rasterio.open(p) as src:
                    b = src.read().astype(np.float32)
                    b08, b12 = b[7], b[11]
                    d = b08 + b12
                    nbr = np.where(d > 0, (b08 - b12) / d, 0)
                    nbr[b[0] <= 0] = np.nan
                    m26.append(nbr)
            except Exception:
                continue
        if m26:
            mh = min(m.shape[0] for m in m26)
            mw = min(m.shape[1] for m in m26)
            m26 = [m[:mh, :mw] for m in m26]
            with np.errstate(all='ignore'):
                yearly_nbr[2026] = np.nanmedian(m26, axis=0)

    if len(yearly_nbr) < 2:
        return np.full(binary.shape, "2306", dtype='<U6')

    # Normalize all yearly NBR to same shape
    min_h = min(a.shape[0] for a in yearly_nbr.values())
    min_w = min(a.shape[1] for a in yearly_nbr.values())
    yearly_nbr = {k: v[:min_h, :min_w] for k, v in yearly_nbr.items()}

    sorted_yrs = sorted(yearly_nbr.keys())
    h, w = binary.shape
    max_drop = np.zeros((h, w), dtype=np.float32)
    drop_year = np.full((h, w), 2023, dtype=np.int32)
    for i in range(1, len(sorted_yrs)):
        prev, curr = sorted_yrs[i-1], sorted_yrs[i]
        with np.errstate(all='ignore'):
            drop = np.nan_to_num(yearly_nbr[prev] - yearly_nbr[curr], nan=0.0)
        better = drop > max_drop[:min_h, :min_w]
        max_drop[:min_h, :min_w][better] = drop[better]
        drop_year[:min_h, :min_w][better] = curr

    labeled, n = ndimage.label(binary)
    year_map = np.full((h, w), "2306", dtype='<U6')
    for cid in range(1, n + 1):
        mask = labeled == cid
        yrs = drop_year[mask]
        drops = max_drop[mask]
        if len(yrs) == 0:
            continue
        if drops.sum() > 0:
            totals = {}
            for y, d in zip(yrs, drops):
                totals[y] = totals.get(y, 0) + d
            modal = max(totals, key=totals.get)
        else:
            vals, cnts = np.unique(yrs, return_counts=True)
            modal = vals[np.argmax(cnts)]
        year_map[mask] = f"{modal % 100:02d}06"

    return year_map


# ─── Vectorize ──────────────────────────────────────────────────────────────
def vectorize(binary, year_map, profile, tile, min_area_ha):
    if binary.sum() == 0:
        return gpd.GeoDataFrame(columns=['geometry', 'time_step', 'confidence', 'tile_id'])

    polys, ts_list = [], []
    for geom, val in shapes(binary.astype(np.uint8), mask=binary.astype(bool), transform=profile['transform']):
        if val == 1:
            polys.append(shape(geom))

    if not polys:
        return gpd.GeoDataFrame(columns=['geometry', 'time_step', 'confidence', 'tile_id'])

    gdf = gpd.GeoDataFrame(geometry=polys, crs=profile['crs'])
    gdf = gdf.to_crs("EPSG:4326")

    utm = gdf.estimate_utm_crs()
    areas_ha = gdf.to_crs(utm).area / 10000.0
    gdf = gdf[areas_ha >= min_area_ha].reset_index(drop=True)

    gdf['time_step'] = "2306"
    gdf['confidence'] = 1.0
    gdf['tile_id'] = tile

    gdf_px = gdf.to_crs(profile['crs'])
    tr = profile['transform']
    for idx, row in gdf_px.iterrows():
        c = row.geometry.centroid
        col = int((c.x - tr.c) / tr.a)
        rpx = int((c.y - tr.f) / tr.e)
        if 0 <= rpx < year_map.shape[0] and 0 <= col < year_map.shape[1]:
            ts = year_map[rpx, col]
            if ts:
                gdf.at[idx, 'time_step'] = ts

    return gdf


# ─── Configuration space ────────────────────────────────────────────────────
# Based on leaderboard analysis:
# - We need MUCH higher recall (47.5% → 65%+)
# - Can afford higher FPR (11.1% → 25-30%)
# - Union IoU area² means missing large polys is catastrophic

CONFIGS = []

# The main knobs:
fusions = ["max", "softmax", "max_recall", "wavg_75_25", "wavg_60_40", "agreement_boost"]

# Per-tile thresholds: (amazon_big, amazon_small, africa, asia_small, asia_big)
# Tile order: 18NVJ_1_6 (Amazon small), 18NYH_2_1 (Amazon big), 33NTE_5_1 (Africa), 47QMA_6_2 (Asia small), 48PWA_0_6 (Asia big)
threshold_sets = {
    "aggressive":  {"18NVJ_1_6": 0.30, "18NYH_2_1": 0.35, "33NTE_5_1": 0.30, "47QMA_6_2": 0.25, "48PWA_0_6": 0.30},
    "moderate":    {"18NVJ_1_6": 0.35, "18NYH_2_1": 0.40, "33NTE_5_1": 0.35, "47QMA_6_2": 0.30, "48PWA_0_6": 0.35},
    "balanced":    {"18NVJ_1_6": 0.40, "18NYH_2_1": 0.45, "33NTE_5_1": 0.40, "47QMA_6_2": 0.30, "48PWA_0_6": 0.38},
    "v5like":      {"18NVJ_1_6": 0.50, "18NYH_2_1": 0.60, "33NTE_5_1": 0.50, "47QMA_6_2": 0.35, "48PWA_0_6": 0.35},
    "ultra_aggr":  {"18NVJ_1_6": 0.25, "18NYH_2_1": 0.30, "33NTE_5_1": 0.25, "47QMA_6_2": 0.20, "48PWA_0_6": 0.25},
}

# Morphology: (close, open, dilate)
morph_sets = {
    "no_erode":     (2, 0, 0),  # close only, no opening (don't erode)
    "close_heavy":  (3, 0, 0),  # heavy closing, no opening
    "v5_default":   (2, 1, 1),  # current v5
    "minimal":      (1, 0, 0),  # minimal processing
    "close_only_1": (1, 0, 0),  # just connect gaps
    "no_morph":     (0, 0, 0),  # raw threshold only
}

min_areas = {
    "tiny":   0.05,
    "small":  0.10,
    "v5std":  0.25,
    "medium": 0.15,
}

# Build config combos — strategic selection, not brute force
# Focus on the highest-impact combos

# Core strategy: MAX fusion + low thresholds + no erosion
for tname, thresholds in threshold_sets.items():
    for mname, morph in [("no_erode", (2, 0, 0)), ("close_heavy", (3, 0, 0)), ("minimal", (1, 0, 0))]:
        for amin_name, amin in [("small", 0.10), ("tiny", 0.05)]:
            CONFIGS.append({
                "name": f"max_{tname}_{mname}_{amin_name}",
                "fusion": "max",
                "thresholds": thresholds,
                "morph": morph,
                "min_area": amin,
            })

# Also test softmax and max_recall with best threshold sets
for fusion_name in ["softmax", "max_recall", "agreement_boost"]:
    for tname in ["aggressive", "moderate", "balanced"]:
        CONFIGS.append({
            "name": f"{fusion_name}_{tname}_noerode_small",
            "fusion": fusion_name,
            "thresholds": threshold_sets[tname],
            "morph": (2, 0, 0),
            "min_area": 0.10,
        })

# wavg_75_25 with lower thresholds (see if just lowering threshold on v5 helps)
for tname in ["aggressive", "moderate", "balanced"]:
    CONFIGS.append({
        "name": f"wavg75_{tname}_noerode_small",
        "fusion": "wavg_75_25",
        "thresholds": threshold_sets[tname],
        "morph": (2, 0, 0),
        "min_area": 0.10,
    })

# Add a few with v5-like thresholds but MAX fusion (test fusion impact alone)
CONFIGS.append({
    "name": "max_v5thresh_v5morph_v5area",
    "fusion": "max",
    "thresholds": threshold_sets["v5like"],
    "morph": (2, 1, 1),
    "min_area": 0.25,
})
CONFIGS.append({
    "name": "max_v5thresh_noerode_small",
    "fusion": "max",
    "thresholds": threshold_sets["v5like"],
    "morph": (2, 0, 0),
    "min_area": 0.10,
})

# Ultra aggressive with no morphology at all
CONFIGS.append({
    "name": "max_ultraaggr_nomorph_tiny",
    "fusion": "max",
    "thresholds": threshold_sets["ultra_aggr"],
    "morph": (0, 0, 0),
    "min_area": 0.05,
})

print(f"Total configs to evaluate: {len(CONFIGS)}")


# ─── Main sweep ─────────────────────────────────────────────────────────────
def run_config(cfg, tile_data):
    """Run a single configuration across all tiles. Returns summary stats."""
    all_gdf = []
    tile_stats = {}

    for tile in TEST_TILES:
        ekl, luis, prof = tile_data[tile]
        region = tile_region(tile)

        # Fuse
        prob = fuse(ekl, luis, cfg["fusion"])

        # Threshold (per-tile)
        thr = cfg["thresholds"].get(tile, 0.40)
        binary = (prob >= thr).astype(np.uint8)

        # Post-process
        close_i, open_i, dilate_i = cfg["morph"]
        binary = postprocess(binary, close_i, open_i, dilate_i, cfg["min_area"])

        # Year estimation
        year_map = tile_data[f"{tile}_year"]

        # Recompute year for this specific binary mask
        labeled, n = ndimage.label(binary)
        drop_year_data = tile_data[f"{tile}_drop_year"]
        max_drop_data = tile_data[f"{tile}_max_drop"]
        h, w = binary.shape
        year_map_local = np.full((h, w), "2306", dtype='<U6')
        for cid in range(1, n + 1):
            mask = labeled == cid
            yrs = drop_year_data[mask]
            drops = max_drop_data[mask]
            if len(yrs) == 0:
                continue
            if drops.sum() > 0:
                totals = {}
                for y, d in zip(yrs, drops):
                    totals[y] = totals.get(y, 0) + d
                modal = max(totals, key=totals.get)
            else:
                vals, cnts = np.unique(yrs, return_counts=True)
                modal = vals[np.argmax(cnts)]
            year_map_local[mask] = f"{modal % 100:02d}06"

        # Vectorize
        gdf = vectorize(binary, year_map_local, prof, tile, cfg["min_area"])
        
        # Compute area
        if len(gdf) > 0:
            utm = gdf.estimate_utm_crs()
            area_ha = gdf.to_crs(utm).area.sum() / 10000.0
        else:
            area_ha = 0.0

        tile_stats[tile] = {"polys": len(gdf), "area_ha": round(area_ha, 1)}
        if len(gdf) > 0:
            all_gdf.append(gdf)

    if all_gdf:
        combined = gpd.pd.concat(all_gdf, ignore_index=True)
    else:
        combined = gpd.GeoDataFrame(columns=['geometry', 'time_step', 'confidence', 'tile_id'])

    total_polys = len(combined)
    total_area = sum(s["area_ha"] for s in tile_stats.values())

    return combined, total_polys, total_area, tile_stats


def precompute_year_data(tile, prof):
    """Pre-compute year estimation data (drop_year, max_drop) once per tile."""
    s2_dir = DATA / "sentinel-2" / "test" / f"{tile}__s2_l2a"
    if not s2_dir.exists():
        s2_dir = DATA / "sentinel-2" / "train" / f"{tile}__s2_l2a"

    h, w = prof['height'], prof['width']

    if not s2_dir.exists():
        return np.full((h, w), 2023, dtype=np.int32), np.zeros((h, w), dtype=np.float32)

    yearly_nbr = {}
    for year in range(2020, 2026):
        monthly = []
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
                    nbr[b[0] <= 0] = np.nan
                    monthly.append(nbr)
            except Exception:
                continue
        if monthly:
            mh = min(m.shape[0] for m in monthly)
            mw = min(m.shape[1] for m in monthly)
            monthly = [m[:mh, :mw] for m in monthly]
            with np.errstate(all='ignore'):
                yearly_nbr[year] = np.nanmedian(monthly, axis=0)

    # 2026 partial
    s2_add = DATA_ADDITIONAL / "sentinel-2" / f"{tile}__s2_l2a"
    if s2_add.exists():
        m26 = []
        for month in [1, 2, 3, 4]:
            p = s2_add / f"{tile}__s2_l2a_2026_{month}.tif"
            if not p.exists():
                continue
            try:
                with rasterio.open(p) as src:
                    b = src.read().astype(np.float32)
                    b08, b12 = b[7], b[11]
                    d = b08 + b12
                    nbr = np.where(d > 0, (b08 - b12) / d, 0)
                    nbr[b[0] <= 0] = np.nan
                    m26.append(nbr)
            except Exception:
                continue
        if m26:
            mh = min(m.shape[0] for m in m26)
            mw = min(m.shape[1] for m in m26)
            m26 = [m[:mh, :mw] for m in m26]
            with np.errstate(all='ignore'):
                yearly_nbr[2026] = np.nanmedian(m26, axis=0)

    if len(yearly_nbr) < 2:
        return np.full((h, w), 2023, dtype=np.int32), np.zeros((h, w), dtype=np.float32)

    # Normalize all to same shape (min across all years)
    min_h = min(a.shape[0] for a in yearly_nbr.values())
    min_w = min(a.shape[1] for a in yearly_nbr.values())
    min_h = min(min_h, h)
    min_w = min(min_w, w)
    yearly_nbr = {k: v[:min_h, :min_w] for k, v in yearly_nbr.items()}

    sorted_yrs = sorted(yearly_nbr.keys())
    max_drop = np.zeros((h, w), dtype=np.float32)
    drop_year = np.full((h, w), 2023, dtype=np.int32)
    for i in range(1, len(sorted_yrs)):
        prev, curr = sorted_yrs[i-1], sorted_yrs[i]
        with np.errstate(all='ignore'):
            drop = np.nan_to_num(yearly_nbr[prev] - yearly_nbr[curr], nan=0.0)
        better = drop > max_drop[:min_h, :min_w]
        max_drop[:min_h, :min_w][better] = drop[better]
        drop_year[:min_h, :min_w][better] = curr

    return drop_year, max_drop


if __name__ == "__main__":
    import time
    t0 = time.time()

    # Pre-load all tile data (do this once)
    print("Loading tile data...")
    tile_data = {}
    for tile in TEST_TILES:
        ekl, luis, prof = load_tile_probs(tile)
        tile_data[tile] = (ekl, luis, prof)
        print(f"  {tile}: ekl={ekl.shape}, luis={'N/A' if luis is None else luis.shape}")

        # Pre-compute year estimation
        print(f"  Computing year data for {tile}...")
        drop_year, max_drop = precompute_year_data(tile, prof)
        tile_data[f"{tile}_drop_year"] = drop_year
        tile_data[f"{tile}_max_drop"] = max_drop
        # Dummy year_map (will be recomputed per-config)
        tile_data[f"{tile}_year"] = np.full(ekl.shape, "2306", dtype='<U6')

    print(f"\nData loaded in {time.time()-t0:.1f}s")
    print(f"\nRunning {len(CONFIGS)} configurations...\n")

    results = []
    for i, cfg in enumerate(CONFIGS):
        t1 = time.time()
        combined, n_polys, total_area, tile_stats = run_config(cfg, tile_data)
        elapsed = time.time() - t1

        # Save submission
        out_dir = OUT / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        if len(combined) > 0:
            combined.insert(0, 'id', range(len(combined)))
            combined[['id', 'time_step', 'confidence', 'tile_id', 'geometry']].to_file(
                out_dir / "submission.geojson", driver="GeoJSON"
            )

        # Summary
        ts = " | ".join(f"{t}:{s['polys']}p/{s['area_ha']}ha" for t, s in sorted(tile_stats.items()))
        results.append({
            "name": cfg["name"],
            "fusion": cfg["fusion"],
            "thresholds": cfg["thresholds"],
            "morph": cfg["morph"],
            "min_area": cfg["min_area"],
            "total_polys": n_polys,
            "total_area_ha": round(total_area, 1),
            "tiles": tile_stats,
        })

        print(f"[{i+1:>3}/{len(CONFIGS)}] {cfg['name']:<50} {n_polys:>5}p {total_area:>8.1f}ha  ({elapsed:.1f}s)")
        print(f"         {ts}")

    # Save results summary
    with open(OUT / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print top candidates sorted by total_polys (proxy for recall)
    print(f"\n{'='*80}")
    print("TOP CANDIDATES (by polygon count — proxy for recall)")
    print(f"{'='*80}")
    results.sort(key=lambda x: x["total_polys"], reverse=True)
    for i, r in enumerate(results[:20]):
        africa = r["tiles"].get("33NTE_5_1", {})
        amazon = r["tiles"].get("18NYH_2_1", {})
        asia = r["tiles"].get("48PWA_0_6", {})
        print(f"{i+1:>3}. {r['name']:<50} {r['total_polys']:>5}p {r['total_area_ha']:>8.1f}ha  "
              f"[NYH:{amazon.get('polys',0)} AFR:{africa.get('polys',0)} PWA:{asia.get('polys',0)}]")

    # Also print by reasonable range (similar total area to top leaderboard teams)
    print(f"\n{'='*80}")
    print("V5 COMPARISON: v5 had 579 polys, 2193 ha")
    print(f"v5 per-tile: NVJ:11/21ha NYH:141/912ha NTE:157/465ha QMA:11/9ha PWA:259/787ha")
    print(f"{'='*80}")

    total_time = time.time() - t0
    print(f"\nDone! {len(CONFIGS)} configs in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Results saved to {OUT}")

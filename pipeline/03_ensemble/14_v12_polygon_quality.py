#!/usr/bin/env python3
"""
V12: POLYGON QUALITY OPTIMIZATION — Closing the 0.81pp gap to #1

Key insight: error404 (#1, 53.45%) has LOWER recall (66.9%) and LOWER FPR (27.3%)
than us (#2, 52.64%, 71.75% recall, 33.59% FPR). They achieve higher IoU with
BETTER polygon quality — their boundaries align better with ground truth.

Strategy: Improve polygon boundary quality through:
1. Polygon simplification (Douglas-Peucker) — smooths raster staircase edges
2. Buffer smoothing — rounds corners
3. Threshold tuning — find optimal recall/FPR sweet spot
4. Dilation + simplification combos — expand then smooth
5. Erosion after dilation — tighten expanded boundaries
"""
import sys, json, time, warnings
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from scipy import ndimage
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.validation import make_valid
from config import TEST_TILES, EKL_SUBMISSION, LUIS_V4_SUBMISSION, DATA, DATA_ADDITIONAL, tile_region

OUT = Path(__file__).resolve().parent.parent / "submission" / "v12_quality"
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


# ─── Fusion ─────────────────────────────────────────────────────────────────
def fuse(ekl, luis, strategy):
    if luis is None:
        return ekl
    if strategy == "wavg_60_40":
        return 0.60 * ekl + 0.40 * luis
    elif strategy == "max":
        return np.maximum(ekl, luis)
    return 0.60 * ekl + 0.40 * luis


# ─── Post-processing ────────────────────────────────────────────────────────
def postprocess(binary, close_iter, open_iter, dilate_iter, erode_iter, min_area_ha):
    result = binary.copy()
    if close_iter > 0:
        s8 = ndimage.generate_binary_structure(2, 2)
        result = ndimage.binary_closing(result, s8, iterations=close_iter).astype(np.uint8)
    if open_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_opening(result, s4, iterations=open_iter).astype(np.uint8)
    if dilate_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_dilation(result, s4, iterations=dilate_iter).astype(np.uint8)
    if erode_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_erosion(result, s4, iterations=erode_iter).astype(np.uint8)
    # Area filter
    labeled, n = ndimage.label(result)
    min_px = int(min_area_ha / 0.01)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_px:
            result[labeled == i] = 0
    return result


# ─── Year estimation ────────────────────────────────────────────────────────
def precompute_year_data(tile, prof):
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

    min_h = min(a.shape[0] for a in yearly_nbr.values())
    min_w = min(a.shape[1] for a in yearly_nbr.values())
    min_h, min_w = min(min_h, h), min(min_w, w)
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


# ─── Vectorize with polygon quality options ─────────────────────────────────
def vectorize(binary, profile, tile, min_area_ha, drop_year, max_drop,
              simplify_tol=0, buffer_smooth=0):
    """
    Vectorize binary mask with optional polygon quality improvements.

    simplify_tol: Douglas-Peucker tolerance in meters (0=off). Smooths
                  staircase raster boundaries. 10m = 1 pixel.
    buffer_smooth: Buffer-unbuffer distance in meters (0=off). Rounds corners.
                   Applied as buffer(d).buffer(-d) to smooth outward corners,
                   then buffer(-d/2).buffer(d/2) to smooth inward corners.
    """
    if binary.sum() == 0:
        return gpd.GeoDataFrame(columns=['geometry', 'time_step', 'confidence', 'tile_id'])

    polys = []
    for geom, val in shapes(binary.astype(np.uint8), mask=binary.astype(bool),
                            transform=profile['transform']):
        if val == 1:
            polys.append(shape(geom))
    if not polys:
        return gpd.GeoDataFrame(columns=['geometry', 'time_step', 'confidence', 'tile_id'])

    # Build GeoDataFrame in source CRS (projected, meters)
    gdf = gpd.GeoDataFrame(geometry=polys, crs=profile['crs'])

    # ── Polygon quality operations in projected CRS (meters) ──
    utm = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm)

    if simplify_tol > 0:
        gdf_utm['geometry'] = gdf_utm.geometry.simplify(simplify_tol, preserve_topology=True)
        # Fix any invalid geometries
        gdf_utm['geometry'] = gdf_utm.geometry.apply(
            lambda g: make_valid(g) if not g.is_valid else g
        )
        # Remove empty geometries
        gdf_utm = gdf_utm[~gdf_utm.geometry.is_empty].reset_index(drop=True)

    if buffer_smooth > 0:
        # Positive buffer then negative: smooths outward corners
        gdf_utm['geometry'] = gdf_utm.geometry.buffer(buffer_smooth).buffer(-buffer_smooth)
        # Also smooth inward corners (half strength)
        half = buffer_smooth / 2
        gdf_utm['geometry'] = gdf_utm.geometry.buffer(-half).buffer(half)
        # Fix invalid
        gdf_utm['geometry'] = gdf_utm.geometry.apply(
            lambda g: make_valid(g) if g is not None and not g.is_empty and not g.is_valid else g
        )
        gdf_utm = gdf_utm[~gdf_utm.geometry.is_empty & gdf_utm.geometry.notna()].reset_index(drop=True)

    # Filter by area
    areas_ha = gdf_utm.area / 10000.0
    gdf_utm = gdf_utm[areas_ha >= min_area_ha].reset_index(drop=True)

    if len(gdf_utm) == 0:
        return gpd.GeoDataFrame(columns=['geometry', 'time_step', 'confidence', 'tile_id'])

    # Convert to EPSG:4326 for output
    gdf = gdf_utm.to_crs("EPSG:4326")

    # ── Year estimation ──
    h, w = binary.shape
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
# Try 11 (52.64%): likely E_ultralow_wavg_dil2 — most aggressive v11 config
# Try 10 (50.43%): B_aggressive_dil2
# error404 (#1):   53.45% IoU, 66.9% recall, 27.3% FPR

TRY8_THR = {
    "18NVJ_1_6": 0.30, "18NYH_2_1": 0.38, "33NTE_5_1": 0.35,
    "47QMA_6_2": 0.30, "48PWA_0_6": 0.38
}
AGGR_THR = {
    "18NVJ_1_6": 0.22, "18NYH_2_1": 0.30, "33NTE_5_1": 0.28,
    "47QMA_6_2": 0.22, "48PWA_0_6": 0.30
}
ULTRA_THR = {
    "18NVJ_1_6": 0.20, "18NYH_2_1": 0.28, "33NTE_5_1": 0.25,
    "47QMA_6_2": 0.20, "48PWA_0_6": 0.28
}
MID_THR = {
    "18NVJ_1_6": 0.25, "18NYH_2_1": 0.33, "33NTE_5_1": 0.30,
    "47QMA_6_2": 0.25, "48PWA_0_6": 0.33
}
GENTLE_THR = {
    "18NVJ_1_6": 0.27, "18NYH_2_1": 0.35, "33NTE_5_1": 0.32,
    "47QMA_6_2": 0.27, "48PWA_0_6": 0.35
}

CONFIGS = []

# ═══════════════════════════════════════════════════════════════════
# GROUP 1: SIMPLIFICATION on try 11's base config (likely best)
# Test if smoothing raster boundaries improves IoU
# ═══════════════════════════════════════════════════════════════════
for stol in [0, 8, 12, 18, 25]:
    CONFIGS.append({
        "name": f"ultra_dil2_simp{stol}",
        "fusion": "wavg_60_40", "thresholds": ULTRA_THR,
        "close": 2, "open": 0, "dilate": 2, "erode": 0,
        "min_area": 0.10, "simplify": stol, "buffer_smooth": 0,
    })

# ═══════════════════════════════════════════════════════════════════
# GROUP 2: BUFFER SMOOTHING on try 11's config
# Buffer(d).buffer(-d) rounds polygon corners
# ═══════════════════════════════════════════════════════════════════
for bsm in [5, 10, 15, 20]:
    CONFIGS.append({
        "name": f"ultra_dil2_buf{bsm}",
        "fusion": "wavg_60_40", "thresholds": ULTRA_THR,
        "close": 2, "open": 0, "dilate": 2, "erode": 0,
        "min_area": 0.10, "simplify": 0, "buffer_smooth": bsm,
    })

# ═══════════════════════════════════════════════════════════════════
# GROUP 3: COMBINED simplify + buffer smooth
# ═══════════════════════════════════════════════════════════════════
for stol, bsm in [(10, 8), (12, 10), (15, 10), (18, 12), (20, 15)]:
    CONFIGS.append({
        "name": f"ultra_dil2_s{stol}_b{bsm}",
        "fusion": "wavg_60_40", "thresholds": ULTRA_THR,
        "close": 2, "open": 0, "dilate": 2, "erode": 0,
        "min_area": 0.10, "simplify": stol, "buffer_smooth": bsm,
    })

# ═══════════════════════════════════════════════════════════════════
# GROUP 4: SIMPLIFICATION on B_aggressive_dil2 (try 10 base)
# ═══════════════════════════════════════════════════════════════════
for stol in [0, 10, 15, 20]:
    CONFIGS.append({
        "name": f"aggr_dil2_simp{stol}",
        "fusion": "wavg_60_40", "thresholds": AGGR_THR,
        "close": 2, "open": 0, "dilate": 2, "erode": 0,
        "min_area": 0.10, "simplify": stol, "buffer_smooth": 0,
    })

# ═══════════════════════════════════════════════════════════════════
# GROUP 5: HIGHER dilation + simplification (expand more, then smooth)
# Dil3 expands 30m, but simplify(20) smooths back → larger smoother polys
# ═══════════════════════════════════════════════════════════════════
for stol in [15, 20, 25]:
    CONFIGS.append({
        "name": f"ultra_dil3_simp{stol}",
        "fusion": "wavg_60_40", "thresholds": ULTRA_THR,
        "close": 2, "open": 0, "dilate": 3, "erode": 0,
        "min_area": 0.10, "simplify": stol, "buffer_smooth": 0,
    })
for stol in [15, 20, 25]:
    CONFIGS.append({
        "name": f"aggr_dil3_simp{stol}",
        "fusion": "wavg_60_40", "thresholds": AGGR_THR,
        "close": 2, "open": 0, "dilate": 3, "erode": 0,
        "min_area": 0.10, "simplify": stol, "buffer_smooth": 0,
    })

# ═══════════════════════════════════════════════════════════════════
# GROUP 6: DILATE then ERODE (morphological smoothing in raster space)
# Dilate 2-3, then erode 1 → net expand 1-2px but smoother boundaries
# ═══════════════════════════════════════════════════════════════════
for dil, ero in [(3, 1), (4, 2), (3, 1), (4, 1)]:
    for thr_name, thr in [("ultra", ULTRA_THR), ("aggr", AGGR_THR)]:
        CONFIGS.append({
            "name": f"{thr_name}_d{dil}e{ero}",
            "fusion": "wavg_60_40", "thresholds": thr,
            "close": 2, "open": 0, "dilate": dil, "erode": ero,
            "min_area": 0.10, "simplify": 0, "buffer_smooth": 0,
        })

# Also with simplification on top
for dil, ero, stol in [(3, 1, 12), (4, 2, 15), (3, 1, 15)]:
    CONFIGS.append({
        "name": f"ultra_d{dil}e{ero}_s{stol}",
        "fusion": "wavg_60_40", "thresholds": ULTRA_THR,
        "close": 2, "open": 0, "dilate": dil, "erode": ero,
        "min_area": 0.10, "simplify": stol, "buffer_smooth": 0,
    })

# ═══════════════════════════════════════════════════════════════════
# GROUP 7: INTERMEDIATE thresholds + quality (error404's operating point)
# Target ~27-30% FPR with ~65-68% recall + smoothing
# ═══════════════════════════════════════════════════════════════════
for thr_name, thr in [("gentle", GENTLE_THR), ("mid", MID_THR)]:
    for dil in [1, 2]:
        for stol in [0, 12, 18]:
            CONFIGS.append({
                "name": f"{thr_name}_dil{dil}_s{stol}",
                "fusion": "wavg_60_40", "thresholds": thr,
                "close": 2, "open": 0, "dilate": dil, "erode": 0,
                "min_area": 0.10, "simplify": stol, "buffer_smooth": 0,
            })

# ═══════════════════════════════════════════════════════════════════
# GROUP 8: MAX fusion + smoothing (different spatial coverage)
# ═══════════════════════════════════════════════════════════════════
for stol in [0, 12, 18]:
    CONFIGS.append({
        "name": f"max_aggr_dil2_s{stol}",
        "fusion": "max", "thresholds": AGGR_THR,
        "close": 2, "open": 0, "dilate": 2, "erode": 0,
        "min_area": 0.10, "simplify": stol, "buffer_smooth": 0,
    })
    CONFIGS.append({
        "name": f"max_ultra_dil2_s{stol}",
        "fusion": "max", "thresholds": ULTRA_THR,
        "close": 2, "open": 0, "dilate": 2, "erode": 0,
        "min_area": 0.10, "simplify": stol, "buffer_smooth": 0,
    })

# ═══════════════════════════════════════════════════════════════════
# GROUP 9: Try 8 base + dilation + heavy smoothing
# Proven detection base, expand + smooth
# ═══════════════════════════════════════════════════════════════════
for dil in [2, 3]:
    for stol in [12, 18, 25]:
        CONFIGS.append({
            "name": f"try8_dil{dil}_s{stol}",
            "fusion": "wavg_60_40", "thresholds": TRY8_THR,
            "close": 2, "open": 0, "dilate": dil, "erode": 0,
            "min_area": 0.10, "simplify": stol, "buffer_smooth": 0,
        })

# Deduplicate configs by name
seen = set()
deduped = []
for c in CONFIGS:
    if c['name'] not in seen:
        seen.add(c['name'])
        deduped.append(c)
CONFIGS = deduped

print(f"Total configs: {len(CONFIGS)}")


# ─── Main sweep ─────────────────────────────────────────────────────────────
def run_config(cfg, tile_data):
    all_gdf = []
    tile_stats = {}
    for tile in TEST_TILES:
        ekl, luis, prof = tile_data[tile]
        prob = fuse(ekl, luis, cfg["fusion"])
        thr = cfg["thresholds"].get(tile, 0.40)
        binary = (prob >= thr).astype(np.uint8)

        binary = postprocess(binary, cfg["close"], cfg["open"],
                             cfg["dilate"], cfg["erode"], cfg["min_area"])

        drop_year = tile_data[f"{tile}_drop_year"]
        max_drop = tile_data[f"{tile}_max_drop"]
        gdf = vectorize(binary, prof, tile, cfg["min_area"],
                        drop_year, max_drop,
                        simplify_tol=cfg["simplify"],
                        buffer_smooth=cfg["buffer_smooth"])
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


if __name__ == "__main__":
    t0 = time.time()

    print("Loading tile data...")
    tile_data = {}
    for tile in TEST_TILES:
        ekl, luis, prof = load_tile_probs(tile)
        tile_data[tile] = (ekl, luis, prof)
        print(f"  {tile}: ekl={ekl.shape}, luis={'N/A' if luis is None else luis.shape}")
        print(f"  Computing year data for {tile}...")
        drop_year, max_drop = precompute_year_data(tile, prof)
        tile_data[f"{tile}_drop_year"] = drop_year
        tile_data[f"{tile}_max_drop"] = max_drop

    print(f"\nData loaded in {time.time()-t0:.1f}s")
    print(f"\nRunning {len(CONFIGS)} configurations...\n")

    # Reference: try 11 (52.64% IoU) — E_ultralow_wavg_dil2 (no smoothing)
    REF_POLYS = 1333
    REF_AREA = 4476.4

    results = []
    for i, cfg in enumerate(CONFIGS):
        t1 = time.time()
        combined, n_polys, total_area, tile_stats = run_config(cfg, tile_data)
        elapsed = time.time() - t1

        out_dir = OUT / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(combined) > 0:
            combined.insert(0, 'id', range(len(combined)))
            combined[['id', 'time_step', 'confidence', 'tile_id', 'geometry']].to_file(
                out_dir / "submission.geojson", driver="GeoJSON"
            )

        d_polys = n_polys - REF_POLYS
        d_area = total_area - REF_AREA
        results.append({
            "name": cfg["name"],
            "fusion": cfg["fusion"],
            "dilate": cfg["dilate"], "erode": cfg["erode"],
            "simplify": cfg["simplify"], "buffer_smooth": cfg["buffer_smooth"],
            "total_polys": n_polys,
            "total_area_ha": round(total_area, 1),
            "tiles": tile_stats,
        })

        simp_tag = f"s{cfg['simplify']}" if cfg['simplify'] > 0 else ""
        buf_tag = f"b{cfg['buffer_smooth']}" if cfg['buffer_smooth'] > 0 else ""
        qual_tag = f" [{simp_tag}{'+' if simp_tag and buf_tag else ''}{buf_tag}]" if simp_tag or buf_tag else ""

        print(f"[{i+1:>3}/{len(CONFIGS)}] {cfg['name']:<35} {n_polys:>5}p ({d_polys:>+5}) {total_area:>8.1f}ha ({d_area:>+8.1f}){qual_tag}  ({elapsed:.1f}s)")

    # Save results
    with open(OUT / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ─── Analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("V12 RESULTS — RANKED BY TOTAL AREA")
    print(f"{'='*110}")
    print(f"{'Rank':>4} {'Name':<38} {'Polys':>6} {'Area':>8} {'ΔArea':>8} {'Simp':>5} {'Buf':>4} {'Dil':>4} {'Ero':>4} {'NVJ':>5} {'NYH':>5} {'NTE':>5} {'QMA':>5} {'PWA':>5}")
    print("-" * 110)
    print(f"{'REF':>4} {'Try 11 baseline (52.64% IoU)':<38} {REF_POLYS:>6} {REF_AREA:>8.1f} {'':>8} {'0':>5} {'0':>4} {'2':>4} {'0':>4}")
    print("-" * 110)

    results.sort(key=lambda x: x["total_area_ha"], reverse=True)
    for i, r in enumerate(results[:50]):
        nvj = r["tiles"].get("18NVJ_1_6", {}).get("polys", 0)
        nyh = r["tiles"].get("18NYH_2_1", {}).get("polys", 0)
        nte = r["tiles"].get("33NTE_5_1", {}).get("polys", 0)
        qma = r["tiles"].get("47QMA_6_2", {}).get("polys", 0)
        pwa = r["tiles"].get("48PWA_0_6", {}).get("polys", 0)
        d_area = r["total_area_ha"] - REF_AREA
        print(f"{i+1:>4} {r['name']:<38} {r['total_polys']:>6} {r['total_area_ha']:>8.1f} {d_area:>+8.1f} {r['simplify']:>5} {r['buffer_smooth']:>4} {r['dilate']:>4} {r['erode']:>4} {nvj:>5} {nyh:>5} {nte:>5} {qma:>5} {pwa:>5}")

    # Show effect of simplification on same base
    print(f"\n{'='*110}")
    print("SIMPLIFICATION EFFECT (ultra_dil2 base = try 11's likely config)")
    print(f"{'='*110}")
    ultra_results = [r for r in results if r['name'].startswith('ultra_dil2_simp')]
    ultra_results.sort(key=lambda x: x['simplify'])
    if ultra_results:
        base_area = ultra_results[0]['total_area_ha']
        for r in ultra_results:
            d = r['total_area_ha'] - base_area
            print(f"  simplify={r['simplify']:>3}m: {r['total_polys']:>5}p  {r['total_area_ha']:>8.1f}ha  (Δ{d:>+7.1f}ha from no-simp)")

    # Show buffer smooth effect
    print(f"\n{'='*110}")
    print("BUFFER SMOOTH EFFECT (ultra_dil2 base)")
    print(f"{'='*110}")
    buf_results = [r for r in results if r['name'].startswith('ultra_dil2_buf')]
    buf_results.sort(key=lambda x: x['buffer_smooth'])
    if buf_results:
        for r in buf_results:
            d = r['total_area_ha'] - REF_AREA
            print(f"  buffer={r['buffer_smooth']:>3}m: {r['total_polys']:>5}p  {r['total_area_ha']:>8.1f}ha  (Δ{d:>+7.1f}ha from ref)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

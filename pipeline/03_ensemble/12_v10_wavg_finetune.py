#!/usr/bin/env python3
"""
V10 FINE-TUNED SWEEP: Build on try 7's success (48.37% IoU, 871 polys).

Key insight: Try 7 used wavg_60_40 fusion, NOT max fusion.
OOF sweep shows wavg_60_40 dominates top-20 configs, MAX fusion ranks 73+.
But OOF is inversely correlated with LB — more aggressive = better on LB.

Try 7 config: wavg_60_40, t=0.45 uniform, close=1, open=1, min_ha=0.25
  -> 871 polys, 48.37% IoU, 53.27% recall, 15.97% FPR

Strategy: Fine grid around try 7 to find the sweet spot.
- wavg_60_40 fusion (proven)
- Threshold: 0.35-0.50 (try 7 used 0.45)
- Open: 0 vs 1 (try 7 used 1)
- Min area: 0.10-0.25 (try 7 used 0.25)
- Close: 1-2 (try 7 used 1)
- Also test per-tile thresholds (lower on NVJ and QMA which have fewest polys)

Target: ~950-1200 polys, aiming for 50%+ IoU.
FPR headroom: can go from 15.97% to ~25% (matching top teams).
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
from config import TEST_TILES, EKL_SUBMISSION, LUIS_V4_SUBMISSION, DATA, DATA_ADDITIONAL, tile_region

OUT = Path(__file__).resolve().parent.parent / "submission" / "v10_finetune"
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
    elif strategy == "wavg_55_45":
        return 0.55 * ekl + 0.45 * luis
    elif strategy == "wavg_50_50":
        return 0.50 * ekl + 0.50 * luis
    elif strategy == "max":
        return np.maximum(ekl, luis)
    elif strategy == "agreement_boost":
        avg = 0.60 * ekl + 0.40 * luis
        both_high = (ekl > 0.35) & (luis > 0.20)
        avg[both_high] = np.clip(avg[both_high] * 1.15, 0, 1)
        return avg
    else:
        return 0.60 * ekl + 0.40 * luis


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
    labeled, n = ndimage.label(result)
    min_px = int(min_area_ha / 0.01)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_px:
            result[labeled == i] = 0
    return result


# ─── Year estimation (precompute once) ─────────────────────────────────────
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


# ─── Vectorize ──────────────────────────────────────────────────────────────
def vectorize(binary, profile, tile, min_area_ha, drop_year, max_drop):
    if binary.sum() == 0:
        return gpd.GeoDataFrame(columns=['geometry', 'time_step', 'confidence', 'tile_id'])

    polys = []
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

    # Year estimation per polygon
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
# Try 7: wavg_60_40, t=0.45 uniform, close=1, open=1, min_ha=0.25 → 871 polys
# Per-tile: NVJ:29, NYH:173, NTE:217, QMA:117, PWA:335
# Target: find the config that gets ~1000-1200 polys

CONFIGS = []

# --- Tier 1: Uniform thresholds around try 7 (wavg_60_40) ---
# Try 7 was t=0.45/c=1/o=1/ha=0.25. Push each knob individually.
for t in [0.38, 0.40, 0.42]:
    for o in [0, 1]:
        for ha in [0.10, 0.15, 0.25]:
            for c in [1, 2]:
                name = f"wavg6040_t{int(t*100)}_c{c}_o{o}_ha{int(ha*100):02d}"
                CONFIGS.append({
                    "name": name,
                    "fusion": "wavg_60_40",
                    "thresholds": {tile: t for tile in TEST_TILES},
                    "morph": (c, o, 0),
                    "min_area": ha,
                })

# --- Tier 2: Per-tile thresholds (lower on NVJ and QMA which had fewest polys) ---
# NVJ only 29 polys (should be 50-80), QMA only 117 (should be 150-200)
pertile_sets = {
    "pertile_mild": {
        # Lower NVJ and QMA by 0.05, keep others at 0.42
        "18NVJ_1_6": 0.37, "18NYH_2_1": 0.42, "33NTE_5_1": 0.42,
        "47QMA_6_2": 0.37, "48PWA_0_6": 0.42
    },
    "pertile_balanced": {
        # More aggressive on low-recall tiles
        "18NVJ_1_6": 0.33, "18NYH_2_1": 0.40, "33NTE_5_1": 0.38,
        "47QMA_6_2": 0.33, "48PWA_0_6": 0.40
    },
    "pertile_push": {
        # Push hard on all tiles
        "18NVJ_1_6": 0.30, "18NYH_2_1": 0.38, "33NTE_5_1": 0.35,
        "47QMA_6_2": 0.30, "48PWA_0_6": 0.38
    },
    "pertile_nvj_focus": {
        # NVJ had only 29 polys — needs huge push, others moderate
        "18NVJ_1_6": 0.25, "18NYH_2_1": 0.42, "33NTE_5_1": 0.40,
        "47QMA_6_2": 0.35, "48PWA_0_6": 0.42
    },
}

for ptname, thresholds in pertile_sets.items():
    for o in [0, 1]:
        for ha in [0.10, 0.15]:
            CONFIGS.append({
                "name": f"{ptname}_c1_o{o}_ha{int(ha*100):02d}",
                "fusion": "wavg_60_40",
                "thresholds": thresholds,
                "morph": (1, o, 0),
                "min_area": ha,
            })
    # Also with close=2
    CONFIGS.append({
        "name": f"{ptname}_c2_o0_ha10",
        "fusion": "wavg_60_40",
        "thresholds": thresholds,
        "morph": (2, 0, 0),
        "min_area": 0.10,
    })

# --- Tier 3: Agreement boost (OOF rank ~10-18, between wavg and max) ---
for t in [0.40, 0.42, 0.45]:
    for o in [0, 1]:
        CONFIGS.append({
            "name": f"agboost_t{int(t*100)}_c1_o{o}_ha15",
            "fusion": "agreement_boost",
            "thresholds": {tile: t for tile in TEST_TILES},
            "morph": (1, o, 0),
            "min_area": 0.15,
        })

# --- Tier 4: Exact try 7 reproduction + slight variants ---
CONFIGS.append({
    "name": "try7_exact",
    "fusion": "wavg_60_40",
    "thresholds": {tile: 0.45 for tile in TEST_TILES},
    "morph": (1, 1, 0),
    "min_area": 0.25,
})
CONFIGS.append({
    "name": "try7_no_open",
    "fusion": "wavg_60_40",
    "thresholds": {tile: 0.45 for tile in TEST_TILES},
    "morph": (1, 0, 0),
    "min_area": 0.25,
})
CONFIGS.append({
    "name": "try7_small_area",
    "fusion": "wavg_60_40",
    "thresholds": {tile: 0.45 for tile in TEST_TILES},
    "morph": (1, 1, 0),
    "min_area": 0.10,
})
CONFIGS.append({
    "name": "try7_no_open_small",
    "fusion": "wavg_60_40",
    "thresholds": {tile: 0.45 for tile in TEST_TILES},
    "morph": (1, 0, 0),
    "min_area": 0.10,
})

# --- Tier 5: MAX fusion comparison (to validate OOF vs LB relationship) ---
for t in [0.45, 0.50]:
    CONFIGS.append({
        "name": f"max_t{int(t*100)}_c1_o1_ha25",
        "fusion": "max",
        "thresholds": {tile: t for tile in TEST_TILES},
        "morph": (1, 1, 0),
        "min_area": 0.25,
    })

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
        close_i, open_i, dilate_i = cfg["morph"]
        binary = postprocess(binary, close_i, open_i, dilate_i, cfg["min_area"])
        drop_year = tile_data[f"{tile}_drop_year"]
        max_drop = tile_data[f"{tile}_max_drop"]
        gdf = vectorize(binary, prof, tile, cfg["min_area"], drop_year, max_drop)
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

    # Pre-load all tile data
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

        ts = " | ".join(f"{t}:{s['polys']}p/{s['area_ha']}ha" for t, s in sorted(tile_stats.items()))
        results.append({
            "name": cfg["name"],
            "fusion": cfg["fusion"],
            "thresholds": cfg["thresholds"],
            "morph": list(cfg["morph"]),
            "min_area": cfg["min_area"],
            "total_polys": n_polys,
            "total_area_ha": round(total_area, 1),
            "tiles": tile_stats,
        })

        print(f"[{i+1:>3}/{len(CONFIGS)}] {cfg['name']:<45} {n_polys:>5}p {total_area:>8.1f}ha  ({elapsed:.1f}s)")
        print(f"         {ts}")

    # Save results
    with open(OUT / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ─── Analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("RESULTS RANKED BY POLYGON COUNT")
    print(f"{'='*90}")
    print(f"{'Rank':>4} {'Name':<45} {'Polys':>6} {'Area':>8} {'NVJ':>5} {'NYH':>5} {'NTE':>5} {'QMA':>5} {'PWA':>5}")
    print("-" * 90)

    # Try 7 reference
    print(f"{'REF':>4} {'TRY 7 (48.37% IoU)':<45} {'871':>6} {'?':>8} {'29':>5} {'173':>5} {'217':>5} {'117':>5} {'335':>5}")
    print("-" * 90)

    results.sort(key=lambda x: x["total_polys"], reverse=True)
    for i, r in enumerate(results):
        nvj = r["tiles"].get("18NVJ_1_6", {}).get("polys", 0)
        nyh = r["tiles"].get("18NYH_2_1", {}).get("polys", 0)
        nte = r["tiles"].get("33NTE_5_1", {}).get("polys", 0)
        qma = r["tiles"].get("47QMA_6_2", {}).get("polys", 0)
        pwa = r["tiles"].get("48PWA_0_6", {}).get("polys", 0)
        marker = " <<<" if 950 <= r["total_polys"] <= 1200 else ""
        print(f"{i+1:>4} {r['name']:<45} {r['total_polys']:>6} {r['total_area_ha']:>8.1f} {nvj:>5} {nyh:>5} {nte:>5} {qma:>5} {pwa:>5}{marker}")

    # Sweet spot analysis
    print(f"\n{'='*90}")
    print("SWEET SPOT: 950-1200 polys (likely optimal range)")
    print(f"{'='*90}")
    sweet = [r for r in results if 950 <= r["total_polys"] <= 1200]
    sweet.sort(key=lambda x: x["total_area_ha"], reverse=True)
    for r in sweet:
        nvj = r["tiles"].get("18NVJ_1_6", {}).get("polys", 0)
        nyh = r["tiles"].get("18NYH_2_1", {}).get("polys", 0)
        nte = r["tiles"].get("33NTE_5_1", {}).get("polys", 0)
        qma = r["tiles"].get("47QMA_6_2", {}).get("polys", 0)
        pwa = r["tiles"].get("48PWA_0_6", {}).get("polys", 0)
        print(f"  {r['name']:<45} {r['total_polys']:>6}p {r['total_area_ha']:>8.1f}ha  "
              f"NVJ:{nvj} NYH:{nyh} NTE:{nte} QMA:{qma} PWA:{pwa}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

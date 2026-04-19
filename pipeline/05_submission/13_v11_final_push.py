#!/usr/bin/env python3
"""
V11 FINAL PUSH: 2 tries remaining, targeting 52-54% IoU.

Analysis of try 8 (50.25% IoU, 56.9% recall, 18.9% FPR):
- FPR headroom: 18.9% → 27% = +8pp FPR budget (matching #1 error404)
- At 0.45 IoU/FPR ratio → +3.6pp IoU → 53.9% (would be #1!)
- Even at 0.35 ratio → +2.8pp → 53.1% (#2)

KEY INSIGHT: Dilation expands polygon boundaries by 10-20m.
- Adds 17-45% more detection AREA per tile
- Doesn't add new FP polygons, just grows existing ones
- If boundaries are mostly correct, this boosts TP area > FP area
- error404/Gelato likely use similar boundary expansion

Strategy tiers:
A) Dilation on try 8 config (safest - proven base + boundary expansion)
B) Lower thresholds + dilation (more detections + expansion)
C) MAX fusion + dilation (captures either-model detections + expansion)
D) Two-pass union: wavg confident + MAX marginal (advanced)
E) Very aggressive push (higher risk, higher reward)
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

OUT = Path(__file__).resolve().parent.parent / "submission" / "v11_final"
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
    elif strategy == "wavg_50_50":
        return 0.50 * ekl + 0.50 * luis
    elif strategy == "max":
        return np.maximum(ekl, luis)
    elif strategy == "softunion":
        # Soft union: boost where either model is confident
        return np.clip(ekl + luis - ekl * luis, 0, 1)
    elif strategy == "power_mean_3":
        # Power mean p=3: strongly favors the higher value
        return np.clip(((ekl**3 + luis**3) / 2) ** (1/3), 0, 1)
    else:
        return 0.60 * ekl + 0.40 * luis


def fuse_twostage(ekl, luis, tile_cfg):
    """Two-stage fusion: confident base + marginal additions."""
    if luis is None:
        return (ekl >= tile_cfg["base_thr"]).astype(np.uint8)
    h, w = min(ekl.shape[0], luis.shape[0]), min(ekl.shape[1], luis.shape[1])
    ekl, luis = ekl[:h, :w], luis[:h, :w]
    
    # Stage 1: wavg confident detections
    wavg = 0.60 * ekl + 0.40 * luis
    base = (wavg >= tile_cfg["base_thr"]).astype(np.uint8)
    
    # Stage 2: add pixels where MAX exceeds a lower threshold
    maxf = np.maximum(ekl, luis)
    marginal = (maxf >= tile_cfg["marginal_thr"]).astype(np.uint8)
    
    # Union
    return np.maximum(base, marginal)


# ─── Post-processing ────────────────────────────────────────────────────────
def postprocess(binary, close_iter, open_iter, dilate_iter, min_area_ha):
    result = binary.copy()
    # Close first (merge nearby fragments)
    if close_iter > 0:
        s8 = ndimage.generate_binary_structure(2, 2)
        result = ndimage.binary_closing(result, s8, iterations=close_iter).astype(np.uint8)
    # Open (remove noise)
    if open_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_opening(result, s4, iterations=open_iter).astype(np.uint8)
    # Dilate AFTER close/open (expand boundaries)
    if dilate_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_dilation(result, s4, iterations=dilate_iter).astype(np.uint8)
    # Area filter
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
# Try 8: pertile_push_c2_o0_ha10
#   wavg_60_40, NVJ=0.30 NYH=0.38 NTE=0.35 QMA=0.30 PWA=0.38
#   close=2, open=0, dilate=0, min_ha=0.10
#   → 1033p, 2869ha, 50.25% IoU, 56.9% recall, 18.9% FPR

TRY8_THRESHOLDS = {
    "18NVJ_1_6": 0.30, "18NYH_2_1": 0.38, "33NTE_5_1": 0.35,
    "47QMA_6_2": 0.30, "48PWA_0_6": 0.38
}

CONFIGS = []

# ─── TIER A: Dilation on try 8 (SAFEST — proven base + boundary expansion) ─
# Dilation adds 17-45% more area per tile. This is PURE area gain on existing
# good detections. Very likely to improve area-weighted IoU.
for dil in [1, 2]:
    CONFIGS.append({
        "name": f"A_try8_dil{dil}",
        "fusion": "wavg_60_40",
        "thresholds": TRY8_THRESHOLDS.copy(),
        "morph": (2, 0, dil),  # close=2, open=0, dilate=dil
        "min_area": 0.10,
        "twostage": False,
    })

# Also try dilation with slightly higher thresholds (to compensate for dilation adding FP area)
for dil in [1, 2]:
    higher_t = {k: v + 0.03 for k, v in TRY8_THRESHOLDS.items()}
    CONFIGS.append({
        "name": f"A_try8_dil{dil}_thr+3",
        "fusion": "wavg_60_40",
        "thresholds": higher_t,
        "morph": (2, 0, dil),
        "min_area": 0.10,
        "twostage": False,
    })

# ─── TIER B: Lower thresholds + dilation (more detections + expansion) ─────
# Push thresholds 0.03-0.05 below try 8, then dilate to fill boundaries
lower_sets = {
    "B_lower3_dil1": {
        "18NVJ_1_6": 0.27, "18NYH_2_1": 0.35, "33NTE_5_1": 0.32,
        "47QMA_6_2": 0.27, "48PWA_0_6": 0.35
    },
    "B_lower5_dil1": {
        "18NVJ_1_6": 0.25, "18NYH_2_1": 0.33, "33NTE_5_1": 0.30,
        "47QMA_6_2": 0.25, "48PWA_0_6": 0.33
    },
    "B_aggressive_dil1": {
        "18NVJ_1_6": 0.22, "18NYH_2_1": 0.30, "33NTE_5_1": 0.28,
        "47QMA_6_2": 0.22, "48PWA_0_6": 0.30
    },
}

for name, thrs in lower_sets.items():
    for dil in [1, 2]:
        CONFIGS.append({
            "name": f"{name.replace('dil1', f'dil{dil}')}",
            "fusion": "wavg_60_40",
            "thresholds": thrs,
            "morph": (2, 0, dil),
            "min_area": 0.10,
            "twostage": False,
        })
    # Also without dilation for comparison
    CONFIGS.append({
        "name": f"{name}_nodil",
        "fusion": "wavg_60_40",
        "thresholds": thrs,
        "morph": (2, 0, 0),
        "min_area": 0.10,
        "twostage": False,
    })

# ─── TIER C: MAX fusion + dilation ────────────────────────────────────────
# MAX captures detections from EITHER model. At matched thresholds, produces
# 3-37% more pixels. Combined with dilation, could push recall significantly.
for t_uniform in [0.38, 0.40, 0.45]:
    for dil in [0, 1]:
        thrs = {tile: t_uniform for tile in TEST_TILES}
        CONFIGS.append({
            "name": f"C_max_t{int(t_uniform*100)}_dil{dil}",
            "fusion": "max",
            "thresholds": thrs,
            "morph": (2, 0, dil),
            "min_area": 0.10,
            "twostage": False,
        })

# MAX with per-tile thresholds similar to try 8
for dil in [0, 1, 2]:
    CONFIGS.append({
        "name": f"C_max_pertile_dil{dil}",
        "fusion": "max",
        "thresholds": TRY8_THRESHOLDS.copy(),
        "morph": (2, 0, dil),
        "min_area": 0.10,
        "twostage": False,
    })

# ─── TIER D: Two-stage union (advanced) ───────────────────────────────────
# Base: wavg_60_40 at try8 thresholds (confident detections)
# + marginal: MAX at higher threshold (add detections either model sees)
twostage_configs = [
    ("D_union_base38_marg45", 0.38, 0.45),
    ("D_union_base40_marg48", 0.40, 0.48),
    ("D_union_base35_marg42", 0.35, 0.42),
    ("D_union_base35_marg38", 0.35, 0.38),
]

for name, base_t_offset, marg_t in twostage_configs:
    for dil in [0, 1]:
        CONFIGS.append({
            "name": f"{name}_dil{dil}",
            "fusion": "twostage",
            "thresholds": {tile: base_t_offset for tile in TEST_TILES},
            "morph": (2, 0, dil),
            "min_area": 0.10,
            "twostage": True,
            "twostage_cfg": {
                tile: {"base_thr": TRY8_THRESHOLDS[tile], "marginal_thr": marg_t}
                for tile in TEST_TILES
            },
        })

# ─── TIER E: Very aggressive (high risk / high reward) ────────────────────
# Target: ~27% FPR (matching error404). Push everything.
CONFIGS.append({
    "name": "E_ultralow_wavg_dil2",
    "fusion": "wavg_60_40",
    "thresholds": {"18NVJ_1_6": 0.20, "18NYH_2_1": 0.28, "33NTE_5_1": 0.25,
                   "47QMA_6_2": 0.20, "48PWA_0_6": 0.28},
    "morph": (2, 0, 2),
    "min_area": 0.10,
    "twostage": False,
})
CONFIGS.append({
    "name": "E_softunion_t35_dil1",
    "fusion": "softunion",
    "thresholds": {tile: 0.35 for tile in TEST_TILES},
    "morph": (2, 0, 1),
    "min_area": 0.10,
    "twostage": False,
})
CONFIGS.append({
    "name": "E_power3_t35_dil1",
    "fusion": "power_mean_3",
    "thresholds": {tile: 0.35 for tile in TEST_TILES},
    "morph": (2, 0, 1),
    "min_area": 0.10,
    "twostage": False,
})
# Try wavg_50_50 (more weight to Luis model)
CONFIGS.append({
    "name": "E_wavg5050_pertile_dil1",
    "fusion": "wavg_50_50",
    "thresholds": TRY8_THRESHOLDS.copy(),
    "morph": (2, 0, 1),
    "min_area": 0.10,
    "twostage": False,
})

# Close=3 (heavy merging) + dilation
CONFIGS.append({
    "name": "E_heavy_close3_dil1",
    "fusion": "wavg_60_40",
    "thresholds": TRY8_THRESHOLDS.copy(),
    "morph": (3, 0, 1),
    "min_area": 0.10,
    "twostage": False,
})

# Try 8 config but with min_area=0.05 (capture even smaller detections)
CONFIGS.append({
    "name": "E_try8_dil1_ha05",
    "fusion": "wavg_60_40",
    "thresholds": TRY8_THRESHOLDS.copy(),
    "morph": (2, 0, 1),
    "min_area": 0.05,
    "twostage": False,
})

print(f"Total configs: {len(CONFIGS)}")


# ─── Main sweep ─────────────────────────────────────────────────────────────
def run_config(cfg, tile_data):
    all_gdf = []
    tile_stats = {}
    for tile in TEST_TILES:
        ekl, luis, prof = tile_data[tile]
        
        if cfg.get("twostage"):
            # Two-stage fusion
            tile_cfg = cfg["twostage_cfg"][tile]
            binary = fuse_twostage(ekl, luis, tile_cfg)
        else:
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
            "morph": list(cfg["morph"]),
            "min_area": cfg["min_area"],
            "total_polys": n_polys,
            "total_area_ha": round(total_area, 1),
            "tiles": tile_stats,
        })

        # Compare to try 8
        area_delta = total_area - 2869.1
        poly_delta = n_polys - 1033
        print(f"[{i+1:>3}/{len(CONFIGS)}] {cfg['name']:<40} {n_polys:>5}p ({poly_delta:+5d}) {total_area:>8.1f}ha ({area_delta:+7.1f})  ({elapsed:.1f}s)")
        print(f"         {ts}")

    # Save results
    with open(OUT / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ─── Analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("V11 RESULTS — RANKED BY TOTAL AREA (best proxy for area-weighted IoU)")
    print(f"{'='*100}")
    print(f"{'Rank':>4} {'Name':<42} {'Polys':>6} {'ΔPoly':>7} {'Area':>8} {'ΔArea':>8} {'NVJ':>5} {'NYH':>5} {'NTE':>5} {'QMA':>5} {'PWA':>5}")
    print("-" * 100)
    print(f"{'REF':>4} {'TRY 8 (50.25% IoU, 18.9% FPR)':<42} {'1033':>6} {'':>7} {'2869.1':>8} {'':>8} {'50':>5} {'186':>5} {'242':>5} {'182':>5} {'373':>5}")
    print("-" * 100)

    results.sort(key=lambda x: x["total_area_ha"], reverse=True)
    for i, r in enumerate(results):
        nvj = r["tiles"].get("18NVJ_1_6", {}).get("polys", 0)
        nyh = r["tiles"].get("18NYH_2_1", {}).get("polys", 0)
        nte = r["tiles"].get("33NTE_5_1", {}).get("polys", 0)
        qma = r["tiles"].get("47QMA_6_2", {}).get("polys", 0)
        pwa = r["tiles"].get("48PWA_0_6", {}).get("polys", 0)
        d_polys = r["total_polys"] - 1033
        d_area = r["total_area_ha"] - 2869.1
        tier = r["name"][0]
        print(f"{i+1:>4} {r['name']:<42} {r['total_polys']:>6} {d_polys:>+7} {r['total_area_ha']:>8.1f} {d_area:>+8.1f} {nvj:>5} {nyh:>5} {nte:>5} {qma:>5} {pwa:>5}")

    # Per-tier summary
    print(f"\n{'='*100}")
    print("TIER SUMMARY (avg area by tier)")
    print(f"{'='*100}")
    tiers = {}
    for r in results:
        t = r["name"][0]
        if t not in tiers:
            tiers[t] = []
        tiers[t].append(r["total_area_ha"])
    
    tier_names = {"A": "Dilation on try8", "B": "Lower thr + dil", "C": "MAX fusion + dil",
                  "D": "Two-stage union", "E": "Very aggressive"}
    for t in sorted(tiers):
        vals = tiers[t]
        best = max(vals)
        print(f"  Tier {t} ({tier_names.get(t, '?'):<25}): avg={np.mean(vals):.1f}ha  best={best:.1f}ha  n={len(vals)}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")

#!/usr/bin/env python3
"""
V9 HYBRID: Per-tile optimal configuration selection.

Leaderboard insights (Oasis = 7th, 44.84% IoU):
- Our recall (47.5%) is 20pp below #1 (66.9%)
- Our FPR (11.1%) is the LOWEST on the board — way too conservative
- #1 has 27.3% FPR, #2 has 37.7% → we have HUGE headroom
- We're massively underpredicting QMA (11 polys) and NVJ (11 polys)
- Union IoU is area-weighted → missing large patches is catastrophic

Strategy: Use different aggression per tile:
- QMA + NVJ: ultra-aggressive (currently ~11 polys each → massive underdetection)
- NYH, NTE, PWA: moderate-to-aggressive (decent coverage but still room to grow)
"""
import sys, json, warnings, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
from pathlib import Path
from scipy import ndimage

# Reuse all the v8 infrastructure
from config import TEST_TILES, EKL_SUBMISSION, LUIS_V4_SUBMISSION, DATA, DATA_ADDITIONAL, tile_region

# Import shared functions from v8 sweep
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("v8", Path(__file__).resolve().parent / "10_v8_sweep.py")
v8 = module_from_spec(spec)
spec.loader.exec_module(v8)

load_tile_probs = v8.load_tile_probs
fuse = v8.fuse
postprocess = v8.postprocess
vectorize = v8.vectorize
precompute_year_data = v8.precompute_year_data

OUT = Path(__file__).resolve().parent.parent / "submission" / "v9_hybrid"
OUT.mkdir(parents=True, exist_ok=True)


# ─── Per-tile configurations ────────────────────────────────────────────────
# Based on v8 sweep results and leaderboard analysis:
#
# Current v5 per tile: NVJ:11p/21ha, NYH:141p/912ha, NTE:157p/465ha, QMA:11p/9ha, PWA:259p/787ha
# v8 ultra_aggr:       NVJ:89p,       NYH:300p,       NTE:355p,       QMA:272p,    PWA:538p
#
# The score is Union IoU (area-weighted), so we want to maximize area overlap.
# We need to be MUCH more aggressive everywhere, especially QMA and NVJ.

HYBRID_CONFIGS = {
    # ─── Strategy A: "Smart Aggressive" ─────────────────────────────────
    # Ultra-aggressive on underpredicted tiles, moderate on others
    "v9_smart_aggressive": {
        "18NVJ_1_6": {"fusion": "max", "threshold": 0.25, "morph": (1, 0, 0), "min_area": 0.05},  # ultra aggr
        "18NYH_2_1": {"fusion": "max", "threshold": 0.35, "morph": (2, 0, 0), "min_area": 0.10},  # moderate
        "33NTE_5_1": {"fusion": "max", "threshold": 0.30, "morph": (1, 0, 0), "min_area": 0.05},  # aggressive
        "47QMA_6_2": {"fusion": "max", "threshold": 0.20, "morph": (0, 0, 0), "min_area": 0.05},  # ultra aggr
        "48PWA_0_6": {"fusion": "max", "threshold": 0.30, "morph": (2, 0, 0), "min_area": 0.10},  # moderate
    },

    # ─── Strategy B: "Uniform Aggressive" ───────────────────────────────
    # Same aggressive settings everywhere (like max_aggressive_minimal_tiny but uniform)
    "v9_uniform_aggressive": {
        "18NVJ_1_6": {"fusion": "max", "threshold": 0.30, "morph": (1, 0, 0), "min_area": 0.05},
        "18NYH_2_1": {"fusion": "max", "threshold": 0.30, "morph": (1, 0, 0), "min_area": 0.05},
        "33NTE_5_1": {"fusion": "max", "threshold": 0.30, "morph": (1, 0, 0), "min_area": 0.05},
        "47QMA_6_2": {"fusion": "max", "threshold": 0.30, "morph": (1, 0, 0), "min_area": 0.05},
        "48PWA_0_6": {"fusion": "max", "threshold": 0.30, "morph": (1, 0, 0), "min_area": 0.05},
    },

    # ─── Strategy C: "Max Recall" ───────────────────────────────────────
    # Absolute maximum detection — our FPR has SO much headroom
    # The winning team has 27.3% FPR, we're at 11.1%. Push it!
    "v9_max_recall": {
        "18NVJ_1_6": {"fusion": "max", "threshold": 0.20, "morph": (0, 0, 0), "min_area": 0.05},
        "18NYH_2_1": {"fusion": "max", "threshold": 0.25, "morph": (0, 0, 0), "min_area": 0.05},
        "33NTE_5_1": {"fusion": "max", "threshold": 0.20, "morph": (0, 0, 0), "min_area": 0.05},
        "47QMA_6_2": {"fusion": "max", "threshold": 0.15, "morph": (0, 0, 0), "min_area": 0.05},
        "48PWA_0_6": {"fusion": "max", "threshold": 0.20, "morph": (0, 0, 0), "min_area": 0.05},
    },

    # ─── Strategy D: "Balanced boost" ───────────────────────────────────
    # Moderate boost everywhere — a safer bet (between v5 and aggressive)
    "v9_balanced_boost": {
        "18NVJ_1_6": {"fusion": "max", "threshold": 0.35, "morph": (2, 0, 0), "min_area": 0.10},
        "18NYH_2_1": {"fusion": "max", "threshold": 0.40, "morph": (2, 0, 0), "min_area": 0.10},
        "33NTE_5_1": {"fusion": "max", "threshold": 0.35, "morph": (2, 0, 0), "min_area": 0.10},
        "47QMA_6_2": {"fusion": "max", "threshold": 0.25, "morph": (1, 0, 0), "min_area": 0.05},
        "48PWA_0_6": {"fusion": "max", "threshold": 0.35, "morph": (2, 0, 0), "min_area": 0.10},
    },

    # ─── Strategy E: "Leaderboard Calibrated" ──────────────────────────
    # Target FPR ~25% (like #1 team). Push recall hard but controlled.
    # Our current 579 polys → aim for ~1200 (2x), keep area growth moderate
    "v9_calibrated": {
        "18NVJ_1_6": {"fusion": "max", "threshold": 0.28, "morph": (1, 0, 0), "min_area": 0.08},
        "18NYH_2_1": {"fusion": "max", "threshold": 0.35, "morph": (1, 0, 0), "min_area": 0.08},
        "33NTE_5_1": {"fusion": "max", "threshold": 0.28, "morph": (1, 0, 0), "min_area": 0.08},
        "47QMA_6_2": {"fusion": "max", "threshold": 0.22, "morph": (0, 0, 0), "min_area": 0.05},
        "48PWA_0_6": {"fusion": "max", "threshold": 0.28, "morph": (1, 0, 0), "min_area": 0.08},
    },

    # ─── Strategy F: "Union of models" ──────────────────────────────────
    # Use max_recall fusion (max + geometric mean blend) + very low thresholds
    # This catches anything EITHER model detects
    "v9_union_models": {
        "18NVJ_1_6": {"fusion": "max_recall", "threshold": 0.25, "morph": (1, 0, 0), "min_area": 0.05},
        "18NYH_2_1": {"fusion": "max_recall", "threshold": 0.30, "morph": (1, 0, 0), "min_area": 0.05},
        "33NTE_5_1": {"fusion": "max_recall", "threshold": 0.25, "morph": (1, 0, 0), "min_area": 0.05},
        "47QMA_6_2": {"fusion": "max_recall", "threshold": 0.20, "morph": (0, 0, 0), "min_area": 0.05},
        "48PWA_0_6": {"fusion": "max_recall", "threshold": 0.25, "morph": (1, 0, 0), "min_area": 0.05},
    },

    # ─── Strategy G: "Medium with close" ────────────────────────────────
    # Moderate thresholds but strong closing to connect fragmented detections
    # Closing helps merge adjacent small detections into larger polygons → more area overlap
    "v9_close_merge": {
        "18NVJ_1_6": {"fusion": "max", "threshold": 0.30, "morph": (3, 0, 0), "min_area": 0.05},
        "18NYH_2_1": {"fusion": "max", "threshold": 0.35, "morph": (3, 0, 0), "min_area": 0.08},
        "33NTE_5_1": {"fusion": "max", "threshold": 0.30, "morph": (3, 0, 0), "min_area": 0.05},
        "47QMA_6_2": {"fusion": "max", "threshold": 0.22, "morph": (2, 0, 0), "min_area": 0.05},
        "48PWA_0_6": {"fusion": "max", "threshold": 0.30, "morph": (3, 0, 0), "min_area": 0.08},
    },
}


def run_hybrid(name, tile_configs, tile_data):
    """Run a hybrid config with per-tile settings."""
    all_gdf = []
    tile_stats = {}

    for tile in TEST_TILES:
        ekl, luis, prof = tile_data[tile]
        cfg = tile_configs[tile]

        # Fuse
        prob = fuse(ekl, luis, cfg["fusion"])

        # Threshold
        binary = (prob >= cfg["threshold"]).astype(np.uint8)

        # Post-process
        close_i, open_i, dilate_i = cfg["morph"]
        binary = postprocess(binary, close_i, open_i, dilate_i, cfg["min_area"])

        # Year estimation from precomputed data
        labeled, n = ndimage.label(binary)
        drop_year_data = tile_data[f"{tile}_drop_year"]
        max_drop_data = tile_data[f"{tile}_max_drop"]
        h, w = binary.shape
        year_map = np.full((h, w), "2306", dtype='<U6')
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
            year_map[mask] = f"{modal % 100:02d}06"

        # Vectorize
        gdf = vectorize(binary, year_map, prof, tile, cfg["min_area"])

        # Stats
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

    return combined, tile_stats


if __name__ == "__main__":
    t0 = time.time()

    # Pre-load all tile data (same as v8)
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
    print(f"\nRunning {len(HYBRID_CONFIGS)} hybrid configurations...\n")
    print(f"{'Config':<30} {'Polys':>6} {'Area':>8} | {'NVJ':>5} {'NYH':>5} {'NTE':>5} {'QMA':>5} {'PWA':>5}")
    print("-" * 85)

    # Reference: v5 submission
    print(f"{'[ref] v5_submitted':<30} {'579':>6} {'2193':>8} | {'11':>5} {'141':>5} {'157':>5} {'11':>5} {'259':>5}")
    print("-" * 85)

    results = []
    for name, tile_configs in HYBRID_CONFIGS.items():
        t1 = time.time()
        combined, tile_stats = run_hybrid(name, tile_configs, tile_data)
        elapsed = time.time() - t1

        total_polys = len(combined)
        total_area = sum(s["area_ha"] for s in tile_stats.values())

        # Save
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if len(combined) > 0:
            combined.insert(0, 'id', range(len(combined)))
            combined[['id', 'time_step', 'confidence', 'tile_id', 'geometry']].to_file(
                out_dir / "submission.geojson", driver="GeoJSON"
            )

        # Print
        per_tile = " ".join(f"{tile_stats[t]['polys']:>5}" for t in sorted(tile_stats))
        print(f"{name:<30} {total_polys:>6} {total_area:>8.1f} | {per_tile}  [{elapsed:.1f}s]")

        results.append({
            "name": name,
            "total_polys": total_polys,
            "total_area": round(total_area, 1),
            "tile_stats": tile_stats,
            "tile_configs": {t: {k: v if not isinstance(v, tuple) else list(v) for k, v in c.items()} for t, c in tile_configs.items()},
        })

    # Save summary
    print(f"\n{'='*85}")
    print(f"Total time: {time.time()-t0:.1f}s")

    with open(OUT / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUT / 'results_summary.json'}")

    # Recommendation
    print("\n" + "=" * 85)
    print("RECOMMENDATION:")
    print("=" * 85)
    # Sort by total_polys as proxy for recall
    by_polys = sorted(results, key=lambda r: r["total_polys"], reverse=True)
    print(f"\nHighest recall (most polys): {by_polys[0]['name']} ({by_polys[0]['total_polys']} polys)")
    print(f"Most balanced:              {by_polys[len(by_polys)//2]['name']} ({by_polys[len(by_polys)//2]['total_polys']} polys)")
    print(f"\nWith 4 tries left, submit in order:")
    print(f"  1. v9_smart_aggressive  — targeted per-tile boost (best expected IoU)")
    print(f"  2. v9_calibrated        — calibrated to match top-team FPR profile")
    print(f"  3. Iterate based on scores from 1 & 2")

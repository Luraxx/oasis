"""Test UNTRIED fusion strategies on the proven try 14/15 post-processing.

Key insight: all submissions 11-15 used wavg_60_40 fusion. softunion and
power_mean create FUNDAMENTALLY different probability maps.

softunion: ekl + luis - ekl*luis (probabilistic OR — boosts where either is confident)
power_mean_3: ((ekl^3 + luis^3)/2)^(1/3) (favors the higher prediction)
max: max(ekl, luis) (takes the more confident model per pixel)

Combined with the proven erosion approach from try 14 (53.30%).
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from scipy import ndimage
from scipy.ndimage import label as cc_label
from rasterio.features import shapes
from shapely.geometry import shape

ROOT = Path("/shared-docker/oasis-luis-v5")
EKL = ROOT / "ekl_submission"
LUIS = ROOT / "luis_v4_submission"

TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]

def load_prob(path):
    with rasterio.open(path) as src:
        d = src.read(1).astype(np.float32)
        p = src.profile.copy()
    if d.max() > 10:
        d /= 1000.0
    return d, p

def fuse_wavg(ekl, luis, w_ekl=0.60):
    return w_ekl * ekl + (1 - w_ekl) * luis

def fuse_softunion(ekl, luis):
    return np.clip(ekl + luis - ekl * luis, 0, 1)

def fuse_power3(ekl, luis):
    return np.clip(((ekl**3 + luis**3) / 2) ** (1/3), 0, 1)

def fuse_max(ekl, luis):
    return np.maximum(ekl, luis)

def postprocess(binary, close_i, open_i, dilate_i, min_area_ha, erode_i=0):
    result = binary.copy()
    if close_i > 0:
        s8 = ndimage.generate_binary_structure(2, 2)
        result = ndimage.binary_closing(result, s8, iterations=close_i).astype(np.uint8)
    if open_i > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_opening(result, s4, iterations=open_i).astype(np.uint8)
    if dilate_i > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_dilation(result, s4, iterations=dilate_i).astype(np.uint8)
    if erode_i > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_erosion(result, s4, iterations=erode_i).astype(np.uint8)
    labeled, n = ndimage.label(result)
    min_px = int(min_area_ha / 0.01)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_px:
            result[labeled == i] = 0
    return result

def vectorize(binary, prof, tile, min_area_ha):
    """Vectorize binary raster to GeoJSON features."""
    feats = []
    if not binary.any():
        return feats
    transform = prof['transform']
    crs = prof['crs']
    for geom, val in shapes(binary.astype(np.uint8), transform=transform):
        if val == 0:
            continue
        poly = shape(geom)
        feats.append({
            "type": "Feature",
            "geometry": poly.__geo_interface__,
            "properties": {
                "tile_id": tile,
                "confidence": 0.5,
                "time_step": None,
            }
        })
    # Filter by area
    filtered = []
    for f in feats:
        poly = shape(f["geometry"])
        # Rough area in hectares (assuming 10m resolution)
        area_ha = poly.area / 100  # each pixel is ~100 sq m at 10m res
        if area_ha >= min_area_ha:
            filtered.append(f)
    return filtered

# Load all tile data
print("Loading tile data...", flush=True)
tile_data = {}
for tile in TEST_TILES:
    ekl, prof = load_prob(EKL / f"prob_{tile}.tif")
    luis, _ = load_prob(LUIS / f"prob_{tile}.tif")
    h, w = min(ekl.shape[0], luis.shape[0]), min(ekl.shape[1], luis.shape[1])
    ekl, luis = ekl[:h, :w], luis[:h, :w]
    tile_data[tile] = (ekl, luis, prof)

# Try 11 thresholds (proven base)
THR_ULTRALOW = {"18NVJ_1_6": 0.20, "18NYH_2_1": 0.28, "33NTE_5_1": 0.25,
                "47QMA_6_2": 0.20, "48PWA_0_6": 0.28}

# Configs to test
CONFIGS = [
    # Baseline: exact try 11 + erode 3 (should match try 14)
    {"name": "base_wavg_e3", "fuse": "wavg", "thresholds": THR_ULTRALOW,
     "morph": (2, 0, 2), "erode": 3, "min_ha": 0.10},

    # Softunion variants
    {"name": "softunion_t30_e2", "fuse": "softunion",
     "thresholds": {t: 0.30 for t in TEST_TILES},
     "morph": (2, 0, 2), "erode": 2, "min_ha": 0.10},
    {"name": "softunion_t35_e2", "fuse": "softunion",
     "thresholds": {t: 0.35 for t in TEST_TILES},
     "morph": (2, 0, 2), "erode": 2, "min_ha": 0.10},
    {"name": "softunion_t35_e3", "fuse": "softunion",
     "thresholds": {t: 0.35 for t in TEST_TILES},
     "morph": (2, 0, 2), "erode": 3, "min_ha": 0.10},
    {"name": "softunion_pertile_e3", "fuse": "softunion",
     "thresholds": {"18NVJ_1_6": 0.28, "18NYH_2_1": 0.35, "33NTE_5_1": 0.32,
                    "47QMA_6_2": 0.28, "48PWA_0_6": 0.35},
     "morph": (2, 0, 2), "erode": 3, "min_ha": 0.10},

    # Power mean variants
    {"name": "power3_t35_e3", "fuse": "power3",
     "thresholds": {t: 0.35 for t in TEST_TILES},
     "morph": (2, 0, 2), "erode": 3, "min_ha": 0.10},
    {"name": "power3_pertile_e3", "fuse": "power3",
     "thresholds": {"18NVJ_1_6": 0.25, "18NYH_2_1": 0.33, "33NTE_5_1": 0.30,
                    "47QMA_6_2": 0.25, "48PWA_0_6": 0.33},
     "morph": (2, 0, 2), "erode": 3, "min_ha": 0.10},

    # Max fusion variants
    {"name": "max_t40_e3", "fuse": "max",
     "thresholds": {t: 0.40 for t in TEST_TILES},
     "morph": (2, 0, 2), "erode": 3, "min_ha": 0.10},
    {"name": "max_pertile_e3", "fuse": "max",
     "thresholds": {"18NVJ_1_6": 0.30, "18NYH_2_1": 0.38, "33NTE_5_1": 0.35,
                    "47QMA_6_2": 0.30, "48PWA_0_6": 0.38},
     "morph": (2, 0, 2), "erode": 3, "min_ha": 0.10},

    # Wavg with different weights + erosion
    {"name": "wavg70_e3", "fuse": "wavg70",
     "thresholds": THR_ULTRALOW,
     "morph": (2, 0, 2), "erode": 3, "min_ha": 0.10},

    # Try erode 4 (more aggressive boundary tightening)
    {"name": "base_wavg_e4", "fuse": "wavg", "thresholds": THR_ULTRALOW,
     "morph": (2, 0, 2), "erode": 4, "min_ha": 0.10},

    # Try erode 2 (less aggressive)
    {"name": "base_wavg_e2", "fuse": "wavg", "thresholds": THR_ULTRALOW,
     "morph": (2, 0, 2), "erode": 2, "min_ha": 0.10},
]

FUSE_FNS = {
    "wavg": lambda e, l: fuse_wavg(e, l, 0.60),
    "wavg70": lambda e, l: fuse_wavg(e, l, 0.70),
    "softunion": fuse_softunion,
    "power3": fuse_power3,
    "max": fuse_max,
}

print(f"\nRunning {len(CONFIGS)} configs...\n", flush=True)

for cfg in CONFIGS:
    total_px = 0
    total_polys = 0
    tile_info = []

    for tile in TEST_TILES:
        ekl, luis, prof = tile_data[tile]
        prob = FUSE_FNS[cfg["fuse"]](ekl, luis)
        thr = cfg["thresholds"][tile]
        binary = (prob >= thr).astype(np.uint8)
        close_i, open_i, dilate_i = cfg["morph"]
        binary = postprocess(binary, close_i, open_i, dilate_i, cfg["min_ha"],
                           erode_i=cfg.get("erode", 0))
        px = int(binary.sum())
        total_px += px

        # Count polygons
        labeled, n = ndimage.label(binary)
        total_polys += n
        tile_info.append(f"{tile[-7:-4]}={px//1000}k/{n}p")

    tiles_str = " ".join(tile_info)
    print(f"  {cfg['name']:<30} px={total_px:>7,}  polys={total_polys:>5}  {tiles_str}", flush=True)

# Now build the most promising ones as actual submissions
print("\n=== Building submission files for top candidates ===\n", flush=True)

for cfg_name in ["softunion_pertile_e3", "softunion_t35_e3", "power3_pertile_e3", "base_wavg_e3"]:
    cfg = [c for c in CONFIGS if c["name"] == cfg_name][0]
    out_dir = Path(f"/shared-docker/oasis-mark-2/submissions/fusion_{cfg_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    for tile in TEST_TILES:
        ekl, luis, prof = tile_data[tile]
        prob = FUSE_FNS[cfg["fuse"]](ekl, luis)
        thr = cfg["thresholds"][tile]
        binary = (prob >= thr).astype(np.uint8)
        close_i, open_i, dilate_i = cfg["morph"]
        binary = postprocess(binary, close_i, open_i, dilate_i, cfg["min_ha"],
                           erode_i=cfg.get("erode", 0))

        # Write binary raster
        out_bin = out_dir / f"pred_{tile}.tif"
        bm = prof.copy()
        bm.update(dtype="uint8", count=1, compress="LZW", nodata=0,
                  height=binary.shape[0], width=binary.shape[1])
        with rasterio.open(out_bin, "w", **bm) as dst:
            dst.write(binary, 1)

        # Vectorize
        feats = vectorize(binary, prof, tile, cfg["min_ha"])
        # Add confidence from prob map
        comp_labels, n_comp = cc_label(binary)
        for i, feat in enumerate(feats):
            cid = i + 1
            if cid <= n_comp:
                pmask = (comp_labels == cid)
                if pmask.any():
                    feat["properties"]["confidence"] = round(float(prob[pmask].mean()), 3)
        all_features.extend(feats)

    cgj = {"type": "FeatureCollection",
           "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
           "features": all_features}
    (out_dir / "submission.geojson").write_text(json.dumps(cgj))
    print(f"  {cfg_name}: {len(all_features)} features, {out_dir}", flush=True)

print("\nDone!", flush=True)

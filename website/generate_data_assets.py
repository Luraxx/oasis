#!/usr/bin/env python3
"""Generate PNG thumbnails and JSON stats for the data-explorer page."""

import json, os, glob, sys
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from PIL import Image

DATA = "/shared-docker/oasis/data/makeathon-challenge"
OUT  = "/shared-docker/oasis-viz-approach/data-assets"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/thumbs", exist_ok=True)

THUMB = 256  # thumbnail pixel size

# ── helpers ──────────────────────────────────────────────────────
def percentile_stretch(arr, lo=2, hi=98):
    """Stretch to 0-255 using percentile clipping."""
    p_lo, p_hi = np.percentile(arr[arr > 0], [lo, hi]) if np.any(arr > 0) else (0, 1)
    if p_hi == p_lo:
        p_hi = p_lo + 1
    out = np.clip((arr - p_lo) / (p_hi - p_lo), 0, 1)
    return (out * 255).astype(np.uint8)

def save_thumb(arr_rgb, path, size=THUMB):
    """Save HxWx3 uint8 array as PNG thumbnail."""
    img = Image.fromarray(arr_rgb)
    img = img.resize((size, size), Image.LANCZOS)
    img.save(path, optimize=True)

def arr_to_thumb(arr_2d, cmap_func, path, size=THUMB):
    """Save single-band with colormap."""
    rgb = cmap_func(arr_2d)
    save_thumb(rgb, path, size)

def green_red_cmap(arr):
    """Green (forest) → Red (deforested) colormap for labels."""
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 1] = 40  # dark green background
    mask = arr > 0
    rgb[mask, 0] = 255  # red for deforestation
    rgb[mask, 1] = 60
    rgb[mask, 2] = 60
    return rgb

def sar_cmap(arr):
    """Cyan-tinted SAR colormap."""
    stretched = percentile_stretch(arr, 1, 99)
    h, w = stretched.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = (stretched * 0.3).astype(np.uint8)
    rgb[:, :, 1] = (stretched * 0.8).astype(np.uint8)
    rgb[:, :, 2] = stretched
    return rgb

# ── collect tile info ────────────────────────────────────────────
train_tiles = sorted([d.replace("__s2_l2a", "") for d in os.listdir(f"{DATA}/sentinel-2/train")])
test_tiles  = sorted([d.replace("__s2_l2a", "") for d in os.listdir(f"{DATA}/sentinel-2/test")])

print(f"Train tiles: {len(train_tiles)}, Test tiles: {len(test_tiles)}")

# ── read geojson metadata ───────────────────────────────────────
with open(f"{DATA}/metadata/train_tiles.geojson") as f:
    train_geo = json.load(f)
with open(f"{DATA}/metadata/test_tiles.geojson") as f:
    test_geo = json.load(f)

def get_tile_coords(geojson_data):
    """Extract {name: {lat, lon, bbox}} from geojson."""
    out = {}
    for ft in geojson_data["features"]:
        name = ft["properties"]["name"]
        coords = ft["geometry"]["coordinates"]
        # Handle nested polygon coords
        while isinstance(coords[0][0], list):
            coords = coords[0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        out[name] = {
            "lat": sum(lats) / len(lats),
            "lon": sum(lons) / len(lons),
            "bbox": [min(lons), min(lats), max(lons), max(lats)]
        }
    return out

train_coords = get_tile_coords(train_geo)
test_coords  = get_tile_coords(test_geo)

# ── region classifier ───────────────────────────────────────────
def get_region(lon):
    if lon < -30:
        return "South America"
    elif lon < 30:
        return "Africa"
    else:
        return "Southeast Asia"

# ── master stats dict ───────────────────────────────────────────
stats = {
    "train": {},
    "test":  {},
    "summary": {
        "total_train": len(train_tiles),
        "total_test":  len(test_tiles),
        "s2_bands": 12,
        "s2_res_m": 10,
        "s2_size": 1002,
        "s1_bands": 1,
        "time_range": "2020-2025",
        "months": 72,
        "label_sources": ["RADD", "GLAD-L", "GLAD-S2"]
    }
}

# ── generate Sentinel-2 RGB thumbs + stats ──────────────────────
def process_s2_tile(tile, split):
    """Generate RGB, false-color, NDVI thumbs for a tile."""
    base = f"{DATA}/sentinel-2/{split}/{tile}__s2_l2a"
    files = sorted(glob.glob(f"{base}/{tile}__s2_l2a_*.tif"))
    if not files:
        return {}

    info = {"s2_count": len(files), "s2_years": {}}

    # Count by year
    for f in files:
        fname = os.path.basename(f)
        parts = fname.replace(".tif", "").split("_")
        # year is at index with 4 digits
        for p in parts:
            if len(p) == 4 and p.isdigit():
                yr = p
                info["s2_years"][yr] = info["s2_years"].get(yr, 0) + 1
                break

    # Generate thumbs for first and last available date
    for label, idx in [("first", 0), ("last", -1), ("mid", len(files)//2)]:
        try:
            with rasterio.open(files[idx]) as src:
                bands = src.read()  # (12, H, W)
                # RGB = bands 4, 3, 2 (1-indexed → 3, 2, 1 in 0-indexed)
                r = percentile_stretch(bands[3].astype(float))
                g = percentile_stretch(bands[2].astype(float))
                b = percentile_stretch(bands[1].astype(float))
                rgb = np.stack([r, g, b], axis=-1)
                save_thumb(rgb, f"{OUT}/thumbs/{tile}_s2_rgb_{label}.png")

                if label == "first":
                    # False color (NIR, Red, Green) = bands 8, 4, 3 → idx 7, 3, 2
                    fc_r = percentile_stretch(bands[7].astype(float))
                    fc_g = percentile_stretch(bands[3].astype(float))
                    fc_b = percentile_stretch(bands[2].astype(float))
                    fc = np.stack([fc_r, fc_g, fc_b], axis=-1)
                    save_thumb(fc, f"{OUT}/thumbs/{tile}_s2_fc.png")

                    # NDVI
                    nir = bands[7].astype(float)
                    red = bands[3].astype(float)
                    denom = nir + red
                    denom[denom == 0] = 1
                    ndvi = (nir - red) / denom
                    # NDVI colormap: brown(-1) → yellow(0) → green(1)
                    h, w = ndvi.shape
                    ndvi_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                    ndvi_n = np.clip((ndvi + 0.2) / 1.0, 0, 1)
                    ndvi_rgb[:, :, 0] = ((1 - ndvi_n) * 180).astype(np.uint8)
                    ndvi_rgb[:, :, 1] = (ndvi_n * 200 + 30).astype(np.uint8)
                    ndvi_rgb[:, :, 2] = (ndvi_n * 40).astype(np.uint8)
                    save_thumb(ndvi_rgb, f"{OUT}/thumbs/{tile}_ndvi.png")

                    # Band stats
                    band_names = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
                    info["band_means"] = [float(np.mean(bands[i][bands[i] > 0])) for i in range(min(12, len(band_names)))]
                    info["band_stds"]  = [float(np.std(bands[i][bands[i] > 0])) for i in range(min(12, len(band_names)))]
                    info["s2_shape"] = [int(bands.shape[1]), int(bands.shape[2])]
        except Exception as e:
            print(f"  WARN: {label} thumb failed for {tile}: {e}")

    return info

# ── generate S1 thumbs ──────────────────────────────────────────
def process_s1_tile(tile, split):
    base = f"{DATA}/sentinel-1/{split}/{tile}__s1_rtc"
    if not os.path.exists(base):
        return {}
    files = sorted(glob.glob(f"{base}/{tile}__s1_rtc_*.tif"))
    info = {"s1_count": len(files), "s1_asc": 0, "s1_desc": 0}
    for f in files:
        if "ascending" in f:
            info["s1_asc"] += 1
        else:
            info["s1_desc"] += 1

    # Thumb from first ascending
    asc_files = [f for f in files if "ascending" in f]
    if asc_files:
        try:
            with rasterio.open(asc_files[0]) as src:
                d = src.read(1)
                db = np.where(d > 0, 10 * np.log10(np.maximum(d, 1e-10)), 0)
                rgb = sar_cmap(db)
                save_thumb(rgb, f"{OUT}/thumbs/{tile}_s1.png")
                info["s1_shape"] = [int(d.shape[0]), int(d.shape[1])]
        except Exception as e:
            print(f"  WARN: S1 thumb failed for {tile}: {e}")
    return info

# ── generate label thumbs ───────────────────────────────────────
def process_labels(tile):
    info = {"radd": {}, "gladl": {}, "glads2": {}}

    # RADD
    radd_path = f"{DATA}/labels/train/radd/radd_{tile}_labels.tif"
    if os.path.exists(radd_path):
        with rasterio.open(radd_path) as src:
            d = src.read(1)
            mask = d > 0
            # Post-2020 alerts: day_offset >= 2193
            day_offset = d % 10000
            post_2020 = (mask) & (day_offset >= 2193)
            info["radd"]["total_alerts"] = int(np.count_nonzero(mask))
            info["radd"]["post2020_alerts"] = int(np.count_nonzero(post_2020))
            info["radd"]["total_pixels"] = int(d.size)
            info["radd"]["deforest_pct"] = round(float(np.count_nonzero(post_2020)) / d.size * 100, 2)
            # Confidence split
            conf = d // 10000
            info["radd"]["high_conf"] = int(np.count_nonzero(conf == 3))
            info["radd"]["low_conf"]  = int(np.count_nonzero(conf == 2))

            # Thumb: show post-2020 alerts
            rgb = green_red_cmap(post_2020.astype(np.uint8))
            save_thumb(rgb, f"{OUT}/thumbs/{tile}_radd.png")

    # GLAD-L (combine years)
    gladl_total = 0
    for yr in [21, 22, 23, 24, 25]:
        alert_path = f"{DATA}/labels/train/gladl/gladl_{tile}_alert{yr}.tif"
        if os.path.exists(alert_path):
            with rasterio.open(alert_path) as src:
                d = src.read(1)
                confirmed = np.count_nonzero(d >= 2)
                gladl_total += confirmed
                info["gladl"][f"20{yr}"] = int(confirmed)
    info["gladl"]["total"] = gladl_total

    # GLAD-S2
    gs2_path = f"{DATA}/labels/train/glads2/glads2_{tile}_alert.tif"
    if os.path.exists(gs2_path):
        with rasterio.open(gs2_path) as src:
            d = src.read(1)
            info["glads2"]["total"] = int(np.count_nonzero(d > 0))
            info["glads2"]["high_conf"] = int(np.count_nonzero(d >= 3))
            info["glads2"]["med_conf"]  = int(np.count_nonzero(d == 2))
            info["glads2"]["low_conf"]  = int(np.count_nonzero(d == 1))

            # Combined label thumb (all 3 sources overlaid with different colors)
            h, w = d.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, 1] = 30  # dark bg
            rgb[d >= 3, 0] = 255  # high conf red
            rgb[d >= 3, 1] = 50
            rgb[d == 2, 0] = 255  # med conf orange
            rgb[d == 2, 1] = 150
            rgb[d == 1, 0] = 255  # low conf yellow
            rgb[d == 1, 1] = 255
            save_thumb(rgb, f"{OUT}/thumbs/{tile}_glads2.png")

    return info

# ── AEF embeddings ──────────────────────────────────────────────
def process_aef(tile, split):
    base = f"{DATA}/aef-embeddings/{split}"
    files = sorted(glob.glob(f"{base}/{tile}_*.tiff"))
    info = {"aef_count": len(files)}
    if files:
        try:
            with rasterio.open(files[0]) as src:
                info["aef_bands"] = src.count
                info["aef_shape"] = [int(src.height), int(src.width)]
                # PCA-like visualization: take first 3 components
                d = src.read()  # (64, H, W)
                # Use components 0, 1, 2 as RGB
                c0 = percentile_stretch(d[0].astype(float))
                c1 = percentile_stretch(d[1].astype(float))
                c2 = percentile_stretch(d[2].astype(float))
                rgb = np.stack([c0, c1, c2], axis=-1)
                save_thumb(rgb, f"{OUT}/thumbs/{tile}_aef.png")
        except Exception as e:
            print(f"  WARN: AEF failed for {tile}: {e}")
    return info

# ── temporal NDVI time series for a tile ─────────────────────────
def compute_ndvi_timeseries(tile, split):
    """Compute mean NDVI for each monthly S2 file."""
    base = f"{DATA}/sentinel-2/{split}/{tile}__s2_l2a"
    files = sorted(glob.glob(f"{base}/{tile}__s2_l2a_*.tif"))
    series = []
    for f in files:
        fname = os.path.basename(f).replace(".tif", "")
        parts = fname.split("_")
        # Extract year and month
        # Pattern: {tile}__s2_l2a_{year}_{month}
        seg = fname.split("__s2_l2a_")[1]  # "2020_1"
        yr, mo = seg.split("_")
        try:
            with rasterio.open(f) as src:
                nir = src.read(8).astype(float)  # B08 (1-indexed band 8)
                red = src.read(4).astype(float)  # B04
                valid = (nir + red) > 0
                if np.any(valid):
                    ndvi_vals = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
                    series.append({
                        "year": int(yr), "month": int(mo),
                        "ndvi_mean": round(float(np.mean(ndvi_vals)), 4),
                        "ndvi_std": round(float(np.std(ndvi_vals)), 4),
                        "valid_pct": round(float(np.sum(valid)) / valid.size * 100, 1)
                    })
        except Exception as e:
            pass
    return sorted(series, key=lambda x: (x["year"], x["month"]))

# ── PROCESS ALL TILES ────────────────────────────────────────────
print("\n=== Processing TRAIN tiles ===")
for tile in train_tiles:
    print(f"  {tile}...")
    coord = train_coords.get(tile, {"lat": 0, "lon": 0, "bbox": [0,0,0,0]})
    info = {
        "split": "train",
        "region": get_region(coord["lon"]),
        "lat": coord["lat"],
        "lon": coord["lon"],
        "bbox": coord["bbox"]
    }
    info.update(process_s2_tile(tile, "train"))
    info.update(process_s1_tile(tile, "train"))
    info["labels"] = process_labels(tile)
    info.update(process_aef(tile, "train"))
    info["ndvi_ts"] = compute_ndvi_timeseries(tile, "train")
    stats["train"][tile] = info

print("\n=== Processing TEST tiles ===")
for tile in test_tiles:
    print(f"  {tile}...")
    coord = test_coords.get(tile, {"lat": 0, "lon": 0, "bbox": [0,0,0,0]})
    info = {
        "split": "test",
        "region": get_region(coord["lon"]),
        "lat": coord["lat"],
        "lon": coord["lon"],
        "bbox": coord["bbox"]
    }
    info.update(process_s2_tile(tile, "test"))
    info.update(process_s1_tile(tile, "test"))
    info.update(process_aef(tile, "test"))
    info["ndvi_ts"] = compute_ndvi_timeseries(tile, "test")
    stats["test"][tile] = info

# ── additional: Generate time-lapse thumbs (6 frames) for one example tile ──
print("\n=== Time-lapse for 18NWG_6_6 ===")
example_tile = "18NWG_6_6"
for year in range(2020, 2026):
    month_path = f"{DATA}/sentinel-2/train/{example_tile}__s2_l2a/{example_tile}__s2_l2a_{year}_6.tif"
    if not os.path.exists(month_path):
        month_path = f"{DATA}/sentinel-2/train/{example_tile}__s2_l2a/{example_tile}__s2_l2a_{year}_1.tif"
    if os.path.exists(month_path):
        with rasterio.open(month_path) as src:
            bands = src.read()
            r = percentile_stretch(bands[3].astype(float))
            g = percentile_stretch(bands[2].astype(float))
            b = percentile_stretch(bands[1].astype(float))
            rgb = np.stack([r, g, b], axis=-1)
            save_thumb(rgb, f"{OUT}/thumbs/timelapse_{year}.png", 320)
        print(f"  {year} OK")

# ── Save JSON ────────────────────────────────────────────────────
with open(f"{OUT}/data_stats.json", "w") as f:
    json.dump(stats, f, indent=1)
print(f"\nDone! Stats → {OUT}/data_stats.json")
print(f"Thumbs → {OUT}/thumbs/ ({len(os.listdir(f'{OUT}/thumbs'))} files)")

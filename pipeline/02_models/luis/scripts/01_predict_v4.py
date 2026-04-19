#!/usr/bin/env python3
"""V4: Probability-level fusion of Eklavya (42.30% leaderboard) + our LGBM/UNet.

Strategy
--------
1. Load Eklavya's calibrated stacked probability raster (5-model ensemble:
   LGBM tabular + TCN + 3 U-Nets, isotonic + logistic stacker).
2. Generate our own probability from the with-2026 LGBM + U-Net (same grid).
3. Weighted fuse: p = w_ekl * p_ekl + (1 - w_ekl) * p_ours.
4. Per-region threshold tuned to roughly match expected positive rates
   (informed by eklavya's leaderboard area).
5. Adaptive fallback per tile: if 0 polygons after threshold, re-threshold
   at the tile's 99.5th percentile (Eklavya trick that recovered 47QMA).
6. Light morphology + per-region min-area filter.
7. Per-component time_step from NBR max-drop year (YYMM, month=06).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape
import geopandas as gpd
from scipy.ndimage import label as cc_label, binary_closing

from src.config import DATA, MODELS, TEST_TILES
from src.regions import tile_region
from src.predict_lib import load_features, predict_lgbm_one, predict_unet_one
from src.timing import estimate_polygon_years

warnings.filterwarnings("ignore")

EKL_DIR = ROOT / "ekl_submission"
SUBMIT_DIR = ROOT / "submissions" / "v4"
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

# Fusion weight on Eklavya (leaderboard-validated 5-model stack).
W_EKL = 0.55

# Per-region thresholds on FUSED probability.
# Lower than Eklavya's solo thresholds (Amazon 0.51 / Asia 0.95 / Africa 0.73)
# because adding our LGBM+UNet drags very-high-confidence pixels toward 0.5.
REGION_THR = {
    "amazon": 0.42,
    "asia":   0.55,
    "africa": 0.50,
}

# Per-region min polygon area in hectares.
REGION_MIN_AREA = {
    "amazon": 0.30,
    "asia":   0.20,
    "africa": 0.20,
}

# Adaptive fallback percentile when tile produces no polygons.
ADAPTIVE_PCT = 99.5
ADAPTIVE_MIN_PIX = 200          # don't trigger if barely any signal at all
ADAPTIVE_MAX_FRAC = 0.02        # cap fallback at 2% of pixels


def get_geo_ref(tile: str):
    s2_dir = DATA / "sentinel-2" / "test" / f"{tile}__s2_l2a"
    s2_files = sorted(s2_dir.glob("*.tif"))
    if not s2_files:
        return None, None
    with rasterio.open(s2_files[0]) as ref:
        return ref.transform, ref.crs


def load_eklavya_prob(tile: str, target_shape: tuple, target_transform, target_crs):
    """Load eklavya prob (uint16/1000) and reproject to our grid if mismatched."""
    src_path = EKL_DIR / f"prob_{tile}.tif"
    if not src_path.exists():
        return None
    with rasterio.open(src_path) as ds:
        if ds.shape == target_shape and ds.crs == target_crs:
            return ds.read(1).astype(np.float32) / 1000.0
        # Reproject onto our grid.
        dst = np.zeros(target_shape, dtype=np.uint16)
        reproject(
            source=rasterio.band(ds, 1),
            destination=dst,
            src_transform=ds.transform, src_crs=ds.crs,
            dst_transform=target_transform, dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
        return dst.astype(np.float32) / 1000.0


def vectorize_with_years(binary, transform, src_crs, years_by_cid, min_area_ha):
    labels, n = cc_label(binary.astype(bool))
    if n == 0:
        return gpd.GeoDataFrame(columns=["time_step", "geometry"], crs=src_crs)
    polygons, cids = [], []
    for geom, value in shapes(labels.astype(np.int32),
                              mask=binary.astype(bool), transform=transform):
        cid = int(value)
        if cid <= 0:
            continue
        polygons.append(shape(geom))
        cids.append(cid)
    gdf = gpd.GeoDataFrame({"cid": cids, "geometry": polygons}, crs=src_crs)
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["time_step", "geometry"], crs=src_crs)
    utm = gdf.estimate_utm_crs()
    areas = gdf.to_crs(utm).geometry.area / 10_000.0
    keep = areas >= min_area_ha
    gdf = gdf[keep].reset_index(drop=True)
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["time_step", "geometry"], crs=src_crs)

    def yr_to_yymm(y):
        if y is None:
            return None
        return (int(y) % 100) * 100 + 6

    gdf_4326 = gdf.to_crs("EPSG:4326")
    gdf_4326["time_step"] = gdf["cid"].map(lambda c: yr_to_yymm(years_by_cid.get(int(c))))
    return gdf_4326[["time_step", "geometry"]]


def main():
    lgbm_full = MODELS / "lgbm_full.txt"
    unet_full = MODELS / "unet_full.pt"
    assert lgbm_full.exists() and unet_full.exists()

    print(f"V4 settings: W_EKL={W_EKL}, REGION_THR={REGION_THR}, ADAPTIVE_PCT={ADAPTIVE_PCT}")

    all_gdfs = []
    summary = []
    for tile in TEST_TILES:
        region = tile_region(tile)
        thr = REGION_THR[region]
        min_area = REGION_MIN_AREA[region]
        print(f"\n{tile}  region={region}  thr={thr:.2f}  min_area={min_area}ha")

        feats, meta = load_features(tile)
        if feats is None:
            print(f"  [skip] no features cached")
            continue
        H, W = meta["shape"]

        transform, crs = get_geo_ref(tile)
        if transform is None:
            print(f"  [skip] no S2 reference")
            continue

        # Our probability.
        p_lgbm = predict_lgbm_one(feats, lgbm_full)
        p_unet = predict_unet_one(feats, meta, unet_full)
        p_ours = (0.5 * p_lgbm + 0.5 * p_unet).reshape(H, W)

        # Eklavya's calibrated probability (5-model stack, leaderboard 42.30%).
        p_ekl = load_eklavya_prob(tile, (H, W), transform, crs)
        if p_ekl is None:
            print(f"  [warn] no eklavya prob, using ours alone")
            p_fused = p_ours
        else:
            p_fused = W_EKL * p_ekl + (1.0 - W_EKL) * p_ours
            print(f"  prob means: ekl={p_ekl.mean():.3f}  ours={p_ours.mean():.3f}  fused={p_fused.mean():.3f}")
            print(f"  prob p99 : ekl={np.percentile(p_ekl,99):.3f}  ours={np.percentile(p_ours,99):.3f}  fused={np.percentile(p_fused,99):.3f}")

        # Initial threshold.
        binary = (p_fused >= thr).astype(np.uint8)
        n_raw = int(binary.sum())
        used_fallback = False

        # Adaptive fallback: if no real signal, try 99.5th percentile.
        if n_raw < ADAPTIVE_MIN_PIX:
            adaptive_thr = float(np.percentile(p_fused, ADAPTIVE_PCT))
            adaptive_thr = max(adaptive_thr, 0.20)        # never accept garbage
            binary = (p_fused >= adaptive_thr).astype(np.uint8)
            n_raw = int(binary.sum())
            cap = int(ADAPTIVE_MAX_FRAC * H * W)
            if n_raw > cap:
                # Too many — bump threshold up further until below cap.
                k = max(1, H * W - cap)
                kth = np.partition(p_fused.ravel(), k - 1)[k - 1]
                binary = (p_fused >= kth).astype(np.uint8)
                n_raw = int(binary.sum())
                adaptive_thr = float(kth)
            used_fallback = True
            print(f"  ADAPTIVE fallback: thr={adaptive_thr:.3f} → {n_raw:,} pixels")

        # Light morphology — closing only (fills holes, doesn't erode).
        binary_pp = binary_closing(binary.astype(bool), iterations=1).astype(np.uint8)
        n_post = int(binary_pp.sum())
        print(f"  pixels: raw={n_raw:,}  post={n_post:,}  ({100*n_post/(H*W):.2f}%)")

        # Save raster artefacts (for debugging / future fusion).
        prob_out = SUBMIT_DIR / f"prob_{tile}.tif"
        with rasterio.open(prob_out, "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="uint16", crs=crs, transform=transform,
                           compress="deflate") as dst:
            dst.write((p_fused * 1000).clip(0, 65535).astype(np.uint16)[np.newaxis])
        pred_out = SUBMIT_DIR / f"pred_{tile}.tif"
        with rasterio.open(pred_out, "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="uint8", crs=crs, transform=transform,
                           compress="deflate") as dst:
            dst.write(binary_pp[np.newaxis])

        # Per-component years.
        years = estimate_polygon_years(binary_pp, tile, "test", transform, crs, (H, W))

        gdf = vectorize_with_years(binary_pp, transform, crs, years, min_area)
        print(f"  polygons: {len(gdf)}")
        if not gdf.empty:
            year_dist = gdf["time_step"].value_counts().to_dict()
            print(f"  YYMM dist: {year_dist}")
            all_gdfs.append(gdf)
        summary.append({"tile": tile, "region": region, "threshold": thr,
                        "fallback": used_fallback,
                        "pix_raw": n_raw, "pix_post": n_post, "polygons": len(gdf)})

    if not all_gdfs:
        print("\nNo predictions.")
        return

    combined = gpd.pd.concat(all_gdfs, ignore_index=True)
    out_path = SUBMIT_DIR / "submission.geojson"
    combined.to_file(out_path, driver="GeoJSON")
    print(f"\nSubmission → {out_path}  ({len(combined)} polygons)")

    with open(SUBMIT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

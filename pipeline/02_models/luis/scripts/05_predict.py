#!/usr/bin/env python3
"""
Step 4: Ensemble inference + GeoJSON submission.

Usage:
    python scripts/04_predict.py                  # full ensemble
    python scripts/04_predict.py --lgbm-only      # LightGBM only
    python scripts/04_predict.py --unet-only      # U-Net only
    python scripts/04_predict.py --threshold 0.5  # custom threshold
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
import geopandas as gpd

from src.config import CACHE, MODELS, SUBMIT, DATA, TEST_TILES, N_FEATURES

warnings.filterwarnings("ignore")


# ── LightGBM inference ────────────────────────────────────────────────────────
def predict_lgbm(tile, model_paths):
    """Average predictions from all available LightGBM models."""
    import lightgbm as lgb

    fp = CACHE / f"{tile}_features.npz"
    mp = CACHE / f"{tile}_meta.json"
    if not fp.exists():
        return None, None

    feats = np.load(fp)["features"]
    with open(mp) as f:
        meta = json.load(f)

    preds = np.zeros(len(feats), dtype=np.float64)
    n_models = 0
    for p in model_paths:
        if not Path(p).exists():
            continue
        m = lgb.Booster(model_file=str(p))
        preds += m.predict(feats)
        n_models += 1

    if n_models == 0:
        return None, None
    preds /= n_models
    return preds.astype(np.float32), meta


# ── U-Net inference ───────────────────────────────────────────────────────────
def predict_unet(tile, model_path, patch_size=256, batch_size=64):
    """Sliding window inference with 50% overlap."""
    import torch
    from src.models.unet import build_model

    fp = CACHE / f"{tile}_features.npz"
    mp = CACHE / f"{tile}_meta.json"
    if not fp.exists():
        return None, None

    feats = np.load(fp)["features"]
    with open(mp) as f:
        meta = json.load(f)
    H, W = meta["shape"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feat_map = feats.reshape(H, W, N_FEATURES).transpose(2, 0, 1)
    pad = patch_size // 2
    feat_pad = np.pad(feat_map, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    patches, coords = [], []
    for y in range(0, H, patch_size // 2):
        for x in range(0, W, patch_size // 2):
            yc, xc = min(y, H - 1), min(x, W - 1)
            patch = feat_pad[:, yc:yc + patch_size, xc:xc + patch_size]
            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                p2 = np.zeros((N_FEATURES, patch_size, patch_size), dtype=np.float32)
                ph, pw = min(patch.shape[1], patch_size), min(patch.shape[2], patch_size)
                p2[:, :ph, :pw] = patch[:, :ph, :pw]
                patch = p2
            patches.append(patch.astype(np.float32))
            coords.append((yc, xc))

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.from_numpy(np.stack(patches[i:i + batch_size])).to(device)
            probs = torch.sigmoid(model(batch).squeeze(1)).cpu().numpy()
            for j, (yc, xc) in enumerate(coords[i:i + batch_size]):
                ph = min(patch_size, H - yc)
                pw = min(patch_size, W - xc)
                prob_map[yc:yc + ph, xc:xc + pw] += probs[j, :ph, :pw]
                count_map[yc:yc + ph, xc:xc + pw] += 1.0

    prob_map /= np.maximum(count_map, 1)
    return prob_map.ravel(), meta


# ── Ensemble ──────────────────────────────────────────────────────────────────
def build_submission(use_lgbm=True, use_unet=True, threshold=0.4):
    from submission_utils import raster_to_geojson

    lgbm_models = sorted(MODELS.glob("lgbm_*.txt")) if use_lgbm else []
    unet_model = MODELS / "unet_full.pt"
    if not unet_model.exists():
        unet_model = MODELS / "unet_holdout.pt"

    lgbm_weight = 0.5 if use_lgbm and use_unet else 1.0
    unet_weight = 0.5 if use_lgbm and use_unet else 1.0

    all_gdfs = []

    for tile in TEST_TILES:
        print(f"\n{tile}")
        probs, meta = None, None

        if use_lgbm and lgbm_models:
            result = predict_lgbm(tile, lgbm_models)
            if result[0] is not None:
                p_lgbm, meta = result
                probs = p_lgbm * lgbm_weight

        if use_unet and unet_model.exists():
            result = predict_unet(tile, unet_model)
            if result[0] is not None:
                p_unet, meta = result
                probs = (probs + p_unet * unet_weight) if probs is not None else p_unet

        if probs is None or meta is None:
            print(f"  Skipping — no predictions")
            continue

        H, W = meta["shape"]
        prob_map = probs.reshape(H, W)
        pred_map = (prob_map >= threshold).astype(np.uint8)
        print(f"  {pred_map.sum():,} deforestation pixels ({100 * pred_map.mean():.2f}%)")

        # Get geo reference from source S2 raster
        split = meta.get("split", "test")
        s2_dir = DATA / "sentinel-2" / split
        tile_dirs = sorted(s2_dir.glob(f"{tile}*"))
        if not tile_dirs:
            continue
        s2_files = sorted(tile_dirs[0].glob("*.tif"))
        if not s2_files:
            continue
        with rasterio.open(s2_files[0]) as ref:
            transform, crs = ref.transform, ref.crs

        # Write prediction raster
        pred_path = SUBMIT / f"{tile}_pred.tif"
        with rasterio.open(pred_path, "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="uint8", crs=crs, transform=transform) as dst:
            dst.write(pred_map[np.newaxis])

        # Vectorize
        try:
            geojson = raster_to_geojson(pred_path)
            gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
            if not gdf.empty:
                all_gdfs.append(gdf)
                print(f"  {len(gdf)} polygons")
        except Exception as e:
            print(f"  Vectorization failed: {e}")

    if not all_gdfs:
        print("\nNo predictions.")
        return None

    combined = gpd.pd.concat(all_gdfs, ignore_index=True)
    out_path = SUBMIT / "submission.geojson"
    combined.to_file(out_path, driver="GeoJSON")
    print(f"\nSubmission → {out_path}  ({len(combined)} polygons)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgbm-only", action="store_true")
    parser.add_argument("--unet-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    use_lgbm = not args.unet_only
    use_unet = not args.lgbm_only

    print(f"Inference (lgbm={use_lgbm}, unet={use_unet}, threshold={args.threshold})")
    build_submission(use_lgbm=use_lgbm, use_unet=use_unet, threshold=args.threshold)


if __name__ == "__main__":
    main()

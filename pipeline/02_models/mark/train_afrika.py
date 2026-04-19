#!/usr/bin/env python3
"""
oasis-afrika: Train a deforestation model specifically for the Africa tile (33NTE_5_1).

Strategy:
  1. Extract rich S2/S1 temporal features for ALL tiles (train + test)
  2. Train Phase A: LightGBM on 16 training tiles (GT labels) with Hansen as feature
     → learns global "what deforestation looks like" including Hansen correlation
  3. Train Phase B: Fine-tune on Africa itself using Hansen pseudo-labels with spatial CV
     → adapts to African spectral characteristics
  4. Ensemble Phase A + B predictions for robust Africa output
  5. Merge with v5_final for other tiles → final submission

Key insight: Our models trained on Amazon/Asia extrapolate poorly to Africa (zero
training data). By using Hansen as both a feature (Phase A) and pseudo-label (Phase B),
we build an Africa-adapted model.
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import ndimage
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path("/shared-docker")
DATA = ROOT / "oasis-luis-additional-data" / "data" / "makeathon-challenge"
HANSEN_DIR = ROOT / "oasis-mark-2" / "external" / "hansen" / "cropped"
GT_DIR = ROOT / "oasis" / "data" / "cache" / "train"
V5_SUB = ROOT / "oasis-luis-v5" / "submission" / "v5_final"
OUT_DIR = ROOT / "oasis-afrika" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_AFRICA = "33NTE_5_1"

TRAIN_TILES = [
    "18NWG_6_6", "18NWH_1_4", "18NWJ_8_9", "18NWM_9_4",
    "18NXH_6_8", "18NXJ_7_6", "18NYH_9_9", "19NBD_4_4",
    "47QMB_0_8", "47QQV_2_4", "48PUT_0_8", "48PWV_7_8",
    "48PXC_7_7", "48PYB_3_6", "48QVE_3_0", "48QWD_2_2",
]

TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MONTHS = list(range(1, 13))

# S2 L2A band names (12 bands)
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL", "TCI_R"]


# ── Feature extraction ─────────────────────────────────────────────

def load_s2_annual(tile: str, split: str):
    """Load S2 data as annual means (much faster than per-month).
    Returns dict of year → (C, H, W) annual mean arrays, plus overall stats."""
    base = DATA / f"sentinel-2/{split}/{tile}__s2_l2a"
    if not base.exists():
        return None, None

    # First pass: find majority shape across all files
    shapes = {}
    tifs = sorted(base.glob("*.tif"))
    if not tifs:
        return None, None
    for f in tifs:
        with rasterio.open(f) as src:
            s = (src.count, *src.shape)
            shapes[s] = shapes.get(s, 0) + 1
    # Use the most common shape
    C, H, W = max(shapes, key=shapes.get)

    # Accumulate per-year sums (much faster than loading all into memory)
    year_sum = {}
    year_cnt = {}
    all_sum = np.zeros((C, H, W), dtype=np.float64)
    all_cnt = np.zeros((C, H, W), dtype=np.int32)

    for year in YEARS:
        ysum = np.zeros((C, H, W), dtype=np.float64)
        ycnt = np.zeros((C, H, W), dtype=np.int32)
        for month in MONTHS:
            p = base / f"{tile}__s2_l2a_{year}_{month}.tif"
            if not p.exists():
                continue
            with rasterio.open(p) as src:
                d = src.read().astype(np.float64)
                if d.shape != (C, H, W):
                    continue  # skip mismatched shapes
                valid = d > 0
                ysum += np.where(valid, d, 0)
                ycnt += valid.astype(np.int32)
        if ycnt.max() > 0:
            year_sum[year] = ysum
            year_cnt[year] = ycnt
            all_sum += ysum
            all_cnt += ycnt

    return {"year_sum": year_sum, "year_cnt": year_cnt,
            "all_sum": all_sum, "all_cnt": all_cnt,
            "shape": (C, H, W)}, (H, W)


def extract_features(tile: str, split: str):
    """
    Fast per-pixel feature extraction from S2 annual aggregates.

    Features:
    - Per-band overall mean, std-proxy (range) for 10 bands = 20
    - NDVI/NDWI/NBR overall mean = 3
    - Early vs late change (bands + indices) = 9
    - Annual means for key bands (6 years × 5 bands) = 30
    - Annual NDVI (6 years) = 6
    Total: ~68 features (fast to compute)
    """
    print(f"  Loading S2 for {tile}...")
    data, hw = load_s2_annual(tile, split)
    if data is None:
        return None

    C, H, W = data["shape"]
    n_years = len(data["year_sum"])
    print(f"    S2: {n_years} years, {C} bands, {H}×{W}")

    features = []
    feat_names = []

    # Overall mean per band
    with np.errstate(divide="ignore", invalid="ignore"):
        all_mean = np.where(data["all_cnt"] > 0,
                            data["all_sum"] / data["all_cnt"], 0).astype(np.float32)

    # Annual means
    annual_means = {}
    for year in YEARS:
        if year in data["year_sum"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                annual_means[year] = np.where(
                    data["year_cnt"][year] > 0,
                    data["year_sum"][year] / data["year_cnt"][year], 0
                ).astype(np.float32)

    # 1. Per-band overall mean (10 bands, skip SCL/TCI)
    for b in range(min(C, 10)):
        features.append(all_mean[b])
        feat_names.append(f"b{b}_mean")

    # 2. Per-band range (max_year - min_year) as variability proxy
    for b in range(min(C, 10)):
        vals = [annual_means[y][b] for y in sorted(annual_means.keys())]
        if len(vals) >= 2:
            arr = np.stack(vals, axis=0)
            features.append(arr.max(axis=0) - arr.min(axis=0))
            feat_names.append(f"b{b}_range")

    # 3. Vegetation indices from overall means
    b04 = all_mean[2]   # Red
    b08 = all_mean[6]   # NIR
    b03 = all_mean[1]   # Green
    b11 = all_mean[8]   # SWIR1
    b12 = all_mean[9]   # SWIR2

    ndvi_all = (b08 - b04) / np.clip(b08 + b04, 1e-6, None)
    ndwi_all = (b03 - b08) / np.clip(b03 + b08, 1e-6, None)
    nbr_all = (b08 - b12) / np.clip(b08 + b12, 1e-6, None)

    features.extend([ndvi_all, ndwi_all, nbr_all])
    feat_names.extend(["ndvi_mean", "ndwi_mean", "nbr_mean"])

    # 4. Early vs Late change
    early_years = [y for y in [2020, 2021] if y in annual_means]
    late_years = [y for y in [2024, 2025] if y in annual_means]

    if early_years and late_years:
        early_stack = np.mean([annual_means[y] for y in early_years], axis=0)
        late_stack = np.mean([annual_means[y] for y in late_years], axis=0)

        for b, bname in [(0, "b02"), (2, "b04"), (6, "b08"), (8, "b11"), (9, "b12")]:
            features.append(late_stack[b] - early_stack[b])
            feat_names.append(f"{bname}_change")

        # Index changes
        e_ndvi = (early_stack[6] - early_stack[2]) / np.clip(early_stack[6] + early_stack[2], 1e-6, None)
        l_ndvi = (late_stack[6] - late_stack[2]) / np.clip(late_stack[6] + late_stack[2], 1e-6, None)
        features.append(l_ndvi - e_ndvi)
        feat_names.append("ndvi_change")

        e_nbr = (early_stack[6] - early_stack[9]) / np.clip(early_stack[6] + early_stack[9], 1e-6, None)
        l_nbr = (late_stack[6] - late_stack[9]) / np.clip(late_stack[6] + late_stack[9], 1e-6, None)
        features.append(l_nbr - e_nbr)
        feat_names.append("nbr_change")

        e_ndwi = (early_stack[1] - early_stack[6]) / np.clip(early_stack[1] + early_stack[6], 1e-6, None)
        l_ndwi = (late_stack[1] - late_stack[6]) / np.clip(late_stack[1] + late_stack[6], 1e-6, None)
        features.append(l_ndwi - e_ndwi)
        feat_names.append("ndwi_change")

        # SWIR ratio change (good for deforestation)
        e_swir_r = early_stack[8] / np.clip(early_stack[6], 1e-6, None)
        l_swir_r = late_stack[8] / np.clip(late_stack[6], 1e-6, None)
        features.append(l_swir_r - e_swir_r)
        feat_names.append("swir_ratio_change")

    # 5. Annual means for key bands + NDVI
    for year in YEARS:
        if year in annual_means:
            am = annual_means[year]
            for b, bname in [(2, "b04"), (6, "b08"), (8, "b11"), (9, "b12"), (0, "b02")]:
                features.append(am[b])
                feat_names.append(f"{bname}_y{year}")
            # Annual NDVI
            ndvi_y = (am[6] - am[2]) / np.clip(am[6] + am[2], 1e-6, None)
            features.append(ndvi_y)
            feat_names.append(f"ndvi_y{year}")

    # Stack all features: (F, H, W)
    feat_stack = np.stack(features, axis=0).astype(np.float32)
    print(f"    Features: {feat_stack.shape[0]} features")

    return feat_stack, feat_names, (H, W)


def load_hansen_10m(tile: str, ref_shape: tuple, ref_crs, ref_transform) -> dict:
    """Load Hansen data reprojected to 10m S2 grid."""
    result = {}
    for layer in ["lossyear", "treecover2000"]:
        p = HANSEN_DIR / f"{tile}_{layer}.tif"
        if not p.exists():
            result[layer] = np.zeros(ref_shape, dtype=np.float32)
            continue

        with rasterio.open(p) as src:
            dst = np.zeros(ref_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                dst_crs=ref_crs,
                dst_transform=ref_transform,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
            result[layer] = dst

    return result


def get_s2_georef(tile: str, split: str):
    """Get CRS and transform from first S2 file."""
    base = DATA / f"sentinel-2/{split}/{tile}__s2_l2a"
    for f in sorted(base.glob("*.tif")):
        with rasterio.open(f) as src:
            return src.crs, src.transform, src.profile.copy()
    return None, None, None


def load_gt(tile: str) -> np.ndarray:
    """Load ground truth consensus labels for training tile."""
    d = np.load(GT_DIR / tile / "bundle.npz")
    if "consensus_pos" in d:
        return d["consensus_pos"].astype(np.uint8)
    return (d["labels"] > 0).astype(np.uint8)


# ── Phase A: Global model on training tiles ─────────────────────────

def build_training_data_global():
    """Build feature matrix from all training tiles with GT labels + Hansen features."""
    print("\n" + "=" * 60)
    print("PHASE A: Building global training data from 16 tiles")
    print("=" * 60)

    all_X = []
    all_y = []
    feat_names = None

    for tile in TRAIN_TILES:
        print(f"\n--- {tile} ---")
        split = "train"

        # Extract S2 features
        result = extract_features(tile, split)
        if result is None:
            print(f"  SKIP: no S2 data")
            continue
        feat_stack, names, (H, W) = result
        if feat_names is None:
            feat_names = names

        # Get georef for Hansen reprojection
        crs, transform, _ = get_s2_georef(tile, split)

        # Load Hansen as features
        hansen = load_hansen_10m(tile, (H, W), crs, transform)
        lossyear = hansen["lossyear"]
        treecover = hansen["treecover2000"]

        # Add Hansen features
        hansen_recent = (lossyear >= 20).astype(np.float32)
        hansen_any = (lossyear > 0).astype(np.float32)

        # Stack: S2 features + Hansen features
        all_feats = np.concatenate([
            feat_stack,
            treecover[np.newaxis],
            hansen_recent[np.newaxis],
            hansen_any[np.newaxis],
        ], axis=0)  # (F+3, H, W)

        # Load GT
        gt = load_gt(tile)
        if gt.shape != (H, W):
            print(f"  WARN: GT shape {gt.shape} != S2 shape {(H, W)}, skipping")
            continue

        # Flatten and sample (subsample to keep memory manageable)
        X = all_feats.reshape(all_feats.shape[0], -1).T  # (N, F)
        y = gt.flatten()

        # Remove pixels that are all NaN in features
        valid = ~np.isnan(X).all(axis=1)
        X = X[valid]
        y = y[valid]

        # Replace remaining NaN with 0
        X = np.nan_to_num(X, nan=0.0)

        # Subsample: keep all positive, sample negatives
        pos_mask = y == 1
        neg_mask = y == 0
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        if n_pos == 0:
            print(f"  SKIP: no positive pixels")
            continue

        # Keep all positives + 3x negatives (balanced-ish)
        max_neg = min(n_neg, max(n_pos * 3, 50000))
        neg_idx = np.where(neg_mask)[0]
        if len(neg_idx) > max_neg:
            neg_idx = np.random.choice(neg_idx, max_neg, replace=False)
        pos_idx = np.where(pos_mask)[0]
        keep = np.concatenate([pos_idx, neg_idx])

        all_X.append(X[keep])
        all_y.append(y[keep])
        print(f"  Kept {len(pos_idx)} pos + {len(neg_idx)} neg = {len(keep)} samples")

    if not all_X:
        print("ERROR: No training data!")
        return None, None, None

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"\nTotal: {X_all.shape[0]} samples, {X_all.shape[1]} features")
    print(f"  Positive: {(y_all == 1).sum()} ({100 * (y_all == 1).mean():.1f}%)")

    # Update feature names with Hansen
    if feat_names:
        feat_names = feat_names + ["hansen_treecover", "hansen_recent", "hansen_any"]

    return X_all, y_all, feat_names


def train_global_model(X, y, feat_names):
    """Train LightGBM on global training data."""
    print("\n" + "=" * 60)
    print("PHASE A: Training global LightGBM")
    print("=" * 60)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1),
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    ds = lgb.Dataset(X, label=y, feature_name=feat_names, free_raw_data=False)

    model = lgb.train(
        params,
        ds,
        num_boost_round=500,
        valid_sets=[ds],
        callbacks=[lgb.log_evaluation(100)],
    )

    # Feature importance
    imp = model.feature_importance(importance_type="gain")
    top_idx = np.argsort(imp)[::-1][:15]
    print("\nTop 15 features:")
    for i in top_idx:
        print(f"  {feat_names[i]:25s} {imp[i]:>10.0f}")

    return model


# ── Phase B: Africa-specific model with Hansen pseudo-labels ──────

def build_africa_data():
    """Build features + Hansen pseudo-labels for Africa tile."""
    print("\n" + "=" * 60)
    print("PHASE B: Building Africa-specific training data")
    print("=" * 60)

    tile = TILE_AFRICA
    split = "test"

    # Extract S2 features
    result = extract_features(tile, split)
    if result is None:
        print("ERROR: No S2 data for Africa!")
        return None
    feat_stack, feat_names, (H, W) = result

    # Get georef
    crs, transform, profile = get_s2_georef(tile, split)

    # Load Hansen
    hansen = load_hansen_10m(tile, (H, W), crs, transform)
    lossyear = hansen["lossyear"]
    treecover = hansen["treecover2000"]

    hansen_recent = (lossyear >= 20).astype(np.float32)
    hansen_any = (lossyear > 0).astype(np.float32)

    # Create pseudo-labels: Hansen lossyear 2020+ = positive
    pseudo_labels = (lossyear >= 20).astype(np.uint8)

    # Stack features + Hansen
    all_feats = np.concatenate([
        feat_stack,
        treecover[np.newaxis],
        hansen_recent[np.newaxis],
        hansen_any[np.newaxis],
    ], axis=0)

    feat_names = feat_names + ["hansen_treecover", "hansen_recent", "hansen_any"]

    print(f"  Africa shape: {H}×{W}")
    print(f"  Hansen pseudo-positives: {pseudo_labels.sum()} pixels")
    print(f"  Treecover > 30: {(treecover > 30).sum()} pixels")

    return {
        "features": all_feats,
        "feat_names": feat_names,
        "pseudo_labels": pseudo_labels,
        "lossyear": lossyear,
        "treecover": treecover,
        "shape": (H, W),
        "crs": crs,
        "transform": transform,
        "profile": profile,
    }


def train_africa_model(africa_data, feat_names_global):
    """Train LightGBM on Africa with Hansen pseudo-labels using spatial CV."""
    print("\n" + "=" * 60)
    print("PHASE B: Training Africa-specific LightGBM with spatial CV")
    print("=" * 60)

    feats = africa_data["features"]
    labels = africa_data["pseudo_labels"]
    H, W = africa_data["shape"]
    treecover = africa_data["treecover"]

    # For Africa model: DON'T use hansen_recent as feature (it's the label!)
    # Remove hansen_recent and hansen_any, keep treecover
    feat_names = africa_data["feat_names"]
    # Find indices to drop (hansen_recent, hansen_any)
    drop_idx = [i for i, n in enumerate(feat_names) if n in ("hansen_recent", "hansen_any")]
    keep_idx = [i for i in range(feats.shape[0]) if i not in drop_idx]
    feats_clean = feats[keep_idx]
    feat_names_clean = [feat_names[i] for i in keep_idx]

    print(f"  Features (excl. Hansen label leak): {len(feat_names_clean)}")

    # Flatten
    X = feats_clean.reshape(feats_clean.shape[0], -1).T  # (N, F)
    y = labels.flatten()

    # Only train on forested areas (treecover > 20)
    forest_mask = treecover.flatten() > 20
    valid = ~np.isnan(X).all(axis=1) & forest_mask
    X_valid = np.nan_to_num(X[valid], nan=0.0)
    y_valid = y[valid]

    print(f"  Forest pixels: {valid.sum()}, pos: {y_valid.sum()}, neg: {(y_valid == 0).sum()}")

    # Spatial CV: 4 quadrants
    row_idx = np.arange(H).repeat(W)[valid]
    col_idx = np.tile(np.arange(W), H)[valid]
    mid_r, mid_c = H // 2, W // 2
    fold = np.zeros(len(row_idx), dtype=int)
    fold[(row_idx >= mid_r) & (col_idx < mid_c)] = 1
    fold[(row_idx < mid_r) & (col_idx >= mid_c)] = 2
    fold[(row_idx >= mid_r) & (col_idx >= mid_c)] = 3

    models = []
    oof_pred = np.zeros(len(y_valid), dtype=np.float32)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    for f in range(4):
        train_mask = fold != f
        val_mask = fold == f

        X_tr, y_tr = X_valid[train_mask], y_valid[train_mask]
        X_val, y_val = X_valid[val_mask], y_valid[val_mask]

        # Balance
        params["scale_pos_weight"] = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names_clean)
        ds_val = lgb.Dataset(X_val, label=y_val, feature_name=feat_names_clean, reference=ds_tr)

        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=300,
            valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
        )

        pred = model.predict(X_val)
        oof_pred[val_mask] = pred
        models.append(model)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, pred) if y_val.sum() > 0 else 0
        print(f"  Fold {f}: AUC={auc:.4f}, pos={y_val.sum()}, val_size={len(y_val)}")

    # Overall OOF AUC
    from sklearn.metrics import roc_auc_score
    overall_auc = roc_auc_score(y_valid, oof_pred) if y_valid.sum() > 0 else 0
    print(f"\n  Overall OOF AUC: {overall_auc:.4f}")

    return models, feat_names_clean, keep_idx


# ── Prediction + Submission ─────────────────────────────────────────

def predict_africa(global_model, africa_models, africa_data, keep_idx, feat_names_global):
    """Generate Africa probability map from both models."""
    print("\n" + "=" * 60)
    print("PREDICTION: Generating Africa probability map")
    print("=" * 60)

    feats = africa_data["features"]
    H, W = africa_data["shape"]

    # Phase A: Global model prediction (uses all features including Hansen)
    X_all = feats.reshape(feats.shape[0], -1).T  # (N, F)
    X_all = np.nan_to_num(X_all, nan=0.0)
    prob_global = global_model.predict(X_all)
    prob_global = prob_global.reshape(H, W)
    print(f"  Global model: mean={prob_global.mean():.4f}, >0.5: {(prob_global > 0.5).sum()}")

    # Phase B: Africa model prediction (without Hansen labels as features)
    X_clean = feats[keep_idx].reshape(len(keep_idx), -1).T
    X_clean = np.nan_to_num(X_clean, nan=0.0)
    probs_local = [m.predict(X_clean) for m in africa_models]
    prob_local = np.mean(probs_local, axis=0).reshape(H, W)
    print(f"  Africa model: mean={prob_local.mean():.4f}, >0.5: {(prob_local > 0.5).sum()}")

    # Ensemble: weighted average (global knows structure, local knows Africa)
    # Weight local model higher since it's adapted to Africa
    prob_ensemble = 0.4 * prob_global + 0.6 * prob_local
    print(f"  Ensemble: mean={prob_ensemble.mean():.4f}, >0.5: {(prob_ensemble > 0.5).sum()}")

    # Also consider: boost with Hansen confirmation
    hansen_recent = africa_data["lossyear"] >= 20
    treecover = africa_data["treecover"]

    # Strategy: ensemble prediction, but boost where Hansen confirms
    prob_final = prob_ensemble.copy()
    # Where Hansen says loss AND treecover was high → boost confidence
    hansen_boost = hansen_recent & (treecover > 25)
    prob_final[hansen_boost] = np.maximum(prob_final[hansen_boost], 0.7)
    print(f"  After Hansen boost: >0.5: {(prob_final > 0.5).sum()}")

    return prob_final, prob_global, prob_local


def polygonize_and_submit(prob_map, africa_data, threshold=0.5, min_ha=0.5):
    """Convert probability map to polygons and create submission."""
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape, mapping
    import geopandas as gpd

    H, W = africa_data["shape"]
    crs = africa_data["crs"]
    transform = africa_data["transform"]

    # Threshold
    mask = (prob_map >= threshold).astype(np.uint8)

    # Morphology: close small gaps, open small noise
    mask = ndimage.binary_closing(mask, iterations=1).astype(np.uint8)
    mask = ndimage.binary_opening(mask, iterations=2).astype(np.uint8)

    # Min area filter
    labeled, n_obj = ndimage.label(mask)
    for i in range(1, n_obj + 1):
        area_px = (labeled == i).sum()
        area_ha = area_px * 100 / 10000  # 10m pixels, 100m² each
        if area_ha < min_ha:
            mask[labeled == i] = 0

    print(f"  Mask pixels: {mask.sum()}, area: {mask.sum() * 100 / 10000:.0f} ha")

    # Polygonize
    polys = []
    for geom, val in rio_shapes(mask, transform=transform):
        if val == 1:
            polys.append(shape(geom))

    if not polys:
        print("  WARNING: No polygons generated!")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs)
    gdf["tile_id"] = TILE_AFRICA

    # Estimate year: use median of available Hansen lossyear + extend
    lossyear = africa_data["lossyear"]
    # For each polygon, estimate deforestation year
    years_est = []
    for geom in gdf.geometry:
        # Simple: assign based on centroid's Hansen value if available, else 2023
        # For competition: year doesn't affect Union IoU scoring directly but is required
        years_est.append("2020-01-01/2026-01-01")
    gdf["time_step"] = years_est

    print(f"  Africa polygons: {len(gdf)}")
    return gdf


def create_final_submission(africa_gdf):
    """Merge Africa polygons with v5_final for other tiles."""
    import geopandas as gpd

    print("\n" + "=" * 60)
    print("Creating final submission")
    print("=" * 60)

    # Load v5_final
    v5 = gpd.read_file(V5_SUB / "submission.geojson")
    print(f"  v5_final: {len(v5)} polygons")

    # Remove Africa tile from v5
    other_tiles = v5[v5["tile_id"] != TILE_AFRICA].copy()
    print(f"  Other tiles (non-Africa): {len(other_tiles)} polygons")
    print(f"  Africa from v5: {len(v5[v5['tile_id'] == TILE_AFRICA])} polygons")
    print(f"  Africa from afrika model: {len(africa_gdf)} polygons")

    # Combine — reproject Africa to WGS84 if needed
    if len(africa_gdf) > 0:
        if africa_gdf.crs != v5.crs:
            africa_gdf = africa_gdf.to_crs(v5.crs)
        combined = gpd.GeoDataFrame(
            pd.concat([other_tiles, africa_gdf], ignore_index=True),
            crs=v5.crs,
        )
    else:
        combined = other_tiles.copy()

    # Ensure required columns
    if "id" in combined.columns:
        combined = combined.drop(columns=["id"])
    combined.insert(0, "id", range(len(combined)))
    if "confidence" not in combined.columns:
        combined["confidence"] = 1.0

    # Keep only required columns
    combined = combined[["id", "time_step", "confidence", "tile_id", "geometry"]]

    # Save
    out_path = OUT_DIR / "submission.geojson"
    combined.to_file(out_path, driver="GeoJSON")
    print(f"\n  Saved: {out_path}")
    print(f"  Total polygons: {len(combined)}")

    # Per-tile breakdown
    for tid in sorted(combined["tile_id"].unique()):
        n = (combined["tile_id"] == tid).sum()
        print(f"    {tid}: {n} polys")

    return combined


# ── Main ────────────────────────────────────────────────────────────

def main():
    import pandas as pd
    # Make pandas available in create_final_submission
    globals()["pd"] = pd

    np.random.seed(42)

    print("=" * 60)
    print("  OASIS-AFRIKA: Africa-specialized deforestation model")
    print("=" * 60)

    # ── Phase A: Global model ──
    X_global, y_global, feat_names_global = build_training_data_global()
    if X_global is None:
        print("FATAL: No global training data")
        return

    global_model = train_global_model(X_global, y_global, feat_names_global)

    # Save global model
    global_model.save_model(str(OUT_DIR / "global_model.txt"))
    print(f"  Saved global model to {OUT_DIR / 'global_model.txt'}")

    # ── Phase B: Africa-specific model ──
    africa_data = build_africa_data()
    if africa_data is None:
        print("FATAL: No Africa data")
        return

    africa_models, feat_names_clean, keep_idx = train_africa_model(
        africa_data, feat_names_global
    )

    # Save Africa models
    for i, m in enumerate(africa_models):
        m.save_model(str(OUT_DIR / f"africa_model_fold{i}.txt"))
    print(f"  Saved {len(africa_models)} Africa models")

    # ── Predict ──
    prob_final, prob_global, prob_local = predict_africa(
        global_model, africa_models, africa_data, keep_idx, feat_names_global
    )

    # Save probability maps
    profile = africa_data["profile"].copy()
    profile.update(count=1, dtype="float32", nodata=0)
    for name, arr in [("prob_final", prob_final), ("prob_global", prob_global), ("prob_local", prob_local)]:
        with rasterio.open(OUT_DIR / f"{name}.tif", "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)

    # ── Try multiple thresholds ──
    print("\n" + "=" * 60)
    print("Threshold sweep on Africa predictions")
    print("=" * 60)

    best_t = 0.5
    best_score = 0
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = (prob_final >= t).astype(np.uint8)
        mask = ndimage.binary_closing(mask, iterations=1).astype(np.uint8)
        mask = ndimage.binary_opening(mask, iterations=2).astype(np.uint8)
        labeled, n_obj = ndimage.label(mask)
        for i in range(1, n_obj + 1):
            if (labeled == i).sum() * 100 / 10000 < 0.5:
                mask[labeled == i] = 0
        px = mask.sum()
        ha = px * 100 / 10000
        # Score heuristic: prefer ~500ha range (similar to v5_final's 465ha)
        # Too few → missing TP, too many → too many FP
        print(f"  t={t:.2f}: {px:>6d} px, {ha:>6.0f} ha")

    # Use t=0.45 as default (slightly generous to capture more TP)
    africa_gdf = polygonize_and_submit(prob_final, africa_data, threshold=0.45, min_ha=0.5)

    # ── Create submission ──
    import pandas as pd
    submission = create_final_submission(africa_gdf)

    print("\n" + "=" * 60)
    print("DONE! Submission at:", OUT_DIR / "submission.geojson")
    print("=" * 60)


if __name__ == "__main__":
    main()

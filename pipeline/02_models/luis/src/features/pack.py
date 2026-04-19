"""Combine all feature sources into a single per-tile matrix."""

import json
import warnings
import numpy as np
import rasterio

from src.config import DATA, CACHE, YEARS, MONTHS, N_FEATURES, TRAIN_TILES, feature_names
from src.features.s2 import extract_s2_features, build_forest_mask
from src.features.s1 import extract_s1_features
from src.features.aef import extract_aef_features
from src.labels.fusion import build_consensus_label

warnings.filterwarnings("ignore")


def get_reference_grid(tile, split):
    """Get shape/transform/crs from first available S2 file."""
    for year in YEARS:
        for month in MONTHS:
            p = DATA / f"sentinel-2/{split}/{tile}__s2_l2a/{tile}__s2_l2a_{year}_{month}.tif"
            if p.exists():
                with rasterio.open(p) as src:
                    return {
                        "shape": (src.height, src.width),
                        "transform": src.transform,
                        "crs": src.crs,
                    }
    return None


def build_tile(tile, split="train"):
    """
    Build feature matrix + labels for a single tile.
    Returns: (features (N,49), labels (N,) or None, metadata dict)
    """
    print(f"  Building features for {tile} ({split}) ...")

    ref = get_reference_grid(tile, split)
    if ref is None:
        print(f"    WARNING: No S2 data for {tile}, skipping.")
        return None, None, None

    ref_shape = ref["shape"]
    ref_transform = ref["transform"]
    ref_crs = ref["crs"]
    H, W = ref_shape
    N = H * W

    # 1. S2 features (30)
    print(f"    S2 features ({H}×{W}) ...")
    s2_feats = extract_s2_features(tile, split, ref_transform, ref_crs, ref_shape)

    # 2. S1 features (6)
    print(f"    S1 features ...")
    s1_feats = extract_s1_features(tile, split, ref_transform, ref_crs, ref_shape)

    # 3. AEF features (13)
    print(f"    AEF features ...")
    aef_feats = extract_aef_features(tile, split, ref_shape, ref_transform, ref_crs)

    # Stack
    all_features = s2_feats + s1_feats + aef_feats
    n_feat = len(all_features)
    assert n_feat == N_FEATURES, f"Expected {N_FEATURES} features, got {n_feat}"

    feature_matrix = np.stack(all_features, axis=0).reshape(n_feat, N).T
    np.nan_to_num(feature_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Labels (train only)
    label_map = None
    label_stats = {}
    if split == "train":
        print(f"    Building consensus labels ...")
        label_2d, label_stats = build_consensus_label(
            tile, ref_transform, ref_crs, ref_shape
        )
        forest_mask = build_forest_mask(tile, ref_shape, ref_transform, ref_crs, split)
        label_2d[~forest_mask] = 0

        label_binary = (label_2d >= 2).astype(np.int8)
        label_binary[label_2d == 1] = -1  # mask weak positives

        label_map = label_binary.reshape(N)

        n_pos = int((label_map == 1).sum())
        n_neg = int((label_map == 0).sum())
        n_unk = int((label_map == -1).sum())
        print(f"    Labels — pos: {n_pos} ({100*n_pos/N:.2f}%), "
              f"neg: {n_neg} ({100*n_neg/N:.2f}%), masked: {n_unk}")
        label_stats.update({"n_pos": n_pos, "n_neg": n_neg, "n_unk": n_unk, "N": N})

    meta = {
        "tile": tile, "split": split, "shape": [H, W],
        "n_features": n_feat, "N": N,
        **label_stats,
        "feature_names": feature_names(),
    }
    return feature_matrix, label_map, meta


def save_tile(tile, split="train"):
    """Build and save a tile's features/labels to cache."""
    cache_feat  = CACHE / f"{tile}_features.npz"
    cache_label = CACHE / f"{tile}_labels.npz"
    cache_meta  = CACHE / f"{tile}_meta.json"

    if cache_feat.exists() and (split != "train" or cache_label.exists()):
        print(f"  {tile}: already cached, skipping.")
        return True

    features, labels, meta = build_tile(tile, split)
    if features is None:
        return False

    np.savez_compressed(cache_feat, features=features)
    if labels is not None:
        np.savez_compressed(cache_label, labels=labels)
    with open(cache_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved → {cache_feat} ({features.shape})")
    return True

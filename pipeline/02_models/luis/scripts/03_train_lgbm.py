#!/usr/bin/env python3
"""
Step 2: Train LightGBM model.

Usage:
    python scripts/02_train_lgbm.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import lightgbm as lgb

from src.config import CACHE, MODELS, REGIONS, TRAIN_TILES, feature_names

# ── Sampling config ───────────────────────────────────────────────────────────
MAX_POS_PER_TILE = 3_000
MAX_NEG_PER_TILE = 15_000

# ── LightGBM params ──────────────────────────────────────────────────────────
PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.1,
    num_leaves=63,
    max_depth=7,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    scale_pos_weight=5,
    n_jobs=8,
    verbose=-1,
    seed=42,
)
NUM_BOOST_ROUND = 300


def sample_tile(tile, rng):
    """Load and subsample a single tile. Returns (X, y) or None."""
    fp = CACHE / f"{tile}_features.npz"
    lp = CACHE / f"{tile}_labels.npz"
    if not fp.exists() or not lp.exists():
        return None

    feats = np.load(fp)["features"]
    labels = np.load(lp)["labels"]

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0:
        return None

    n_pos = min(len(pos_idx), MAX_POS_PER_TILE)
    n_neg = min(len(neg_idx), MAX_NEG_PER_TILE)
    p_sel = rng.choice(pos_idx, n_pos, replace=False)
    n_sel = rng.choice(neg_idx, n_neg, replace=False)
    idx = np.concatenate([p_sel, n_sel])

    print(f"  {tile}: {n_pos} pos, {n_neg} neg")
    return feats[idx], labels[idx]


def train_model(tiles, name="full"):
    """Train a single LightGBM model on the given tiles."""
    rng = np.random.default_rng(42)
    Xs, ys = [], []

    for tile in tiles:
        result = sample_tile(tile, rng)
        if result is not None:
            Xs.append(result[0])
            ys.append(result[1])

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"\nTotal: {X.shape}  pos={int((y==1).sum())} ({100*(y==1).mean():.1f}%)")

    ds = lgb.Dataset(X, label=y, free_raw_data=True)
    model = lgb.train(PARAMS, ds, num_boost_round=NUM_BOOST_ROUND,
                      callbacks=[lgb.log_evaluation(50)])

    save_path = MODELS / f"lgbm_{name}.txt"
    model.save_model(str(save_path))
    print(f"Saved → {save_path}")

    # Feature importance
    imp = model.feature_importance("gain")
    names = feature_names()
    top = np.argsort(imp)[::-1][:15]
    print("\nTop 15 features by gain:")
    for r, i in enumerate(top):
        n = names[i] if i < len(names) else f"f{i}"
        print(f"  {r + 1:2d}. {n:<35s} {imp[i]:.0f}")

    return model


def main():
    # Train full model on all tiles
    print("=" * 60)
    print("Training full model")
    train_model(TRAIN_TILES, "full")

    # LORO: train one model per holdout region
    for region_name, region_tiles in REGIONS.items():
        print(f"\n{'=' * 60}")
        print(f"LORO: holding out {region_name} ({len(region_tiles)} tiles)")
        fit_tiles = [t for t in TRAIN_TILES if t not in region_tiles]
        train_model(fit_tiles, f"loro_{region_name}")

    print("\nDone!")


if __name__ == "__main__":
    main()

"""Load cached per-tile features + labels and turn into (X, y, meta) arrays.

Key design choices:
  * only pixels inside ``forest_2020`` are used (positives AND negatives).
  * training mode sub-samples negatives to a requested positive:negative ratio.
  * returns float32 arrays ready for LightGBM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path("/shared-docker/oasis-mark-2")
FEAT = ROOT / "cache/features"
LABEL = ROOT / "cache/labels"


@dataclass
class TileBundle:
    tile: str
    X: np.ndarray              # (N, F) float32
    y_hard: np.ndarray         # (N,) uint8
    y_soft: np.ndarray         # (N,) float32
    in_forest: np.ndarray      # (H, W) bool — the original mask
    pixel_index: np.ndarray    # (N,) int64 — linear indices into (H, W)
    feature_names: list[str]
    shape: tuple[int, int]
    sample_weight: np.ndarray | None = None  # (N,) float32 — per-row weight, optional


def load_tile(tile: str, sample_neg_ratio: float | None = None,
              rng: np.random.Generator | None = None,
              tile_uniform_weight: bool = False) -> TileBundle:
    """Load features + labels for a single tile.

    ``sample_neg_ratio``: when set (e.g. 10.0), negatives are sub-sampled to
    this multiplier of the positive count. Positives are always kept in full.
    This is appropriate for *training*; set to ``None`` for evaluation/inference.

    ``tile_uniform_weight``: when True, sets ``sample_weight`` so each tile
    contributes the same total weight (1.0) to training. This boosts the
    influence of low-positive-rate tiles whose total pixel count after
    sub-sampling is small. Use during training for fairer cross-tile balance.
    """
    rng = rng or np.random.default_rng(0)
    f = np.load(FEAT / f"{tile}.npz", allow_pickle=False)
    data = f["data"]                       # (F, H, W) float16
    names = [str(x) for x in f["feature_names"]]
    H, W = data.shape[1:]

    has_label = (LABEL / f"{tile}.npz").exists()
    if has_label:
        l = np.load(LABEL / f"{tile}.npz", allow_pickle=False)
        y_hard = l["y_hard"].astype(np.uint8)
        y_soft = l["y_soft"].astype(np.float32)
        forest = l["forest_2020"]
    else:
        y_hard = np.zeros((H, W), dtype=np.uint8)
        y_soft = np.zeros((H, W), dtype=np.float32)
        # no label file for test tiles — use Hansen-based forest mask derived from features
        tc_i = names.index("hansen_treecover2000")
        ly_i = names.index("hansen_lossyear_encoded")
        tc = data[tc_i].astype(np.float32)
        ly = data[ly_i].astype(np.float32)
        # "was forest in 2020" = tc ≥ 30 AND lossyear not in encoded 2 (our encoding for 2001-2020)
        forest = (tc >= 30.0) & (ly != 2)

    # flatten to pixel-wise
    y_hard_f = y_hard.ravel()
    y_soft_f = y_soft.ravel()
    forest_f = forest.ravel()
    keep = forest_f  # only use pixels inside forest_2020

    if sample_neg_ratio is not None:
        pos = keep & (y_hard_f > 0)
        neg = keep & (y_hard_f == 0)
        n_pos = int(pos.sum())
        if n_pos == 0:
            # nothing to train on this tile; keep a small neg sample anyway
            neg_idx = np.flatnonzero(neg)
            take = min(len(neg_idx), 10_000)
            sampled_neg = rng.choice(neg_idx, size=take, replace=False) if take else np.array([], dtype=np.int64)
            keep_mask = np.zeros_like(keep)
            keep_mask[sampled_neg] = True
        else:
            neg_target = int(n_pos * sample_neg_ratio)
            neg_idx = np.flatnonzero(neg)
            if len(neg_idx) > neg_target:
                neg_idx = rng.choice(neg_idx, size=neg_target, replace=False)
            keep_mask = np.zeros_like(keep)
            keep_mask[np.flatnonzero(pos)] = True
            keep_mask[neg_idx] = True
        keep = keep_mask

    idx = np.flatnonzero(keep)
    X = data[:, idx // W, idx % W].T.astype(np.float32)  # (N, F)

    sample_weight = None
    if tile_uniform_weight and len(idx) > 0:
        # Each tile contributes total weight 1.0 → per-row weight = 1/N.
        # Internally we re-scale to mean ~ 1.0 in concat_bundles so LGBM's
        # behaviour is comparable to unweighted but with per-tile balance.
        sample_weight = np.full(len(idx), 1.0 / len(idx), dtype=np.float32)

    return TileBundle(
        tile=tile,
        X=X,
        y_hard=y_hard_f[idx],
        y_soft=y_soft_f[idx],
        in_forest=forest,
        pixel_index=idx.astype(np.int64),
        feature_names=names,
        shape=(H, W),
        sample_weight=sample_weight,
    )


def concat_bundles(bundles: list[TileBundle]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], np.ndarray | None]:
    """Concatenate training bundles into (X, y_hard, y_soft, feature_names, tile_ids_per_row, sample_weight).

    ``sample_weight`` is None unless every bundle has one. When present it is
    re-normalised so the global mean weight is 1.0 (so LGBM gradients keep the
    same scale as unweighted training).
    """
    X = np.concatenate([b.X for b in bundles], axis=0)
    yh = np.concatenate([b.y_hard for b in bundles])
    ys = np.concatenate([b.y_soft for b in bundles])
    names = bundles[0].feature_names
    tile_ids = np.concatenate([np.full(len(b.y_hard), b.tile, dtype=object) for b in bundles])

    if all(b.sample_weight is not None for b in bundles):
        w = np.concatenate([b.sample_weight for b in bundles])
        # rescale so the average weight is 1.0
        w = w * (len(w) / max(w.sum(), 1e-12))
        sample_weight = w.astype(np.float32)
    else:
        sample_weight = None
    return X, yh, ys, names, tile_ids.tolist(), sample_weight

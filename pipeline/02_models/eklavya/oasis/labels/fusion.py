"""Weak-label fusion policy.

We have three weak alert sources after 2020:

* RADD (radd_{tile}_labels.tif) - encoded as conf_digit*10000 + days_since_2014_12_31
* GLAD-L (gladl_{tile}_alert{YY}.tif, gladl_{tile}_alertDate{YY}.tif)
* GLAD-S2 (glads2_{tile}_alert.tif, glads2_{tile}_alertDate.tif)

This module:

1. Loads each source as a binary post-2020 alert raster on a target grid.
2. Produces a soft fusion: ``labels``, ``valid_mask``, ``sample_weight``,
   used to train models with confidence-weighted samples.
3. Produces a *consensus* ground-truth subset for honest validation:
   - High-confidence positive: at least 2 sources agree on alert.
   - High-confidence negative: every available source says no alert AND
     no alert in a 1-pixel buffer (suppresses near-edge confusion).
   - Ignored: anything else.

The consensus subset is the only thing used for threshold tuning,
calibration, and reported metrics. Training samples can still be drawn
from the soft fusion to maximize sample volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from oasis import paths


# Days since 2014-12-31 to 2021-01-01.
RADD_DAY_2021 = 2193
# Days since 2019-01-01 to 2021-01-01.
GLADS2_DAY_2021 = 731


def _reproject_nearest(src_path: Path, dst_crs, dst_transform, dst_shape) -> np.ndarray:
    with rasterio.open(src_path) as src:
        data = src.read(1)
        out = np.zeros(dst_shape, dtype=data.dtype)
        reproject(
            source=data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return out


def load_radd_binary(
    tile_id: str, dst_crs, dst_transform, dst_shape
) -> tuple[np.ndarray, bool]:
    """Return (binary_alert_post_2020, present)."""
    path = paths.LABEL_DIR / "radd" / f"radd_{tile_id}_labels.tif"
    if not path.exists():
        return np.zeros(dst_shape, dtype=np.uint8), False
    arr = _reproject_nearest(path, dst_crs, dst_transform, dst_shape).astype(np.int32)
    has_alert = arr > 0
    # FIX: previous code used ``arr % 100000`` which kept the leading
    # confidence digit and effectively disabled the post-2020 gate.
    day_offset = np.where(has_alert, arr % 10000, 0)
    after_2020 = day_offset >= RADD_DAY_2021
    return (has_alert & after_2020).astype(np.uint8), True


def load_gladl_binary(
    tile_id: str, dst_crs, dst_transform, dst_shape
) -> tuple[np.ndarray, bool]:
    """OR over GLAD-L alert{YY} where YY in 21..25 and value >= 2."""
    binary = np.zeros(dst_shape, dtype=np.uint8)
    found = False
    for year in range(21, 26):
        alert_path = paths.LABEL_DIR / "gladl" / f"gladl_{tile_id}_alert{year}.tif"
        if alert_path.exists():
            arr = _reproject_nearest(alert_path, dst_crs, dst_transform, dst_shape)
            binary |= (arr >= 2).astype(np.uint8)
            found = True
    return binary, found


def load_glads2_binary(
    tile_id: str, dst_crs, dst_transform, dst_shape
) -> tuple[np.ndarray, bool]:
    alert_path = paths.LABEL_DIR / "glads2" / f"glads2_{tile_id}_alert.tif"
    date_path = paths.LABEL_DIR / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    if not alert_path.exists() or not date_path.exists():
        return np.zeros(dst_shape, dtype=np.uint8), False
    alert = _reproject_nearest(alert_path, dst_crs, dst_transform, dst_shape)
    date_arr = _reproject_nearest(date_path, dst_crs, dst_transform, dst_shape).astype(np.int32)
    has_alert = alert >= 2
    after_2020 = date_arr >= GLADS2_DAY_2021
    return (has_alert & after_2020).astype(np.uint8), True


@dataclass
class LabelStack:
    """Container for everything fusion produces for one tile."""

    sources: dict[str, np.ndarray]      # name -> binary alert raster
    available: dict[str, bool]
    labels: np.ndarray                  # uint8 fused training label
    valid_mask: np.ndarray              # bool: pixels eligible for training
    sample_weight: np.ndarray           # float32 in [0, 1]
    consensus_pos: np.ndarray           # bool: hi-conf positive (>=2 agree)
    consensus_neg: np.ndarray           # bool: hi-conf negative (all-zero + buffered)
    n_available: int


def _binary_dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """Dependency-light 4-connected dilation."""
    out = mask.copy()
    for _ in range(iters):
        shifted = np.zeros_like(out)
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = out | shifted
    return out


def fuse_labels(
    tile_id: str, dst_crs, dst_transform, dst_shape, verbose: bool = False
) -> LabelStack:
    """Build training labels + consensus ground truth for one tile."""
    radd, radd_ok = load_radd_binary(tile_id, dst_crs, dst_transform, dst_shape)
    gladl, gladl_ok = load_gladl_binary(tile_id, dst_crs, dst_transform, dst_shape)
    glads2, glads2_ok = load_glads2_binary(tile_id, dst_crs, dst_transform, dst_shape)

    sources = {"radd": radd, "gladl": gladl, "glads2": glads2}
    available = {"radd": radd_ok, "gladl": gladl_ok, "glads2": glads2_ok}

    active = [name for name, ok in available.items() if ok]
    n_available = len(active)
    h, w = dst_shape

    if n_available == 0:
        empty = np.zeros(dst_shape, dtype=np.uint8)
        return LabelStack(
            sources=sources,
            available=available,
            labels=empty,
            valid_mask=np.zeros(dst_shape, dtype=bool),
            sample_weight=np.zeros(dst_shape, dtype=np.float32),
            consensus_pos=np.zeros(dst_shape, dtype=bool),
            consensus_neg=np.zeros(dst_shape, dtype=bool),
            n_available=0,
        )

    stack = np.stack([sources[name] for name in active], axis=0)
    votes = stack.sum(axis=0)

    positive = votes > (n_available / 2.0)
    negative = votes == 0
    valid_mask = positive | negative

    labels = positive.astype(np.uint8)

    agreement = np.maximum(votes, n_available - votes).astype(np.float32) / float(n_available)
    coverage = np.float32(n_available / len(paths.LABEL_SOURCES))
    sample_weight = np.zeros(dst_shape, dtype=np.float32)
    sample_weight[valid_mask] = agreement[valid_mask] * coverage

    if n_available >= 2:
        consensus_pos = votes >= 2
    else:
        # With only one source we cannot define a hi-conf positive.
        consensus_pos = np.zeros(dst_shape, dtype=bool)

    any_alert = stack.sum(axis=0) > 0
    buffered_alert = _binary_dilate(any_alert, iters=1)
    consensus_neg = (~buffered_alert) & negative

    if verbose:
        print(
            f"  fuse[{tile_id}] sources={n_available} "
            f"pos={int(labels.sum())} hi_pos={int(consensus_pos.sum())} "
            f"hi_neg={int(consensus_neg.sum())} "
            f"valid={int(valid_mask.sum())}/{h * w}"
        )

    return LabelStack(
        sources=sources,
        available=available,
        labels=labels,
        valid_mask=valid_mask,
        sample_weight=sample_weight,
        consensus_pos=consensus_pos,
        consensus_neg=consensus_neg,
        n_available=n_available,
    )


def consensus_subset(stack: LabelStack) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, mask) restricted to consensus pixels.

    ``mask`` is True where ``labels`` is meaningful (hi-conf pos OR neg).
    """
    mask = stack.consensus_pos | stack.consensus_neg
    labels = stack.consensus_pos.astype(np.uint8)
    return labels, mask

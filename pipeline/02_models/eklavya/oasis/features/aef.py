"""AlphaEarth foundation-model embedding features.

AEF tiles ship as 64-channel float32 tiffs in EPSG:4326, one per year
2020..2025. We reproject every year onto the modeling grid (S2 UTM 10 m)
and stack into a (Y, 64, H, W) cube cached on disk.

Downstream consumers either:

* Use the full (Y * 64) flattened vector per pixel.
* Use yearly cosine distance / L2 change features.
* Use the 2025-2020 delta as a compact change vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


N_AEF_BANDS = 64


def parse_aef_filename(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def reproject_aef(
    src_path: Path,
    dst_crs,
    dst_transform,
    dst_shape,
    *,
    bands: list[int] | None = None,
    resampling=Resampling.bilinear,
) -> np.ndarray:
    """Reproject all (or selected) AEF bands onto the modeling grid."""
    with rasterio.open(src_path) as src:
        if bands is None:
            bands = list(range(1, src.count + 1))
        out = np.zeros((len(bands), *dst_shape), dtype=np.float32)
        for i, b in enumerate(bands):
            data = src.read(b).astype(np.float32)
            reproject(
                source=data,
                destination=out[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling,
            )
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel cosine distance between two (C, H, W) embedding stacks."""
    dot = (a * b).sum(axis=0)
    na = np.sqrt((a * a).sum(axis=0))
    nb = np.sqrt((b * b).sum(axis=0))
    denom = np.maximum(na * nb, 1e-6)
    cos = np.clip(dot / denom, -1.0, 1.0)
    return (1.0 - cos).astype(np.float32)


def l2_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    return np.sqrt((diff * diff).sum(axis=0)).astype(np.float32)

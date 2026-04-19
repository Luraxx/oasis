"""Reproject arbitrary rasters to the tile's canonical grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from .canonical_grid import CanonicalGrid


def reproject_to_canonical(
    src_path: Path,
    grid: CanonicalGrid,
    bands: list[int] | None = None,
    resampling: Resampling = Resampling.bilinear,
    dst_dtype=np.float32,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Read and reproject one or more bands to ``grid`` at 10 m UTM.

    Returns array of shape ``(n_bands, H, W)`` — the leading dim is dropped if
    ``bands`` is a single int.
    """
    with rasterio.open(src_path) as src:
        if bands is None:
            bands = list(range(1, src.count + 1))
        if isinstance(bands, int):
            bands = [bands]
        out = np.full((len(bands),) + grid.shape, fill_value, dtype=dst_dtype)
        for i, b in enumerate(bands):
            reproject(
                source=rasterio.band(src, b),
                destination=out[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=resampling,
            )
    return out

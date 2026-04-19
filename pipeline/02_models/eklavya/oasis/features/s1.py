"""Sentinel-1 feature extraction.

Provider files are RTC sigma0 in linear power (float32) at 30 m, single
band (VV). Filename pattern: ``{tile}__s1_rtc_{YYYY}_{M}_{ascending|descending}.tif``.

We:

* convert linear -> dB (10 * log10).
* apply a 5x5 mean Lee-style speckle filter (NaN-safe, dependency-free).
* keep ascending and descending stacks separate.
* reproject to the modeling grid (10 m UTM S2) with bilinear.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter


def parse_s1_filename(path: Path) -> tuple[int, int, str]:
    """Return (year, month, orbit) where orbit in {"ascending","descending"}."""
    parts = path.stem.split("_")
    rtc_idx = next(i for i, t in enumerate(parts) if t == "rtc")
    year = int(parts[rtc_idx + 1])
    month = int(parts[rtc_idx + 2])
    orbit = parts[rtc_idx + 3]
    return year, month, orbit


def linear_to_db(linear: np.ndarray) -> np.ndarray:
    out = np.where(linear > 0, 10.0 * np.log10(linear + 1e-10), np.nan)
    return out.astype(np.float32)


def lee_filter(arr: np.ndarray, win: int = 5) -> np.ndarray:
    """NaN-safe Lee speckle filter on an already-dB array.

    Local mean and variance are computed over a ``win x win`` window via
    ``scipy.ndimage.uniform_filter`` (preserves shape). The output blends
    the local mean and the pixel value weighted by ``var / (var + noise_var)``.
    """
    if arr.ndim != 2:
        raise ValueError("expected 2D array")

    arr = arr.astype(np.float32)
    valid = np.isfinite(arr)
    arr0 = np.where(valid, arr, 0.0)
    valid_f = valid.astype(np.float32)

    sum_v = uniform_filter(valid_f, size=win, mode="nearest")
    sum_a = uniform_filter(arr0, size=win, mode="nearest")
    sum_a2 = uniform_filter(arr0 * arr0, size=win, mode="nearest")

    counts = np.maximum(sum_v, 1e-3)
    means = sum_a / counts
    var = np.maximum(sum_a2 / counts - means * means, 0.0)

    if valid.any():
        noise_var = float(np.median(var[valid]))
    else:
        noise_var = 1.0
    noise_var = max(noise_var, 1e-3)

    weight = var / (var + noise_var)
    out = means + weight * (arr0 - means)
    out = np.where(valid, out, np.nan).astype(np.float32)
    return out


def reproject_s1_to_grid(
    src_path: Path, dst_crs, dst_transform, dst_shape, *, resampling=Resampling.bilinear
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        data = src.read(1).astype(np.float32)
        out = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=data,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return out

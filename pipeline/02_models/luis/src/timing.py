"""Estimate the year a deforestation event occurred via NBR max-drop.

For each polygon (connected component of the binary mask), we look at the
yearly NBR (Normalized Burn Ratio) median time series and find the year with
the largest negative drop relative to the previous year. The polygon's
`time_step` is set to that year.
"""
from collections import Counter
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import label as cc_label

from src.config import DATA, DATA_ADDITIONAL, B04, B08

YEARS_FOR_TIMING = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def _load_nbr_year(tile, year, split, ref_transform, ref_crs, ref_shape):
    """Load yearly median NBR for a tile, reprojected to ref grid. Returns (H,W) or None."""
    monthly = []
    for month in range(1, 13):
        p = DATA / f"sentinel-2/{split}/{tile}__s2_l2a/{tile}__s2_l2a_{year}_{month}.tif"
        if not p.exists():
            p = DATA_ADDITIONAL / f"sentinel-2/{tile}__s2_l2a/{tile}__s2_l2a_{year}_{month}.tif"
        if not p.exists():
            continue
        try:
            with rasterio.open(p) as src:
                bands = src.read().astype(np.float32)
                src_t, src_c = src.transform, src.crs
        except Exception:
            continue
        nir, swir2 = bands[B08], bands[11]  # B12 = index 11
        denom = nir + swir2
        denom[denom == 0] = 1e-6
        nbr = (nir - swir2) / denom
        nbr[bands[B08] == 0] = np.nan
        if nbr.shape != tuple(ref_shape):
            out = np.zeros(ref_shape, dtype=np.float32)
            reproject(source=nbr, destination=out,
                      src_transform=src_t, src_crs=src_c,
                      dst_transform=ref_transform, dst_crs=ref_crs,
                      resampling=Resampling.bilinear)
            nbr = out
        monthly.append(nbr)
    if not monthly:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(np.stack(monthly, axis=0), axis=0)


def estimate_polygon_years(binary: np.ndarray, tile: str, split: str,
                           ref_transform, ref_crs, ref_shape,
                           default_year: int = 2023) -> dict:
    """Return {component_id (1-based) → year} based on per-pixel max NBR drop.

    Algorithm:
    1. Compute yearly median NBR per pixel for years 2020..2026.
    2. For each pixel, compute year-over-year deltas; the year of the most
       negative delta (largest drop) is the candidate event year.
    3. For each connected component in `binary`, take the modal candidate year
       across its pixels.
    """
    H, W = ref_shape
    if binary.sum() == 0:
        return {}

    nbr_stack = []
    used_years = []
    for y in YEARS_FOR_TIMING:
        nbr_y = _load_nbr_year(tile, y, split, ref_transform, ref_crs, ref_shape)
        if nbr_y is not None:
            nbr_stack.append(nbr_y)
            used_years.append(y)

    if len(nbr_stack) < 2:
        # Fallback: assign default year to all components.
        labels, n = cc_label(binary.astype(bool))
        return {cid: default_year for cid in range(1, n + 1)}

    nbr_arr = np.stack(nbr_stack, axis=0)  # (T, H, W)
    # Year-over-year delta. delta[i] = nbr[i+1] - nbr[i]; large negative = drop.
    deltas = np.diff(nbr_arr, axis=0)  # (T-1, H, W)
    # Year ASSIGNED to a drop between years[i] and years[i+1] = years[i+1].
    drop_year_for_idx = used_years[1:]
    # For pixels where all NaN, skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Replace NaNs with +inf so they're never the min.
        deltas_filled = np.where(np.isnan(deltas), np.inf, deltas)
        drop_idx = np.argmin(deltas_filled, axis=0)  # (H, W) — which year had biggest drop

    labels, n = cc_label(binary.astype(bool))
    out = {}
    for cid in range(1, n + 1):
        mask = labels == cid
        years_in_comp = drop_idx[mask]
        if len(years_in_comp) == 0:
            out[cid] = default_year
            continue
        # Mode (most frequent drop year)
        c = Counter(years_in_comp.tolist())
        modal_idx, _ = c.most_common(1)[0]
        out[cid] = int(drop_year_for_idx[modal_idx])
    return out

"""Sentinel-2 spectral index computation and temporal feature extraction."""

import warnings
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from src.config import (
    DATA, DATA_ADDITIONAL, USE_2026,
    YEARS, YEARS_ADDITIONAL, YEARS_ALL, MONTHS, MONTHS_ADDITIONAL,
    BASELINE_YEAR, B02, B03, B04, B08, B11, B12,
)


def compute_indices(bands: np.ndarray) -> np.ndarray:
    """Compute 6 spectral indices from S2 12-band array. Returns (6, H, W)."""
    def safe_ratio(a, b):
        d = a + b
        d = np.where(d == 0, 1e-6, d)
        return (a - b) / d

    nir   = bands[B08].astype(np.float32)
    red   = bands[B04].astype(np.float32)
    grn   = bands[B03].astype(np.float32)
    blu   = bands[B02].astype(np.float32)
    swir1 = bands[B11].astype(np.float32)
    swir2 = bands[B12].astype(np.float32)

    ndvi = safe_ratio(nir, red)
    nbr  = safe_ratio(nir, swir2)
    ndmi = safe_ratio(nir, swir1)
    ndwi = safe_ratio(grn, nir)
    bsi  = safe_ratio(swir1 + red, nir + blu)
    evi  = 2.5 * (nir - red) / (nir + 6*red - 7.5*blu + 1 + 1e-6)
    evi  = np.clip(evi, -2, 2)

    return np.stack([ndvi, nbr, ndmi, ndwi, bsi, evi], axis=0)


def load_indices(tile, year, month, split="train",
                 dst_transform=None, dst_crs=None, dst_shape=None):
    """Load S2 file, compute indices, optionally reproject. Returns (6,H,W) or None."""
    # Check Makeathon data first, then additional data
    p = DATA / f"sentinel-2/{split}/{tile}__s2_l2a/{tile}__s2_l2a_{year}_{month}.tif"
    if not p.exists():
        p = DATA_ADDITIONAL / f"sentinel-2/{tile}__s2_l2a/{tile}__s2_l2a_{year}_{month}.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        bands = src.read().astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs

    invalid = (bands[B08] == 0) | (bands[B04] == 0)
    indices = compute_indices(bands)
    indices[:, invalid] = np.nan

    if dst_shape is not None and indices.shape[1:] != tuple(dst_shape):
        n_idx = indices.shape[0]
        out = np.zeros((n_idx,) + tuple(dst_shape), dtype=np.float32)
        for c in range(n_idx):
            reproject(
                source=indices[c], destination=out[c],
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        indices = out

    return indices


def extract_s2_features(tile, split, ref_transform, ref_crs, ref_shape):
    """
    Extract 42 S2 features (6 indices × 7 temporal stats).
    Extract S2 features per index.
    Baseline: 30 features (6 indices × 5 stats)
    With-2026: 42 features (6 indices × 7 stats)
    """
    # Load all available months
    s2_ts = {}
    for year in YEARS:
        for month in MONTHS:
            idx = load_indices(tile, year, month, split,
                               dst_transform=ref_transform, dst_crs=ref_crs,
                               dst_shape=ref_shape)
            s2_ts[(year, month)] = idx
    if USE_2026:
        for year in YEARS_ADDITIONAL:
            for month in MONTHS_ADDITIONAL:
                idx = load_indices(tile, year, month, split,
                                   dst_transform=ref_transform, dst_crs=ref_crs,
                                   dst_shape=ref_shape)
                s2_ts[(year, month)] = idx

    features = []
    n_idx = 6

    for i in range(n_idx):
        # 2020 baseline
        stack_2020 = [s2_ts[(BASELINE_YEAR, m)][i]
                      for m in MONTHS if s2_ts.get((BASELINE_YEAR, m)) is not None]
        if stack_2020:
            arr_2020 = np.stack(stack_2020, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_2020 = np.nanmean(arr_2020, axis=0)
                std_2020  = np.nanstd(arr_2020, axis=0)
        else:
            mean_2020 = np.zeros(ref_shape, dtype=np.float32)
            std_2020  = np.zeros(ref_shape, dtype=np.float32)

        # Post-2020 stats
        stack_post = []
        post_end = 2027 if USE_2026 else 2026
        for year in range(2021, post_end):
            months_range = MONTHS_ADDITIONAL if year >= 2026 else MONTHS
            for m in months_range:
                d = s2_ts.get((year, m))
                if d is not None:
                    stack_post.append(d[i])

        if stack_post:
            arr_post = np.stack(stack_post, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                min_post = np.nanmin(arr_post, axis=0)
            max_drop = mean_2020 - min_post
        else:
            min_post = mean_2020.copy()
            max_drop = np.zeros(ref_shape, dtype=np.float32)

        # Trend slope (linear fit over annual medians)
        annual_medians = []
        years_for_slope = YEARS_ALL if USE_2026 else YEARS
        for year in years_for_slope:
            months_range = MONTHS_ADDITIONAL if year >= 2026 else MONTHS
            yr_vals = [s2_ts[(year, m)][i]
                       for m in months_range if s2_ts.get((year, m)) is not None]
            if yr_vals:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    annual_medians.append(np.nanmedian(np.stack(yr_vals, axis=0), axis=0))

        if len(annual_medians) >= 2:
            T = len(annual_medians)
            t_vals = np.arange(T, dtype=np.float32)
            stack_ann = np.stack(annual_medians, axis=0)
            t_mean = t_vals.mean()
            stack_norm = stack_ann - np.nanmean(stack_ann, axis=0, keepdims=True)
            t_norm = t_vals - t_mean
            slope = np.nansum(t_norm[:, None, None] * stack_norm, axis=0) / (np.sum(t_norm**2) + 1e-9)
        else:
            slope = np.zeros(ref_shape, dtype=np.float32)

        for arr in [mean_2020, std_2020, min_post, max_drop, slope]:
            np.nan_to_num(arr, copy=False, nan=0.0)

        features += [mean_2020, std_2020, min_post, max_drop, slope]

        # 2026 recent features (only in with2026 mode)
        if USE_2026:
            stack_2026 = [s2_ts[(2026, m)][i]
                          for m in MONTHS_ADDITIONAL if s2_ts.get((2026, m)) is not None]
            if stack_2026:
                arr_2026 = np.stack(stack_2026, axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean_2026 = np.nanmean(arr_2026, axis=0)
                drop_2026 = mean_2020 - mean_2026
            else:
                mean_2026 = np.zeros(ref_shape, dtype=np.float32)
                drop_2026 = np.zeros(ref_shape, dtype=np.float32)

            for arr in [mean_2026, drop_2026]:
                np.nan_to_num(arr, copy=False, nan=0.0)

            features += [mean_2026, drop_2026]

    return features


def build_forest_mask(tile, ref_shape, ref_transform, ref_crs, split="train",
                      ndvi_threshold=0.4):
    """Pixels where median NDVI in 2020 > threshold → forest."""
    ndvi_stack = []
    for month in MONTHS:
        p = DATA / f"sentinel-2/{split}/{tile}__s2_l2a/{tile}__s2_l2a_2020_{month}.tif"
        if not p.exists():
            continue
        with rasterio.open(p) as src:
            bands = src.read().astype(np.float32)
            src_transform = src.transform
            src_crs_file = src.crs
        nir = bands[B08]; red = bands[B04]
        denom = nir + red; denom[denom == 0] = 1e-6
        nv = (nir - red) / denom
        nv[bands[B08] == 0] = np.nan
        if nv.shape != tuple(ref_shape):
            out = np.zeros(ref_shape, dtype=np.float32)
            reproject(source=nv, destination=out,
                      src_transform=src_transform, src_crs=src_crs_file,
                      dst_transform=ref_transform, dst_crs=ref_crs,
                      resampling=Resampling.bilinear)
            nv = out
        ndvi_stack.append(nv)

    if not ndvi_stack:
        return np.ones(ref_shape, dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ndvi_median = np.nanmedian(np.stack(ndvi_stack, axis=0), axis=0)
    return (ndvi_median > ndvi_threshold) & np.isfinite(ndvi_median)

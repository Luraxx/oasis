"""Sentinel-1 radar feature extraction."""

import warnings
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from src.config import DATA, DATA_ADDITIONAL, USE_2026, YEARS, YEARS_ADDITIONAL, MONTHS, MONTHS_ADDITIONAL, BASELINE_YEAR, S1_ORBITS


def load_s1_db(tile, year, month, orbit="ascending", split="train",
               dst_transform=None, dst_crs=None, dst_shape=None):
    """Load S1 VV backscatter, convert to dB, reproject to S2 grid."""
    # Check Makeathon data first, then additional data
    p = DATA / f"sentinel-1/{split}/{tile}__s1_rtc/{tile}__s1_rtc_{year}_{month}_{orbit}.tif"
    if not p.exists():
        p = DATA_ADDITIONAL / f"sentinel-1/{tile}__s1_rtc/{tile}__s1_rtc_{year}_{month}_{orbit}.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as src:
        if dst_transform is not None and dst_shape is not None:
            out = np.zeros(dst_shape, dtype=np.float32)
            reproject(
                source=src.read(1).astype(np.float32), destination=out,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            data = out
        else:
            data = src.read(1).astype(np.float32)
    db = np.where(data > 0, 10.0 * np.log10(data + 1e-9), np.nan)
    return db


def extract_s1_features(tile, split, ref_transform, ref_crs, ref_shape):
    """
    Extract 8 S1 features (2 orbits × 4 stats: mean20, std20, maxdrop, mean26).
    Returns list of 8 (H,W) arrays.
    """
    features = []

    for orbit in S1_ORBITS:
        s1_ts = {}
        for year in YEARS:
            for month in MONTHS:
                d = load_s1_db(tile, year, month, orbit, split,
                               dst_transform=ref_transform, dst_crs=ref_crs,
                               dst_shape=ref_shape)
                s1_ts[(year, month)] = d
        if USE_2026:
            for year in YEARS_ADDITIONAL:
                for month in MONTHS_ADDITIONAL:
                    d = load_s1_db(tile, year, month, orbit, split,
                                   dst_transform=ref_transform, dst_crs=ref_crs,
                                   dst_shape=ref_shape)
                    s1_ts[(year, month)] = d

        # 2020 baseline
        stack_2020 = [s1_ts[(BASELINE_YEAR, m)]
                      for m in MONTHS if s1_ts.get((BASELINE_YEAR, m)) is not None]
        if stack_2020:
            arr_2020 = np.stack(stack_2020, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_2020 = np.nanmean(arr_2020, axis=0)
                std_2020  = np.nanstd(arr_2020, axis=0)
        else:
            mean_2020 = np.zeros(ref_shape, dtype=np.float32)
            std_2020  = np.zeros(ref_shape, dtype=np.float32)

        # Post-2020 max drop
        stack_post = []
        post_end = 2027 if USE_2026 else 2026
        for y in range(2021, post_end):
            months_range = MONTHS_ADDITIONAL if y >= 2026 else MONTHS
            for m in months_range:
                v = s1_ts.get((y, m))
                if v is not None:
                    stack_post.append(v)
        if stack_post:
            arr_post = np.stack(stack_post, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                min_post = np.nanmin(arr_post, axis=0)
            max_drop = mean_2020 - min_post
        else:
            max_drop = np.zeros(ref_shape, dtype=np.float32)

        for arr in [mean_2020, std_2020, max_drop]:
            np.nan_to_num(arr, copy=False, nan=0.0)

        features += [mean_2020, std_2020, max_drop]

        # 2026 mean (only in with2026 mode)
        if USE_2026:
            stack_2026 = [s1_ts.get((2026, m))
                          for m in MONTHS_ADDITIONAL if s1_ts.get((2026, m)) is not None]
            if stack_2026:
                arr_2026 = np.stack(stack_2026, axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mean_2026 = np.nanmean(arr_2026, axis=0)
            else:
                mean_2026 = np.zeros(ref_shape, dtype=np.float32)
            np.nan_to_num(mean_2026, copy=False, nan=0.0)
            features.append(mean_2026)

    return features

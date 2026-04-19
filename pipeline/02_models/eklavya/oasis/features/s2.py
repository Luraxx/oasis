"""Sentinel-2 feature extraction.

Band order in provider stack (1-indexed):
    1=B01 2=B02(B) 3=B03(G) 4=B04(R) 5=B05 6=B06 7=B07 8=B08(NIR)
    9=B8A 10=B09 11=B11(SWIR1) 12=B12(SWIR2)

S2 L2A here ships without an SCL band, so we derive a pragmatic cloud
and shadow mask from spectral thresholds (Hollstein-style, simplified).
The mask is applied before any temporal statistic so that monthly NDVI
means are not dragged by clouds.

All reflectance arithmetic uses raw uint16 / 10000 scale (provider
convention). Functions return float32 arrays in physical NDVI/NBR/...
units in [-1, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


SCALE = 10000.0  # provider reflectance scale

# Band index aliases (1-based, matches rasterio.read(idx)).
B02, B03, B04, B05, B08, B8A, B09, B11, B12 = 2, 3, 4, 5, 8, 9, 10, 11, 12

INDEX_NAMES = ("ndvi", "nbr", "ndmi", "evi", "mndwi", "bsi", "savi", "ndii")


def s2_cloud_mask(stack: np.ndarray) -> np.ndarray:
    """Derive a boolean ``valid`` mask (True = clear, usable).

    Args:
        stack: float32 array of shape (12, H, W) in raw provider units
            (uint16 cast to float32, NOT divided by SCALE).
    """
    b02 = stack[B02 - 1]
    b03 = stack[B03 - 1]
    b04 = stack[B04 - 1]
    b08 = stack[B08 - 1]
    b11 = stack[B11 - 1]
    b12 = stack[B12 - 1]

    saturated = (
        (b02 == 0) | (b03 == 0) | (b04 == 0) | (b08 == 0) | (b11 == 0) | (b12 == 0)
    )

    # Bright targets (clouds + ice + concrete look similar in optical only;
    # combine with SWIR to discriminate clouds).
    bright = b02 > 2200
    high_swir = b11 > 3500
    cloud = bright & high_swir

    # Thin clouds / haze: high blue with elevated B12.
    thin_cloud = (b02 > 1700) & (b12 > 1500) & ((b04 / np.maximum(b03, 1)) < 1.3)

    # Cloud shadows: low NIR, low VIS, but on land (NDVI ~ 0..0.3).
    ndvi_denom = b08 + b04
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(ndvi_denom > 0, (b08 - b04) / np.where(ndvi_denom > 0, ndvi_denom, 1.0), 0)
    shadow = (b08 < 800) & (b11 < 800) & (ndvi < 0.3) & (~saturated)

    # Snow / ice (NDSI heuristic).
    ndsi_denom = b03 + b11
    with np.errstate(invalid="ignore", divide="ignore"):
        ndsi = np.where(ndsi_denom > 0, (b03 - b11) / np.where(ndsi_denom > 0, ndsi_denom, 1.0), 0)
    snow = (ndsi > 0.4) & (b03 > 1500)

    invalid = saturated | cloud | thin_cloud | shadow | snow
    return ~invalid


def compute_indices(stack: np.ndarray, valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Return dict of float32 spectral indices on the same H,W grid.

    Pixels marked invalid (by `valid` mask) are set to NaN so downstream
    nan-aware reductions skip them.
    """
    s = stack.astype(np.float32) / SCALE
    b02 = s[B02 - 1]
    b03 = s[B03 - 1]
    b04 = s[B04 - 1]
    b08 = s[B08 - 1]
    b11 = s[B11 - 1]
    b12 = s[B12 - 1]

    eps = 1e-6

    def safe(numer, denom):
        return np.where(np.abs(denom) > eps, numer / denom, 0.0).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.clip(safe(b08 - b04, b08 + b04), -1.0, 1.0)
        nbr = np.clip(safe(b08 - b12, b08 + b12), -1.0, 1.0)
        ndmi = np.clip(safe(b08 - b11, b08 + b11), -1.0, 1.0)
        evi = np.clip(safe(2.5 * (b08 - b04), b08 + 6 * b04 - 7.5 * b02 + 1.0), -1.5, 2.5)
        mndwi = np.clip(safe(b03 - b11, b03 + b11), -1.0, 1.0)
        bsi = np.clip(safe((b11 + b04) - (b08 + b02), (b11 + b04) + (b08 + b02)), -1.0, 1.0)
        savi = np.clip(safe(1.5 * (b08 - b04), b08 + b04 + 0.5), -1.5, 1.5)
        ndii = ndmi  # alias of ndmi but we keep both for stack symmetry

    out = {
        "ndvi": ndvi,
        "nbr": nbr,
        "ndmi": ndmi,
        "evi": evi,
        "mndwi": mndwi,
        "bsi": bsi,
        "savi": savi,
        "ndii": ndii,
    }
    if valid is not None:
        for k in out:
            out[k] = np.where(valid, out[k], np.nan)
    return out


def parse_s2_filename(path: Path) -> tuple[int, int]:
    """Extract (year, month) from ``..._YYYY_M.tif`` filename."""
    stem = path.stem
    parts = stem.split("_")
    year = int(parts[-2])
    month = int(parts[-1])
    return year, month


def read_s2_stack(path: Path) -> np.ndarray:
    """Read a 12-band S2 file as float32 (H, W, 12) -> returns (12, H, W)."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

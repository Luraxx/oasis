"""Postprocessing: morphological cleanup + connected-component area filter."""

import numpy as np
from scipy.ndimage import binary_opening, binary_closing, label as cc_label, sum_labels


def clean_binary(binary: np.ndarray, opening: int = 1, closing: int = 1) -> np.ndarray:
    """Apply morphological opening then closing on a binary mask.

    - Opening removes salt-and-pepper noise (lowers FP).
    - Closing bridges 1-pixel gaps (raises Recall).
    """
    out = binary.astype(bool)
    if opening > 0:
        out = binary_opening(out, iterations=opening)
    if closing > 0:
        out = binary_closing(out, iterations=closing)
    return out.astype(np.uint8)


def filter_components_by_area(binary: np.ndarray, transform, min_area_ha: float = 0.5) -> np.ndarray:
    """Drop connected components smaller than `min_area_ha` (hectares).

    Pixel area is derived from the affine transform (assumes a projected CRS in metres).
    """
    if binary.sum() == 0:
        return binary
    pixel_area_m2 = abs(transform.a * transform.e)  # |sx * sy|
    if pixel_area_m2 <= 0 or pixel_area_m2 > 1e6:
        # Geographic CRS or something odd — skip area filter (raster_to_geojson will catch it).
        return binary
    labels, n = cc_label(binary.astype(bool))
    if n == 0:
        return binary
    sizes = sum_labels(np.ones_like(labels, dtype=np.float32), labels, index=np.arange(1, n + 1))
    areas_ha = sizes * pixel_area_m2 / 10_000.0
    keep_ids = np.where(areas_ha >= min_area_ha)[0] + 1
    if len(keep_ids) == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    keep_mask = np.isin(labels, keep_ids)
    return keep_mask.astype(np.uint8)


def postprocess(binary: np.ndarray, transform=None, opening: int = 1, closing: int = 1,
                min_area_ha: float = 0.5) -> np.ndarray:
    """Full chain: morphology then area filter (if transform given)."""
    out = clean_binary(binary, opening=opening, closing=closing)
    if transform is not None:
        out = filter_components_by_area(out, transform, min_area_ha=min_area_ha)
    return out

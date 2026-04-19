"""Consensus label fusion from RADD, GLAD-S2, GLAD-L alert datasets."""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from src.config import DATA


def _reproj(path, ref_shape, ref_transform, ref_crs, dtype=np.uint16):
    """Reproject a label raster to the reference grid."""
    if not Path(path).exists():
        return None
    with rasterio.open(path) as src:
        out = np.zeros(ref_shape, dtype=np.float32)
        reproject(
            source=src.read(1).astype(np.float32),
            destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )
    return out.astype(dtype)


def build_consensus_label(tile, ref_transform, ref_crs, ref_shape):
    """
    Build consensus labels from 3 alert datasets.

    Returns:
        label_2d (H,W) int8: 0=negative, 1=weak (1 source), 2=strong (>=2 sources)
        stats dict
    """
    # --- RADD ---
    radd_path = DATA / f"labels/train/radd/radd_{tile}_labels.tif"
    radd_raw = _reproj(radd_path, ref_shape, ref_transform, ref_crs, np.uint16)
    if radd_raw is not None:
        days = (radd_raw.astype(np.int32) % 10000)
        radd_bin = (days > 2192).astype(np.uint8)  # post-2020, any confidence
    else:
        radd_bin = np.zeros(ref_shape, dtype=np.uint8)

    # --- GLAD-S2 ---
    gs2_alert_path = DATA / f"labels/train/glads2/glads2_{tile}_alert.tif"
    gs2_date_path  = DATA / f"labels/train/glads2/glads2_{tile}_alertDate.tif"
    gs2_conf = _reproj(gs2_alert_path, ref_shape, ref_transform, ref_crs, np.uint8)
    gs2_date = _reproj(gs2_date_path, ref_shape, ref_transform, ref_crs, np.uint16)
    if gs2_conf is not None and gs2_date is not None:
        gs2_bin = ((gs2_date.astype(np.int32) > 730) & (gs2_conf > 0)).astype(np.uint8)
    else:
        gs2_bin = np.zeros(ref_shape, dtype=np.uint8)

    # --- GLAD-L ---
    gladl_bin = np.zeros(ref_shape, dtype=np.uint8)
    for yy in ["21", "22", "23", "24", "25"]:
        gl_path = DATA / f"labels/train/gladl/gladl_{tile}_alert{yy}.tif"
        gl = _reproj(gl_path, ref_shape, ref_transform, ref_crs, np.uint8)
        if gl is not None:
            gladl_bin = np.maximum(gladl_bin, (gl >= 2).astype(np.uint8))

    # --- Consensus fusion ---
    agreement = radd_bin.astype(np.int16) + gs2_bin.astype(np.int16) + gladl_bin.astype(np.int16)
    strong_pos = (agreement >= 2).astype(np.uint8)
    weak_pos   = (agreement == 1).astype(np.uint8)

    label = np.zeros(ref_shape, dtype=np.int8)
    label[weak_pos == 1]   = 1
    label[strong_pos == 1] = 2

    stats = {
        "radd_pct":   float(100 * radd_bin.mean()),
        "gs2_pct":    float(100 * gs2_bin.mean()),
        "gladl_pct":  float(100 * gladl_bin.mean()),
        "strong_pct": float(100 * strong_pos.mean()),
        "weak_pct":   float(100 * weak_pos.mean()),
    }
    return label, stats

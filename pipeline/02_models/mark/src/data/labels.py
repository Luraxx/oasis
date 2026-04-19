"""Per-tile label fusion on the canonical UTM 10 m grid.

Produces one compressed NPZ file per tile under ``cache/labels/{tile}.npz`` with:

  - ``y_any``        : ``(H, W)`` bool — 1 if any available source reports
                       post-2020 (≥ 2021) deforestation at ≥ low confidence.
  - ``y_hard``       : ``(H, W)`` bool — 1 if ≥ 2 sources agree (or the single
                       source is high confidence when only one is available).
  - ``y_soft``       : ``(H, W)`` float16 — weighted confidence in [0, 1].
  - ``time_step``    : ``(H, W)`` int16 — year of earliest post-2020 alert
                       (0 = no alert). Kept as optional "when" information.
  - ``forest_2020``  : ``(H, W)`` bool — was forest in 2020 gate
                       (Hansen treecover2000 ≥ 30 % AND lossyear ∉ {1..20}).
  - ``sources_avail``: ``(H, W)`` uint8 — bit flags of which sources had any
                       data on that pixel (bit0=RADD, bit1=GLAD-L, bit2=GLAD-S2,
                       bit3=Hansen).

Post-2020 rule:
  * alerts with date ≤ 2020-12-31 are excluded.
  * alerts with date > 2025-12-31 are clipped (no imagery to support them).

Source confidence weights (for y_soft):
  * RADD conf=2 (low) → 0.5,  conf=3 (high) → 1.0
  * GLAD-L conf=2 → 0.5, conf=3 → 1.0
  * GLAD-S2 conf=1 → 0.25, 2 → 0.5, 3 → 0.75, 4 → 1.0
  * Hansen post-2020 loss → 0.75 (reasonably reliable but annual-resolution)
y_soft = sum(weight_i) / number_of_sources_available.

Hard-label rule:
  * 2+ of available sources contribute at ≥ 0.5 weight.
  * if only 1 source available, that source must be ≥ 1.0 (i.e. high conf).

time_step:
  * Year of the earliest post-2020 alert in any source (2021..2025).
  * 0 if no post-2020 alert.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from .canonical_grid import CanonicalGrid, grid_for
from .reproject import reproject_to_canonical

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path("/shared-docker/oasis-mark-2")
LBL = ROOT / "data/makeathon-challenge/labels/train"
LBL_EXTRA = ROOT / "external/makeathon-extras/labels/train"  # writable overlay
HANSEN = ROOT / "external/hansen/cropped"
CACHE = ROOT / "cache/labels"
CACHE.mkdir(parents=True, exist_ok=True)


def _first_label(subpath: str) -> Path | None:
    for root in (LBL, LBL_EXTRA):
        p = root / subpath
        if p.exists():
            return p
    return None

RADD_EPOCH = date(2014, 12, 31)
GLADS2_EPOCH = date(2019, 1, 1)

POST_CUTOFF_LOW = (date(2020, 12, 31) - RADD_EPOCH).days  # > this day means year ≥ 2021
POST_CUTOFF_HIGH = (date(2025, 12, 31) - RADD_EPOCH).days  # clip upper end

GLADS2_LOW = (date(2020, 12, 31) - GLADS2_EPOCH).days
GLADS2_HIGH = (date(2025, 12, 31) - GLADS2_EPOCH).days


def _radd_sources(tile: str, grid: CanonicalGrid) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return (conf_soft_weight, year_of_earliest, availability_mask).

    conf_soft_weight in {0, 0.5, 1.0} per pixel.
    year_of_earliest is 0 if no post-2020 alert.
    availability_mask is 1 where the raster has a valid pixel (currently any pixel)."""
    p = _first_label(f"radd/radd_{tile}_labels.tif")
    if p is None:
        return None, None, None
    raw = reproject_to_canonical(p, grid, resampling=Resampling.nearest,
                                 dst_dtype=np.float32, fill_value=0.0)[0].astype(np.int64)
    conf_code = np.where(raw > 0, raw // 10000, 0)
    days = np.where(raw > 0, raw % 10000, 0)
    post = (conf_code > 0) & (days > POST_CUTOFF_LOW) & (days <= POST_CUTOFF_HIGH)
    weight = np.zeros(grid.shape, dtype=np.float32)
    weight[post & (conf_code == 2)] = 0.5
    weight[post & (conf_code == 3)] = 1.0
    year = np.zeros(grid.shape, dtype=np.int16)
    year[post] = np.array([(RADD_EPOCH + timedelta(days=int(d))).year for d in days[post]], dtype=np.int16)
    avail = np.ones(grid.shape, dtype=bool)  # RADD covers the tile fully
    return weight, year, avail


def _gladl_sources(tile: str, grid: CanonicalGrid) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """GLAD-L: yearly alertYY + alertDateYY. Only years 2021-2025 exist."""
    weight = np.zeros(grid.shape, dtype=np.float32)
    year_out = np.zeros(grid.shape, dtype=np.int16)
    any_found = False
    for yy in [21, 22, 23, 24, 25]:
        year = 2000 + yy
        p = _first_label(f"gladl/gladl_{tile}_alert{yy:02d}.tif")
        if p is None:
            continue
        any_found = True
        conf = reproject_to_canonical(p, grid, resampling=Resampling.nearest,
                                      dst_dtype=np.float32, fill_value=0.0)[0].astype(np.int16)
        # code 2 → 0.5, code 3 → 1.0
        hit2 = conf == 2
        hit3 = conf == 3
        # take MAX weight across years (a pixel flagged once stays flagged)
        new_weight = np.zeros_like(weight)
        new_weight[hit2] = 0.5
        new_weight[hit3] = 1.0
        stronger = new_weight > weight
        weight = np.where(stronger, new_weight, weight)
        # earliest year
        alerted = (hit2 | hit3)
        need_year = alerted & (year_out == 0)
        year_out[need_year] = year
    if not any_found:
        return None, None, None
    avail = np.ones(grid.shape, dtype=bool)
    return weight, year_out, avail


def _glads2_sources(tile: str, grid: CanonicalGrid) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """GLAD-S2: only available for 8 Colombia tiles."""
    p_alert = _first_label(f"glads2/glads2_{tile}_alert.tif")
    p_date = _first_label(f"glads2/glads2_{tile}_alertDate.tif")
    if not (p_alert is not None and p_date is not None):
        return None, None, None
    conf = reproject_to_canonical(p_alert, grid, resampling=Resampling.nearest,
                                  dst_dtype=np.float32, fill_value=0.0)[0].astype(np.int16)
    days = reproject_to_canonical(p_date, grid, resampling=Resampling.nearest,
                                  dst_dtype=np.float32, fill_value=0.0)[0].astype(np.int32)
    post = (conf > 0) & (days > GLADS2_LOW) & (days <= GLADS2_HIGH)
    weight = np.zeros(grid.shape, dtype=np.float32)
    for code, w in [(1, 0.25), (2, 0.5), (3, 0.75), (4, 1.0)]:
        weight[post & (conf == code)] = w
    year = np.zeros(grid.shape, dtype=np.int16)
    # Convert days to year by mapping via epoch
    if post.any():
        dd = days[post]
        yrs = np.array([(GLADS2_EPOCH + timedelta(days=int(d))).year for d in dd], dtype=np.int16)
        year[post] = yrs
    avail = np.ones(grid.shape, dtype=bool)
    return weight, year, avail


def _hansen_sources(tile: str, grid: CanonicalGrid) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Hansen GFC lossyear 2021-2023 → post-2020 loss signal."""
    p = HANSEN / f"{tile}_lossyear.tif"
    if not p.exists():
        return None, None, None
    ly = reproject_to_canonical(p, grid, resampling=Resampling.nearest,
                                dst_dtype=np.float32, fill_value=0.0)[0].astype(np.int16)
    post = (ly >= 21) & (ly <= 23)  # 2021, 2022, 2023
    weight = np.where(post, 0.75, 0.0).astype(np.float32)
    year = np.where(post, 2000 + ly, 0).astype(np.int16)
    # hansen does not cover 2024-2025 → treat as not-available for those years;
    # but availability within the tile is always true (data covers land pixels)
    avail = np.ones(grid.shape, dtype=bool)
    return weight, year, avail


def _forest_2020(tile: str, grid: CanonicalGrid) -> np.ndarray:
    """Forest-in-2020 boolean gate."""
    tc_path = HANSEN / f"{tile}_treecover2000.tif"
    ly_path = HANSEN / f"{tile}_lossyear.tif"
    tc = reproject_to_canonical(tc_path, grid, resampling=Resampling.bilinear,
                                dst_dtype=np.float32, fill_value=0.0)[0]
    ly = reproject_to_canonical(ly_path, grid, resampling=Resampling.nearest,
                                dst_dtype=np.float32, fill_value=0.0)[0].astype(np.int16)
    return (tc >= 30.0) & ~((ly >= 1) & (ly <= 20))


def fuse(tile: str, force: bool = False) -> Path:
    grid = grid_for(tile)
    out = CACHE / f"{tile}.npz"
    if out.exists() and not force:
        log.info(f"[skip] {tile} label cache exists at {out}")
        return out

    log.info(f"[{tile}] fusing labels")
    sources = {
        "radd": _radd_sources(tile, grid),
        "gladl": _gladl_sources(tile, grid),
        "glads2": _glads2_sources(tile, grid),
        "hansen": _hansen_sources(tile, grid),
    }

    sum_weight = np.zeros(grid.shape, dtype=np.float32)
    n_hot = np.zeros(grid.shape, dtype=np.int16)  # #sources flagging this pixel
    n_avail = 0
    min_year = np.full(grid.shape, 9999, dtype=np.int16)
    sources_bitmask = np.zeros(grid.shape, dtype=np.uint8)
    bit_map = {"radd": 1, "gladl": 2, "glads2": 4, "hansen": 8}

    for name, triple in sources.items():
        if triple[0] is None:
            log.info(f"  [{tile}] {name} NOT AVAILABLE")
            continue
        weight, year, avail = triple
        n_avail += 1
        sum_weight += weight
        hot = weight > 0
        n_hot += hot.astype(np.int16)
        sources_bitmask[avail] |= bit_map[name]
        # update earliest-year
        year_is_real = year > 0
        update = year_is_real & (year.astype(np.int32) < min_year.astype(np.int32))
        min_year[update] = year[update]

    time_step = np.where(min_year == 9999, 0, min_year).astype(np.int16)

    y_soft = (sum_weight / max(n_avail, 1)).clip(0, 1).astype(np.float16)
    y_any = (sum_weight > 0)
    # hard: ≥ 2 sources each contributing ≥ 0.5
    #       or 1 source contributing ≥ 1.0 when only one source is available for this tile
    # We compute per-pixel: count of sources contributing ≥ 0.5, and max weight
    count_ge_05 = np.zeros(grid.shape, dtype=np.int16)
    max_weight = np.zeros(grid.shape, dtype=np.float32)
    for name, triple in sources.items():
        if triple[0] is None:
            continue
        w = triple[0]
        count_ge_05 += (w >= 0.5).astype(np.int16)
        max_weight = np.maximum(max_weight, w)
    y_hard = (count_ge_05 >= 2) | (count_ge_05 >= 1) & (max_weight >= 1.0) & (n_avail == 1)

    forest_2020 = _forest_2020(tile, grid)

    # apply post-2020 forest gate to all labels
    y_any = y_any & forest_2020
    y_hard = y_hard & forest_2020
    y_soft = np.where(forest_2020, y_soft, 0).astype(np.float16)
    time_step = np.where(forest_2020, time_step, 0).astype(np.int16)

    # summary
    log.info(f"  [{tile}] sources available: {n_avail}  forest_2020 px: {int(forest_2020.sum())}")
    log.info(f"  [{tile}] y_any rate: {float(y_any.mean()):.4f}  y_hard rate: {float(y_hard.mean()):.4f}")

    meta = {
        "tile": tile, "split": grid.split, "n_sources": n_avail,
        "sources_bitmask_legend": {k: v for k, v in bit_map.items()},
    }
    np.savez_compressed(
        out,
        y_any=y_any, y_hard=y_hard, y_soft=y_soft, time_step=time_step,
        forest_2020=forest_2020, sources_avail=sources_bitmask,
        meta=np.asarray(str(meta)),
    )
    log.info(f"  [{tile}] saved {out}  ({out.stat().st_size/1e6:.2f} MB)")
    return out


if __name__ == "__main__":
    import sys
    tiles = sys.argv[1:] or ["47QMB_0_8"]
    for t in tiles:
        fuse(t)

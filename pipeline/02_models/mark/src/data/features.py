"""Per-tile feature extraction on the canonical UTM 10 m grid.

Produces one compressed NPZ file per tile under ``cache/features/{tile}.npz``
with:

  - ``data``          : ``(n_features, H, W)`` float16 — all per-pixel features
  - ``feature_names`` : 1-D array of strings, length ``n_features``
  - ``valid``         : ``(H, W)`` bool — pixels where the core modalities are present

Feature groups (see ``FEATURE_GROUPS`` below for exact layout):

  1. AEF 2020                     (64 dims) — foundation-model baseline
  2. AEF 2025                     (64 dims) — latest state
  3. AEF cosine(AEF_y, AEF_2020)  (5 dims for y = 2021..2025)
  4. AEF ||AEF_y - AEF_2020||     (5 dims)
  5. S2 per-year spectral indices: NDVI / NBR / NDMI / NDWI
     year medians + p10 + p90 + cloud-free-month count (→ 4*3 + 1 = 13 per year × 6 years = 78)
  6. S2 spectral deltas vs 2020  (NDVI / NBR / NDMI Δ × 5 years = 15)
  7. S1 per-year VV (dB) mean, min, std                (3 × 6 = 18)
  8. S1 VV Δ vs 2020 (mean, min)                       (2 × 5 = 10)
  9. Hansen treecover2000                               (1)
 10. Hansen lossyear code as ordinal                    (1)
 11. Hansen post-2020 loss (0/1 binary)                 (1)
 12. Validity masks                                    (3)

Totals: 64 + 64 + 5 + 5 + 78 + 15 + 18 + 10 + 1 + 1 + 1 + 3 = 265.

Computation is ~3-5 min per tile (dominated by S2 cloud-mask over 6 years × ~12
scenes). Results are cached — re-running is near-instant.
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling
from s2cloudless import S2PixelCloudDetector

from .canonical_grid import CanonicalGrid, grid_for
from .reproject import reproject_to_canonical

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path("/shared-docker/oasis-mark-2")
DATA = ROOT / "data/makeathon-challenge"
# Writable overlay for user-added tiles (data/ is root-owned, can't modify)
DATA_EXTRA = ROOT / "external/makeathon-extras"
DATA_ROOTS = [DATA, DATA_EXTRA]
HANSEN = ROOT / "external/hansen/cropped"
CACHE = ROOT / "cache/features"
CACHE.mkdir(parents=True, exist_ok=True)


def _first_existing(subpath: str) -> Path | None:
    for root in DATA_ROOTS:
        p = root / subpath
        if p.exists():
            return p
    return None

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
POST_YEARS = [2021, 2022, 2023, 2024, 2025]  # "delta vs 2020" years

S2_FNAME_RE = re.compile(r"^(?P<tile>[\w_]+)__s2_l2a_(?P<y>\d{4})_(?P<m>\d{1,2})\.tif$")
S1_FNAME_RE = re.compile(r"^(?P<tile>[\w_]+)__s1_rtc_(?P<y>\d{4})_(?P<m>\d{1,2})_(?P<orb>ascending|descending)\.tif$")


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _usable_s2_files(tile: str, split: str) -> list[tuple[int, int, Path]]:
    """Return (year, month, path) for S2 files whose shape is reasonable (≥900×900)."""
    d = _first_existing(f"sentinel-2/{split}/{tile}__s2_l2a")
    if d is None:
        return []
    out = []
    for f in sorted(d.glob("*.tif")):
        m = S2_FNAME_RE.match(f.name)
        if not m:
            continue
        with rasterio.open(f) as src:
            if min(src.shape) < 900:
                continue
        out.append((int(m["y"]), int(m["m"]), f))
    return out


def _usable_s1_files(tile: str, split: str) -> list[tuple[int, int, str, Path]]:
    d = _first_existing(f"sentinel-1/{split}/{tile}__s1_rtc")
    if d is None:
        return []
    out = []
    for f in sorted(d.glob("*.tif")):
        m = S1_FNAME_RE.match(f.name)
        if not m:
            continue
        if f.stat().st_size < 50_000:
            continue
        with rasterio.open(f) as src:
            if min(src.shape) < 200:
                continue
        out.append((int(m["y"]), int(m["m"]), m["orb"], f))
    return out


def _best_s1_orbit(s1_files: list[tuple[int, int, str, Path]]) -> str:
    """Pick the orbit direction with the most years of full monthly coverage.

    Some tiles (e.g. Cameroon 33NTE_5_1) only have ascending; Colombia train
    tiles lose ascending for 2022-24. Using a single orbit per tile avoids
    month-level NaN holes in the feature stack.
    """
    cov = defaultdict(set)  # orb -> set of (year, month)
    for y, m, orb, _ in s1_files:
        cov[orb].add((y, m))
    if not cov:
        return "descending"
    # prefer the one with most distinct months overall
    return max(cov.keys(), key=lambda o: len(cov[o]))


# --------------------------------------------------------------------------
# Cloud masking (s2cloudless, run per scene)
# --------------------------------------------------------------------------

_DETECTOR: S2PixelCloudDetector | None = None


def _detector() -> S2PixelCloudDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = S2PixelCloudDetector(
            threshold=0.3, average_over=1, dilation_size=0, all_bands=False
        )
    return _DETECTOR


# Downsample factor for s2cloudless: we detect at H/_CLOUD_DS × W/_CLOUD_DS
# then upsample the binary mask with nearest-neighbour. s2cloudless is a
# per-pixel LGBM model on top of spatial averaging, so operating at 1/4
# resolution is ~16× faster with virtually no loss on sub-pixel clouds.
_CLOUD_DS = 4


def _s2_cloud_mask(bands_raw: np.ndarray) -> np.ndarray:
    """Compute cloud mask from reprojected S2 bands.

    ``bands_raw`` is (12, H, W) uint16-ish float, band order
    B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 (no B10).
    s2cloudless expects (1, H, W, 10) float 0-1 with bands:
    B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12.
    We pass zeros for B10 and scale by /10000.

    To keep this fast we run s2cloudless at 1/_CLOUD_DS resolution and
    upsample the binary mask.
    """
    H, W = bands_raw.shape[1:]
    ds = _CLOUD_DS
    h, w = H // ds, W // ds
    # Block-mean downsample each needed band.
    def dsample(b):
        return b[:h * ds, :w * ds].reshape(h, ds, w, ds).mean(axis=(1, 3))
    zero_b10 = np.zeros((h, w), dtype=np.float32)
    stack = np.stack([
        dsample(bands_raw[0]),   # B01
        dsample(bands_raw[1]),   # B02
        dsample(bands_raw[3]),   # B04
        dsample(bands_raw[4]),   # B05
        dsample(bands_raw[7]),   # B08
        dsample(bands_raw[8]),   # B8A
        dsample(bands_raw[9]),   # B09
        zero_b10,                # B10 (absent)
        dsample(bands_raw[10]),  # B11
        dsample(bands_raw[11]),  # B12
    ], axis=-1).astype(np.float32) / 10000.0
    probs_small = _detector().get_cloud_probability_maps(stack[None])[0]
    mask_small = (probs_small > 0.3).astype(np.uint8)
    # Nearest-neighbour upsample to (H, W)
    mask = np.repeat(np.repeat(mask_small, ds, axis=0), ds, axis=1)
    if mask.shape != (H, W):
        # pad or crop to exact H×W (in case H or W isn't divisible by ds)
        padded = np.zeros((H, W), dtype=np.uint8)
        padded[: mask.shape[0], : mask.shape[1]] = mask[: H, : W]
        mask = padded
    return mask.astype(bool)


# --------------------------------------------------------------------------
# Feature builders
# --------------------------------------------------------------------------

def _aef_features(tile: str, split: str, grid: CanonicalGrid
                  ) -> tuple[np.ndarray, list[str]]:
    """Return AEF features and names.

    Stacks: AEF_2020 (64), AEF_2025 (64), cos(y, 2020) for y=2021..2025 (5),
    ||AEF_y - AEF_2020|| for y=2021..2025 (5). Total 138.
    """
    aefs: dict[int, np.ndarray] = {}
    for y in YEARS:
        p = _first_existing(f"aef-embeddings/{split}/{tile}_{y}.tiff")
        if p is None:
            aefs[y] = np.full((64, *grid.shape), np.nan, dtype=np.float32)
            continue
        aefs[y] = reproject_to_canonical(p, grid, resampling=Resampling.bilinear)

    baseline = aefs[2020]
    feats: list[np.ndarray] = []
    names: list[str] = []

    # (1) AEF 2020
    feats.extend(baseline[i] for i in range(64))
    names.extend(f"aef2020_{i:02d}" for i in range(64))

    # (2) AEF 2025
    feats.extend(aefs[2025][i] for i in range(64))
    names.extend(f"aef2025_{i:02d}" for i in range(64))

    baseline_norm = np.sqrt(np.sum(baseline ** 2, axis=0) + 1e-8)
    for y in POST_YEARS:
        cur = aefs[y]
        cur_norm = np.sqrt(np.sum(cur ** 2, axis=0) + 1e-8)
        dot = np.sum(baseline * cur, axis=0)
        cos = dot / (baseline_norm * cur_norm + 1e-8)
        delta = cur - baseline
        delta_norm = np.sqrt(np.sum(delta ** 2, axis=0))
        feats.append(cos)
        names.append(f"aef_cos_{y}_vs_2020")
        feats.append(delta_norm)
        names.append(f"aef_dnorm_{y}_vs_2020")

    return np.stack(feats, axis=0), names


def _s2_yearly_features(tile: str, split: str, grid: CanonicalGrid
                        ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """S2 annual aggregates. Returns (features, names, cloud_frac_per_year).

    For each year: NDVI, NBR, NDMI, NDWI → median, p10, p90; plus cloud-free month count.
    For each post-2020 year: ΔNDVI, ΔNBR, ΔNDMI vs 2020-median.
    """
    s2_files = _usable_s2_files(tile, split)
    # group by year
    by_year: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for y, m, p in s2_files:
        by_year[y].append((m, p))

    # accumulators per year & index
    indices = ["ndvi", "nbr", "ndmi", "ndwi"]
    stats = ["min", "median", "max"]  # fast NaN-aware aggregates
    idx_stack: dict[int, dict[str, np.ndarray]] = {}
    cloud_frac = np.full(len(YEARS), np.nan, dtype=np.float32)
    cf_counts = np.zeros(len(YEARS), dtype=np.int16)

    # suppress "All-NaN slice" warnings: we produce NaN on purpose for empty pixels
    import warnings
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    for yi, y in enumerate(YEARS):
        months = sorted(by_year.get(y, []))
        year_clouds = []
        year_idx: dict[str, list[np.ndarray]] = {k: [] for k in indices}

        for mi, p in months:
            bands = reproject_to_canonical(
                p, grid, resampling=Resampling.bilinear,
                dst_dtype=np.float32, fill_value=0.0,
            )  # (12, H, W) — bands B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
            valid = bands[1] > 0  # B02 > 0 = has data
            cloud = _s2_cloud_mask(bands)
            ok = valid & ~cloud

            # compute spectral indices (float) with nodata
            b03, b04, b08, b11, b12 = bands[2], bands[3], bands[7], bands[10], bands[11]
            ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
            nbr = (b08 - b12) / (b08 + b12 + 1e-6)
            ndmi = (b08 - b11) / (b08 + b11 + 1e-6)
            ndwi = (b03 - b08) / (b03 + b08 + 1e-6)

            for name, arr in [("ndvi", ndvi), ("nbr", nbr), ("ndmi", ndmi), ("ndwi", ndwi)]:
                masked = np.where(ok, arr, np.nan)
                year_idx[name].append(masked)
            year_clouds.append(float(cloud.mean()))

        # aggregate per year
        year_stack = {}
        cf_count = 0
        for name in indices:
            arrs = year_idx[name]
            if not arrs:
                year_stack[name] = {s: np.full(grid.shape, np.nan, dtype=np.float32) for s in stats}
                continue
            stacked = np.stack(arrs, axis=0)
            if name == "ndvi":
                cf_count = np.isfinite(stacked).sum(axis=0)
            year_stack[name] = {
                "min": np.nanmin(stacked, axis=0).astype(np.float32),
                "median": np.nanmedian(stacked, axis=0).astype(np.float32),
                "max": np.nanmax(stacked, axis=0).astype(np.float32),
            }
        idx_stack[y] = year_stack
        cf_counts[yi] = int(np.nanmedian(cf_count)) if np.any(cf_count) else 0
        cloud_frac[yi] = float(np.mean(year_clouds)) if year_clouds else float("nan")
        log.info(f"[{tile}] S2 year {y} done ({len(months)} scenes, cloud_frac={cloud_frac[yi]:.2f})")

    feats: list[np.ndarray] = []
    names: list[str] = []
    for yi, y in enumerate(YEARS):
        for name in indices:
            for s in stats:
                feats.append(idx_stack[y][name][s])
                names.append(f"s2_{name}_{s}_{y}")
        feats.append(np.full(grid.shape, cf_counts[yi], dtype=np.float32))
        names.append(f"s2_cloudfree_months_{y}")

    # deltas vs 2020 (on the per-pixel median)
    base_map = idx_stack[2020]
    for y in POST_YEARS:
        cur_map = idx_stack[y]
        for idx_name in ["ndvi", "nbr", "ndmi"]:
            feats.append(cur_map[idx_name]["median"] - base_map[idx_name]["median"])
            names.append(f"s2_{idx_name}_delta_{y}_vs_2020")

    return np.stack(feats, axis=0), names, cloud_frac


def _s1_yearly_features(tile: str, split: str, grid: CanonicalGrid
                        ) -> tuple[np.ndarray, list[str], str]:
    """S1 annual VV (dB) aggregates: mean, min, std; Δ(mean/min) vs 2020. Returns (features, names, orbit_used)."""
    s1_files = _usable_s1_files(tile, split)
    orbit = _best_s1_orbit(s1_files)
    # keep only chosen orbit
    by_year: dict[int, list[Path]] = defaultdict(list)
    for y, m, orb, p in s1_files:
        if orb == orbit:
            by_year[y].append(p)

    year_stack: dict[int, dict[str, np.ndarray]] = {}
    for y in YEARS:
        paths = sorted(by_year.get(y, []))
        month_arrays: list[np.ndarray] = []
        for p in paths:
            lin = reproject_to_canonical(p, grid, resampling=Resampling.bilinear,
                                         dst_dtype=np.float32, fill_value=np.nan)[0]
            # convert to dB (avoid log(0))
            db = np.where(lin > 0, 10 * np.log10(lin), np.nan)
            month_arrays.append(db)
        if not month_arrays:
            year_stack[y] = {s: np.full(grid.shape, np.nan, dtype=np.float32) for s in ("mean", "min", "std")}
            continue
        stacked = np.stack(month_arrays, axis=0)
        year_stack[y] = {
            "mean": np.nanmean(stacked, axis=0),
            "min": np.nanmin(stacked, axis=0),
            "std": np.nanstd(stacked, axis=0),
        }

    feats, names = [], []
    for y in YEARS:
        for s in ("mean", "min", "std"):
            feats.append(year_stack[y][s])
            names.append(f"s1_vv_{s}_{y}")
    # deltas vs 2020
    for y in POST_YEARS:
        for s in ("mean", "min"):
            feats.append(year_stack[y][s] - year_stack[2020][s])
            names.append(f"s1_vv_{s}_delta_{y}_vs_2020")
    return np.stack(feats, axis=0), names, orbit


def _hansen_features(tile: str, grid: CanonicalGrid) -> tuple[np.ndarray, list[str]]:
    """Treecover2000 (0-100), lossyear-encoded (0=none, 1=pre-2001, 2=2001-2020, 3=2021, 4=2022, 5=2023),
    and a 0/1 post-2020 loss flag."""
    tc_path = HANSEN / f"{tile}_treecover2000.tif"
    ly_path = HANSEN / f"{tile}_lossyear.tif"
    dm_path = HANSEN / f"{tile}_datamask.tif"
    tc = reproject_to_canonical(tc_path, grid, resampling=Resampling.bilinear,
                                dst_dtype=np.float32, fill_value=0.0)[0]
    ly = reproject_to_canonical(ly_path, grid, resampling=Resampling.nearest,
                                dst_dtype=np.float32, fill_value=0.0)[0]
    # encode lossyear: 0 none, 1..20 = 2001..2020, 21..23 = 2021..2023
    ly_int = ly.astype(np.int16)
    encoded = np.zeros_like(ly, dtype=np.float32)
    encoded[(ly_int >= 1) & (ly_int <= 20)] = 2  # 2001-2020
    encoded[ly_int == 21] = 3
    encoded[ly_int == 22] = 4
    encoded[ly_int == 23] = 5
    post2020_loss = ((ly_int >= 21) & (ly_int <= 23)).astype(np.float32)
    return np.stack([tc, encoded, post2020_loss], axis=0), [
        "hansen_treecover2000", "hansen_lossyear_encoded", "hansen_post2020_loss",
    ]


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def extract(tile: str, force: bool = False) -> Path:
    grid = grid_for(tile)
    out = CACHE / f"{tile}.npz"
    if out.exists() and not force:
        log.info(f"[skip] {tile} cache exists at {out}")
        return out

    log.info(f"[{tile}] start")
    t0 = time.time()

    aef, aef_n = _aef_features(tile, grid.split, grid)
    log.info(f"[{tile}] AEF done: {aef.shape} ({time.time() - t0:.0f}s)")

    s2, s2_n, cloud_frac = _s2_yearly_features(tile, grid.split, grid)
    log.info(f"[{tile}] S2 done: {s2.shape} ({time.time() - t0:.0f}s)")

    s1, s1_n, orbit = _s1_yearly_features(tile, grid.split, grid)
    log.info(f"[{tile}] S1 done ({orbit}): {s1.shape} ({time.time() - t0:.0f}s)")

    hs, hs_n = _hansen_features(tile, grid)
    log.info(f"[{tile}] Hansen done: {hs.shape} ({time.time() - t0:.0f}s)")

    aef_valid = np.any(np.isfinite(aef[:64]), axis=0)
    s1_valid = np.any(np.isfinite(s1[::3][:6]), axis=0)  # year means
    hansen_valid = hs[0] >= 0
    valid = aef_valid & s1_valid & hansen_valid  # S2 may be broken on some tiles; keep it optional

    # Validity channels (3)
    s2_has_any = np.isfinite(s2[:12]).any(axis=0).astype(np.float32)  # at least one 2020 stat valid
    validity = np.stack([
        aef_valid.astype(np.float32),
        s1_valid.astype(np.float32),
        s2_has_any,
    ], axis=0)
    val_n = ["valid_aef", "valid_s1", "valid_s2"]

    data = np.concatenate([aef, s2, s1, hs, validity], axis=0).astype(np.float16)
    names = aef_n + s2_n + s1_n + hs_n + val_n
    assert data.shape[0] == len(names), (data.shape, len(names))

    # Metadata
    meta = {
        "tile": tile, "split": grid.split, "epsg": grid.epsg,
        "shape": list(grid.shape), "pixel_size_m": grid.pixel_size,
        "ul_x": grid.ul_x, "ul_y": grid.ul_y,
        "cloud_frac_per_year": cloud_frac.tolist(),
        "s1_orbit": orbit,
        "n_features": data.shape[0],
    }
    np.savez(
        out, data=data, feature_names=np.asarray(names),
        valid=valid, meta=np.asarray(str(meta)),
    )
    log.info(f"[{tile}] saved {out} ({out.stat().st_size/1e6:.1f} MB, {time.time() - t0:.0f}s total)")
    return out


if __name__ == "__main__":
    import sys
    tiles = sys.argv[1:] or [grid_for("47QMB_0_8").tile_id]  # default small test
    for t in tiles:
        extract(t)

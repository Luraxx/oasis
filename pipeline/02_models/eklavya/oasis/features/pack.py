"""Build the per-pixel tabular feature pack from a cached tile bundle.

The feature pack is the input to LightGBM and the temporal CNN (the
latter also gets the raw S2 monthly stack alongside).

Feature groups produced:

S2 (per index in INDEX_NAMES, default 6: ndvi, nbr, ndmi, evi, mndwi, bsi):
  for each index, per pixel:
    yearly mean 2020..2025                          (6)
    overall mean / std / min / max                  (4)
    linear slope across yearly means + R^2          (2)
    max year-to-year drop and the year of max drop  (2)
    dry-season vs wet-season mean (proxy: month %)  (2)

S1 (per orbit, ascending/descending):
  yearly mean dB                                    (6)
  overall mean / std / slope                        (3)
  max year-to-year drop                             (1)
  rolling-3 month coherence proxy (std of std)      (1)

AEF:
  every (year, channel) flattened                   (6 * 64 = 384)
  year-to-year cosine distance                      (5)
  L2 norm of (latest - earliest)                    (1)

External (loaded if cache is present, else zero-filled):
  worldcover one-hot 11 classes                     (11)
  hansen treecover2000, lossyear                    (2)
  jrc tmf 2020 / 2024 class                         (2)
  distance to nearest road (km)                     (1)

Returns a (C, H, W) float32 stack and the per-feature names. Float32 NaN
handling: every reduction goes through ``np.nan*`` aware code; the final
stack is ``nan_to_num``'d before returning.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from oasis.features import s2 as s2_feat

# These reductions are intentionally NaN-tolerant; suppress chatty
# numpy warnings that would otherwise dominate logs.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")


DEFAULT_INDICES = ("ndvi", "nbr", "ndmi", "evi", "mndwi", "bsi")
YEARS_FULL = (2020, 2021, 2022, 2023, 2024, 2025)
DRY_MONTHS = (6, 7, 8, 9)  # proxy applicable to the Amazon dry season


def _yearly_reduce(values: np.ndarray, ym: np.ndarray, years: Sequence[int]) -> np.ndarray:
    """For a (T, H, W) stack, return (Y, H, W) yearly means using only timesteps
    with year in ``years``. Uses np.nanmean."""
    out = np.full((len(years), *values.shape[1:]), np.nan, dtype=np.float32)
    for i, yr in enumerate(years):
        idx = np.where(ym[:, 0] == yr)[0]
        if len(idx) == 0:
            continue
        with np.errstate(invalid="ignore"):
            out[i] = np.nanmean(values[idx], axis=0).astype(np.float32)
    return out


def _slope_and_r2(yearly: np.ndarray, years: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel linear slope and R^2 over ``yearly`` of shape (Y, H, W)."""
    x = np.array(years, dtype=np.float32)
    x = x - x.mean()
    y = yearly
    valid = np.isfinite(y)
    yfilled = np.where(valid, y, 0.0)
    n = valid.sum(axis=0).astype(np.float32)
    sumx = (x[:, None, None] * valid).sum(axis=0)
    sumxx = (x[:, None, None] ** 2 * valid).sum(axis=0)
    sumy = (yfilled * valid).sum(axis=0)
    sumxy = (x[:, None, None] * yfilled * valid).sum(axis=0)

    denom = n * sumxx - sumx * sumx
    safe_denom = np.where(np.abs(denom) > 1e-6, denom, 1.0)
    slope = (n * sumxy - sumx * sumy) / safe_denom
    intercept = (sumy - slope * sumx) / np.where(n > 0, n, 1.0)

    pred = intercept[None] + slope[None] * x[:, None, None]
    resid = (yfilled - pred) * valid
    ss_res = (resid * resid).sum(axis=0)
    ymean = sumy / np.where(n > 0, n, 1.0)
    ss_tot = (((yfilled - ymean[None]) * valid) ** 2).sum(axis=0)
    safe_tot = np.where(ss_tot > 1e-6, ss_tot, 1.0)
    r2 = 1.0 - ss_res / safe_tot
    r2 = np.where(np.isfinite(r2), r2, 0.0)
    slope = np.where(np.abs(denom) > 1e-6, slope, 0.0)
    return slope.astype(np.float32), r2.astype(np.float32)


def _max_drop(yearly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Largest negative year-to-year delta and the index (year position) it occurred."""
    deltas = np.diff(yearly, axis=0)  # (Y-1, H, W); positive = increase
    deltas = np.where(np.isfinite(deltas), deltas, 0.0)
    idx = np.argmin(deltas, axis=0).astype(np.int8)  # most-negative
    drop = np.take_along_axis(deltas, idx[None], axis=0)[0]
    drop = np.where(drop < 0, drop, 0.0)
    return drop.astype(np.float32), idx.astype(np.float32)


def _seasonal_mean(values: np.ndarray, ym: np.ndarray, months: Sequence[int]) -> np.ndarray:
    idx = np.where(np.isin(ym[:, 1], list(months)))[0]
    if len(idx) == 0:
        return np.full(values.shape[1:], np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmean(values[idx], axis=0).astype(np.float32)


def s2_feature_pack(
    bundle: dict,
    indices: Sequence[str] = DEFAULT_INDICES,
    years: Sequence[int] = YEARS_FULL,
) -> tuple[np.ndarray, list[str]]:
    """Build all S2-derived features as a (C, H, W) stack."""
    s2_stack = bundle["s2_stack"]  # (T, 12, H, W) uint16
    s2_valid = bundle["s2_valid"]  # (T, H, W) bool
    s2_ym = bundle["s2_ym"]        # (T, 2) int16

    feats: list[np.ndarray] = []
    names: list[str] = []

    # Compute monthly indices once per index name
    h, w = s2_stack.shape[-2:]
    t = s2_stack.shape[0]
    monthly_idx: dict[str, np.ndarray] = {name: np.full((t, h, w), np.nan, dtype=np.float32) for name in indices}
    for ti in range(t):
        all_idx = s2_feat.compute_indices(s2_stack[ti], valid=s2_valid[ti])
        for name in indices:
            monthly_idx[name][ti] = all_idx[name]

    for name in indices:
        values = monthly_idx[name]

        yearly = _yearly_reduce(values, s2_ym, years)
        for yi, yr in enumerate(years):
            feats.append(yearly[yi])
            names.append(f"s2_{name}_y{yr}")

        with np.errstate(invalid="ignore"):
            mean_all = np.nanmean(values, axis=0)
            std_all = np.nanstd(values, axis=0)
            min_all = np.nanmin(values, axis=0)
            max_all = np.nanmax(values, axis=0)
        feats.extend([mean_all, std_all, min_all, max_all])
        names.extend([f"s2_{name}_mean", f"s2_{name}_std", f"s2_{name}_min", f"s2_{name}_max"])

        slope, r2 = _slope_and_r2(yearly, years)
        feats.extend([slope, r2])
        names.extend([f"s2_{name}_slope", f"s2_{name}_r2"])

        drop, drop_year = _max_drop(yearly)
        feats.extend([drop, drop_year])
        names.extend([f"s2_{name}_max_drop", f"s2_{name}_drop_year"])

        dry = _seasonal_mean(values, s2_ym, DRY_MONTHS)
        wet = _seasonal_mean(values, s2_ym, [m for m in range(1, 13) if m not in DRY_MONTHS])
        feats.extend([dry, wet])
        names.extend([f"s2_{name}_dry", f"s2_{name}_wet"])

    stack = np.stack(feats, axis=0)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return stack, names


def _s1_orbit_feature_names(orbit: str, years: Sequence[int]) -> list[str]:
    names: list[str] = [f"s1_{orbit}_y{yr}" for yr in years]
    names.extend(
        [f"s1_{orbit}_mean", f"s1_{orbit}_std", f"s1_{orbit}_slope", f"s1_{orbit}_max_drop"]
    )
    names.append(f"s1_{orbit}_coherence")
    return names


def _s1_features_for_orbit(
    db_stack: np.ndarray, ym: np.ndarray, orbit: str, years: Sequence[int]
) -> tuple[list[np.ndarray], list[str]]:
    names = _s1_orbit_feature_names(orbit, years)
    if db_stack.shape[0] == 0:
        h, w = db_stack.shape[-2:]
        feats = [np.zeros((h, w), dtype=np.float32) for _ in names]
        return feats, names

    feats: list[np.ndarray] = []
    yearly = _yearly_reduce(db_stack, ym, years)
    for yi, _yr in enumerate(years):
        feats.append(yearly[yi])

    with np.errstate(invalid="ignore"):
        mean_all = np.nanmean(db_stack, axis=0).astype(np.float32)
        std_all = np.nanstd(db_stack, axis=0).astype(np.float32)
    slope, _ = _slope_and_r2(yearly, years)
    drop, _ = _max_drop(yearly)
    feats.extend([mean_all, std_all, slope, drop])

    if db_stack.shape[0] >= 3:
        roll = np.full_like(db_stack, np.nan)
        for ti in range(db_stack.shape[0]):
            lo = max(0, ti - 1)
            hi = min(db_stack.shape[0], ti + 2)
            with np.errstate(invalid="ignore"):
                roll[ti] = np.nanstd(db_stack[lo:hi], axis=0)
        with np.errstate(invalid="ignore"):
            coh = np.nanmean(roll, axis=0).astype(np.float32)
    else:
        coh = np.zeros(db_stack.shape[-2:], dtype=np.float32)
    feats.append(coh)
    return feats, names


def s1_feature_pack(
    bundle: dict, years: Sequence[int] = YEARS_FULL
) -> tuple[np.ndarray, list[str]]:
    asc = bundle["s1_asc_db"]
    desc = bundle["s1_desc_db"]
    asc_ym = bundle["s1_asc_ym"]
    desc_ym = bundle["s1_desc_ym"]
    h, w = asc.shape[-2:] if asc.shape[0] > 0 else desc.shape[-2:]

    if asc.shape[0] == 0:
        asc = np.zeros((0, h, w), dtype=np.float32)
    if desc.shape[0] == 0:
        desc = np.zeros((0, h, w), dtype=np.float32)

    asc_feats, asc_names = _s1_features_for_orbit(asc, asc_ym, "asc", years)
    desc_feats, desc_names = _s1_features_for_orbit(desc, desc_ym, "desc", years)

    stack = np.stack(asc_feats + desc_feats, axis=0)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return stack, asc_names + desc_names


def aef_feature_pack(bundle: dict) -> tuple[np.ndarray, list[str]]:
    aef = bundle["aef_stack"]   # (Y, 64, H, W)
    years = bundle["aef_years"]  # (Y,)
    if aef.shape[0] == 0:
        h, w = bundle["s2_stack"].shape[-2:]
        return np.zeros((0, h, w), dtype=np.float32), []

    feats: list[np.ndarray] = []
    names: list[str] = []
    n_y, n_c, h, w = aef.shape
    for yi in range(n_y):
        for ci in range(n_c):
            feats.append(aef[yi, ci])
            names.append(f"aef_y{int(years[yi])}_c{ci:02d}")

    # Year-to-year cosine distance and L2 change
    for yi in range(1, n_y):
        a = aef[yi - 1]
        b = aef[yi]
        dot = (a * b).sum(axis=0)
        na = np.sqrt((a * a).sum(axis=0))
        nb = np.sqrt((b * b).sum(axis=0))
        cos = np.clip(dot / np.maximum(na * nb, 1e-6), -1.0, 1.0)
        feats.append((1.0 - cos).astype(np.float32))
        names.append(f"aef_cosdist_{int(years[yi - 1])}_{int(years[yi])}")
    if n_y >= 2:
        diff = aef[-1] - aef[0]
        l2 = np.sqrt((diff * diff).sum(axis=0)).astype(np.float32)
        feats.append(l2)
        names.append(f"aef_l2_{int(years[0])}_{int(years[-1])}")

    stack = np.stack(feats, axis=0)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return stack, names


def external_feature_pack(
    tile_id: str, split: str, *, h: int, w: int
) -> tuple[np.ndarray, list[str]]:
    """Optional external priors (WorldCover, Hansen, JRC TMF, OSM roads).

    Loads from ``data/externals/{tile_id}/<layer>.npy`` if present;
    otherwise returns zeros so models stay schema-stable across runs.
    """
    from oasis import paths

    base = paths.EXTERNALS_ROOT / split / tile_id
    feats: list[np.ndarray] = []
    names: list[str] = []

    # WorldCover one-hot (11 ESA WorldCover classes - codes mapped to 0..10)
    wc_path = base / "worldcover.npy"
    n_wc = 11
    if wc_path.exists():
        wc = np.load(wc_path)
        for c in range(n_wc):
            feats.append((wc == c).astype(np.float32))
            names.append(f"ext_worldcover_{c}")
    else:
        for c in range(n_wc):
            feats.append(np.zeros((h, w), dtype=np.float32))
            names.append(f"ext_worldcover_{c}")

    # Hansen 2 layers
    for layer in ("hansen_treecover2000", "hansen_lossyear"):
        p = base / f"{layer}.npy"
        if p.exists():
            feats.append(np.load(p).astype(np.float32))
        else:
            feats.append(np.zeros((h, w), dtype=np.float32))
        names.append(f"ext_{layer}")

    # JRC TMF 2 layers
    for layer in ("jrc_tmf_2020", "jrc_tmf_2024"):
        p = base / f"{layer}.npy"
        if p.exists():
            feats.append(np.load(p).astype(np.float32))
        else:
            feats.append(np.zeros((h, w), dtype=np.float32))
        names.append(f"ext_{layer}")

    # OSM road distance
    p = base / "osm_road_distance_km.npy"
    if p.exists():
        feats.append(np.load(p).astype(np.float32))
    else:
        feats.append(np.zeros((h, w), dtype=np.float32))
    names.append("ext_osm_road_distance_km")

    stack = np.stack(feats, axis=0)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return stack, names


def assemble_feature_pack(
    bundle: dict, *, tile_id: str, split: str, include_external: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Concatenate every feature group into one (C, H, W) float32 stack."""
    s2, n_s2 = s2_feature_pack(bundle)
    s1, n_s1 = s1_feature_pack(bundle)
    aef, n_aef = aef_feature_pack(bundle)
    h, w = s2.shape[-2:]
    parts = [s2, s1, aef]
    names = n_s2 + n_s1 + n_aef
    if include_external:
        ext, n_ext = external_feature_pack(tile_id, split, h=h, w=w)
        parts.append(ext)
        names.extend(n_ext)
    full = np.concatenate(parts, axis=0)
    return full.astype(np.float32), names

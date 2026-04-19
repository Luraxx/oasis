"""Test-tile inference + GeoJSON packaging.

Loads the trained model bundle (LightGBM + TCN + UNet + calibrated
stacker), produces per-tile probability rasters, post-processes to
binary, writes a confidence GeoTIFF, and emits the submission GeoJSON.

Also computes the per-polygon timing estimate (modal year of max NBR
drop) for the optional bonus.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from oasis import cache, paths, validation
from oasis.ensemble import CalibratedStack
from oasis.features.pack import _max_drop, _yearly_reduce
from oasis.features import s2 as s2_feat
from oasis.postprocess import clean_and_filter, pixel_area_m2
from submission_utils import raster_to_geojson


@dataclass
class PredictBundle:
    """Inference bundle.

    ``unet_paths`` is a mapping from model namespace (e.g. ``unet_eb3``)
    to a list of checkpoint paths; predictions across all paths in a
    namespace are averaged at inference (multi-fold averaging). Each
    namespace contributes one row to the calibrated-stack input. The
    convenience ``unet_path`` field is kept for backward compatibility.
    """
    lgbm_paths: list[Path] = None  # type: ignore[assignment]
    tcn_paths: list[Path] = None   # type: ignore[assignment]
    unet_paths: dict[str, list[Path]] = None  # type: ignore[assignment]
    stack_path: Path | None = None
    threshold: float = 0.5
    min_area_ha: float = 0.5
    use_external: bool = True
    # UNet inference knobs. Default = single-scale + half-overlap (stride
    # = patch_size/2) which is tractable on one GPU and still benefits
    # from the 8-way D4 TTA plus the multi-fold averaging within each
    # encoder namespace.  Override to (0.75, 1.0, 1.25) with stride_div=4
    # only for a final accuracy squeeze when compute is cheap.
    unet_tta: bool = True
    unet_multiscale: tuple[float, ...] = (1.0,)
    unet_stride_div: int = 2

    # Backward compat aliases (single-checkpoint shortcuts).
    lgbm_path: Path | None = None
    tcn_path: Path | None = None
    unet_path: Path | None = None

    def __post_init__(self) -> None:
        if self.lgbm_paths is None:
            self.lgbm_paths = [self.lgbm_path] if self.lgbm_path else []
        if self.tcn_paths is None:
            self.tcn_paths = [self.tcn_path] if self.tcn_path else []
        if self.unet_paths is None:
            self.unet_paths = {"unet": [self.unet_path]} if self.unet_path else {}


def _ensure_meta(meta: dict) -> dict:
    rmeta = cache.restore_rasterio_meta(meta)
    return rmeta


def _write_tiff(path: Path, data: np.ndarray, base_meta: dict, dtype: str) -> None:
    out_meta = base_meta.copy()
    out_meta.update(dtype=dtype, count=1, nodata=0 if dtype == "uint8" else None, compress="LZW")
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(data, 1)


def _avg(probs: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)


def _model_probs_for_tile(
    tile_id: str, split: str, bundle: PredictBundle, *, verbose: bool = True
) -> dict[str, np.ndarray]:
    """Return one calibrated-stack input per model namespace.

    Within a namespace, averaging across LORO + final checkpoints reduces
    variance ("k-fold averaging").
    """
    probs: dict[str, np.ndarray] = {}

    if bundle.lgbm_paths:
        from oasis.models import lgbm as lgbm_mod

        per_fold = []
        for ckpt in bundle.lgbm_paths:
            clf, feature_names = lgbm_mod.load_model(ckpt)
            if verbose:
                print(f"  [lgbm:{ckpt.name}] predicting {tile_id}")
            per_fold.append(
                lgbm_mod.predict_tile(
                    clf,
                    tile_id,
                    split,
                    include_external=bundle.use_external,
                    feature_names=feature_names,
                )
            )
        probs["lgbm"] = _avg(per_fold)

    if bundle.tcn_paths:
        from oasis.models import tcn as tcn_mod

        per_fold = []
        for ckpt in bundle.tcn_paths:
            model, _ = tcn_mod.load_tcn(ckpt)
            if verbose:
                print(f"  [tcn:{ckpt.name}] predicting {tile_id}")
            per_fold.append(tcn_mod.predict_tile(model, tile_id, split))
        probs["tcn"] = _avg(per_fold)

    if bundle.unet_paths:
        from oasis.models import unet as unet_mod

        for ns, ckpts in bundle.unet_paths.items():
            per_fold = []
            for ckpt in ckpts:
                model, cfg = unet_mod.load_unet(ckpt)
                if verbose:
                    print(f"  [{ns}:{ckpt.name}] predicting {tile_id}")
                per_fold.append(
                    unet_mod.predict_tile_unet(
                        model, tile_id, split, cfg,
                        tta=bundle.unet_tta,
                        multiscale=bundle.unet_multiscale,
                        stride_div=bundle.unet_stride_div,
                    )
                )
            probs[ns] = _avg(per_fold)

    return probs


def estimate_polygon_timings(
    binary: np.ndarray, arrays: dict, *, default_year: int = 2023
) -> dict[int, int]:
    """Return ``{component_id: YYMM}`` derived from NBR temporal drops.

    YYMM = (year % 100) * 100 + month, e.g. June 2024 -> 2406.
    The month is chosen as the modal month (within the modal year) of the
    largest month-to-month NBR drop per pixel. Falls back to month=06 if
    monthly evidence is unavailable.
    """
    from scipy.ndimage import label as cc_label

    labels, n = cc_label(binary)
    if n == 0:
        return {}

    s2_stack = arrays["s2_stack"]
    s2_valid = arrays["s2_valid"]
    s2_ym = arrays["s2_ym"]
    h, w = s2_stack.shape[-2:]

    T = s2_stack.shape[0]
    monthly_nbr = np.full((T, h, w), np.nan, dtype=np.float32)
    for ti in range(T):
        idx = s2_feat.compute_indices(s2_stack[ti].astype(np.float32), valid=s2_valid[ti])
        monthly_nbr[ti] = idx["nbr"]

    years = (2020, 2021, 2022, 2023, 2024, 2025)
    yearly = _yearly_reduce(monthly_nbr, s2_ym, years)
    _, drop_year = _max_drop(yearly)
    drop_year_int = drop_year.astype(np.int8)

    # Per-pixel month of largest month-to-month NBR drop (1..12, 0 if unknown)
    if T >= 2:
        with np.errstate(invalid="ignore"):
            diffs = monthly_nbr[1:] - monthly_nbr[:-1]
        diffs_filled = np.where(np.isnan(diffs), np.inf, diffs)
        drop_t_idx = np.argmin(diffs_filled, axis=0)  # 0..T-2
        valid_drop = np.isfinite(np.take_along_axis(diffs, drop_t_idx[None], axis=0)[0])
        # Map t-index -> calendar (year, month). Use the *later* month of the pair.
        ym_pair = np.array(s2_ym, dtype=np.int32)  # shape (T,2)
        later_year = ym_pair[1:, 0][drop_t_idx]
        later_month = ym_pair[1:, 1][drop_t_idx]
        later_year = np.where(valid_drop, later_year, 0)
        later_month = np.where(valid_drop, later_month, 0)
    else:
        later_year = np.zeros((h, w), dtype=np.int32)
        later_month = np.zeros((h, w), dtype=np.int32)

    out: dict[int, int] = {}
    for cid in range(1, n + 1):
        m = labels == cid
        if not m.any():
            yy = default_year % 100
            out[cid] = int(yy * 100 + 6)
            continue
        years_in = drop_year_int[m]
        counts = np.bincount(np.clip(years_in, 0, len(years) - 1), minlength=len(years))
        modal_idx = int(counts.argmax())
        modal_year = int(years[min(max(modal_idx + 1, 0), len(years) - 1)])
        # Pick modal month among pixels whose finer-grained drop year matches modal_year
        in_year = m & (later_year == modal_year) & (later_month > 0)
        if in_year.any():
            mc = np.bincount(later_month[in_year].astype(np.int64), minlength=13)
            month = int(mc[1:].argmax() + 1)
        else:
            month = 6  # safe mid-year fallback
        yy = modal_year % 100
        out[cid] = int(yy * 100 + month)
    return out


def predict_tile(
    tile_id: str,
    split: str,
    bundle: PredictBundle,
    *,
    out_root: Path = paths.SUBMISSION_ROOT,
    verbose: bool = True,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    arrays, meta = cache.load_tile_cache(tile_id, split)
    rmeta = _ensure_meta(meta)
    probs = _model_probs_for_tile(tile_id, split, bundle, verbose=verbose)

    if bundle.stack_path is not None and bundle.stack_path.exists():
        stack = CalibratedStack.load(bundle.stack_path)
        prob = stack.stack(probs, region=validation.region_of(tile_id))
    else:
        prob = np.mean(np.stack(list(probs.values()), axis=0), axis=0).astype(np.float32)

    binary_raw = (prob >= bundle.threshold).astype(np.uint8)
    binary = clean_and_filter(binary_raw, rmeta["transform"], min_area_ha=bundle.min_area_ha)

    # Persist artifacts.
    bin_path = out_root / f"pred_{tile_id}.tif"
    prob_path = out_root / f"prob_{tile_id}.tif"
    _write_tiff(bin_path, binary.astype(np.uint8), rmeta, dtype="uint8")
    _write_tiff(prob_path, (prob * 1000).astype(np.uint16), rmeta, dtype="uint16")

    geojson_path = out_root / f"pred_{tile_id}.geojson"
    n_polys = 0
    if binary.any():
        try:
            geojson = raster_to_geojson(
                bin_path, output_path=geojson_path, min_area_ha=bundle.min_area_ha
            )
            n_polys = len(geojson["features"])
            # Bonus: enrich each polygon with confidence + timing.
            timings = estimate_polygon_timings(binary, arrays)
            mean_prob_per_poly: dict[int, float] = {}
            from scipy.ndimage import label as cc_label

            comp_labels, n_comp = cc_label(binary)
            for cid in range(1, n_comp + 1):
                m = comp_labels == cid
                if m.any():
                    mean_prob_per_poly[cid] = float(prob[m].mean())
            # Order in raster_to_geojson follows shapely.shapes - re-derive
            # by re-vectorising and pairing with mean_prob ordered the
            # same way (best effort: identity map by polygon index).
            for i, feat in enumerate(geojson["features"]):
                cid = i + 1
                feat["properties"] = feat.get("properties") or {}
                feat["properties"]["confidence"] = mean_prob_per_poly.get(cid, None)
                feat["properties"]["time_step"] = timings.get(cid, None)
            with open(geojson_path, "w") as f:
                json.dump(geojson, f)
        except ValueError as exc:
            print(f"  [{tile_id}] geojson warning: {exc}")
            with open(geojson_path, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)
    else:
        with open(geojson_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)

    return {
        "tile_id": tile_id,
        "binary_pixels": int(binary.sum()),
        "polygons": int(n_polys),
        "probability_path": str(prob_path),
        "binary_path": str(bin_path),
        "geojson_path": str(geojson_path),
    }


def rethreshold_tile(
    tile_id: str,
    split: str,
    threshold: float,
    *,
    out_root: Path = paths.SUBMISSION_ROOT,
    min_area_ha: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Re-generate the binary raster + GeoJSON from an already-computed
    probability raster without re-running any model.

    Useful when tuning the threshold or applying an adaptive fallback:
    the expensive deep-learning forward passes are skipped entirely.
    """
    arrays, meta = cache.load_tile_cache(tile_id, split)
    rmeta = _ensure_meta(meta)
    prob_path = out_root / f"prob_{tile_id}.tif"
    if not prob_path.exists():
        raise FileNotFoundError(f"Probability raster missing: {prob_path}")

    with rasterio.open(prob_path) as src:
        prob = src.read(1).astype(np.float32) / 1000.0

    binary_raw = (prob >= threshold).astype(np.uint8)
    binary = clean_and_filter(binary_raw, rmeta["transform"], min_area_ha=min_area_ha)

    bin_path = out_root / f"pred_{tile_id}.tif"
    _write_tiff(bin_path, binary.astype(np.uint8), rmeta, dtype="uint8")

    geojson_path = out_root / f"pred_{tile_id}.geojson"
    n_polys = 0
    if binary.any():
        try:
            geojson = raster_to_geojson(
                bin_path, output_path=geojson_path, min_area_ha=min_area_ha
            )
            n_polys = len(geojson["features"])
            from scipy.ndimage import label as cc_label

            comp_labels, n_comp = cc_label(binary)
            mean_prob_per_poly: dict[int, float] = {}
            for cid in range(1, n_comp + 1):
                m = comp_labels == cid
                if m.any():
                    mean_prob_per_poly[cid] = float(prob[m].mean())
            timings = estimate_polygon_timings(binary, arrays)
            for i, feat in enumerate(geojson["features"]):
                cid = i + 1
                feat["properties"] = feat.get("properties") or {}
                feat["properties"]["confidence"] = mean_prob_per_poly.get(cid, None)
                feat["properties"]["time_step"] = timings.get(cid, None)
            with open(geojson_path, "w") as f:
                json.dump(geojson, f)
        except ValueError as exc:
            if verbose:
                print(f"  [{tile_id}] rethreshold geojson warning: {exc}")
            with open(geojson_path, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f)
    else:
        with open(geojson_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)

    if verbose:
        print(
            f"  [{tile_id}] rethreshold thr={threshold:.3f} "
            f"pixels={int(binary.sum()):,} polygons={n_polys}"
        )
    return {
        "tile_id": tile_id,
        "binary_pixels": int(binary.sum()),
        "polygons": int(n_polys),
        "probability_path": str(prob_path),
        "binary_path": str(bin_path),
        "geojson_path": str(geojson_path),
    }

"""Run final inference on test tiles and write submission GeoJSON.

Auto-discovers every trained model checkpoint:

* ``artifacts/models/lgbm_*.joblib``  -> LightGBM (LORO folds + full)
* ``artifacts/models/tcn_*.pt``       -> TCN (LORO folds + full)
* ``artifacts/models/unet_<tag>_*.pt`` -> one entry per encoder namespace

Within each namespace, predictions across all available checkpoints are
averaged at inference for variance reduction.

Threshold is chosen per test tile by region (Amazon / Asia tiles match
their LORO threshold; Africa falls back to the median).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from oasis import audit, paths, validation
from oasis.ensemble import CalibratedStack
from oasis.predict import PredictBundle, predict_tile, rethreshold_tile


_UNET_RE = re.compile(r"^(unet_[A-Za-z0-9_]+?)_(loro_[a-z]+|full)\.pt$")
_LGBM_RE = re.compile(r"^lgbm_(loro_[a-z]+|full(?:_[a-z]+)?)\.joblib$")
_TCN_RE = re.compile(r"^tcn_(loro_[a-z]+|full)\.pt$")


def _discover_checkpoints(models_root: Path) -> tuple[list[Path], list[Path], dict[str, list[Path]]]:
    lgbm: list[Path] = []
    tcn: list[Path] = []
    unets: dict[str, list[Path]] = defaultdict(list)
    if not models_root.exists():
        return lgbm, tcn, dict(unets)
    for p in sorted(models_root.iterdir()):
        if _LGBM_RE.match(p.name):
            lgbm.append(p)
        elif _TCN_RE.match(p.name):
            tcn.append(p)
        else:
            m = _UNET_RE.match(p.name)
            if m:
                unets[m.group(1)].append(p)
    return lgbm, tcn, dict(unets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", default="", help="Optional comma-separated subset of test tiles.")
    parser.add_argument("--lgbm-paths", default="", help="Comma-sep override.")
    parser.add_argument("--tcn-paths", default="", help="Comma-sep override.")
    parser.add_argument("--unet-paths", default="", help="Comma-sep override (use namespace::path).")
    parser.add_argument("--stack", type=Path, default=paths.MODELS_ROOT / "stack.joblib")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Single threshold; if omitted, per-tile region thresholds are used.")
    parser.add_argument("--threshold-json", type=Path,
                        default=paths.OOF_ROOT / "stack" / "threshold.json")
    parser.add_argument("--postprocess-json", type=Path,
                        default=paths.OOF_ROOT / "stack" / "postprocess.json")
    parser.add_argument("--min-area-ha", type=float, default=None)
    parser.add_argument("--no-external", action="store_true")
    parser.add_argument("--out-root", type=Path, default=paths.SUBMISSION_ROOT)
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable UNet test-time augmentation (faster CPU inference).")
    parser.add_argument("--multiscale", default="0.75,1.0,1.25",
                        help="Comma-separated UNet multiscale factors. Use '1.0' for single-scale.")
    parser.add_argument("--stride-div", type=int, default=4,
                        help="UNet sliding-window stride = patch_size // stride_div. Lower => faster.")
    parser.add_argument("--adaptive-threshold", action="store_true", default=True,
                        help="If region threshold yields 0 polygons, retry with a tile-specific percentile.")
    parser.add_argument("--no-adaptive-threshold", dest="adaptive_threshold", action="store_false")
    parser.add_argument("--adaptive-percentile", type=float, default=None,
                        help="Percentile of tile's probability distribution used for adaptive retry.")
    parser.add_argument("--africa-stack-region", choices=("global", "amazon", "asia"), default="global",
                        help="Regional stacker override for Africa test tiles.")
    parser.add_argument("--africa-threshold-region", choices=("global", "amazon", "asia"), default="global",
                        help="Threshold-source override for Africa test tiles.")
    args = parser.parse_args()

    paths.ensure_dirs()
    _, test_tiles = audit.audit(strict=False)
    if args.tiles:
        selected = {tile.strip() for tile in args.tiles.split(",") if tile.strip()}
        test_tiles = [tile for tile in test_tiles if tile in selected]

    lgbm, tcn, unets = _discover_checkpoints(paths.MODELS_ROOT)

    if args.lgbm_paths:
        lgbm = [Path(p) for p in args.lgbm_paths.split(",") if p]
    if args.tcn_paths:
        tcn = [Path(p) for p in args.tcn_paths.split(",") if p]
    if args.unet_paths:
        overridden_unets: dict[str, list[Path]] = defaultdict(list)
        for tok in args.unet_paths.split(","):
            tok = tok.strip()
            if not tok:
                continue
            ns, _, p = tok.partition("::")
            overridden_unets[ns].append(Path(p))
        unets = dict(overridden_unets)

    active_names: set[str] | None = None
    if args.stack.exists():
        stack = CalibratedStack.load(args.stack)
        active_names = set(stack.model_names)
        for region_models in stack.regional_model_names.values():
            active_names.update(region_models)
    if active_names:
        lgbm = lgbm if "lgbm" in active_names else []
        tcn = tcn if "tcn" in active_names else []
        unets = {ns: ps for ns, ps in unets.items() if ns in active_names}
        print(f"Active stack model namespaces: {sorted(active_names)}")

    print("Discovered checkpoints:")
    print(f"  lgbm  ({len(lgbm)}): {[p.name for p in lgbm]}")
    print(f"  tcn   ({len(tcn)}): {[p.name for p in tcn]}")
    for ns, ps in unets.items():
        print(f"  {ns} ({len(ps)}): {[p.name for p in ps]}")

    # Threshold resolution.
    threshold_data = None
    if args.threshold_json.exists():
        with open(args.threshold_json) as f:
            threshold_data = json.load(f)
    postprocess_data = None
    if args.postprocess_json.exists():
        with open(args.postprocess_json) as f:
            postprocess_data = json.load(f)

    if args.threshold is not None:
        global_threshold = float(args.threshold)
        region_thresholds: dict[str, float] = {}
    elif threshold_data is not None:
        global_threshold = float(threshold_data.get("test_threshold", 0.5))
        region_thresholds = {k: float(v) for k, v in threshold_data.get("region_thresholds", {}).items()}
    else:
        global_threshold = 0.5
        region_thresholds = {}

    postprocess_global = (postprocess_data or {}).get("global", {})
    postprocess_regions = (postprocess_data or {}).get("regions", {})

    print(f"Global fallback threshold: {global_threshold:.3f}")
    if region_thresholds:
        print(f"Region thresholds: {region_thresholds}")
    if postprocess_regions:
        print(f"Postprocess regions: {sorted(postprocess_regions)}")

    multiscale = tuple(float(x) for x in args.multiscale.split(",") if x.strip())
    base_bundle = PredictBundle(
        lgbm_paths=lgbm,
        tcn_paths=tcn,
        unet_paths=unets,
        stack_path=args.stack if args.stack.exists() else None,
        threshold=global_threshold,
        min_area_ha=args.min_area_ha if args.min_area_ha is not None else float(postprocess_global.get("min_area_ha", 0.5)),
        use_external=not args.no_external,
        stack_region_overrides={},
        unet_tta=not args.no_tta,
        unet_multiscale=multiscale,
        unet_stride_div=args.stride_div,
    )

    results = []
    for tile in test_tiles:
        region = validation.region_of(tile)
        stack_region = region
        threshold_region = region
        if region == "africa":
            stack_region = args.africa_stack_region
            threshold_region = args.africa_threshold_region
        region_cfg = postprocess_global if threshold_region == "global" else postprocess_regions.get(threshold_region, {})
        thr = (
            float(args.threshold)
            if args.threshold is not None
            else float(region_cfg.get("threshold", region_thresholds.get(threshold_region, global_threshold)))
        )
        min_area_ha = (
            float(args.min_area_ha)
            if args.min_area_ha is not None
            else float(region_cfg.get("min_area_ha", postprocess_global.get("min_area_ha", 0.5)))
        )
        adaptive_percentile = (
            float(args.adaptive_percentile)
            if args.adaptive_percentile is not None
            else float(region_cfg.get("adaptive_percentile", postprocess_global.get("adaptive_percentile", 99.0)))
        )
        print(
            f"\n=== predict {tile} (region={region}, stack_region={stack_region}, "
            f"threshold_region={threshold_region}, thr={thr:.3f}, "
            f"min_area_ha={min_area_ha:.2f}, adaptive_pctl={adaptive_percentile:.2f}) ==="
        )
        tile_bundle = replace(
            base_bundle,
            threshold=thr,
            min_area_ha=min_area_ha,
            stack_region_overrides={tile: stack_region} if stack_region != region else {},
        )
        res = predict_tile(tile, "test", tile_bundle, out_root=args.out_root)

        # Adaptive fallback: if the region threshold killed every polygon the
        # calibration is off for this tile; re-threshold the already-computed
        # probability raster at the tile's own percentile (no model rerun).
        if args.adaptive_threshold and adaptive_percentile > 0 and res["polygons"] == 0:
            import numpy as np
            import rasterio

            prob_path = args.out_root / f"prob_{tile}.tif"
            if prob_path.exists():
                with rasterio.open(prob_path) as src:
                    prob = src.read(1).astype(np.float32) / 1000.0
                pct_thr = float(np.quantile(prob, adaptive_percentile / 100.0))
                if pct_thr < thr - 0.05 and pct_thr > 0.3:
                    print(
                        f"  [adaptive] region thr={thr:.2f} -> 0 polys; "
                        f"rethresholding at p{adaptive_percentile:.1f}={pct_thr:.3f}"
                    )
                    res = rethreshold_tile(
                        tile, "test", pct_thr,
                        out_root=args.out_root, min_area_ha=min_area_ha,
                    )

        results.append(res)

    print("\nDone:")
    for r in results:
        print(f"  {r['tile_id']}: pixels={r['binary_pixels']:,} polygons={r['polygons']}")


if __name__ == "__main__":
    main()

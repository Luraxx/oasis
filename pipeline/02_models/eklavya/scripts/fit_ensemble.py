"""Fit per-model isotonic calibration + logistic stacker on OOF probabilities,
then pick the test-time threshold as the median of region-best thresholds.

Inputs:
* artifacts/oof/{model}/{tile}.npy

Outputs:
* artifacts/models/stack.joblib
* artifacts/oof/stack/threshold.json   (per-region + chosen median)
* artifacts/oof/stack/{tile}.npy       (calibrated stacked OOF prob)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from oasis import audit, cache, paths, validation
from oasis.ensemble import CalibratedStack, fit_calibration, fit_stacker
from oasis.metrics import best_threshold_f1


def _discover_models(explicit: list[str] | None = None) -> list[str]:
    """Find every directory under artifacts/oof/ that has at least one .npy.

    Excludes ``stack`` (the post-stacking output dir).
    """
    if explicit:
        return explicit
    root = paths.OOF_ROOT
    if not root.exists():
        return []
    out = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name in ("stack",):
            continue
        if any(sub.glob("*.npy")):
            out.append(sub.name)
    return out


def _load_oof(model: str, tile: str) -> np.ndarray | None:
    p = paths.OOF_ROOT / model / f"{tile}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def _load_model_scores(models: list[str]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for model in models:
        summary_path = paths.OOF_ROOT / model / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        by_region: dict[str, list[float]] = defaultdict(list)
        for tile, info in summary.get("tiles", {}).items():
            if "f1" not in info:
                continue
            by_region[validation.region_of(tile)].append(float(info["f1"]))
        if not by_region:
            continue
        scores[model] = {
            region: float(np.mean(vals)) for region, vals in by_region.items() if vals
        }
        all_vals = [v for vals in by_region.values() for v in vals]
        scores[model]["all"] = float(np.mean(all_vals))
    return scores


def _select_top_models(
    models: list[str],
    scores: dict[str, dict[str, float]],
    *,
    region: str,
    top_k: int,
) -> list[str]:
    ranked = sorted(
        models,
        key=lambda m: (
            scores.get(m, {}).get(region, float("-inf")),
            scores.get(m, {}).get("all", float("-inf")),
            m,
        ),
        reverse=True,
    )
    picked = ranked[: max(1, min(top_k, len(ranked)))]
    return picked or list(models)


def _parse_region_models(spec: str, *, models: list[str]) -> dict[str, list[str]]:
    if not spec.strip():
        return {}
    allowed = set(models)
    out: dict[str, list[str]] = {}
    for chunk in spec.split(";"):
        item = chunk.strip()
        if not item:
            continue
        region, sep, raw_models = item.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid --region-models entry '{item}'. Expected region=model1,model2"
            )
        region = region.strip()
        selected = [m.strip() for m in raw_models.split(",") if m.strip()]
        if not selected:
            raise ValueError(f"No models listed for region '{region}'")
        unknown = [m for m in selected if m not in allowed]
        if unknown:
            raise ValueError(f"Unknown models for region '{region}': {unknown}")
        out[region] = selected
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to stack. Empty = auto-discover from artifacts/oof/.",
    )
    parser.add_argument(
        "--region-models",
        default="",
        help="Optional semicolon-separated overrides, e.g. amazon=lgbm,unet_mit_b1;asia=lgbm,tcn,unet_mit_b1",
    )
    parser.add_argument("--no-region-stackers", action="store_true")
    parser.add_argument("--global-top-k", type=int, default=3)
    parser.add_argument("--region-top-k", type=int, default=3)
    parser.add_argument("--no-auto-model-selection", action="store_true")
    args = parser.parse_args()

    paths.ensure_dirs()
    train_tiles, _ = audit.audit(strict=False)
    explicit = [m for m in args.models.split(",") if m] if args.models else None
    models = _discover_models(explicit)
    if not models:
        raise RuntimeError(
            "No OOF model directories found under artifacts/oof/. "
            "Run scripts/train_*.py first to populate OOF predictions."
        )

    print(f"Stacking models: {models}")
    region_model_overrides = _parse_region_models(args.region_models, models=models)
    if region_model_overrides:
        print(f"Region model overrides: {region_model_overrides}")
    model_scores = _load_model_scores(models)
    if model_scores:
        print(f"Loaded OOF summary scores for: {sorted(model_scores)}")
    if args.no_auto_model_selection:
        global_models = list(models)
    else:
        global_models = _select_top_models(
            models, model_scores, region="all", top_k=args.global_top_k
        )
    print(f"Global stack models: {global_models}")

    # Concatenate consensus pixels across tiles for calibration / stacker fit.
    per_model_p: dict[str, list[np.ndarray]] = {m: [] for m in models}
    y_all: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    tile_meta: dict[str, dict] = {}

    for tile in train_tiles:
        arrays, _ = cache.load_tile_cache(tile, "train")
        mask = arrays["consensus_pos"] | arrays["consensus_neg"]
        labels = arrays["consensus_pos"].astype(np.uint8)
        if not mask.any():
            continue

        probs_for_tile = {}
        for m in models:
            p = _load_oof(m, tile)
            if p is None:
                continue
            probs_for_tile[m] = p
        if len(probs_for_tile) != len(models):
            print(f"  skip {tile}: missing OOF for {set(models) - set(probs_for_tile)}")
            continue

        for m in models:
            per_model_p[m].append(probs_for_tile[m][mask])
        y_all.append(labels[mask])
        masks.append(mask)
        tile_meta[tile] = {"region": validation.region_of(tile), "n_pix": int(mask.sum())}

    if not y_all:
        raise RuntimeError("No OOF data found - run scripts/train_*.py first")

    y_concat = np.concatenate(y_all)
    print(f"Total consensus pixels: {len(y_concat):,} (pos={int(y_concat.sum()):,})")

    stack = CalibratedStack(model_names=list(global_models))
    for m in models:
        p = np.concatenate(per_model_p[m])
        stack.calibrators[m] = fit_calibration(p, y_concat, np.ones_like(y_concat, dtype=bool))
        print(f"  calibrated {m}")

    cal_concat = np.stack(
        [stack.calibrators[m].predict(np.concatenate(per_model_p[m])) for m in global_models], axis=0
    )
    stacker = fit_stacker(
        cal_concat[:, :, None],  # (M, N, 1) -> caller takes flat after mask
        y_concat[:, None],
        np.ones_like(y_concat[:, None], dtype=bool),
    )
    stack.stacker = stacker
    if not args.no_region_stackers:
        regions = sorted({info["region"] for info in tile_meta.values() if info["region"] != "unknown"})
        for region in regions:
            selected_models = region_model_overrides.get(region)
            if selected_models is None:
                if args.no_auto_model_selection:
                    selected_models = list(global_models)
                else:
                    selected_models = _select_top_models(
                        models, model_scores, region=region, top_k=args.region_top_k
                    )
            per_region_p: dict[str, list[np.ndarray]] = {m: [] for m in selected_models}
            y_region: list[np.ndarray] = []
            for tile, info in tile_meta.items():
                if info["region"] != region:
                    continue
                arrays, _ = cache.load_tile_cache(tile, "train")
                mask = arrays["consensus_pos"] | arrays["consensus_neg"]
                labels = arrays["consensus_pos"].astype(np.uint8)
                if not mask.any():
                    continue
                y_region.append(labels[mask])
                for m in selected_models:
                    p = _load_oof(m, tile)
                    if p is None:
                        raise RuntimeError(f"Missing OOF for model={m} tile={tile}")
                    per_region_p[m].append(stack.calibrators[m].predict(p[mask]))
            if not y_region:
                continue
            y_concat_region = np.concatenate(y_region)
            cal_concat_region = np.stack(
                [np.concatenate(per_region_p[m]) for m in selected_models], axis=0
            )
            stack.regional_model_names[region] = list(selected_models)
            stack.regional_stackers[region] = fit_stacker(
                cal_concat_region[:, :, None],
                y_concat_region[:, None],
                np.ones_like(y_concat_region[:, None], dtype=bool),
            )
            print(f"  fitted regional stacker {region}: models={selected_models}")

    stack_path = paths.MODELS_ROOT / "stack.joblib"
    stack.save(stack_path)
    print(f"  saved {stack_path}")

    # Per-region thresholds.
    (paths.OOF_ROOT / "stack").mkdir(parents=True, exist_ok=True)
    region_thresholds: dict[str, float] = {}
    region_ys: dict[str, list[np.ndarray]] = defaultdict(list)
    region_ps: dict[str, list[np.ndarray]] = defaultdict(list)

    for tile, info in tile_meta.items():
        arrays, _ = cache.load_tile_cache(tile, "train")
        mask = arrays["consensus_pos"] | arrays["consensus_neg"]
        labels = arrays["consensus_pos"].astype(np.uint8)
        probs = {m: _load_oof(m, tile) for m in models}
        stacked = stack.stack(probs, region=info["region"])
        region_ys[info["region"]].append(labels[mask])
        region_ps[info["region"]].append(stacked[mask])
        np.save(paths.OOF_ROOT / "stack" / f"{tile}.npy", stacked.astype(np.float16))

    for region, ys in region_ys.items():
        y = np.concatenate(ys)
        p = np.concatenate(region_ps[region])
        t, rep = best_threshold_f1(y, p)
        region_thresholds[region] = t
        print(f"  region {region}: best_t={t:.2f} f1={rep.f1:.4f}")

    median_t = float(np.median(list(region_thresholds.values())))
    print(f"\nChosen test-time threshold (median of region thresholds): {median_t:.2f}")

    out = paths.OOF_ROOT / "stack" / "threshold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "models": models,
            "region_thresholds": region_thresholds,
            "test_threshold": median_t,
            "region_models": stack.regional_model_names,
        }, f, indent=2)


if __name__ == "__main__":
    main()

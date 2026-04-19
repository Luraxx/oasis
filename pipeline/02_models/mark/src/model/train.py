"""Train a LightGBM per-pixel deforestation classifier with tile-holdout validation.

Strategy:
  * Split: 13 train tiles, 3 val tiles — chosen for biome + intensity diversity.
  * Training target: y_hard (strict 2-of-available consensus) — binary.
  * Positives: all kept. Negatives: sub-sampled 10:1 per tile.
  * Model: LightGBM gradient-boosted decision trees, ~500 rounds with early stopping.
  * Evaluation on val: reconstruct per-tile predictions on the **full** forest_2020
    mask (no sub-sampling) and compute pixel IoU at several thresholds.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.data.canonical_grid import all_tile_ids, grid_for
from src.model.dataset import TileBundle, concat_bundles, load_tile

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path("/shared-docker/oasis-mark-2")
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


# Val tiles chosen for diversity (different regions, different pos rates).
VAL_TILES = [
    "18NXH_6_8",   # Colombia, very high pos rate (~32%), 95% forest baseline
    "47QQV_2_4",   # SE Asia, very low pos rate (~5%), clean test
    "48PYB_3_6",   # SE Asia, medium pos rate (~21%), partially forested
]


def lgbm_params(scale_pos_weight: float = 1.0) -> dict:
    return dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=63,           # smaller trees → much faster
        max_depth=-1,
        min_data_in_leaf=500,    # more regularization, fewer splits
        feature_fraction=0.5,    # subsample features per tree for speed + reg
        bagging_fraction=0.7,
        bagging_freq=5,
        lambda_l2=1.0,
        scale_pos_weight=scale_pos_weight,
        num_threads=20,
        verbose=-1,
    )


def iou(pred_bin: np.ndarray, y: np.ndarray) -> float:
    inter = int((pred_bin & y.astype(bool)).sum())
    union = int((pred_bin | y.astype(bool)).sum())
    return inter / union if union else 0.0


def per_tile_evaluate(model: lgb.Booster, val_bundles: list[TileBundle],
                      thresholds: list[float]) -> dict:
    """Evaluate on each val tile at multiple thresholds. Returns dict of metrics."""
    results = {"per_tile": {}, "per_threshold": {}}
    all_preds_by_thr: dict[float, list] = {t: [] for t in thresholds}
    all_ys = []

    for b in val_bundles:
        probs = model.predict(b.X)
        # Reconstruct full-tile prediction map for per-tile IoU computation
        H, W = b.shape
        pred_map = np.zeros(H * W, dtype=np.float32)
        pred_map[b.pixel_index] = probs
        pred_map = pred_map.reshape(H, W)

        # Load full labels (y_hard) for this tile on the full forest mask
        # Since b already has the full forest mask, and positives are fully kept in y_hard,
        # we need to load the full tile WITHOUT sub-sampling to evaluate properly.
        full_b = load_tile(b.tile, sample_neg_ratio=None)
        # full_b has ALL forest pixels. Extract pred at those indices.
        full_probs = pred_map.ravel()[full_b.pixel_index]
        full_y = full_b.y_hard.astype(bool)

        results["per_tile"][b.tile] = {}
        for t in thresholds:
            bin_pred = full_probs > t
            v = iou(bin_pred, full_y)
            results["per_tile"][b.tile][f"iou_t{t:.2f}"] = round(v, 4)
            all_preds_by_thr[t].append(bin_pred)
        all_ys.append(full_y)

    # Aggregate across tiles (micro-IoU: concatenate first)
    concat_y = np.concatenate(all_ys)
    for t in thresholds:
        concat_p = np.concatenate(all_preds_by_thr[t])
        results["per_threshold"][f"iou_t{t:.2f}"] = round(float(iou(concat_p, concat_y)), 4)

    # macro-IoU: mean of per-tile IoUs
    for t in thresholds:
        vals = [results["per_tile"][ti][f"iou_t{t:.2f}"] for ti in results["per_tile"]]
        results["per_threshold"][f"iou_t{t:.2f}_macro"] = round(float(np.mean(vals)), 4)

    return results


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-tiles", action="store_true",
                    help="Train on all 16 train tiles (no val holdout). Used for final submission.")
    ap.add_argument("--tile-uniform-weight", action="store_true",
                    help="Per-tile uniform sample weights (each tile contributes equally).")
    ap.add_argument("--num-rounds", type=int, default=300)
    ap.add_argument("--out", type=str, default="lgbm_baseline.txt",
                    help="Model file name under models/")
    args = ap.parse_args()

    all_train = [t for t in all_tile_ids() if grid_for(t).split == "train"]
    if args.all_tiles:
        train_tiles = sorted(all_train)
        val_tiles = []
    else:
        train_tiles = [t for t in all_train if t not in VAL_TILES]
        val_tiles = VAL_TILES

    log.info(f"Train tiles ({len(train_tiles)}): {train_tiles}")
    log.info(f"Val tiles ({len(val_tiles)}): {val_tiles}")
    log.info(f"tile_uniform_weight={args.tile_uniform_weight}  num_rounds={args.num_rounds}")

    rng = np.random.default_rng(42)

    t0 = time.time()
    log.info("Loading train bundles...")
    train_bundles = [load_tile(t, sample_neg_ratio=10.0, rng=rng,
                               tile_uniform_weight=args.tile_uniform_weight)
                     for t in train_tiles]
    X_train, yh_train, ys_train, names, _, w_train = concat_bundles(train_bundles)
    log.info(f"Train matrix: {X_train.shape}  positives: {yh_train.sum()} "
             f"({100*yh_train.mean():.2f}%)  weighted={w_train is not None}  "
             f"({time.time() - t0:.1f}s)")

    val_bundles = []
    X_val = yh_val = w_val = None
    if val_tiles:
        log.info("Loading val bundles (full forest pixels, no subsample)...")
        val_bundles = [load_tile(t, sample_neg_ratio=None) for t in val_tiles]
        X_val, yh_val, _, _, _, w_val = concat_bundles(val_bundles)
        log.info(f"Val matrix: {X_val.shape}  positives: {yh_val.sum()} ({100*yh_val.mean():.2f}%)")

    params = lgbm_params(scale_pos_weight=1.0)
    log.info(f"LGBM params: {params}")

    dtrain = lgb.Dataset(X_train, label=yh_train, feature_name=names, weight=w_train)
    valid_sets = [dtrain]
    valid_names = ["train"]
    if X_val is not None:
        dval = lgb.Dataset(X_val, label=yh_val, feature_name=names, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("val")

    log.info("Training...")
    callbacks = [lgb.log_evaluation(20)]
    if X_val is not None:
        callbacks.insert(0, lgb.early_stopping(30))
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=args.num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    log.info(f"Training done in {time.time() - t0:.1f}s")
    log.info(f"Best iteration: {booster.best_iteration}")

    # Save
    model_path = MODELS / args.out
    booster.save_model(str(model_path))
    log.info(f"Saved model: {model_path}")

    # Feature importance always
    importance = booster.feature_importance(importance_type="gain")
    imp_sorted = sorted(zip(names, importance), key=lambda x: -x[1])[:25]
    log.info("Top feature importances (gain):")
    for n, v in imp_sorted:
        log.info(f"  {n:<40s} {v:10.1f}")

    if val_tiles:
        log.info("Evaluating on validation tiles at multiple thresholds...")
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
        results = per_tile_evaluate(booster, val_bundles, thresholds)
        results["top_features"] = [(n, float(v)) for n, v in imp_sorted]
        metrics_path = MODELS / args.out.replace(".txt", "_metrics.json")
        with open(metrics_path, "w") as fp:
            json.dump(results, fp, indent=2)
        log.info("\nPER-TILE IOU:")
        for tile, scores in results["per_tile"].items():
            log.info(f"  {tile}: {scores}")
        log.info("\nAGGREGATE (macro=mean of per-tile IoU; micro=concat pixels):")
        for k, v in results["per_threshold"].items():
            log.info(f"  {k}: {v}")
    else:
        log.info("[--all-tiles mode] no validation; skipping IoU evaluation.")
        metrics_path = MODELS / args.out.replace(".txt", "_features.json")
        with open(metrics_path, "w") as fp:
            json.dump({"top_features": [(n, float(v)) for n, v in imp_sorted]}, fp, indent=2)


if __name__ == "__main__":
    main()

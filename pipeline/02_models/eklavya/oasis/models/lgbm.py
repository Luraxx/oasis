"""LightGBM tabular trainer over the per-pixel feature pack.

Workflow:

* For each training tile, materialise the (C, H, W) feature pack and
  sample (N, 2) pixel coordinates via ``oasis.sampling.sample_pixels``.
* Stack across tiles into one design matrix.
* Fit ``LGBMClassifier`` with sample weights = label-confidence.
* Optionally produce a (H, W) probability raster per tile via
  ``predict_tile``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from oasis import cache, paths, sampling
from oasis.features.pack import assemble_feature_pack


@dataclass
class LGBMConfig:
    samples_per_tile: int = 30_000
    pos_neg_ratio: float = 1.5
    num_leaves: int = 255
    max_depth: int = -1
    learning_rate: float = 0.03
    n_estimators: int = 2_000
    min_child_samples: int = 50
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    early_stopping_rounds: int = 100
    random_state: int = 42
    n_jobs: int = -1
    include_external: bool = True
    feature_names: list[str] = field(default_factory=list)


@dataclass
class TrainingMatrix:
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    feature_names: list[str]


def model_uses_external(feature_names: list[str]) -> bool:
    return any(name.startswith("ext_") for name in feature_names)


def tile_has_external_data(tile_id: str, split: str) -> bool:
    ext_dir = paths.EXTERNALS_ROOT / split / tile_id
    return ext_dir.exists() and any(ext_dir.glob("*.npy"))


def resolve_external_usage(
    tiles: list[str],
    *,
    split: str,
    mode: str | bool | None = "auto",
    verbose: bool = True,
) -> bool:
    if isinstance(mode, bool):
        return mode
    resolved = (mode or "auto").strip().lower()
    if resolved not in {"auto", "on", "off"}:
        raise ValueError(f"Unsupported external mode: {mode}")
    if resolved == "on":
        return True
    if resolved == "off":
        return False
    missing = [tile for tile in tiles if not tile_has_external_data(tile, split)]
    if missing:
        if verbose:
            sample = ", ".join(missing[:3])
            more = "" if len(missing) <= 3 else f", +{len(missing) - 3} more"
            print(
                f"  externals: auto -> OFF ({len(missing)}/{len(tiles)} tiles missing cached layers: "
                f"{sample}{more})"
            )
        return False
    if verbose:
        print(f"  externals: auto -> ON ({len(tiles)} tiles with cached external layers)")
    return True


def build_training_matrix(
    fit_tiles: list[str],
    *,
    cfg: LGBMConfig,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> TrainingMatrix:
    rng = rng or np.random.default_rng(cfg.random_state)
    Xs, ys, ws = [], [], []
    feature_names: list[str] = []

    for tile_id in tqdm(fit_tiles, desc="building train matrix", disable=not verbose):
        arrays, _ = cache.load_tile_cache(tile_id, "train")
        feats, names = assemble_feature_pack(
            arrays, tile_id=tile_id, split="train", include_external=cfg.include_external
        )
        if not feature_names:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError(
                f"Feature schema mismatch on {tile_id}: {len(names)} vs {len(feature_names)}"
            )

        coords = sampling.sample_pixels(
            labels=arrays["labels"],
            valid_mask=arrays["valid_mask"],
            sample_weight=arrays["sample_weight"],
            consensus_pos=arrays["consensus_pos"],
            n_samples=cfg.samples_per_tile,
            pos_neg_ratio=cfg.pos_neg_ratio,
            rng=rng,
        )
        if len(coords) == 0:
            if verbose:
                print(f"  [{tile_id}] no usable pixels, skipping")
            continue
        rows, cols = coords[:, 0], coords[:, 1]
        Xs.append(feats[:, rows, cols].T.astype(np.float32))
        ys.append(arrays["labels"][rows, cols].astype(np.uint8))
        ws.append(np.maximum(arrays["sample_weight"][rows, cols], 1e-3).astype(np.float32))

    if not Xs:
        raise RuntimeError("No training data assembled")

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if verbose:
        print(
            f"  matrix: X={X.shape} y_pos={int(y.sum())} y_neg={int(len(y) - y.sum())} "
            f"weight_mean={w.mean():.3f}"
        )
    return TrainingMatrix(X=X, y=y, w=w, feature_names=feature_names)


def fit_lgbm(
    train: TrainingMatrix,
    *,
    val: TrainingMatrix | None = None,
    cfg: LGBMConfig,
    verbose: bool = True,
) -> lgb.LGBMClassifier:
    callbacks = []
    if val is not None and cfg.early_stopping_rounds:
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=verbose))
    if verbose:
        callbacks.append(lgb.log_evaluation(period=100))
    clf = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        min_child_samples=cfg.min_child_samples,
        feature_fraction=cfg.feature_fraction,
        bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        class_weight="balanced",
        verbose=-1,
    )
    fit_kwargs = {"sample_weight": train.w}
    if val is not None:
        fit_kwargs["eval_set"] = [(val.X, val.y)]
        fit_kwargs["eval_sample_weight"] = [val.w]
        fit_kwargs["eval_metric"] = ["binary_logloss", "auc"]
        fit_kwargs["callbacks"] = callbacks
    clf.fit(train.X, train.y, **fit_kwargs)
    return clf


def predict_tile(
    model: lgb.LGBMClassifier,
    tile_id: str,
    split: str,
    *,
    include_external: bool | None = True,
    feature_names: list[str] | None = None,
    chunk_size: int = 250_000,
) -> np.ndarray:
    if feature_names is not None:
        include_external = model_uses_external(feature_names)
    arrays, _ = cache.load_tile_cache(tile_id, split)
    feats, names = assemble_feature_pack(
        arrays, tile_id=tile_id, split=split, include_external=include_external
    )
    if feature_names is not None and names != feature_names:
        raise RuntimeError(
            f"Feature schema mismatch for {tile_id} ({split}): "
            f"expected {len(feature_names)} features, got {len(names)}"
        )
    c, h, w = feats.shape
    flat = feats.reshape(c, -1).T.astype(np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(flat.shape[0], dtype=np.float32)
    for s in range(0, flat.shape[0], chunk_size):
        e = min(s + chunk_size, flat.shape[0])
        out[s:e] = model.predict_proba(flat[s:e])[:, 1].astype(np.float32)
    return out.reshape(h, w)


def save_model(model: lgb.LGBMClassifier, path: Path, feature_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "uses_external": model_uses_external(feature_names),
        },
        path,
    )


def load_model(path: Path) -> tuple[lgb.LGBMClassifier, list[str]]:
    payload = joblib.load(path)
    return payload["model"], payload["feature_names"]

"""Train LightGBM with leave-one-region-out validation + OOF stacks.

Outputs:

* artifacts/models/lgbm_full.joblib              (trained on all 16 tiles)
* artifacts/models/lgbm_loro_{region}.joblib     (one per region fold)
* artifacts/oof/lgbm/{tile}.npy                  (OOF probability rasters)
* artifacts/oof/lgbm/summary.json                (per-tile metrics)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from oasis import audit, cache, paths, validation
from oasis.metrics import best_threshold_f1, binary_report
from oasis.models import lgbm as lgbm_mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-tile", type=int, default=30_000)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--external-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--skip-oof", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()

    paths.ensure_dirs()
    train_tiles, _ = audit.audit(strict=False)
    use_external = lgbm_mod.resolve_external_usage(
        train_tiles, split="train", mode=args.external_mode
    )

    cfg = lgbm_mod.LGBMConfig(
        samples_per_tile=args.samples_per_tile,
        n_estimators=args.n_estimators,
        include_external=use_external,
    )

    oof_dir = paths.OOF_ROOT / "lgbm"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"folds": [], "tiles": {}}

    if not args.skip_oof:
        for split in validation.loro_splits(train_tiles):
            print(f"\n=== LORO fold {split.name} ===")
            print(f"train: {split.train}")
            print(f"val  : {split.val}")
            train_mat = lgbm_mod.build_training_matrix(split.train, cfg=cfg)
            val_mat = lgbm_mod.build_training_matrix(split.val, cfg=cfg)
            clf = lgbm_mod.fit_lgbm(train_mat, val=val_mat, cfg=cfg)

            fold_path = paths.MODELS_ROOT / f"lgbm_{split.name}.joblib"
            lgbm_mod.save_model(clf, fold_path, train_mat.feature_names)
            summary["folds"].append({"name": split.name, "n_train": int(len(train_mat.y)), "n_val": int(len(val_mat.y))})

            for tile in split.val:
                arrays, _ = cache.load_tile_cache(tile, "train")
                prob = lgbm_mod.predict_tile(
                    clf,
                    tile,
                    "train",
                    include_external=cfg.include_external,
                    feature_names=train_mat.feature_names,
                )
                np.save(oof_dir / f"{tile}.npy", prob.astype(np.float16))

                mask = arrays["consensus_pos"] | arrays["consensus_neg"]
                if mask.sum() > 0:
                    y = arrays["consensus_pos"].astype(np.uint8)[mask]
                    p = prob[mask]
                    t, rep = best_threshold_f1(y, p)
                    summary["tiles"][tile] = {"fold": split.name, "best_threshold": t, **rep.to_dict()}
                    print(f"  [{tile}] best_t={t:.2f} f1={rep.f1:.4f} prec={rep.precision:.4f} rec={rep.recall:.4f}")

        with open(oof_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    if not args.skip_final:
        print("\n=== Final LightGBM on all tiles ===")
        full_mat = lgbm_mod.build_training_matrix(train_tiles, cfg=cfg)
        clf = lgbm_mod.fit_lgbm(full_mat, val=None, cfg=cfg)
        full_path = paths.MODELS_ROOT / "lgbm_full.joblib"
        lgbm_mod.save_model(clf, full_path, full_mat.feature_names)
        print(f"  saved {full_path}")


if __name__ == "__main__":
    main()

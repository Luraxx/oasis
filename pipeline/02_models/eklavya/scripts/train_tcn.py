"""Train the per-pixel Temporal CNN with LORO + OOF stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from oasis import audit, cache, paths, validation
from oasis.metrics import best_threshold_f1
from oasis.models import tcn as tcn_mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-tile", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-oof", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()

    paths.ensure_dirs()
    train_tiles, _ = audit.audit(strict=False)

    cfg = tcn_mod.TCNConfig(
        samples_per_tile=args.samples_per_tile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    oof_dir = paths.OOF_ROOT / "tcn"
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"folds": [], "tiles": {}}

    if not args.skip_oof:
        for split in validation.loro_splits(train_tiles):
            print(f"\n=== TCN LORO fold {split.name} ===")
            train_ds = tcn_mod.build_tcn_dataset(split.train, cfg=cfg)
            val_ds = tcn_mod.build_tcn_dataset(split.val, cfg=cfg)
            model = tcn_mod.train_tcn(train_ds, val_ds, cfg)

            fold_path = paths.MODELS_ROOT / f"tcn_{split.name}.pt"
            tcn_mod.save_tcn(model, fold_path, cfg)
            summary["folds"].append({"name": split.name, "n_train": int(len(train_ds)), "n_val": int(len(val_ds))})

            for tile in split.val:
                prob = tcn_mod.predict_tile(model, tile, "train")
                np.save(oof_dir / f"{tile}.npy", prob.astype(np.float16))
                arrays, _ = cache.load_tile_cache(tile, "train")
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
        print("\n=== Final TCN on all tiles ===")
        full_ds = tcn_mod.build_tcn_dataset(train_tiles, cfg=cfg)
        model = tcn_mod.train_tcn(full_ds, None, cfg)
        tcn_mod.save_tcn(model, paths.MODELS_ROOT / "tcn_full.pt", cfg)


if __name__ == "__main__":
    main()

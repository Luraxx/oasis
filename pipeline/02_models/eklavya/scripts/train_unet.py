"""Train the multi-temporal U-Net with LORO + OOF stacks.

Supports a ``--encoder-list`` flag for the encoder-diversity ensemble:
each entry trains its own LORO + final pass and writes its OOF / models
under a unique ``unet_<short_tag>`` namespace so the ensemble fitter can
pick all of them up automatically.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from oasis import audit, cache, paths, validation
from oasis.metrics import best_threshold_f1
from oasis.models import unet as unet_mod


def short_tag(encoder: str) -> str:
    """Map e.g. ``efficientnet-b3`` -> ``eb3``, ``resnet34`` -> ``r34``."""
    if not encoder:
        return "unk"
    if encoder.startswith("efficientnet-"):
        return "e" + encoder.split("-", 1)[1]  # e.g. eb3
    if encoder.startswith("resnet"):
        return "r" + encoder.replace("resnet", "")  # e.g. r34
    if encoder.startswith("mit_"):
        return "mit_" + encoder.split("_", 1)[1]
    if encoder.startswith("se_"):
        return encoder
    cleaned = re.sub(r"[^a-z0-9]+", "_", encoder.lower())
    return cleaned[:8]


def train_one_encoder(
    encoder: str,
    base_cfg: unet_mod.UNetConfig,
    train_tiles: list[str],
    *,
    skip_oof: bool,
    skip_final: bool,
) -> None:
    tag = short_tag(encoder)
    model_namespace = f"unet_{tag}"
    print(f"\n##### Encoder: {encoder} (namespace: {model_namespace}) #####")

    oof_dir = paths.OOF_ROOT / model_namespace
    oof_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"encoder": encoder, "folds": [], "tiles": {}}

    if not skip_oof:
        for split in validation.loro_splits(train_tiles):
            print(f"\n=== UNet[{tag}] LORO fold {split.name} ===")
            print(f"train: {split.train}")
            print(f"val  : {split.val}")
            cfg = unet_mod.UNetConfig(**{**base_cfg.__dict__, "encoder": encoder})
            model, cfg = unet_mod.fit_unet(split.train, split.val, cfg)
            fold_path = paths.MODELS_ROOT / f"{model_namespace}_{split.name}.pt"
            unet_mod.save_unet(model, fold_path, cfg)
            summary["folds"].append({"name": split.name, "n_train_tiles": len(split.train)})

            for tile in split.val:
                prob = unet_mod.predict_tile_unet(model, tile, "train", cfg, tta=True)
                np.save(oof_dir / f"{tile}.npy", prob.astype(np.float16))
                arrays, _ = cache.load_tile_cache(tile, "train")
                mask = arrays["consensus_pos"] | arrays["consensus_neg"]
                if mask.sum() > 0:
                    y = arrays["consensus_pos"].astype(np.uint8)[mask]
                    p = prob[mask]
                    t, rep = best_threshold_f1(y, p)
                    summary["tiles"][tile] = {
                        "fold": split.name, "best_threshold": t, **rep.to_dict()
                    }
                    print(
                        f"  [{tile}] best_t={t:.2f} f1={rep.f1:.4f} "
                        f"prec={rep.precision:.4f} rec={rep.recall:.4f}"
                    )

        with open(oof_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    if not skip_final:
        print(f"\n=== Final UNet[{tag}] on all tiles ===")
        cfg = unet_mod.UNetConfig(**{**base_cfg.__dict__, "encoder": encoder})
        model, cfg = unet_mod.fit_unet(train_tiles, [], cfg)
        unet_mod.save_unet(model, paths.MODELS_ROOT / f"{model_namespace}_full.pt", cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--encoder", default="efficientnet-b3")
    parser.add_argument(
        "--encoder-list",
        default="",
        help="Comma-separated encoder names; if set, overrides --encoder and trains one model per encoder.",
    )
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default="bfloat16")
    parser.add_argument("--skip-oof", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--lovasz-phase", type=float, default=0.3)
    parser.add_argument("--lovasz-weight", type=float, default=0.7)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    args = parser.parse_args()

    paths.ensure_dirs()
    train_tiles, _ = audit.audit(strict=False)

    base_cfg = unet_mod.UNetConfig(
        encoder=args.encoder,
        encoder_weights=args.encoder_weights or None,
        patch_size=args.patch_size,
        stride=args.stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        amp_dtype=args.amp,
        lovasz_phase=args.lovasz_phase,
        lovasz_weight=args.lovasz_weight,
        ema_decay=args.ema_decay,
    )

    encoders = [e.strip() for e in args.encoder_list.split(",") if e.strip()] if args.encoder_list else [args.encoder]
    for encoder in encoders:
        train_one_encoder(encoder, base_cfg, train_tiles, skip_oof=args.skip_oof, skip_final=args.skip_final)


if __name__ == "__main__":
    main()

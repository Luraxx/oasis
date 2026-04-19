#!/usr/bin/env python3
"""
Step 3: Train U-Net segmentation model.

Usage:
    python -u scripts/03_train_unet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.config import MODELS, TRAIN_TILES, REGIONS
from src.models.unet import (
    build_model, FocalLoss, f1_score_gpu, GPUTileStore, DEVICE,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
PATCH_SIZE  = 128
BATCH_SIZE  = 256
N_EPOCHS    = 80
LR          = 3e-4
PATIENCE    = 15   # early stopping patience (improvement over oasis-luis)

# Validation: hold out one region
VAL_TILES = REGIONS["amazon"][:4]  # 4 Amazon tiles for validation


def train(name, train_tiles, val_tiles, save_as, n_epochs=N_EPOCHS):
    print(f"\n{'=' * 60}\nTraining: {name}")
    print(f"  Train tiles: {len(train_tiles)}, Val tiles: {len(val_tiles)}")

    train_store = GPUTileStore(train_tiles, patch_size=PATCH_SIZE,
                               n_patches_epoch=40_000, pos_ratio=0.5, augment=True)
    val_store = GPUTileStore(val_tiles, patch_size=PATCH_SIZE,
                             n_patches_epoch=8_000, pos_ratio=0.4, augment=False)

    if len(train_store) == 0:
        print("  No data!"); return None

    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  VRAM after load: {used:.1f} / {total:.1f} GB")

    model = build_model().to(DEVICE)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_f1, best_ep, no_improve = 0.0, 0, 0
    save_path = MODELS / save_as

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        t_loss, t_f1, n_b = 0.0, 0.0, 0
        for start in range(0, len(train_store), BATCH_SIZE):
            x, y, w = train_store.get_batch(start, min(start + BATCH_SIZE, len(train_store)))
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y, w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item(); t_f1 += f1_score_gpu(logits, y, w); n_b += 1

        scheduler.step()
        train_store.on_epoch_end()

        # Validate
        model.eval()
        v_loss, v_f1, n_v = 0.0, 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_store), BATCH_SIZE):
                x, y, w = val_store.get_batch(start, min(start + BATCH_SIZE, len(val_store)))
                logits = model(x)
                v_loss += criterion(logits, y, w).item()
                v_f1 += f1_score_gpu(logits, y, w)
                n_v += 1

        t_loss /= max(n_b, 1); t_f1 /= max(n_b, 1)
        v_loss /= max(n_v, 1); v_f1 /= max(n_v, 1)

        vram_str = ""
        if epoch % 10 == 0 and torch.cuda.is_available():
            vram_str = f"  VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB"

        print(f"  Ep {epoch:3d}/{n_epochs}  "
              f"loss={t_loss:.4f} f1={t_f1:.4f}  |  "
              f"val_loss={v_loss:.4f} val_f1={v_f1:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.1e}{vram_str}", flush=True)

        if v_f1 > best_f1:
            best_f1, best_ep = v_f1, epoch
            no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_f1": v_f1, "name": name}, save_path)
            print(f"    *** Best model → {save_path}  (F1={v_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\n  Done. Best F1={best_f1:.4f} @ Ep {best_ep}")
    return save_path


def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Holdout training
    sa_train = [t for t in TRAIN_TILES if t not in VAL_TILES]
    train("holdout", sa_train, VAL_TILES, "unet_holdout.pt", N_EPOCHS)

    # Full training (all tiles, same val for monitoring)
    train("full", TRAIN_TILES, VAL_TILES, "unet_full.pt", N_EPOCHS)

    print("\nAll done!")


if __name__ == "__main__":
    main()

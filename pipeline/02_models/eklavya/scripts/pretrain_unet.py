"""Self-supervised pretraining of the U-Net encoder via masked autoencoding.

For every cached tile (train and test) we sample 256x256 patches of the
multi-temporal channel stack, randomly mask out 25-50% of the spatial
positions (independent of the per-pixel features), and train the U-Net
to reconstruct the masked spectrum at the masked locations using L1
loss. The encoder weights are then exported and used as the initialiser
for the supervised U-Net (replacing ImageNet).

Why it helps:
* Africa test tile has zero training labels but is in the unlabeled
  pretraining pool - the encoder learns its texture statistics.
* Reduces overfitting on the small (16 tile) labeled set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from oasis import audit, cache, paths
from oasis.models import unet as unet_mod


class MaskedPatchDataset(Dataset):
    def __init__(self, tile_split: list[tuple[str, str]], patch_size: int = 256, stride: int = 128, mask_ratio: float = 0.4):
        self.patches: list[tuple[int, int, np.ndarray]] = []
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        rng = np.random.default_rng(0)
        for tile, split in tile_split:
            arrays, _ = cache.load_tile_cache(tile, split)
            channels, _ = unet_mod.build_unet_channels(arrays)
            h, w = channels.shape[-2:]
            ys = list(range(0, max(h - patch_size, 0) + 1, stride)) + [max(h - patch_size, 0)]
            xs = list(range(0, max(w - patch_size, 0) + 1, stride)) + [max(w - patch_size, 0)]
            for y in sorted(set(ys)):
                for x in sorted(set(xs)):
                    self.patches.append((y, x, channels))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        y, x, channels = self.patches[idx]
        ps = self.patch_size
        patch = channels[:, y : y + ps, x : x + ps]
        # Build mask: ratio of patches dropped at the spatial level.
        mask = (np.random.rand(ps, ps) < self.mask_ratio).astype(np.float32)
        masked = patch.copy()
        masked[:, mask.astype(bool)] = 0.0
        target = patch.copy()
        return torch.from_numpy(masked), torch.from_numpy(target), torch.from_numpy(mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--encoder", default="efficientnet-b3")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=paths.MODELS_ROOT / "unet_pretrain.pt")
    args = parser.parse_args()

    paths.ensure_dirs()
    train_tiles, test_tiles = audit.audit(strict=False)
    pairs = [(t, "train") for t in train_tiles] + [(t, "test") for t in test_tiles]

    ds = MaskedPatchDataset(pairs, patch_size=args.patch_size)
    print(f"Pretrain patches: {len(ds)}")
    if len(ds) == 0:
        print("No patches available - build cache first")
        return

    sample_ch, _, _ = ds[0]
    in_channels = sample_ch.shape[0]
    device = unet_mod.auto_device(args.device)
    print(f"Pretrain device: {device}")

    model = unet_mod.build_unet(in_channels, args.encoder, encoder_weights=None)
    # Replace the segmentation head with a per-pixel reconstruction head:
    # reuse the model's decoder and project to in_channels via a 1x1 conv.
    head = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    model.segmentation_head = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.GELU(),
        head,
    )
    # Replace the first encoder conv to take in_channels (smp default does
    # this via in_channels arg above already).
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for masked, target, mask in loader:
            masked = masked.to(device)
            target = target.to(device)
            mask = mask.to(device)
            pred = model(masked)
            # Loss only on masked pixels.
            m = mask.unsqueeze(1)
            loss = (F.l1_loss(pred, target, reduction="none") * m).sum() / m.sum().clamp(min=1.0) / pred.shape[1]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            n += 1
        print(f"  pretrain epoch {epoch + 1}/{args.epochs} loss={loss_sum / max(n, 1):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state_dict": model.encoder.state_dict(), "encoder": args.encoder, "in_channels": in_channels}, args.out)
    print(f"saved encoder weights to {args.out}")


if __name__ == "__main__":
    main()

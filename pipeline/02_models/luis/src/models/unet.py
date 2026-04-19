"""U-Net model, GPU tile store, and training utilities."""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.config import CACHE, MODELS, N_FEATURES

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(encoder="efficientnet-b4"):
    """Create U-Net with specified encoder."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=N_FEATURES,
        classes=1,
        activation=None,
    )


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced segmentation."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, weights):
        logits = logits.squeeze(1)
        p = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device),
            torch.tensor(1 - self.alpha, device=logits.device),
        )
        loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t.clamp(1e-7))
        return (loss * weights).sum() / weights.sum().clamp(1)


@torch.no_grad()
def f1_score_gpu(logits, targets, weights, thr=0.4):
    pred = torch.sigmoid(logits.squeeze(1)) >= thr
    valid = weights > 0
    tp = (pred & (targets == 1) & valid).sum().float()
    fp = (pred & (targets == 0) & valid).sum().float()
    fn = (~pred & (targets == 1) & valid).sum().float()
    return (2 * tp / (2 * tp + fp + fn + 1e-7)).item()


class GPUTileStore:
    """Load all feature maps as GPU tensors. Patch sampling on GPU."""

    def __init__(self, tiles, patch_size=128, n_patches_epoch=50_000,
                 pos_ratio=0.5, augment=True):
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.n_patches = n_patches_epoch
        self.pos_ratio = pos_ratio
        self.augment = augment

        self.feat_tensors = []
        self.label_tensors = []
        self.pos_coords = []
        self.tile_names = []

        total_mb = 0
        for tile in tiles:
            fp = CACHE / f"{tile}_features.npz"
            lp = CACHE / f"{tile}_labels.npz"
            mp = CACHE / f"{tile}_meta.json"
            if not fp.exists():
                continue

            feats = np.load(fp)["features"]
            labels = np.load(lp)["labels"]
            with open(mp) as f:
                meta = json.load(f)

            H, W = meta["shape"]
            if H < patch_size or W < patch_size:
                continue

            np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            feat_map = feats.reshape(H, W, N_FEATURES).transpose(2, 0, 1).astype(np.float32)
            label_map = labels.reshape(H, W).astype(np.int8)

            f_gpu = torch.from_numpy(feat_map).to(DEVICE)
            l_gpu = torch.from_numpy(label_map).to(DEVICE)
            mb = f_gpu.element_size() * f_gpu.numel() / 1e6
            total_mb += mb

            h = self.half
            pos_y, pos_x = torch.where(l_gpu == 1)
            valid = (pos_y >= h) & (pos_y < H - h) & (pos_x >= h) & (pos_x < W - h)
            pos_centers = torch.stack([pos_y[valid], pos_x[valid]], dim=1).cpu()

            self.feat_tensors.append(f_gpu)
            self.label_tensors.append(l_gpu)
            self.pos_coords.append(pos_centers)
            self.tile_names.append(tile)

            n_pos = int((labels == 1).sum())
            print(f"  [store] {tile}: {H}×{W}  pos={n_pos:,}  ({mb:.0f} MB)")

        print(f"  Total VRAM for features: {total_mb / 1024:.1f} GB")
        self._build_index()

    def _build_index(self):
        n_tiles = len(self.feat_tensors)
        if n_tiles == 0:
            self.index = torch.zeros(0, 3, dtype=torch.long)
            return

        n_pos_total = int(self.n_patches * self.pos_ratio)
        n_neg_total = self.n_patches - n_pos_total
        rows = []

        for ti, centers in enumerate(self.pos_coords):
            if len(centers) == 0:
                continue
            n_sel = min(n_pos_total // n_tiles, len(centers))
            idx = torch.randperm(len(centers))[:n_sel]
            sel = centers[idx]
            rows.append(torch.cat([torch.full((n_sel, 1), ti, dtype=torch.long), sel], dim=1))

        for ti, f in enumerate(self.feat_tensors):
            H, W = f.shape[1], f.shape[2]
            h = self.half
            n = n_neg_total // n_tiles
            ys = torch.randint(h, H - h, (n,))
            xs = torch.randint(h, W - h, (n,))
            rows.append(torch.cat([torch.full((n, 1), ti, dtype=torch.long),
                                   ys.unsqueeze(1), xs.unsqueeze(1)], dim=1))

        self.index = torch.cat(rows)[torch.randperm(sum(len(r) for r in rows))]

    def __len__(self):
        return len(self.index)

    def get_batch(self, start, end):
        batch = self.index[start:end]
        h = self.half
        feats_list, labels_list = [], []
        for row in batch:
            ti, cy, cx = row[0].item(), row[1].item(), row[2].item()
            feats_list.append(self.feat_tensors[ti][:, cy - h:cy + h, cx - h:cx + h])
            labels_list.append(self.label_tensors[ti][cy - h:cy + h, cx - h:cx + h])

        x = torch.stack(feats_list)
        y = torch.stack(labels_list)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = x.flip(3); y = y.flip(2)
            if torch.rand(1).item() > 0.5:
                x = x.flip(2); y = y.flip(1)
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                x = torch.rot90(x, k, [2, 3])
                y = torch.rot90(y, k, [1, 2])

        w = (y >= 0).float()
        return x.float(), y.clamp(min=0).long(), w

    def on_epoch_end(self):
        self._build_index()

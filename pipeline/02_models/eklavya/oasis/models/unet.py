"""Multi-temporal U-Net for spatial-context deforestation prediction.

Input tensor per patch: ``(C, H, W)`` float32.

Channels (default 60-ish, all derivable from cache):

* 6 yearly NDVI means          (Y=2020..2025)
* 6 yearly NBR means
* 6 yearly NDMI means
* 6 yearly EVI means
* 1 NDVI slope
* 1 NDVI max-drop
* 1 NBR slope
* 1 NBR max-drop
* 6 yearly S1 ascending dB means
* 6 yearly S1 descending dB means
* 1 S1 ascending mean
* 1 S1 descending mean
* 16 AEF channels of the latest year (channels 0..15)
* 16 AEF channels of the earliest year (channels 0..15)

Total = ~73 channels (small enough for a EfficientNet-B3 encoder).

Patches are 256x256 sampled with overlap from training tiles. Sampling
oversamples patches that contain at least one consensus-positive pixel.

Loss: BCE + Dice + Lovasz, weighted by per-pixel ``sample_weight``.
TTA at inference: D4 group (4 rotations x 2 flips = 8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from oasis import cache
from oasis.features import s2 as s2_feat
from oasis.features.pack import _yearly_reduce, _slope_and_r2, _max_drop


YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
INDEX_NAMES_4 = ("ndvi", "nbr", "ndmi", "evi")
N_AEF_CHANNELS_USE = 16


def build_unet_channels(arrays: dict) -> tuple[np.ndarray, list[str]]:
    """Assemble the per-tile (C, H, W) U-Net input tensor."""
    s2_stack = arrays["s2_stack"]
    s2_valid = arrays["s2_valid"]
    s2_ym = arrays["s2_ym"]

    h, w = s2_stack.shape[-2:]

    # Compute monthly indices then yearly means for each requested index.
    feats: list[np.ndarray] = []
    names: list[str] = []
    monthly_idx: dict[str, np.ndarray] = {n: np.full((s2_stack.shape[0], h, w), np.nan, dtype=np.float32) for n in INDEX_NAMES_4}
    for ti in range(s2_stack.shape[0]):
        idx_dict = s2_feat.compute_indices(s2_stack[ti].astype(np.float32), valid=s2_valid[ti])
        for n in INDEX_NAMES_4:
            monthly_idx[n][ti] = idx_dict[n]

    for n in INDEX_NAMES_4:
        yearly = _yearly_reduce(monthly_idx[n], s2_ym, YEARS)
        for yi, yr in enumerate(YEARS):
            feats.append(np.nan_to_num(yearly[yi], nan=0.0))
            names.append(f"unet_{n}_y{yr}")
        if n in ("ndvi", "nbr"):
            slope, _ = _slope_and_r2(yearly, YEARS)
            drop, _ = _max_drop(yearly)
            feats.append(slope)
            names.append(f"unet_{n}_slope")
            feats.append(drop)
            names.append(f"unet_{n}_max_drop")

    # S1 yearly per orbit
    for orbit, key in [("asc", "s1_asc_db"), ("desc", "s1_desc_db")]:
        ym = arrays[f"s1_{orbit}_ym"]
        db = arrays[key]
        if db.shape[0] > 0:
            yearly = _yearly_reduce(db, ym, YEARS)
            for yi, yr in enumerate(YEARS):
                feats.append(np.nan_to_num(yearly[yi], nan=0.0))
                names.append(f"unet_s1_{orbit}_y{yr}")
            feats.append(np.nanmean(db, axis=0).astype(np.float32))
            names.append(f"unet_s1_{orbit}_mean")
        else:
            for yr in YEARS:
                feats.append(np.zeros((h, w), dtype=np.float32))
                names.append(f"unet_s1_{orbit}_y{yr}")
            feats.append(np.zeros((h, w), dtype=np.float32))
            names.append(f"unet_s1_{orbit}_mean")

    # AEF latest + earliest (first 16 dims each)
    aef = arrays["aef_stack"]  # (Y, 64, H, W)
    aef_years = arrays["aef_years"]
    if aef.shape[0] > 0:
        for ci in range(N_AEF_CHANNELS_USE):
            feats.append(aef[-1, ci])
            names.append(f"unet_aef_latest_c{ci:02d}")
        early_idx = 0
        for ci in range(N_AEF_CHANNELS_USE):
            feats.append(aef[early_idx, ci])
            names.append(f"unet_aef_earliest_c{ci:02d}")
    else:
        for ci in range(2 * N_AEF_CHANNELS_USE):
            feats.append(np.zeros((h, w), dtype=np.float32))
            names.append(f"unet_aef_zero_{ci}")

    stack = np.stack(feats, axis=0).astype(np.float32)
    stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
    return stack, names


class UNetTilePatches(Dataset):
    """Yield patches sampled from a single tile cache.

    Tiles smaller than ``patch_size`` in either dimension produce zero
    patches and contribute nothing to the dataset. They are typically
    sliver tiles (e.g. 18NWM_9_4 is only 2 pixels wide) where the
    spatial U-Net cannot do useful work.
    """

    def __init__(
        self,
        tile_id: str,
        split: str,
        *,
        patch_size: int = 256,
        stride: int = 128,
        positive_oversample: float = 4.0,
        rng: np.random.Generator | None = None,
        min_dim: int | None = None,
    ):
        arrays, _ = cache.load_tile_cache(tile_id, split)
        self.tile_id = tile_id
        self.channels, self.names = build_unet_channels(arrays)
        self.labels = arrays["labels"].astype(np.float32)
        self.weights = np.maximum(arrays["sample_weight"], 1e-3).astype(np.float32)
        self.consensus_pos = arrays["consensus_pos"]
        self.patch_size = patch_size
        self.stride = stride
        self.positive_oversample = positive_oversample
        self.rng = rng or np.random.default_rng(42)
        self.min_dim = min_dim if min_dim is not None else patch_size
        h, w = self.labels.shape
        self.skipped_for_size = (h < self.min_dim or w < self.min_dim)
        self._patch_starts = [] if self.skipped_for_size else self._compute_patch_starts()

    def _compute_patch_starts(self) -> list[tuple[int, int]]:
        h, w = self.labels.shape
        ps = self.patch_size
        if h < ps or w < ps:
            return []
        starts: list[tuple[int, int]] = []
        ys = list(range(0, max(h - ps, 0) + 1, self.stride)) + [max(h - ps, 0)]
        xs = list(range(0, max(w - ps, 0) + 1, self.stride)) + [max(w - ps, 0)]
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                starts.append((y, x))

        # Oversample patches that contain consensus positives.
        cp = self.consensus_pos
        weighted: list[tuple[int, int]] = []
        for y, x in starts:
            crop = cp[y : y + ps, x : x + ps]
            if crop.any():
                k = int(round(self.positive_oversample))
                weighted.extend([(y, x)] * max(k, 1))
            else:
                weighted.append((y, x))
        return weighted

    def __len__(self) -> int:
        return len(self._patch_starts)

    def __getitem__(self, idx):
        y, x = self._patch_starts[idx]
        ps = self.patch_size
        ch = self.channels[:, y : y + ps, x : x + ps]
        lbl = self.labels[y : y + ps, x : x + ps]
        wt = self.weights[y : y + ps, x : x + ps]

        # Random flip + 90 rotation augmentation (training only, caller controls).
        if self.rng.random() < 0.5:
            ch = ch[:, :, ::-1].copy()
            lbl = lbl[:, ::-1].copy()
            wt = wt[:, ::-1].copy()
        if self.rng.random() < 0.5:
            ch = ch[:, ::-1, :].copy()
            lbl = lbl[::-1, :].copy()
            wt = wt[::-1, :].copy()
        k = int(self.rng.integers(0, 4))
        if k:
            ch = np.rot90(ch, k=k, axes=(1, 2)).copy()
            lbl = np.rot90(lbl, k=k).copy()
            wt = np.rot90(wt, k=k).copy()
        return (
            torch.from_numpy(ch),
            torch.from_numpy(lbl),
            torch.from_numpy(wt),
        )


class MultiTileDataset(Dataset):
    """Concatenates UNetTilePatches across tiles."""

    def __init__(self, datasets: list[UNetTilePatches]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cum = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return int(self.cum[-1]) if len(self.cum) else 0

    def __getitem__(self, idx):
        d = int(np.searchsorted(self.cum, idx, side="right"))
        local = idx - (self.cum[d - 1] if d > 0 else 0)
        return self.datasets[d][local]


def lovasz_grad(gt: torch.Tensor) -> torch.Tensor:
    p = len(gt)
    gts = gt.sum()
    intersection = gts - gt.float().cumsum(0)
    union = gts + (1 - gt).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for b in range(logits.shape[0]):
        losses.append(lovasz_hinge_flat(logits[b].view(-1), labels[b].view(-1)))
    return torch.stack(losses).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    w = weight
    num = 2.0 * (p * targets * w).sum(dim=(1, 2)) + 1.0
    den = ((p + targets) * w).sum(dim=(1, 2)) + 1.0
    return (1.0 - num / den).mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    *,
    use_lovasz: bool = True,
    lovasz_weight: float = 0.5,
) -> torch.Tensor:
    """BCE + Dice (+ optional Lovasz) blended loss.

    Phase A of training disables Lovasz (BCE+Dice converges faster from
    random init). Phase B switches it on and lowers LR.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, weight=weight)
    d = dice_loss(logits, targets, weight)
    out = bce + d
    if use_lovasz:
        out = out + lovasz_weight * lovasz_hinge(logits, targets)
    return out


@dataclass
class UNetConfig:
    encoder: str = "efficientnet-b3"
    encoder_weights: str | None = "imagenet"
    patch_size: int = 256
    stride: int = 128
    positive_oversample: float = 4.0
    batch_size: int = 8
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "auto"
    amp_dtype: str = "bfloat16"  # "bfloat16" or "float32"
    random_state: int = 42
    in_channels: int = 0  # filled at fit time
    feature_names: list[str] = field(default_factory=list)
    # Two-phase loss: Lovasz only kicks in for the last `lovasz_phase`
    # fraction of epochs and the LR is divided by `lovasz_lr_div` at that point.
    lovasz_phase: float = 0.3
    lovasz_lr_div: float = 5.0
    lovasz_weight: float = 0.7
    # Exponential-moving-average weights (decay) and snapshot averaging.
    ema_decay: float = 0.999
    snapshot_keep: int = 3


def auto_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_unet(in_channels: int, encoder: str = "efficientnet-b3", encoder_weights: str | None = "imagenet"):
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        decoder_attention_type="scse",
    )


def _ema_update(ema_state: dict, model: nn.Module, decay: float) -> None:
    msd = model.state_dict()
    for k, v in msd.items():
        if k not in ema_state:
            ema_state[k] = v.detach().clone()
            continue
        if v.dtype.is_floating_point:
            ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
        else:
            ema_state[k] = v.detach().clone()


def _average_state_dicts(states: list[dict]) -> dict:
    if not states:
        raise ValueError("no states to average")
    out: dict = {}
    n = float(len(states))
    for k in states[0].keys():
        if states[0][k].dtype.is_floating_point:
            stacked = torch.stack([s[k].float() for s in states], dim=0)
            out[k] = (stacked.sum(dim=0) / n).to(states[0][k].dtype)
        else:
            out[k] = states[0][k].clone()
    return out


def fit_unet(
    train_tiles: list[str],
    val_tiles: list[str],
    cfg: UNetConfig,
    *,
    verbose: bool = True,
):
    device = auto_device(cfg.device)
    if verbose:
        print(f"  UNet device: {device}")
    rng = np.random.default_rng(cfg.random_state)

    train_datasets_all = [
        UNetTilePatches(t, "train", patch_size=cfg.patch_size, stride=cfg.stride,
                        positive_oversample=cfg.positive_oversample, rng=rng)
        for t in train_tiles
    ]
    val_datasets_all = [
        UNetTilePatches(t, "train", patch_size=cfg.patch_size, stride=cfg.patch_size,
                        positive_oversample=1.0, rng=rng)
        for t in val_tiles
    ]
    train_datasets = [d for d in train_datasets_all if len(d) > 0]
    val_datasets = [d for d in val_datasets_all if len(d) > 0]
    skipped_train = [d.tile_id for d in train_datasets_all if d.skipped_for_size]
    skipped_val = [d.tile_id for d in val_datasets_all if d.skipped_for_size]
    if skipped_train and verbose:
        print(f"  UNet skipping undersized train tiles: {skipped_train}")
    if skipped_val and verbose:
        print(f"  UNet skipping undersized val tiles: {skipped_val}")

    if not train_datasets:
        raise RuntimeError("No usable train tiles after size filter for U-Net")
    train_ds = MultiTileDataset(train_datasets)
    val_ds = MultiTileDataset(val_datasets) if val_datasets else None

    in_channels = train_datasets[0].channels.shape[0]
    cfg.in_channels = in_channels
    cfg.feature_names = train_datasets[0].names

    model = build_unet(in_channels, cfg.encoder, cfg.encoder_weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0) if val_ds else None

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" and device.type == "cuda" else torch.float32

    # EMA of weights, evaluated independently from the fast-moving live model.
    ema_state: dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema_model = build_unet(in_channels, cfg.encoder, cfg.encoder_weights).to(device)

    best_f1 = -1.0
    best_state = None
    best_snapshots: list[tuple[float, dict]] = []
    lovasz_start = max(1, int(round(cfg.epochs * (1.0 - cfg.lovasz_phase))))

    for epoch in range(cfg.epochs):
        # Phase B: enable Lovasz and drop LR on the first epoch we cross the threshold.
        if epoch == lovasz_start and cfg.lovasz_lr_div > 1.0:
            for pg in opt.param_groups:
                pg["lr"] = pg["lr"] / cfg.lovasz_lr_div
            if verbose:
                print(f"  UNet phase B: enabling Lovasz, lr/= {cfg.lovasz_lr_div:g}")
        use_lovasz = epoch >= lovasz_start

        model.train()
        loss_sum = 0.0
        n_batches = 0
        for ch, lbl, wt in train_loader:
            ch = ch.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            wt = wt.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype == torch.bfloat16)):
                logits = model(ch).squeeze(1)
                loss = combined_loss(
                    logits, lbl, wt,
                    use_lovasz=use_lovasz,
                    lovasz_weight=cfg.lovasz_weight,
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            _ema_update(ema_state, model, cfg.ema_decay)
            loss_sum += loss.item()
            n_batches += 1
        sched.step()

        msg = (
            f"  UNet epoch {epoch + 1}/{cfg.epochs} "
            f"train_loss={loss_sum / max(n_batches, 1):.4f} "
            f"phase={'B' if use_lovasz else 'A'}"
        )

        if val_loader is not None:
            ema_model.load_state_dict(ema_state)
            ema_model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for ch, lbl, _ in val_loader:
                    ch = ch.to(device); lbl = lbl.to(device)
                    logits = ema_model(ch).squeeze(1)
                    ys.append(lbl.cpu().numpy().reshape(-1))
                    ps.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            from oasis.metrics import best_threshold_f1
            yarr = np.concatenate(ys)
            parr = np.concatenate(ps)
            _, rep = best_threshold_f1(yarr, parr)
            msg += f" ema_val_f1={rep.f1:.4f}"
            snap = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
            best_snapshots.append((rep.f1, snap))
            best_snapshots.sort(key=lambda t: t[0], reverse=True)
            best_snapshots = best_snapshots[: cfg.snapshot_keep]
            if rep.f1 > best_f1:
                best_f1 = rep.f1
                best_state = snap
        if verbose:
            print(msg)

    # Choose the published checkpoint:
    # 1. If we collected snapshots, average the top-K EMA snapshots.
    # 2. Else fall back to the last EMA state, then to the live model.
    if best_snapshots:
        avg = _average_state_dicts([s for _, s in best_snapshots])
        model.load_state_dict(avg)
        if verbose:
            print(
                f"  UNet final = mean of top-{len(best_snapshots)} EMA snapshots "
                f"(best individual val_f1={best_f1:.4f})"
            )
    elif best_state is not None:
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(ema_state)
    return model, cfg


def _sliding_window_predict(
    model,
    channels: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    transforms: list[tuple[int, bool]],
    device,
) -> np.ndarray:
    """Predict on (C, H, W) with reflective pad + sliding window + per-patch TTA.

    Handles tiles smaller than ``patch_size`` by zero-padding up to one
    full patch (rather than requiring multiple) so we still get a valid
    prediction back at the original native resolution.
    """
    c, h, w = channels.shape
    ps = patch_size
    # Tile must hold at least one full patch; pad to the next multiple
    # of ``ps`` along each dimension. ``mode="reflect"`` requires
    # input >= 2; fall back to ``edge`` when tile is degenerate.
    pad_h = max(0, ps - h) + (ps - max(h, ps) % ps) % ps
    pad_w = max(0, ps - w) + (ps - max(w, ps) % ps) % ps
    pad_mode = "reflect" if min(h, w) >= 2 else "edge"
    channels = np.pad(channels, ((0, 0), (0, pad_h), (0, pad_w)), mode=pad_mode)
    H, W = channels.shape[-2:]

    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, max(H - ps, 0) + 1, stride)) + [max(H - ps, 0)]
    xs = list(range(0, max(W - ps, 0) + 1, stride)) + [max(W - ps, 0)]
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = channels[:, y : y + ps, x : x + ps]
                acc = np.zeros((ps, ps), dtype=np.float32)
                for k, flip in transforms:
                    p = np.rot90(patch, k=k, axes=(1, 2))
                    if flip:
                        p = p[:, :, ::-1]
                    p = p.copy()
                    t = torch.from_numpy(p).unsqueeze(0).to(device)
                    logits = model(t).squeeze(1)
                    out = torch.sigmoid(logits).cpu().numpy()[0]
                    if flip:
                        out = out[:, ::-1]
                    out = np.rot90(out, k=-k)
                    acc += out
                acc /= len(transforms)
                pred_sum[y : y + ps, x : x + ps] += acc
                pred_count[y : y + ps, x : x + ps] += 1
    pred = pred_sum / np.maximum(pred_count, 1.0)
    return pred[:h, :w].astype(np.float32)


def _resize_channels(channels: np.ndarray, scale: float) -> np.ndarray:
    """Bilinear resize (C, H, W) by `scale` using torch.nn.functional.interpolate."""
    if abs(scale - 1.0) < 1e-3:
        return channels
    t = torch.from_numpy(channels).unsqueeze(0).float()
    h, w = channels.shape[-2:]
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    out = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
    return out.squeeze(0).numpy().astype(channels.dtype)


def _resize_prob(prob: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if prob.shape == target_hw:
        return prob
    t = torch.from_numpy(prob).unsqueeze(0).unsqueeze(0).float()
    out = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0).numpy().astype(np.float32)


def predict_tile_unet(
    model,
    tile_id: str,
    split: str,
    cfg: UNetConfig,
    *,
    tta: bool = True,
    multiscale: tuple[float, ...] = (0.75, 1.0, 1.25),
    stride_div: int = 4,
    device: torch.device | None = None,
) -> np.ndarray:
    """Multi-scale + D4-TTA tile prediction with (ps / stride_div) sliding stride.

    Args:
        multiscale: List of input scales to evaluate. Predictions are bilinearly
            resampled back to the native grid and averaged. Pass ``(1.0,)`` for
            single-scale inference.
        stride_div: Sliding-window stride is ``patch_size // stride_div``.
            ``stride_div=4`` is roughly 4x slower than ``=2`` but gives better
            edge stitching.
    """
    device = device or auto_device(cfg.device)
    model = model.to(device).eval()
    arrays, _ = cache.load_tile_cache(tile_id, split)
    channels, _ = build_unet_channels(arrays)
    c, h, w = channels.shape
    ps = cfg.patch_size
    stride = max(1, ps // stride_div)

    transforms = [(0, False)]
    if tta:
        transforms = [(k, flip) for k in (0, 1, 2, 3) for flip in (False, True)]

    scales = multiscale or (1.0,)
    accum = np.zeros((h, w), dtype=np.float32)
    for s in scales:
        scaled = _resize_channels(channels, s)
        pred = _sliding_window_predict(
            model, scaled,
            patch_size=ps, stride=stride,
            transforms=transforms, device=device,
        )
        accum += _resize_prob(pred, (h, w))
    return (accum / float(len(scales))).astype(np.float32)


def save_unet(model, path: Path, cfg: UNetConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "cfg": cfg.__dict__}
    torch.save(payload, path)


def load_unet(path: Path):
    payload = torch.load(path, map_location="cpu")
    cfg = UNetConfig(**payload["cfg"])
    model = build_unet(cfg.in_channels, cfg.encoder, encoder_weights=None)
    model.load_state_dict(payload["state_dict"])
    return model, cfg

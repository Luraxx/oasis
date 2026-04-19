"""Per-pixel Temporal 1D-CNN.

Input: aligned monthly time series for each pixel.

We construct a (B, C, T) tensor per sample where:

* C = 12 raw S2 bands (normalized to [0, 1] by SCALE) + 6 derived
       indices (NDVI, NBR, NDMI, EVI, MNDWI, BSI) + 1 valid mask channel
       = 19 channels.
* T = padded to 72 monthly time steps (Jan 2020 .. Dec 2025). Missing
       months are zero-filled and the valid channel marks them.

Architecture: 4 dilated conv blocks (dilations 1, 2, 4, 8), kernel 3,
hidden width 96, GELU activations, BatchNorm, residual connections.
Followed by global mean+max pooling then a small MLP -> single logit.

Training is per-pixel with sample weights (label confidence). Inference
runs in batched chunks across the whole tile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from oasis import cache, sampling
from oasis.features import s2 as s2_feat


YEARS_FULL = (2020, 2021, 2022, 2023, 2024, 2025)
MONTHS_PER_YEAR = 12
N_TIME = len(YEARS_FULL) * MONTHS_PER_YEAR  # 72
S2_BANDS = 12
INDEX_NAMES = ("ndvi", "nbr", "ndmi", "evi", "mndwi", "bsi")
N_CHANNELS = S2_BANDS + len(INDEX_NAMES) + 1  # +1 valid mask


def _ym_to_index(year: int, month: int) -> int:
    return (int(year) - YEARS_FULL[0]) * MONTHS_PER_YEAR + (int(month) - 1)


def build_pixel_time_series(arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, C, H, W) float32 series and (T, H, W) bool valid mask.

    Missing months are zero in the data channels and False in the valid
    mask. The valid mask is also passed as channel C-1 of the data
    tensor for direct consumption by the network.
    """
    s2 = arrays["s2_stack"]            # (T_real, 12, H, W) uint16
    valid = arrays["s2_valid"]         # (T_real, H, W) bool
    ym = arrays["s2_ym"]               # (T_real, 2) int16
    h, w = s2.shape[-2:]

    out = np.zeros((N_TIME, N_CHANNELS, h, w), dtype=np.float16)
    valid_t = np.zeros((N_TIME, h, w), dtype=bool)

    for t_real in range(s2.shape[0]):
        year, month = int(ym[t_real, 0]), int(ym[t_real, 1])
        if year < YEARS_FULL[0] or year > YEARS_FULL[-1]:
            continue
        ti = _ym_to_index(year, month)
        if ti < 0 or ti >= N_TIME:
            continue

        bands = s2[t_real].astype(np.float32) / s2_feat.SCALE
        bands = np.clip(bands, 0.0, 1.5).astype(np.float16)
        out[ti, :S2_BANDS] = bands

        idx_dict = s2_feat.compute_indices(s2[t_real].astype(np.float32), valid=valid[t_real])
        for ci, name in enumerate(INDEX_NAMES):
            out[ti, S2_BANDS + ci] = np.nan_to_num(idx_dict[name], nan=0.0).astype(np.float16)

        v = valid[t_real]
        out[ti, -1] = v.astype(np.float16)
        valid_t[ti] = v
    return out, valid_t


class PixelTimeSeriesDataset(Dataset):
    """Holds (sample_count, C, T) torch tensors in CPU RAM."""

    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        # X: (N, C, T) float32
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.w = torch.from_numpy(w.astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


class DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.gelu(x + h)


class TemporalCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        hidden: int = 96,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [DilatedBlock(hidden, d, dropout=dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: (B, C, T)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        # global mean + max
        gmean = h.mean(dim=-1)
        gmax = h.max(dim=-1).values
        z = torch.cat([gmean, gmax], dim=-1)
        return self.head(z).squeeze(-1)


@dataclass
class TCNConfig:
    samples_per_tile: int = 30_000
    pos_neg_ratio: float = 1.5
    hidden: int = 96
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 4096
    epochs: int = 15
    device: str = "auto"  # "auto" -> cuda > mps > cpu
    random_state: int = 42
    feature_names: list[str] = field(default_factory=list)


def auto_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_tcn_dataset(
    fit_tiles: list[str], cfg: TCNConfig, *, rng: np.random.Generator | None = None, verbose: bool = True
) -> PixelTimeSeriesDataset:
    rng = rng or np.random.default_rng(cfg.random_state)
    Xs, ys, ws = [], [], []
    for tile_id in tqdm(fit_tiles, desc="building TCN dataset", disable=not verbose):
        arrays, _ = cache.load_tile_cache(tile_id, "train")
        series, _ = build_pixel_time_series(arrays)  # (T, C, H, W) float16
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
            continue
        rows, cols = coords[:, 0], coords[:, 1]
        # Result shape: (N, C, T)
        X_tile = np.transpose(series[:, :, rows, cols], (2, 1, 0)).astype(np.float32)
        y_tile = arrays["labels"][rows, cols].astype(np.float32)
        w_tile = np.maximum(arrays["sample_weight"][rows, cols], 1e-3).astype(np.float32)
        Xs.append(X_tile)
        ys.append(y_tile)
        ws.append(w_tile)
    if not Xs:
        raise RuntimeError("No TCN training data assembled")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    if verbose:
        print(f"  TCN dataset: X={X.shape} pos={int(y.sum())} neg={int(len(y) - y.sum())}")
    return PixelTimeSeriesDataset(X, y, w)


def train_tcn(
    train_ds: PixelTimeSeriesDataset,
    val_ds: PixelTimeSeriesDataset | None,
    cfg: TCNConfig,
    *,
    verbose: bool = True,
) -> TemporalCNN:
    device = auto_device(cfg.device)
    if verbose:
        print(f"  TCN device: {device}")
    torch.manual_seed(cfg.random_state)
    model = TemporalCNN(in_channels=N_CHANNELS, hidden=cfg.hidden, dilations=cfg.dilations, dropout=cfg.dropout)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False) if val_ds is not None else None
    )

    best_val = -1.0
    best_state = None
    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for X, y, w in train_loader:
            X, y, w = X.to(device), y.to(device), w.to(device)
            logits = model(X)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(y)
            n_seen += len(y)
        sched.step()
        if verbose:
            msg = f"  TCN epoch {epoch + 1}/{cfg.epochs} train_loss={loss_sum / max(n_seen, 1):.4f}"
            if val_loader is not None:
                model.eval()
                vl, vp, vy = 0.0, [], []
                vw_sum = 0.0
                with torch.no_grad():
                    for X, y, w in val_loader:
                        X, y, w = X.to(device), y.to(device), w.to(device)
                        logits = model(X)
                        l = F.binary_cross_entropy_with_logits(logits, y, weight=w, reduction="sum")
                        vl += l.item()
                        vw_sum += w.sum().item()
                        vp.append(torch.sigmoid(logits).cpu().numpy())
                        vy.append(y.cpu().numpy())
                pred = np.concatenate(vp)
                tgt = np.concatenate(vy)
                from oasis.metrics import best_threshold_f1
                _, rep = best_threshold_f1(tgt, pred)
                msg += f" val_loss={vl / max(vw_sum, 1):.4f} val_f1={rep.f1:.4f}"
                if rep.f1 > best_val:
                    best_val = rep.f1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_tile(
    model: TemporalCNN,
    tile_id: str,
    split: str,
    *,
    chunk_size: int = 32_768,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or auto_device()
    model = model.to(device).eval()
    arrays, _ = cache.load_tile_cache(tile_id, split)
    series, _ = build_pixel_time_series(arrays)
    t, c, h, w = series.shape
    flat = np.transpose(series.reshape(t, c, -1), (2, 1, 0))  # (H*W, C, T)
    out = np.zeros(flat.shape[0], dtype=np.float32)
    with torch.no_grad():
        for s in range(0, flat.shape[0], chunk_size):
            e = min(s + chunk_size, flat.shape[0])
            x = torch.from_numpy(flat[s:e].astype(np.float32)).to(device)
            logits = model(x)
            out[s:e] = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    return out.reshape(h, w)


def save_tcn(model: TemporalCNN, path: Path, cfg: TCNConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, path)


def load_tcn(path: Path) -> tuple[TemporalCNN, TCNConfig]:
    payload = torch.load(path, map_location="cpu")
    cfg = TCNConfig(**payload["cfg"])
    model = TemporalCNN(in_channels=N_CHANNELS, hidden=cfg.hidden, dilations=cfg.dilations, dropout=cfg.dropout)
    model.load_state_dict(payload["state_dict"])
    return model, cfg

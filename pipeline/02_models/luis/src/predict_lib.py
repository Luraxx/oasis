"""Shared inference utilities used by OOF + final prediction."""

import json
from pathlib import Path

import numpy as np
import torch

from src.config import CACHE, N_FEATURES


def load_features(tile: str):
    fp = CACHE / f"{tile}_features.npz"
    mp = CACHE / f"{tile}_meta.json"
    if not fp.exists():
        return None, None
    feats = np.load(fp)["features"]
    with open(mp) as f:
        meta = json.load(f)
    return feats, meta


def load_labels(tile: str):
    """Load the binary consensus labels (1=pos, 0=neg, -1=unknown)."""
    lp = CACHE / f"{tile}_labels.npz"
    if not lp.exists():
        return None
    return np.load(lp)["labels"]


def predict_lgbm_one(feats: np.ndarray, model_path: Path) -> np.ndarray:
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(model_path))
    return booster.predict(feats).astype(np.float32)


def predict_unet_one(feats: np.ndarray, meta: dict, model_path: Path,
                     patch_size: int = 256, batch_size: int = 64,
                     device: str | None = None) -> np.ndarray:
    """Sliding-window U-Net inference. Returns flat (N,) probability vector."""
    from src.models.unet import build_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    H, W = meta["shape"]
    ckpt = torch.load(model_path, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feat_map = feats.reshape(H, W, N_FEATURES).transpose(2, 0, 1)
    pad = patch_size // 2
    feat_pad = np.pad(feat_map, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    patches, coords = [], []
    for y in range(0, H, patch_size // 2):
        for x in range(0, W, patch_size // 2):
            yc, xc = min(y, H - 1), min(x, W - 1)
            patch = feat_pad[:, yc:yc + patch_size, xc:xc + patch_size]
            if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                p2 = np.zeros((N_FEATURES, patch_size, patch_size), dtype=np.float32)
                ph = min(patch.shape[1], patch_size)
                pw = min(patch.shape[2], patch_size)
                p2[:, :ph, :pw] = patch[:, :ph, :pw]
                patch = p2
            patches.append(patch.astype(np.float32))
            coords.append((yc, xc))

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.from_numpy(np.stack(patches[i:i + batch_size])).to(device)
            probs = torch.sigmoid(model(batch).squeeze(1)).cpu().numpy()
            for j, (yc, xc) in enumerate(coords[i:i + batch_size]):
                ph = min(patch_size, H - yc)
                pw = min(patch_size, W - xc)
                prob_map[yc:yc + ph, xc:xc + pw] += probs[j, :ph, :pw]
                count_map[yc:yc + ph, xc:xc + pw] += 1.0

    prob_map /= np.maximum(count_map, 1)
    return prob_map.ravel()

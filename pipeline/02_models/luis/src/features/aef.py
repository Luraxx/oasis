"""AlphaEarth Foundation model embedding features."""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import DATA, AEF_PCA_DIM, AEF_DELTA_DIM


def load_aef(tile, year, split="train"):
    """Load AEF embedding raster. Returns (64, H', W') array and profile."""
    p = DATA / f"aef-embeddings/{split}/{tile}_{year}.tiff"
    if not p.exists():
        return None, None
    with rasterio.open(p) as src:
        data = src.read().astype(np.float32)
        profile = src.profile
    return data, profile


def _reproject_aef(tile, year, split, ref_shape, ref_transform, ref_crs):
    """Reproject AEF from EPSG:4326 to the tile's UTM grid."""
    aef, _ = load_aef(tile, year, split)
    if aef is None:
        return None
    C, _, _ = aef.shape
    H, W = ref_shape
    out = np.zeros((C, H, W), dtype=np.float32)
    with rasterio.open(DATA / f"aef-embeddings/{split}/{tile}_{year}.tiff") as src:
        for c in range(C):
            reproject(
                source=aef[c], destination=out[c],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
    return out


def extract_aef_features(tile, split, ref_shape, ref_transform, ref_crs):
    """
    Extract 13 AEF features: PCA(8) on 2020 + cosine_sim + delta_PCA(4).
    Returns list of 13 (H,W) arrays.
    """
    H, W = ref_shape
    N = H * W
    features = []

    aef_2020 = _reproject_aef(tile, 2020, split, ref_shape, ref_transform, ref_crs)
    if aef_2020 is None:
        return [np.zeros(ref_shape, dtype=np.float32) for _ in range(AEF_PCA_DIM + 1 + AEF_DELTA_DIM)]

    C = aef_2020.shape[0]
    aef_flat = aef_2020.reshape(C, N).T  # (N, 64)

    # PCA on 2020 embeddings → landscape characterization
    valid_mask = np.isfinite(aef_flat).all(axis=1)
    aef_pca_result = np.zeros((N, AEF_PCA_DIM), dtype=np.float32)
    if valid_mask.sum() > 100:
        scaler = StandardScaler()
        pca = PCA(n_components=AEF_PCA_DIM, random_state=42)
        aef_pca_result[valid_mask] = pca.fit_transform(
            scaler.fit_transform(aef_flat[valid_mask])
        )
    features += [aef_pca_result[:, k].reshape(H, W) for k in range(AEF_PCA_DIM)]

    # Change detection: 2020 vs 2025
    aef_2025 = _reproject_aef(tile, 2025, split, ref_shape, ref_transform, ref_crs)
    if aef_2025 is not None:
        aef_2025_flat = aef_2025.reshape(C, N).T
        delta_flat = aef_2025_flat - aef_flat

        # Cosine similarity
        norm_2020 = np.linalg.norm(aef_flat, axis=1, keepdims=True) + 1e-9
        norm_2025 = np.linalg.norm(aef_2025_flat, axis=1, keepdims=True) + 1e-9
        cos_sim = (aef_flat / norm_2020 * aef_2025_flat / norm_2025).sum(axis=1)
        features.append(np.nan_to_num(cos_sim, nan=1.0).reshape(H, W))

        # Delta PCA
        delta_pca_result = np.zeros((N, AEF_DELTA_DIM), dtype=np.float32)
        valid_delta = np.isfinite(delta_flat).all(axis=1)
        if valid_delta.sum() > 100:
            pca_delta = PCA(n_components=AEF_DELTA_DIM, random_state=42)
            delta_pca_result[valid_delta] = pca_delta.fit_transform(
                np.nan_to_num(delta_flat[valid_delta], nan=0.0)
            )
        features += [delta_pca_result[:, k].reshape(H, W) for k in range(AEF_DELTA_DIM)]
    else:
        features += [np.zeros(ref_shape, dtype=np.float32) for _ in range(1 + AEF_DELTA_DIM)]

    return features

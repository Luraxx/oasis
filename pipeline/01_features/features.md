# Feature Engineering

## Overview

Each pixel in the 10m-resolution tiles is described by up to **508 features** derived from four data sources. All features are computed as **change relative to a 2020 baseline** for geographic robustness across Amazon, Southeast Asia, and Africa.

## Data Sources

### 1. Sentinel-2 L2A (~84 features)

Six spectral indices computed per pixel from 12 multispectral bands:

| Index | Formula | Signal |
|-------|---------|--------|
| NDVI | (B08−B04)/(B08+B04) | Vegetation greenness |
| NBR | (B08−B12)/(B08+B12) | Burn/clearing severity |
| NDMI | (B08−B11)/(B08+B11) | Moisture stress |
| EVI | 2.5·(B08−B04)/(B08+6·B04−7.5·B02+1) | Enhanced vegetation |
| MNDWI | (B03−B11)/(B03+B11) | Water presence |
| BSI | ((B11+B04)−(B08+B02))/((B11+B04)+(B08+B02)) | Bare soil exposure |

**Temporal statistics** per index (6 years × monthly → 72 time steps):
- `mean_2020`: Baseline stability (pre-deforestation reference)
- `std_2020`: Noise / cloud contamination level
- `min_post`: Minimum post-2020 value (recovery detection)
- `maxdrop`: Maximum year-to-year drop (deforestation signal strength)
- `slope`: Linear trend across years
- `drop_year`: Year of largest negative change
- `mean_2026`: Recent 2026 signal (from additional data)
- `dry_mean` / `wet_mean`: Seasonal decomposition (Jun-Sep vs Oct-May)

→ 6 indices × 14 stats = **84 features**

### 2. Sentinel-1 RTC Radar (~20 features)

VV backscatter in dB, separated by orbit direction (ascending/descending):

- Lee speckle filtering applied before feature extraction
- Per orbit: yearly means (2020–2025), slope, max year-to-year drop
- Rolling 3-month std-of-stds (texture stability proxy)
- `mean_2020`, `mean_2026`, `drop_signal` per orbit

→ 2 orbits × ~10 stats = **~20 features**

### 3. AlphaEarth Foundation Embeddings (~395 features)

64-dimensional learned representations per pixel per year (2020–2025):

- **Raw**: 64 channels × 6 years = 384 features (flattened)
- **Derived**: 5 year-to-year cosine distances + 1 L2 norm change
- **PCA variants**: 8 PCA components + delta-PCA (4 components)
- `aef_cossim`: Cosine similarity between 2020 and latest embedding (top predictive feature)
- `aef_delta_pca0`: First principal component of embedding change

→ **~395 features**

### 4. External Priors (~17 features)

| Source | Features | Purpose |
|--------|----------|---------|
| ESA WorldCover 2020 | 11 land-cover classes (one-hot) | Forest mask baseline |
| Hansen GFC | `treecover2000`, `lossyear` (2021–2024) | Known historical deforestation |
| JRC TMF | 2020/2024 disturbance class | Tropical forest monitoring |
| Road distance | Distance to nearest road (km) | Accessibility proxy |

→ **~17 features**

## Feature Assembly

```python
# Per-tile feature matrix construction (simplified)
features = np.column_stack([
    extract_s2_features(tile),    # 84 features: spectral indices × temporal stats
    extract_s1_features(tile),    # 20 features: VV backscatter per orbit
    extract_aef_features(tile),   # 395 features: embeddings + change metrics
    extract_external(tile),       # 17 features: WorldCover, Hansen, JRC, roads
])
# Shape: (H*W, 508) — one row per pixel, cached as .npz per tile
```

## Label Generation (Training)

Ground truth from **consensus fusion** of 3 weak reference sources:

```
Positive: pixel flagged by ≥2 of {RADD, GLAD-S2, GLAD-L}
Negative: pixel flagged by 0 of 3 sources
Ambiguous: pixel flagged by exactly 1 source → excluded from training
```

Applied within forest mask (ESA WorldCover 2020 tree cover).

## Source Code

| File | Repository | Purpose |
|------|-----------|---------|
| `oasis/features/s2.py` | oasis-eklavya-v2 | Sentinel-2 feature extraction |
| `oasis/features/s1.py` | oasis-eklavya-v2 | Sentinel-1 feature extraction |
| `oasis/features/aef.py` | oasis-eklavya-v2 | AEF embedding features |
| `oasis/features/pack.py` | oasis-eklavya-v2 | Feature matrix assembly + caching |
| `oasis/labels/fusion.py` | oasis-eklavya-v2 | Consensus label generation |
| `src/features/` | oasis-luis-v4 | Luis feature extraction variants |

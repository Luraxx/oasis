# System Architecture

## End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW SATELLITE DATA                           │
│  Sentinel-2 L2A (12 bands, monthly, 2020–2026)                 │
│  Sentinel-1 RTC  (VV backscatter, ascending/descending)        │
│  AlphaEarth Foundation Embeddings (64d per year)                │
│  External: WorldCover, Hansen GFC, JRC TMF, Road Distance      │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               ① FEATURE ENGINEERING                              │
│                                                                  │
│  Per pixel: 508 features                                         │
│  ├── S2 spectral indices × temporal stats          (84 feat)    │
│  ├── S1 radar backscatter per orbit                (20 feat)    │
│  ├── AEF embeddings flattened + change metrics     (395 feat)   │
│  └── External priors (WorldCover, Hansen, JRC)     (17 feat)    │
│                                                                  │
│  Labels: Consensus fusion (≥2 of RADD/GLAD-S2/GLAD-L agree)    │
│  Validation: Leave-One-Region-Out (Amazon / Asia)               │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               ② MODEL TRAINING (3 families)                      │
│                                                                  │
│  EKLAVYA (5 models → calibrated stack)                          │
│  ├── LightGBM        (508 tabular features)                    │
│  ├── Temporal 1D-CNN  (19ch × 72 months)                       │
│  ├── U-Net EB3        (74ch × 256² patches)                    │
│  ├── U-Net R34        (same input, encoder diversity)          │
│  └── U-Net MiT-B1     (ViT-based encoder)                     │
│  → IsotonicRegression + LogisticRegression stacker              │
│  → Output: prob_{tile}.tif (OOF IoU: 71.67%)                   │
│                                                                  │
│  LUIS (2 models → ensemble)                                     │
│  ├── LightGBM        (49–63 features, LORO per-region)         │
│  └── U-Net EB4        (128² patches, FocalLoss)                │
│  → 50/50 average + Eklavya fusion (55/45)                      │
│  → Output: prob_{tile}.tif (OOF IoU: 57.99%)                   │
│                                                                  │
│  MARK (LightGBM + MLP → per-tile adaptive)                     │
│  ├── v1: 204-feature MLP (baseline)                            │
│  └── v2: LightGBM + 3-way ensemble with per-tile weights      │
│  → Output: prob_{tile}.tif (informing fusion strategy)         │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               ③ PROBABILITY FUSION                               │
│                                                                  │
│  fused = 0.60 × Eklavya + 0.40 × Luis                          │
│                                                                  │
│  Simple weighted average won over:                              │
│  - Learned meta-stackers (overfit to OOF)                      │
│  - Region-specific smart fusion (overfit)                       │
│  - Max/softunion/power-mean (too aggressive)                    │
│                                                                  │
│  Combined OOF IoU: 73.87% (+2.2pp over Eklavya alone)          │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               ④ POST-PROCESSING                                  │
│                                                                  │
│  Per-tile threshold (ultra-low: 0.20–0.28)                      │
│       ↓                                                          │
│  Morphological closing (8-conn, iter=2) — merge fragments       │
│       ↓                                                          │
│  Morphological dilation (4-conn, iter=2) — expand boundaries    │
│       ↓                                                          │
│  Binary erosion (4-conn, iter=4) — ★ KEY: tighten boundaries   │
│       ↓                                                          │
│  Area filter (≥ 0.10 ha)                                        │
│                                                                  │
│  Erosion insight: each 1m costs 0.5pp recall, saves 0.9pp FPR  │
│  → 4m erosion = optimal IoU (53.46% vs 52.64% without)         │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               ⑤ SUBMISSION                                       │
│                                                                  │
│  Vectorize binary mask → shapely polygons                       │
│  Reproject UTM → WGS84 (EPSG:4326)                             │
│  Assign time_step from NBR max-drop year (YYMM format)         │
│  Assign confidence from fused probability at centroid           │
│  Export as GeoJSON FeatureCollection                            │
│                                                                  │
│  Final: ~1100 polygons, ~4450 ha across 5 test tiles            │
│  Score: 53.46% Union IoU (4th place)                            │
└─────────────────────────────────────────────────────────────────┘
```

## Test Tiles

| Tile | EPSG | Region | Training Analogs | Challenge |
|------|------|--------|------------------|-----------|
| 18NVJ_1_6 | 32618 | Amazon | 8 Amazon train tiles | Cloud cover |
| 18NYH_2_1 | 32618 | Amazon | 8 Amazon train tiles | Cloud cover |
| 33NTE_5_1 | 32633 | Africa | **None** | Zero-shot transfer |
| 47QMA_6_2 | 32647 | SE Asia | 8 Asia train tiles | Diverse land use |
| 48PWA_0_6 | 32648 | SE Asia | 8 Asia train tiles | Diverse land use |

The **Africa tile (33NTE)** is the hardest — no training data from Africa exists, so all predictions rely on cross-continental generalization.

## Computational Requirements

- **Training**: AMD MI300X GPU (for U-Nets, TCN), CPU for LightGBM
- **Features**: ~4 hours for full feature extraction across 16 train + 5 test tiles
- **Training**: ~6 hours total (LGBM ~30min, TCN ~1hr, 3×UNet ~4.5hr)
- **Inference**: ~20 minutes for all 5 test tiles
- **Post-processing**: ~5 minutes for full sweep of configurations

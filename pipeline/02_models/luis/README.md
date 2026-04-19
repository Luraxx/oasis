# Luis Model Family — LightGBM + U-Net with LORO Validation

## Architecture

Two-model ensemble with progressive fusion into Eklavya's outputs.

```
Input: 49–63 features/pixel (S2 indices + S1 stats + AEF embeddings)
       ↓
┌──────────────────┬──────────────────────────────┐
│  LightGBM        │  U-Net (EfficientNet-B4)     │
│  300 rounds      │  128×128 patches             │
│  scale_pos=5     │  FocalLoss (α=0.75, γ=2.0)  │
│  LORO per-region │  Sliding window 50% overlap  │
└────────┬─────────┴──────────────┬───────────────┘
         ↓                        ↓
    50/50 weighted average → Luis probability map
         ↓
    55% Eklavya + 45% Luis → v4 fusion map
```

## Feature Set (49–63 features)

More compact than Eklavya, focused on change signals:

| Group | Count | Details |
|-------|-------|---------|
| S2 Spectral Indices | 30–42 | 6 indices × 5–7 temporal stats (mean_2020, std_2020, min_post, maxdrop, slope, [mean_2026, drop_2026]) |
| S1 Radar | 6–8 | 2 orbits × 3–4 stats (mean_2020, std_2020, maxdrop, [mean_2026]) |
| AEF Embeddings | 13 | 8 PCA components + cosine_sim + 4 delta-PCA components |

All features computed as **change relative to 2020 baseline**.

## Validation

**LORO (Leave-One-Region-Out):**
- Amazon fold: 8 tiles (18NW\*, 18NX\*, 18NY\*, 19NB\*)
- Asia fold: 8 tiles (47Q\*, 48P\*, 48Q\*)
- Separate LGBM models per region for geographic generalization

**OOF IoU**: 57.99%

## Evolution (v1 → v5)

| Version | Key Change | Impact |
|---------|------------|--------|
| v1 | Basic LGBM+UNet, global threshold 0.4 | Baseline (29.8% leaderboard) |
| v2 | Modular src/ package, LORO, 2026 data | Better organization |
| v4 | **Fusion with Eklavya** (55/45 weight) | Regional adaptation |
| v5 | **Meta-stacker** on 9 OOF predictions | Learned ensemble weights |

## Key Innovation: Adaptive Fallback

If a test tile produces zero polygons after thresholding, re-threshold at the tile's 99.5th percentile probability to ensure minimum signal. This recovered tile 47QMA from 0 → 73 polygons.

## Source Code

| File | Purpose |
|------|---------|
| `oasis-luis-v4/scripts/01_predict_v4.py` | Eklavya+Luis fusion prediction |
| `oasis-luis-v4/src/predict_lib.py` | Shared inference utilities |
| `oasis-luis-v4/scripts/02_train_lgbm.py` | LGBM training with LORO |
| `oasis-luis-v4/scripts/03_train_unet.py` | U-Net training |
| `oasis-luis-v5/scripts/01_train_stacker.py` | Meta-stacker (v5) |
| `oasis-luis-v5/scripts/02_predict_v5.py` | Stacker-based fusion |

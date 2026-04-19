# Eklavya Model Family — 5-Model Calibrated Stack

## Architecture

Five diverse models, each producing independent per-pixel probability maps, combined via isotonic calibration + logistic stacking.

```
Input: 508 features/pixel + temporal sequences
       ↓
┌──────────────┬─────────────────┬───────────────────────────────────┐
│  LightGBM    │  Temporal CNN   │  3× U-Net                        │
│  (tabular)   │  (1D per-pixel) │  (spatial segmentation)          │
│              │                 │  ├── EfficientNet-B3 encoder     │
│              │                 │  ├── ResNet34 encoder            │
│              │                 │  └── MiT-B1 (ViT) encoder       │
└──────┬───────┴────────┬────────┴──────────────┬──────────────────┘
       ↓                ↓                       ↓
  IsotonicRegression calibration per model
       ↓                ↓                       ↓
  LogisticRegression stacker → final calibrated probability
```

## Model Details

### LightGBM (Tabular)

- **Input**: 508 features per pixel (tabular)
- **Sampling**: 30,000 pixels/tile, pos:neg ratio 1.5:1
- **Config**: 2000 boosting rounds, 255 leaves, lr=0.03, feature_fraction=0.7
- **Validation**: LORO (Leave-One-Region-Out) — hold out Amazon or Asia
- **Top features**: `aef_cossim`, `aef_delta_pca0`, `s2_ndvi_maxdrop`

### Temporal 1D-CNN (TCN)

- **Input**: 19 channels × 72 monthly time steps per pixel
  - 12 S2 bands + 6 spectral indices + valid mask
- **Architecture**: 4 dilated convolutional blocks
- **Training**: 20,000 samples/tile, batch=4096, 12 epochs
- **Purpose**: Captures temporal deforestation dynamics pixel-by-pixel

### U-Net (Spatial Segmentation) — 3 variants

- **Input**: 74-channel tensor (yearly indices + S1 + AEF), 256×256 patches
- **Stride**: 128 (4× overlap, averaged at inference)
- **Loss**: BCE + Dice + Lovász (70/30 weight after 30% epochs)
- **TTA**: D4 augmentation (8× flip/rotate ensemble)
- **Encoders**: EfficientNet-B3, ResNet34, MiT-B1 (transformer) for diversity

### Calibrated Stack

- **Step 1**: IsotonicRegression per model — maps raw probabilities to calibrated probabilities using OOF predictions
- **Step 2**: LogisticRegression stacker — learns optimal combination weights from calibrated OOF
- **Regional stackers**: Separate stacker per region (Amazon/Asia) for adaptive weighting

## Validation Strategy

**Leave-One-Region-Out (LORO):**
- Fold 1: Train on Asia (8 tiles), validate on Amazon (8 tiles)
- Fold 2: Train on Amazon, validate on Asia
- Africa (33NTE test tile): No training analog → uses median thresholds

**OOF Performance**: 71.67% IoU (but 29% gap to test — domain shift on unseen tiles)

## Key Outputs

- Per-tile probability TIFs: `prob_{tile}.tif` (uint16, 0–1000 scale)
- OOF predictions: `artifacts/oof/{model}/{tile}.npy`

## Source Code

| File | Purpose |
|------|---------|
| `oasis-eklavya-v2/oasis/models/lgbm.py` | LightGBM model |
| `oasis-eklavya-v2/oasis/models/tcn.py` | Temporal CNN model |
| `oasis-eklavya-v2/oasis/models/unet.py` | U-Net model |
| `oasis-eklavya-v2/oasis/ensemble.py` | Isotonic calibration + stacking |
| `oasis-eklavya-v2/scripts/train_lgbm.py` | LGBM training script |
| `oasis-eklavya-v2/scripts/train_tcn.py` | TCN training script |
| `oasis-eklavya-v2/scripts/train_unet.py` | U-Net training script |
| `oasis-eklavya-v2/scripts/fit_ensemble.py` | Calibration + stacking |
| `oasis-eklavya-v2/scripts/infer_test.py` | Final test inference |

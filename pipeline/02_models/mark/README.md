# Mark Model Family — Per-Pixel MLP + LightGBM with Per-Tile Adaptation

## Architecture

Two iterations: v1 (MLP baseline) → v2 (LightGBM + 3-way ensemble).

### v1: Per-Pixel MLP

```
Input: 204 features/pixel
  ├── AEF Embeddings (193d): 2020 baseline + year + difference + cosine + L2
  ├── Sentinel-1 Stats (10d): mean/min VV per orbit for 2020 + target year
  ├── Sentinel-2 Stats (9d): NDVI, NBR, clear_frac × 2020/target/drops
  └── Forest Mask (1d): binary flag
       ↓
  3-layer MLP → BCEWithLogitsLoss → per-pixel probability
```

- **Training**: All positives (~20K/tile-year) + 40K random negatives
- **Labels**: Soft consensus (mean of available sources × forest mask)
- **Threshold**: Global 0.375 | F1 = 0.789

### v2: LightGBM + 3-Way Ensemble

```
Mark's new LightGBM (CMR model)
       ↓
Ensemble fusion (per-tile adaptive weights):
  final_prob = w_ekl × ekl + w_luis × luis + w_cmr × cmr
       ↓
Per-tile Gaussian smoothing + morphological ops
```

**Per-tile weight optimization** (key innovation):
- 47QMA: 15% ekl / 10% luis / **75% cmr** (Mark model dominates)
- 33NTE: Balanced weights + strong Gaussian smoothing
- 18NYH: Higher threshold to reduce false positives

**LightGBM config**: 63 leaves, 500 min_data_in_leaf, 10:1 neg subsampling, tile-holdout validation.

## Key Contribution

Mark's approach provided the **CMR (Cameroon) specialized model** and **per-tile weight tuning** methodology that informed the broader ensemble strategy. The mega_t40 variant scored 50.25% (Try 9).

## Source Code

| File | Purpose |
|------|---------|
| `oasis-mark/scripts/` | v1 MLP training and inference |
| `oasis-mark-2/scripts/build_ultimate.py` | v2 3-way ensemble with per-tile tuning |
| `oasis-mark-2/scripts/build_final_submissions.py` | Submission variant generation |

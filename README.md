# Team Oasis — Deforestation Detection Pipeline

**osapiens Makeathon Challenge 2026**  
**Final Ranking: 4th Place — 53.46% Union IoU**

---

## Challenge

Detect **post-2020 deforestation** across 5 test tiles spanning 3 continents (Amazon, Southeast Asia, Central Africa) using Sentinel-1/2 time series, AlphaEarth Foundation (AEF) embeddings, and weak reference labels (RADD, GLAD-S2, GLAD-L).

**Evaluation metric:** Union IoU — intersection-over-union of predicted vs ground-truth deforestation polygons, weighted by polygon area.

## Our Approach

A **7-model multi-stage ensemble** combining three independently developed model families, fused at the probability map level with optimized post-processing.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE OVERVIEW                                │
│                                                                         │
│  Raw Data (S1, S2, AEF, External)                                      │
│       ↓                                                                 │
│  ① Feature Engineering  (~508 features/pixel)                          │
│       ↓                                                                 │
│  ② Model Training (3 independent model families)                       │
│       ├── Eklavya: LightGBM + TCN + 3×U-Net → Calibrated Stack        │
│       ├── Luis:    LightGBM + U-Net → LORO Ensemble                   │
│       └── Mark:    LightGBM + MLP → Per-Tile Adaptive                 │
│       ↓                                                                 │
│  ③ Probability Fusion  (60% Eklavya + 40% Luis weighted avg)          │
│       ↓                                                                 │
│  ④ Post-Processing  (threshold → morphology → 4m erosion)             │
│       ↓                                                                 │
│  ⑤ Vectorization + Year Estimation → GeoJSON Submission                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Results

| Try | Score | Method | Recall | FPR | Year |
|-----|-------|--------|--------|-----|------|
| **15** | **53.46%** | Ekl+Luis wavg 60/40, ultra-low thresholds, 4m erosion | 69.3% | 30.0% | 8.2% |
| 11 | 52.64% | Same as above, without erosion | 71.8% | 33.6% | 7.7% |
| 9 | 50.25% | Mark's 3-way ensemble (ekl+luis+cmr) | 56.9% | 18.9% | 12.8% |

**Leaderboard (final):**

| Rank | Team | Score |
|------|------|-------|
| 1 | Non Deterministic | 54.29% |
| 2 | OMEGA-EARTH | 54.23% |
| 3 | Aguacates | 54.05% |
| **4** | **Oasis** | **53.46%** |

Gap to #1: 0.83 percentage points.

## Repository Structure

```
oasis-final/
├── challenge-repo/              # Cloned osapiens challenge repo
├── pipeline/
│   ├── ★ run_pipeline.py        # RUNNABLE: Reproduziert Try 15 in ~36s
│   ├── ★ config.py              # Alle Pfade, Tiles, Hyperparameter
│   ├── 01_features/             # Feature engineering documentation
│   │   └── features.md          # All 508 features explained
│   ├── 02_models/               # Complete model source code + docs
│   │   ├── eklavya/             # 5-model ensemble (LGBM, TCN, 3×UNet)
│   │   │   ├── oasis/           # Core ML package (features, models, ensemble)
│   │   │   │   ├── features/    # S1, S2, AEF feature extractors
│   │   │   │   ├── labels/      # Consensus label fusion
│   │   │   │   └── models/      # LightGBM, TCN, U-Net implementations
│   │   │   ├── scripts/         # Training & inference scripts
│   │   │   └── train_and_predict.py
│   │   ├── luis/                # LightGBM + UNet with LORO validation
│   │   │   ├── src/             # Feature extraction, postprocessing, timing
│   │   │   └── scripts/         # Cache, train, predict pipeline
│   │   └── mark/                # LightGBM + MLP per-tile adaptive
│   │       ├── src/             # Data loading, model training
│   │       └── scripts/         # Ensemble building, CMR probs, fusion
│   ├── 03_ensemble/             # Probability fusion & sweeps (14 scripts)
│   │   ├── src/config.py        # Tile definitions, paths, regions
│   │   ├── 13_v11_final_push.py # ★ Generates Try 11 base (52.64%)
│   │   ├── 14_v12_polygon_quality.py  # ★ Erosion → Try 15 best (53.46%)
│   │   └── fusion.md            # Fusion strategy documentation
│   ├── 04_postprocessing/       # Morphology, erosion, area filtering
│   │   ├── postprocess_luis.py  # Luis postprocessing code
│   │   ├── predict_eklavya.py   # Eklavya vectorization code
│   │   ├── timing_luis.py       # Year estimation from NBR
│   │   └── postprocessing.md    # Full postprocessing documentation
│   ├── 05_submission/           # Final GeoJSON generation
│   │   ├── 13_v11_final_push.py # Vectorization + year estimation
│   │   ├── 14_v12_polygon_quality.py  # Erosion variants
│   │   └── submission.md        # Submission format documentation
│   ├── output/                  # Generated outputs
│   ├── submission_utils.py      # Challenge evaluation utilities
│   └── requirements.txt         # Python dependencies
├── docs/
│   ├── architecture.md          # Full system architecture diagram
│   ├── lessons_learned.md       # What worked, what didn't
│   └── score_history.md         # All 17 submission attempts with source refs
└── README.md                    # This file
```

## Quick Links

- [Full Architecture](docs/architecture.md) — end-to-end system diagram
- [Feature Engineering](pipeline/01_features/features.md) — all 508 features explained
- [Eklavya Models](pipeline/02_models/eklavya/README.md) — 5-model calibrated stack
- [Luis Models](pipeline/02_models/luis/README.md) — LGBM + UNet with LORO
- [Mark Models](pipeline/02_models/mark/README.md) — per-tile adaptive ensemble
- [Ensemble Fusion](pipeline/03_ensemble/fusion.md) — probability map combination
- [Post-Processing](pipeline/04_postprocessing/postprocessing.md) — morphology + erosion
- [Submission Format](pipeline/05_submission/submission.md) — vectorization + year estimation
- [Score History](docs/score_history.md) — all 17 tries with source code references
- [Lessons Learned](docs/lessons_learned.md) — what worked and what didn't

## Reproducing the Best Submission (Try 15)

### Quick: Run the Pipeline (requires pre-trained probability maps)

```bash
cd pipeline/

# Reproduziert Try 15 (53.46% IoU) — dauert ~36 Sekunden
python run_pipeline.py

# Ohne Erosion → Try 11 Basis (52.64% IoU)
python run_pipeline.py --no-erosion

# Nur ein Tile
python run_pipeline.py --tiles 18NVJ_1_6

# Dry-run (nur Statistiken, kein Output)
python run_pipeline.py --dry-run
```

Output: `pipeline/output/try15_best/submission.geojson`

**Was passiert dabei?**
1. Lädt Probability Maps (Eklavya + Luis) für jedes Tile
2. Fusioniert: 60% Eklavya + 40% Luis (gewichteter Durchschnitt)
3. Binarisiert mit ultra-niedrigen Schwellenwerten (0.20–0.28 pro Tile)
4. Morphologie: Close(2) → Dilate(2) → Erode(4)
5. Year Estimation via NBR-Drop auf Sentinel-2 Zeitreihe
6. Vektorisierung → GeoJSON mit Polygonen in EPSG:4326

Alle Parameter sind in `config.py` dokumentiert und erklärbar.

### Full: Train Models from Scratch (separate environments)
```bash
# Eklavya: 5-model ensemble
cd pipeline/02_models/eklavya
python scripts/build_cache.py          # Feature extraction → .npz per tile
python scripts/train_lgbm.py           # LightGBM (LORO)
python scripts/train_tcn.py            # Temporal CNN
python scripts/train_unet.py           # 3× U-Net (EB3, R34, MiT-B1)
python scripts/fit_ensemble.py         # Isotonic calibration + stacking
python scripts/infer_test.py           # → prob_{tile}.tif

# Luis: LGBM + UNet
cd pipeline/02_models/luis
python scripts/02_build_cache.py       # Feature extraction
python scripts/03_train_lgbm.py        # LightGBM (LORO)
python scripts/04_train_unet.py        # U-Net (EB4)
python scripts/05_predict.py           # → prob_{tile}.tif
python scripts/01_predict_v4.py        # Ekl+Luis fusion → prob_{tile}.tif
```

### Stage 2: Ensemble Fusion + Post-Processing
```bash
cd pipeline/03_ensemble
# 13_v11_final_push.py: Fuses Ekl (60%) + Luis (40%), applies ultra-low
# thresholds, morphological close=2 + dilate=2 → Try 11 base (52.64%)
python 13_v11_final_push.py

# 14_v12_polygon_quality.py: Applies 4m erosion to Try 11 → Try 15 (53.46%)
python 14_v12_polygon_quality.py
```

### Stage 3: Submit
```bash
# Output: submission/v13_shrink/erode4_raw/submission.geojson
# Contains ~1100 polygons across 5 test tiles
```

## Team

Built by Team Oasis — Eklavya, Luis, Mark, Afrika — during the osapiens Makeathon 2026.

## Source Repositories (original working directories)

| Component | Directory | Purpose |
|-----------|-----------|---------|
| Eklavya v2 | `oasis-eklavya-v2/` | 5-model ensemble training & inference |
| Luis v4 | `oasis-luis-v4/` | LightGBM + UNet with LORO & Eklavya fusion |
| Luis v5 | `oasis-luis-v5/` | Final ensemble fusion & postprocessing |
| Mark v2 | `oasis-mark-2/` | Per-tile adaptive ensemble |
| Afrika | `oasis-afrika/` | Africa-specific LightGBM with Hansen features |

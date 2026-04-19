# Submission Score History

**Max Tries: 20** | Used: 17/20 | Remaining: 3

## All Submissions

| # | Score | Recall | FPR | Year | Method | Source |
|---|-------|--------|-----|------|--------|--------|
| **15** | **53.46%** 🏆 | 69.34% | 29.99% | 8.23% | Ekl+Luis wavg 60/40, ultra-low thresholds, close=2, dil=2, **4m erosion** | `oasis-luis-v5/scripts/13_v11_final_push.py` + `14_v12_polygon_quality.py` |
| 14 | 53.30% | 70.10% | 31.00% | 8.10% | Same as 15 but 3m erosion | `oasis-luis-v5/scripts/14_v12_polygon_quality.py` |
| 13 | 52.87% | 71.42% | 32.94% | 7.78% | Try 11 + prune <0.20ha + 1m erosion | `oasis-luis-v5/scripts/14_v12_polygon_quality.py` |
| 12 | 52.34% | 72.59% | 34.77% | 7.52% | Try 11 + Douglas-Peucker simplify 8m | `oasis-luis-v5/scripts/14_v12_polygon_quality.py` |
| 11 | 52.64% | 71.75% | 33.59% | 7.69% | `E_ultralow_wavg_dil2` — wavg 60/40, ultra-low per-tile thresholds, close=2, dil=2 | `oasis-luis-v5/scripts/13_v11_final_push.py` |
| 10 | 50.43% | 64.66% | 30.38% | 0.00% | `B_aggressive_dil2` — lower thresholds, close=2, dil=2 | `oasis-luis-v5/scripts/13_v11_final_push.py` |
| 9 | 50.25% | 56.91% | 18.89% | 12.77% | Mark's `mega_t40` — 3-way ensemble (ekl+luis+cmr), per-tile adaptive | `oasis-mark-2/scripts/build_ultimate.py` |
| 8 | 48.37% | 53.27% | 15.97% | 11.72% | Ekl+Luis 60/40, t=0.45, close=1, open=1, min_ha=0.25 | `oasis-luis-v5/scripts/12_v10_wavg_finetune.py` |
| 7 | 29.00% | 38.67% | 46.28% | 0.00% | Mark v1 standalone | `oasis-mark/` |
| 6 | 41.70% | 46.49% | 19.84% | 20.50% | v5 + Afrika LightGBM ensemble on Africa tile | `oasis-afrika/train_afrika.py` |
| 5 | 44.84% | 47.52% | 11.15% | 25.70% | Ekl+Luis 60/40, t=0.59, close=1, open=2, min_ha=0.50 | `oasis-luis-v5/scripts/02_predict_v5.py` |
| 4 | 35.53% | 39.23% | 20.95% | 0.00% | Mark v2 standalone | `oasis-mark-2/` |
| 3 | 42.30% | 45.80% | 15.28% | 14.37% | Eklavya 5-model stack standalone | `oasis-eklavya-v2/scripts/infer_test.py` |
| 2 | 17.39% | 18.14% | 19.16% | 11.16% | Luis v2 standalone | `oasis-luis-v2/` |
| 1 | 29.80% | 36.26% | 37.41% | 0.00% | Luis v1 baseline | `oasis-luis/` |

(Tries 16, 17 were experimental and scored 49.76% and 47.34% respectively.)

## Score Progression Chart

```
Score %
55 ┤
54 ┤                                                    ●━━ #1 Non Deterministic (54.29%)
   ┤                                         ■ 15 (53.46%) ← OUR BEST
53 ┤                                    ■ 14
   ┤                               ■ 13  ■ 11
52 ┤                              ■ 12
   ┤
51 ┤
   ┤                         ■ 10
50 ┤                     ■ 9
   ┤
49 ┤
   ┤                ■ 8
48 ┤
   ┤
   ┤
45 ┤           ■ 5
   ┤
   ┤       ■ 3
42 ┤      ■ 6
   ┤
   ┤  ■ 4
35 ┤
   ┤
   ┤■ 1  ■ 7
30 ┤
   ┤
   ┤
   ┤■ 2
17 ┤
   └──────────────────────────────────────────────────────
    1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
                        Try #
```

## Key Transitions

- **Try 1→3** (+12.5pp): Model quality (Luis baseline → Eklavya 5-model stack)
- **Try 3→5** (+2.5pp): Fusion (Eklavya standalone → Ekl+Luis 60/40 ensemble)
- **Try 5→8** (+3.5pp): Threshold tuning (conservative → aggressive)
- **Try 8→11** (+4.3pp): Ultra-low thresholds + morphological dilation
- **Try 11→15** (+0.8pp): Boundary erosion optimization

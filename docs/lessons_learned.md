# Lessons Learned

## What Worked

### 1. Model Diversity > Model Complexity
Combining three independently developed model families (Eklavya 5-model stack, Luis 2-model LORO, Mark per-tile adaptive) provided genuine ensemble diversity. Each team member made different architectural choices, feature selections, and validation strategies — this is the kind of diversity that single-team ensembles struggle to achieve.

### 2. Simple Fusion Beats Sophisticated Methods
We tested learned meta-stackers, region-specific smart fusion, max/softunion/power-mean — but a **simple 60/40 weighted average** of two probability maps won. Complex methods overfit to OOF patterns that didn't transfer to test tiles (29% OOF→test gap).

### 3. Post-Processing is Underrated
The difference between Try 11 (52.64%) and Try 15 (53.46%) was **entirely post-processing** — no model retraining, no new features, just 4 meters of binary erosion. This is a nearly free +0.82pp improvement.

### 4. Dilation Then Erosion > Neither
Counterintuitively, dilating boundaries by 20m then eroding by 40m outperformed doing nothing. The dilation captures boundary pixels that thresholding misses, while erosion removes the false positive fringe. The net effect is tighter, more accurate boundaries.

### 5. Ultra-Low Thresholds + Erosion = High IoU
Conventional wisdom says to find the "right" threshold. Instead, we used very low thresholds (0.20–0.28) to maximize recall, then used morphological erosion to remove the FPR. This approach works because:
- Low thresholds capture all true positives + some boundary noise
- Erosion preferentially removes the noise (thin boundary artifacts erode away, but large true positive regions survive)

### 6. LORO Validation Matters
Leave-One-Region-Out validation was essential for honest model assessment. Per-pixel random splits grossly overestimate performance due to spatial autocorrelation.

## What Didn't Work

### 1. 3-Way Ensemble with CMR Model (Try 17: 47.34%)
Adding Mark's CMR (Cameroon-specialized) model as a third voice in the ensemble — with ultra-low thresholds — pushed recall to 76.63% but FPR exploded to 44.67%. The CMR model agreed with the other models on true positives but added many unique false positives.

### 2. Afrika-Specific LightGBM with Hansen Pseudo-Labels
Training a specialized model for the Africa tile using Hansen forest loss as pseudo-labels. The Hansen data is noisy and doesn't match the challenge's deforestation definition well enough.

### 3. Polygon Simplification (Try 12: 52.34%)
Douglas-Peucker simplification at 8m tolerance was supposed to smooth raster staircase edges. Instead it slightly worsened IoU by distorting polygon shapes.

### 4. Year Estimation Accuracy
Our year predictions scored only 8.23% (vs Aguacates' 15.3%). The NBR max-drop heuristic is noisy — small changes in cloud coverage can shift the detected year. Monthly-level analysis would likely improve this but we ran out of tries.

### 5. Learned Meta-Stackers
Training a logistic regression stacker on OOF predictions from all 9 base models (6 Eklavya + 3 Luis). Performance was good on OOF (73.87%) but didn't generalize — the stacker learned OOF-specific patterns rather than robust fusion rules.

## Key Insights for Future Work

1. **The OOF→Test gap is the main challenge**: 71.67% OOF vs 42.30% initial test score. Domain shift between train tiles and test tiles (especially zero-shot Africa) is the bottleneck, not model capacity.

2. **Post-processing should be tuned as carefully as models**: We found that the erosion sweep (0–10m) had as much impact as months of model development. Budget time for systematic post-processing optimization.

3. **The recall/FPR trade-off has a sharp optimum**: At 4m erosion, each additional meter costs 0.5pp recall and saves 0.9pp FPR. At 5m, recall crashes. This cliff behavior means the optimal point is narrow.

4. **Boundary quality matters more than detection count**: The top teams (Non Deterministic, OMEGA-EARTH) had similar or lower recall than us but better FPR — their polygon boundaries were more precisely aligned.

## Score Evolution

| Try | Score | Key Change | Lesson |
|-----|-------|-----------|---------|
| 1 | 29.80% | Luis v1 baseline | Starting point |
| 3 | 42.30% | Eklavya 5-model stack | Model quality matters |
| 5 | 44.84% | 60/40 Ekl+Luis ensemble | Fusion helps |
| 8 | 48.37% | Lower thresholds + less opening | More recall = more IoU |
| 9 | 50.25% | Mark's per-tile tuning | Per-tile adaptation |
| 11 | 52.64% | Ultra-low thresholds + dilation | Aggressive expansion |
| **15** | **53.46%** | **+4m erosion** | **Post-processing FTW** |
| 17 | 47.34% | 3-way ensemble, too aggressive | More models ≠ better |

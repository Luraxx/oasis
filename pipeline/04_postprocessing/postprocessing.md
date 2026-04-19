# Post-Processing Pipeline

## Overview

After probability fusion, the continuous probability map is converted to binary deforestation masks through thresholding, morphological operations, erosion, and area filtering.

```
Fused probability map [0.0 – 1.0]
       ↓
① Per-tile thresholding → binary mask
       ↓
② Morphological closing (fill gaps between fragments)
       ↓
③ Morphological dilation (expand polygon boundaries)
       ↓
④ Binary erosion (shrink boundaries to reduce FPR) ← KEY STEP
       ↓
⑤ Area filtering (remove components < 0.10 ha)
       ↓
Binary deforestation mask
```

## Step 1: Per-Tile Thresholding

Ultra-low thresholds to maximize recall (boundary quality fixed later by erosion):

| Tile | Region | Threshold | Rationale |
|------|--------|-----------|-----------|
| 18NVJ_1_6 | Amazon | 0.20 | Strong S2 signal, low cloud |
| 18NYH_2_1 | Amazon | 0.28 | More cloud interference |
| 33NTE_5_1 | Africa | 0.25 | No training data — median fallback |
| 47QMA_6_2 | Asia | 0.20 | Strong radar signal |
| 48PWA_0_6 | Asia | 0.28 | More variable landscape |

## Step 2: Morphological Closing (iterations=2)

Fills small holes and bridges narrow gaps between fragments:

```python
# 8-connected structuring element (diagonal connections)
s8 = ndimage.generate_binary_structure(2, 2)
result = ndimage.binary_closing(result, s8, iterations=2)
```

**Effect**: Merges nearby deforestation patches that should be a single clearing.

## Step 3: Morphological Dilation (iterations=2)

Expands polygon boundaries outward by ~20m:

```python
# 4-connected structuring element (cardinal directions only)
s4 = ndimage.generate_binary_structure(2, 1)
result = ndimage.binary_dilation(result, s4, iterations=2)
```

**Effect**: Captures boundary pixels missed by thresholding. At 10m resolution, each iteration adds ~10m to boundaries. This step was critical — dilation alone added +2pp IoU by improving boundary alignment with ground truth polygons.

## Step 4: Binary Erosion — THE KEY INNOVATION (iterations=4)

Shrinks polygon boundaries inward by ~40m:

```python
s4 = ndimage.generate_binary_structure(2, 1)
result = ndimage.binary_erosion(result, s4, iterations=4)
```

**This is the step that made our best submission.** After aggressive dilation, boundaries overshoot ground truth. Erosion tightens them back:

| Erosion | IoU | Recall | FPR | Delta IoU |
|---------|-----|--------|-----|-----------|
| 0m (Try 11) | 52.64% | 71.75% | 33.59% | baseline |
| 1m (Try 13) | 52.87% | 71.42% | 32.94% | +0.23pp |
| 3m (Try 14) | 53.30% | 70.10% | 31.00% | +0.66pp |
| **4m (Try 15)** | **53.46%** | **69.34%** | **29.99%** | **+0.82pp** |
| 5m (Try 16) | 49.76% | 62.06% | 28.48% | −2.88pp |

**Trade-off**: Each meter of erosion costs ~0.5pp recall but saves ~0.9pp FPR. At 4m, the IoU gain from FPR reduction outweighs the recall loss. At 5m, too much area is lost and polygons start disappearing entirely.

## Step 5: Area Filtering

Remove connected components smaller than 0.10 hectares (~10 pixels at 10m):

```python
labeled, n = ndimage.label(result)
min_px = int(0.10 / 0.01)  # 0.01 ha per pixel at 10m
for i in range(1, n + 1):
    if (labeled == i).sum() < min_px:
        result[labeled == i] = 0
```

## Full Post-Processing Code

```python
def postprocess(binary, close_iter=2, open_iter=0, dilate_iter=2, 
                erode_iter=4, min_area_ha=0.10):
    result = binary.copy()
    
    # Close (merge fragments, 8-connected)
    if close_iter > 0:
        s8 = ndimage.generate_binary_structure(2, 2)
        result = ndimage.binary_closing(result, s8, iterations=close_iter)
    
    # Open (remove noise, 4-connected) — not used in best submission
    if open_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_opening(result, s4, iterations=open_iter)
    
    # Dilate (expand boundaries, 4-connected)
    if dilate_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_dilation(result, s4, iterations=dilate_iter)
    
    # Erode (shrink boundaries, 4-connected) — KEY STEP
    if erode_iter > 0:
        s4 = ndimage.generate_binary_structure(2, 1)
        result = ndimage.binary_erosion(result, s4, iterations=erode_iter)
    
    # Area filter
    labeled, n = ndimage.label(result)
    min_px = int(min_area_ha / 0.01)
    for i in range(1, n + 1):
        if (labeled == i).sum() < min_px:
            result[labeled == i] = 0
    
    return result.astype(np.uint8)
```

## Source Code

| File | Purpose |
|------|---------|
| `oasis-luis-v5/scripts/14_v12_polygon_quality.py` | Erosion sweep experiments |
| `oasis-luis-v5/scripts/13_v11_final_push.py` | Base postprocessing (close + dilate) |
| `oasis-eklavya-v2/oasis/postprocess.py` | Original morphology pipeline |

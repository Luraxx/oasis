# Probability Fusion — Combining Model Families

## Strategy

The final submission uses a **weighted average** of two calibrated probability maps:

```
fused_prob = 0.60 × Eklavya_prob + 0.40 × Luis_prob
```

### Why 60/40?

OOF analysis on 16 training tiles showed:
- Eklavya alone: 71.67% OOF IoU (5-model calibrated stack, stronger)
- Luis alone: 57.99% OOF IoU (2-model, more compact)
- Combined 60/40: **73.87% OOF IoU** (+2.2pp over Eklavya alone)

The models make **complementary errors** — Luis catches some events Eklavya misses (especially in Asia) and vice versa.

## Fusion Methods Explored

| Method | Formula | Best Score | Notes |
|--------|---------|------------|-------|
| **wavg_60_40** | 0.6·E + 0.4·L | **53.46%** | Optimal, used in final |
| wavg_50_50 | 0.5·E + 0.5·L | ~51% | Too much weight on weaker model |
| max | max(E, L) | ~47% | High recall but FPR explodes |
| softunion | E + L − E·L | ~46% | Similar to max, too aggressive |
| power_mean_3 | ((E³+L³)/2)^(1/3) | ~48% | Favors higher value |
| smart_fusion | Region-specific rules | ~50% | Overfit to OOF patterns |
| meta_stacker | Learned LogReg on OOF | ~49% | Didn't generalize well to test |
| two-stage | wavg base + max marginal | ~51% | Overcomplicated |

**Key lesson**: Simple weighted average beat all sophisticated fusion methods on the test set. The OOF→test gap (29%) means complex methods overfit to training distribution.

## Input Probability Maps

| Model | File | Scale | Resolution |
|-------|------|-------|------------|
| Eklavya | `ekl_submission/prob_{tile}.tif` | uint16, 0–1000 | 10m |
| Luis v4 | `luis_v4_submission/prob_{tile}.tif` | uint16, 0–1000 | 10m |

Both maps are aligned to the same UTM grid per tile. Minor shape mismatches are handled by cropping to the smaller dimensions.

## Fusion Code

```python
def load_prob(path):
    """Load probability TIF, normalize to 0-1 float."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    if data.max() > 10:  # stored as uint16 × 1000
        data /= 1000.0
    return data, profile

def fuse(ekl, luis):
    """Weighted average fusion — simple and robust."""
    if luis is None:
        return ekl
    h = min(ekl.shape[0], luis.shape[0])
    w = min(ekl.shape[1], luis.shape[1])
    return 0.60 * ekl[:h, :w] + 0.40 * luis[:h, :w]
```

## Source Code

| File | Purpose |
|------|---------|
| `oasis-luis-v5/scripts/13_v11_final_push.py` | Main fusion + sweep logic |
| `oasis-luis-v5/scripts/05_smart_fusion.py` | Region-specific fusion experiments |
| `oasis-luis-v5/src/config.py` | Tile definitions, paths, region mapping |

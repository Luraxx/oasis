"""Local validation: compute IoU of a submission directory against weak labels.
This is an approximation since weak labels != ground truth, but it helps
compare variants before spending submission tries.
"""
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from datetime import date
import os, sys, json
from pathlib import Path

HANSEN_DIR = '/shared-docker/oasis-mark-2/external/hansen/cropped'
LABEL_DIR = '/shared-docker/oasis-mark-2/external/makeathon-extras/labels/train'

TEST_TILES = ['18NVJ_1_6', '18NYH_2_1', '33NTE_5_1', '47QMA_6_2', '48PWA_0_6']
RADD_EPOCH = date(2014, 12, 31)
POST_LOW = (date(2020, 12, 31) - RADD_EPOCH).days
POST_HIGH = (date(2025, 12, 31) - RADD_EPOCH).days

def reproj(p, meta, dtype=np.uint8):
    if not os.path.exists(p): return None
    dst = np.zeros((meta['height'], meta['width']), dtype=dtype)
    with rasterio.open(p) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=meta['transform'], dst_crs=meta['crs'],
                  resampling=Resampling.nearest)
    return dst

def load_labels(tile, meta, h, w):
    """Load and union weak labels for a tile."""
    ly = reproj(f'{HANSEN_DIR}/{tile}_lossyear.tif', meta)
    tc = reproj(f'{HANSEN_DIR}/{tile}_treecover2000.tif', meta)
    hansen = np.zeros((h,w), bool)
    if ly is not None and tc is not None:
        ly, tc = ly[:h,:w], tc[:h,:w]
        hansen = (tc >= 30) & (ly >= 21) & (ly <= 24)

    radd_raw = reproj(f'{LABEL_DIR}/radd/radd_{tile}_labels.tif', meta, dtype=np.int32)
    radd = np.zeros((h,w), bool)
    if radd_raw is not None:
        radd_raw = radd_raw[:h,:w]
        days = radd_raw % 10000
        conf = radd_raw // 10000
        radd = (conf >= 2) & (days > POST_LOW) & (days <= POST_HIGH)

    gladl = np.zeros((h,w), bool)
    for yy in [21,22,23,24,25]:
        gp = f'{LABEL_DIR}/gladl/gladl_{tile}_alert{yy:02d}.tif'
        gc = reproj(gp, meta, dtype=np.uint8)
        if gc is not None:
            gladl |= (gc[:h,:w] >= 2)

    return hansen, radd, gladl

def compute_metrics(pred, hansen, radd, gladl):
    """Compute IoU against different label combinations."""
    any_label = hansen | radd | gladl
    n_agree = hansen.astype(int) + radd.astype(int) + gladl.astype(int)
    consensus_2 = n_agree >= 2

    # IoU vs any label
    inter_any = (pred & any_label).sum()
    union_any = (pred | any_label).sum()
    iou_any = inter_any / max(union_any, 1)

    # IoU vs 2+ consensus
    inter_c2 = (pred & consensus_2).sum()
    union_c2 = (pred | consensus_2).sum()
    iou_c2 = inter_c2 / max(union_c2, 1)

    # Precision / Recall vs any
    tp = (pred & any_label).sum()
    fp = (pred & ~any_label).sum()
    fn = (any_label & ~pred).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        'iou_any': float(iou_any),
        'iou_c2': float(iou_c2),
        'precision': float(precision),
        'recall': float(recall),
        'pred_px': int(pred.sum()),
        'label_px': int(any_label.sum()),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
    }


def validate_submission(sub_dir):
    """Validate a submission directory."""
    sub_dir = Path(sub_dir)
    print(f'\n=== Validating: {sub_dir.name} ===')

    all_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    total_pred_union, total_label_union = 0, 0

    for tile in TEST_TILES:
        pred_path = sub_dir / f'pred_{tile}.tif'
        if not pred_path.exists():
            print(f'  [{tile}] MISSING')
            continue

        with rasterio.open(pred_path) as src:
            pred = src.read(1).astype(bool)
            meta = {'height': src.height, 'width': src.width,
                    'transform': src.transform, 'crs': src.crs}

        h, w = pred.shape
        hansen, radd, gladl = load_labels(tile, meta, h, w)
        m = compute_metrics(pred, hansen, radd, gladl)
        all_metrics[tile] = m

        total_tp += m['tp']
        total_fp += m['fp']
        total_fn += m['fn']

        any_label = hansen | radd | gladl
        total_pred_union += int(pred.sum())
        total_label_union += int(any_label.sum())

        print(f'  [{tile}] IoU_any={m["iou_any"]:.3f} IoU_c2={m["iou_c2"]:.3f} '
              f'P={m["precision"]:.3f} R={m["recall"]:.3f} '
              f'pred={m["pred_px"]:,} labels={m["label_px"]:,}')

    # Overall
    overall_iou = total_tp / max(total_tp + total_fp + total_fn, 1)
    overall_p = total_tp / max(total_tp + total_fp, 1)
    overall_r = total_tp / max(total_tp + total_fn, 1)

    print(f'\n  OVERALL: IoU={overall_iou:.3f} P={overall_p:.3f} R={overall_r:.3f}')
    print(f'  Total: pred={total_pred_union:,} labels={total_label_union:,}')
    return overall_iou, all_metrics


if __name__ == '__main__':
    dirs = sys.argv[1:]
    if not dirs:
        # Compare multiple submission dirs
        sub_root = Path('/shared-docker/oasis-mark-2/submissions')
        dirs = ['ultimate_v1', 'try11_repro', 'tri_mega_mid', 'smart_t40_n20']

    results = {}
    for d in dirs:
        p = Path(d) if Path(d).is_absolute() else Path('/shared-docker/oasis-mark-2/submissions') / d
        if p.exists():
            iou, metrics = validate_submission(p)
            results[d] = iou

    print('\n=== COMPARISON ===')
    for name, iou in sorted(results.items(), key=lambda x: -x[1]):
        print(f'  {name}: IoU={iou:.4f}')

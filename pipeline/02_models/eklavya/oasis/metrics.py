"""Binary classification metrics restricted to consensus-truth pixels.

Every reported number in this project must be computable from a
probability raster and a consensus mask. We never claim a number against
the noisy fused training labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinaryReport:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    iou: float
    n: int

    def to_dict(self) -> dict:
        return self.__dict__


def binary_report(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> BinaryReport:
    y_true = y_true.astype(np.uint8)
    pred = (prob >= threshold).astype(np.uint8)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return BinaryReport(
        threshold=threshold,
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, iou=iou,
        n=int(y_true.size),
    )


def best_threshold_f1(
    y_true: np.ndarray, prob: np.ndarray, *, lo: float = 0.05, hi: float = 0.95, step: float = 0.02
) -> tuple[float, BinaryReport]:
    """Sweep thresholds and return the (threshold, report) with best F1."""
    grid = np.arange(lo, hi + 1e-9, step)
    best = None
    for t in grid:
        rep = binary_report(y_true, prob, float(t))
        if best is None or rep.f1 > best.f1:
            best = rep
    return float(best.threshold), best

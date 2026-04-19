"""Probability calibration + stacking meta-learner.

Workflow:

1. After each LORO fold, every model writes a per-tile probability raster
   under ``artifacts/oof/{model}/{tile}.npy``.
2. ``calibrate_per_model`` fits one ``IsotonicRegression`` per model on
   the union of consensus pixels across all OOF tiles.
3. ``fit_stacker`` trains a logistic regression on the (M, N) stack of
   calibrated probabilities and emits the meta-learner.
4. ``stack_predict`` accepts a list of model probability rasters and
   returns the calibrated, stacked probability raster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibratedStack:
    model_names: list[str]
    calibrators: dict[str, IsotonicRegression] = field(default_factory=dict)
    stacker: LogisticRegression | None = None
    regional_stackers: dict[str, LogisticRegression] = field(default_factory=dict)
    regional_model_names: dict[str, list[str]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_names": self.model_names,
                "calibrators": self.calibrators,
                "stacker": self.stacker,
                "regional_stackers": self.regional_stackers,
                "regional_model_names": self.regional_model_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "CalibratedStack":
        payload = joblib.load(path)
        out = cls(model_names=payload["model_names"])
        out.calibrators = payload["calibrators"]
        out.stacker = payload["stacker"]
        out.regional_stackers = payload.get("regional_stackers", {})
        out.regional_model_names = payload.get("regional_model_names", {})
        return out

    def calibrate(self, model_name: str, prob: np.ndarray) -> np.ndarray:
        calib = self.calibrators.get(model_name)
        if calib is None:
            return prob
        flat = prob.reshape(-1)
        out = calib.predict(flat).astype(np.float32)
        return out.reshape(prob.shape)

    def _stack_with(
        self,
        probs: dict[str, np.ndarray],
        model_names: list[str],
        stacker: LogisticRegression | None,
    ) -> np.ndarray:
        cal = np.stack(
            [self.calibrate(name, probs[name]) for name in model_names], axis=0
        )
        if stacker is None:
            # Fallback: simple averaging.
            return cal.mean(axis=0).astype(np.float32)
        h, w = cal.shape[1:]
        flat = cal.reshape(len(model_names), -1).T  # (N, M)
        out = stacker.predict_proba(flat)[:, 1].astype(np.float32)
        return out.reshape(h, w)

    def stack(self, probs: dict[str, np.ndarray], region: str | None = None) -> np.ndarray:
        if region is not None and region in self.regional_model_names:
            model_names = self.regional_model_names[region]
            if all(name in probs for name in model_names):
                return self._stack_with(probs, model_names, self.regional_stackers.get(region))
        return self._stack_with(probs, self.model_names, self.stacker)


def fit_calibration(
    probs: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> IsotonicRegression:
    p = probs[mask].astype(np.float32)
    y = labels[mask].astype(np.uint8)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def fit_stacker(
    cal_probs: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> LogisticRegression:
    """``cal_probs`` shape (M, H, W); returns LR over M features."""
    M = cal_probs.shape[0]
    flat = cal_probs.reshape(M, -1).T[mask.reshape(-1)]
    y = labels.reshape(-1)[mask.reshape(-1)]
    lr_kwargs = dict(max_iter=200, class_weight="balanced", C=1.0)
    try:
        # The stacker sees millions of pixels but only a few model features.
        # Newton-Cholesky converges much faster than the default LBFGS here.
        lr = LogisticRegression(solver="newton-cholesky", **lr_kwargs)
        lr.fit(flat, y)
    except ValueError:
        lr = LogisticRegression(solver="lbfgs", **lr_kwargs)
        lr.fit(flat, y)
    return lr


def write_oof_summary(out_path: Path, summary: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

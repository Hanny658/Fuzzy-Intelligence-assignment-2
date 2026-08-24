"""ROC / EER / operating-point metrics. Scores: higher = more likely cancer (positive class)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class OperatingPoint:
    threshold: float
    fpr: float
    fnr: float
    sensitivity: float
    specificity: float
    accuracy: float


def eer(y: np.ndarray, scores: np.ndarray):
    """Equal error rate and its threshold, by linear interpolation of the ROC where FPR = FNR."""
    fpr, tpr, thr = roc_curve(y, scores)
    fnr = 1 - tpr
    diff = fpr - fnr  # goes from -1 (at (0,0)) to +1 (at (1,1))
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:  # degenerate: take the closest point
        i = int(np.argmin(np.abs(diff)))
        return float((fpr[i] + fnr[i]) / 2), float(thr[i])
    i = int(idx[0])
    d0, d1 = diff[i], diff[i + 1]
    w = d0 / (d0 - d1) if d0 != d1 else 0.0
    e = fpr[i] + w * (fpr[i + 1] - fpr[i])
    t0, t1 = thr[i], thr[i + 1]
    if np.isinf(t0):
        t0 = t1 + 1e-6
    t = t0 + w * (t1 - t0)
    return float(e), float(t)


def operating_point(y: np.ndarray, scores: np.ndarray, threshold: float) -> OperatingPoint:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return OperatingPoint(float(threshold), 1 - spec, 1 - sens, sens, spec, (tp + tn) / len(y))


def cost_threshold(y: np.ndarray, scores: np.ndarray, c_fn: float = 5.0, c_fp: float = 1.0) -> float:
    """Threshold minimising expected cost c_fn*FNR*P(pos) + c_fp*FPR*P(neg) (secondary analysis)."""
    fpr, tpr, thr = roc_curve(y, scores)
    p = y.mean()
    cost = c_fn * (1 - tpr) * p + c_fp * fpr * (1 - p)
    i = int(np.argmin(cost))
    t = thr[i]
    return float(t if np.isfinite(t) else thr[min(i + 1, len(thr) - 1)])


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y, scores))

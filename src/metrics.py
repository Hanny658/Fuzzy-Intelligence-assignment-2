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
    """Return the interpolated EER and the closest attainable deterministic threshold.

    The EER value is obtained from the linearly interpolated ROC intersection with
    FPR = FNR.  On a finite sample, that interpolated point generally represents a
    mixture of two adjacent operating points rather than one score threshold.  The
    returned threshold is therefore selected separately from the complete empirical
    ROC: it minimises |FPR - FNR|, with balanced error as the tie-breaker.
    """
    fpr, tpr, thr = roc_curve(y, scores, drop_intermediate=False)
    fnr = 1 - tpr
    diff = fpr - fnr  # goes from -1 (at (0,0)) to +1 (at (1,1))
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:  # degenerate: take the closest point
        e = float(np.min((fpr + fnr) / 2))
    else:
        i = int(idx[0])
        d0, d1 = diff[i], diff[i + 1]
        w = d0 / (d0 - d1) if d0 != d1 else 0.0
        e = float(fpr[i] + w * (fpr[i + 1] - fpr[i]))

    gap = np.abs(diff)
    candidates = np.flatnonzero(gap == gap.min())
    j = int(candidates[np.argmin((fpr[candidates] + fnr[candidates]) / 2)])
    return e, float(thr[j])


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

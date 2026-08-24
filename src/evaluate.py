"""Unified evaluation protocol.

For each (dataset, model):
  A. Given train/test split
     - inner stratified K-fold on the training set -> out-of-fold (OOF) scores
       -> EER threshold theta_EER chosen on OOF scores (no test leakage)
     - refit on full training set -> test scores
     - report: test AUC, test EER (best boundary on the test ROC), and FPR/FNR/acc on the test set
       at theta_EER chosen from training; plus a cost-sensitive point (secondary).
  B. Repeated stratified K-fold on the pooled data -> mean +- std of AUC and EER (small-sample robustness).
Preprocessing (log1p + standardisation) is fit inside every training fold.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from data import Dataset
from metrics import auc, cost_threshold, eer, operating_point
from models.base import ScoringModel, seed_everything
from preprocess import make_preprocessor


@dataclass
class Result:
    dataset: str
    model: str
    family: str
    # A. given split
    test_auc: float
    test_eer: float
    test_eer_threshold: float
    train_oof_auc: float
    train_oof_eer: float
    theta_eer: float  # chosen on training OOF scores
    test_fpr_at_theta: float
    test_fnr_at_theta: float
    test_sens_at_theta: float
    test_spec_at_theta: float
    test_acc_at_theta: float
    test_sens_at_cost: float  # secondary: cost-sensitive threshold (FN 5x FP) chosen on training OOF
    test_spec_at_cost: float
    fit_time_s: float
    # B. repeated CV
    cv_auc_mean: float = float("nan")
    cv_auc_std: float = float("nan")
    cv_eer_mean: float = float("nan")
    cv_eer_std: float = float("nan")
    note: str = ""
    # raw material for plots (not written to the summary table)
    scores: dict = field(default_factory=dict, repr=False)

    def row(self) -> dict:
        d = asdict(self)
        d.pop("scores")
        return d


def _fit_score(factory, X_tr, y_tr, X_te, heavy_cols, seed):
    pre = make_preprocessor(heavy_cols).fit(X_tr)
    seed_everything(seed)
    m: ScoringModel = factory(seed=seed)
    m.fit(pre.transform(X_tr), y_tr)
    return m, m.decision_scores(pre.transform(X_te))


def evaluate(factory, ds: Dataset, n_splits: int = 5, n_repeats: int = 5, seed: int = 0,
             c_fn: float = 5.0, verbose: bool = True) -> Result:
    name = factory(seed=seed).name
    family = factory(seed=seed).family
    t0 = time.time()

    # ---- A. given split: inner CV for the threshold -------------------------------------------
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(ds.y_train))
    for k, (tr, va) in enumerate(inner.split(ds.X_train, ds.y_train)):
        _, s = _fit_score(factory, ds.X_train[tr], ds.y_train[tr], ds.X_train[va], ds.heavy_tail_cols, seed + k)
        oof[va] = s
    oof_auc = auc(ds.y_train, oof)
    oof_eer, theta = eer(ds.y_train, oof)
    theta_cost = cost_threshold(ds.y_train, oof, c_fn=c_fn)

    t_fit = time.time()
    model, test_scores = _fit_score(factory, ds.X_train, ds.y_train, ds.X_test, ds.heavy_tail_cols, seed)
    fit_time = time.time() - t_fit

    test_auc = auc(ds.y_test, test_scores)
    test_eer, test_thr = eer(ds.y_test, test_scores)
    op = operating_point(ds.y_test, test_scores, theta)
    opc = operating_point(ds.y_test, test_scores, theta_cost)

    # ---- B. repeated CV on pooled data ----------------------------------------------------------
    cv_aucs, cv_eers = [], []
    cv_scores = []
    if n_repeats > 0:
        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        Xa, ya = ds.X_all, ds.y_all
        for k, (tr, va) in enumerate(rkf.split(Xa, ya)):
            _, s = _fit_score(factory, Xa[tr], ya[tr], Xa[va], ds.heavy_tail_cols, seed + 100 + k)
            cv_aucs.append(auc(ya[va], s))
            cv_eers.append(eer(ya[va], s)[0])
            cv_scores.append((va, s))

    res = Result(
        dataset=ds.name, model=name, family=family,
        test_auc=test_auc, test_eer=test_eer, test_eer_threshold=test_thr,
        train_oof_auc=oof_auc, train_oof_eer=oof_eer, theta_eer=theta,
        test_fpr_at_theta=op.fpr, test_fnr_at_theta=op.fnr, test_sens_at_theta=op.sensitivity,
        test_spec_at_theta=op.specificity, test_acc_at_theta=op.accuracy,
        test_sens_at_cost=opc.sensitivity, test_spec_at_cost=opc.specificity,
        fit_time_s=fit_time,
        cv_auc_mean=float(np.mean(cv_aucs)) if cv_aucs else float("nan"),
        cv_auc_std=float(np.std(cv_aucs)) if cv_aucs else float("nan"),
        cv_eer_mean=float(np.mean(cv_eers)) if cv_eers else float("nan"),
        cv_eer_std=float(np.std(cv_eers)) if cv_eers else float("nan"),
        note=model.describe(),
        scores={"y_test": ds.y_test, "test_scores": test_scores, "y_train": ds.y_train, "oof_scores": oof,
                "cv_aucs": np.array(cv_aucs), "cv_eers": np.array(cv_eers)},
    )
    if verbose:
        print(f"  {ds.name:7s} {name:14s} testAUC={test_auc:.3f} testEER={test_eer:.3f} "
              f"| theta(train-OOF): FPR={op.fpr:.3f} FNR={op.fnr:.3f} acc={op.accuracy:.3f} "
              f"| CV AUC={res.cv_auc_mean:.3f}+-{res.cv_auc_std:.3f} EER={res.cv_eer_mean:.3f}+-{res.cv_eer_std:.3f} "
              f"| fit {fit_time:.2f}s total {time.time() - t0:.1f}s {model.describe()}")
    return res

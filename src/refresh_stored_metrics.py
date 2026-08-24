"""Refresh derived metrics in saved Result objects from the stored raw score arrays.

This is useful when a metric implementation is clarified without changing any fitted
model or score. It updates results/*_results.pkl; run ``run_all.py --models none
--merge`` afterwards to regenerate the CSV, tables, and figures.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from metrics import auc, cost_threshold, eer, operating_point


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MODEL_KEYS = {
    "LogReg": "logreg",
    "SVM-RBF": "svm",
    "RandomForest": "rf",
    "ELM": "elm",
    "MLP-BP": "mlp",
    "PC-2017": "pc",
    "Forward-Forward": "ff",
    "CMA-ES-MLP": "cmaes",
    "GA-MLP": "ga",
    "ANFIS": "anfis",
    "TabPFN": "tabpfn",
}


def refresh(dataset: str) -> None:
    pickle_path = os.path.join(RESULTS, f"{dataset}_results.pkl")
    with open(pickle_path, "rb") as stream:
        payload = pickle.load(stream)

    for model_name, result in payload["results"].items():
        key = MODEL_KEYS[model_name]
        score_path = os.path.join(RESULTS, f"{dataset}_{key}_scores.npz")
        with np.load(score_path, allow_pickle=False) as scores:
            y_test = scores["y_test"]
            test_scores = scores["test_scores"]
            y_train = scores["y_train"]
            oof_scores = scores["oof_scores"]
            cv_aucs = scores["cv_aucs"]
            cv_eers = scores["cv_eers"]

            result.test_auc = auc(y_test, test_scores)
            result.test_eer, result.test_eer_threshold = eer(y_test, test_scores)
            result.train_oof_auc = auc(y_train, oof_scores)
            result.train_oof_eer, result.theta_eer = eer(y_train, oof_scores)

            operating = operating_point(y_test, test_scores, result.theta_eer)
            result.test_fpr_at_theta = operating.fpr
            result.test_fnr_at_theta = operating.fnr
            result.test_sens_at_theta = operating.sensitivity
            result.test_spec_at_theta = operating.specificity
            result.test_acc_at_theta = operating.accuracy

            cost_theta = cost_threshold(y_train, oof_scores)
            cost_operating = operating_point(y_test, test_scores, cost_theta)
            result.test_sens_at_cost = cost_operating.sensitivity
            result.test_spec_at_cost = cost_operating.specificity

            result.cv_auc_mean = float(np.mean(cv_aucs))
            result.cv_auc_std = float(np.std(cv_aucs))
            result.cv_eer_mean = float(np.mean(cv_eers))
            result.cv_eer_std = float(np.std(cv_eers))
            result.scores = {name: scores[name].copy() for name in scores.files}

    with open(pickle_path, "wb") as stream:
        pickle.dump(payload, stream)
    print(f"refreshed {dataset}: {len(payload['results'])} models")


if __name__ == "__main__":
    refresh("nuh")
    refresh("wdbc")

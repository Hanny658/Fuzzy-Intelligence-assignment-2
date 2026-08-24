"""Sensitivity of NUH results to the clinically ambiguous borderline subgroup.

Runs repeated stratified 5-fold CV under three label policies and writes a compact
CSV used by the report:

    python src/borderline_sensitivity.py --repeats 5 --seed 0
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from data import load_nuh_g2
from evaluate import evaluate
from models.classical import RandomForest, SVMRBF
from models.mlp_bp import MLPBP


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "results", "nuh_borderline_sensitivity.csv")

POLICIES = {
    "Borderline as cancer": ((1, 3, 4), (2,)),
    "Borderline excluded": ((3, 4), (2,)),
    "Borderline as non-cancer": ((3, 4), (1, 2)),
}

MODELS = [SVMRBF, RandomForest, MLPBP]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for policy, (cancer_groups, noncancer_groups) in POLICIES.items():
        ds = load_nuh_g2(cancer_groups=cancer_groups, noncancer_groups=noncancer_groups)
        print(f"== {policy}: {ds.summary()}", flush=True)
        for factory in MODELS:
            result = evaluate(factory, ds, n_repeats=args.repeats, seed=args.seed, verbose=True)
            rows.append({
                "policy": policy,
                "n": len(ds.y_all),
                "positive": int(ds.y_all.sum()),
                "negative": int(len(ds.y_all) - ds.y_all.sum()),
                "model": result.model,
                "cv_auc_mean": result.cv_auc_mean,
                "cv_auc_std": result.cv_auc_std,
                "cv_eer_mean": result.cv_eer_mean,
                "cv_eer_std": result.cv_eer_std,
            })

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

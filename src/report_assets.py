"""Build the compact tables and focused figures used by report/report.tex."""
from __future__ import annotations

import os
import pickle

import pandas as pd

import plots


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

FOCUS_MODELS = [
    "LogReg",
    "SVM-RBF",
    "RandomForest",
    "MLP-BP",
    "MLP-FF",
    "MLP-CMA-ES",
    "ANFIS",
    "TabPFN",
]


def load_results(dataset: str):
    with open(os.path.join(RESULTS, f"{dataset}_results.pkl"), "rb") as stream:
        payload = pickle.load(stream)
    return {name: payload["results"][name] for name in FOCUS_MODELS}


def compact_table(results: dict) -> str:
    best_auc = max(r.cv_auc_mean for r in results.values())
    best_eer = min(r.cv_eer_mean for r in results.values())
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Test AUC & Test EER$^{*}$ & CV AUC & CV EER \\",
        r"\midrule",
    ]
    for result in results.values():
        cv_auc = f"{result.cv_auc_mean:.3f} $\\pm$ {result.cv_auc_std:.3f}"
        cv_eer = f"{result.cv_eer_mean:.3f} $\\pm$ {result.cv_eer_std:.3f}"
        if result.cv_auc_mean == best_auc:
            cv_auc = rf"\textbf{{{cv_auc}}}"
        if result.cv_eer_mean == best_eer:
            cv_eer = rf"\textbf{{{cv_eer}}}"
        lines.append(
            f"{result.model} & {result.test_auc:.3f} & {result.test_eer:.3f} & "
            f"{cv_auc} & {cv_eer} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def sensitivity_table() -> str:
    frame = pd.read_csv(os.path.join(RESULTS, "nuh_borderline_sensitivity.csv"))
    policy_names = {
        "Borderline as cancer": "Cancer",
        "Borderline excluded": "Excluded",
        "Borderline as non-cancer": "Non-cancer",
    }
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Borderline policy & Model & $n$ (cancer/non-cancer) & CV AUC & CV EER \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{policy_names[row.policy]} & {row.model} & {int(row.n)} "
            f"({int(row.positive)}/{int(row.negative)}) & "
            f"{row.cv_auc_mean:.3f} $\\pm$ {row.cv_auc_std:.3f} & "
            f"{row.cv_eer_mean:.3f} $\\pm$ {row.cv_eer_std:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    os.makedirs(FIGURES, exist_ok=True)
    focused = {}
    for dataset in ("nuh", "wdbc", "support2"):
        focused[dataset] = load_results(dataset)
        with open(os.path.join(RESULTS, f"{dataset}_focus_table.tex"), "w", encoding="utf-8") as stream:
            stream.write(compact_table(focused[dataset]))
    plots.roc_panels(focused["nuh"], "NUH-g2", os.path.join(FIGURES, "nuh_roc_focused.png"))
    plots.roc_panels(focused["support2"], "SUPPORT2", os.path.join(FIGURES, "support2_roc_focused.png"))
    # both data sets share one dot plot, one hue each, so each model is read across the two
    plots.cv_dotplot_paired({"NUH-g2": focused["nuh"], "SUPPORT2": focused["support2"]},
                            os.path.join(FIGURES, "cv_auc_nuh_support2.png"), "auc")
    with open(os.path.join(RESULTS, "nuh_borderline_table.tex"), "w", encoding="utf-8") as stream:
        stream.write(sensitivity_table())
    print("wrote focused report tables and figures")


if __name__ == "__main__":
    main()

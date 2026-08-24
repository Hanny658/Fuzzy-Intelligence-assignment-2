"""Run every model on every dataset; write results/, figures/ and LaTeX tables.

    python src/run_all.py --datasets nuh wdbc --models all --repeats 5

To (re)run only some models and fold them into the existing results instead of overwriting them:

    python src/run_all.py --models tabpfn --merge --exclude kan

--merge loads results/<dataset>_results.pkl and replaces/adds the models named in --models;
--exclude drops models (by key) from every derived output (summary CSV, LaTeX table, figures).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from data import LOADERS  # noqa: E402
from evaluate import evaluate  # noqa: E402
from models.anfis import ANFIS  # noqa: E402
from models.classical import ELM, LogReg, RandomForest, SVMRBF  # noqa: E402
from models.ff import ForwardForward  # noqa: E402
from models.kan import KAN  # noqa: E402
from models.mlp_bp import MLPBP  # noqa: E402
from models.neuroevo import CMAESMLP, GAMLP  # noqa: E402
from models.pc import PredictiveCoding  # noqa: E402
from models.tabpfn_wrap import TabPFN  # noqa: E402
import plots  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

MODELS = {
    "logreg": LogReg, "svm": SVMRBF, "rf": RandomForest, "elm": ELM,
    "mlp": MLPBP, "pc": PredictiveCoding, "ff": ForwardForward, "kan": KAN,
    "cmaes": CMAESMLP, "ga": GAMLP, "anfis": ANFIS, "tabpfn": TabPFN,
}

COLS = [("model", "Model"), ("test_auc", "AUC"), ("test_eer", "EER"),
        ("theta_pair", "$\\theta_{\\mathrm{EER}}$ test\\,/\\,OOF"),
        ("test_fpr_at_theta", "FPR@$\\theta$"), ("test_fnr_at_theta", "FNR@$\\theta$"),
        ("test_acc_at_theta", "Acc@$\\theta$"), ("cv_auc", "CV AUC"), ("cv_eer", "CV EER"), ("fit_time_s", "Fit (s)")]


def latex_table(df: pd.DataFrame, dataset: str) -> str:
    d = df.copy()
    d["cv_auc"] = d.apply(lambda r: f"{r.cv_auc_mean:.3f} $\\pm$ {r.cv_auc_std:.3f}", axis=1)
    d["cv_eer"] = d.apply(lambda r: f"{r.cv_eer_mean:.3f} $\\pm$ {r.cv_eer_std:.3f}", axis=1)
    # the EER threshold of the test ROC and the one pre-chosen on training OOF scores (model's own score scale)
    d["theta_pair"] = d.apply(lambda r: f"${r.test_eer_threshold:.2f}$ / ${r.theta_eer:.2f}$", axis=1)
    best_auc, best_eer = d.test_auc.max(), d.test_eer.min()
    lines = ["\\begin{tabular}{lrrcrrrccr}", "\\toprule",
             " & ".join(h for _, h in COLS) + " \\\\", "\\midrule"]
    for _, r in d.iterrows():
        cells = []
        for k, _ in COLS:
            v = r[k]
            if isinstance(v, float):
                s = f"{v:.3f}" if k != "fit_time_s" else f"{v:.2f}"
                if (k == "test_auc" and v == best_auc) or (k == "test_eer" and v == best_eer):
                    s = f"\\textbf{{{s}}}"
            else:
                s = str(v).replace("_", "\\_")
            cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nuh", "wdbc"])
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true",
                    help="update the existing results/<dataset>_results.pkl instead of starting from scratch")
    ap.add_argument("--exclude", nargs="*", default=[], help="model keys to drop from all derived outputs")
    args = ap.parse_args()
    # --models none (with --merge) re-fits nothing and only regenerates tables/figures from the saved results
    names = list(MODELS) if args.models == ["all"] else [m for m in args.models if m != "none"]
    excluded = {MODELS[k].name for k in args.exclude}
    names = [n for n in names if MODELS[n].name not in excluded]
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    for dname in args.datasets:
        ds = LOADERS[dname]()
        print(f"== {ds.summary()}")
        results, hists, skipped = {}, {}, {}
        pkl_path = os.path.join(RES, f"{dname}_results.pkl")
        if args.merge and os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                prev = pickle.load(f)
            results, hists, skipped = prev["results"], prev["hists"], prev["skipped"]
            for k in [k for k in results if k in excluded]:
                results.pop(k)
                hists.pop(k, None)
            for n in names:  # models about to be re-run drop their old skip note
                skipped.pop(MODELS[n].name, None)
            print(f"   merging into existing results: {list(results)}")
        for mname in names:
            factory = MODELS[mname]
            t0 = time.time()
            try:
                r = evaluate(factory, ds, n_repeats=args.repeats, seed=args.seed)
            except Exception as e:  # e.g. TabPFN licence/token missing
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
                print(f"  {ds.name:7s} {factory.name:14s} SKIPPED - {msg}")
                skipped[factory.name] = msg
                continue
            results[r.model] = r
            np.savez(os.path.join(RES, f"{dname}_{mname}_scores.npz"), **r.scores)
            # keep the training curves of the three paradigms on the full training set
            if mname in ("mlp", "pc", "ff"):
                from preprocess import make_preprocessor
                from models.base import seed_everything
                pre = make_preprocessor(ds.heavy_tail_cols).fit(ds.X_train)
                seed_everything(args.seed)
                m = factory(seed=args.seed).fit(pre.transform(ds.X_train), ds.y_train)
                hists[r.model] = m.hist
        df = pd.DataFrame([r.row() for r in results.values()])
        df.to_csv(os.path.join(RES, f"{dname}_summary.csv"), index=False)
        with open(os.path.join(RES, f"{dname}_table.tex"), "w") as f:
            f.write(latex_table(df, ds.name))
        with open(os.path.join(RES, f"{dname}_results.pkl"), "wb") as f:
            pickle.dump({"results": results, "skipped": skipped, "hists": hists}, f)
        plots.roc_panels(results, ds.name, os.path.join(FIG, f"{dname}_roc_panels.png"))
        plots.cv_dotplot(results, ds.name, os.path.join(FIG, f"{dname}_cv_auc.png"), "auc")
        plots.cv_dotplot(results, ds.name, os.path.join(FIG, f"{dname}_cv_eer.png"), "eer")
        if hists:
            plots.training_curves(hists, ds.name, os.path.join(FIG, f"{dname}_training_curves.png"))
        for r in results.values():
            plots.roc_single(r, ds.name, os.path.join(FIG, f"{dname}_roc_{r.model.lower().replace('-', '_')}.png"))
        skipped_path = os.path.join(RES, f"{dname}_skipped.txt")
        if skipped:
            with open(skipped_path, "w") as f:
                f.write("\n".join(f"{k}: {v}" for k, v in skipped.items()))
        elif os.path.exists(skipped_path):
            os.remove(skipped_path)
        print(df[["model", "test_auc", "test_eer", "test_acc_at_theta", "cv_auc_mean", "cv_eer_mean", "fit_time_s"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()

"""Run every model on every dataset; write results/, figures/ and LaTeX tables.

    python src/run_all.py --datasets nuh wdbc --models all --repeats 5

To (re)run only some models and fold them into the existing results instead of overwriting them:

    python src/run_all.py --models tabpfn --merge --exclude kan

--merge loads results/<dataset>_results.pkl and replaces/adds the models named in --models;
--exclude drops models (by key) from every derived output (summary CSV, LaTeX table, figures).
--jobs N fits N models in parallel worker processes (results are identical: every model has its
own seeds and never sees another model's state).  On SUPPORT2, which is ~80x larger than WDBC:

    python src/run_all.py --datasets support2 --repeats 1 --jobs 4

Every model is checkpointed to results/parts/ as soon as it finishes, so an interrupted run only
loses the models still in flight; --resume then skips the ones already on disk.  If --jobs cannot
be used (see below), --fit-only lets several independent processes fit disjoint model sets at the
same time without racing on the shared tables and figures, and a final assembly pass folds every
checkpoint in:

    python src/run_all.py --datasets support2 --models ga    --repeats 1 --resume --fit-only &
    python src/run_all.py --datasets support2 --models tabpfn --repeats 1 --resume --fit-only &
    wait
    python src/run_all.py --datasets support2 --models none  --repeats 1          # assemble only

Note: on this machine, after a hibernate/resume cycle, creating subprocesses deadlocks -- both
--jobs > 1 and joblib's loky backend hang with the children spawned but idle at 0% CPU.  The
independent-process form above plus JOBLIB_MULTIPROCESSING=0 (which makes the SVM grid search run
sequentially, same numbers) is the way around it without rebooting.
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
from models.classical import ELM, LightGBM, LogReg, RandomForest, SVMRBF  # noqa: E402
from models.ff import ForwardForward  # noqa: E402
from models.kan import KAN  # noqa: E402
from models.mlp_bp import MLPBP  # noqa: E402
from models.neuroevo import CMAESMLP, GAMLP  # noqa: E402
from models.pc import PredictiveCoding  # noqa: E402
from models.tabpfn_wrap import TabPFN  # noqa: E402
import plots  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
PARTS = os.path.join(RES, "parts")  # per-model checkpoints, so a killed run can be resumed

MODELS = {
    "logreg": LogReg, "svm": SVMRBF, "rf": RandomForest, "lgbm": LightGBM, "elm": ELM,
    "mlp": MLPBP, "pc": PredictiveCoding, "ff": ForwardForward, "kan": KAN,
    "cmaes": CMAESMLP, "ga": GAMLP, "anfis": ANFIS, "tabpfn": TabPFN,
}

# rough relative cost, measured on SUPPORT2; used only to schedule the parallel run
SLOW_ORDER = {"ga": 100, "ff": 88, "tabpfn": 58, "cmaes": 44, "pc": 37, "svm": 20, "anfis": 2, "mlp": 2}

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


def part_path(dname: str, mname: str) -> str:
    return os.path.join(PARTS, f"{dname}_{mname}.pkl")


def has_result(dname: str, mname: str) -> bool:
    """True only if the checkpoint holds a fitted model.

    A model that raised is checkpointed too, so that a licence or environment failure is recorded
    rather than silently retried forever -- but --resume must still re-run it, because the usual
    reason to resume is that the cause has been fixed.
    """
    path = part_path(dname, mname)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            return "result" in pickle.load(f)
    except Exception:
        return False


def run_model(mname: str, dname: str, repeats: int, seed: int) -> str:
    """Fit and evaluate one model on one dataset, and checkpoint it to results/parts/.

    Module level so it can run in a worker process. Each model is written the moment it finishes,
    so an interrupted run loses at most the models still in flight; --resume picks up the rest.
    """
    factory = MODELS[mname]
    ds = LOADERS[dname]()
    try:
        r = evaluate(factory, ds, n_repeats=repeats, seed=seed)
        payload = {"key": mname, "result": r, "hist": None}
        if mname in ("mlp", "pc", "ff"):  # training curves of the three paradigms, on the full training set
            from preprocess import make_preprocessor
            from models.base import seed_everything
            pre = make_preprocessor(ds.heavy_tail_cols).fit(ds.X_train)
            seed_everything(seed)
            payload["hist"] = factory(seed=seed).fit(pre.transform(ds.X_train), ds.y_train).hist
    except Exception as e:  # e.g. TabPFN licence/token missing
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
        print(f"  {ds.name:7s} {factory.name:14s} SKIPPED - {msg}", flush=True)
        payload = {"key": mname, "skipped": (factory.name, msg)}
    os.makedirs(PARTS, exist_ok=True)
    tmp = part_path(dname, mname) + ".tmp"  # write-then-rename, so a kill cannot leave a half file
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, part_path(dname, mname))
    return mname


def _worker_init(threads: int) -> None:
    """Cap the BLAS/torch thread pools so N workers do not oversubscribe the machine."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nuh", "wdbc"])
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true",
                    help="update the existing results/<dataset>_results.pkl instead of starting from scratch")
    ap.add_argument("--exclude", nargs="*", default=[], help="model keys to drop from all derived outputs")
    ap.add_argument("--jobs", type=int, default=1, help="fit this many models in parallel processes")
    ap.add_argument("--resume", action="store_true",
                    help="skip models that already have a checkpoint in results/parts/")
    ap.add_argument("--fit-only", action="store_true",
                    help="fit and checkpoint only; skip assembling tables and figures. Lets several "
                         "independent processes fit disjoint model sets at once without racing on the "
                         "derived outputs; run once more with --models none afterwards to assemble.")
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
        t0 = time.time()
        todo = [m for m in names if not (args.resume and has_result(dname, m))]
        if args.resume and len(todo) < len(names):
            print(f"   resuming: {len(names) - len(todo)} models already checkpointed, {len(todo)} to run")
        if args.jobs > 1 and len(todo) > 1:
            from concurrent.futures import ProcessPoolExecutor
            threads = max(1, (os.cpu_count() or 4) // args.jobs)
            with ProcessPoolExecutor(max_workers=args.jobs, initializer=_worker_init,
                                     initargs=(threads,)) as pool:
                # slowest models first, so the long tail is not left running on its own at the end
                order = sorted(todo, key=lambda m: -SLOW_ORDER.get(m, 0))
                list(pool.map(run_model, order, *[[v] * len(order) for v in
                                                  (dname, args.repeats, args.seed)]))
        else:
            for m in todo:
                run_model(m, dname, args.repeats, args.seed)

        if args.fit_only:
            print(f"   fitted {len(todo)} models in {(time.time() - t0) / 60:.1f} min (checkpoints only)")
            continue

        # Assemble from every checkpoint this dataset has, in MODELS order, so tables and figures
        # do not depend on which worker finished first and do not depend on what this particular
        # invocation was asked to fit. "--models none" is therefore a pure re-assembly step.
        for mname in MODELS:
            path = part_path(dname, mname)
            if mname in args.exclude or not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                p = pickle.load(f)
            if "skipped" in p:
                skipped[p["skipped"][0]] = p["skipped"][1]
                continue
            r = p["result"]
            results[r.model] = r
            np.savez(os.path.join(RES, f"{dname}_{mname}_scores.npz"), **r.scores)
            if p["hist"] is not None:
                hists[r.model] = p["hist"]
        print(f"   {len(todo)} models fitted in {(time.time() - t0) / 60:.1f} min "
              f"({'parallel x' + str(args.jobs) if args.jobs > 1 else 'serial'}); "
              f"{len(results)} in the assembled results")
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

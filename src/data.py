"""Dataset loaders for Assignment 2.

Three datasets, chosen to span three difficulty tiers rather than three cancers:
  * WDBC     - Wisconsin Diagnostic Breast Cancer (569 x 30), label M=1 (malignant), B=0.
  * NUH      - NUH ovarian-cancer blood-test data (Tan, Quek, Ng, Razvi 2008), group g2
               (28 features, 55 train / 54 test after de-duplication).
  * SUPPORT2 - SUPPORT study, in-hospital mortality of seriously ill adults (9105 patients).
               Added because WDBC is saturated (every model scores AUC ~0.99) and NUH is too
               small to separate models reliably; SUPPORT2 sits in between, with class
               imbalance, mixed feature types and real missingness.

NUH file facts (verified by inspection, not documented in the repo):
  * *Train.txt begins with a header line "N 1"; *Test.txt has no header.
  * Within a group, the c1..cN files hold the SAME samples with one-vs-rest labels:
      g2c1 = borderline, g2c2 = benign/normal, g2c3 = stage I&II, g2c4 = stage III&IV.
  * g2c1Test contains 3 extra rows that duplicate g2c1Train rows -> removed.
Cancer label follows the source paper: cancer = borderline + I&II + III&IV, non-cancer = benign.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUH_DIR = os.path.join(ROOT, "Ovarian-NUH")
WDBC_PATH = os.path.join(ROOT, "wisconsin_breast_cancer_diagnostic", "wdbc.data")
SUPPORT2_PATH = os.path.join(ROOT, "support2", "support2.csv")
SUPPORT2_URL = "https://archive.ics.uci.edu/static/public/880/data.csv"


@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    heavy_tail_cols: list  # columns to log1p-transform before scaling

    @property
    def X_all(self) -> np.ndarray:
        return np.vstack([self.X_train, self.X_test])

    @property
    def y_all(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_test])

    def summary(self) -> str:
        return (f"{self.name}: d={self.X_train.shape[1]} | train n={len(self.y_train)} "
                f"(pos={int(self.y_train.sum())}) | test n={len(self.y_test)} "
                f"(pos={int(self.y_test.sum())}) | heavy-tail cols={self.heavy_tail_cols}")


# ----------------------------------------------------------------------------- NUH
def _read_nuh(path: str) -> np.ndarray:
    """Read a NUH text file, skipping the optional 'N 1' header line."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) > 2:
                rows.append([float(p) for p in parts])
    return np.array(rows, dtype=float)


def _heavy_tail_cols(X: np.ndarray, ratio: float = 20.0) -> list:
    """Columns whose max/median ratio is large (right-skewed lab values).

    NaN-aware, because SUPPORT2 keeps its missing values until the in-fold imputer runs.
    On the two complete data sets this is identical to the plain median/max version.
    """
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(X, axis=0)
        mx = np.nanmax(X, axis=0)
    return [int(j) for j in range(X.shape[1])
            if np.isfinite(med[j]) and np.isfinite(mx[j]) and med[j] > 0 and mx[j] / med[j] > ratio]


def load_nuh_g2(cancer_groups=(1, 3, 4), noncancer_groups=(2,)) -> Dataset:
    """Build a binary NUH-g2 dataset from selected one-vs-rest subgroups.

    Groups not listed in either argument are excluded.  This makes the clinically
    ambiguous borderline group (c1) explicit: it can be treated as cancer (the
    historical default), excluded, or treated as non-cancer in sensitivity analyses.
    """
    train_parts = {c: _read_nuh(os.path.join(NUH_DIR, f"g2c{c}Train.txt")) for c in range(1, 5)}
    test_parts = {c: _read_nuh(os.path.join(NUH_DIR, f"g2c{c}Test.txt")) for c in range(1, 5)}

    def build(parts):
        # map feature-row -> subgroup id (each sample is positive in exactly one file)
        label_of = {}
        for c, arr in parts.items():
            for row in arr:
                key = tuple(row[:-1])
                if int(row[-1]) == 1:
                    assert key not in label_of or label_of[key] == c, "sample positive in two groups"
                    label_of[key] = c
        keys = list(label_of)
        X = np.array(keys, dtype=float)
        sub = np.array([label_of[k] for k in keys])
        selected = np.isin(sub, (*cancer_groups, *noncancer_groups))
        X, sub = X[selected], sub[selected]
        y = np.where(np.isin(sub, cancer_groups), 1, 0)
        assert set(np.unique(sub)) <= set(cancer_groups) | set(noncancer_groups)
        return X, y

    X_tr, y_tr = build(train_parts)
    X_te, y_te = build(test_parts)
    # remove test rows that duplicate training rows (leakage)
    train_keys = {tuple(r) for r in X_tr}
    keep = np.array([tuple(r) not in train_keys for r in X_te])
    X_te, y_te = X_te[keep], y_te[keep]

    d = X_tr.shape[1]
    names = [f"f{j + 1}" for j in range(d)]
    return Dataset("NUH-g2", X_tr, y_tr, X_te, y_te, names, _heavy_tail_cols(X_tr))


# ----------------------------------------------------------------------------- WDBC
WDBC_BASE = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
             "concavity", "concave_points", "symmetry", "fractal_dimension"]


def load_wdbc(test_size: float = 0.2, seed: int = 0) -> Dataset:
    from sklearn.model_selection import train_test_split

    raw = np.genfromtxt(WDBC_PATH, delimiter=",", dtype=str)
    y = (raw[:, 1] == "M").astype(int)
    X = raw[:, 2:].astype(float)
    names = [f"{b}_{s}" for s in ("mean", "se", "worst") for b in WDBC_BASE]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return Dataset("WDBC", X_tr, y_tr, X_te, y_te, names, _heavy_tail_cols(X_tr))


# ----------------------------------------------------------------------------- SUPPORT2
# Columns that must not be used as predictors. Two kinds:
#   identifier  - id
#   post-hoc    - recorded at or after the outcome, so knowing them presupposes the answer:
#                 death/d.time (died at any time / time to death), slos (length of stay),
#                 sfdm2 (functional status at 2 months), charges/totcst/totmcst (billed at
#                 discharge), dnr/dnrday (resuscitation order set during the stay).
# dzgroup is dropped as redundant with dzclass (dzclass is its 4-level grouping); adlp/adlsc are
# dropped as alternative encodings of adls, which is kept. prg2m/prg6m are the attending
# physician's own survival estimates, which would make the task "copy the clinician".
SUPPORT2_DROP = ["id", "death", "d.time", "slos", "sfdm2", "charges", "totcst", "totmcst",
                 "dnr", "dnrday", "prg2m", "prg6m", "dzgroup", "adlp", "adlsc"]
SUPPORT2_TARGET = "hospdead"


def load_support2(test_size: float = 0.2, seed: int = 0) -> Dataset:
    """SUPPORT2: in-hospital death of seriously ill hospitalised adults (9105 patients).

    Kept predictors are baseline demographics, comorbidity and physiology recorded on study
    entry, including the SUPPORT/APACHE severity scores (sps, aps) and the SUPPORT model's own
    2- and 6-month survival estimates (surv2m, surv6m).  Those are model outputs but they are
    computed from day-3 data, i.e. available at prediction time; no single one dominates
    (surv2m alone reaches AUC 0.84 against 0.90 for the full set).

    Categorical columns are one-hot encoded here, with missing treated as its own level: that is
    a fixed schema, not a fitted statistic, so it cannot leak.  Missing numeric values are left
    as NaN and imputed by the preprocessing pipeline, which is fit inside each training fold.
    """
    import pandas as pd

    if not os.path.isfile(SUPPORT2_PATH):  # one-off download, no login required
        import urllib.request
        os.makedirs(os.path.dirname(SUPPORT2_PATH), exist_ok=True)
        urllib.request.urlretrieve(SUPPORT2_URL, SUPPORT2_PATH)

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(SUPPORT2_PATH, low_memory=False)
    y = df[SUPPORT2_TARGET].to_numpy(dtype=int)
    df = df.drop(columns=[c for c in (*SUPPORT2_DROP, SUPPORT2_TARGET) if c in df])

    num = df.select_dtypes(include=[np.number])
    cat = df.drop(columns=num.columns)
    dummies = pd.get_dummies(cat.astype("string").fillna("missing"), prefix_sep="=", dtype=float)
    frame = pd.concat([num, dummies], axis=1)

    X = frame.to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    # only the genuinely continuous columns can be heavy-tailed; the one-hot block is 0/1
    heavy = [j for j in _heavy_tail_cols(X_tr) if j < num.shape[1]]
    return Dataset("SUPPORT2", X_tr, y_tr, X_te, y_te, list(frame.columns), heavy)


LOADERS = {"nuh": load_nuh_g2, "wdbc": load_wdbc, "support2": load_support2}


if __name__ == "__main__":
    for k, fn in LOADERS.items():
        print(fn().summary())

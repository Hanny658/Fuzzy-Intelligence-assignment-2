"""Dataset loaders for Assignment 2.

Two datasets:
  * WDBC  - Wisconsin Diagnostic Breast Cancer (569 x 30), label M=1 (malignant), B=0.
  * NUH   - NUH ovarian-cancer blood-test data (Tan, Quek, Ng, Razvi 2008), group g2
            (28 features, 55 train / 54 test after de-duplication).

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
    """Columns whose max/median ratio is large (right-skewed lab values)."""
    med = np.median(X, axis=0)
    mx = X.max(axis=0)
    return [int(j) for j in range(X.shape[1]) if med[j] > 0 and mx[j] / med[j] > ratio]


def load_nuh_g2(cancer_groups=(1, 3, 4), noncancer_groups=(2,)) -> Dataset:
    """Build cancer-vs-non-cancer labels from the one-vs-rest files of group g2."""
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


LOADERS = {"nuh": load_nuh_g2, "wdbc": load_wdbc}


if __name__ == "__main__":
    for k, fn in LOADERS.items():
        print(fn().summary())

"""Preprocessing: log1p on heavy-tailed columns, then standardisation. Fit on train only."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Log1pColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols=()):
        self.cols = list(cols)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if self.cols:
            X[:, self.cols] = np.log1p(np.clip(X[:, self.cols], 0, None))
        return X


def make_preprocessor(heavy_cols=()) -> Pipeline:
    return Pipeline([("log", Log1pColumns(heavy_cols)), ("scale", StandardScaler())])

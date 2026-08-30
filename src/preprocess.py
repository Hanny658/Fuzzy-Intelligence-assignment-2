"""Preprocessing: median imputation, log1p on heavy-tailed columns, then standardisation.

Fit on train only. The imputer matters for SUPPORT2, whose numeric columns are up to 53%
missing; on the two complete data sets it passes the values through unchanged.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
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
    # keep_empty_features: a fold can contain a column that is missing throughout; dropping it
    # would change the feature count between folds, so it is filled with zeros instead.
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("log", Log1pColumns(heavy_cols)),
                     ("scale", StandardScaler())])

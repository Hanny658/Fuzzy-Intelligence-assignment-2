"""Classical baselines: logistic regression, RBF-SVM, random forest, extreme learning machine."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from .base import ScoringModel


class LogReg(ScoringModel):
    name, family = "LogReg", "classical"

    def fit(self, X, y):
        self.m = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        return self

    def decision_scores(self, X):
        return self.m.decision_function(X)


class SVMRBF(ScoringModel):
    """RBF SVM; C and gamma chosen by an inner 5-fold grid search on AUC."""
    name, family = "SVM-RBF", "classical"

    def fit(self, X, y):
        grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
        cv = StratifiedKFold(n_splits=min(5, int(np.bincount(y).min())), shuffle=True, random_state=self.seed)
        gs = GridSearchCV(SVC(kernel="rbf"), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        gs.fit(X, y)
        self.m = gs.best_estimator_
        self.best_params_ = gs.best_params_
        return self

    def decision_scores(self, X):
        return self.m.decision_function(X)

    def describe(self):
        return f"best={self.best_params_}"


class RandomForest(ScoringModel):
    name, family = "RandomForest", "classical"

    def fit(self, X, y):
        self.m = RandomForestClassifier(n_estimators=500, random_state=self.seed, n_jobs=-1).fit(X, y)
        return self

    def decision_scores(self, X):
        return self.m.predict_proba(X)[:, 1]


class ELM(ScoringModel):
    """Extreme learning machine: random tanh hidden layer + ridge-regression read-out (no iterative training)."""
    name, family = "ELM", "classical"

    def __init__(self, seed=0, n_hidden=200, ridge=1.0):
        super().__init__(seed)
        self.n_hidden, self.ridge = n_hidden, ridge

    def _h(self, X):
        return np.tanh(X @ self.W + self.b)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self.W = rng.normal(0, 1 / np.sqrt(d), (d, self.n_hidden))
        self.b = rng.uniform(-1, 1, self.n_hidden)
        H = self._h(X)
        t = 2.0 * y - 1.0  # targets in {-1, +1}
        self.beta = np.linalg.solve(H.T @ H + self.ridge * np.eye(self.n_hidden), H.T @ t)
        return self

    def decision_scores(self, X):
        return self._h(X) @ self.beta

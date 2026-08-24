"""First-order Takagi-Sugeno ANFIS (Jang, 1993) with R scatter-partition rules.

Rule r:  IF x_1 is A_r1 AND ... AND x_d is A_rd  THEN  y_r = a_r^T x + b_r
  * Gaussian membership functions A_rj(x) = exp(-(x - c_rj)^2 / (2 s_rj^2)); product t-norm
    computed in log-space (28 antecedents would otherwise underflow)
  * normalised firing strength = softmax_r(log w_r);  output logit = sum_r w_r y_r
Membership width is initialised to sqrt(d) so the product of d Gaussians stays smooth.
Premise centres are initialised by k-means on the training data (scatter partition, avoids the
2^d grid explosion); all premise + consequent parameters are then tuned by gradient descent.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from .base import ScoringModel
from .torch_utils import predict_logits, to_t, train_autograd


class TSANFIS(torch.nn.Module):
    def __init__(self, centres: np.ndarray, sigma0: float | None = None):
        super().__init__()
        R, d = centres.shape
        sigma0 = float(np.sqrt(d)) if sigma0 is None else sigma0  # width scales with #antecedents
        self.c = torch.nn.Parameter(to_t(centres))
        self.log_s = torch.nn.Parameter(torch.full((R, d), float(np.log(sigma0))))
        self.a = torch.nn.Parameter(torch.zeros(R, d))
        self.b = torch.nn.Parameter(torch.zeros(R))

    def firing(self, x):  # (N, R) normalised firing strengths
        diff = x.unsqueeze(1) - self.c  # (N, R, d)
        log_w = -(diff ** 2 / (2 * torch.exp(2 * self.log_s))).sum(-1)
        return torch.softmax(log_w, dim=1)

    def forward(self, x):
        w = self.firing(x)
        y_r = x @ self.a.T + self.b  # (N, R)
        return (w * y_r).sum(1, keepdim=True)


class ANFIS(ScoringModel):
    name, family = "ANFIS", "fuzzy"

    def __init__(self, seed=0, n_rules=4, epochs=300, lr=5e-3, weight_decay=1e-3):
        super().__init__(seed)
        self.n_rules, self.epochs, self.lr, self.weight_decay = n_rules, epochs, lr, weight_decay

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        km = KMeans(n_clusters=self.n_rules, n_init=10, random_state=self.seed).fit(X)
        self.net = TSANFIS(km.cluster_centers_)
        self.hist = train_autograd(self.net, X, y, self.epochs, self.lr, self.weight_decay, seed=self.seed)
        return self

    def decision_scores(self, X):
        return predict_logits(self.net, X)

    def describe(self):
        return f"rules={self.n_rules} gaussian-MF final_loss={self.hist[-1]:.3f}"

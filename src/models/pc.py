"""Supervised Predictive Coding network (Whittington & Bogacz, Neural Computation 2017).

Same architecture as the BP-MLP (d-64-32-1, tanh hidden units) but NO back-propagation and NO autograd:
  * value nodes x^l, prediction mu^l = W^l f(x^{l-1}) + b^l, error nodes eps^l = x^l - mu^l
  * energy  F = 1/2 sum_l ||eps^l||^2
  * inference (relaxation, input and target clamped):
        x^l <- x^l + gamma * ( -eps^l + f'(x^l) * (W^{l+1})^T eps^{l+1} ),   l = 1..L-1
  * learning (local Hebbian-like rule after T relaxation steps):
        dW^l = eta * eps^l  f(x^{l-1})^T ,   db^l = eta * eps^l
  Every update uses only quantities available at the two ends of a synapse.
Prediction = feed-forward pass with the output node unclamped (x^L = mu^L).
"""
from __future__ import annotations

import numpy as np
import torch

from .base import ScoringModel
from .torch_utils import HIDDEN, to_t


class PCNet:
    def __init__(self, sizes, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.sizes = list(sizes)
        self.L = len(sizes) - 1
        self.W, self.b = [None], [None]  # index 1..L
        for l in range(1, self.L + 1):
            n_out, n_in = sizes[l], sizes[l - 1]
            self.W.append(torch.randn(n_out, n_in, generator=g) / np.sqrt(n_in))
            self.b.append(torch.zeros(n_out))
        # Adam state for the local updates (optimiser only re-scales the local gradient)
        self.mW = [None] + [torch.zeros_like(w) for w in self.W[1:]]
        self.vW = [None] + [torch.zeros_like(w) for w in self.W[1:]]
        self.mb = [None] + [torch.zeros_like(b) for b in self.b[1:]]
        self.vb = [None] + [torch.zeros_like(b) for b in self.b[1:]]
        self.t = 0

    @staticmethod
    def f(x):
        return torch.tanh(x)

    @staticmethod
    def df(x):
        return 1 - torch.tanh(x) ** 2

    def act(self, l, x):  # activation applied to layer l's value nodes when predicting layer l+1
        return x if l == 0 else self.f(x)

    def dact(self, l, x):
        return torch.ones_like(x) if l == 0 else self.df(x)

    def mu(self, l, x_prev):
        return self.act(l - 1, x_prev) @ self.W[l].T + self.b[l]

    def forward(self, X):
        x = [X]
        for l in range(1, self.L + 1):
            x.append(self.mu(l, x[-1]))
        return x

    def relax(self, X, Y, T, gamma):
        x = self.forward(X)
        x[self.L] = Y  # clamp the output node to the target
        for _ in range(T):
            eps = [None] + [x[l] - self.mu(l, x[l - 1]) for l in range(1, self.L + 1)]
            for l in range(1, self.L):  # hidden value nodes only
                top_down = eps[l + 1] @ self.W[l + 1]
                x[l] = x[l] + gamma * (-eps[l] + self.dact(l, x[l]) * top_down)
        eps = [None] + [x[l] - self.mu(l, x[l - 1]) for l in range(1, self.L + 1)]
        return x, eps

    def local_update(self, x, eps, eta, weight_decay=0.0, betas=(0.9, 0.999), eps_adam=1e-8):
        self.t += 1
        B = x[0].shape[0]
        for l in range(1, self.L + 1):
            gW = eps[l].T @ self.act(l - 1, x[l - 1]) / B - weight_decay * self.W[l]  # local: eps^l x f(x^{l-1})
            gb = eps[l].mean(0)
            for p, g, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                m[l] = betas[0] * m[l] + (1 - betas[0]) * g
                v[l] = betas[1] * v[l] + (1 - betas[1]) * g * g
                mh = m[l] / (1 - betas[0] ** self.t)
                vh = v[l] / (1 - betas[1] ** self.t)
                p[l] = p[l] + eta * mh / (vh.sqrt() + eps_adam)


class PredictiveCoding(ScoringModel):
    name, family = "PC-2017", "local-learning"

    def __init__(self, seed=0, epochs=300, eta=1e-3, gamma=0.1, T=20, batch_size=128, weight_decay=1e-3):
        super().__init__(seed)
        self.epochs, self.eta, self.gamma, self.T, self.batch_size, self.weight_decay = \
            epochs, eta, gamma, T, batch_size, weight_decay

    def fit(self, X, y):
        g = torch.Generator().manual_seed(self.seed)
        Xt, Yt = to_t(X), to_t(y).unsqueeze(1)
        self.net = PCNet([X.shape[1], *HIDDEN, 1], seed=self.seed)
        n = len(Xt)
        self.hist = []
        with torch.no_grad():
            for _ in range(self.epochs):
                perm = torch.randperm(n, generator=g)
                tot = 0.0
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    x, eps = self.net.relax(Xt[idx], Yt[idx], self.T, self.gamma)
                    self.net.local_update(x, eps, self.eta, self.weight_decay)
                    tot += (eps[self.net.L] ** 2).sum().item()
                self.hist.append(tot / n)
        return self

    def decision_scores(self, X):
        with torch.no_grad():
            return self.net.forward(to_t(X))[-1].squeeze(1).numpy()

    def describe(self):
        return f"arch={HIDDEN} T={self.T} gamma={self.gamma} final_out_err={self.hist[-1]:.3f}"

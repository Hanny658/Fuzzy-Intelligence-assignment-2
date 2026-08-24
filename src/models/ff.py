"""Forward-Forward algorithm (Hinton, 2022) for tabular binary classification.

Same hidden sizes as the BP-MLP (64-32, ReLU). The label is embedded in the input vector
(one-hot appended and scaled); every layer is trained with its OWN local objective:
    goodness g = mean(h^2);   loss = softplus(-(g_pos - theta)) + softplus(g_neg - theta)
Positive data = (x, true label), negative data = (x, wrong label). The input to each layer is
length-normalised and detached, so no error signal ever flows backwards between layers.
Inference: score(x) = sum_layers g(x, label=1) - g(x, label=0).
"""
from __future__ import annotations

import numpy as np
import torch

from .base import ScoringModel
from .torch_utils import HIDDEN, to_t


class FFLayer(torch.nn.Module):
    def __init__(self, d_in, d_out, lr, threshold):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.threshold = threshold

    def forward(self, x):
        x = x / (x.norm(dim=1, keepdim=True) + 1e-6)  # layer-norm-like length normalisation
        return torch.relu(self.lin(x))

    def goodness(self, h):
        return (h ** 2).mean(1)

    def train_step(self, x_pos, x_neg):
        g_pos = self.goodness(self.forward(x_pos))
        g_neg = self.goodness(self.forward(x_neg))
        loss = torch.nn.functional.softplus(-(g_pos - self.threshold)).mean() + \
            torch.nn.functional.softplus(g_neg - self.threshold).mean()
        self.opt.zero_grad()
        loss.backward()  # gradient is local to this layer only (inputs are detached)
        self.opt.step()
        return loss.item(), self.forward(x_pos).detach(), self.forward(x_neg).detach()


class ForwardForward(ScoringModel):
    name, family = "Forward-Forward", "local-learning"

    def __init__(self, seed=0, epochs=300, lr=3e-3, threshold=2.0, label_scale=3.0, batch_size=128):
        super().__init__(seed)
        self.epochs, self.lr, self.threshold, self.label_scale, self.batch_size = \
            epochs, lr, threshold, label_scale, batch_size

    def _embed(self, X, y):
        onehot = torch.zeros(len(X), 2)
        onehot[torch.arange(len(X)), y.long()] = self.label_scale
        return torch.cat([X, onehot], 1)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        Xt, yt = to_t(X), to_t(y)
        sizes = [X.shape[1] + 2, *HIDDEN]
        self.layers = [FFLayer(sizes[i], sizes[i + 1], self.lr, self.threshold) for i in range(len(HIDDEN))]
        n = len(Xt)
        self.hist = []
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                h_pos = self._embed(Xt[idx], yt[idx])
                h_neg = self._embed(Xt[idx], 1 - yt[idx])
                for layer in self.layers:
                    loss, h_pos, h_neg = layer.train_step(h_pos, h_neg)
                    tot += loss * len(idx)
            self.hist.append(tot / n)
        return self

    @torch.no_grad()
    def _total_goodness(self, Xt, label):
        h = self._embed(Xt, torch.full((len(Xt),), float(label)))
        g = torch.zeros(len(Xt))
        for layer in self.layers:
            h = layer(h)
            g += layer.goodness(h)
        return g

    def decision_scores(self, X):
        Xt = to_t(X)
        return (self._total_goodness(Xt, 1) - self._total_goodness(Xt, 0)).numpy()

    def describe(self):
        return f"arch={HIDDEN} theta={self.threshold} final_loss={self.hist[-1]:.3f}"

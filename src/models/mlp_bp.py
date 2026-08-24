"""Standard MLP trained with back-propagation (the reference for the PC / FF / neuro-evolution variants)."""
from __future__ import annotations

import torch

from .base import ScoringModel
from .torch_utils import HIDDEN, predict_logits, train_autograd


def build_mlp(d_in: int, hidden=HIDDEN, act=torch.nn.Tanh) -> torch.nn.Sequential:
    layers, d = [], d_in
    for h in hidden:
        layers += [torch.nn.Linear(d, h), act()]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers)


class MLPBP(ScoringModel):
    name, family = "MLP-BP", "backprop"

    def __init__(self, seed=0, epochs=300, lr=1e-3, weight_decay=1e-3):
        super().__init__(seed)
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.net = build_mlp(X.shape[1])
        self.hist = train_autograd(self.net, X, y, self.epochs, self.lr, self.weight_decay, seed=self.seed)
        return self

    def decision_scores(self, X):
        return predict_logits(self.net, X)

    def describe(self):
        return f"arch={HIDDEN} final_loss={self.hist[-1]:.3f}"

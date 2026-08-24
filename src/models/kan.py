"""Minimal Kolmogorov-Arnold Network (Liu et al., 2024), efficient-kan style.

Each edge carries a learnable univariate function  phi(x) = w_b * silu(x) + w_s * sum_i c_i B_i(x)
(B_i = cubic B-spline basis on a fixed grid). Layers sum the edge functions -> no fixed activation.
NB: KAN replaces the MLP *architecture*; it is still trained by gradient descent (autograd/backprop).
"""
from __future__ import annotations

import torch

from .base import ScoringModel
from .torch_utils import predict_logits, train_autograd

KAN_HIDDEN = (8,)


class KANLinear(torch.nn.Module):
    def __init__(self, d_in, d_out, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0)):
        super().__init__()
        self.d_in, self.d_out, self.k = d_in, d_out, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid)  # (G + 2k + 1,)
        n_basis = grid_size + spline_order
        self.base_weight = torch.nn.Parameter(torch.empty(d_out, d_in))
        self.spline_weight = torch.nn.Parameter(torch.empty(d_out, d_in, n_basis))
        self.spline_scaler = torch.nn.Parameter(torch.ones(d_out, d_in))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        torch.nn.init.normal_(self.spline_weight, 0, 0.1 / n_basis)

    def b_splines(self, x):  # x: (N, d_in) -> (N, d_in, n_basis) via Cox-de Boor recursion
        g = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)
        for k in range(1, self.k + 1):
            left = (x - g[: -(k + 1)]) / (g[k:-1] - g[: -(k + 1)]) * bases[..., :-1]
            right = (g[k + 1:] - x) / (g[k + 1:] - g[1:-k]) * bases[..., 1:]
            bases = left + right
        return bases

    def forward(self, x):
        base = torch.nn.functional.silu(x) @ self.base_weight.T
        B = self.b_splines(x.clamp(self.grid[self.k], self.grid[-self.k - 1] - 1e-4))
        W = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = torch.einsum("nib,oib->no", B, W)
        return base + spline


def build_kan(d_in, hidden=KAN_HIDDEN):
    layers, d = [], d_in
    for h in hidden:
        layers += [KANLinear(d, h), torch.nn.LayerNorm(h)]  # keep activations inside the spline grid
        d = h
    layers.append(KANLinear(d, 1))
    return torch.nn.Sequential(*layers)


class KAN(ScoringModel):
    name, family = "KAN", "backprop"

    def __init__(self, seed=0, epochs=300, lr=5e-3, weight_decay=1e-3):
        super().__init__(seed)
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.net = build_kan(X.shape[1])
        self.hist = train_autograd(self.net, X, y, self.epochs, self.lr, self.weight_decay, seed=self.seed)
        return self

    def decision_scores(self, X):
        return predict_logits(self.net, X)

    def describe(self):
        return f"arch={KAN_HIDDEN} grid=5 k=3 final_loss={self.hist[-1]:.3f}"

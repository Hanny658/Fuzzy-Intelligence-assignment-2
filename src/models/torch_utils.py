"""Shared helpers for the torch-based models (CPU only; data sets are tiny so we train full-batch)."""
from __future__ import annotations

import numpy as np
import torch

HIDDEN = (64, 32)  # common architecture d-64-32-1 for MLP-BP, MLP-PC, MLP-FF, MLP-CMA-ES, MLP-GA


def to_t(X) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X), dtype=torch.float32)


def train_autograd(model: torch.nn.Module, X, y, epochs=300, lr=1e-3, weight_decay=1e-3, batch_size=None,
                   seed=0) -> list:
    """Generic BCE-with-logits training loop (Adam). Used by MLP-BP, KAN and ANFIS."""
    g = torch.Generator().manual_seed(seed)
    Xt, yt = to_t(X), to_t(y).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = len(Xt)
    bs = n if batch_size is None else batch_size
    hist = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / n)
    model.eval()
    return hist


@torch.no_grad()
def predict_logits(model: torch.nn.Module, X) -> np.ndarray:
    model.eval()
    return model(to_t(X)).squeeze(1).cpu().numpy()

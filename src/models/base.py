"""Common model interface: fit(X, y) then decision_scores(X) -> 1-D array (higher = cancer)."""
from __future__ import annotations

import numpy as np


class ScoringModel:
    name: str = "base"
    family: str = "base"  # classical | backprop | local-learning | neuro-evolution | fuzzy | foundation

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScoringModel":
        raise NotImplementedError

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        return ""


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

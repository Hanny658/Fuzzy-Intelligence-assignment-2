"""Neuro-evolution: the SAME MLP (d-64-32-1, tanh) as MLP-BP, but weights found by gradient-free search.

  * CMAESMLP : (separable) CMA-ES from the `cma` package
  * GAMLP    : hand-written real-coded GA (tournament selection, BLX-alpha crossover, Gaussian mutation, elitism)
Fitness = binary cross-entropy on the training set + L2 penalty. A whole population is evaluated with one
batched numpy forward pass, so each generation costs a few matrix products.
"""
from __future__ import annotations

import numpy as np

from .base import ScoringModel
from .torch_utils import HIDDEN


class PopMLP:
    """Batched forward pass for a population of flat weight vectors."""

    def __init__(self, d_in, hidden=HIDDEN):
        self.sizes = [d_in, *hidden, 1]
        self.shapes = [(self.sizes[i], self.sizes[i + 1]) for i in range(len(self.sizes) - 1)]
        self.n_params = sum(a * b + b for a, b in self.shapes)

    def forward(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        """theta: (P, n_params) -> logits (P, N)."""
        P = theta.shape[0]
        h = np.broadcast_to(X, (P, *X.shape))
        off = 0
        for i, (a, b) in enumerate(self.shapes):
            W = theta[:, off:off + a * b].reshape(P, a, b)
            off += a * b
            bias = theta[:, off:off + b].reshape(P, 1, b)
            off += b
            h = h @ W + bias
            if i < len(self.shapes) - 1:
                h = np.tanh(h)
        return h[..., 0]

    def init(self, rng, P):
        """Glorot-style random initial population."""
        parts = []
        for a, b in self.shapes:
            parts.append(rng.normal(0, np.sqrt(2 / (a + b)), (P, a * b)))
            parts.append(np.zeros((P, b)))
        return np.concatenate(parts, 1)


def bce_fitness(logits: np.ndarray, y: np.ndarray, theta: np.ndarray, l2: float) -> np.ndarray:
    """Mean BCE per population member (lower is better) + L2."""
    z = logits
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return loss.mean(1) + l2 * (theta ** 2).mean(1)


class _EvoBase(ScoringModel):
    l2 = 1e-3

    def decision_scores(self, X):
        return self.net.forward(self.best[None], X)[0]


class CMAESMLP(_EvoBase):
    name, family = "CMA-ES-MLP", "neuro-evolution"

    def __init__(self, seed=0, generations=300, sigma0=0.3, popsize=32):
        super().__init__(seed)
        self.generations, self.sigma0, self.popsize = generations, sigma0, popsize

    def fit(self, X, y):
        import cma

        self.net = PopMLP(X.shape[1])
        rng = np.random.default_rng(self.seed)
        x0 = self.net.init(rng, 1)[0]
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, {
            "seed": self.seed + 1, "popsize": self.popsize, "CMA_diagonal": True,  # separable CMA-ES
            "verbose": -9, "maxiter": self.generations})
        self.hist = []
        while not es.stop():
            sols = np.array(es.ask())
            fit = bce_fitness(self.net.forward(sols, X), y, sols, self.l2)
            es.tell(sols, fit.tolist())
            self.hist.append(float(fit.min()))
        self.best = np.array(es.result.xbest)
        return self

    def describe(self):
        return f"arch={HIDDEN} sep-CMA-ES pop={self.popsize} gens={len(self.hist)} best_loss={self.hist[-1]:.3f}"


class GAMLP(_EvoBase):
    name, family = "GA-MLP", "neuro-evolution"

    def __init__(self, seed=0, generations=300, popsize=60, p_cross=0.9, alpha=0.5,
                 p_mut=0.05, sigma_mut=0.1, elite=2, tournament=3):
        super().__init__(seed)
        self.generations, self.popsize, self.p_cross, self.alpha = generations, popsize, p_cross, alpha
        self.p_mut, self.sigma_mut, self.elite, self.tournament = p_mut, sigma_mut, elite, tournament

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.net = PopMLP(X.shape[1])
        P, n = self.popsize, self.net.n_params
        pop = self.net.init(rng, P)
        fit = bce_fitness(self.net.forward(pop, X), y, pop, self.l2)
        self.hist = []
        for _ in range(self.generations):
            order = np.argsort(fit)
            elites = pop[order[:self.elite]]
            # tournament selection
            cand = rng.integers(0, P, (P - self.elite, 2, self.tournament))
            winners = cand[np.arange(P - self.elite)[:, None], np.arange(2)[None, :],
                           np.argmin(fit[cand], axis=2)]
            p1, p2 = pop[winners[:, 0]], pop[winners[:, 1]]
            # BLX-alpha crossover
            lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
            span = hi - lo
            child = rng.uniform(lo - self.alpha * span, hi + self.alpha * span)
            no_cross = rng.random(P - self.elite) > self.p_cross
            child[no_cross] = p1[no_cross]
            # Gaussian mutation
            mask = rng.random(child.shape) < self.p_mut
            child = child + mask * rng.normal(0, self.sigma_mut, child.shape)
            pop = np.vstack([elites, child])
            fit = bce_fitness(self.net.forward(pop, X), y, pop, self.l2)
            self.hist.append(float(fit.min()))
        self.best = pop[np.argmin(fit)]
        return self

    def describe(self):
        return f"arch={HIDDEN} GA pop={self.popsize} gens={self.generations} best_loss={self.hist[-1]:.3f}"

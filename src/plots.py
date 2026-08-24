"""ROC and summary figures (matplotlib, print-oriented light palette)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from metrics import eer  # noqa: E402

# fixed categorical order (validated palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5cf6", "#d1477a"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": SURFACE, "savefig.dpi": 200,
})

# ROC panels are derived from the families of the models actually passed in, so the figure cannot go
# stale when the reported model set changes. The groups mirror the paragraphs of the Models section;
# every model is drawn exactly once (no model is repeated across panels as a shared reference).
PANEL_GROUPS = [
    ("Classical baselines", ("classical",)),
    ("Neural networks", ("backprop", "local-learning", "neuro-evolution")),
    ("Fuzzy and pre-trained models", ("fuzzy", "foundation")),
]


def _panels(results: dict) -> list:
    """[(labelled title, [Result, ...]), ...] for the groups that contain at least one model."""
    groups = []
    for title, families in PANEL_GROUPS:
        members = [r for r in results.values() if r.family in families]
        if members:
            groups.append((title, members))
    return [(f"({chr(97 + i)}) {t}", m) for i, (t, m) in enumerate(groups)]


def _style_axes(ax):
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, linestyle=":", zorder=1)
    ax.plot([0, 1], [1, 0], color=AXIS, linewidth=0.8, linestyle="--", zorder=1)  # FPR = FNR line
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_aspect("equal")


def _draw_roc(ax, y, scores, color, label):
    fpr, tpr, _ = roc_curve(y, scores)
    e, _ = eer(y, scores)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (EER {e:.2f})", zorder=3)
    ax.scatter([e], [1 - e], s=48, color=color, edgecolor="white", linewidth=2, zorder=4)


def roc_panels(results: dict, dataset: str, path: str):
    """results: model name -> Result. Draws the test-set ROC in one panel per model family."""
    panels = _panels(results)
    ncol = min(len(panels), 3)
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.5 * nrow), squeeze=False)
    for ax, (title, members) in zip(axes.ravel(), panels):
        for k, r in enumerate(members):
            _draw_roc(ax, r.scores["y_test"], r.scores["test_scores"], SERIES[k % len(SERIES)], r.model)
        _style_axes(ax)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.legend(loc="lower right", fontsize=8)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{dataset}: test-set ROC curves; dots mark the interpolated equal-error point "
                 f"(dashed: FPR = FNR)", fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def roc_single(r, dataset: str, path: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    _draw_roc(ax, r.scores["y_test"], r.scores["test_scores"], SERIES[0], r.model)
    # also show the training-OOF threshold operating point
    ax.scatter([r.test_fpr_at_theta], [1 - r.test_fnr_at_theta], s=56, marker="s", facecolor="white",
               edgecolor=SERIES[0], linewidth=1.5, zorder=5,
               label=f"theta from train OOF: FPR {r.test_fpr_at_theta:.2f}, FNR {r.test_fnr_at_theta:.2f}")
    _style_axes(ax)
    ax.set_title(f"{dataset} - {r.model}  (AUC {r.test_auc:.3f})", loc="left", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cv_dotplot(results: dict, dataset: str, path: str, metric: str = "auc"):
    """Repeated-CV mean +- std per model, grouped by family, single hue."""
    order = ["classical", "backprop", "local-learning", "neuro-evolution", "fuzzy", "foundation"]
    rows = sorted(results.values(), key=lambda r: (order.index(r.family), -getattr(r, f"cv_{metric}_mean")))
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 1.2))
    ys = np.arange(len(rows))[::-1]
    for y0, r in zip(ys, rows):
        m, s = getattr(r, f"cv_{metric}_mean"), getattr(r, f"cv_{metric}_std")
        ax.plot([m - s, m + s], [y0, y0], color=SERIES[0], linewidth=2, alpha=0.5, zorder=2)
        ax.scatter([m], [y0], s=40, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(m + s + 0.01, y0, f"{m:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.model}  [{r.family}]" for r in rows], fontsize=8.5, color=INK)
    # family separators
    fams = [r.family for r in rows]
    for i in range(1, len(fams)):
        if fams[i] != fams[i - 1]:
            ax.axhline(ys[i] + 0.5, color=GRID, linewidth=0.8)
    ax.set_xlabel("Repeated stratified 5-fold CV " + ("AUC" if metric == "auc" else "EER") + " (mean +- std)")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{dataset}: cross-validated {'AUC' if metric == 'auc' else 'EER'} by model", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def training_curves(hists: dict, dataset: str, path: str):
    """Loss curves for the three training paradigms (normalised to their own first value)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for k, (name, h) in enumerate(hists.items()):
        h = np.asarray(h, dtype=float)
        ax.plot(np.arange(1, len(h) + 1), h / h[0], color=SERIES[k], linewidth=2, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training objective / initial value")
    ax.set_yscale("log")
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8)
    ax.set_title(f"{dataset}: convergence of BP, PC and FF (each its own objective)", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

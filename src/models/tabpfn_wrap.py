"""TabPFN (Hollmann et al., Nature 2025; here the v3 checkpoint of the `tabpfn` package): a transformer
pre-trained on synthetic tabular tasks that classifies by in-context learning -- no gradient training on
our data. Used as a modern reference.

Requirements / quirks (verified with tabpfn 8.4.0):
  * The weights need a one-off licence acceptance. Put your PriorLabs API key in the project `.env` as
    `TOKEN=...` (or export `TABPFN_TOKEN`); the package verifies it against api.priorlabs.ai and caches it.
  * The ~213 MB checkpoint is fetched from HuggingFace. The `hf_xet` download backend stalls on this
    machine, so we disable it (HF_HUB_DISABLE_XET=1); if a download still hangs, fetch the file with
    curl into the cache directory printed by `tabpfn.model_loading.get_cache_dir()`.
  * `fit` only stores the training set; the compute happens in `predict_proba`, so the reported fit time
    understates the cost of this model.
"""
from __future__ import annotations

import os

from .base import ScoringModel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_token_from_env_file() -> None:
    """Export TABPFN_TOKEN from the project .env (keys TOKEN or TABPFN_TOKEN) if not already set."""
    if os.environ.get("TABPFN_TOKEN"):
        return
    path = os.path.join(ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("TOKEN", "TABPFN_TOKEN"):
                os.environ["TABPFN_TOKEN"] = v.strip().strip("'\"")
                return


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401
        return True
    except Exception:
        return False


class TabPFN(ScoringModel):
    name, family = "TabPFN", "foundation"

    def fit(self, X, y):
        _load_token_from_env_file()
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # never block a headless run on a browser login
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        # On CPU the package refuses contexts above 5000 rows because inference gets slow. SUPPORT2
        # has 7284 training rows, so the guard is lifted deliberately: one predict call over the
        # full context takes ~5 min here, against ~1 s for every other model in the comparison.
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        if not os.environ.get("TABPFN_TOKEN") and not os.environ.get("TABPFN_ALLOW_INTERACTIVE"):
            raise RuntimeError("TabPFN weights need a licence token: set TABPFN_TOKEN or TOKEN in .env "
                               "(see https://ux.priorlabs.ai/account) - model skipped")
        import tabpfn
        from tabpfn import TabPFNClassifier
        from tabpfn.settings import settings

        self.version = f"tabpfn {tabpfn.__version__} / {settings.tabpfn.model_version.value}"
        self.m = TabPFNClassifier(device="cpu", random_state=self.seed)
        self.m.fit(X, y)
        return self

    def decision_scores(self, X):
        return self.m.predict_proba(X)[:, 1]

    def describe(self):
        return f"{getattr(self, 'version', 'tabpfn')} pretrained in-context classifier (no training on our data)"

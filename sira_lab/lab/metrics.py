"""Evaluation metrics (numpy only)."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Sequence

import numpy as np


def tpr_at_fpr(pos: Sequence[float], neg: Sequence[float], fpr: float = 0.01) -> float:
    """TPR of a score-threshold detector at the given false-positive rate.
    pos = scores of texts that SHOULD be detected; neg = scores of human/unwatermarked texts."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    thr = np.quantile(neg, 1.0 - fpr)
    return float(np.mean(pos > thr))


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def edit_ratio(a: str, b: str) -> float:
    """Word-level normalized edit similarity in [0, 1] (1 = identical)."""
    return SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def perplexity(mean_surprisal_nats: float) -> float:
    return float(np.exp(mean_surprisal_nats))


def changed_share(orig: str, new: str) -> float:
    """Share of the ORIGINAL words that were replaced, deleted or rewritten (0 = identical, 1 = everything changed)."""
    wa, wb = orig.split(), new.split()
    if not wa:
        return 0.0
    sm = SequenceMatcher(None, wa, wb, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return float(min(1.0, changed / len(wa)))

"""Reference (numpy) maths for the two zero-shot detectors. The GPU versions in detectors.py must match these.

Both take the next-token logits [L, V] for a text of L tokens (row t predicts token t) and the token ids [L].
Convention for everything in the Lab: HIGHER score = more likely machine-written.
"""
from __future__ import annotations

import numpy as np


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def fast_detectgpt_np(logits_score: np.ndarray, logits_ref: np.ndarray, ids: np.ndarray) -> float:
    """Analytic Fast-DetectGPT criterion (Bao et al., ICLR 2024):
    (log p(x) - sum_t E_{x~p_ref}[log p_score(x)]) / sqrt(sum_t Var)."""
    lp = _log_softmax(logits_score.astype(np.float64))
    pr = np.exp(_log_softmax(logits_ref.astype(np.float64)))
    ll = lp[np.arange(len(ids)), ids]
    mu = (pr * lp).sum(-1)
    var = (pr * lp ** 2).sum(-1) - mu ** 2
    return float((ll.sum() - mu.sum()) / np.sqrt(var.sum()))


def binoculars_np(logits_observer: np.ndarray, logits_performer: np.ndarray, ids: np.ndarray) -> float:
    """Binoculars (Hans et al., ICML 2024): log-perplexity under the performer divided by the cross-perplexity
    between observer and performer. Lower raw ratio = more machine-like, so we return the NEGATIVE ratio."""
    lq = _log_softmax(logits_performer.astype(np.float64))
    log_ppl = -lq[np.arange(len(ids)), ids].mean()
    p_obs = np.exp(_log_softmax(logits_observer.astype(np.float64)))
    x_ppl = -(p_obs * lq).sum(-1).mean()
    return float(-(log_ppl / x_ppl))

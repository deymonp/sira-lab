"""Green-list text watermarks (Kirchenbauer et al. 2023 "KGW" and Zhao et al. 2023 "Unigram").

Own re-implementation so the Lab has no fragile dependencies and can be unit-tested without a GPU.
Statistics are identical to the reference versions (green fraction gamma, bias delta, z-test);
only the pseudo-random function that decides which tokens are green differs.

  kgw     : green(prev_token, token)  -> depends on the previous token (context width 1)
  unigram : green(token)              -> one fixed green list for the whole vocabulary
  exp     : Aaronson/Kirchner exponential-minimum ("Gumbel") sampling. Distribution-preserving: the next token is
            argmax_v log(u_v)/p_v with u_v a keyed pseudo-random number. Detection sums -log(1-u) over the text.
  synthid : SynthID-Text (Dathathri et al. 2024) via the reference implementation shipped in `transformers`.
            NOTE: that implementation hashes differently from Google's production (Gemini) watermark, so
            results here do NOT transfer to Gemini output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

_A = 0x9E3779B1
_B = 0x85EBCA6B
_M32 = 0xFFFFFFFF


@dataclass(frozen=True)
class WatermarkSpec:
    algo: str = "kgw"          # kgw | unigram
    gamma: float = 0.5         # fraction of the vocabulary that is green
    delta: float = 2.0         # logit bias added to green tokens while generating
    key: int = 15485863        # secret key
    z_threshold: float = 4.0   # conventional detection threshold


def hash_bits(prev, tok, spec: "WatermarkSpec"):
    """32-bit keyed pseudo-random value for (prev, tok). Works on ints, numpy and torch int64 tensors."""
    return _hash32((prev * _A + tok * _B + spec.key) & _M32) & _M32


def synthid_keys(spec: "WatermarkSpec", depth: int = 9) -> List[int]:
    rs = np.random.RandomState(spec.key % (2 ** 31))
    return [int(x) for x in rs.randint(0, 100_000, depth)]


def _hash32(x):
    x = (((x >> 16) ^ x) * 0x45D9F3B) & _M32
    x = (((x >> 16) ^ x) * 0x45D9F3B) & _M32
    return (x >> 16) ^ x


def green_mask(prev, tok, spec: WatermarkSpec):
    """True where `tok` is green after `prev`. Works on python ints, numpy and torch int64 tensors alike."""
    if spec.algo == "unigram":
        x0 = (tok * _B + spec.key) & _M32
    elif spec.algo == "kgw":
        x0 = (prev * _A + tok * _B + spec.key) & _M32
    else:
        raise ValueError(f"unknown watermark algorithm {spec.algo!r}")
    return (_hash32(x0) & 0xFFFF) < int(spec.gamma * 65536)


def z_score(n_green: float, n: int, gamma: float) -> float:
    if n <= 0:
        return 0.0
    return (n_green - gamma * n) / math.sqrt(n * gamma * (1.0 - gamma))


def detect_ids(ids: Sequence[int], spec: WatermarkSpec) -> Dict[str, float]:
    """Score a token-id sequence. The first token has no context and is skipped."""
    a = np.asarray(ids, dtype=np.int64)
    if len(a) < 2:
        return {"z": 0.0, "green_frac": 0.0, "n_scored": 0}
    if spec.algo == "synthid":
        raise ValueError("synthid is scored by detectors.SynthIDMean (needs torch/transformers)")
    if spec.algo == "exp":
        u = (hash_bits(a[:-1], a[1:], spec).astype(np.float64) + 0.5) / 4294967296.0
        n = len(a) - 1
        s = float(np.sum(-np.log1p(-u)))                     # ~ Gamma(n, 1) for unwatermarked text
        return {"z": (s - n) / math.sqrt(n), "green_frac": float(np.mean(u > 0.5)), "n_scored": n}
    green = green_mask(a[:-1], a[1:], spec)
    n, g = len(a) - 1, int(np.sum(green))
    return {"z": z_score(g, n, spec.gamma), "green_frac": g / n, "n_scored": n}


def detect_text(text: str, encode: Callable[[str], List[int]], spec: WatermarkSpec) -> Dict[str, float]:
    return detect_ids(encode(text), spec)


def make_logits_processor(spec: WatermarkSpec):
    """A transformers LogitsProcessor implementing the watermark (kgw/unigram: +delta on green; exp: forced pick)."""
    import torch
    from transformers import LogitsProcessor

    if spec.algo == "exp":
        class _ExpProcessor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                prev = input_ids[:, -1].to(torch.int64)
                tok = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int64)
                h = hash_bits(prev[:, None], tok[None, :], spec)
                u = (h.to(torch.float64) + 0.5) / 4294967296.0
                p = torch.softmax(scores.float(), dim=-1).to(torch.float64)
                pick = torch.argmax(torch.log(u) / p.clamp_min(1e-30), dim=-1)
                out = torch.full_like(scores, float("-inf"))
                out.scatter_(1, pick[:, None], 0.0)
                return out

        return _ExpProcessor()
    if spec.algo not in ("kgw", "unigram"):
        raise ValueError(f"no logits processor for {spec.algo!r}")

    class _WatermarkProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            prev = input_ids[:, -1].to(torch.int64)
            tok = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int64)
            mask = green_mask(prev[:, None], tok[None, :], spec)
            return scores + spec.delta * mask.to(scores.dtype)

    return _WatermarkProcessor()

"""Quality gates: semantic similarity and fluency (perplexity ratio)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Protocol

from .config import GateCfg


class SimModel(Protocol):
    def similarity(self, a: str, b: str) -> float: ...


class STSimilarity:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name)

    def similarity(self, a: str, b: str) -> float:
        from sentence_transformers import util

        ea, eb = self.model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(ea, eb))


@dataclass
class GateResult:
    ok: bool
    sim: float
    ppl_ratio: float
    reasons: List[str] = field(default_factory=list)


def check(orig: str, cand: str, orig_mean_surprisal: float, sim_model: SimModel, scorer, cfg: GateCfg) -> GateResult:
    sim = sim_model.similarity(orig, cand)
    ratio = math.exp(scorer.mean_surprisal(cand) - orig_mean_surprisal)
    reasons = []
    if sim < cfg.min_sim:
        reasons.append(f"similarity {sim:.2f} < {cfg.min_sim}")
    if ratio > cfg.max_ppl_ratio:
        reasons.append(f"perplexity ratio {ratio:.2f} > {cfg.max_ppl_ratio}")
    return GateResult(not reasons or not cfg.enforce, sim, ratio, reasons)

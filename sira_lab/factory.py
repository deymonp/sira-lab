"""Build a real (GPU) rewriter from a Config. Imports torch/transformers lazily."""
from __future__ import annotations

from .backends import HFBackend
from .config import Config
from .gates import STSimilarity
from .pipeline import SiraRewriter
from .scorer import HFScorer


def build_rewriter(cfg: Config) -> SiraRewriter:
    scorer = HFScorer(cfg.models.scorer, dtype=cfg.models.dtype, device_map=cfg.models.device_map)
    backend = HFBackend(cfg.models.infiller, quantize_4bit=cfg.models.quantize_4bit, dtype=cfg.models.dtype,
                        device_map=cfg.models.device_map, batch_size=cfg.gen.batch_size)
    sim = STSimilarity(cfg.models.sim)
    return SiraRewriter(cfg, scorer, backend, sim)

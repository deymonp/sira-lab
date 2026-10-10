"""Typed configuration. YAML files in configs/ override these defaults."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelCfg:
    scorer: str = "Qwen/Qwen2.5-0.5B"            # small LM used only to compute self-information
    infiller: str = "Qwen/Qwen2.5-7B-Instruct"   # open-weights LLM that fills the masks
    sim: str = "sentence-transformers/all-MiniLM-L6-v2"
    quantize_4bit: bool = True                    # True on Colab T4; False on ZeroGPU
    dtype: str = "float16"                        # T4 has no bf16
    device_map: str = "auto"


@dataclass
class MaskCfg:
    mask_ratio: float = 0.30        # fraction of ALL words in a chunk to mask
    min_surprisal: float = 0.0      # nats; words below this are never masked
    agg: str = "max"                # how token surprisals combine into a word score: max | mean | sum
    min_gap: int = 0                # unmasked words required between two masks
    protect_acronyms: bool = True
    protect_capitalized: bool = True  # mid-sentence capitalised words (names) are kept
    protect_quotes: bool = True


@dataclass
class GenCfg:
    max_new_tokens: int = 700
    temperature: float = 0.7
    top_p: float = 0.9
    batch_size: int = 4
    max_attempts: int = 3           # covers parse failures and gate failures
    strict: bool = True             # only accept outputs whose unmasked text is untouched


@dataclass
class GateCfg:
    min_sim: float = 0.85           # cosine similarity vs. the ORIGINAL chunk
    max_ppl_ratio: float = 1.6      # perplexity(out) / perplexity(orig) under the scorer LM
    ratio_decay: float = 0.7        # mask_ratio multiplier after a failed gate
    on_fail: str = "best"           # best | original


@dataclass
class ChunkCfg:
    max_words: int = 250
    min_words: int = 8
    skip_headings: bool = True
    min_alpha_ratio: float = 0.6    # paragraphs with fewer letters than this (equations, tables) are skipped


@dataclass
class Config:
    models: ModelCfg = field(default_factory=ModelCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    gen: GenCfg = field(default_factory=GenCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    chunk: ChunkCfg = field(default_factory=ChunkCfg)
    rounds: int = 1
    seed: int = 42
    max_input_words: int = 1200


def _merge(obj: Any, data: dict) -> Any:
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {key!r}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    cfg = Config()
    if path:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _merge(cfg, data)
    if overrides:
        _merge(cfg, overrides)
    return cfg


def config_hash(cfg: Config) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]

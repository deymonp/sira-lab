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
    strategy: str = "surprisal"     # surprisal (SIRA) | random (control) | lowest (control: mask the most predictable words)
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
    strict: bool = True             # passage mode only: reject outputs whose unmasked text was altered
    mode: str = "list"              # list = model returns only the fills ("1: word"); passage = model rewrites the whole passage
    fallback_to_passage: bool = True  # if list mode fails to parse, retry in passage mode
    hygiene: bool = False           # fill hygiene (v8): plain-word fills only, no repeats of neighbouring words, a/an fixed


@dataclass
class GateCfg:
    min_sim: float = 0.85           # cosine similarity vs. the ORIGINAL chunk
    max_ppl_ratio: float = 1.6      # perplexity(out) / perplexity(orig) under the scorer LM
    ratio_decay: float = 0.7        # mask_ratio multiplier after a failed gate
    on_fail: str = "best"           # best | original
    enforce: bool = True            # False = still measure similarity/perplexity, but never reject a rewrite because of them


@dataclass
class ChunkCfg:
    max_words: int = 250
    min_words: int = 8
    skip_headings: bool = True
    min_alpha_ratio: float = 0.6    # paragraphs with fewer letters than this (equations, tables) are skipped


@dataclass
class LabCfg:
    generator: str = "Qwen/Qwen2.5-1.5B"        # model that writes the (water)marked text
    source: str = "hf"                           # hf (arXiv abstracts) | builtin (small prompt list)
    dataset: str = "gfissore/arxiv-abstracts-2021"
    text_field: str = "abstract"
    n_samples: int = 100                         # per class (watermarked per algorithm, unwatermarked, human)
    prompt_words: int = 30
    max_new_tokens: int = 200
    min_new_tokens: int = 150
    gen_batch_size: int = 16
    attack_batch_size: int = 16                  # texts per rewrite_many call
    temperature: float = 0.8                     # sampling settings used for ALL generated text
    top_p: float = 0.95                          # (zero-shot detectors only see machine text when it is not pure T=1 sampling)
    algorithms: list = field(default_factory=lambda: ["kgw", "unigram"])   # kgw | unigram | exp | synthid
    detectors: list = field(default_factory=lambda: ["fast_detectgpt", "binoculars"])   # zero-shot panel
    fdg_model: str = "Qwen/Qwen2.5-1.5B"
    bino_observer: str = "Qwen/Qwen2.5-1.5B"
    bino_performer: str = "Qwen/Qwen2.5-1.5B-Instruct"
    detect_tokens: int = 150                     # zero-shot detectors score only the first N tokens of every text
    ho_fdg_model: str = "HuggingFaceTB/SmolLM2-1.7B"            # HELD-OUT zero-shot detectors: a different model family
    ho_bino_observer: str = "HuggingFaceTB/SmolLM2-1.7B"        # than the generator, so they are not 'home turf' detectors
    ho_bino_performer: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    judge_batch_size: int = 8                    # LLM-judge prompts per GPU batch
    judge_b_model: str = "microsoft/Phi-3.5-mini-instruct"   # independent second judge (different model family) for final grades
    prep_attacks: list = field(default_factory=list)          # computed first, but NOT scored or judged (inputs for chains)
    min_fluency: int = 4                         # a rewrite counts as 'good quality' if the judge gives at least this fluency...
    min_meaning: int = 4                         # ...and at least this meaning-preservation score (both 1-5)
    gamma: float = 0.5
    delta: float = 2.0
    key: int = 15485863
    z_threshold: float = 4.0
    attacks: list = field(default_factory=lambda: ["sira", "random", "lowest", "paraphrase"])
    fpr: float = 0.01


@dataclass
class Config:
    models: ModelCfg = field(default_factory=ModelCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    gen: GenCfg = field(default_factory=GenCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    chunk: ChunkCfg = field(default_factory=ChunkCfg)
    lab: LabCfg = field(default_factory=LabCfg)
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

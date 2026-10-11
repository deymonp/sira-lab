"""Prompt sources. 'hf' streams arXiv abstracts; 'builtin' is a small offline list for smoke tests."""
from __future__ import annotations

import random
from typing import List

from ..config import LabCfg

BUILTIN = [
    "Recent advances in large language models have raised questions about how reliably machine-generated text can be identified.",
    "We study the trade-off between model size and calibration in neural networks trained on small scientific datasets.",
    "Graph neural networks have become a standard tool for predicting molecular properties from structural data.",
    "In this paper we propose a new method for estimating causal effects from observational data with hidden confounders.",
    "Sparse attention mechanisms promise to reduce the quadratic cost of transformers on long documents.",
    "Climate models disagree on regional precipitation changes, which complicates adaptation planning for agriculture.",
    "We present a randomized trial comparing two physiotherapy protocols for patients recovering from knee surgery.",
    "Quantum error correction remains the central obstacle to building fault-tolerant quantum computers at scale.",
    "The relationship between sleep duration and cognitive performance in older adults has been studied extensively.",
    "Reinforcement learning from human feedback has been widely adopted to align language models with user preferences.",
    "Single-cell sequencing has revealed unexpected heterogeneity in tumour microenvironments across cancer types.",
    "We investigate whether contrastive pretraining improves robustness of image classifiers under distribution shift.",
    "Federated learning allows hospitals to train shared models without exchanging patient records.",
    "Perovskite solar cells have reached high efficiencies, but their long-term stability under heat and humidity is limited.",
    "This work examines how urban green space affects air quality and respiratory health in dense cities.",
    "Bayesian optimization offers a sample-efficient approach to tuning expensive simulations in materials science.",
    "We analyze the energy consumption of training large neural networks and propose scheduling strategies to reduce it.",
    "Machine translation for low-resource languages is limited by the scarcity of parallel corpora.",
    "Protein structure prediction has been transformed by deep learning, yet predicting dynamics remains difficult.",
    "Social media data has been used to track the spread of misinformation during public health emergencies.",
]


def _first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def load_prompts(cfg: LabCfg, seed: int = 42) -> List[dict]:
    """Return [{id, prompt, human}] where `human` is real continuation text (may be empty for builtin)."""
    n = cfg.n_samples
    rows: List[dict] = []
    if cfg.source == "hf":
        try:
            from datasets import load_dataset

            ds = load_dataset(cfg.dataset, split="train", streaming=True).shuffle(seed=seed, buffer_size=2000)
            for ex in ds:
                words = (ex.get(cfg.text_field) or "").split()
                if len(words) < cfg.prompt_words + 90:
                    continue
                rows.append({
                    "id": len(rows),
                    "prompt": " ".join(words[:cfg.prompt_words]),
                    "human": " ".join(words[cfg.prompt_words:cfg.prompt_words + 200]),
                })
                if len(rows) >= n:
                    break
        except Exception as e:  # network/dataset problems must not kill the run
            print(f"[prompts] could not load {cfg.dataset!r} ({type(e).__name__}: {e}); using the builtin list instead.")
            rows = []
    if len(rows) < n:
        rng = random.Random(seed)
        base = [r for r in rows]
        i = len(rows)
        while len(rows) < n:
            rows.append({"id": i, "prompt": _first_words(rng.choice(BUILTIN), cfg.prompt_words), "human": ""})
            i += 1
        if not base:
            print("[prompts] builtin prompts have no human reference text; the human-FPR check will be skipped.")
    return rows[:n]

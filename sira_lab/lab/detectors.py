"""Detector panel. Every detector exposes:  .name  .kind ('watermark' | 'zeroshot')  .score(texts) -> np.ndarray
Convention: HIGHER score = more likely watermarked / machine-written."""
from __future__ import annotations

import math
from typing import Callable, List, Optional

import numpy as np

from .watermark import WatermarkSpec, detect_text, synthid_keys


def make_encoder(model_name: str) -> Callable[[str], List[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    return lambda t: tok(t, add_special_tokens=False)["input_ids"]


# ------------------------------------------------------------------ watermark detectors
class WatermarkZ:
    kind = "watermark"

    def __init__(self, spec: WatermarkSpec, encode: Callable[[str], List[int]]):
        self.spec, self.encode = spec, encode
        self.algo, self.name = spec.algo, f"wm_{spec.algo}"
        self.threshold = spec.z_threshold

    def score(self, texts: List[str]) -> np.ndarray:
        return np.array([detect_text(t, self.encode, self.spec)["z"] for t in texts], dtype=float)


class SynthIDMean:
    """Mean g-value detector for the transformers SynthID-Text reference implementation (no training needed).
    Under the null the mean g-value is 0.5; we convert it to a z-score. Must run on the same device type
    (cuda/cpu) that generated the text, because the sampling table is drawn with a device-bound RNG."""
    kind = "watermark"
    algo, name = "synthid", "wm_synthid"

    def __init__(self, spec: WatermarkSpec, tokenizer_name: str, ngram_len: int = 5):
        import torch
        from transformers import AutoTokenizer, SynthIDTextWatermarkLogitsProcessor

        self.torch, self.spec, self.ngram_len = torch, spec, ngram_len
        self.threshold = spec.z_threshold
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("[synthid] WARNING: no GPU; if the texts were generated on a GPU the g-values will not match.")
        self.proc = SynthIDTextWatermarkLogitsProcessor(
            ngram_len=ngram_len, keys=synthid_keys(spec), sampling_table_size=2 ** 16,
            sampling_table_seed=0, context_history_size=1024, device=self.device)

    def _z(self, text: str) -> float:
        torch = self.torch
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < self.ngram_len + 5:
            return 0.0
        x = torch.tensor([ids], device=self.device)
        n = self.ngram_len
        g = self.proc.compute_g_values(x)                                           # [1, L-(n-1), depth]
        rep = self.proc.compute_context_repetition_mask(x)                          # [1, L-(n-1)]
        eos = self.proc.compute_eos_token_mask(x, self.tok.eos_token_id)[:, n - 1:]
        mask = (rep * eos).float()
        total = float((g.float() * mask[..., None]).sum())
        m = float(mask.sum()) * g.shape[-1]
        return 0.0 if m <= 0 else (total - 0.5 * m) / math.sqrt(0.25 * m)

    def score(self, texts: List[str]) -> np.ndarray:
        return np.array([self._z(t) for t in texts], dtype=float)


# ------------------------------------------------------------------ zero-shot detectors
class _LM:
    def __init__(self, name: str, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..scorer import dtype_kwarg

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", **dtype_kwarg(torch, dtype)).eval()
        self.start = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    def ids(self, text: str, max_tokens: int) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"][:max_tokens]

    def logits(self, ids: List[int]):
        """[L, V] float32 logits, row t predicts ids[t] (a start token is prepended so every token is scored)."""
        torch = self.torch
        inp = torch.tensor([[self.start] + ids], device=self.model.device)
        with torch.inference_mode():
            return self.model(inp).logits[0, :-1].float()


def fast_detectgpt_torch(logits_score, logits_ref, ids):
    import torch

    lp = torch.log_softmax(logits_score, dim=-1)
    pr = torch.softmax(logits_ref, dim=-1)
    ll = lp.gather(1, ids[:, None]).squeeze(1)
    mu = (pr * lp).sum(-1)
    var = (pr * lp ** 2).sum(-1) - mu ** 2
    return float((ll.sum() - mu.sum()) / var.sum().sqrt())


def binoculars_torch(logits_observer, logits_performer, ids):
    import torch

    lq = torch.log_softmax(logits_performer, dim=-1)
    log_ppl = -lq.gather(1, ids[:, None]).squeeze(1).mean()
    p_obs = torch.softmax(logits_observer, dim=-1)
    x_ppl = -(p_obs * lq).sum(-1).mean()
    return float(-(log_ppl / x_ppl))


class FastDetectGPT:
    kind, name, algo = "zeroshot", "fast_detectgpt", None

    def __init__(self, model_name: str, dtype: str = "float16", max_tokens: int = 150):
        self.lm, self.max_tokens, self.threshold = _LM(model_name, dtype), max_tokens, None

    def score(self, texts: List[str]) -> np.ndarray:
        out = []
        for t in texts:
            ids = self.lm.ids(t, self.max_tokens)
            if len(ids) < 20:
                out.append(0.0)
                continue
            lg = self.lm.logits(ids)
            out.append(fast_detectgpt_torch(lg, lg, self.lm.torch.tensor(ids, device=lg.device)))
        return np.array(out, dtype=float)


class Binoculars:
    kind, name, algo = "zeroshot", "binoculars", None

    def __init__(self, observer: str, performer: str, dtype: str = "float16", max_tokens: int = 150):
        self.obs, self.perf, self.max_tokens, self.threshold = _LM(observer, dtype), _LM(performer, dtype), max_tokens, None
        probe = "The quick brown fox jumps over the lazy dog."
        if self.obs.ids(probe, 99) != self.perf.ids(probe, 99):
            raise ValueError("Binoculars needs observer and performer to share a tokenizer.")

    def score(self, texts: List[str]) -> np.ndarray:
        out = []
        for t in texts:
            ids = self.obs.ids(t, self.max_tokens)
            if len(ids) < 20:
                out.append(0.0)
                continue
            i = self.obs.torch.tensor(ids, device=self.perf.model.device)
            lo, lp = self.obs.logits(ids), self.perf.logits(ids)
            out.append(binoculars_torch(lo.to(lp.device), lp, i))
        return np.array(out, dtype=float)


# ------------------------------------------------------------------ the panel
def detector_factories(cfg) -> List[Callable[[], object]]:
    """One zero-argument factory per detector, so evaluation can load -> score -> free them one at a time."""
    from .generate import spec_for

    lab, facs = cfg.lab, []
    for algo in lab.algorithms:
        spec = spec_for(lab, algo)
        if algo == "synthid":
            facs.append(lambda spec=spec: SynthIDMean(spec, lab.generator))
        else:
            facs.append(lambda spec=spec: WatermarkZ(spec, make_encoder(lab.generator)))
    for name in lab.detectors:
        if name == "fast_detectgpt":
            facs.append(lambda: FastDetectGPT(lab.fdg_model, cfg.models.dtype, lab.detect_tokens))
        elif name == "binoculars":
            facs.append(lambda: Binoculars(lab.bino_observer, lab.bino_performer, cfg.models.dtype, lab.detect_tokens))
        else:
            raise ValueError(f"unknown detector {name!r}")
    return facs

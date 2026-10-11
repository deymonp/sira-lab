"""Stage 1: write watermarked / unwatermarked texts with an open LM (needs torch + transformers)."""
from __future__ import annotations

from typing import List, Optional

from ..config import LabCfg
from .watermark import WatermarkSpec, make_logits_processor, synthid_keys


def spec_for(lab: LabCfg, algo: str) -> WatermarkSpec:
    return WatermarkSpec(algo=algo, gamma=lab.gamma, delta=lab.delta, key=lab.key, z_threshold=lab.z_threshold)


def load_generator(name: str, dtype: str = "float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..scorer import dtype_kwarg

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", **dtype_kwarg(torch, dtype)).eval()
    return model, tok


def generate_texts(model, tok, prompts: List[str], lab: LabCfg, spec: Optional[WatermarkSpec], seed: int = 0) -> List[str]:
    """Sample a continuation for every prompt. spec=None -> ordinary sampling (negative class)."""
    import torch
    from transformers import LogitsProcessorList

    out_texts: List[str] = []
    for i in range(0, len(prompts), lab.gen_batch_size):
        batch = prompts[i:i + lab.gen_batch_size]
        torch.manual_seed(seed + i)
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        kwargs = dict(
            do_sample=True, top_k=0, top_p=lab.top_p, temperature=lab.temperature,
            max_new_tokens=lab.max_new_tokens, min_new_tokens=lab.min_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
        if spec is not None and spec.algo == "synthid":
            from transformers import SynthIDTextWatermarkingConfig

            kwargs["watermarking_config"] = SynthIDTextWatermarkingConfig(keys=synthid_keys(spec), ngram_len=5)
        elif spec is not None:
            kwargs["logits_processor"] = LogitsProcessorList([make_logits_processor(spec)])
        with torch.inference_mode():
            out = model.generate(**enc, **kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]
        out_texts += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return out_texts

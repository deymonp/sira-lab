"""Text-generation backends for the infiller."""
from __future__ import annotations

from typing import Dict, List, Protocol

Messages = List[Dict[str, str]]


class Backend(Protocol):
    def generate(self, batch: List[Messages], max_new_tokens: int, temperature: float, top_p: float) -> List[str]: ...


class HFBackend:
    def __init__(self, name: str, quantize_4bit: bool = True, dtype: str = "float16",
                 device_map: str = "auto", batch_size: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {"device_map": device_map}
        if quantize_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, dtype),
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()

    def generate(self, batch, max_new_tokens, temperature, top_p):
        torch = self.torch
        outs: List[str] = []
        for i in range(0, len(batch), self.batch_size):
            sub = batch[i:i + self.batch_size]
            prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in sub]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.inference_mode():
                out = self.model.generate(**enc, **gen_kwargs)
            outs += self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

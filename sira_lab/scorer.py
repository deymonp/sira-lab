"""Per-token self-information (surprisal, in nats) from a small causal LM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple


@dataclass
class TokenScores:
    spans: List[Tuple[int, int]]   # character offsets of each token in the scored text
    surprisal: List[float]         # -log p(token | previous tokens), nats


class Scorer(Protocol):
    def score(self, text: str) -> TokenScores: ...
    def mean_surprisal(self, text: str) -> float: ...


class HFScorer:
    def __init__(self, name: str, dtype: str = "float16", device_map: str = "auto", max_tokens: int = 1536):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        if not getattr(self.tok, "is_fast", False):
            raise ValueError("Scorer needs a fast tokenizer (offset mappings).")
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=getattr(torch, dtype), device_map=device_map
        ).eval()
        self.max_tokens = max_tokens
        # Models like Qwen have no BOS; use EOS as the context token so the first real token is scored too.
        self.start_id = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    def score(self, text: str) -> TokenScores:
        torch = self.torch
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if not ids:
            return TokenScores([], [])
        if len(ids) > self.max_tokens:
            raise ValueError(f"Chunk has {len(ids)} tokens (> {self.max_tokens}); lower chunk.max_words.")
        dev = self.model.device
        inp = torch.tensor([[self.start_id] + ids], device=dev)
        with torch.inference_mode():
            logits = self.model(inp).logits[0, :-1].float()
        tgt = torch.tensor(ids, device=dev)
        s = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
        return TokenScores([(int(a), int(b)) for a, b in offs], s.cpu().tolist())

    def mean_surprisal(self, text: str) -> float:
        s = self.score(text).surprisal
        return sum(s) / len(s) if s else 0.0

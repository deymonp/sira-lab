"""Fake components so the whole pipeline (and the UI) can run on CPU with no downloads."""
from __future__ import annotations

import re
import zlib
from difflib import SequenceMatcher

from .config import Config
from .masker import MASK_RE
from .pipeline import SiraRewriter
from .scorer import TokenScores


class FakeScorer:
    """Deterministic pseudo-surprisal per word/punctuation token."""

    def score(self, text: str) -> TokenScores:
        spans, vals = [], []
        for m in re.finditer(r"\w+|[^\w\s]", text):
            spans.append(m.span())
            vals.append((zlib.crc32(m.group(0).lower().encode()) % 1000) / 100.0)
        return TokenScores(spans, vals)

    def mean_surprisal(self, text: str) -> float:
        v = self.score(text).surprisal
        return sum(v) / len(v) if v else 0.0


class FakeBackend:
    """Fills every <mask_N> with 'altN'. Options simulate misbehaving models."""

    def __init__(self, fail_first: int = 0, corrupt: bool = False):
        self.fail_first, self.corrupt, self.calls = fail_first, corrupt, 0

    def generate(self, batch, max_new_tokens, temperature, top_p):
        outs = []
        for messages in batch:
            self.calls += 1
            user = messages[-1]["content"]
            m = re.search(r"<passage>\n(.*?)\n</passage>", user, flags=re.S)
            template = m.group(1)
            if self.calls <= self.fail_first:
                outs.append("Sorry, I cannot do that.")
                continue
            if "one line per placeholder" in user:
                nums = MASK_RE.findall(template)
                outs.append("\n".join(f"{n}: alt{n}" for n in nums))
                continue
            text = MASK_RE.sub(lambda mm: f"alt{mm.group(1)}", template)
            if self.corrupt:
                text = "Completely unrelated sentence about cooking pasta."
            outs.append(f"<passage>\n{text}\n</passage>")
        return outs


class DifflibSim:
    def similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.split(), b.split()).ratio()


def build_fake_rewriter(cfg: Config, **backend_kwargs) -> SiraRewriter:
    return SiraRewriter(cfg, FakeScorer(), FakeBackend(**backend_kwargs), DifflibSim())


class FakeJudgeBackend:
    """Judge stand-in: texts or rewrites containing 'alt' (the fake infiller's fills) get low scores."""

    def generate(self, batch, max_new_tokens, temperature, top_p):
        outs = []
        for messages in batch:
            u = messages[-1]["content"]
            if "<rewrite>" in u:
                outs.append("2" if "alt" in u.split("<rewrite>")[1] else "5")
            else:
                outs.append("3" if "alt" in u else "5")
        return outs

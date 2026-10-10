"""Split an article into paragraph-sized chunks and decide which ones are safe to rewrite."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .config import ChunkCfg

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u201c(\[])")


@dataclass
class Chunk:
    index: int
    para_index: int
    text: str
    rewrite: bool
    skip_reason: Optional[str] = None


def count_words(text: str) -> int:
    return len(text.split())


def skip_reason(par: str, cfg: ChunkCfg) -> Optional[str]:
    stripped = par.strip()
    words = stripped.split()
    if re.match(r"^\[\d+\]", stripped) or "doi.org" in stripped.lower():
        return "reference"
    if re.match(r"^(Figure|Fig\.|Table)\s*\d+", stripped):
        return "caption"
    if stripped.startswith("#"):
        return "heading"
    if cfg.skip_headings and "\n" not in stripped and len(words) <= 12 and not re.search(r"[.!?;,]$", stripped):
        return "heading"
    if len(words) < cfg.min_words:
        return "too short"
    letters = sum(c.isalpha() for c in stripped)
    nonspace = sum(not c.isspace() for c in stripped)
    if nonspace and letters / nonspace < cfg.min_alpha_ratio:
        return "non-prose"
    return None


def split_long(par: str, max_words: int) -> List[str]:
    if count_words(par) <= max_words:
        return [par]
    pieces, cur, cur_words = [], [], 0
    for sent in _SENT_SPLIT.split(par):
        w = count_words(sent)
        if cur and cur_words + w > max_words:
            pieces.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(sent)
        cur_words += w
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def make_chunks(text: str, cfg: ChunkCfg) -> List[Chunk]:
    chunks: List[Chunk] = []
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    for pi, par in enumerate(paras):
        reason = skip_reason(par, cfg)
        if reason:
            chunks.append(Chunk(len(chunks), pi, par, False, reason))
            continue
        for piece in split_long(par, cfg.max_words):
            chunks.append(Chunk(len(chunks), pi, piece, True))
    return chunks


def join_chunks(chunks: List[Chunk], texts: List[str]) -> str:
    paras: List[List[str]] = []
    last = None
    for ch, t in zip(chunks, texts):
        if ch.para_index != last:
            paras.append([])
            last = ch.para_index
        paras[-1].append(t)
    return "\n\n".join(" ".join(p) for p in paras)

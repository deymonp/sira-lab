"""Turn per-token self-information into word-level masks.

Key idea (SIRA): words whose tokens have HIGH self-information (low probability in
context) are the high-entropy positions where text watermarks tend to live. We mask
those words and let a different LLM fill them in.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import MaskCfg

MASK_RE = re.compile(r"<mask_(\d+)>")
WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019\-][A-Za-z]+)*")

_PROTECT_PATTERNS = [
    r"\[\d+(?:\s*[,\-\u2013]\s*\d+)*\]",                       # [12], [3-5], [1, 2]
    r"\([^()]*\b(?:19|20)\d{2}[a-z]?\b[^()]*\)",                # (Smith et al., 2024)
    r"\$[^$\n]+\$",                                            # inline math
    r"`[^`\n]+`",                                              # inline code
    r"https?://\S+",
    r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    r"\b(?:Fig(?:ure)?s?|Tables?|Eq(?:uation)?s?|Sections?|Appendix)\.?\s*[A-Z]?\d+[A-Za-z]?\b",
]
_QUOTE_PATTERNS = [r"\"[^\"\n]+\"", r"\u201c[^\u201d\n]+\u201d"]
_BAD_NEIGHBORS = set("0123456789_^\\=<>$/@#%&*+~-")

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how
however i if in into is it its itself just may me might more most must my no nor not now of off on once only or other
our out over own same shall she should so some such than that the their them then there these they this those through
thus to too under until up upon very was we were what when where whether which while who whom why will with would you
your
""".split())


@dataclass
class Word:
    idx: int
    start: int
    end: int
    text: str
    candidate: bool = False
    score: float = 0.0


@dataclass
class MaskInfo:
    number: int
    start: int
    end: int
    original: str
    score: float


@dataclass
class MaskResult:
    template: str
    masks: List[MaskInfo]
    n_words: int
    n_candidates: int


def protected_ranges(text: str, cfg: MaskCfg) -> List[Tuple[int, int]]:
    patterns = list(_PROTECT_PATTERNS)
    if cfg.protect_quotes:
        patterns += _QUOTE_PATTERNS
    ranges: List[Tuple[int, int]] = []
    for p in patterns:
        ranges += [m.span() for m in re.finditer(p, text)]
    ranges.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in ranges:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _in_ranges(ranges: Sequence[Tuple[int, int]], starts: Sequence[int], s: int, e: int) -> bool:
    i = bisect.bisect_right(starts, s) - 1
    if i >= 0 and ranges[i][1] > s:
        return True
    j = i + 1
    return j < len(ranges) and ranges[j][0] < e


def _at_sentence_start(text: str, start: int) -> bool:
    i = start - 1
    while i >= 0 and text[i] in " \t\"'\u201c\u2018([":
        i -= 1
    return i < 0 or text[i] in ".?!:\n"


def find_words(text: str, cfg: MaskCfg) -> List[Word]:
    ranges = protected_ranges(text, cfg)
    starts = [r[0] for r in ranges]
    words: List[Word] = []
    for i, m in enumerate(WORD_RE.finditer(text)):
        s, e = m.span()
        w = Word(idx=i, start=s, end=e, text=m.group(0))
        ok = len(w.text) >= 3 and w.text.lower() not in STOPWORDS
        if ok and _in_ranges(ranges, starts, s, e):
            ok = False
        if ok:
            before = text[s - 1] if s > 0 else " "
            after = text[e] if e < len(text) else " "
            if before in _BAD_NEIGHBORS or after in _BAD_NEIGHBORS:
                ok = False
        if ok and cfg.protect_acronyms:
            if (w.text.isupper() and len(w.text) >= 2) or (not w.text.isupper() and any(c.isupper() for c in w.text[1:])):
                ok = False
        if ok and cfg.protect_capitalized and w.text[0].isupper() and not _at_sentence_start(text, s):
            ok = False
        w.candidate = ok
        words.append(w)
    return words


def score_words(words: List[Word], spans: Sequence[Tuple[int, int]], surprisal: Sequence[float], agg: str = "max") -> None:
    j, n = 0, len(spans)
    for w in words:
        while j < n and spans[j][1] <= w.start:
            j += 1
        k, vals = j, []
        while k < n and spans[k][0] < w.end:
            vals.append(surprisal[k])
            k += 1
        if not vals:
            w.score = 0.0
        elif agg == "mean":
            w.score = sum(vals) / len(vals)
        elif agg == "sum":
            w.score = float(sum(vals))
        else:
            w.score = float(max(vals))


def select_masks(words: List[Word], cfg: MaskCfg, ratio: Optional[float] = None) -> List[Word]:
    ratio = cfg.mask_ratio if ratio is None else ratio
    target = int(round(ratio * len(words)))
    cands = sorted((w for w in words if w.candidate and w.score >= cfg.min_surprisal), key=lambda w: -w.score)
    chosen: List[Word] = []
    taken: List[int] = []
    for w in cands:
        if len(chosen) >= target:
            break
        if cfg.min_gap > 0 and any(abs(w.idx - t) <= cfg.min_gap for t in taken):
            continue
        chosen.append(w)
        taken.append(w.idx)
    return sorted(chosen, key=lambda w: w.start)


def make_template(text: str, chosen: List[Word], words: List[Word]) -> MaskResult:
    parts, masks, pos = [], [], 0
    for n, w in enumerate(chosen, start=1):
        parts.append(text[pos:w.start])
        parts.append(f"<mask_{n}>")
        masks.append(MaskInfo(n, w.start, w.end, w.text, w.score))
        pos = w.end
    parts.append(text[pos:])
    return MaskResult("".join(parts), masks, len(words), sum(w.candidate for w in words))


def heat_segments(text: str, words: List[Word], chosen: List[Word]) -> List[Tuple[str, Optional[str]]]:
    chosen_ids = {w.idx for w in chosen}
    scores = sorted(w.score for w in words if w.candidate)
    hi = scores[int(0.75 * (len(scores) - 1))] if scores else float("inf")
    segs: List[Tuple[str, Optional[str]]] = []
    pos = 0
    for w in words:
        if w.start > pos:
            segs.append((text[pos:w.start], None))
        label = "masked" if w.idx in chosen_ids else ("high" if w.candidate and w.score >= hi else None)
        segs.append((w.text, label))
        pos = w.end
    if pos < len(text):
        segs.append((text[pos:], None))
    return segs

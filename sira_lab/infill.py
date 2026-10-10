"""Prompting and parsing for the mask-filling LLM. Pure Python (no ML imports)."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .masker import MASK_RE

SYSTEM = "You are a meticulous academic copy editor."

USER_TMPL = """Below is a passage from a research article in which some words were replaced by placeholders of the form <mask_N>.
Replace every placeholder with a word or short phrase (1 to 3 words) that fits naturally, so the passage stays fluent and keeps its original meaning.

Rules:
- Do not change any text outside the placeholders.
- Do not add or remove sentences.
- Return ONLY the completed passage between <passage> and </passage> tags.

<passage>
{template}
</passage>"""

_EX_IN = "The results <mask_1> that larger models <mask_2> better calibration, although the <mask_3> is small."
_EX_OUT = "<passage>\nThe results suggest that larger models achieve better calibration, although the effect is small.\n</passage>"


def build_messages(template: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(template=_EX_IN)},
        {"role": "assistant", "content": _EX_OUT},
        {"role": "user", "content": USER_TMPL.format(template=template)},
    ]


def extract_passage(raw: str) -> str:
    m = re.search(r"<passage>(.*?)</passage>", raw, flags=re.S)
    if m:
        return m.group(1).strip()
    if "<passage>" in raw:
        return raw.split("<passage>", 1)[1].strip()
    return raw.strip()


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def extract_fills(template: str, passage: str) -> Optional[List[str]]:
    """Align the model output with the template; return the fill for each mask, or None
    if any text outside the masks was altered."""
    T, O = _norm_ws(template), _norm_ws(passage)
    parts = MASK_RE.split(T)
    segs = parts[0::2]
    n = len(segs) - 1
    if n <= 0 or not O.startswith(segs[0]):
        return None
    pos, fills = len(segs[0]), []
    for i in range(1, n + 1):
        seg = segs[i]
        if i == n:
            if not O.endswith(seg):
                return None
            end = len(O) - len(seg)
            if end <= pos:
                return None
            fills.append(O[pos:end].strip())
        else:
            if not seg:
                return None
            j = O.find(seg, pos + 1)
            if j < 0:
                return None
            fills.append(O[pos:j].strip())
            pos = j + len(seg)
    return fills


def valid_fill(f: str) -> bool:
    return bool(f) and 1 <= len(f.split()) <= 4 and len(f) <= 40 and "<" not in f and ">" not in f


def fill_template(template: str, fills: List[str]) -> str:
    return MASK_RE.sub(lambda m: fills[int(m.group(1)) - 1], template)


def lenient_accept(template: str, passage: str) -> Optional[str]:
    """Accept a rewrite that touched text outside the masks, if it still contains
    at least 80% of the unmasked words in order and has a plausible length."""
    if "<mask_" in passage:
        return None
    n_masks = len(MASK_RE.findall(template))
    base = _norm_ws(MASK_RE.sub(" ", template)).split()
    out = passage.split()
    if not base or not out:
        return None
    if abs(len(out) - (len(base) + n_masks)) > 0.25 * len(base) + 2 * n_masks:
        return None
    matched = sum(b.size for b in SequenceMatcher(None, base, out, autojunk=False).get_matching_blocks())
    if matched / len(base) < 0.8:
        return None
    return passage.strip()


def parse_candidate(template: str, originals: List[str], raw: str, strict: bool = True):
    """Return (text, swaps) or None. swaps = [(original_word, fill), ...]."""
    passage = extract_passage(raw)
    fills = extract_fills(template, passage)
    if fills is not None and all(valid_fill(f) for f in fills):
        return fill_template(template, fills), list(zip(originals, fills))
    if not strict:
        text = lenient_accept(template, passage)
        if text:
            return text, []
    return None

"""Prompting and parsing for the mask-filling LLM. Pure Python (no ML imports)."""
from __future__ import annotations

import math
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


LIST_USER_TMPL = """Below is a passage from a research article in which some words were replaced by numbered placeholders <mask_N>.
For each placeholder, give ONE word or short phrase (1 to 3 words) that fits naturally and keeps the original meaning.

Rules:
- Output exactly one line per placeholder, in order, in the form `N: replacement`.
- Output nothing else: no explanations and no repeated text.

<passage>
{template}
</passage>"""

_LIST_EX_OUT = "1: suggest\n2: achieve\n3: effect"

_HYGIENE_RULES = ("- The replacement must not repeat the word directly before or directly after the placeholder.\n"
                  "- Use plain words only: no brackets, parentheses or other punctuation.\n")


def _list_template(hygiene: bool) -> str:
    if not hygiene:
        return LIST_USER_TMPL
    return LIST_USER_TMPL.replace("- Output nothing else", _HYGIENE_RULES + "- Output nothing else")


def build_messages(template: str, mode: str = "passage", hygiene: bool = False) -> List[Dict[str, str]]:
    if mode == "list":
        t = _list_template(hygiene)
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": t.format(template=_EX_IN)},
            {"role": "assistant", "content": _LIST_EX_OUT},
            {"role": "user", "content": t.format(template=template)},
        ]
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


_PLAIN_WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9'\-]*$")
_AN_EXCEPTIONS = ("uni", "use", "usu", "eu", "one", "once", "ubi")
_ART = re.compile(r"(?i)\b(a|an)\s+$")


def valid_fill(f: str, strict: bool = False) -> bool:
    ok = bool(f) and 1 <= len(f.split()) <= 4 and len(f) <= 40 and "<" not in f and ">" not in f
    if ok and strict:                         # hygiene: plain words only (no brackets, parentheses, odd punctuation)
        ok = all(_PLAIN_WORD.match(w) for w in f.split())
    return ok


def _fix_article(prefix: str, fill: str) -> str:
    """If the text just before a fill ends in 'a'/'an', make the article agree with the fill's first letter."""
    m, words = _ART.search(prefix), fill.split()
    if not m or not words:
        return prefix
    first = words[0].lower()
    new = "an" if (first[0] in "aeiou" and not first.startswith(_AN_EXCEPTIONS)) else "a"
    if m.group(1).lower() == new:
        return prefix
    if m.group(1)[0].isupper():
        new = new.capitalize()
    return prefix[:m.start(1)] + new + prefix[m.end(1):]


def fill_template(template: str, fills: List[str], fix_articles: bool = False) -> str:
    parts = MASK_RE.split(template)               # [seg0, n1, seg1, n2, seg2, ...]
    out = parts[0]
    for i in range(1, len(parts), 2):
        fill = fills[int(parts[i]) - 1]
        if fix_articles:
            out = _fix_article(out, fill)
        out += fill + parts[i + 1]
    return out


def neighbor_words(template: str) -> List[tuple]:
    """For every mask: (word directly before it, word directly after it), lower-case, or None when punctuation intervenes."""
    segs = MASK_RE.split(template)[0::2]
    out = []
    for i in range(len(segs) - 1):
        left = re.search(r"([A-Za-z0-9'\-]+)\s+$", segs[i])
        right = re.match(r"\s+([A-Za-z0-9'\-]+)", segs[i + 1])
        out.append((left.group(1).lower() if left else None, right.group(1).lower() if right else None))
    return out


def dedupe_fill(fill: str, original: str, left: Optional[str], right: Optional[str]) -> str:
    """Remove words of the fill that merely repeat the neighbouring word ('lengths lengths'); keep the original if nothing is left."""
    words = fill.split()
    if left and words and words[0].lower().strip("'-") == left.strip("'-"):
        words = words[1:]
    if right and words and words[-1].lower().strip("'-") == right.strip("'-"):
        words = words[:-1]
    return " ".join(words) if words else original


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


# ---------------------------------------------------------------- list mode
_LIST_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:<?mask_)?(\d+)>?\s*[:.)\]\-]\s*(.+?)\s*$")


def _clean_fill(fill: str, original: str) -> str:
    f = fill.strip().strip(" `\"'\u201c\u201d\u2018\u2019.,;:").strip()
    if not f:
        return f
    if original[:1].isupper() and f[0].islower():
        f = f[0].upper() + f[1:]
    elif original[:1].islower() and f[0].isupper() and not f.isupper():
        f = f[0].lower() + f[1:]
    return f


def parse_list_output(raw: str) -> Dict[int, str]:
    fills: Dict[int, str] = {}
    for line in raw.splitlines():
        m = _LIST_LINE.match(line)
        if m and int(m.group(1)) not in fills:
            fills[int(m.group(1))] = m.group(2)
    return fills


def parse_list_candidate(template: str, originals: List[str], raw: str, min_frac: float = 0.5, hygiene: bool = False):
    """Return ((text, swaps) | None, reason). Missing fills keep the original word.
    hygiene=True: plain-word fills only, neighbour repeats removed, a/an agreement fixed."""
    n = len(originals)
    parsed = parse_list_output(raw)
    nb = neighbor_words(template) if hygiene else None
    good = {}
    for i, f in parsed.items():
        if 1 <= i <= n:
            f = _clean_fill(f, originals[i - 1])
            if hygiene and valid_fill(f, strict=True):
                f = _clean_fill(dedupe_fill(f, originals[i - 1], *nb[i - 1]), originals[i - 1])
            if valid_fill(f, strict=hygiene):
                good[i] = f
    if len(good) < max(1, math.ceil(min_frac * n)):
        return None, f"only {len(good)} of {n} usable fills in the model output"
    full = [good.get(i + 1, originals[i]) for i in range(n)]
    return (fill_template(template, full, fix_articles=hygiene), list(zip(originals, full))), ""


def parse_output(mode: str, template: str, originals: List[str], raw: str, strict: bool = True, hygiene: bool = False):
    """Dispatch on mode. Returns ((text, swaps) | None, reason)."""
    if mode == "list":
        return parse_list_candidate(template, originals, raw, hygiene=hygiene)
    cand = parse_candidate(template, originals, raw, strict)
    return cand, ("" if cand else "output did not match the template (text outside the masks changed, wrong format, or a fill was too long)")

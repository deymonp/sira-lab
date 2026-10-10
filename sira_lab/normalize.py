"""Layer A: deterministic text hygiene. No ML, no meaning changes."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Tuple

_INVISIBLE = {
    0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x200E, 0x200F,
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069,
}
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
_SPACE_LIKE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
# Cyrillic look-alikes that sometimes sneak into English words
_HOMOGLYPHS = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0443": "y", "\u0445": "x", "\u0456": "i",
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
    "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T", "\u0425": "X",
}


def normalize(text: str) -> Tuple[str, Dict[str, int]]:
    stats = {"invisible_removed": 0, "spaces_normalized": 0, "homoglyphs_fixed": 0, "ligatures_expanded": 0}
    text = unicodedata.normalize("NFC", text)  # NFC (not NFKC) so superscripts etc. survive
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    out = []
    for ch in text:
        if ord(ch) in _INVISIBLE:
            stats["invisible_removed"] += 1
        elif ch in _LIGATURES:
            out.append(_LIGATURES[ch])
            stats["ligatures_expanded"] += 1
        else:
            out.append(ch)
    text = "".join(out)

    text, n = _SPACE_LIKE.subn(" ", text)
    stats["spaces_normalized"] = n

    def _fix_word(m: "re.Match[str]") -> str:
        w = m.group(0)
        if any(c in _HOMOGLYPHS for c in w) and any(c.isascii() and c.isalpha() for c in w):
            stats["homoglyphs_fixed"] += sum(c in _HOMOGLYPHS for c in w)
            return "".join(_HOMOGLYPHS.get(c, c) for c in w)
        return w

    text = re.sub(r"\S+", _fix_word, text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), stats

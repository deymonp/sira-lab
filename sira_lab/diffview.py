"""Word-level diff for the UI."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Optional, Tuple


def word_diff(a: str, b: str) -> List[Tuple[str, Optional[str]]]:
    wa, wb = a.split(), b.split()
    segs: List[Tuple[str, Optional[str]]] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        if op == "equal":
            segs.append((" ".join(wa[i1:i2]) + " ", None))
        else:
            if i2 > i1:
                segs.append((" ".join(wa[i1:i2]) + " ", "removed"))
            if j2 > j1:
                segs.append((" ".join(wb[j1:j2]) + " ", "added"))
    return segs

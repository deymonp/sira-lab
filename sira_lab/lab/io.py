"""Tiny JSONL helpers. Every Lab stage writes JSONL incrementally, so a dropped Colab session loses nothing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


def read_jsonl(path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

"""LLM quality judge: fluency of each text and meaning preservation of each rewrite (scores 1 to 5).

Caveats to quote with the numbers: the judge is a 7B open model (noisy; it is of the same family as the generator
and the rewriter, so it may favour that family's style). It is a much better check than embedding similarity,
but it is not a human reader. Human texts and unattacked machine texts are judged too, as reference points."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional

from ..infill import SYSTEM  # noqa: F401  (kept for parity with the rewriter prompts)
from .attacks import attack_filename
from .io import append_jsonl, read_jsonl

JUDGE_SYSTEM = "You are a strict, careful English language editor."

FLUENCY_USER = """Rate the fluency and grammatical correctness of the following passage from a research article on a scale from 1 to 5.
1 = broken or ungrammatical, 3 = understandable but clearly awkward, 5 = fully fluent and natural.
The passage may begin or end in the middle of a sentence; ignore that.
Answer with a single digit and nothing else.

<passage>
{text}
</passage>"""

MEANING_USER = """Below are an original passage and a rewritten version of it. Rate how well the rewrite preserves the meaning and the technical content of the original on a scale from 1 to 5.
1 = meaning lost or changed, 3 = partly preserved, 5 = the same meaning.
Answer with a single digit and nothing else.

<original>
{orig}
</original>

<rewrite>
{new}
</rewrite>"""


def parse_digit(raw: str) -> Optional[int]:
    m = re.search(r"[1-5]", raw or "")
    return int(m.group()) if m else None


def _ask(backend, prompts: List[str]) -> List[Optional[int]]:
    msgs = [[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": p}] for p in prompts]
    return [parse_digit(o) for o in backend.generate(msgs, 4, 0.0, 1.0)]


def run_judge(cfg, run_dir, backend=None, attacks=None, log: Callable[[str], None] = print) -> None:
    """Writes quality.jsonl (resumable) and quality.csv in the run folder."""
    run_dir = Path(run_dir)
    if backend is None:
        from ..backends import HFBackend
        backend = HFBackend(cfg.models.infiller, quantize_4bit=cfg.models.quantize_4bit, dtype=cfg.models.dtype,
                            device_map=cfg.models.device_map, batch_size=cfg.lab.judge_batch_size)
    path = run_dir / "quality.jsonl"
    done = {(r["kind"], r["algo"], r["id"]) for r in read_jsonl(path)}
    chunk = max(cfg.lab.judge_batch_size * 4, 8)

    def process(kind: str, algo: str, recs: List[dict], with_meaning: bool) -> None:
        todo = [r for r in recs if (kind, algo, r["id"]) not in done and r["text"].strip()]
        for i in range(0, len(todo), chunk):
            b = todo[i:i + chunk]
            flu = _ask(backend, [FLUENCY_USER.format(text=r["text"]) for r in b])
            mea = _ask(backend, [MEANING_USER.format(orig=r["orig_text"], new=r["text"]) for r in b]) if with_meaning else [None] * len(b)
            append_jsonl(path, [{"kind": kind, "algo": algo, "id": r["id"], "fluency": f, "meaning": m}
                                for r, f, m in zip(b, flu, mea)])
        if todo:
            log(f"[judge] {kind}/{algo}: {len(todo)} texts")

    gen = read_jsonl(run_dir / "generated.jsonl")
    for kind in ("human", "unwatermarked"):                      # reference points: how fluent is untouched text?
        process(kind, "none", [r for r in gen if r["kind"] == kind], False)
    for algo in cfg.lab.algorithms:                              # the unattacked watermarked text
        process("none", algo, [r for r in gen if r["kind"] == "watermarked" and r["algo"] == algo], False)
    for name in attacks or cfg.lab.attacks:
        recs = read_jsonl(run_dir / attack_filename(name))
        for algo in cfg.lab.algorithms:
            process(name, algo, [r for r in recs if r["algo"] == algo], True)

    import pandas as pd
    pd.DataFrame(read_jsonl(path)).to_csv(run_dir / "quality.csv", index=False)
    log(f"[judge] wrote quality.csv in {run_dir}")

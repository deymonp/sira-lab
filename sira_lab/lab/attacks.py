"""Stage 2: apply attacks to watermarked texts and record fidelity metrics.

Attack names:
  sira | random | lowest        use the rewriter's default mask ratio
  sira@0.3                      mask ratio 0.3 (also random@0.3, lowest@0.3)
  sira@0.3x2                    mask ratio 0.3, two rewrite rounds
  paraphrase                    plain LLM paraphrase of the whole passage
"""
from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from ..chunking import count_words
from ..infill import SYSTEM, extract_passage
from .io import append_jsonl, read_jsonl
from .metrics import changed_share, edit_ratio

STRATEGY = {"sira": "surprisal", "random": "random", "lowest": "lowest"}
_ATTACK_RE = re.compile(r"^(sira|random|lowest)(?:@(\d*\.?\d+))?(?:x(\d+))?$")

PARAPHRASE_USER = """Paraphrase the following passage from a research article. Keep the meaning, technical terms, numbers and citations, but change the wording and sentence structure.
Return ONLY the paraphrase between <passage> and </passage> tags.

<passage>
{text}
</passage>"""


def parse_attack(name: str) -> dict:
    if name == "paraphrase":
        return {"kind": "paraphrase"}
    m = _ATTACK_RE.match(name)
    if not m:
        raise ValueError(f"unknown attack {name!r} (use sira, random, lowest, paraphrase, e.g. 'sira@0.3' or 'sira@0.3x2')")
    return {"kind": "rewrite", "strategy": STRATEGY[m.group(1)],
            "ratio": float(m.group(2)) if m.group(2) else None,
            "rounds": int(m.group(3)) if m.group(3) else None}


def attack_filename(name: str) -> str:
    return "attacked_" + name.replace("@", "_r") + ".jsonl"


def paraphrase_many(texts: List[str], backend, cfg) -> List[str]:
    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": PARAPHRASE_USER.format(text=t)}] for t in texts]
    max_new = min(900, int(1.8 * max(count_words(t) for t in texts)) + 100)
    raws = backend.generate(msgs, max_new, cfg.gen.temperature, cfg.gen.top_p)
    return [extract_passage(r) or t for r, t in zip(raws, texts)]


def _quality(rewriter, orig: str, out: str) -> Dict[str, float]:
    sim = 1.0 if out == orig else rewriter.sim_model.similarity(orig, out)
    ppl = math.exp(rewriter.scorer.mean_surprisal(out) - rewriter.scorer.mean_surprisal(orig)) if out != orig else 1.0
    return {"sim": float(sim), "ppl_ratio": float(ppl), "edit_ratio": float(edit_ratio(orig, out)),
            "changed": changed_share(orig, out)}


def run_attack_name(rewriter, name: str, texts: List[str]) -> List[dict]:
    """Return one dict per text: {text, status, warnings, words_masked, attempts, mask_ratio_used}."""
    cfg, spec = rewriter.cfg, parse_attack(name)
    if spec["kind"] == "rewrite":
        old = cfg.mask.strategy
        cfg.mask.strategy = spec["strategy"]
        rewriter._rng = random.Random(cfg.seed)
        try:
            reps = rewriter.rewrite_many(texts, mask_ratio=spec["ratio"], rounds=spec["rounds"])
        finally:
            cfg.mask.strategy = old
        res = []
        for rep in reps:
            if rep.words_changed == 0:
                status = "unchanged"
            elif any(c.status == "best-effort" for c in rep.chunks):
                status = "best-effort"
            else:
                status = "rewritten"
            ratios = [c.mask_ratio_used for c in rep.chunks if c.mask_ratio_used is not None]
            res.append({
                "text": rep.text_out, "status": status,
                "warnings": [w[:300] for c in rep.chunks for w in c.warnings][:4],
                "words_masked": rep.words_masked, "attempts": sum(c.attempts for c in rep.chunks),
                "mask_ratio_used": float(np.mean(ratios)) if ratios else None})
        return res
    outs = paraphrase_many(texts, rewriter.backend, cfg)
    return [{"text": o, "status": "unchanged" if o == t else "rewritten", "warnings": []} for o, t in zip(outs, texts)]


def run_attacks(cfg, run_dir, rewriter, attacks=None, limit=None, log: Callable[[str], None] = print) -> None:
    run_dir = Path(run_dir)
    gen = read_jsonl(run_dir / "generated.jsonl")
    wm = [r for r in gen if r["kind"] == "watermarked" and r["text"].strip()]
    if limit:
        wm = wm[:limit]
    bs = cfg.lab.attack_batch_size
    for name in attacks or cfg.lab.attacks:
        parse_attack(name)                                   # fail early on typos
        path = run_dir / attack_filename(name)
        done = {(r["algo"], r["id"]) for r in read_jsonl(path)}
        todo = [r for r in wm if (r["algo"], r["id"]) not in done]
        log(f"[attack:{name}] {len(todo)} texts to process ({len(done)} already done)")
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            results = run_attack_name(rewriter, name, [r["text"] for r in batch])
            rows = []
            for r, res in zip(batch, results):
                extra = {k: v for k, v in res.items() if k not in ("text", "status")}
                rows.append({"id": r["id"], "algo": r["algo"], "attack": name, "orig_text": r["text"],
                             "text": res["text"], "status": res["status"], **extra,
                             **_quality(rewriter, r["text"], res["text"])})
            append_jsonl(path, rows)
            log(f"[attack:{name}] {min(i + bs, len(todo))}/{len(todo)}")

"""Stage 2: apply attacks to watermarked texts and record fidelity metrics."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, Dict, List

from ..chunking import count_words
from ..infill import SYSTEM, extract_passage
from .io import append_jsonl, read_jsonl
from .metrics import edit_ratio

REWRITE_ATTACKS = {"sira": "surprisal", "random": "random", "lowest": "lowest"}

PARAPHRASE_USER = """Paraphrase the following passage from a research article. Keep the meaning, technical terms, numbers and citations, but change the wording and sentence structure.
Return ONLY the paraphrase between <passage> and </passage> tags.

<passage>
{text}
</passage>"""


def paraphrase_many(texts: List[str], backend, cfg) -> List[str]:
    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": PARAPHRASE_USER.format(text=t)}] for t in texts]
    max_new = min(900, int(1.8 * max(count_words(t) for t in texts)) + 100)
    raws = backend.generate(msgs, max_new, cfg.gen.temperature, cfg.gen.top_p)
    return [extract_passage(r) or t for r, t in zip(raws, texts)]


def _quality(rewriter, orig: str, out: str) -> Dict[str, float]:
    sim = 1.0 if out == orig else rewriter.sim_model.similarity(orig, out)
    ppl = math.exp(rewriter.scorer.mean_surprisal(out) - rewriter.scorer.mean_surprisal(orig)) if out != orig else 1.0
    return {"sim": float(sim), "ppl_ratio": float(ppl), "edit_ratio": float(edit_ratio(orig, out))}


def run_attack_name(rewriter, name: str, texts: List[str]) -> List[dict]:
    """Return one dict per text: {text, status}."""
    cfg = rewriter.cfg
    if name in REWRITE_ATTACKS:
        old = cfg.mask.strategy
        cfg.mask.strategy = REWRITE_ATTACKS[name]
        rewriter._rng = random.Random(cfg.seed)
        try:
            reps = rewriter.rewrite_many(texts)
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
            warns = [w[:300] for c in rep.chunks for w in c.warnings][:4]
            res.append({"text": rep.text_out, "status": status, "warnings": warns})
        return res
    if name == "paraphrase":
        outs = paraphrase_many(texts, rewriter.backend, cfg)
        return [{"text": o, "status": "unchanged" if o == t else "rewritten"} for o, t in zip(outs, texts)]
    raise ValueError(f"unknown attack {name!r}")


def run_attacks(cfg, run_dir, rewriter, attacks=None, limit=None, log: Callable[[str], None] = print) -> None:
    run_dir = Path(run_dir)
    gen = read_jsonl(run_dir / "generated.jsonl")
    wm = [r for r in gen if r["kind"] == "watermarked" and r["text"].strip()]
    if limit:
        wm = wm[:limit]
    bs = cfg.lab.attack_batch_size
    for name in attacks or cfg.lab.attacks:
        path = run_dir / f"attacked_{name}.jsonl"
        done = {(r["algo"], r["id"]) for r in read_jsonl(path)}
        todo = [r for r in wm if (r["algo"], r["id"]) not in done]
        log(f"[attack:{name}] {len(todo)} texts to process ({len(done)} already done)")
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            results = run_attack_name(rewriter, name, [r["text"] for r in batch])
            rows = []
            for r, res in zip(batch, results):
                rows.append({"id": r["id"], "algo": r["algo"], "attack": name, "orig_text": r["text"],
                             "text": res["text"], "status": res["status"], "warnings": res.get("warnings", []),
                             **_quality(rewriter, r["text"], res["text"])})
            append_jsonl(path, rows)
            log(f"[attack:{name}] {min(i + bs, len(todo))}/{len(todo)}")

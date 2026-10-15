"""Stage 2: apply attacks to watermarked texts and record fidelity metrics.

Attack names:
  sira | random | lowest        use the rewriter's default mask ratio
  sira@0.3                      mask ratio 0.3 (also random@0.3, lowest@0.3)
  sira@0.3x2                    mask ratio 0.3, two rewrite rounds
  paraphrase                    plain LLM paraphrase of the whole passage
  A+B                           chain: apply A, then B to A's output (e.g. paraphrase+random@0.15).
                                A finished attacked_<A>.jsonl in the run folder is re-used as the first stage.
  sira2 | random2 | lowest2     same strategies with v8 "fill hygiene" (plain-word fills, no repeats of neighbouring
                                words, a/an fixed) and no two masks next to each other
  NAME~k                        variant k of NAME: same recipe, different random seed (k = 1, 2, ...; no suffix = variant 0)
                                e.g. paraphrase+random2@0.15~2.  Used for best-of-n selection.
  bestN                         derived by select.select_best from variants; never run as an attack
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
_ATTACK_RE = re.compile(r"^(sira|random|lowest)(2)?(?:@(\d*\.?\d+))?(?:x(\d+))?$")
_DERIVED_RE = re.compile(r"^best\d+$")

PARAPHRASE_USER = """Paraphrase the following passage from a research article. Keep the meaning, technical terms, numbers and citations, but change the wording and sentence structure.
Return ONLY the paraphrase between <passage> and </passage> tags.

<passage>
{text}
</passage>"""


def parse_attack(name: str) -> dict:
    if _DERIVED_RE.match(name):
        return {"kind": "derived"}
    m = re.match(r"^(.*?)(?:~(\d+))?$", name)
    core, variant = m.group(1), int(m.group(2)) if m.group(2) else 0
    if "+" in core:
        parts = core.split("+")
        for part in parts:
            if "~" in part:
                raise ValueError(f"{name!r}: put the variant marker once, at the very end (e.g. 'paraphrase+random@0.15~2')")
            parse_attack(part)                              # validates every stage (and rejects empty ones)
        return {"kind": "chain", "parts": parts, "variant": variant}
    if core == "paraphrase":
        return {"kind": "paraphrase", "variant": variant}
    m = _ATTACK_RE.match(core)
    if not m:
        raise ValueError(f"unknown attack {name!r} (use sira, random, lowest, paraphrase, e.g. 'sira@0.3', 'sira2@0.3x2', "
                         "'paraphrase+random2@0.15~1')")
    return {"kind": "rewrite", "strategy": STRATEGY[m.group(1)], "hygiene": bool(m.group(2)),
            "ratio": float(m.group(3)) if m.group(3) else None,
            "rounds": int(m.group(4)) if m.group(4) else None, "variant": variant}


def attack_filename(name: str) -> str:
    return "attacked_" + name.replace("@", "_r").replace("+", "_plus_").replace("~", "_v") + ".jsonl"


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


def _cached_stage(run_dir, part: str, keys) -> List[str] | None:
    """Texts of an already finished single attack for exactly these (algo, id) keys, or None."""
    recs = {(r["algo"], r["id"]): r["text"] for r in read_jsonl(Path(run_dir) / attack_filename(part))}
    return [recs[k] for k in keys] if keys and all(k in recs for k in keys) else None


def _seed(rewriter, variant: int) -> None:
    """Variant 0 keeps the original behaviour; other variants get their own random streams."""
    cfg = rewriter.cfg
    rewriter._rng = random.Random(cfg.seed + variant)
    if variant:
        try:
            from transformers import set_seed
            set_seed(cfg.seed + 1000 * variant)
        except Exception:
            pass


def run_attack_name(rewriter, name: str, texts: List[str], keys=None, run_dir=None) -> List[dict]:
    """Return one dict per text: {text, status, warnings, words_masked, attempts, mask_ratio_used}.
    keys = [(algo, id), ...] and run_dir enable re-using a finished first stage of a chain."""
    cfg, spec = rewriter.cfg, parse_attack(name)
    if spec["kind"] == "derived":
        raise ValueError(f"{name!r} is derived by select.select_best and cannot be run as an attack")
    k = spec.get("variant", 0)
    if spec["kind"] == "chain":
        cur, last = list(texts), None
        for j, part in enumerate(spec["parts"]):
            cached = None
            if j == 0 and run_dir is not None:
                cached = _cached_stage(run_dir, part + (f"~{k}" if k else ""), keys)
            if cached is not None:
                last = [{"text": t, "status": "rewritten", "warnings": []} for t in cached]
            else:
                last = run_attack_name(rewriter, part + (f"~{k}" if k else ""), cur)
            cur = [r["text"] for r in last]
        return [{**r, "status": "rewritten" if (r["status"] == "unchanged" and r["text"] != t0) else r["status"]}
                for r, t0 in zip(last, texts)]
    if spec["kind"] == "rewrite":
        old = (cfg.mask.strategy, cfg.gen.hygiene, cfg.mask.min_gap)
        cfg.mask.strategy = spec["strategy"]
        if spec["hygiene"]:
            cfg.gen.hygiene, cfg.mask.min_gap = True, max(1, cfg.mask.min_gap)
        _seed(rewriter, k)
        try:
            reps = rewriter.rewrite_many(texts, mask_ratio=spec["ratio"], rounds=spec["rounds"])
        finally:
            cfg.mask.strategy, cfg.gen.hygiene, cfg.mask.min_gap = old
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
    _seed(rewriter, k)
    outs = paraphrase_many(texts, rewriter.backend, cfg)
    return [{"text": o, "status": "unchanged" if o == t else "rewritten", "warnings": []} for o, t in zip(outs, texts)]


def run_attacks(cfg, run_dir, rewriter, attacks=None, limit=None, log: Callable[[str], None] = print) -> None:
    run_dir = Path(run_dir)
    gen = read_jsonl(run_dir / "generated.jsonl")
    wm = [r for r in gen if r["kind"] == "watermarked" and r["text"].strip()]
    if limit:
        wm = wm[:limit]
    bs = cfg.lab.attack_batch_size
    names = list(cfg.lab.prep_attacks) + list(attacks or cfg.lab.attacks)       # prep attacks first: inputs for chains
    for name in names:
        if parse_attack(name)["kind"] == "derived":           # fail early on typos; skip names made by select_best
            continue
        path = run_dir / attack_filename(name)
        done = {(r["algo"], r["id"]) for r in read_jsonl(path)}
        todo = [r for r in wm if (r["algo"], r["id"]) not in done]
        log(f"[attack:{name}] {len(todo)} texts to process ({len(done)} already done)")
        for i in range(0, len(todo), bs):
            batch = todo[i:i + bs]
            results = run_attack_name(rewriter, name, [r["text"] for r in batch],
                                      keys=[(r["algo"], r["id"]) for r in batch], run_dir=run_dir)
            rows = []
            for r, res in zip(batch, results):
                extra = {k: v for k, v in res.items() if k not in ("text", "status")}
                rows.append({"id": r["id"], "algo": r["algo"], "attack": name, "variant": parse_attack(name).get("variant", 0),
                             "orig_text": r["text"],
                             "text": res["text"], "status": res["status"], **extra,
                             **_quality(rewriter, r["text"], res["text"])})
            append_jsonl(path, rows)
            log(f"[attack:{name}] {min(i + bs, len(todo))}/{len(todo)}")

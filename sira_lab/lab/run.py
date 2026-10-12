"""Stage runner + CLI.   python -m sira_lab.lab.run all --config configs/lab_quick.yaml --run /content/runs/quick"""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

from ..config import load_config
from .attacks import run_attacks
from .detectors import detector_factories, make_encoder
from .evaluate import evaluate, plot_dose_response, plot_tradeoff
from .generate import generate_texts, load_generator, spec_for
from .io import append_jsonl, read_jsonl
from .prompts import load_prompts


def run_generate(cfg, run_dir, log=print) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    path = run_dir / "generated.jsonl"
    existing = read_jsonl(path)
    have = {(r["kind"], r["algo"], r["id"]) for r in existing}

    prompts = load_prompts(cfg.lab, cfg.seed)
    jobs = [("unwatermarked", "none", None)] + [("watermarked", a, spec_for(cfg.lab, a)) for a in cfg.lab.algorithms]

    # human reference texts need no model
    rows = [{"kind": "human", "algo": "none", "id": p["id"], "prompt": p["prompt"], "text": p["human"]}
            for p in prompts if ("human", "none", p["id"]) not in have and p["human"]]
    append_jsonl(path, rows)

    model = tok = None
    try:
        for kind, algo, spec in jobs:
            todo = [p for p in prompts if (kind, algo, p["id"]) not in have]
            if not todo:
                continue
            if model is None:
                model, tok = load_generator(cfg.lab.generator, cfg.models.dtype)
            log(f"[generate] {kind}/{algo}: {len(todo)} texts")
            texts = generate_texts(model, tok, [p["prompt"] for p in todo], cfg.lab, spec, cfg.seed)
            append_jsonl(path, [{"kind": kind, "algo": algo, "id": p["id"], "prompt": p["prompt"], "text": t}
                                for p, t in zip(todo, texts)])
    finally:
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    log(f"[generate] done -> {path}")


def run_attack(cfg, run_dir, rewriter=None, attacks=None, limit=None, log=print) -> None:
    if rewriter is None:
        from ..factory import build_rewriter
        rewriter = build_rewriter(cfg)
    run_attacks(cfg, run_dir, rewriter, attacks, limit, log)


def load_encoder(cfg):
    return make_encoder(cfg.lab.generator)


def run_evaluate(cfg, run_dir, detectors=None, log=print):
    detectors = detectors if detectors is not None else detector_factories(cfg)
    df = evaluate(cfg, run_dir, detectors)
    plot_tradeoff(df, Path(run_dir) / "tradeoff.png")
    plot_dose_response(df, Path(run_dir) / "dose_response.png")
    try:
        from .analysis import load_scores, paired_table
        paired_table(load_scores(run_dir)).to_csv(Path(run_dir) / "paired.csv", index=False)
    except Exception as e:                      # the main results must survive an analysis problem
        log(f"[evaluate] paired analysis skipped: {type(e).__name__}: {e}")
    log(f"[evaluate] wrote results.csv, results.md, scores.csv, paired.csv, tradeoff.png, dose_response.png in {run_dir}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "attack", "evaluate", "all"])
    ap.add_argument("--config", default="configs/lab_quick.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.stage in ("generate", "all"):
        run_generate(cfg, a.run)
    if a.stage in ("attack", "all"):
        run_attack(cfg, a.run, limit=a.limit)
    if a.stage in ("evaluate", "all"):
        print(run_evaluate(cfg, a.run).to_string())


if __name__ == "__main__":
    main()

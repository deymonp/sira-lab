"""Stage 3: run the detector panel -> results table and plot. Detectors are loaded, used and freed one at a time."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List

import numpy as np

from .io import read_jsonl
from .metrics import auroc, tpr_at_fpr

_IDENT = {"sim": 1.0, "edit_ratio": 1.0, "ppl_ratio": 1.0, "status": "none"}


def _row(det, algo: str, attack: str, pos: np.ndarray, neg: np.ndarray, neg_set: str, fpr: float, recs: List[dict]) -> Dict:
    def agg(fn, key):
        v = [r[key] for r in recs if r.get(key) is not None]
        return float(fn(v)) if v else float("nan")

    thr = getattr(det, "threshold", None)
    n = len(recs) or 1
    return {
        "detector": det.name, "algo": algo, "attack": attack, "n": len(pos), "n_neg": len(neg), "neg_set": neg_set,
        "TPR@FPR": tpr_at_fpr(pos, neg, fpr) if len(pos) and len(neg) else float("nan"),
        "AUROC": auroc(pos, neg) if len(pos) and len(neg) else float("nan"),
        "mean_score": float(np.mean(pos)) if len(pos) else float("nan"),
        "TPR@thr": float(np.mean(pos >= thr)) if (thr is not None and len(pos)) else float("nan"),
        "FPR@thr": float(np.mean(neg >= thr)) if (thr is not None and len(neg)) else float("nan"),
        "sim_median": agg(np.median, "sim"), "edit_mean": agg(np.mean, "edit_ratio"),
        "ppl_ratio_mean": agg(np.mean, "ppl_ratio"),
        "frac_unchanged": sum(r.get("status") == "unchanged" for r in recs) / n,
        "frac_best_effort": sum(r.get("status") == "best-effort" for r in recs) / n,
    }


def _rows_for(det, cfg, gen: List[dict], attacks: Dict[str, List[dict]]) -> List[Dict]:
    fpr = cfg.lab.fpr
    unwm = [r["text"] for r in gen if r["kind"] == "unwatermarked" and r["text"].strip()]
    human = [r["text"] for r in gen if r["kind"] == "human" and r["text"].strip()]
    rows: List[Dict] = []
    if det.kind == "watermark":
        algos, neg_set, neg_texts = [det.algo], "unwatermarked", unwm
    else:   # zero-shot: question is 'machine vs human', so negatives are human texts
        algos, neg_set, neg_texts = list(cfg.lab.algorithms), "human", human
        if not human:
            print(f"[evaluate] {det.name}: no human reference texts (builtin prompts) -> skipped")
            return rows
    neg = det.score(neg_texts) if neg_texts else np.array([])
    if det.kind == "zeroshot" and unwm:
        rows.append(_row(det, "-", "unwatermarked (AI baseline)", det.score(unwm), neg, neg_set, fpr, [dict(_IDENT)] * len(unwm)))
    for algo in algos:
        wm = [r["text"] for r in gen if r["kind"] == "watermarked" and r["algo"] == algo and r["text"].strip()]
        if not wm:
            continue
        rows.append(_row(det, algo, "none", det.score(wm), neg, neg_set, fpr, [dict(_IDENT)] * len(wm)))
        for attack, recs_all in attacks.items():
            recs = [r for r in recs_all if r["algo"] == algo]
            if recs:
                rows.append(_row(det, algo, attack, det.score([r["text"] for r in recs]), neg, neg_set, fpr, recs))
    return rows


def evaluate(cfg, run_dir, detectors: list):
    """`detectors`: detector instances and/or zero-argument factories (loaded and freed one at a time)."""
    import pandas as pd

    run_dir = Path(run_dir)
    gen = read_jsonl(run_dir / "generated.jsonl")
    attacks = {a: read_jsonl(run_dir / f"attacked_{a}.jsonl") for a in cfg.lab.attacks}
    rows: List[Dict] = []
    for item in detectors:
        det = item if hasattr(item, "score") else item()
        try:
            rows += _rows_for(det, cfg, gen, attacks)
        finally:
            del det
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "results.csv", index=False)
    (run_dir / "results.md").write_text(to_markdown(df), encoding="utf-8")
    return df


def to_markdown(df) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.tolist()]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def plot_tradeoff(df, path) -> None:
    """One panel per (detector, algorithm): detection power left (y, lower = better attack) vs meaning kept (x)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["sim_median", "TPR@FPR"])
    pairs = list(d[["detector", "algo"]].drop_duplicates().itertuples(index=False, name=None))
    if not pairs:
        return
    ncols = min(3, len(pairs))
    nrows = -(-len(pairs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    for ax, (det, algo) in zip(axes.flat, pairs):
        sub = d[(d["detector"] == det) & (d["algo"] == algo)]
        ax.scatter(sub["sim_median"], sub["TPR@FPR"], s=60)
        for _, r in sub.iterrows():
            ax.annotate(r["attack"], (r["sim_median"], r["TPR@FPR"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(f"{det} / {algo}", fontsize=10)
        ax.set_xlabel("median similarity to original (higher = meaning kept)", fontsize=8)
        ax.set_ylabel("TPR @ low FPR (lower = evaded)", fontsize=8)
        ax.set_xlim(0.4, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

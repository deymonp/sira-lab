"""Stage 3: run the detector panel -> results table, per-text scores and plots.
Detectors are loaded, used and freed one at a time."""
from __future__ import annotations

import gc
import hashlib
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from .attacks import attack_filename
from .io import read_jsonl
from .metrics import auroc, tpr_at_fpr

_IDENT = {"sim": 1.0, "edit_ratio": 1.0, "ppl_ratio": 1.0, "changed": 0.0, "status": "none"}


def _boot_ci(flags: np.ndarray, n_boot: int = 1000, seed: int = 0):
    flags = np.asarray(flags, float)
    if len(flags) == 0:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(flags), (n_boot, len(flags)))
    m = flags[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _row(det, algo: str, attack: str, pos: np.ndarray, neg: np.ndarray, neg_set: str, fpr: float, recs: List[dict]) -> Dict:
    def agg(fn, key):
        v = [r[key] for r in recs if r.get(key) is not None]
        return float(fn(v)) if v else float("nan")

    thr = getattr(det, "threshold", None)
    n = len(recs) or 1
    if thr is not None:
        flags = pos >= thr
    elif len(neg):
        flags = pos > np.quantile(neg, 1.0 - fpr)         # threshold that gives `fpr` false alarms on the negatives
    else:
        flags = np.array([], dtype=bool)
    lo, hi = _boot_ci(flags)
    return {
        "detector": det.name, "algo": algo, "attack": attack, "n": len(pos), "n_neg": len(neg), "neg_set": neg_set,
        "detect_rate": float(np.mean(flags)) if len(flags) else float("nan"), "rate_lo": lo, "rate_hi": hi,
        "changed_mean": agg(np.mean, "changed"),
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


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


class _Cached:
    """Wraps a detector (instance or tagged factory). Scores already in scores.csv are reused; the real detector
    (and its model) is only loaded if something is actually missing. A changed text invalidates its cached score."""

    def __init__(self, item, cache: dict):
        self.cache, self._item, self._det = cache, item, None
        if hasattr(item, "score"):
            self._det, meta = item, {"name": item.name, "kind": item.kind, "algo": getattr(item, "algo", None),
                                     "threshold": getattr(item, "threshold", None)}
        else:
            meta = getattr(item, "meta", None)
            if meta is None:
                self._det = item()
                meta = {"name": self._det.name, "kind": self._det.kind, "algo": getattr(self._det, "algo", None),
                        "threshold": getattr(self._det, "threshold", None)}
        self.name, self.kind, self.algo, self.threshold = meta["name"], meta["kind"], meta["algo"], meta["threshold"]

    def score(self, texts: List[str], keys: list) -> np.ndarray:
        hashes = [_hash(t) for t in texts]
        out = []
        for k, h in zip(keys, hashes):
            row = self.cache.get(k)
            th = row.get("text_hash") if row is not None else None
            usable = row is not None and (th is None or th != th or th == h)     # None/NaN = file from an older version: trust it
            out.append(float(row["score"]) if usable else None)
        miss = [i for i, v in enumerate(out) if v is None]
        if miss:
            if self._det is None:
                self._det = self._item()
            for i, v in zip(miss, self._det.score([texts[i] for i in miss])):
                out[i] = float(v)
        return np.array(out, dtype=float)


def _scored(det, recs: List[dict], algo: str, attack: str, sink: List[dict]) -> np.ndarray:
    keys = [(det.name, algo, attack, int(r["id"])) for r in recs]
    sc = det.score([r["text"] for r in recs], keys)
    for r, s in zip(recs, sc):
        sink.append({"detector": det.name, "algo": algo, "attack": attack, "id": int(r["id"]), "score": float(s),
                     "sim": r.get("sim"), "changed": r.get("changed"), "text_hash": _hash(r["text"])})
    return sc


def _rows_for(det, cfg, gen: List[dict], attacks: Dict[str, List[dict]], sink: List[dict]) -> List[Dict]:
    fpr = cfg.lab.fpr
    unwm = [r for r in gen if r["kind"] == "unwatermarked" and r["text"].strip()]
    human = [r for r in gen if r["kind"] == "human" and r["text"].strip()]
    rows: List[Dict] = []
    if det.kind == "watermark":
        algos, neg_set, neg_recs = [det.algo], "unwatermarked", unwm
    else:   # zero-shot: question is 'machine vs human', so negatives are human texts
        algos, neg_set, neg_recs = list(cfg.lab.algorithms), "human", human
        if not human:
            print(f"[evaluate] {det.name}: no human reference texts (builtin prompts) -> skipped")
            return rows
    neg = _scored(det, neg_recs, "none", f"NEG:{neg_set}", sink) if neg_recs else np.array([])
    ident = lambda rs: [{**r, **_IDENT} for r in rs]
    if det.kind == "zeroshot" and unwm:
        rows.append(_row(det, "-", "unwatermarked (AI baseline)", _scored(det, ident(unwm), "-", "unwatermarked (AI baseline)", sink),
                         neg, neg_set, fpr, ident(unwm)))
    for algo in algos:
        wm = [r for r in gen if r["kind"] == "watermarked" and r["algo"] == algo and r["text"].strip()]
        if not wm:
            continue
        rows.append(_row(det, algo, "none", _scored(det, ident(wm), algo, "none", sink), neg, neg_set, fpr, ident(wm)))
        for attack, recs_all in attacks.items():
            recs = [r for r in recs_all if r["algo"] == algo]
            if recs:
                rows.append(_row(det, algo, attack, _scored(det, recs, algo, attack, sink), neg, neg_set, fpr, recs))
    return rows


def _flush(path: Path, cache: dict, sink: List[dict]) -> None:
    import pandas as pd

    merged = dict(cache)
    for r in sink:
        merged[(r["detector"], r["algo"], r["attack"], r["id"])] = r
    pd.DataFrame(list(merged.values())).to_csv(path, index=False)


def evaluate(cfg, run_dir, detectors: list):
    """`detectors`: detector instances and/or tagged factories. Scores already stored in scores.csv are re-used
    (a detector whose scores are all stored is not even loaded), and scores.csv is saved after every detector, so
    an interrupted evaluation continues instead of starting over."""
    import pandas as pd

    run_dir = Path(run_dir)
    gen = read_jsonl(run_dir / "generated.jsonl")
    attacks = {a: read_jsonl(run_dir / attack_filename(a)) for a in cfg.lab.attacks}
    missing = [a for a, r in attacks.items() if not r]
    if missing:
        print(f"[evaluate] no results yet for: {missing} (skipped)")
    cache: dict = {}
    sp = run_dir / "scores.csv"
    if sp.exists():
        for r in pd.read_csv(sp).to_dict("records"):
            cache[(r["detector"], r["algo"], r["attack"], int(r["id"]))] = r
    rows: List[Dict] = []
    sink: List[dict] = []
    for item in detectors:
        det = _Cached(item, cache)
        try:
            rows += _rows_for(det, cfg, gen, attacks, sink)
            _flush(sp, cache, sink)
        finally:
            det._det = None
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


def _family(attack: str) -> str:
    m = re.match(r"^(sira|random|lowest)(?:@[\d.]+)?(x\d+)?$", attack)
    if m:
        return m.group(1) + (f" ({m.group(2)[1:]} rounds)" if m.group(2) else "")
    return attack


def plot_dose_response(df, path) -> None:
    """Detection rate (y, lower = evaded) against the share of words changed (x), one line per attack family.
    The question it answers: at the SAME edit level, does targeting (sira) beat random / lowest editing?"""
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[~df["attack"].str.startswith("unwatermarked")].dropna(subset=["changed_mean", "detect_rate"])
    pairs = list(d[["detector", "algo"]].drop_duplicates().itertuples(index=False, name=None))
    if not pairs:
        return
    ncols = min(3, len(pairs))
    nrows = -(-len(pairs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    for ax, (det, algo) in zip(axes.flat, pairs):
        sub = d[(d["detector"] == det) & (d["algo"] == algo)].copy()
        sub["family"] = sub["attack"].map(_family)
        base = sub[sub["attack"] == "none"]
        for fam, g in sub[sub["attack"] != "none"].groupby("family"):
            g = g.sort_values("changed_mean")
            g = g if base.empty else pd.concat([g, base]).sort_values("changed_mean")
            ax.errorbar(g["changed_mean"], g["detect_rate"],
                        yerr=[g["detect_rate"] - g["rate_lo"], g["rate_hi"] - g["detect_rate"]],
                        marker="o", capsize=2, label=fam, linewidth=1.2)
        ax.set_title(f"{det} / {algo}", fontsize=10)
        ax.set_xlabel("share of original words changed", fontsize=8)
        ax.set_ylabel("detection rate (lower = evaded)", fontsize=8)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


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

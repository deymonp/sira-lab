"""CPU-only analysis of a finished run: paired comparisons between attacks, with bootstrap confidence intervals.
Reads scores.csv, which evaluation writes (one detector score per text)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_scores(run_dir):
    import pandas as pd

    return pd.read_csv(Path(run_dir) / "scores.csv")


def auto_pairs(attacks: List[str]) -> List[Tuple[str, str]]:
    """Pair every sira attack with the random / lowest attack that has the same settings."""
    pairs = []
    for a in attacks:
        m = re.match(r"^sira(2?)((?:@[\d.]+)?(?:x\d+)?)$", a)
        if m:
            for other in ("random", "lowest"):
                if other + m.group(1) + m.group(2) in attacks:
                    pairs.append((a, other + m.group(1) + m.group(2)))
    return pairs


def paired_table(scores, pairs: Optional[List[Tuple[str, str]]] = None, n_boot: int = 2000, seed: int = 0):
    """For each detector/watermark and attack pair (a, b): mean per-text score difference a - b with a 95% CI.
    Scores are 'higher = more detectable', so a NEGATIVE difference means attack `a` hides the text better."""
    import pandas as pd

    s = scores[~scores["attack"].astype(str).str.startswith("NEG")]
    pairs = pairs or auto_pairs(sorted(s["attack"].unique()))
    rng = np.random.default_rng(seed)
    rows = []
    for (det, algo), g in s.groupby(["detector", "algo"]):
        for a, b in pairs:
            A, B = g[g["attack"] == a].set_index("id"), g[g["attack"] == b].set_index("id")
            ids = A.index.intersection(B.index)
            if len(ids) < 5:
                continue
            d = (A.loc[ids, "score"] - B.loc[ids, "score"]).to_numpy(float)
            boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            verdict = f"{a} hides better" if hi < 0 else (f"{b} hides better" if lo > 0 else "no clear difference")
            rows.append({
                "detector": det, "algo": algo, "a": a, "b": b, "n": len(ids),
                "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "frac_a_lower": float(np.mean(d < 0)),
                "changed_a": float(A.loc[ids, "changed"].mean()), "changed_b": float(B.loc[ids, "changed"].mean()),
                "sim_a": float(A.loc[ids, "sim"].mean()), "sim_b": float(B.loc[ids, "sim"].mean()),
                "verdict": verdict})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- success rate
def detector_thresholds(sc, z_threshold: float = 4.0, fpr: float = 0.01) -> dict:
    """Watermark detectors: fixed z threshold. Zero-shot detectors: the score that gives `fpr` false alarms on human text."""
    thr = {}
    for det, g in sc.groupby("detector"):
        if det.startswith("wm_"):
            thr[det] = z_threshold
        else:
            neg = g[g["attack"] == "NEG:human"]["score"].to_numpy(float)
            thr[det] = float(np.quantile(neg, 1 - fpr)) if len(neg) else float("nan")
    return thr


def is_evaded(detector: str, score: float, thr: dict) -> bool:
    return (score < thr[detector]) if detector.startswith("wm_") else (score <= thr[detector])


def success_table(run_dir, z_threshold: float = 4.0, fpr: float = 0.01, min_fluency: int = 4, min_meaning: int = 4,
                  min_sim: float = 0.85, n_boot: int = 1000, seed: int = 0, quality_file: str = "quality.csv",
                  log=print):
    """Per-text success: a rewritten text SUCCEEDS if it slips past EVERY detector in the panel AND is good quality.

    * Watermark detector: evaded if z < z_threshold. Zero-shot detector: evaded if its score is at or below the
      threshold that gives `fpr` false alarms on human text.
    * Quality: judge fluency >= min_fluency and meaning >= min_meaning from `quality_file` (quality.csv = judge A;
      quality_b.csv = the independent judge B) when it exists, otherwise embedding similarity >= min_sim (a weak proxy).
      Attacks that have no judge scores, or are missing some detector, are left out (and listed) instead of showing a
      misleading 0%. `quality_basis` says which quality measure was used.
    Columns evade_zs / evade_zs_ho: the original / the held-out (other model family) zero-shot detectors."""
    import pandas as pd

    run_dir = Path(run_dir)
    sc = load_scores(run_dir)
    qpath = run_dir / quality_file
    q = pd.read_csv(qpath) if qpath.exists() else None

    thr = detector_thresholds(sc, z_threshold, fpr)
    a = sc["attack"].astype(str)
    sc = sc[~a.str.startswith("NEG") & ~a.str.startswith("unwatermarked")].copy()
    sc = sc[[not np.isnan(thr[d]) for d in sc["detector"]]]
    sc["evaded"] = [is_evaded(d, s_, thr) for s_, d in zip(sc["score"], sc["detector"])]

    rows, rng, dropped = [], np.random.default_rng(seed), []
    for algo in sorted(sc["algo"].unique()):
        g = sc[sc["algo"] == algo]
        g = g[[(not d.startswith("wm_")) or d == f"wm_{algo}" for d in g["detector"]]]
        n_det = g.groupby("attack")["detector"].nunique()
        complete = set(n_det[n_det == n_det.max()].index)
        has_q = set(q[q["algo"] == algo]["kind"]) if q is not None else complete
        dropped += [(algo, x) for x in sorted(set(n_det.index) - (complete & has_q))]
        g = g[g["attack"].isin(complete & has_q)]
        if g.empty:
            continue
        piv = g.pivot_table(index=["attack", "id"], columns="detector", values="evaded", aggfunc="first").dropna(axis=1, how="all")
        zs = [c for c in piv.columns if not c.startswith("wm_") and not c.endswith("_ho")]
        zs_ho = [c for c in piv.columns if c.endswith("_ho")]
        out = pd.DataFrame(index=piv.index)
        out["evade_wm"] = piv[f"wm_{algo}"].astype(bool) if f"wm_{algo}" in piv else True
        if zs:
            out["evade_zs"] = piv[zs].astype(bool).all(axis=1)
        if zs_ho:
            out["evade_zs_ho"] = piv[zs_ho].astype(bool).all(axis=1)
        out["evade_all"] = piv.astype(bool).all(axis=1)
        out = out.reset_index()
        if q is not None:
            qq = q[q["algo"] == algo].rename(columns={"kind": "attack"})[["attack", "id", "fluency", "meaning"]]
            out = out.merge(qq, on=["attack", "id"], how="left")
            out.loc[out["attack"] == "none", "meaning"] = 5
            out["quality_ok"] = (out["fluency"] >= min_fluency) & (out["meaning"] >= min_meaning)
            basis = "judge" if quality_file == "quality.csv" else f"judge ({quality_file})"
        else:
            sim = g.drop_duplicates(["attack", "id"])[["attack", "id", "sim"]]
            out = out.merge(sim, on=["attack", "id"], how="left")
            out["quality_ok"] = out["sim"] >= min_sim
            basis = "similarity"
        out["success"] = out["evade_all"] & out["quality_ok"]
        for attack, d in out.groupby("attack", sort=False):
            flags = d["success"].to_numpy(float)
            boots = flags[rng.integers(0, len(flags), (n_boot, len(flags)))].mean(axis=1)
            row = {"algo": algo, "attack": attack, "n": len(d)}
            for c in ("evade_wm", "evade_zs", "evade_zs_ho", "evade_all", "quality_ok", "success"):
                if c in d:
                    row[c] = float(d[c].mean())
            row.update({"success_lo": float(np.percentile(boots, 2.5)), "success_hi": float(np.percentile(boots, 97.5)),
                        "quality_basis": basis})
            if q is not None:
                row["fluency_mean"], row["meaning_mean"] = float(d["fluency"].mean()), float(d["meaning"].mean())
            rows.append(row)
    if dropped:
        log("[success] left out (no judge scores or missing detectors): " + ", ".join(f"{a_}/{x}" for a_, x in dropped))
    return pd.DataFrame(rows)


def plot_success(df, path) -> None:
    """Horizontal bars: share of texts that beat every detector AND keep good quality, per attack."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = list(df["algo"].unique())
    fig, axes = plt.subplots(1, len(algos), figsize=(6.5 * len(algos), 0.45 * df["attack"].nunique() + 2.2), squeeze=False)
    for ax, algo in zip(axes[0], algos):
        d = df[df["algo"] == algo].sort_values("success")
        err = [d["success"] - d["success_lo"], d["success_hi"] - d["success"]]
        ax.barh(d["attack"], d["success"], xerr=err, capsize=2, color="#4c78a8")
        ax.set_xlim(0, 1)
        ax.set_title(f"{algo} texts: success rate", fontsize=10)
        ax.set_xlabel("share of texts that beat ALL detectors and keep good quality", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

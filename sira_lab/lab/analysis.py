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
        m = re.match(r"^sira((?:@[\d.]+)?(?:x\d+)?)$", a)
        if m:
            for other in ("random", "lowest"):
                if other + m.group(1) in attacks:
                    pairs.append((a, other + m.group(1)))
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

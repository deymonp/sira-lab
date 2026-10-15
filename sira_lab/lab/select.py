"""Best-of-n selection (adaptive attack with feedback).

For every watermarked text there are n variants of the same attack (names  BASE, BASE~1, BASE~2 ...). This step keeps, per text,
the variant that does best on the SELECTION panel:
  * the watermark detector and the two original zero-shot detectors (detectors whose name ends in _ho are held out), and
  * the judge-A quality scores (if use_judge).
Choice: fewest detectors that still flag the text, plus a penalty for failing the quality bar; ties go to higher judge scores.

HONEST EVALUATION: the held-out detectors never take part in the choice, so they measure whether the gain generalizes.
Judge A takes part in the choice, so grading the result with judge A is optimistic (winner's curse); use judge B
(judge.run_judge_b, another model family and wording) for the final quality grade.

It is CPU-only: it re-uses scores.csv and quality.csv. The chosen variant's score and quality rows are copied under the
new attack name, so evaluation finds them in its cache and loads no detector models."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .analysis import detector_thresholds, is_evaded
from .attacks import attack_filename
from .io import read_jsonl


def variant_names(base: str, n: int) -> List[str]:
    return [base] + [f"{base}~{k}" for k in range(1, n)]


def select_best(cfg, run_dir, base: str, n: int = 3, out_name: Optional[str] = None, use_judge: bool = True,
                log=print):
    import json

    import pandas as pd

    run_dir = Path(run_dir)
    out_name = out_name or f"best{n}"
    sc = pd.read_csv(run_dir / "scores.csv")
    sc = sc[sc["attack"] != out_name]                                    # a re-run replaces the old selection
    q = None
    if use_judge:
        q = pd.read_csv(run_dir / "quality.csv")
        q = q[q["kind"] != out_name]
    thr = detector_thresholds(sc, cfg.lab.z_threshold, cfg.lab.fpr)
    sel_dets = [d for d in sc["detector"].unique() if not d.endswith("_ho") and not np.isnan(thr[d])]
    S = {(r.detector, r.algo, r.attack, int(r.id)): r.score for r in sc.itertuples()}
    Q = {(r.kind, r.algo, int(r.id)): (r.fluency, r.meaning) for r in q.itertuples()} if q is not None else {}

    names = variant_names(base, n)
    recs = {nm: {(r["algo"], r["id"]): r for r in read_jsonl(run_dir / attack_filename(nm))} for nm in names}
    keys = sorted({k for nm in names for k in recs[nm]})
    min_f, min_m = cfg.lab.min_fluency, cfg.lab.min_meaning

    chosen, summary = [], []
    for algo, tid in keys:
        dets = [d for d in sel_dets if (not d.startswith("wm_")) or d == f"wm_{algo}"]
        cands = []
        for k, nm in enumerate(names):
            r = recs[nm].get((algo, tid))
            if r is None or any((d, algo, nm, tid) not in S for d in dets):
                continue
            flags = sum(not is_evaded(d, S[(d, algo, nm, tid)], thr) for d in dets)
            qs, qfail = 0.0, 0
            if use_judge:
                f, m = Q.get((nm, algo, tid), (np.nan, np.nan))
                if np.isnan(f) or np.isnan(m):
                    continue
                qs, qfail = f + m, int(not (f >= min_f and m >= min_m))
            cands.append((flags + 2 * qfail, -qs, k, nm, flags, qfail))
        if not cands:
            continue
        obj, _, k, nm, flags, qfail = min(cands)
        row = dict(recs[nm][(algo, tid)])
        row.update({"attack": out_name, "chosen_variant": k, "chosen_from": nm})
        chosen.append(row)
        summary.append({"algo": algo, "id": tid, "n_candidates": len(cands), "chosen_variant": k, "flags": flags,
                        "quality_fail": qfail, "any_perfect": any(c[0] == 0 for c in cands)})

    if not chosen:
        raise RuntimeError(f"no complete candidates found for {base!r}: run the attacks, evaluation and the judge first")
    (run_dir / attack_filename(out_name)).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")

    # copy the chosen variant's scores (all detectors, incl. held-out) and judge-A rows under the new name
    new_rows = []
    by_key = {(r["algo"], r["id"]): r["chosen_from"] for r in chosen}
    sub = sc[sc["attack"].isin(names)]
    for r in sub.to_dict("records"):
        if by_key.get((r["algo"], int(r["id"]))) == r["attack"]:
            new_rows.append({**r, "attack": out_name})
    pd.concat([sc, pd.DataFrame(new_rows)], ignore_index=True).to_csv(run_dir / "scores.csv", index=False)
    if q is not None:
        qrows = [{"kind": out_name, "algo": a_, "id": i_, "fluency": Q[(nm, a_, i_)][0], "meaning": Q[(nm, a_, i_)][1]}
                 for (a_, i_), nm in by_key.items()]
        pd.concat([q, pd.DataFrame(qrows)], ignore_index=True).to_csv(run_dir / "quality.csv", index=False)

    df = pd.DataFrame(summary)
    for algo, g in df.groupby("algo"):
        log(f"[select] {algo}: {len(g)} texts, mean candidates {g.n_candidates.mean():.1f}; "
            f"a variant passed ALL selection checks for {g.any_perfect.mean():.0%} of texts; "
            f"variant chosen: {g.chosen_variant.value_counts().sort_index().to_dict()}")
    log(f"[select] wrote {attack_filename(out_name)} ({len(chosen)} texts) and copied scores/quality rows as '{out_name}'")
    return df

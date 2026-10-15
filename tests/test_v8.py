import json

import numpy as np
import pandas as pd
import pytest

from sira_lab.config import load_config
from sira_lab.infill import (build_messages, dedupe_fill, fill_template, neighbor_words, parse_list_candidate,
                             valid_fill)
from sira_lab.lab.analysis import success_table
from sira_lab.lab.attacks import attack_filename, parse_attack, run_attacks
from sira_lab.lab.evaluate import evaluate
from sira_lab.lab.io import append_jsonl, read_jsonl
from sira_lab.lab.judge import run_judge, run_judge_b
from sira_lab.lab.select import select_best, variant_names
from sira_lab.testing import FakeJudgeBackend, build_fake_rewriter
from test_lab_pipeline import _make_run

T = "The orientation distribution of <mask_1> is determined by a <mask_2> process in the network."


# ------------------------------------------------------------------------------ Option A: fill hygiene
def test_strict_fills_reject_brackets_and_punctuation():
    assert valid_fill("(1D) modeling") and not valid_fill("(1D) modeling", strict=True)
    assert valid_fill("state-of-the-art", strict=True) and valid_fill("don't", strict=True)
    assert not valid_fill("a; b", strict=True) and not valid_fill("[x]", strict=True) and not valid_fill("", strict=True)


def test_neighbors_and_dedupe():
    assert neighbor_words(T) == [("of", "is"), ("a", "process")]
    assert neighbor_words("Results, <mask_1>. We <mask_2> it.") == [(None, None), ("we", "it")]     # punctuation blocks neighbours
    assert dedupe_fill("lengths", "orientations", "distribution", "is") == "lengths"
    assert dedupe_fill("distribution", "orientations", "distribution", "is") == "orientations"       # nothing left -> keep original
    assert dedupe_fill("stochastic process", "random", "a", "process") == "stochastic"
    assert dedupe_fill("the new", "old", "the", "model") == "new"


def test_article_agreement():
    assert fill_template("It is a <mask_1> result.", ["efficient"], fix_articles=True) == "It is an efficient result."
    assert fill_template("It is an <mask_1> result.", ["good"], fix_articles=True) == "It is a good result."
    assert fill_template("A <mask_1> idea.", ["unique"], fix_articles=True) == "A unique idea."
    assert fill_template("a <mask_1> study", ["university"], fix_articles=True) == "a university study"
    assert fill_template("A <mask_1> idea.", ["old"], fix_articles=True) == "An old idea."
    assert fill_template("It is a <mask_1> result.", ["efficient"]) == "It is a efficient result."   # legacy: untouched


def test_hygiene_changes_parsing_but_legacy_is_unchanged():
    raw = "1: (1D) modeling\n2: stochastic process"
    (legacy, _), _ = parse_list_candidate(T, ["shapes", "random"], raw)
    (clean, swaps), _ = parse_list_candidate(T, ["shapes", "random"], raw, hygiene=True)
    assert "(1D) modeling" in legacy and "stochastic process process" in legacy      # the v7 glitches
    assert "(1D)" not in clean and "process process" not in clean
    assert clean == "The orientation distribution of shapes is determined by a stochastic process in the network."
    assert dict(swaps)["shapes"] == "shapes"                                          # rejected fill -> original kept


def test_hygiene_prompt_has_the_extra_rules_only_when_asked():
    assert "must not repeat" in build_messages(T, "list", hygiene=True)[-1]["content"]
    assert "must not repeat" not in build_messages(T, "list")[-1]["content"]


# ------------------------------------------------------------------------------ variants, prep attacks, new families
def test_hygiene_family_restores_settings_and_variants_differ(tmp_path):
    cfg = _make_run(tmp_path, n=12)
    cfg.gate.enforce = False
    rw = build_fake_rewriter(cfg)
    before = (cfg.gen.hygiene, cfg.mask.min_gap, cfg.mask.strategy)
    run_attacks(cfg, tmp_path, rw, attacks=["random2@0.3", "random@0.3", "random@0.3~1"], log=lambda m: None)
    assert (cfg.gen.hygiene, cfg.mask.min_gap, cfg.mask.strategy) == before
    h = {(r["algo"], r["id"]): r["text"] for r in read_jsonl(tmp_path / attack_filename("random2@0.3"))}
    v0 = {(r["algo"], r["id"]): r["text"] for r in read_jsonl(tmp_path / attack_filename("random@0.3"))}
    v1 = {(r["algo"], r["id"]): r["text"] for r in read_jsonl(tmp_path / attack_filename("random@0.3~1"))}
    assert len(h) == len(v0) == len(v1) == 12
    assert sum(v0[k] != v1[k] for k in v0) >= 10                      # a different seed picks different words
    assert read_jsonl(tmp_path / attack_filename("random@0.3~1"))[0]["variant"] == 1
    assert read_jsonl(tmp_path / attack_filename("random@0.3"))[0]["variant"] == 0


def test_chain_variant_uses_matching_first_stage(tmp_path):
    cfg = _make_run(tmp_path, n=8)
    cfg.gate.enforce = False
    rw = build_fake_rewriter(cfg)
    run_attacks(cfg, tmp_path, rw, attacks=["random@0.3~2"], log=lambda m: None)
    calls = rw.backend.calls
    run_attacks(cfg, tmp_path, rw, attacks=["random@0.3+random@0.15~2"], log=lambda m: None)
    assert rw.backend.calls - calls <= 8                               # first stage (variant 2) was re-used, only stage 2 ran
    assert parse_attack("random@0.3+random@0.15~2")["variant"] == 2
    with pytest.raises(ValueError):
        parse_attack("random@0.3~2+random@0.15~2")                    # variant marker only at the very end


def test_prep_attacks_run_but_are_not_evaluated_and_derived_names_are_skipped(tmp_path):
    cfg = _make_run(tmp_path, n=8)
    cfg.gate.enforce = False
    cfg.lab.prep_attacks = ["random@0.3~1"]
    cfg.lab.attacks = ["random@0.15", "best3"]
    run_attacks(cfg, tmp_path, build_fake_rewriter(cfg), log=lambda m: None)
    assert (tmp_path / attack_filename("random@0.3~1")).exists() and (tmp_path / attack_filename("random@0.15")).exists()
    assert not (tmp_path / attack_filename("best3")).exists()


# ------------------------------------------------------------------------------ Option C: selection
BASE = "paraphrase+random2@0.15"


def _selection_run(tmp_path, n_texts=6):
    """3 variants per text.  v0: evades everything but poor quality.  v1: good quality but flagged by a selection detector.
    v2: evades the selection detectors, good quality, but flagged by the HELD-OUT detector."""
    names = variant_names(BASE, 3)
    ids = list(range(n_texts))
    sc, q = [], []
    for det in ("fast_detectgpt", "binoculars", "binoculars_ho"):                    # human negatives -> threshold 0.891
        sc += [{"detector": det, "algo": "none", "attack": "NEG:human", "id": i, "score": i / 10, "sim": 1.0, "changed": 0.0} for i in range(10)]
    for k, nm in enumerate(names):
        recs = []
        for i in ids:
            recs.append({"id": i, "algo": "kgw", "attack": nm, "orig_text": f"orig {i}", "text": f"text {i} variant {k}",
                         "status": "rewritten", "sim": 0.9, "ppl_ratio": 1.5, "edit_ratio": 0.5, "changed": 0.5})
            wm = 1.0
            fdg = 5.0 if k == 1 else 0.1
            bino = 0.1
            ho = 5.0 if k == 2 else 0.1
            for det, s in (("wm_kgw", wm), ("fast_detectgpt", fdg), ("binoculars", bino), ("binoculars_ho", ho)):
                sc.append({"detector": det, "algo": "kgw", "attack": nm, "id": i, "score": s, "sim": 0.9, "changed": 0.5})
            q.append({"kind": nm, "algo": "kgw", "id": i, "fluency": 2 if k == 0 else 5, "meaning": 2 if k == 0 else 5})
        append_jsonl(tmp_path / attack_filename(nm), recs)
    pd.DataFrame(sc).to_csv(tmp_path / "scores.csv", index=False)
    pd.DataFrame(q).to_csv(tmp_path / "quality.csv", index=False)
    cfg = load_config(overrides={"lab": {"algorithms": ["kgw"], "attacks": [BASE, "best3"]}})
    return cfg


def test_selection_picks_by_selection_panel_and_keeps_heldout_honest(tmp_path):
    cfg = _selection_run(tmp_path)
    logs = []
    df = select_best(cfg, tmp_path, BASE, n=3, log=logs.append)
    assert (df["chosen_variant"] == 2).all()                           # passes the selection detectors AND the judge
    assert df["any_perfect"].all() and (df["flags"] == 0).all() and (df["quality_fail"] == 0).all()
    chosen = read_jsonl(tmp_path / attack_filename("best3"))
    assert len(chosen) == 6 and all(r["chosen_from"] == BASE + "~2" and r["attack"] == "best3" for r in chosen)
    sc = pd.read_csv(tmp_path / "scores.csv")
    b = sc[sc.attack == "best3"]
    assert len(b) == 6 * 4 and set(b.detector) == {"wm_kgw", "fast_detectgpt", "binoculars", "binoculars_ho"}
    assert (b[b.detector == "binoculars_ho"]["score"] == 5.0).all()   # the held-out verdict was NOT used for choosing
    st = success_table(tmp_path).set_index("attack")
    assert st.loc["best3", "evade_zs"] == 1.0 and st.loc["best3", "evade_zs_ho"] == 0.0 and st.loc["best3", "success"] == 0.0
    assert any("[select]" in m for m in logs)


def test_selection_without_judge_and_rerun_replaces(tmp_path):
    cfg = _selection_run(tmp_path)
    df = select_best(cfg, tmp_path, BASE, n=3, use_judge=False)
    assert (df["chosen_variant"] == 0).all()                           # without quality info the first perfect-detector variant wins
    select_best(cfg, tmp_path, BASE, n=3, use_judge=True)
    select_best(cfg, tmp_path, BASE, n=3, use_judge=True)
    sc = pd.read_csv(tmp_path / "scores.csv")
    assert len(sc[sc.attack == "best3"]) == 6 * 4                      # no duplicate rows after re-running
    assert len(pd.read_csv(tmp_path / "quality.csv").query("kind == 'best3'")) == 6


def test_selection_falls_back_to_least_bad_candidate(tmp_path):
    cfg = _selection_run(tmp_path, n_texts=4)
    q = pd.read_csv(tmp_path / "quality.csv")
    q.loc[q.kind != BASE, ["fluency", "meaning"]] = 2                 # now every variant has poor quality
    q.to_csv(tmp_path / "quality.csv", index=False)
    df = select_best(cfg, tmp_path, BASE, n=3)
    assert not df["any_perfect"].any() and (df["quality_fail"] == 1).all()


def test_evaluation_after_selection_loads_no_detector(tmp_path):
    cfg = _selection_run(tmp_path)
    select_best(cfg, tmp_path, BASE, n=3)
    (tmp_path / "generated.jsonl").write_text("")
    loads = []
    def stub(name, kind="zeroshot", algo=None, thr=None):
        def f():
            loads.append(name)
            raise RuntimeError("must not load")
        f.meta = {"name": name, "kind": kind, "algo": algo, "threshold": thr}
        return f
    gen = [{"kind": "watermarked", "algo": "kgw", "id": i, "text": f"orig {i}", "prompt": ""} for i in range(6)]
    append_jsonl(tmp_path / "generated.jsonl", gen)
    sc = pd.read_csv(tmp_path / "scores.csv")                         # the unattacked rows an earlier evaluation would have stored
    none_rows = pd.DataFrame([{"detector": "wm_kgw", "algo": "kgw", "attack": "none", "id": i, "score": 8.0, "sim": 1.0, "changed": 0.0} for i in range(6)])
    pd.concat([sc, none_rows]).to_csv(tmp_path / "scores.csv", index=False)
    cfg.lab.attacks = ["best3"]
    df = evaluate(cfg, tmp_path, [stub("wm_kgw", "watermark", "kgw", 4.0), stub("fast_detectgpt"), stub("binoculars_ho")])
    assert loads == [] and "best3" in set(df["attack"])


# ------------------------------------------------------------------------------ judge B and success-table cleanup
def test_judge_b_runs_with_second_rubric_and_writes_separate_files(tmp_path):
    cfg = _make_run(tmp_path, n=8)
    cfg.gate.enforce = False
    cfg.lab.attacks = ["random@0.3"]
    run_attacks(cfg, tmp_path, build_fake_rewriter(cfg), log=lambda m: None)
    label = run_judge_b(cfg, tmp_path, kinds=["random@0.3"], backend=FakeJudgeBackend(), log=lambda m: None)
    qb = pd.read_csv(tmp_path / "quality_b.csv")
    assert {"human", "none", "random@0.3"} <= set(qb["kind"]) and not (tmp_path / "quality.csv").exists()
    assert (qb[qb.kind == "random@0.3"]["meaning"] == 2).all() and (tmp_path / "quality_b_model.txt").read_text() == label


def test_judge_b_falls_back_when_the_model_cannot_be_used(tmp_path, monkeypatch):
    import sira_lab.backends as backends
    cfg = _make_run(tmp_path, n=6)
    cfg.gate.enforce = False
    cfg.lab.attacks = ["random@0.3"]
    run_attacks(cfg, tmp_path, build_fake_rewriter(cfg), log=lambda m: None)
    def fake_hf(name, **kw):
        if name == cfg.lab.judge_b_model:
            raise OSError("model not found")
        return FakeJudgeBackend()
    monkeypatch.setattr(backends, "HFBackend", fake_hf)
    msgs = []
    label = run_judge_b(cfg, tmp_path, kinds=["random@0.3"], log=msgs.append)
    assert "second wording only" in label and any("Falling back" in m for m in msgs)

    class _Silent:
        def generate(self, *a, **k):
            return ["no digits here"] * len(a[0])
    monkeypatch.setattr(backends, "HFBackend", lambda name, **kw: _Silent() if name == cfg.lab.judge_b_model else FakeJudgeBackend())
    (tmp_path / "quality_b.jsonl").unlink()
    assert "second wording only" in run_judge_b(cfg, tmp_path, kinds=["random@0.3"], log=lambda m: None)


def test_success_table_leaves_out_attacks_without_judge_scores(tmp_path):
    cfg = _selection_run(tmp_path)
    sc = pd.read_csv(tmp_path / "scores.csv")
    extra = sc[sc.attack == BASE].copy(); extra["attack"] = "legacy@0.15"          # attack with scores but never judged
    pd.concat([sc, extra]).to_csv(tmp_path / "scores.csv", index=False)
    logs = []
    st = success_table(tmp_path, log=logs.append)
    assert "legacy@0.15" not in set(st["attack"]) and any("legacy@0.15" in m for m in logs)
    q = pd.read_csv(tmp_path / "quality.csv"); q["fluency"] = 3
    q.to_csv(tmp_path / "quality_b.csv", index=False)
    stb = success_table(tmp_path, quality_file="quality_b.csv").set_index("attack")
    assert stb.loc[BASE, "quality_ok"] == 0.0 and stb["quality_basis"].iloc[0].startswith("judge (quality_b")

import numpy as np
import pandas as pd
import pytest

from sira_lab.config import load_config
from sira_lab.lab.analysis import plot_success, success_table
from sira_lab.lab.attacks import attack_filename, parse_attack, run_attacks
from sira_lab.lab.detectors import detector_factories
from sira_lab.lab.io import append_jsonl, read_jsonl
from sira_lab.lab.judge import parse_digit, run_judge
from sira_lab.testing import FakeJudgeBackend, build_fake_rewriter
from test_lab_pipeline import _make_run


def test_chain_parsing_and_filenames():
    c = parse_attack("paraphrase+random@0.15")
    assert c["kind"] == "chain" and c["parts"] == ["paraphrase", "random@0.15"] and c["variant"] == 0
    assert attack_filename("paraphrase+sira@0.3x2") == "attacked_paraphrase_plus_sira_r0.3x2.jsonl"
    for bad in ("paraphrase+", "+sira", "sira+banana", "paraphrase++sira"):
        with pytest.raises(ValueError):
            parse_attack(bad)


def test_chain_reuses_finished_first_stage_and_edits_further(tmp_path):
    cfg = _make_run(tmp_path, n=16)
    cfg.lab.attacks = ["random@0.3", "random@0.3+sira@0.15"]
    cfg.gate.enforce = False
    rw = build_fake_rewriter(cfg)
    run_attacks(cfg, tmp_path, rw, attacks=["random@0.3"], log=lambda m: None)
    calls_before = rw.backend.calls
    run_attacks(cfg, tmp_path, rw, attacks=["random@0.3+sira@0.15"], log=lambda m: None)
    first = {(r["algo"], r["id"]): r for r in read_jsonl(tmp_path / attack_filename("random@0.3"))}
    chain = read_jsonl(tmp_path / attack_filename("random@0.3+sira@0.15"))
    assert len(chain) == 16
    # stage 1 was NOT recomputed (only one extra rewrite pass happened) ...
    assert rw.backend.calls - calls_before <= 16
    # ... and the chain changed MORE words than stage 1 alone, measured against the ORIGINAL text
    assert np.mean([r["changed"] for r in chain]) > np.mean([first[(r["algo"], r["id"])]["changed"] for r in chain])
    assert all(r["orig_text"] == first[(r["algo"], r["id"])]["orig_text"] for r in chain)


def test_judge_parse_and_run_resumes(tmp_path):
    assert parse_digit("4") == 4 and parse_digit("Score: 5/5") == 5 and parse_digit("n/a") is None and parse_digit(None) is None
    cfg = _make_run(tmp_path, n=8)
    cfg.lab.attacks = ["sira@0.3"]
    cfg.gate.enforce = False
    run_attacks(cfg, tmp_path, build_fake_rewriter(cfg), log=lambda m: None)
    backend = FakeJudgeBackend()
    run_judge(cfg, tmp_path, backend, log=lambda m: None)
    q = pd.read_csv(tmp_path / "quality.csv")
    assert {"kind", "algo", "id", "fluency", "meaning"} <= set(q.columns)
    assert set(q["kind"]) == {"human", "unwatermarked", "none", "sira@0.3"}
    assert (q[q.kind == "none"]["fluency"] == 5).all()                       # untouched text: fluent
    assert (q[q.kind == "sira@0.3"]["fluency"] <= 3).all() and (q[q.kind == "sira@0.3"]["meaning"] == 2).all()
    assert q[q.kind != "sira@0.3"]["meaning"].isna().all()                   # no meaning score for untouched text
    n_lines = len(read_jsonl(tmp_path / "quality.jsonl"))
    run_judge(cfg, tmp_path, backend, log=lambda m: None)                    # second run: nothing left to do
    assert len(read_jsonl(tmp_path / "quality.jsonl")) == n_lines


def _synthetic_run(tmp_path, with_quality):
    rows = []
    def add(det, algo, attack, ids, scores, sim=1.0):
        for i, s in zip(ids, scores):
            rows.append({"detector": det, "algo": algo, "attack": attack, "id": i, "score": s, "sim": sim, "changed": 0.0})
    ids = list(range(10))
    for det in ("fast_detectgpt", "binoculars_ho"):                     # human negatives: scores 0..0.9, 1% threshold = 0.891
        add(det, "none", "NEG:human", ids, [x / 10 for x in range(10)])
    add("wm_kgw", "kgw", "none", ids, [8.0] * 10)                       # unattacked: everything detected
    add("fast_detectgpt", "kgw", "none", ids, [3.0] * 10); add("binoculars_ho", "kgw", "none", ids, [3.0] * 10)
    # attack A: evades wm for all, evades zero-shot for first 6 only; attack B: evades everything for ids < 4
    add("wm_kgw", "kgw", "A", ids, [1.0] * 10, sim=0.9)
    add("fast_detectgpt", "kgw", "A", ids, [0.1] * 6 + [5.0] * 4, sim=0.9)
    add("binoculars_ho", "kgw", "A", ids, [0.1] * 10, sim=0.9)
    add("wm_kgw", "kgw", "B", ids, [1.0] * 4 + [9.0] * 6, sim=0.5)
    add("fast_detectgpt", "kgw", "B", ids, [0.1] * 10, sim=0.5)
    add("binoculars_ho", "kgw", "B", ids, [0.1] * 10, sim=0.5)
    pd.DataFrame(rows).to_csv(tmp_path / "scores.csv", index=False)
    if with_quality:
        q = [{"kind": "A", "algo": "kgw", "id": i, "fluency": 5 if i < 3 else 2, "meaning": 5} for i in ids]
        q += [{"kind": "B", "algo": "kgw", "id": i, "fluency": 5, "meaning": 5} for i in ids]
        q += [{"kind": "none", "algo": "kgw", "id": i, "fluency": 5, "meaning": None} for i in ids]
        pd.DataFrame(q).to_csv(tmp_path / "quality.csv", index=False)


def test_success_table_logic_with_judge(tmp_path):
    _synthetic_run(tmp_path, with_quality=True)
    t = success_table(tmp_path).set_index("attack")
    assert t.loc["none", "success"] == 0.0 and t.loc["none", "evade_all"] == 0.0
    assert t.loc["A", "evade_wm"] == 1.0 and t.loc["A", "evade_zs"] == 0.6 and t.loc["A", "evade_zs_ho"] == 1.0
    assert t.loc["A", "evade_all"] == 0.6 and t.loc["A", "quality_ok"] == 0.3
    assert t.loc["A", "success"] == pytest.approx(0.3)                   # ids 0,1,2: evade everything AND fluent
    assert t.loc["B", "evade_all"] == 0.4 and t.loc["B", "success"] == 0.4
    assert (t["quality_basis"] == "judge").all()
    assert (t["success_lo"] <= t["success"]).all() and (t["success"] <= t["success_hi"]).all()


def test_success_table_falls_back_to_similarity(tmp_path):
    _synthetic_run(tmp_path, with_quality=False)
    t = success_table(tmp_path).set_index("attack")
    assert (t["quality_basis"] == "similarity").all()
    assert t.loc["A", "success"] == pytest.approx(0.6) and t.loc["B", "success"] == 0.0      # B has sim 0.5 < 0.85
    plot_success(t.reset_index(), tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 1000


def test_v7_config_is_valid():
    cfg = load_config("configs/lab_v7.yaml")
    for a in cfg.lab.attacks:
        parse_attack(a)
    assert cfg.gate.enforce is False and "fast_detectgpt_ho" in cfg.lab.detectors
    assert len(detector_factories(cfg)) == len(cfg.lab.algorithms) + len(cfg.lab.detectors)


# ------------------------------------------------------------------------------------------ score caching
from sira_lab.lab.detectors import WatermarkZ
from sira_lab.lab.evaluate import evaluate
from sira_lab.lab.generate import spec_for
from toylm import encode


class _Counting:
    kind, algo, threshold = "zeroshot", None, None

    def __init__(self, name="count_zs"):
        self.name, self.n_scored = name, 0

    def score(self, texts):
        self.n_scored += len(texts)
        return np.array([len(t) % 7 + 0.5 for t in texts], dtype=float)


def _small_run(tmp_path):
    cfg = _make_run(tmp_path, n=10)
    cfg.lab.attacks = ["sira@0.3"]
    cfg.gate.enforce = False
    run_attacks(cfg, tmp_path, build_fake_rewriter(cfg), log=lambda m: None)
    return cfg


def test_scores_are_cached_and_models_not_reloaded(tmp_path):
    cfg = _small_run(tmp_path)
    spec = spec_for(cfg.lab, "kgw")
    first = _Counting()
    df1 = evaluate(cfg, tmp_path, [WatermarkZ(spec, encode), first])
    assert first.n_scored > 0

    loads = []
    def factory():
        loads.append(1)
        return _Counting()
    factory.meta = {"name": "count_zs", "kind": "zeroshot", "algo": None, "threshold": None}
    df2 = evaluate(cfg, tmp_path, [WatermarkZ(spec, encode), factory])
    assert loads == []                                                     # every score cached -> model never loaded
    key = ["detector", "algo", "attack"]
    pd.testing.assert_frame_equal(df1.sort_values(key).reset_index(drop=True), df2.sort_values(key).reset_index(drop=True))


def test_new_detector_only_scores_itself_and_changed_texts_are_rescored(tmp_path):
    cfg = _small_run(tmp_path)
    spec = spec_for(cfg.lab, "kgw")
    evaluate(cfg, tmp_path, [WatermarkZ(spec, encode)])
    new = _Counting("another_zs")
    evaluate(cfg, tmp_path, [WatermarkZ(spec, encode), new])
    sc = pd.read_csv(tmp_path / "scores.csv")
    assert {"wm_kgw", "another_zs"} <= set(sc["detector"])                 # old detector's scores kept, new ones added
    n_first = new.n_scored
    # change one attacked text: only that text should be re-scored for the detector
    path = tmp_path / "attacked_sira_r0.3.jsonl"
    recs = read_jsonl(path)
    recs[0]["text"] += " extra words appended"
    path.write_text("".join(__import__("json").dumps(r) + "\n" for r in recs))
    again = _Counting("another_zs")
    evaluate(cfg, tmp_path, [WatermarkZ(spec, encode), again])
    assert 0 < again.n_scored <= 2 and n_first > again.n_scored


def test_interrupted_evaluation_keeps_finished_detectors(tmp_path):
    cfg = _small_run(tmp_path)
    spec = spec_for(cfg.lab, "kgw")

    class _Boom(_Counting):
        def score(self, texts):
            raise RuntimeError("GPU quota exhausted")

    with pytest.raises(RuntimeError):
        evaluate(cfg, tmp_path, [WatermarkZ(spec, encode), _Boom("crashes")])
    sc = pd.read_csv(tmp_path / "scores.csv")
    assert "wm_kgw" in set(sc["detector"]) and len(sc) > 0                 # first detector's scores survived the crash

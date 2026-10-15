import numpy as np
import pandas as pd
import pytest

from sira_lab.config import GateCfg, load_config
from sira_lab.gates import check
from sira_lab.lab.analysis import auto_pairs, load_scores, paired_table
from sira_lab.lab.attacks import attack_filename, parse_attack, run_attacks
from sira_lab.lab.detectors import WatermarkZ
from sira_lab.lab.evaluate import evaluate, plot_dose_response
from sira_lab.lab.generate import spec_for
from sira_lab.lab.io import read_jsonl
from sira_lab.lab.metrics import changed_share
from sira_lab.testing import DifflibSim, FakeScorer, build_fake_rewriter
from test_lab_pipeline import _make_run
from toylm import encode

DOSE = ["sira@0.3", "random@0.3", "sira@0.15", "random@0.15", "sira@0.3x2", "random@0.3x2", "paraphrase"]


def test_parse_attack_and_filenames():
    a = parse_attack("sira")
    assert (a["kind"], a["strategy"], a["ratio"], a["rounds"], a["hygiene"], a["variant"]) == ("rewrite", "surprisal", None, None, False, 0)
    assert parse_attack("random@0.15")["ratio"] == 0.15
    b = parse_attack("lowest@0.3x2")
    assert (b["strategy"], b["ratio"], b["rounds"]) == ("lowest", 0.3, 2)
    assert parse_attack("paraphrase")["kind"] == "paraphrase"
    assert attack_filename("sira") == "attacked_sira.jsonl" and attack_filename("sira@0.3x2") == "attacked_sira_r0.3x2.jsonl"
    for bad in ("siraa", "sira@", "random@x", "banana@0.3"):
        with pytest.raises(ValueError):
            parse_attack(bad)


def test_changed_share():
    assert changed_share("a b c d", "a b c d") == 0.0
    assert changed_share("a b c d", "a x c d") == 0.25
    assert changed_share("a b c d", "w x y z q") == 1.0 and changed_share("", "anything") == 0.0


def test_gate_enforce_flag_only_changes_acceptance():
    scorer, sim = FakeScorer(), DifflibSim()
    strict = check("one two three four", "totally different words here", 3.0, sim, scorer, GateCfg(min_sim=0.9, enforce=True))
    loose = check("one two three four", "totally different words here", 3.0, sim, scorer, GateCfg(min_sim=0.9, enforce=False))
    assert not strict.ok and loose.ok and loose.reasons == strict.reasons and loose.sim == strict.sim


def test_dose_attacks_record_edit_levels(tmp_path):
    cfg = _make_run(tmp_path, n=16)
    cfg.lab.attacks = DOSE
    cfg.gate.enforce = False
    rw = build_fake_rewriter(cfg)
    run_attacks(cfg, tmp_path, rw, log=lambda m: None)
    mean_changed = {}
    for a in DOSE:
        recs = read_jsonl(tmp_path / attack_filename(a))
        assert len(recs) == 16 and all("changed" in r for r in recs)
        mean_changed[a] = np.mean([r["changed"] for r in recs])
        if a != "paraphrase":
            assert all(r["status"] == "rewritten" for r in recs) and all(r["mask_ratio_used"] is not None for r in recs)
    assert mean_changed["sira@0.3"] > mean_changed["sira@0.15"] * 1.5          # more mask ratio -> more edits
    assert mean_changed["sira@0.3x2"] > mean_changed["sira@0.3"]                # a second round edits more
    assert abs(mean_changed["sira@0.3"] - mean_changed["random@0.3"]) < 0.08    # matched edit budget
    assert rw.cfg.mask.strategy == "surprisal"


def test_evaluate_adds_rates_scores_and_paired_table(tmp_path):
    cfg = _make_run(tmp_path, n=24)
    cfg.lab.attacks = DOSE
    cfg.gate.enforce = False
    run_attacks(cfg, tmp_path, build_fake_rewriter(cfg), log=lambda m: None)
    spec = spec_for(cfg.lab, "kgw")
    df = evaluate(cfg, tmp_path, [WatermarkZ(spec, encode)])
    assert {"detect_rate", "rate_lo", "rate_hi", "changed_mean"} <= set(df.columns)
    r = df.set_index("attack")
    assert r.loc["none", "detect_rate"] == 1.0 and r.loc["none", "changed_mean"] == 0.0
    assert (r["rate_lo"] <= r["detect_rate"] + 1e-9).all() and (r["detect_rate"] <= r["rate_hi"] + 1e-9).all()
    assert r.loc["sira@0.3", "detect_rate"] <= r.loc["sira@0.15", "detect_rate"] + 1e-9    # more editing, less detection
    sc = load_scores(tmp_path)
    assert {"detector", "algo", "attack", "id", "score", "sim", "changed"} <= set(sc.columns)
    assert sc["attack"].str.startswith("NEG:").any()
    pt = paired_table(sc)
    assert set(pt["a"]) == {"sira@0.3", "sira@0.15", "sira@0.3x2"} and {"verdict", "ci_lo", "ci_hi"} <= set(pt.columns)
    plot_dose_response(df, tmp_path / "d.png")
    assert (tmp_path / "d.png").stat().st_size > 1000


def test_missing_attack_files_are_skipped_not_fatal(tmp_path, capsys):
    cfg = _make_run(tmp_path, n=12)
    cfg.lab.attacks = ["sira@0.3", "random@0.3"]
    run_attacks(cfg, tmp_path, build_fake_rewriter(cfg), attacks=["sira@0.3"], log=lambda m: None)
    df = evaluate(cfg, tmp_path, [WatermarkZ(spec_for(cfg.lab, "kgw"), encode)])
    assert "random@0.3" not in set(df["attack"]) and "sira@0.3" in set(df["attack"])
    assert "no results yet" in capsys.readouterr().out


def _fake_scores(a_shift, n=40, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(5, 1, n)
    rows = []
    for i in range(n):
        for atk, shift in (("sira@0.3", a_shift), ("random@0.3", 0.0)):
            rows.append({"detector": "wm_kgw", "algo": "kgw", "attack": atk, "id": i,
                         "score": base[i] + shift + rng.normal(0, 0.3), "sim": 0.9, "changed": 0.2})
    return pd.DataFrame(rows)


def test_paired_table_detects_real_differences_only():
    better = paired_table(_fake_scores(-1.0)).iloc[0]
    assert better["mean_diff"] < -0.8 and better["ci_hi"] < 0 and better["verdict"] == "sira@0.3 hides better"
    worse = paired_table(_fake_scores(+1.0)).iloc[0]
    assert worse["verdict"] == "random@0.3 hides better"
    same = paired_table(_fake_scores(0.0)).iloc[0]
    assert same["verdict"] == "no clear difference" and same["ci_lo"] < 0 < same["ci_hi"]
    assert auto_pairs(["sira@0.3", "random@0.3", "lowest@0.3", "sira", "paraphrase"]) == [("sira@0.3", "random@0.3"), ("sira@0.3", "lowest@0.3")]

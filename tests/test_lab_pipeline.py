import json
import numpy as np
import pytest

from sira_lab.config import load_config
from sira_lab.lab.attacks import run_attacks
from sira_lab.lab.detectors import WatermarkZ
from sira_lab.lab.evaluate import evaluate, to_markdown, plot_tradeoff
from sira_lab.lab.watermark import detect_text
from sira_lab.lab.generate import spec_for
from sira_lab.lab.io import append_jsonl, read_jsonl
from sira_lab.lab.prompts import BUILTIN, load_prompts
from sira_lab.testing import build_fake_rewriter
from toylm import encode, sample_text


def _make_run(tmp_path, n=24):
    cfg = load_config(overrides={
        "lab": {"algorithms": ["kgw"], "attacks": ["sira", "random", "lowest", "paraphrase"], "n_samples": n,
                "attack_batch_size": 8},
        "gate": {"min_sim": 0.0, "max_ppl_ratio": 1e9},
        "mask": {"mask_ratio": 0.3},
    })
    spec, rng = spec_for(cfg.lab, "kgw"), np.random.default_rng(3)
    rows = []
    for i in range(n):
        rows.append({"kind": "watermarked", "algo": "kgw", "id": i, "prompt": "", "text": sample_text(150, spec, rng)})
        rows.append({"kind": "unwatermarked", "algo": "none", "id": i, "prompt": "", "text": sample_text(150, None, rng)})
        rows.append({"kind": "human", "algo": "none", "id": i, "prompt": "", "text": sample_text(150, None, rng)})
    append_jsonl(tmp_path / "generated.jsonl", rows)
    return cfg


def test_prompts_builtin_and_fallback():
    cfg = load_config(overrides={"lab": {"source": "builtin", "n_samples": 30}})
    rows = load_prompts(cfg.lab)
    assert len(rows) == 30 and all(r["prompt"] for r in rows) and rows[0]["human"] == ""
    assert {r["id"] for r in rows} == set(range(30))


def test_full_lab_flow_with_fakes(tmp_path):
    cfg = _make_run(tmp_path)
    rw = build_fake_rewriter(cfg)
    logs = []
    run_attacks(cfg, tmp_path, rw, log=logs.append)
    for a in cfg.lab.attacks:
        recs = read_jsonl(tmp_path / f"attacked_{a}.jsonl")
        assert len(recs) == 24 and {"sim", "ppl_ratio", "edit_ratio", "status"} <= set(recs[0])

    spec = spec_for(cfg.lab, "kgw")

    class FakeZeroShot:                                    # 'machine-ness' = share of green tokens (plumbing test only)
        kind, name, algo, threshold = "zeroshot", "fake_zs", None, None
        def score(self, texts):
            return np.array([detect_text(t, encode, spec)["green_frac"] for t in texts])

    df = evaluate(cfg, tmp_path, [WatermarkZ(spec, encode), lambda: FakeZeroShot()])
    wm = df[df["detector"] == "wm_kgw"].set_index("attack")
    assert set(wm.index) == {"none", "sira", "random", "lowest", "paraphrase"}
    assert wm.loc["none", "TPR@FPR"] == 1.0 and wm.loc["none", "mean_score"] > 6
    assert wm.loc["sira", "mean_score"] < wm.loc["none", "mean_score"] - 2     # masking words erases signal
    assert wm.loc["sira", "edit_mean"] < 1.0 and wm.loc["none", "FPR@thr"] == 0.0
    assert wm.loc["paraphrase", "frac_unchanged"] == 1.0                       # fake paraphraser returns the input
    zs = df[df["detector"] == "fake_zs"]
    assert (zs["neg_set"] == "human").all() and "unwatermarked (AI baseline)" in set(zs["attack"])
    assert zs.set_index("attack").loc["none", "AUROC"] > 0.9
    assert (tmp_path / "results.csv").exists() and "TPR@FPR" in to_markdown(df)
    plot_tradeoff(df, tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000


def test_attacks_are_resumable(tmp_path):
    cfg = _make_run(tmp_path, n=16)
    rw = build_fake_rewriter(cfg)
    run_attacks(cfg, tmp_path, rw, attacks=["sira"], limit=8, log=lambda m: None)
    assert len(read_jsonl(tmp_path / "attacked_sira.jsonl")) == 8
    run_attacks(cfg, tmp_path, rw, attacks=["sira"], log=lambda m: None)
    recs = read_jsonl(tmp_path / "attacked_sira.jsonl")
    assert len(recs) == 16 and len({r["id"] for r in recs}) == 16


def test_strategies_pick_different_words(tmp_path):
    cfg = _make_run(tmp_path, n=8)
    rw = build_fake_rewriter(cfg)
    run_attacks(cfg, tmp_path, rw, attacks=["sira", "lowest"], log=lambda m: None)
    a = {r["id"]: r["text"] for r in read_jsonl(tmp_path / "attacked_sira.jsonl")}
    b = {r["id"]: r["text"] for r in read_jsonl(tmp_path / "attacked_lowest.jsonl")}
    assert any(a[i] != b[i] for i in a)
    assert rw.cfg.mask.strategy == "surprisal"                         # restored after the attack


def test_failed_rewrites_store_the_models_answer(tmp_path):
    cfg = _make_run(tmp_path, n=4)
    rw = build_fake_rewriter(cfg, fail_first=10_000)          # model never follows the format
    run_attacks(cfg, tmp_path, rw, attacks=["sira"], log=lambda m: None)
    recs = read_jsonl(tmp_path / "attacked_sira.jsonl")
    assert all(r["status"] == "unchanged" for r in recs)
    assert any("Sorry, I cannot do that" in w for r in recs for w in r["warnings"])

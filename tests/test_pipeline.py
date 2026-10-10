from sira_lab.config import load_config
from sira_lab.testing import build_fake_rewriter

DOC = """2. Methods

We evaluate watermark robustness on a corpus of scientific abstracts, and we compare several rewriting strategies under a fixed budget. The analysis suggests that targeted masking preserves meaning better than random masking in most settings tested here.

Figure 3: Distribution of scores across the evaluated detectors.

Results indicate that detectors based on token statistics degrade quickly when high-entropy words are replaced, whereas classifier based detectors remain noisy across domains and writing styles."""


def _cfg(**over):
    return load_config(overrides=over)


def test_end_to_end_skips_and_rewrites():
    rep = build_fake_rewriter(_cfg(gate={"min_sim": 0.0, "max_ppl_ratio": 100})).rewrite(DOC)
    statuses = [c.status for c in rep.chunks]
    assert statuses == ["skipped:heading", "rewritten", "skipped:caption", "rewritten"]
    assert rep.words_masked > 0 and "alt1" in rep.text_out
    assert "2. Methods" in rep.text_out and "Figure 3:" in rep.text_out
    assert rep.sim is not None and rep.to_markdown()


def test_parse_failure_is_retried():
    rw = build_fake_rewriter(_cfg(gate={"min_sim": 0.0, "max_ppl_ratio": 100}), fail_first=2)
    rep = rw.rewrite(DOC)
    assert all(c.status in ("rewritten",) or c.status.startswith("skipped") for c in rep.chunks)
    assert max(c.attempts for c in rep.chunks) >= 2


def test_gate_failure_lowers_ratio_then_keeps_best():
    rw = build_fake_rewriter(_cfg(gate={"min_sim": 0.99}))
    rep = rw.rewrite(DOC)
    bad = [c for c in rep.chunks if c.status == "best-effort"]
    assert bad and rep.warnings
    assert all(c.mask_ratio_used is not None for c in bad)


def test_on_fail_original():
    rw = build_fake_rewriter(_cfg(gate={"min_sim": 0.99, "on_fail": "original"}))
    rep = rw.rewrite(DOC)
    assert all(c.out == c.orig for c in rep.chunks if c.status == "unchanged")


def test_multi_round_and_overrides():
    rw = build_fake_rewriter(_cfg(gate={"min_sim": 0.0, "max_ppl_ratio": 100}))
    rep = rw.rewrite(DOC, rounds=2, mask_ratio=0.1, temperature=0.0)
    assert rep.words_masked > 0


def test_input_limit():
    import pytest
    rw = build_fake_rewriter(_cfg(max_input_words=10))
    with pytest.raises(ValueError):
        rw.rewrite(DOC)


def test_heatmap_roundtrip():
    segs = build_fake_rewriter(_cfg()).heatmap(DOC)
    assert any(lbl == "masked" for _, lbl in segs)

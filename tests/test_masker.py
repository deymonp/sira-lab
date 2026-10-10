from sira_lab.config import MaskCfg
from sira_lab.masker import find_words, make_template, score_words, select_masks, heat_segments
from sira_lab.testing import FakeScorer

TEXT = ("Recent work by Smith et al. (Smith et al., 2024) shows that GPT-4 outperforms baselines [12] "
        "on 3 datasets, e.g. \"state of the art\" results appear in Figure 2. The analysis suggests substantial gains.")


def _words(text=TEXT):
    cfg = MaskCfg()
    ts = FakeScorer().score(text)
    words = find_words(text, cfg)
    score_words(words, ts.spans, ts.surprisal)
    return cfg, words


def test_protected_things_are_never_candidates():
    _, words = _words()
    cand = {w.text for w in words if w.candidate}
    for banned in ("Smith", "GPT", "Figure", "state", "art", "et", "al"):
        assert banned not in cand
    assert "outperforms" in cand or "baselines" in cand


def test_sentence_start_capital_is_allowed_midsentence_is_not():
    cfg = MaskCfg()
    words = find_words("The analysis suggests that Berlin hosts results.", cfg)
    cand = {w.text for w in words if w.candidate}
    assert "analysis" in cand and "Berlin" not in cand


def test_select_and_template_roundtrip():
    cfg, words = _words()
    chosen = select_masks(words, cfg, ratio=0.2)
    assert 0 < len(chosen) <= round(0.2 * len(words))
    assert all(w.candidate for w in chosen)
    mr = make_template(TEXT, chosen, words)
    assert mr.template.count("<mask_") == len(chosen)
    rebuilt = mr.template
    for m in reversed(mr.masks):
        rebuilt = rebuilt.replace(f"<mask_{m.number}>", m.original, 1)
    assert rebuilt == TEXT


def test_selection_prefers_high_scores():
    cfg, words = _words()
    chosen = select_masks(words, cfg, ratio=0.1)
    best = max((w for w in words if w.candidate), key=lambda w: w.score)
    assert best in chosen


def test_min_gap():
    cfg, words = _words()
    cfg.min_gap = 2
    chosen = select_masks(words, cfg, ratio=0.5)
    idx = [w.idx for w in chosen]
    assert all(b - a >= 3 for a, b in zip(idx, idx[1:]))


def test_heat_segments_cover_text():
    cfg, words = _words()
    chosen = select_masks(words, cfg, ratio=0.2)
    assert "".join(s for s, _ in heat_segments(TEXT, words, chosen)) == TEXT

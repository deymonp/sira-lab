from sira_lab.infill import (build_messages, extract_fills, extract_passage, fill_template,
                             lenient_accept, parse_candidate)

T = "The results <mask_1> that larger models <mask_2> better calibration, although the <mask_3> is small."


def test_extract_and_fill():
    out = "<passage>\nThe results suggest that larger models achieve better calibration, although the effect is small.\n</passage>"
    fills = extract_fills(T, extract_passage(out))
    assert fills == ["suggest", "achieve", "effect"]
    assert fill_template(T, fills).startswith("The results suggest that")


def test_altered_outside_text_is_rejected():
    bad = "The findings suggest that larger models achieve better calibration, although the effect is small."
    assert extract_fills(T, bad) is None
    assert parse_candidate(T, ["a", "b", "c"], bad, strict=True) is None


def test_lenient_mode_accepts_close_rewrite():
    bad = "The findings suggest that larger models achieve better calibration, although the effect is small."
    got = parse_candidate(T, ["a", "b", "c"], bad, strict=False)
    assert got is not None and got[0] == bad


def test_multiword_fill_and_whitespace_tolerance():
    out = "The results   clearly indicate\nthat larger models tend to show better calibration, although the net effect is small."
    fills = extract_fills(T, out)
    assert fills == ["clearly indicate", "tend to show", "net effect"]


def test_too_long_fill_rejected():
    out = "The results one two three four five that larger models achieve better calibration, although the effect is small."
    assert parse_candidate(T, ["a", "b", "c"], out) is None


def test_adjacent_masks_resplit_harmlessly():
    t = "We <mask_1> <mask_2> results today."
    out = "We report very strong results today."
    fills = extract_fills(t, out)
    assert " ".join(fill_template(t, fills).split()) == out


def test_messages_shape():
    m = build_messages(T)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"] and T in m[-1]["content"]


# ---------------------------------------------------------------- list mode
from sira_lab.infill import parse_list_candidate, parse_list_output, parse_output


def test_list_mode_basic_and_formats():
    raw = "1: suggest\n2. achieve\n3) effect"
    got, reason = parse_list_candidate(T, ["a", "b", "c"], raw)
    assert reason == "" and got[0] == "The results suggest that larger models achieve better calibration, although the effect is small."
    assert parse_list_output("- mask_1: x\n<mask_2>: y") == {1: "x", 2: "y"}


def test_list_mode_cleans_case_and_punctuation():
    t = "<mask_1> results show that the model is <mask_2>."
    got, _ = parse_list_candidate(t, ["Recent", "robust"], '1: "recent",\n2: Strong.')
    assert got[0] == "Recent results show that the model is strong."


def test_list_mode_partial_keeps_original_and_junk_fails():
    got, _ = parse_list_candidate(T, ["suggest", "achieve", "effect"], "1: indicate\n2: reach")
    assert got[0].startswith("The results indicate that larger models reach") and got[0].endswith("the effect is small.")
    none, reason = parse_list_candidate(T, ["a", "b", "c"], "I'm sorry, I can't help with that.")
    assert none is None and "usable fills" in reason


def test_build_messages_list_mode_and_dispatch():
    m = build_messages(T, "list")
    assert "one line per placeholder" in m[-1]["content"] and m[2]["content"].startswith("1: ")
    cand, reason = parse_output("passage", T, ["a", "b", "c"], "nonsense")
    assert cand is None and reason

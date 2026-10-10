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

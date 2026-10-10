from sira_lab.normalize import normalize


def test_strips_invisible_and_spaces():
    t, s = normalize("Hel\u200blo\u00a0world\u00ad  again")
    assert t == "Hello world again"
    assert s["invisible_removed"] == 2 and s["spaces_normalized"] == 1


def test_ligatures_and_homoglyphs():
    t, s = normalize("The \ufb01nal r\u0435sult")  # Cyrillic 'e'
    assert t == "The final result"
    assert s["homoglyphs_fixed"] == 1 and s["ligatures_expanded"] == 1


def test_keeps_superscripts_and_paragraphs():
    t, _ = normalize("Area in m\u00b2.\n\n\n\nNext paragraph.")
    assert "m\u00b2" in t and t.count("\n\n") == 1

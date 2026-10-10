from sira_lab.chunking import join_chunks, make_chunks
from sira_lab.config import ChunkCfg

DOC = """1. Introduction

Large language models can be watermarked by biasing token choice during sampling, and detectors look for that bias in a statistical test.

Figure 1: Overview of the proposed method and its components.

[1] Smith, J. (2024). A paper. doi.org/10.1/abc

x = a + b * c / d - e = 0 ( 1 2 3 )"""


def test_skip_reasons():
    ch = make_chunks(DOC, ChunkCfg())
    reasons = [c.skip_reason for c in ch]
    assert reasons == ["heading", None, "caption", "reference", "non-prose"]


def test_long_paragraph_is_split_and_rejoins():
    sent = "This is a reasonably long sentence about watermark detection in text. "
    par = (sent * 40).strip()
    ch = make_chunks(par, ChunkCfg(max_words=60))
    assert len(ch) > 3 and all(len(c.text.split()) <= 70 for c in ch)
    assert join_chunks(ch, [c.text for c in ch]).split() == par.split()

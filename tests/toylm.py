"""A numpy 'language model' for testing watermarks without any ML framework."""
import zlib
import numpy as np

from sira_lab.lab.watermark import green_mask, hash_bits

V = 2000


def _word(i: int) -> str:
    s, n = "", i
    for _ in range(4):
        s += "abcdefghijklmnopqrstuvwxyz"[n % 26]
        n //= 26
    return "q" + s + "x"          # 6 letters, never a stopword


VOCAB = [_word(i) for i in range(V)]
W2I = {w: i for i, w in enumerate(VOCAB)}


def encode(text: str):
    return [W2I.get(w, zlib.crc32(w.encode()) % V) for w in text.split()]


def sample_text(n_tokens: int, spec, rng: np.random.Generator, base_scale: float = 0.0) -> str:
    """Sample from a toy 'language model' (random logits, spread controlled by base_scale).
    With a spec: kgw/unigram bias green tokens by +delta; exp picks argmax log(u)/p (Aaronson)."""
    toks = [int(rng.integers(V))]
    allv = np.arange(V, dtype=np.int64)
    for _ in range(n_tokens - 1):
        logits = rng.normal(size=V) * base_scale
        p = np.exp(logits - logits.max())
        p /= p.sum()
        if spec is not None and spec.algo == "exp":
            u = (hash_bits(np.int64(toks[-1]), allv, spec).astype(np.float64) + 0.5) / 4294967296.0
            toks.append(int(np.argmax(np.log(u) / np.maximum(p, 1e-30))))
            continue
        if spec is not None:
            logits = logits + spec.delta * green_mask(np.int64(toks[-1]), allv, spec)
            p = np.exp(logits - logits.max())
            p /= p.sum()
        toks.append(int(rng.choice(V, p=p)))
    return " ".join(VOCAB[t] for t in toks)

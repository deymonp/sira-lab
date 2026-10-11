import numpy as np
import pytest

from sira_lab.lab.watermark import WatermarkSpec, detect_ids, detect_text, green_mask, z_score
from toylm import encode, sample_text, V


def test_green_fraction_matches_gamma_and_is_deterministic():
    rng = np.random.default_rng(0)
    prev = rng.integers(0, 150_000, 200_000).astype(np.int64)
    tok = rng.integers(0, 150_000, 200_000).astype(np.int64)
    for algo in ("kgw", "unigram"):
        spec = WatermarkSpec(algo=algo, gamma=0.5)
        m = green_mask(prev, tok, spec)
        assert abs(m.mean() - 0.5) < 0.01
        assert (m == green_mask(prev, tok, spec)).all()
    spec = WatermarkSpec(algo="kgw", gamma=0.25)
    assert abs(green_mask(prev, tok, spec).mean() - 0.25) < 0.01


def test_kgw_depends_on_previous_token_unigram_does_not():
    tok = np.arange(5000, dtype=np.int64)
    kgw, uni = WatermarkSpec(algo="kgw"), WatermarkSpec(algo="unigram")
    a, b = green_mask(np.int64(1), tok, kgw), green_mask(np.int64(2), tok, kgw)
    assert 0.4 < (a == b).mean() < 0.6            # independent lists -> ~50% agreement
    assert (green_mask(np.int64(1), tok, uni) == green_mask(np.int64(2), tok, uni)).all()


def test_key_changes_the_green_list():
    tok = np.arange(5000, dtype=np.int64)
    a = green_mask(np.int64(7), tok, WatermarkSpec(key=1))
    b = green_mask(np.int64(7), tok, WatermarkSpec(key=2))
    assert 0.4 < (a == b).mean() < 0.6


@pytest.mark.parametrize("algo", ["kgw", "unigram"])
def test_detection_separates_watermarked_from_plain(algo):
    spec = WatermarkSpec(algo=algo, delta=2.0)
    rng = np.random.default_rng(1)
    wm = [detect_text(sample_text(150, spec, rng), encode, spec)["z"] for _ in range(12)]
    plain = [detect_text(sample_text(150, None, rng), encode, spec)["z"] for _ in range(40)]
    assert np.mean(wm) > 6 and min(wm) > spec.z_threshold
    # Unigram's single fixed green list covers slightly more or less than gamma of this tiny 2,000-word toy
    # vocabulary (about +-1%), which shifts plain-text z by a few tenths. Real 150k-token vocabularies don't.
    assert abs(np.mean(plain)) < 1.0 and max(plain) < spec.z_threshold


def test_wrong_key_does_not_detect():
    spec = WatermarkSpec(key=1)
    rng = np.random.default_rng(2)
    z = [detect_text(sample_text(150, spec, rng), encode, WatermarkSpec(key=999))["z"] for _ in range(10)]
    assert abs(np.mean(z)) < 1.0


def test_edge_cases_and_formula():
    spec = WatermarkSpec()
    assert detect_ids([], spec)["n_scored"] == 0 and detect_ids([5], spec)["z"] == 0.0
    assert z_score(75, 100, 0.5) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        green_mask(1, 2, WatermarkSpec(algo="nope"))


def test_torch_processor_matches_numpy():
    torch = pytest.importorskip("torch")
    from sira_lab.lab.watermark import make_logits_processor

    spec = WatermarkSpec(algo="kgw", delta=2.0)
    proc = make_logits_processor(spec)
    ids = torch.tensor([[5, 17, 99], [3, 4, 5]])
    scores = torch.zeros(2, V)
    out = proc(ids, scores)
    for b in range(2):
        expect = 2.0 * green_mask(np.int64(int(ids[b, -1])), np.arange(V, dtype=np.int64), spec)
        assert np.allclose(out[b].numpy(), expect)


def test_exp_detection_separates_and_is_calibrated():
    spec = WatermarkSpec(algo="exp")
    rng = np.random.default_rng(5)
    wm = [detect_text(sample_text(120, spec, rng, base_scale=2.0), encode, spec)["z"] for _ in range(10)]
    plain = [detect_text(sample_text(120, None, rng, base_scale=2.0), encode, spec)["z"] for _ in range(60)]
    assert min(wm) > 6                                      # strong signal
    assert abs(np.mean(plain)) < 0.5 and 0.6 < np.std(plain) < 1.5   # null is ~N(0,1): sum(-log(1-u)) ~ Gamma(n,1)


def test_exp_survives_light_edits_but_not_heavy_ones():
    spec = WatermarkSpec(algo="exp")
    rng = np.random.default_rng(6)
    words = sample_text(200, spec, rng, base_scale=2.0).split()
    def edit(frac):
        w = list(words)
        for i in rng.choice(len(w), int(frac * len(w)), replace=False):
            w[i] = "qrandomx"
        return " ".join(w)
    z0, z_light, z_heavy = (detect_text(t, encode, spec)["z"] for t in (" ".join(words), edit(0.1), edit(0.6)))
    assert z0 > z_light > z_heavy and z_heavy < z0 / 2


def test_synthid_keys_deterministic_and_detect_ids_refuses_synthid():
    from sira_lab.lab.watermark import synthid_keys
    a, b = WatermarkSpec(algo="synthid", key=1), WatermarkSpec(algo="synthid", key=2)
    assert synthid_keys(a) == synthid_keys(a) and synthid_keys(a) != synthid_keys(b) and len(synthid_keys(a)) == 9
    with pytest.raises(ValueError):
        detect_ids([1, 2, 3], a)


def test_torch_exp_processor_picks_the_same_token_as_numpy():
    torch = pytest.importorskip("torch")
    from sira_lab.lab.watermark import make_logits_processor, hash_bits
    spec = WatermarkSpec(algo="exp")
    proc = make_logits_processor(spec)
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(2, V, generator=g) * 2
    ids = torch.tensor([[5, 17, 99], [3, 4, 5]])
    out = proc(ids, scores)
    for b in range(2):
        p = torch.softmax(scores[b], -1).double().numpy()
        u = (hash_bits(np.int64(int(ids[b, -1])), np.arange(V, dtype=np.int64), spec).astype(np.float64) + 0.5) / 4294967296.0
        expect = int(np.argmax(np.log(u) / np.maximum(p, 1e-30)))
        assert int(torch.argmax(out[b])) == expect and torch.isinf(out[b]).sum() == V - 1

import numpy as np
import pytest

from sira_lab.lab.zeroshot_math import binoculars_np, fast_detectgpt_np, _log_softmax

L, V = 150, 400


def _logits(rng, scale=3.0):
    return rng.normal(size=(L, V)) * scale


def _sample(rng, logits, temperature=1.0):
    p = np.exp(_log_softmax(logits / temperature))
    return np.array([rng.choice(V, p=row) for row in p])


def test_fast_detectgpt_flags_low_temperature_text_not_random_text():
    rng = np.random.default_rng(0)
    machine, human = [], []
    for _ in range(25):
        lg = _logits(rng)
        machine.append(fast_detectgpt_np(lg, lg, _sample(rng, lg, temperature=0.7)))     # typical chatbot sampling
        human.append(fast_detectgpt_np(lg, lg, rng.integers(0, V, L)))                   # unrelated token choice
    assert np.mean(machine) > 1.0 and np.mean(human) < -1.0
    assert np.mean(machine) > np.mean(human) + 3


def test_fast_detectgpt_pure_sampling_is_near_zero():
    """Documents a real limitation: text sampled at T=1 straight from the model is NOT flagged."""
    rng = np.random.default_rng(1)
    z = []
    for _ in range(40):
        lg = _logits(rng)
        z.append(fast_detectgpt_np(lg, lg, _sample(rng, lg, 1.0)))
    assert abs(np.mean(z)) < 0.4


def test_binoculars_orientation_and_scale_invariance():
    rng = np.random.default_rng(2)
    machine, human = [], []
    for _ in range(25):
        obs = _logits(rng)
        perf = obs + rng.normal(size=(L, V)) * 0.5            # observer/performer are similar but not identical
        machine.append(binoculars_np(obs, perf, _sample(rng, perf, temperature=0.7)))
        human.append(binoculars_np(obs, perf, rng.integers(0, V, L)))
    assert np.mean(machine) > np.mean(human)                  # higher = more machine-like
    obs = _logits(rng); perf = obs + 0.3
    ids = rng.integers(0, V, L)
    assert binoculars_np(obs, perf, ids) == pytest.approx(binoculars_np(obs, perf + 5.0, ids))  # logit shifts cancel


def test_torch_matches_numpy_reference():
    torch = pytest.importorskip("torch")
    from sira_lab.lab.detectors import binoculars_torch, fast_detectgpt_torch
    rng = np.random.default_rng(3)
    lo, lp = _logits(rng), _logits(rng)
    ids = rng.integers(0, V, L)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    ti = torch.tensor(ids)
    assert fast_detectgpt_torch(t(lp), t(lo), ti) == pytest.approx(fast_detectgpt_np(lp, lo, ids), rel=1e-4)
    assert binoculars_torch(t(lo), t(lp), ti) == pytest.approx(binoculars_np(lo, lp, ids), rel=1e-4)

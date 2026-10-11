import numpy as np

from vtrack import LandmarkSmoother, OneEuroFilter


def test_constant_signal_unchanged():
    f = OneEuroFilter()
    for i in range(50):
        y = f(np.array([1.0, 2.0, 3.0]), i / 30)
    assert np.allclose(y, [1, 2, 3])


def test_reduces_jitter_when_still():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = 5.0 + rng.normal(0, 0.1, 300)
    out = np.array([float(f(x, i / 30)) for i, x in enumerate(raw)])
    assert out[100:].std() < 0.5 * raw[100:].std()


def test_follows_step_with_speed_adaptation():
    f = OneEuroFilter(min_cutoff=1.0, beta=5.0)
    for i in range(30):
        f(0.0, i / 30)
    y = [float(f(10.0, (30 + i) / 30)) for i in range(10)]
    assert y[-1] > 9.0


def test_smoother_resets_after_gap():
    s = LandmarkSmoother(max_gap_s=0.25)
    s(np.zeros((21, 3)), 0.0)
    s(np.zeros((21, 3)), 0.03)
    out = s(np.ones((21, 3)) * 7, 5.0)
    assert np.allclose(out, 7)

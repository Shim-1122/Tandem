import pytest

from vtrack import JointChannel, SlewLimiter


def test_channel_scaling_and_clamp():
    ch = JointChannel("elbow", 0, 140)
    assert ch.command(0) == 0.0
    assert ch.command(70) == pytest.approx(0.5)
    assert ch.command(200) == 1.0
    assert ch.command(-20) == 0.0


def test_channel_invert():
    assert JointChannel("x", 0, 100, invert=True).command(25) == pytest.approx(0.75)


def test_slew_limiter_rate():
    s = SlewLimiter(max_rate=2.0)
    s(0.0, 0.0)
    assert s(1.0, 0.1) == pytest.approx(0.2)
    assert s(1.0, 0.6) == pytest.approx(1.0)
    assert s(0.0, 0.7) == pytest.approx(0.8)

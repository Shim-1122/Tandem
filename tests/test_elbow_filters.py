import sys
from pathlib import Path

import numpy as np
import pytest

from vtrack.elbow import FOREARM_PER_SHOULDER_WIDTH, UPPER_ARM_PER_SHOULDER_WIDTH, ElbowPredictor, SegmentLengths
from vtrack.filters import AngleSmoother, JitterMeter, make_arm_smoothers
from vtrack.synthetic import arm_chain

L = (0.30, 0.26)
UP = np.array([0.0, 1.0, 0.0])


def rand_angles(rng):
    return dict(shoulder_flexion=rng.uniform(-30, 120), shoulder_abduction=rng.uniform(-20, 70),
                shoulder_rotation=rng.uniform(-60, 80), elbow_flexion=rng.uniform(20, 130),
                forearm_rotation=0.0, wrist_flexion=0.0, wrist_deviation=0.0)


def test_hidden_elbow_with_noisy_guess_beats_the_raw_guess():
    rng = np.random.default_rng(0)
    est_err, raw_err = [], []
    for _ in range(300):
        c = arm_chain(rand_angles(rng))
        g = c["E"] + rng.normal(0, 0.04, 3)
        e = ElbowPredictor().estimate(c["S"], c["W"], L, guess=g, guess_vis=0.2, up=UP)
        assert e.mode == "predicted"
        assert np.linalg.norm(e.elbow - c["S"]) == pytest.approx(L[0], abs=1e-6)   # segment lengths kept
        assert np.linalg.norm(e.elbow - c["W"]) == pytest.approx(L[1], abs=1e-6)
        est_err.append(np.linalg.norm(e.elbow - c["E"]))
        raw_err.append(np.linalg.norm(g - c["E"]))
    assert np.median(est_err) < 0.5 * np.median(raw_err)


def test_continuity_holds_elbow_while_hidden_and_bending():
    rng = np.random.default_rng(1)
    for _ in range(30):
        a = rand_angles(rng)
        p = ElbowPredictor()
        c = arm_chain(a)
        p.estimate(c["S"], c["W"], L, guess=c["E"], guess_vis=1.0, t=0.0)
        for k in range(1, 31):
            c = arm_chain(dict(a, elbow_flexion=float(np.clip(a["elbow_flexion"] + 2 * k, 5, 150))))
            e = p.estimate(c["S"], c["W"], L, guess=None, up=UP, t=k / 30)
        assert np.linalg.norm(e.elbow - c["E"]) < 0.01


def test_gravity_prior_alone_picks_the_hanging_elbow():
    c = arm_chain(dict(shoulder_flexion=0, shoulder_abduction=0, shoulder_rotation=0, elbow_flexion=90,
                       forearm_rotation=0, wrist_flexion=0, wrist_deviation=0))
    e = ElbowPredictor().estimate(c["S"], c["W"], L, up=UP)
    assert np.linalg.norm(e.elbow - c["E"]) < 1e-3


def test_visible_elbow_is_used_as_measured_and_straight_arm_lies_on_the_line():
    S, W = np.zeros(3), np.array([0.0, -0.6, 0.0])
    g = np.array([0.01, -0.29, 0.02])
    assert np.allclose(ElbowPredictor().estimate(S, W, L, guess=g, guess_vis=0.9).elbow, g)
    e = ElbowPredictor().estimate(S, W, L, guess=g, guess_vis=0.1)
    assert e.mode == "straight" and np.allclose(e.elbow, [0, -0.30, 0])


def test_segment_lengths_fallback_then_median():
    s = SegmentLengths(min_samples=5)
    assert s.get(0.36) == pytest.approx((UPPER_ARM_PER_SHOULDER_WIDTH * 0.36, FOREARM_PER_SHOULDER_WIDTH * 0.36))
    for d in (0.29, 0.30, 0.31, 0.30, 0.90):              # one outlier frame
        s.add(np.zeros(3), np.array([0, -d, 0]), np.array([0, -d - 0.26, 0]))
    assert s.calibrated and s.get(0.36) == pytest.approx((0.30, 0.26))


def test_angle_smoother_unwraps_across_180_and_resets_on_nan():
    f = AngleSmoother(min_cutoff=0.5, beta=0.0)
    out = [f({"a": v}, i / 30)["a"] for i, v in enumerate([170, 176, -178, -172, -175])]
    assert all(abs(o) > 160 for o in out)                   # never swings through 0
    assert np.isnan(f({"a": float("nan")}, 0.3)["a"])
    assert f({"a": 10.0}, 0.4)["a"] == pytest.approx(10.0)  # restarted cleanly


def test_jitter_meter():
    j = JitterMeter(window_s=1.0)
    for i in range(30):
        j.add({"x": 5.0 + (1.0 if i % 2 else -1.0)}, i / 30)
    assert j.std()["x"] == pytest.approx(1.0, abs=0.05)


def test_balanced_preset_reduces_jitter_and_tracks_motion_in_simulation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_stability as ev

    class Args:
        xy_cm, z_cm, hand_cm = 0.6, 2.0, 0.2
    raw_j = np.mean(list(ev.run(ev.PIPELINES["raw"], False, Args, seconds=4)[0].values()))
    prev_j = np.mean(list(ev.run(ev.PIPELINES["previous"], False, Args, seconds=4)[0].values()))
    bal_j = np.mean(list(ev.run(ev.PIPELINES["balanced"], False, Args, seconds=4)[0].values()))
    prev_m = np.mean(list(ev.run(ev.PIPELINES["previous"], True, Args, seconds=4)[1].values()))
    bal_m = np.mean(list(ev.run(ev.PIPELINES["balanced"], True, Args, seconds=4)[1].values()))
    assert bal_j < 0.4 * raw_j and bal_j < prev_j and bal_m < prev_m


def test_make_arm_smoothers_presets():
    for name in ("responsive", "balanced", "stable"):
        ps, ang = make_arm_smoothers(name)
        out = ps(np.ones((33, 3)), 0.0)
        assert out.shape == (33, 3)

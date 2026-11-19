"""The Rust core (native/, tandem_core) gives the same numbers as vtrack's Python code: filters and the hand, arm,
leg and core solvers, over random motion including the singular cases (straight elbow or knee, limbs along the
lateral axis, feet out of view, filter resets, angles wrapping past 180, NaN and None)."""
import math

import numpy as np
import pytest

from vtrack import native
from vtrack.filters import AngleSmoother, LandmarkSmoother, OneEuroFilter, SMOOTHING_PRESETS
from vtrack.fingers import FINGERS4, HandKinematics, hand_fk
from vtrack.kinematics import ArmKinematics
from vtrack.legs import CAMERA_UP, LegKinematics, core_angles
from vtrack.synthetic import forward_arm, forward_body

pytestmark = pytest.mark.skipif(not native.AVAILABLE, reason="tandem_core is not installed (pip install ./native)")
TOL = 1e-9


def same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
        return
    if isinstance(a, (str, bool)):
        assert a == b, (a, b)
        return
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a, b, atol=TOL, rtol=0, equal_nan=True), np.max(np.abs(a - b))


def times(rng, n):
    t = np.cumsum(rng.uniform(0.002, 0.05, n))
    t[n // 3] = t[n // 3 - 1]                 # a repeated timestamp resets the filter
    return t


def test_one_euro_arrays_scalars_and_per_axis_cutoffs():
    rng = np.random.default_rng(1)
    for shape, kw in (((7,), dict(min_cutoff=0.3, beta=0.05)), ((33, 3), SMOOTHING_PRESETS["balanced"]["pose"]),
                      ((21, 3), dict(min_cutoff=2.0, beta=5.0)), ((), dict(min_cutoff=1.0, beta=0.3))):
        py, rs = OneEuroFilter(native=False, **kw), OneEuroFilter(native=True, **kw)
        for i, t in enumerate(times(rng, 200)):
            x = rng.normal(size=shape).cumsum() if shape == () else rng.normal(size=shape)
            same(py(x, t), rs(x, t))
            if i == 120:
                py.reset()
                rs.reset()


def test_landmark_smoother_resets_after_a_gap():
    rng = np.random.default_rng(2)
    py, rs = LandmarkSmoother(native=False, min_cutoff=2.0, beta=5.0), LandmarkSmoother(native=True, min_cutoff=2.0, beta=5.0)
    t = np.cumsum(rng.uniform(0.005, 0.03, 100))
    t[50:] += 1.0                              # a dropout longer than max_gap_s
    for ti in t:
        x = rng.normal(size=(21, 3))
        same(py(x, ti), rs(x, ti))


def test_angle_smoother_unwraps_and_passes_nan():
    rng = np.random.default_rng(3)
    py, rs = AngleSmoother(native=False, min_cutoff=0.3, beta=0.05), AngleSmoother(native=True, min_cutoff=0.3, beta=0.05)
    yaw = 150.0
    for i, t in enumerate(times(rng, 300)):
        yaw = (yaw + rng.normal(scale=8) + 180) % 360 - 180          # wraps past +-180
        vals = {"yaw": yaw, "pitch": float(rng.normal(scale=30)), "roll": np.float64(rng.normal()),
                "gone": float("nan") if i % 7 == 0 else float(rng.normal()), "none": None if i % 5 else 1.0}
        a, b = py(vals, t), rs(vals, t)
        assert list(a) == list(b) and a["none"] == b["none"]
        same({k: v for k, v in a.items() if k != "none"}, {k: v for k, v in b.items() if k != "none"})
        if i == 200:
            py.reset()
            rs.reset()


def random_arm(rng):
    return dict(shoulder_flexion=rng.uniform(-60, 175), shoulder_abduction=rng.uniform(-30, 89),
                shoulder_rotation=rng.uniform(-80, 80), elbow_flexion=rng.choice([rng.uniform(0, 12), rng.uniform(15, 150)]),
                forearm_rotation=rng.uniform(-90, 90), wrist_flexion=rng.uniform(-70, 80), wrist_deviation=rng.uniform(-30, 20))


@pytest.mark.parametrize("side", ["right", "left"])
def test_arm_solver(side):
    rng = np.random.default_rng(4)
    py, rs = ArmKinematics(side, native=False), ArmKinematics(side, native=True)
    for i in range(400):
        a = random_arm(rng)
        if i % 9 == 0:
            a.update(shoulder_flexion=0.0, shoulder_abduction=90.0)               # along the lateral axis
        pose, hand = forward_arm(a, side)
        pose = pose + rng.normal(scale=0.003, size=pose.shape)
        up = None if i % 4 else np.array([0.0, -1.0, 0.0]) + rng.normal(scale=0.05, size=3)
        h = hand if i % 3 else None
        same(py.solve(pose, h, up).as_dict(), rs.solve(pose, h, up).as_dict())
        if i == 300:
            py.reset()
            rs.reset()


def random_legs(rng):
    def leg():
        return dict(hip_flexion=rng.uniform(-25, 120), hip_abduction=rng.uniform(-25, 45),
                    hip_rotation=rng.uniform(-45, 45), knee_flexion=rng.choice([rng.uniform(0, 15), rng.uniform(20, 140)]),
                    ankle_dorsiflexion=rng.uniform(-45, 25), ankle_inversion=rng.uniform(-20, 30),
                    ankle_abduction=rng.uniform(-25, 25))
    return {"right": leg(), "left": leg()}


@pytest.mark.parametrize("side", ["right", "left"])
def test_leg_solver_and_core(side):
    rng = np.random.default_rng(5)
    py, rs = LegKinematics(side, native=False), LegKinematics(side, native=True)
    for i in range(400):
        legs = random_legs(rng)
        if i % 11 == 0:
            legs[side].update(hip_flexion=0.0, hip_abduction=89.0)
        core = dict(core_flexion=rng.uniform(-20, 60), core_lateral_bend=rng.uniform(-25, 25),
                    core_rotation=rng.uniform(-40, 40))
        pose = forward_body(legs, core, rng.uniform(-10, 10))
        pose = pose + rng.normal(scale=0.003, size=pose.shape)
        up = CAMERA_UP if i % 5 else CAMERA_UP + rng.normal(scale=0.05, size=3)
        foot = bool(i % 4)
        if i == 150:
            for k in (py, rs):
                k.calibrate(pose, up)
        same(py.solve(pose, up, foot).as_dict(), rs.solve(pose, up, foot).as_dict())
        same(vars(core_angles(pose, up, native=False)), vars(core_angles(pose, up, native=True)))
        if i == 300:
            py.reset()
            rs.reset()


@pytest.mark.parametrize("side", ["right", "left"])
def test_hand_solver(side):
    from tests.test_gestures import GEOMETRY
    rng = np.random.default_rng(6)
    py, rs = HandKinematics(side, native=False), HandKinematics(side, native=True)
    for i in range(400):
        a = {f: dict(mcp_flex=rng.uniform(-20, 90), mcp_abd=rng.uniform(-20, 20), pip=rng.uniform(0, 100),
                     dip=rng.uniform(0, 80)) for f in FINGERS4}
        straight = i % 6 == 0                                                 # thumb rotation unobservable
        a["thumb"] = dict(cmc_palmar=rng.uniform(-10, 60), cmc_radial=rng.uniform(-10, 50),
                          cmc_rotation=rng.uniform(-80, 80), mcp=0.0 if straight else rng.uniform(0, 60),
                          ip=0.0 if straight else rng.uniform(0, 70))
        P = hand_fk(a, GEOMETRY)
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        w = (P @ R.T + rng.normal(scale=0.002, size=P.shape)) * ([-1, 1, 1] if side == "left" else 1)
        same(py.solve(w), rs.solve(w))


def test_native_is_faster():
    """Not a correctness check: the point of the Rust core."""
    import time
    pose, hand = forward_arm(random_arm(np.random.default_rng(7)), "right")
    out = {}
    for flag in (False, True):
        k = ArmKinematics("right", native=flag)
        t0 = time.perf_counter()
        for _ in range(300):
            k.solve(pose, hand)
        out[flag] = time.perf_counter() - t0
    assert out[True] < out[False] / 3, out
    assert math.isfinite(out[True])

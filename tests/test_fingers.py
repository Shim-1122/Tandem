import numpy as np
import pytest

from tests.conftest import make_hand
from vtrack.fingers import (FINGER_KEYS, THUMB_KEYS, HandKinematics, flat, hand_fk, hand_geometry,
                            mirror_if_left, palm_to_world)
from vtrack.kinematics import rot_x, rot_y, rot_z

# A right hand in palm coordinates (x thumb side, y toward the wrist, z back of hand), meters.
GEOMETRY = {
    "index": dict(base=np.array([0.022, -0.088, 0.0]), lengths=[0.045, 0.026, 0.021]),
    "middle": dict(base=np.array([0.0, -0.090, 0.0]), lengths=[0.050, 0.030, 0.022]),
    "ring": dict(base=np.array([-0.018, -0.085, 0.0]), lengths=[0.046, 0.028, 0.021]),
    "pinky": dict(base=np.array([-0.034, -0.075, 0.0]), lengths=[0.036, 0.020, 0.019]),
    "thumb": dict(base=np.array([0.025, -0.025, -0.010]), lengths=[0.045, 0.031, 0.026]),
}


def random_angles(rng):
    a = {n: dict(mcp_flex=rng.uniform(-20, 90), mcp_abd=rng.uniform(-20, 20), pip=rng.uniform(0, 100),
                 dip=rng.uniform(0, 80)) for n in ("index", "middle", "ring", "pinky")}
    a["thumb"] = dict(cmc_palmar=rng.uniform(0, 60), cmc_radial=rng.uniform(0, 60),
                      cmc_rotation=rng.uniform(-60, 60), mcp=rng.uniform(10, 60), ip=rng.uniform(10, 80))
    return a


def to_world(P, rng, side):
    R = rot_z(rng.uniform(-3, 3)) @ rot_y(rng.uniform(-3, 3)) @ rot_x(rng.uniform(-3, 3))
    return mirror_if_left(P @ R.T + rng.normal(scale=0.1, size=3), side)


@pytest.mark.parametrize("side", ["right", "left"])
def test_round_trip_all_finger_angles(side):
    rng = np.random.default_rng(11)
    for _ in range(300):
        truth = random_angles(rng)
        world = to_world(hand_fk(truth, GEOMETRY), rng, side)
        out = HandKinematics(side).solve(world)
        for name, keys in [(n, FINGER_KEYS) for n in ("index", "middle", "ring", "pinky")] + [("thumb", THUMB_KEYS)]:
            for k in keys:
                assert out[name][k] == pytest.approx(truth[name][k], abs=1e-6), (side, name, k)


def test_fk_of_solved_angles_reproduces_the_landmarks():
    rng = np.random.default_rng(12)
    truth = random_angles(rng)
    world = to_world(hand_fk(truth, GEOMETRY), rng, "left")
    kin = HandKinematics("left")
    angles = kin.solve(world)
    rebuilt = palm_to_world(hand_fk(angles, hand_geometry(world, "left")), world, "left")
    assert np.allclose(rebuilt, world, atol=1e-9)


def test_open_hand_and_fist_from_simple_test_hand():
    open_ = HandKinematics("right").solve(make_hand(extended=("index", "middle", "ring", "pinky"), thumb=True))
    fist = HandKinematics("right").solve(make_hand())
    for f in ("index", "middle", "ring", "pinky"):
        assert open_[f]["pip"] == pytest.approx(0, abs=1e-6)
        assert fist[f]["pip"] == pytest.approx(90, abs=1e-6)


def test_signs_flexion_toward_palm_and_abduction_toward_thumb():
    base = {n: dict(mcp_flex=0.0, mcp_abd=0.0, pip=0.0, dip=0.0) for n in ("index", "middle", "ring", "pinky")}
    base["thumb"] = dict(cmc_palmar=10.0, cmc_radial=30.0, cmc_rotation=0.0, mcp=20.0, ip=20.0)
    P = hand_fk(base, GEOMETRY)
    bent = dict(base, index=dict(mcp_flex=40.0, mcp_abd=15.0, pip=0.0, dip=0.0))
    Q = hand_fk(bent, GEOMETRY)
    assert Q[6][2] < P[6][2]          # flexed toward the palm: z (back of hand) decreases
    assert Q[6][0] > P[6][0] - 1e-9   # abducted toward the thumb side: x increases


def test_thumb_rotation_held_while_thumb_is_straight():
    rng = np.random.default_rng(13)
    kin = HandKinematics("right")
    a = random_angles(rng)
    a["thumb"].update(cmc_rotation=35.0, mcp=30.0, ip=30.0)
    kin.solve(hand_fk(a, GEOMETRY))
    a["thumb"].update(cmc_rotation=-50.0, mcp=0.0, ip=0.0)        # straight thumb: rotation not visible
    out = kin.solve(hand_fk(a, GEOMETRY))
    assert out["thumb"]["cmc_rotation"] == pytest.approx(35.0, abs=1e-6)


def test_flat_names():
    out = flat(HandKinematics("right").solve(make_hand()))
    assert "index_mcp_flex" in out and "thumb_cmc_rotation" in out and len(out) == 21

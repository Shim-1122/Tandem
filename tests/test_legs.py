import numpy as np
import pytest

from vtrack.head import FACING_CAMERA
from vtrack.kinematics import rot_x, rot_y
from vtrack.legs import (CAMERA_UP, CORE_FIELDS, LEG, LegAngles, LegKinematics, core_angles, core_visible,
                         leg_out_of_range, legs_visible, pelvis_frame, up_from_standing)
from vtrack.synthetic import forward_body, leg_chain

JOINT = ["hip_flexion", "hip_abduction", "hip_rotation", "knee_flexion", "ankle_dorsiflexion", "ankle_inversion",
         "ankle_abduction"]
OFFSET = (0.1, 0.2, 2.5)


def random_leg(rng, knee=(25, 130), ankle_abd=(-25, 25)):
    return dict(hip_flexion=rng.uniform(-20, 100), hip_abduction=rng.uniform(-20, 40),
                hip_rotation=rng.uniform(-40, 40), knee_flexion=rng.uniform(*knee),
                ankle_dorsiflexion=rng.uniform(-40, 25), ankle_inversion=rng.uniform(-20, 30),
                ankle_abduction=rng.uniform(*ankle_abd))


def level(yaw_deg=0.0, camera_pitch_deg=0.0):
    """Level body frame for a person facing the camera, turned by yaw; the camera may be pitched."""
    return rot_x(np.radians(camera_pitch_deg)) @ FACING_CAMERA @ rot_y(np.radians(yaw_deg))


def foot_attitude(R_foot_level):
    """Expected foot pitch and roll to the floor from the foot frame in level axes (Y up)."""
    xf, yf, zf = R_foot_level.T
    U = np.array([0.0, 1.0, 0.0])
    return np.degrees(np.arcsin(xf @ U)), np.degrees(np.arctan2(-(zf @ U), yf @ U))


def test_round_trip_both_legs_and_core_with_any_heading_and_hike():
    rng = np.random.default_rng(0)
    for _ in range(300):
        legs = {sd: random_leg(rng) for sd in ("left", "right")}
        core = dict(core_flexion=rng.uniform(-20, 60), core_lateral_bend=rng.uniform(-30, 30),
                    core_rotation=rng.uniform(-40, 40))
        hike, R_level = rng.uniform(-10, 10), level(rng.uniform(-60, 60))
        pose = forward_body(legs, core, hike, R_level, OFFSET)
        up = R_level[:, 1]
        for sd in ("left", "right"):
            a = LegKinematics(sd).solve(pose, up)
            assert a.hip_rotation_source == "knee" and a.ankle_valid
            for k in JOINT:
                assert getattr(a, k) == pytest.approx(legs[sd][k], abs=1e-6), k
            assert a.pelvic_hike == pytest.approx(hike if sd == "right" else -hike, abs=1e-6)
        c = core_angles(pose, up)
        for k in CORE_FIELDS:
            assert getattr(c, k) == pytest.approx(core[k], abs=1e-6), k


def test_foot_pitch_and_roll_are_the_foot_attitude_to_the_floor():
    rng = np.random.default_rng(1)
    for _ in range(200):
        legs = {sd: random_leg(rng) for sd in ("left", "right")}
        hike = rng.uniform(-8, 8)
        pose = forward_body(legs, None, hike, level(), OFFSET)
        for sd in ("left", "right"):
            a = LegKinematics(sd).solve(pose, CAMERA_UP)
            # expected: foot frame in the (mirrored) level frame, including the pelvic hike
            # the pelvis is the level frame turned about the forward axis; raising this side is a negative turn
            R_tilt = rot_x(-np.radians(hike if sd == "right" else -hike))
            pitch, roll = foot_attitude(R_tilt @ leg_chain(legs[sd])["R_foot"])
            assert (a.foot_pitch, a.foot_roll) == pytest.approx((pitch, roll), abs=1e-6)


def test_standing_reads_zero_and_foot_signs():
    zero = {k: 0.0 for k in JOINT}
    pose = forward_body({"left": zero, "right": zero}, None, 0.0, level(), OFFSET)
    for sd in ("left", "right"):
        a = LegKinematics(sd).solve(pose, CAMERA_UP)
        for k in JOINT + ["pelvic_hike", "foot_pitch", "foot_roll"]:
            assert getattr(a, k) == pytest.approx(0.0, abs=1e-6), (sd, k)
    toes_up = dict(zero, ankle_dorsiflexion=15.0)
    inverted = dict(zero, ankle_inversion=12.0)
    pose = forward_body({"left": toes_up, "right": inverted}, None, 0.0, level(), OFFSET)
    assert LegKinematics("left").solve(pose, CAMERA_UP).foot_pitch == pytest.approx(15.0, abs=1e-6)
    assert LegKinematics("right").solve(pose, CAMERA_UP).foot_roll == pytest.approx(12.0, abs=1e-6)


def test_straight_knee_takes_hip_rotation_from_the_foot():
    rng = np.random.default_rng(2)
    for _ in range(200):
        legs = {sd: random_leg(rng, knee=(0, 15), ankle_abd=(0, 0)) for sd in ("left", "right")}
        pose = forward_body(legs, None, rng.uniform(-5, 5), level(rng.uniform(-40, 40)), OFFSET)
        up = level(0)[:, 1]
        for sd in ("left", "right"):
            a = LegKinematics(sd).solve(pose, up)
            assert a.hip_rotation_source == "foot"
            for k in JOINT:
                assert getattr(a, k) == pytest.approx(legs[sd][k], abs=1e-6), k


def test_straight_knee_without_the_foot_holds_hip_rotation():
    rng = np.random.default_rng(3)
    bent = random_leg(rng)
    straight = dict(bent, knee_flexion=5.0, hip_rotation=bent["hip_rotation"] + 20)
    k = LegKinematics("right")
    a1 = k.solve(forward_body({"right": bent}, None, 0.0, level(), OFFSET), CAMERA_UP)
    a2 = k.solve(forward_body({"right": straight}, None, 0.0, level(), OFFSET), CAMERA_UP, foot=False)
    assert a2.hip_rotation_source == "held" and a2.hip_rotation == pytest.approx(a1.hip_rotation)
    assert not a2.ankle_valid and np.isnan(a2.ankle_dorsiflexion) and np.isnan(a2.foot_pitch)
    assert a2.hip_flexion == pytest.approx(straight["hip_flexion"], abs=1e-6)


def test_signs_as_seen_by_the_camera():
    """A person facing the camera: their right is the camera's -x, up is -y, toward the camera is -z."""
    zero = {k: 0.0 for k in JOINT}

    def body(right=None, core=None, hike=0.0):
        return forward_body({"left": zero, "right": dict(zero, **(right or {}))}, core, hike, level(), OFFSET)

    base = body()
    assert body({"hip_flexion": 30})[26, 2] < base[26, 2] - 0.1           # knee comes toward the camera
    assert body({"hip_abduction": 20})[26, 0] < base[26, 0] - 0.1         # knee goes out to your right
    assert body({"knee_flexion": 60})[28, 2] > base[28, 2] + 0.1          # ankle goes behind you
    assert body({"ankle_dorsiflexion": 20})[32, 1] < base[32, 1] - 0.02   # toes rise
    assert body({"ankle_abduction": 20})[32, 0] < base[32, 0] - 0.02      # toes turn out to your right
    assert body(hike=8)[24, 1] < base[24, 1] - 0.01                       # right hip rises
    assert body(core={"core_flexion": 30})[11, 2] < base[11, 2] - 0.1     # shoulders come forward
    assert body(core={"core_lateral_bend": 20})[11, 0] < base[11, 0] - 0.1    # lean to your right
    assert body(core={"core_rotation": 30})[12, 2] < base[12, 2] - 0.05   # turn left: right shoulder forward
    a = LegKinematics("right").solve(body({"hip_rotation": 25, "knee_flexion": 90}), CAMERA_UP)
    assert a.hip_rotation == pytest.approx(25.0, abs=1e-6)


def test_foot_calibration_removes_landmark_offsets():
    zero = {k: 0.0 for k in JOINT}
    # MediaPipe's heel point sits higher than its toe point on a flat foot: about 7 degrees of fake plantarflexion
    lengths = dict(heel=(-0.06, -0.05, 0.0), toe=(0.17, -0.08, 0.0))
    stand = forward_body({"left": zero, "right": zero}, None, 0.0, level(), OFFSET, **lengths)
    k = LegKinematics("right")
    assert k.solve(stand, CAMERA_UP).ankle_dorsiflexion < -5
    assert k.calibrate(stand, CAMERA_UP)
    a = k.solve(stand, CAMERA_UP)
    assert (a.ankle_dorsiflexion, a.ankle_inversion, a.foot_pitch, a.foot_roll) == pytest.approx((0, 0, 0, 0), abs=1e-6)
    moved = dict(zero, knee_flexion=40, ankle_dorsiflexion=15, ankle_inversion=10, ankle_abduction=8)
    a = k.solve(forward_body({"left": zero, "right": moved}, None, 0.0, level(), OFFSET, **lengths), CAMERA_UP)
    assert (a.ankle_dorsiflexion, a.ankle_inversion, a.ankle_abduction) == pytest.approx((15, 10, 8), abs=1e-6)
    # a bent knee is not a standing pose: refused, previous calibration kept
    C = k.foot_correction.copy()
    assert not k.calibrate(forward_body({"left": zero, "right": moved}, None, 0.0, level(), OFFSET), CAMERA_UP)
    assert np.allclose(k.foot_correction, C)
    k.clear_calibration()
    assert np.allclose(k.foot_correction, np.eye(3))


def test_standing_up_direction_fixes_a_tilted_camera():
    zero = {k: 0.0 for k in JOINT}
    R_level = level(camera_pitch_deg=20)                # camera tilted 20 degrees
    pose = forward_body({"left": zero, "right": dict(zero, hip_flexion=30)}, None, 0.0, R_level, OFFSET)
    standing = forward_body({"left": zero, "right": zero}, None, 0.0, R_level, OFFSET)
    wrong = LegKinematics("right").solve(pose, CAMERA_UP).hip_flexion
    up = up_from_standing(standing)
    assert abs(wrong - 30) > 15
    assert LegKinematics("right").solve(pose, up).hip_flexion == pytest.approx(30, abs=1e-6)
    assert core_angles(pose, up).core_flexion == pytest.approx(0, abs=1e-6)


def test_pelvis_frame_and_visibility_helpers():
    pose = forward_body({}, None, 6.0, level(), OFFSET)
    R_p, mid, obliquity = pelvis_frame(pose, CAMERA_UP)
    assert np.allclose(R_p.T @ R_p, np.eye(3)) and np.degrees(obliquity) == pytest.approx(6.0)
    vis = np.ones(33)
    assert legs_visible(vis, "right") == ([], True) and core_visible(vis)
    vis[LEG["right"][1]] = 0.2                           # right knee hidden
    vis[LEG["right"][4]] = 0.3                           # right toe hidden
    assert legs_visible(vis, "right") == (["knee"], False)
    vis[23] = 0.1                                        # left hip hidden: the pelvis is unknown for both legs
    assert legs_visible(vis, "right")[0] == ["hip", "knee"] and not core_visible(vis)
    assert legs_visible(None, "left") == (["hip", "knee", "ankle"], False)


def test_out_of_range_flags():
    assert leg_out_of_range(LegAngles(knee_flexion=90, hip_flexion=40)) == []
    assert leg_out_of_range(LegAngles(knee_flexion=-40, ankle_inversion=80)) == ["knee_flexion", "ankle_inversion"]

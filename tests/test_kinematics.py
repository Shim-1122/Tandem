import numpy as np
import pytest

from vtrack.geometry import L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST
from vtrack.kinematics import ArmKinematics, euler_yxz, rot_x, rot_y, rot_z, shortest_arc
from vtrack.synthetic import forward_arm

KEYS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion",
        "forearm_rotation", "wrist_flexion", "wrist_deviation"]


def torso_pose(elbow, wrist):
    """Right arm in thorax axes (X forward, Y up, Z right), identity body rotation."""
    p = np.zeros((33, 3))
    p[R_SHOULDER], p[L_SHOULDER] = (0, 0, 0.2), (0, 0, -0.2)
    p[R_HIP], p[L_HIP] = (0, -0.5, 0.1), (0, -0.5, -0.1)
    p[R_ELBOW], p[R_WRIST] = elbow, wrist
    return p


def hand_pts(mid, thumb_dir):
    """Hand landmarks with wrist at origin, middle knuckle at `mid`, index/pinky 4 cm along +-thumb_dir."""
    h = np.zeros((21, 3))
    h[9] = mid
    h[5] = np.asarray(mid) + 0.04 * np.asarray(thumb_dir)
    h[17] = np.asarray(mid) - 0.04 * np.asarray(thumb_dir)
    return h


def solve_right(pose, hand):
    return ArmKinematics("right").solve(pose, hand)


# ---- independent anatomical checks, no forward kinematics involved ----

NEUTRAL_E, NEUTRAL_W = (0, -0.3, 0.2), (0.26, -0.3, 0.2)   # arm hanging, elbow 90 degrees, forearm forward


def test_neutral_pose_reads_zero_shoulder_and_wrist():
    a = solve_right(torso_pose(NEUTRAL_E, NEUTRAL_W), hand_pts((0.09, 0, 0), (0, 1, 0)))
    assert a.elbow_flexion == pytest.approx(90)
    for k in ("shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "forearm_rotation",
              "wrist_flexion", "wrist_deviation"):
        assert getattr(a, k) == pytest.approx(0, abs=1e-6), k


def test_forearm_across_body_is_internal_rotation():
    # forearm points to the left (toward the midline), thumb still up
    a = solve_right(torso_pose(NEUTRAL_E, (0, -0.3, -0.06)), hand_pts((0, 0, -0.09), (0, 1, 0)))
    assert a.shoulder_rotation == pytest.approx(90, abs=1e-6)
    assert a.forearm_rotation == pytest.approx(0, abs=1e-6)


def test_forearm_pointing_outward_is_external_rotation():
    a = solve_right(torso_pose(NEUTRAL_E, (0, -0.3, 0.46)), hand_pts((0, 0, 0.09), (0, 1, 0)))
    assert a.shoulder_rotation == pytest.approx(-90, abs=1e-6)


def test_pronation_positive_when_thumb_moves_toward_midline():
    a = solve_right(torso_pose(NEUTRAL_E, NEUTRAL_W), hand_pts((0.09, 0, 0), (0, 0, -1)))
    assert a.forearm_rotation == pytest.approx(90, abs=1e-6)   # palm faces down
    s = solve_right(torso_pose(NEUTRAL_E, NEUTRAL_W), hand_pts((0.09, 0, 0), (0, 0, 1)))
    assert s.forearm_rotation == pytest.approx(-90, abs=1e-6)  # palm faces up


def test_wrist_flexion_positive_toward_palm_side():
    t = np.radians(30)
    a = solve_right(torso_pose(NEUTRAL_E, NEUTRAL_W), hand_pts((0.09 * np.cos(t), 0, -0.09 * np.sin(t)), (0, 1, 0)))
    assert a.wrist_flexion == pytest.approx(30, abs=1e-6)
    assert a.wrist_deviation == pytest.approx(0, abs=1e-6)


def test_radial_deviation_positive_toward_thumb_side():
    t = np.radians(30)
    mid = (0.09 * np.cos(t), 0.09 * np.sin(t), 0)
    a = solve_right(torso_pose(NEUTRAL_E, NEUTRAL_W), hand_pts(mid, (-np.sin(t), np.cos(t), 0)))
    assert a.wrist_deviation == pytest.approx(30, abs=1e-6)
    assert a.wrist_flexion == pytest.approx(0, abs=1e-6)


def test_arm_raised_forward_and_sideways_swing_angles():
    fwd = solve_right(torso_pose((0.3, 0, 0.2), (0.56, 0, 0.2)), None)
    assert fwd.shoulder_flexion == pytest.approx(90) and fwd.shoulder_abduction == pytest.approx(0, abs=1e-6)
    side = solve_right(torso_pose((0, 0, 0.5), (0, 0, 0.76)), None)
    assert side.shoulder_abduction == pytest.approx(90)


# ---- round trips through forward kinematics ----

def random_angles(rng, elbow_min=30):
    """Random pose away from the lateral-axis pole (abduction within +-80 degrees)."""
    return dict(shoulder_flexion=rng.uniform(-60, 170), shoulder_abduction=rng.uniform(-30, 80),
                shoulder_rotation=rng.uniform(-70, 90), elbow_flexion=rng.uniform(elbow_min, 140),
                forearm_rotation=rng.uniform(-80, 80), wrist_flexion=rng.uniform(-70, 70),
                wrist_deviation=rng.uniform(-25, 35))


def random_body(rng):
    return rot_y(rng.uniform(-1, 1)) @ rot_x(rng.uniform(-0.5, 0.5)) @ rot_z(rng.uniform(-0.5, 0.5))


@pytest.mark.parametrize("side", ["right", "left"])
def test_round_trip_random_poses_with_hand_landmarks(side):
    rng = np.random.default_rng(1)
    for _ in range(300):
        ang = random_angles(rng)
        pose, hand = forward_arm(ang, side, body_rotation=random_body(rng), body_offset=rng.normal(size=3))
        out = ArmKinematics(side).solve(pose, hand)
        assert out.shoulder_rotation_valid and out.wrist_valid
        for k in KEYS:
            assert getattr(out, k) == pytest.approx(ang[k], abs=1e-6), (k, ang)


@pytest.mark.parametrize("side", ["right", "left"])
def test_round_trip_pose_only_hand(side):
    rng = np.random.default_rng(2)
    for _ in range(100):
        ang = random_angles(rng)
        pose, _ = forward_arm(ang, side, body_rotation=random_body(rng))
        out = ArmKinematics(side).solve(pose, None)
        for k in KEYS:
            assert getattr(out, k) == pytest.approx(ang[k], abs=1e-6), (k, ang)


def test_left_arm_mirrors_right_arm():
    ang = dict(shoulder_flexion=35, shoulder_abduction=40, shoulder_rotation=-20, elbow_flexion=80,
               forearm_rotation=30, wrist_flexion=-15, wrist_deviation=12)
    r = ArmKinematics("right").solve(*forward_arm(ang, "right"))
    l = ArmKinematics("left").solve(*forward_arm(ang, "left"))
    for k in KEYS:
        assert getattr(r, k) == pytest.approx(getattr(l, k), abs=1e-6)


# ---- observability limits ----

BENT = dict(shoulder_flexion=20, shoulder_abduction=10, shoulder_rotation=25, elbow_flexion=70,
            forearm_rotation=15, wrist_flexion=10, wrist_deviation=5)


def test_straight_elbow_holds_shoulder_rotation_and_flags_it():
    k = ArmKinematics("right")
    k.solve(*forward_arm(BENT))
    straight = dict(BENT, elbow_flexion=0.0, shoulder_rotation=60)   # shoulder rotation really changed
    out = k.solve(*forward_arm(straight))
    assert out.shoulder_rotation_valid is False
    assert out.shoulder_rotation == pytest.approx(25)                 # held
    assert out.wrist_valid is True


def test_total_twist_is_preserved_when_elbow_is_straight():
    k = ArmKinematics("right")
    k.solve(*forward_arm(BENT))
    straight = dict(BENT, elbow_flexion=0.0, shoulder_rotation=60, forearm_rotation=15)
    out = k.solve(*forward_arm(straight))
    # the unseen 35 degrees of extra shoulder rotation shows up as forearm pronation instead
    assert out.shoulder_rotation + out.forearm_rotation == pytest.approx(60 + 15, abs=1e-6)


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def test_arm_overhead_is_regular_not_singular():
    ang = dict(BENT, shoulder_flexion=180.0, shoulder_abduction=0.0, shoulder_rotation=20.0, elbow_flexion=60.0)
    out = ArmKinematics("right").solve(*forward_arm(ang))
    assert out.shoulder_rotation_valid
    assert wrap(out.shoulder_flexion - 180.0) == pytest.approx(0, abs=1e-6)
    assert out.shoulder_rotation == pytest.approx(20, abs=1e-6)


def test_arm_raised_past_horizontal_to_the_side_reads_as_flexion_near_180():
    # 120 degrees of pure abduction = long axis 60 degrees above horizontal, outward
    ang = dict(BENT, shoulder_flexion=180.0, shoulder_abduction=60.0, shoulder_rotation=0.0)
    out = ArmKinematics("right").solve(*forward_arm(ang))
    assert out.shoulder_abduction == pytest.approx(60, abs=1e-6)
    assert abs(wrap(out.shoulder_flexion)) == pytest.approx(180, abs=1e-6)


@pytest.mark.parametrize("sign", [1, -1])
def test_lateral_pole_keeps_rotation_held_and_flexion_consistent(sign):
    k = ArmKinematics("right")
    k.solve(*forward_arm(dict(BENT, shoulder_rotation=25.0)))          # establishes rotation = 25
    at_pole = dict(BENT, shoulder_flexion=10.0, shoulder_abduction=90.0 * sign, shoulder_rotation=25.0)
    out = k.solve(*forward_arm(at_pole))
    assert out.shoulder_rotation_valid is False
    assert out.shoulder_rotation == pytest.approx(25, abs=1e-6)
    assert out.shoulder_abduction == pytest.approx(90.0 * sign, abs=1e-6)
    assert out.shoulder_flexion == pytest.approx(10.0, abs=1e-6)       # the held rotation was right


@pytest.mark.parametrize("abd", [90.0, -90.0])
def test_lateral_pole_output_reproduces_the_true_shoulder_rotation(abd):
    """The held rotation changes flexion to compensate, so the reported angles still describe the same pose."""
    k = ArmKinematics("right")
    k.solve(*forward_arm(dict(BENT, shoulder_rotation=25.0)))
    true = dict(BENT, shoulder_flexion=10.0, shoulder_abduction=abd, shoulder_rotation=55.0)
    out = k.solve(*forward_arm(true))
    assert out.shoulder_rotation_valid is False
    assert out.shoulder_rotation == pytest.approx(25, abs=1e-6)                  # held, not 55

    def R(f, a, t):
        d = np.radians
        return rot_z(d(f)) @ rot_x(-d(a)) @ rot_y(d(t))

    assert np.allclose(R(out.shoulder_flexion, out.shoulder_abduction, out.shoulder_rotation),
                       R(10.0, abd, 55.0), atol=1e-9)


def test_up_hint_replaces_hips_when_they_are_wrong():
    ang = dict(BENT)
    pose, hand = forward_arm(ang)
    good = ArmKinematics("right").solve(pose, hand)
    bad = pose.copy()
    bad[L_HIP] += (0.3, 0.0, 0.0)                     # hips guessed 30 cm forward, as when out of view
    bad[R_HIP] += (0.3, 0.0, 0.0)
    wrong = ArmKinematics("right").solve(bad, hand)
    fixed = ArmKinematics("right").solve(bad, hand, up_hint=np.array([0.0, 1.0, 0.0]))
    assert abs(wrong.shoulder_flexion - good.shoulder_flexion) > 10
    for k in KEYS:
        assert getattr(fixed, k) == pytest.approx(getattr(good, k), abs=1e-6), k

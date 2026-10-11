import numpy as np
import pytest

from tests.conftest import make_hand
from vtrack import arm_angles, body_frame, finger_flexion, joint_angle


def test_joint_angle_basic():
    assert joint_angle((1, 0, 0), (0, 0, 0), (-1, 0, 0)) == pytest.approx(180)
    assert joint_angle((1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(90)
    assert joint_angle((1, 0, 0), (0, 0, 0), (1, 0, 0)) == pytest.approx(0)


def test_joint_angle_scale_invariant():
    assert joint_angle((10, 0, 0), (0, 0, 0), (0, 3, 0)) == pytest.approx(90)


def test_finger_flexion_straight_vs_curled():
    straight = finger_flexion(make_hand(extended=("index",)), "index")
    curled = finger_flexion(make_hand(), "index")
    assert straight["pip"] == pytest.approx(0, abs=1e-6)
    assert curled["pip"] == pytest.approx(90, abs=1e-6)


def test_body_frame_orthonormal(make_pose):
    r, u, f = body_frame(make_pose())
    for v in (r, u, f):
        assert np.linalg.norm(v) == pytest.approx(1)
    assert np.dot(r, u) == pytest.approx(0, abs=1e-9)
    assert np.dot(r, f) == pytest.approx(0, abs=1e-9)
    assert f == pytest.approx([0, 0, -1])


def test_arm_hanging_is_zero(make_pose):
    a = arm_angles(make_pose(), "right")
    for k in a:
        assert a[k] == pytest.approx(0, abs=1e-6)


def test_arm_forward_raise(make_pose):
    a = arm_angles(make_pose(r_elbow=(0.2, 1.4, -0.3), r_wrist=(0.2, 1.4, -0.6)), "right")
    assert a["shoulder_flexion"] == pytest.approx(90)
    assert a["shoulder_abduction"] == pytest.approx(0, abs=1e-6)
    assert a["shoulder_elevation"] == pytest.approx(90)
    assert a["elbow_flexion"] == pytest.approx(0, abs=1e-6)


def test_arm_abduction_both_sides_positive_outward(make_pose):
    p = make_pose(r_elbow=(0.5, 1.4, 0), r_wrist=(0.8, 1.4, 0), l_elbow=(-0.5, 1.4, 0), l_wrist=(-0.8, 1.4, 0))
    assert arm_angles(p, "right")["shoulder_abduction"] == pytest.approx(90)
    assert arm_angles(p, "left")["shoulder_abduction"] == pytest.approx(90)


def test_elbow_flexion_90(make_pose):
    a = arm_angles(make_pose(r_wrist=(0.2, 1.1, -0.3)), "right")
    assert a["elbow_flexion"] == pytest.approx(90)


def test_backward_swing_is_negative_flexion(make_pose):
    a = arm_angles(make_pose(r_elbow=(0.2, 1.1, 0.2), r_wrist=(0.2, 0.9, 0.2)), "right")
    assert a["shoulder_flexion"] < 0


def test_pure_sideways_raise_has_zero_flexion(make_pose):
    a = arm_angles(make_pose(r_elbow=(0.5, 1.4, 0), r_wrist=(0.8, 1.4, 0)), "right")
    assert a["shoulder_flexion"] == pytest.approx(0, abs=1e-6)
    assert a["shoulder_abduction"] == pytest.approx(90)


def test_diagonal_raise_splits_between_planes(make_pose):
    # 45 degrees forward of the sagittal plane's outward side: equal forward and outward components, horizontal
    a = arm_angles(make_pose(r_elbow=(0.2 + 0.212, 1.4, -0.212), r_wrist=(0.8, 1.4, -0.4)), "right")
    assert a["shoulder_abduction"] == pytest.approx(45, abs=0.1)
    assert a["shoulder_elevation"] == pytest.approx(90, abs=0.1)

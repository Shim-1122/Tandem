import numpy as np
import pytest

from vtrack import frame_agreement_deg
from vtrack.synthetic import forward_arm
from vtrack.tracker import FrameResult, HandResult, pick_hand


def hand_at(x, y):
    img = np.zeros((21, 3))
    img[0, :2] = (x, y)
    return HandResult("Left", np.zeros((21, 3)), img)


def pose_with_wrists(left, right):
    img = np.zeros((33, 3))
    img[15, :2], img[16, :2] = left, right
    return img


def test_pick_hand_matches_by_position_not_label():
    near_right, near_left = hand_at(0.30, 0.50), hand_at(0.70, 0.50)
    res = FrameResult(hands=[near_left, near_right], pose_image=pose_with_wrists((0.7, 0.5), (0.3, 0.5)))
    assert pick_hand(res, "right") is near_right
    assert pick_hand(res, "left") is near_left


def test_pick_hand_none_when_too_far_or_missing():
    res = FrameResult(hands=[hand_at(0.9, 0.9)], pose_image=pose_with_wrists((0.1, 0.1), (0.2, 0.2)))
    assert pick_hand(res, "right") is None
    assert pick_hand(FrameResult(), "right") is None


def test_frame_agreement_zero_for_consistent_landmarks_without_rotation():
    ang = dict(shoulder_flexion=20, shoulder_abduction=10, shoulder_rotation=10, elbow_flexion=70,
               forearm_rotation=15, wrist_flexion=10, wrist_deviation=5)
    pose, hand = forward_arm(ang, "right")
    out = frame_agreement_deg(pose, hand, "right")
    assert out["long_axis_deg"] == pytest.approx(0, abs=1e-6)
    assert out["palm_normal_deg"] == pytest.approx(0, abs=1e-6)


def test_frame_agreement_flags_a_rotated_hand_frame():
    ang = dict(shoulder_flexion=20, shoulder_abduction=10, shoulder_rotation=10, elbow_flexion=70,
               forearm_rotation=15, wrist_flexion=10, wrist_deviation=5)
    pose, hand = forward_arm(ang, "right")
    c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
    rotated = hand @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T   # hand axes 40 degrees off the pose axes
    assert frame_agreement_deg(pose, rotated, "right")["long_axis_deg"] > 10

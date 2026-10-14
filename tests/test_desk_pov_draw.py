import numpy as np
import pytest

from tests.conftest import make_hand
from vtrack import ArmKinematics
from vtrack.geometry import L_HIP, R_HIP
from vtrack.handpose import HandPoseZero, all_finger_angles, camera_matrix, hand_frame
from vtrack.kinematics import rot_x, rot_y, rot_z, up_from_neutral
from vtrack.synthetic import forward_arm

KEYS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion",
        "forearm_rotation", "wrist_flexion", "wrist_deviation"]
NEUTRAL = dict(shoulder_flexion=0.0, shoulder_abduction=0.0, shoulder_rotation=0.0, elbow_flexion=90.0,
               forearm_rotation=0.0, wrist_flexion=0.0, wrist_deviation=0.0)
CAM = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], float)   # thorax axes into camera axes (person facing camera)


# ---------- desk mode: no hips ----------

@pytest.mark.parametrize("side", ["right", "left"])
def test_desk_calibration_recovers_angles_with_hips_garbage_and_tilted_camera(side):
    tilt = rot_x(np.radians(25))                  # laptop camera pitched 25 degrees
    body = tilt @ CAM
    cal_pose, _ = forward_arm(NEUTRAL, side, body_rotation=body)
    up = up_from_neutral(cal_pose, side)
    truth = dict(shoulder_flexion=55, shoulder_abduction=25, shoulder_rotation=35, elbow_flexion=75,
                 forearm_rotation=-30, wrist_flexion=20, wrist_deviation=-10)
    pose, hand = forward_arm(truth, side, body_rotation=body)
    pose[L_HIP] = pose[R_HIP] = (5.0, 5.0, 5.0)    # hips out of view: landmarks are nonsense
    out = ArmKinematics(side).solve(pose, hand, up_hint=up)
    for k in KEYS:
        assert getattr(out, k) == pytest.approx(truth[k], abs=1e-6), k


def test_camera_vertical_without_calibration_is_off_by_the_camera_tilt():
    body = rot_x(np.radians(25)) @ CAM
    pose, hand = forward_arm(NEUTRAL, "right", body_rotation=body)
    pose[L_HIP] = pose[R_HIP] = (5.0, 5.0, 5.0)
    out = ArmKinematics("right").solve(pose, hand, up_hint=np.array([0.0, -1.0, 0.0]))
    assert abs(out.shoulder_flexion) == pytest.approx(25, abs=1e-6)   # the error calibration removes


# ---------- POV mode ----------

def test_camera_matrix_from_fov():
    K = camera_matrix(1280, 720, 90)
    assert K[0, 0] == pytest.approx(640) and K[0, 2] == 640 and K[1, 2] == 360


def project(K, pts):
    q = (K @ pts.T).T
    return np.column_stack([q[:, 0] / q[:, 2] / (2 * K[0, 2]), q[:, 1] / q[:, 2] / (2 * K[1, 2])])


def test_pnp_recovers_hand_position_and_rotation():
    cv2 = pytest.importorskip("cv2")
    from vtrack.handpose import hand_pose_camera
    K = camera_matrix(1280, 720, 70)
    w = make_hand(extended=("index", "middle", "ring", "pinky"), thumb=True)
    w = w - w.mean(axis=0)
    rng = np.random.default_rng(5)
    for _ in range(20):
        R, _ = cv2.Rodrigues(rng.normal(scale=0.4, size=3))
        t = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), rng.uniform(0.3, 0.8)])
        R_est, t_est = hand_pose_camera(w, project(K, (R @ w.T).T + t), K)
        assert t_est == pytest.approx(t, abs=1e-5)
        assert np.allclose(R_est, R, atol=1e-5)


def test_hand_pose_zero_reports_relative_motion():
    z = HandPoseZero()
    assert z.relative(np.eye(3), np.zeros(3)) is None
    R0 = rot_y(0.3)
    z.zero(R0, np.array([0.1, 0.0, 0.5]))
    m = z.relative(R0 @ rot_y(np.radians(20)) @ rot_x(np.radians(10)), np.array([0.12, -0.03, 0.45]))
    assert (m.dx_cm, m.dy_cm, m.dz_cm) == pytest.approx((2.0, -3.0, -5.0))
    assert (m.twist, m.pitch, m.yaw) == pytest.approx((20.0, 10.0, 0.0), abs=1e-9)


def test_hand_frame_is_a_rotation_and_rotates_with_the_hand():
    w = make_hand(extended=("index", "middle", "ring", "pinky"), thumb=True)
    F = hand_frame(w)
    assert F.T @ F == pytest.approx(np.eye(3), abs=1e-9) and np.linalg.det(F) == pytest.approx(1)
    R = rot_z(0.7) @ rot_x(-0.3)
    assert hand_frame((R @ w.T).T) == pytest.approx(R @ F, abs=1e-9)


def test_all_finger_angles_open_vs_fist():
    open_ = all_finger_angles(make_hand(extended=("index", "middle", "ring", "pinky"), thumb=True))
    fist = all_finger_angles(make_hand())
    for f in ("index", "middle", "ring", "pinky"):
        assert open_[f]["pip"] == pytest.approx(0, abs=1e-6)
        assert fist[f]["pip"] == pytest.approx(90, abs=1e-6)


# ---------- drawing ----------

def test_wireframe_draws_and_skips_bad_points():
    pytest.importorskip("cv2")
    from vtrack.draw import HAND_EDGES, draw_wireframe
    img = np.zeros((200, 200, 3), np.uint8)
    pts = np.full((21, 3), 0.5)
    pts[:, 0] += np.linspace(-0.3, 0.3, 21)
    pts[3] = np.nan
    draw_wireframe(img, pts, HAND_EDGES, (0, 255, 0))
    assert img.any()


def test_panels_render_with_and_without_data():
    pytest.importorskip("cv2")
    from vtrack.draw import render_arm_panel, render_hand_panel
    from vtrack.kinematics import ArmAngles
    a = render_arm_panel(dict(NEUTRAL, shoulder_flexion=30.0), "left", size=240)
    b = render_arm_panel(ArmAngles(), size=240)                      # all NaN: no pose message
    c = render_hand_panel(make_hand(), size=240)
    d = render_hand_panel(None, size=240)
    for img in (a, b, c, d):
        assert img.shape == (240, 240, 3) and img.any()


def test_desk_calibration_with_both_arms_hanging():
    body = rot_x(np.radians(20)) @ CAM
    pr, _ = forward_arm(NEUTRAL, "right", body_rotation=body)
    pl, _ = forward_arm(NEUTRAL, "left", body_rotation=body)
    pose = pr.copy()
    for i in (13, 15, 17, 19):
        pose[i] = pl[i]
    up = up_from_neutral(pose, "both")
    assert up == pytest.approx(body @ np.array([0.0, 1.0, 0.0]), abs=1e-9)


def test_both_arms_solved_independently_from_one_pose():
    from vtrack.geometry import L_ELBOW, L_WRIST
    right = dict(shoulder_flexion=40, shoulder_abduction=20, shoulder_rotation=30, elbow_flexion=80,
                 forearm_rotation=20, wrist_flexion=15, wrist_deviation=5)
    left = dict(shoulder_flexion=10, shoulder_abduction=45, shoulder_rotation=-20, elbow_flexion=110,
                forearm_rotation=-40, wrist_flexion=-10, wrist_deviation=-15)
    pr, hr = forward_arm(right, "right", body_rotation=CAM)
    pl, hl = forward_arm(left, "left", body_rotation=CAM)
    pose = pr.copy()
    for i in (L_ELBOW, L_WRIST, 17, 19):
        pose[i] = pl[i]
    out_r = ArmKinematics("right").solve(pose, hr)
    out_l = ArmKinematics("left").solve(pose, hl)
    for k in KEYS:
        assert getattr(out_r, k) == pytest.approx(right[k], abs=1e-6)
        assert getattr(out_l, k) == pytest.approx(left[k], abs=1e-6)


def test_two_arm_panel_and_column_text_render():
    pytest.importorskip("cv2")
    from vtrack.draw import render_arms_panel, text_block
    from vtrack.kinematics import ArmAngles
    img = render_arms_panel({"left": dict(NEUTRAL), "right": ArmAngles()}, size=240)
    assert img.shape == (240, 240, 3) and img.any()
    canvas = np.zeros((200, 600, 3), np.uint8)
    text_block(canvas, ["\tLEFT\tRIGHT", "Elbow flexion\t110.0\t80.0", "torso up: calibrated"])
    assert canvas.any()


def test_wireframes_match_mediapipe_topology():
    vision = pytest.importorskip("mediapipe.tasks.python.vision")
    from vtrack.draw import HAND_EDGES, POSE_ARM_EDGES, POSE_LEG_EDGES
    edges = lambda cs: {tuple(sorted((c.start, c.end))) for c in cs}            # noqa: E731
    assert {tuple(sorted(e)) for e in HAND_EDGES} == edges(vision.HandLandmarksConnections.HAND_CONNECTIONS)
    body = edges(vision.PoseLandmarksConnections.POSE_LANDMARKS)
    assert {tuple(sorted(e)) for e in POSE_ARM_EDGES + POSE_LEG_EDGES} <= body   # the face is drawn separately

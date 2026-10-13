import math

import numpy as np
import pytest

from vtrack.gaze import (EYEBALL_RADIUS_M, GazeCalibrationSession, TargetLock, calibration_targets, eye_state,
                         fit_affine, gaze_direction_in_head)
from vtrack.handpose import camera_matrix
from vtrack.head import ANCHOR_IDS, HEAD_AXES_IN_MODEL, MODEL_POINTS_M, head_axes_from_angles

W, H = 1280, 720
K = camera_matrix(W, H, 70)
# Eye corners in head model coordinates (m): x toward the person's left, y up, z forward
# (the outer corners are head-pose anchors, so they use the anchors' exact model positions)
EYE_MODEL = {"right": dict(outer=tuple(MODEL_POINTS_M[ANCHOR_IDS.index(33)]), inner=(-0.0190, 0.0262, 0.0330),
                           ids=dict(outer=33, inner=133, iris=468, upper=159, lower=145)),
             "left": dict(outer=tuple(MODEL_POINTS_M[ANCHOR_IDS.index(263)]), inner=(0.0190, 0.0262, 0.0330),
                          ids=dict(outer=263, inner=362, iris=473, upper=386, lower=374))}


class Pose:
    def __init__(self, yaw, pitch, roll, pos):
        self.axes = head_axes_from_angles(yaw, pitch, roll)
        self.position_m = np.asarray(pos, float)


def face(head=(0, 0, 0), pos=(0.0, 0.05, 0.6), eye=(0.0, 0.0), target=None, closed=False):
    """478 synthetic face landmarks (MediaPipe conventions) and the matching head pose.

    eye: eye-in-head yaw (+ left) and pitch (+ up) in degrees; or target: a camera-frame point to look at.
    The eyeball center sits straight behind the middle of the eye corners, so the iris is centered there
    when looking straight ahead.
    """
    pose = Pose(*head, pos)
    R_model = pose.axes @ HEAD_AXES_IN_MODEL.T
    t = pose.position_m
    pts = {i: MODEL_POINTS_M[k] for k, i in enumerate(ANCHOR_IDS)}
    mids = [(np.array(e["outer"]) + np.array(e["inner"])) / 2 for e in EYE_MODEL.values()]
    if target is not None:
        origin = R_model @ np.mean(mids, axis=0) + t
        d_model = R_model.T @ ((np.asarray(target, float) - origin) / np.linalg.norm(np.asarray(target, float) - origin))
    else:
        y, p = np.radians(eye)
        d_model = np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])
    for e, mid in zip(EYE_MODEL.values(), mids):
        width = np.linalg.norm(np.array(e["outer"]) - np.array(e["inner"]))
        gap = (0.02 if closed else 0.3) * width
        pts[e["ids"]["outer"]], pts[e["ids"]["inner"]] = np.array(e["outer"]), np.array(e["inner"])
        pts[e["ids"]["iris"]] = mid - np.array([0, 0, EYEBALL_RADIUS_M]) + EYEBALL_RADIUS_M * d_model
        pts[e["ids"]["upper"]] = mid + np.array([0, gap / 2, 0.002])
        pts[e["ids"]["lower"]] = mid - np.array([0, gap / 2, -0.002])
    cam = {i: R_model @ p + t for i, p in pts.items()}
    z_ref = np.mean([cam[i][2] for i in ANCHOR_IDS])
    L = np.zeros((478, 3))
    for i, c in cam.items():
        L[i] = [(K[0, 0] * c[0] / c[2] + K[0, 2]) / W, (K[1, 1] * c[1] / c[2] + K[1, 2]) / H,
                (c[2] - z_ref) * K[0, 0] / z_ref / W]
    return L, pose, d_model


def test_eye_angles_and_direction_are_recovered_for_any_head_pose():
    rng = np.random.default_rng(0)
    for _ in range(200):
        head = (rng.uniform(-30, 30), rng.uniform(-20, 20), rng.uniform(-15, 15))
        pos = (rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), rng.uniform(0.4, 1.0))
        eye = (rng.uniform(-25, 25), rng.uniform(-20, 20))
        L, pose, d_model = face(head, pos, eye)
        s = eye_state(L, pose, W, H, K, neutral_dy=0.0)
        assert s.valid
        assert (s.yaw, s.pitch) == pytest.approx(eye, abs=1e-6)
        assert s.direction == pytest.approx(pose.axes @ HEAD_AXES_IN_MODEL.T @ d_model, abs=1e-9)


def test_gaze_meets_the_camera_plane_where_you_look():
    rng = np.random.default_rng(1)
    for _ in range(100):
        target = np.array([rng.uniform(-0.25, 0.25), rng.uniform(0.0, 0.25), 0.0])      # screen below the camera
        L, pose, _ = face((rng.uniform(-20, 20), rng.uniform(-15, 15), rng.uniform(-10, 10)),
                          (rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), rng.uniform(0.45, 0.8)), target=target)
        s = eye_state(L, pose, W, H, K, neutral_dy=0.0)
        assert s.plane_point == pytest.approx(target[:2], abs=1e-9)


def test_closed_eyes_and_missing_data_are_not_valid():
    L, pose, _ = face(closed=True)
    s = eye_state(L, pose, W, H, K)
    assert not s.valid and s.openness[0] < 0.13
    L, pose, _ = face()
    assert eye_state(L, pose, W, H, K).openness[0] > 0.2
    assert not eye_state(None, pose, W, H, K).valid and not eye_state(L, None, W, H, K).valid
    assert not eye_state(L[:468], pose, W, H, K).valid            # no iris landmarks


def test_looking_away_from_the_screen_has_no_plane_point():
    L, pose, _ = face(eye=(0, 0), head=(0, 0, 0))
    assert eye_state(L, pose, W, H, K, neutral_dy=0.0).plane_point is not None
    L, pose, _ = face(eye=(0, 0), head=(120, 0, 0))                  # head turned far away
    assert eye_state(L, pose, W, H, K, neutral_dy=0.0).plane_point is None


def test_direction_in_head_axes():
    assert gaze_direction_in_head(0, 0) == pytest.approx([1, 0, 0])
    assert gaze_direction_in_head(90, 0) == pytest.approx([0, 0, -1])      # toward your left = -right
    assert gaze_direction_in_head(0, 90) == pytest.approx([0, 1, 0])


# A virtual screen just below the webcam, shown mirrored: video pixel = M @ camera-plane meters + c
SCREEN_M = np.array([[-2000.0, 0.0], [0.0, 2000.0]])
SCREEN_C = np.array([640.0, 40.0])


def plane_of(display_px):
    return np.append(np.linalg.solve(SCREEN_M, np.asarray(display_px, float) - SCREEN_C), 0.0)


def calibrate(head, pos, noise_px=0.0, rng=None, dots=None):
    targets = calibration_targets(W, H)
    s = GazeCalibrationSession(targets, t0=0.0, settle_s=1.0, capture_s=1.0)
    for i, dot in enumerate(targets):
        if dots is not None and i not in dots:
            continue
        for k in range(10):
            t = i * 2.0 + 1.05 + k * 0.09
            L, pose, _ = face(head, pos, target=plane_of(dot))
            if noise_px:
                L[:, :2] += rng.normal(scale=noise_px, size=(478, 2)) / np.array([W, H])
            st = eye_state(L, pose, W, H, K, neutral_dy=0.0)
            s.add(st.plane_point, float(np.linalg.norm(st.origin)), t)
    return s


def test_calibration_maps_gaze_to_the_video_and_survives_head_motion():
    s = calibrate(head=(0, -5, 0), pos=(0.0, 0.08, 0.6))
    assert s.finished(18.0) and not s.finished(17.9)
    m = s.result()
    assert m.dots == 9 and m.error_px < 1e-6 and m.error_deg < 1e-6
    # move 6 cm, turn 12 degrees and tilt: the mapping, built on the geometric gaze point, still hits the spot
    for spot in ([200.0, 150.0], [1000.0, 600.0], [640.0, 360.0]):
        L, pose, _ = face((12, 4, -6), (0.06, 0.03, 0.7), target=plane_of(spot))
        st = eye_state(L, pose, W, H, K, neutral_dy=0.0)
        assert m.predict(st.plane_point) == pytest.approx(spot, abs=1e-6)


def test_calibration_with_landmark_noise_reports_its_own_error():
    rng = np.random.default_rng(2)
    m = calibrate(head=(0, -5, 0), pos=(0.0, 0.08, 0.6), noise_px=0.15, rng=rng).result()
    assert 0 < m.error_px < 150 and 0 < m.error_deg < 8
    assert "deg" in m.summary()


def test_calibration_needs_enough_dots():
    s = calibrate(head=(0, 0, 0), pos=(0, 0.08, 0.6), dots={0, 1, 2, 3})
    with pytest.raises(ValueError, match="only 4 of 9"):
        s.result()
    s = GazeCalibrationSession(calibration_targets(W, H), 0.0, 1.0, 1.0)     # eyes fixed on one spot
    for i in range(9):
        for k in range(5):
            s.add(np.array([0.01, 0.12]), 0.6, i * 2.0 + 1.1 + 0.1 * k)
    with pytest.raises(ValueError, match="hardly moved"):
        s.result()
    s = GazeCalibrationSession(calibration_targets(W, H), 10.0, 1.0, 1.0)
    assert s.phase(10.5)[:2] == (0, "settle") and s.phase(11.5)[:2] == (0, "capture")
    assert not s.add(np.zeros(2), 0.6, 10.5) and not s.add(None, 0.6, 11.5) and s.add(np.zeros(2), 0.6, 11.5)


def test_fit_affine_is_exact_for_an_affine_map():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(9, 2))
    A = np.array([[3.0, -1.0, 5.0], [0.5, 2.0, -4.0]])
    Y = X @ A[:, :2].T + A[:, 2]
    assert fit_affine(X, Y) == pytest.approx(A)


def test_target_lock_toggle_mirror_and_ray():
    lock = TargetLock(W, H, K, mirrored=True)
    assert lock.toggle(None, 0.0) == "no gaze" and not lock.locked
    assert lock.toggle([W - 1 - K[0, 2], K[1, 2]], 1.0) == "locked"
    assert lock.camera_point() == pytest.approx([K[0, 2], K[1, 2]])
    assert lock.ray() == pytest.approx([0, 0, 1])                    # the image center is straight ahead
    assert lock.toggle([100, 100], 2.0) == "released" and lock.ray() is None
    lock = TargetLock(W, H, K, mirrored=False)
    lock.lock([K[0, 2] + K[0, 0], K[1, 2]], 0.0)                     # one focal length right: 45 degrees
    r = lock.ray()
    assert math.degrees(math.atan2(r[0], r[2])) == pytest.approx(45.0)


def test_dwell_locking():
    lock = TargetLock(W, H, K, dwell_s=0.8, radius_px=40)
    t = 0.0
    for k in range(5):                                               # wandering: never locks
        assert not lock.update([100 + 100 * k, 300], t)
        t += 0.3
    assert lock.dwell_progress(t) < 0.5
    for k in range(10):
        lock.update([500 + (k % 2) * 5, 400], t)
        t += 0.1
    assert lock.locked and lock.display_point == pytest.approx([502.5, 400], abs=3)
    assert not lock.update([900, 100], t + 1.0) and lock.locked      # stays locked until released
    assert not TargetLock(W, H, K).update([1, 1], 0.0)               # dwell off by default

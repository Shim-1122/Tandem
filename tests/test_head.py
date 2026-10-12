import numpy as np
import pytest

pytest.importorskip("cv2")

from vtrack.handpose import camera_matrix
from vtrack.head import (ANCHOR_IDS, ANCHORS, FACING_CAMERA, HEAD_AXES_IN_MODEL, MODEL_POINTS_M, HeadPoseEstimator,
                         HeadPoseZero, angles_from_axes, head_axes_from_angles, project_model_points)

W, H = 1280, 720
K = camera_matrix(W, H, 70)


def landmarks_for(yaw, pitch, roll, pos=(0.02, -0.03, 0.6), noise_px=0.0, rng=None):
    """Fake 478-point face landmarks: the 4 anchors at their true perspective projections, with a depth z
    in MediaPipe's convention (relative to the face, same scale as x, i.e. divided by image width)."""
    axes = head_axes_from_angles(yaw, pitch, roll)
    pos = np.array(pos, float)
    px = project_model_points(axes, pos, K)
    cam = (axes @ HEAD_AXES_IN_MODEL.T @ MODEL_POINTS_M.T).T + pos
    z_px = (cam[:, 2] - cam[:, 2].mean()) * K[0, 0] / cam[:, 2].mean()
    if noise_px:
        px = px + rng.normal(scale=noise_px, size=px.shape)
        z_px = z_px + rng.normal(scale=noise_px * 2, size=z_px.shape)     # depth is noisier
    L = np.zeros((478, 3))
    L[ANCHOR_IDS, 0], L[ANCHOR_IDS, 1], L[ANCHOR_IDS, 2] = px[:, 0] / W, px[:, 1] / H, z_px / W
    return L, px


def test_facing_camera_is_zero():
    L, _ = landmarks_for(0, 0, 0)
    p = HeadPoseEstimator().solve(L, W, H, K)
    assert (p.yaw, p.pitch, p.roll) == pytest.approx((0, 0, 0), abs=1e-4)
    assert p.position_cm == pytest.approx([2.0, -3.0, 60.0], abs=1e-3)
    assert np.allclose(p.axes, FACING_CAMERA, atol=1e-6)


def test_round_trip_random_poses():
    rng = np.random.default_rng(0)
    for _ in range(200):
        yaw, pitch, roll = rng.uniform(-60, 60), rng.uniform(-35, 35), rng.uniform(-35, 35)
        pos = (rng.uniform(-0.25, 0.25), rng.uniform(-0.15, 0.15), rng.uniform(0.35, 1.2))
        L, _ = landmarks_for(yaw, pitch, roll, pos)
        p = HeadPoseEstimator().solve(L, W, H, K)
        assert (p.yaw, p.pitch, p.roll) == pytest.approx((yaw, pitch, roll), abs=1e-3)
        assert p.position_m == pytest.approx(pos, abs=1e-5)
        assert p.reprojection_px < 1e-3


def test_signs_match_what_you_see_in_the_camera():
    _, n0 = landmarks_for(0, 0, 0)           # anchor order: forehead, nose tip, right eye outer, left eye outer
    _, yl = landmarks_for(20, 0, 0)
    eyes_mid = (yl[2] + yl[3]) / 2
    assert yl[1, 0] > eyes_mid[0] + 5                       # turn to your left: nose moves right in the image
    _, up = landmarks_for(0, 20, 0)
    assert (up[1, 1] - (up[2, 1] + up[3, 1]) / 2) < (n0[1, 1] - (n0[2, 1] + n0[3, 1]) / 2) - 3   # look up
    _, tr = landmarks_for(0, 0, 20)
    assert tr[2, 1] > tr[3, 1] + 5                          # tilt right: your right eye (image left) drops
    assert n0[2, 0] < n0[3, 0]                              # your right eye is on the image's left


def test_noisy_anchors_stay_close_and_tracking_is_stable():
    rng = np.random.default_rng(1)
    est = HeadPoseEstimator()
    errs = []
    for k in range(100):
        yaw, pitch, roll = 25 * np.sin(k / 15), 10 * np.sin(k / 23), 8 * np.sin(k / 31)
        L, _ = landmarks_for(yaw, pitch, roll, noise_px=1.0, rng=rng)
        p = est.solve(L, W, H, K)
        errs.append(np.abs([p.yaw - yaw, p.pitch - pitch, p.roll - roll]))
    errs = np.array(errs)
    assert np.median(errs) < 2.0 and errs.max() < 10.0      # 1 px anchor noise: degrees of error, no flips


def test_zero_reports_relative_motion():
    z = HeadPoseZero()
    est = HeadPoseEstimator()
    L, _ = landmarks_for(10, -5, 3, pos=(0.0, 0.0, 0.6))
    p0 = est.solve(L, W, H, K)
    assert z.relative(p0) is None
    z.zero(p0)
    assert z.relative(p0) == pytest.approx((0, 0, 0, 0, 0, 0), abs=1e-4)
    L, _ = landmarks_for(10, -5, 3, pos=(0.05, 0.0, 0.7))
    rel = z.relative(est.solve(L, W, H, K))
    assert rel[:3] == pytest.approx((5.0, 0.0, 10.0), abs=1e-2)


def test_angles_from_axes_inverts_head_axes_from_angles():
    rng = np.random.default_rng(2)
    for _ in range(100):
        a = rng.uniform([-80, -80, -80], [80, 80, 80])
        assert angles_from_axes(head_axes_from_angles(*a)) == pytest.approx(tuple(a), abs=1e-9)


def test_anchor_names():
    assert ANCHORS == {"forehead": 151, "nose_tip": 1, "right_eye_outer": 33, "left_eye_outer": 263}

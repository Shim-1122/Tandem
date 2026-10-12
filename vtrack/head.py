"""Head pose with 6 degrees of freedom from 4 anchor points on the face.

Anchors (MediaPipe face mesh indices) are on the rigid part of the skull, so talking, smiling or
opening the jaw does not move them:
    forehead (151), nose tip (1), right eye outer corner (33), left eye outer corner (263)
("right" and "left" are your own sides). Their reference 3D positions come from MediaPipe's canonical face model.

Rotation: the best rigid fit (Kabsch) between the 4 reference anchors and the 4 anchors as MediaPipe
measures them in 3D (image x and y plus its per-landmark depth estimate). A 2D-only fit of 4 points was
tested and rejected: on a photo of a person looking straight at the camera it read 27 degrees of pitch,
because pitch then depends on each person's forehead-eye-nose proportions. The 3D fit read within 3
degrees of straight on that photo and followed in-plane image rotations of 15 and 30 degrees to within
1 degree. Position: with the rotation fixed, the head position that best reprojects the anchors (linear
least squares), which needs the camera's field of view (--fov). The anchors are then unprojected at that
distance and the fit is repeated, which removes perspective error for heads near the edge of the frame.

Output, relative to the camera (camera axes: x right, y down, z away from the camera):
    x, y, z   head position in cm (the canonical model's origin, inside the head behind the nose)
    yaw       turning: + toward your left (your nose moves to the right in the camera image)
    pitch     nodding: + looking up
    roll      tilting: + top of the head toward your right shoulder
All angles are zero when you face the camera squarely and upright. Yaw, pitch and roll are applied in
that order (intrinsic Y-Z-X). Distance assumes an average-size face, so a larger or smaller face reads
proportionally farther or nearer; relative motion after zeroing is more accurate than absolute distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import rot_x, rot_y, rot_z

ANCHORS = {"forehead": 151, "nose_tip": 1, "right_eye_outer": 33, "left_eye_outer": 263}
ANCHOR_IDS = list(ANCHORS.values())
# Canonical face model (MediaPipe face_geometry/data/canonical_face_model.obj), centimeters.
# Model axes: x toward the person's left, y up, z out of the face (forward).
CANONICAL_CM = {
    151: (0.000000, 6.545390, 5.027311),
    1: (0.000000, -1.126865, 7.475604),
    33: (-4.445859, 2.663991, 3.173422),
    263: (4.445859, 2.663991, 3.173422),
}
MODEL_POINTS_M = np.array([CANONICAL_CM[i] for i in ANCHOR_IDS], dtype=np.float64) / 100.0

# Head axes (forward, up, right) expressed in model coordinates.
HEAD_AXES_IN_MODEL = np.column_stack([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
# Head axes of a person facing the camera upright, in camera coordinates.
FACING_CAMERA = np.column_stack([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])


def euler_yzx(R: np.ndarray) -> tuple[float, float, float]:
    """(a, b, c) radians with R = Ry(a) Rz(b) Rx(c); b is limited to +-90 degrees."""
    b = np.arcsin(np.clip(R[1, 0], -1.0, 1.0))
    return float(np.arctan2(-R[2, 0], R[0, 0])), float(b), float(np.arctan2(-R[1, 2], R[1, 1]))


def head_axes_from_angles(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Head axes (forward, up, right) in camera coordinates for the given angles."""
    d = np.radians([yaw_deg, pitch_deg, roll_deg])
    return FACING_CAMERA @ rot_y(d[0]) @ rot_z(d[1]) @ rot_x(d[2])


def angles_from_axes(R_head: np.ndarray, reference: np.ndarray = FACING_CAMERA) -> tuple[float, float, float]:
    """(yaw, pitch, roll) degrees of head axes relative to reference axes (both as forward, up, right columns)."""
    return tuple(float(np.degrees(v)) for v in euler_yzx(reference.T @ R_head))


@dataclass
class HeadPose:
    axes: np.ndarray            # (3, 3) columns forward, up, right in camera coordinates
    position_m: np.ndarray      # (3,) model origin in camera coordinates, meters
    yaw: float
    pitch: float
    roll: float
    anchors_px: np.ndarray      # (4, 2) anchor pixel positions in ANCHORS order
    reprojection_px: float      # mean anchor reprojection error, pixels

    @property
    def position_cm(self) -> np.ndarray:
        return self.position_m * 100.0


def kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rotation R minimizing sum |R (A_i - mean A) - (B_i - mean B)|^2 (scale-free, proper rotation)."""
    a, b = A - A.mean(axis=0), B - B.mean(axis=0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def translation_given_rotation(R_model: np.ndarray, pts_px: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Position t (meters) that best reprojects the model anchors for a known rotation (linear least squares)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rows, rhs = [], []
    for p, (u, v) in zip(MODEL_POINTS_M, pts_px):
        q = R_model @ p
        # fx (qx + tx) = (u - cx)(qz + tz)   and   fy (qy + ty) = (v - cy)(qz + tz)
        rows.append([fx, 0.0, -(u - cx)]); rhs.append((u - cx) * q[2] - fx * q[0])
        rows.append([0.0, fy, -(v - cy)]); rhs.append((v - cy) * q[2] - fy * q[1])
    t, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return t


class HeadPoseEstimator:
    """Solves the head pose from one frame's face landmarks."""

    def __init__(self, iterations: int = 4):
        self.iterations = iterations

    def reset(self):
        pass

    def solve(self, face_landmarks: np.ndarray, width: int, height: int, K: np.ndarray) -> HeadPose | None:
        """face_landmarks: (468 or 478, 3) face landmarker output (normalized x, y and relative depth z)."""
        L = np.asarray(face_landmarks, dtype=np.float64)[ANCHOR_IDS]
        if L.shape[1] < 3:
            return None
        # MediaPipe's z uses roughly the same scale as x (image width), so this is an isotropic point set,
        # but x and y are perspective projections: first fit as if the camera were orthographic, then
        # unproject each anchor at its estimated depth and refit, a few times (converges quickly).
        pts = np.column_stack([L[:, 0] * width, L[:, 1] * height])
        z_px = L[:, 2] * width
        R_model = kabsch(MODEL_POINTS_M, np.column_stack([pts, z_px]))
        t = translation_given_rotation(R_model, pts, K)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        for _ in range(self.iterations):
            if not np.all(np.isfinite(t)) or t[2] <= 0:
                return None
            z_mean = float(np.mean((R_model @ MODEL_POINTS_M.T).T[:, 2] + t[2]))
            Z = z_mean + (z_px - z_px.mean()) * z_mean / fx
            cam = np.column_stack([(pts[:, 0] - cx) * Z / fx, (pts[:, 1] - cy) * Z / fy, Z])
            R_model = kabsch(MODEL_POINTS_M, cam)
            t = translation_given_rotation(R_model, pts, K)
        if not np.all(np.isfinite(t)) or t[2] <= 0:
            return None
        axes = R_model @ HEAD_AXES_IN_MODEL
        yaw, pitch, roll = angles_from_axes(axes)
        proj = project_model_points(axes, t, K)
        return HeadPose(axes, t, yaw, pitch, roll, pts, float(np.mean(np.linalg.norm(proj - pts, axis=1))))


def project_model_points(axes: np.ndarray, position_m: np.ndarray, K: np.ndarray, points_m=MODEL_POINTS_M):
    """Pixel positions of model points for a head pose (for drawing and for tests)."""
    R_model = axes @ HEAD_AXES_IN_MODEL.T
    cam = (R_model @ np.asarray(points_m, float).T).T + position_m
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3]


class HeadPoseZero:
    """Head motion relative to a pose captured with zero(): (dx, dy, dz cm; dyaw, dpitch, droll deg)."""

    def __init__(self):
        self.axes0 = self.pos0 = None

    @property
    def is_set(self) -> bool:
        return self.axes0 is not None

    def zero(self, pose: HeadPose):
        self.axes0, self.pos0 = pose.axes.copy(), pose.position_m.copy()

    def relative(self, pose: HeadPose):
        if not self.is_set:
            return None
        d = (pose.position_m - self.pos0) * 100.0
        return (float(d[0]), float(d[1]), float(d[2]), *angles_from_axes(pose.axes, self.axes0))

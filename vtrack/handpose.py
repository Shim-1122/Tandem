"""Hand-only tracking for a head-mounted (POV) camera, where the shoulders and torso are not visible.

From a POV camera the shoulder and elbow angles are not observable, so this tracks what is:
  * the hand's 3D position relative to the camera, by solving a perspective-n-point (PnP) problem
    between the metric hand world landmarks and their pixel locations,
  * the hand's 3D orientation relative to the camera,
  * every finger joint angle, and gestures.
For teleoperation, `HandPoseZero` reports motion relative to a pose you set with one key press, which
is what a robot arm's end-effector target would follow. Head motion moves the camera too, so a
head-mounted setup needs head tracking (an IMU in the goggles) to separate hand motion from head motion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import FINGERS, finger_flexion, unit
from .kinematics import euler_yxz


def camera_matrix(width: int, height: int, hfov_deg: float = 70.0) -> np.ndarray:
    """Pinhole intrinsics from a horizontal field of view. Typical: laptop webcam 65 to 78, GoPro wide 120+."""
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=float)


def hand_frame(world: np.ndarray) -> np.ndarray:
    """Hand frame columns (X toward thumb side, Y proximal, Z = X x Y), in the landmarks' own axes.

    Same convention as the wrist frame in kinematics.py. Note Z = X x Y points out of the BACK of a
    right hand and out of the palm of a left hand.
    """
    y = unit(world[0] - world[9])
    x = world[5] - world[17]
    x = unit(x - np.dot(x, y) * y)
    return np.column_stack([x, y, np.cross(x, y)])


def hand_pose_camera(world: np.ndarray, image_norm: np.ndarray, K: np.ndarray):
    """Hand position (meters) and rotation in the camera frame (x right, y down, z forward).

    world: (21, 3) hand world landmarks (meters, hand centered). image_norm: (21, 2+) normalized pixels.
    Returns (R, t): world-landmark axes to camera axes, and the hand center position. Depth assumes an
    average-size hand, so a larger or smaller hand reads proportionally farther or nearer.
    """
    import cv2
    w, h = 2 * K[0, 2], 2 * K[1, 2]
    px = np.asarray(image_norm, dtype=float)[:, :2] * np.array([w, h])
    obj = np.asarray(world, dtype=np.float64)
    flags = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj, px.astype(np.float64), K, None, flags=flags)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def all_finger_angles(world: np.ndarray) -> dict[str, dict[str, float]]:
    """Flexion in degrees at every finger joint: {finger: {"mcp", "pip", "dip"}} (thumb: CMC/MCP/IP order)."""
    return {name: finger_flexion(world, name) for name in FINGERS}


@dataclass
class HandMotion:
    dx_cm: float          # camera right
    dy_cm: float          # camera down
    dz_cm: float          # away from camera
    twist: float          # degrees about the hand's long axis (pronation-like)
    pitch: float          # degrees about the thumb-side axis (flexion-like)
    yaw: float            # degrees about the palm normal (deviation-like)


class HandPoseZero:
    """Hand position and orientation relative to a zero pose captured with `zero()`."""

    def __init__(self):
        self.R0 = self.t0 = None

    @property
    def is_set(self) -> bool:
        return self.R0 is not None

    def zero(self, R_hand_cam: np.ndarray, t_cam: np.ndarray):
        self.R0, self.t0 = R_hand_cam.copy(), np.asarray(t_cam, dtype=float).copy()

    def relative(self, R_hand_cam: np.ndarray, t_cam: np.ndarray) -> HandMotion | None:
        if not self.is_set:
            return None
        d = (np.asarray(t_cam) - self.t0) * 100.0
        a, b, c = euler_yxz(self.R0.T @ R_hand_cam)
        return HandMotion(float(d[0]), float(d[1]), float(d[2]),
                          float(np.degrees(a)), float(np.degrees(b)), float(np.degrees(c)))

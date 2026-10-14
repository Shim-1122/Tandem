"""Joint angle math on 3D landmarks (hand: 21 points, pose: 33 points).

All functions take numpy arrays of shape (N, 3). Use MediaPipe *world* landmarks
(meters, hip or hand centered) rather than normalized image coordinates, because
angles computed in image space are distorted by perspective and aspect ratio.
"""
from __future__ import annotations

import math

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
FINGERS = {  # name: (MCP, PIP, DIP, TIP); thumb uses (CMC, MCP, IP, TIP)
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# MediaPipe pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24


def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = math.sqrt(float(np.dot(v, v))) if v.ndim == 1 else float(np.linalg.norm(v))
    return v / n if n > eps else np.zeros_like(v)


def cross3(a, b) -> np.ndarray:
    """Cross product of two 3-vectors; about 10 times faster than np.cross for single vectors, which matters
    because the kinematics take dozens of them per frame."""
    a0, a1, a2 = np.asarray(a, dtype=float).tolist()
    b0, b1, b2 = np.asarray(b, dtype=float).tolist()
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def joint_angle(a, b, c) -> float:
    """Interior angle at b in degrees between segments b->a and b->c (180 = straight)."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    u, v = unit(a - b), unit(c - b)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def finger_flexion(hand: np.ndarray, finger: str) -> dict[str, float]:
    """Flexion in degrees at the two bending joints of a finger (0 = straight).

    Returns {"mcp": ..., "pip": ..., "dip": ...}. MCP flexion is measured against the
    wrist to MCP direction, so it needs the wrist landmark.
    """
    mcp, pip, dip, tip = FINGERS[finger]
    return {
        "mcp": 180.0 - joint_angle(hand[WRIST], hand[mcp], hand[pip]),
        "pip": 180.0 - joint_angle(hand[mcp], hand[pip], hand[dip]),
        "dip": 180.0 - joint_angle(hand[pip], hand[dip], hand[tip]),
    }


def body_frame(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (right, up, forward) unit vectors of the torso from shoulders and hips.

    right = person's right-hand direction, up = hips to shoulders (orthogonalized),
    forward = up x right (a person facing +north with right = east and up = +z).
    """
    right = unit(pose[R_SHOULDER] - pose[L_SHOULDER])
    mid_sh = (pose[R_SHOULDER] + pose[L_SHOULDER]) / 2
    mid_hip = (pose[R_HIP] + pose[L_HIP]) / 2
    up = mid_sh - mid_hip
    up = unit(up - np.dot(up, right) * right)
    forward = unit(cross3(up, right))
    return right, up, forward


def arm_angles(pose: np.ndarray, side: str = "right") -> dict[str, float]:
    """Shoulder and elbow angles for one arm, in degrees, in the torso frame.

    shoulder_flexion: upper arm raised forward (0 hanging, +90 horizontal forward, - backward)
    shoulder_abduction: angle of the upper arm out of the sagittal plane (0 in the plane, +90 straight outward)
    shoulder_elevation: total angle of the upper arm from hanging, any direction
    elbow_flexion: 0 straight, ~145 fully bent

    Flexion is the angle in the sagittal plane and abduction the angle out of it, so a pure
    forward raise reads (90, 0) and a pure sideways raise reads (0, 90). With combined motion the
    two are not independent Euler angles; the arm pointing straight up makes flexion undefined
    (it flips between +180 and -180). The pneumatic arm will need a proper shoulder model later.
    """
    s, e, w = ((L_SHOULDER, L_ELBOW, L_WRIST) if side == "left" else (R_SHOULDER, R_ELBOW, R_WRIST))
    right, up, fwd = body_frame(pose)
    u = unit(pose[e] - pose[s])
    out_sign = 1.0 if side == "right" else -1.0
    down = -np.dot(u, up) + 0.0  # + 0.0 turns -0.0 into 0.0 so atan2 does not return 180
    return {
        "shoulder_flexion": float(np.degrees(np.arctan2(np.dot(u, fwd), down))),
        "shoulder_abduction": float(np.degrees(np.arcsin(np.clip(out_sign * np.dot(u, right), -1.0, 1.0)))),
        "shoulder_elevation": float(np.degrees(np.arccos(np.clip(down, -1.0, 1.0)))),
        "elbow_flexion": 180.0 - joint_angle(pose[s], pose[e], pose[w]),
    }

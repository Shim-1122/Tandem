"""Finger kinematics: joint angles of all five digits from 21 hand landmarks, and the reverse.

Angles are measured in a palm frame built from the wrist, index knuckle and pinky knuckle
(x toward the thumb side, y toward the wrist, z out of the back of the hand). A left hand is mirrored
into a right hand first, so every sign below means the same thing on both hands.

Index, middle, ring, pinky (4 angles each, degrees):
  mcp_flex   knuckle flexion, toward the palm positive (hyperextension negative)
  mcp_abd    knuckle side-to-side, toward the thumb side positive
  pip        middle joint flexion
  dip        end joint flexion
Each finger's zero is straight, in line with its own metacarpal (wrist to knuckle line).

Thumb (5 angles, degrees):
  cmc_palmar   metacarpal raised out of the palm plane toward the palm side (palmar abduction)
  cmc_radial   metacarpal swung away from the index finger within the palm plane (radial abduction)
  cmc_rotation rotation of the thumb about its own metacarpal (opposition positive). Zero means the
               thumb bends across the palm toward the pinky side; +90 means it bends straight toward the
               palm. Only observable while the thumb is bent; otherwise the last value is held.
  mcp, ip      flexion of the thumb's two hinge joints

MediaPipe landmark noise is a few millimeters over phalanges 2 to 5 cm long, so expect several degrees
of noise per joint, and PIP/DIP readings that are partly coupled through the model's hand prior.
"""
from __future__ import annotations

import numpy as np

from .geometry import unit
from .kinematics import rot_x, rot_y, rot_z

FINGERS4 = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
THUMB = (1, 2, 3, 4)
FINGER_KEYS = ["mcp_flex", "mcp_abd", "pip", "dip"]
THUMB_KEYS = ["cmc_palmar", "cmc_radial", "cmc_rotation", "mcp", "ip"]
_EX = np.array([1.0, 0.0, 0.0])
_BEND_MIN = np.sin(np.radians(8.0))


def mirror_if_left(world: np.ndarray, side: str) -> np.ndarray:
    w = np.asarray(world, dtype=float)
    return w * np.array([-1.0, 1.0, 1.0]) if side == "left" else w


def palm_frame(world: np.ndarray) -> np.ndarray:
    """Columns: x thumb side, y proximal (knuckles to wrist), z = x cross y (back of a right hand)."""
    y = unit(world[0] - world[9])
    x = world[5] - world[17]
    x = unit(x - np.dot(x, y) * y)
    return np.column_stack([x, y, np.cross(x, y)])


def _finger_base(mcp_palm: np.ndarray) -> np.ndarray:
    """Columns u (along the metacarpal, distal), v (palmar), w = u x v (thumb side)."""
    u = unit(mcp_palm)
    v = np.array([0.0, 0.0, -1.0])
    v = unit(v - np.dot(v, u) * u)
    return np.column_stack([u, v, np.cross(u, v)])


_THUMB_BASE = np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _hinge(R: np.ndarray, d: np.ndarray) -> float:
    """Signed flexion of bone direction d about the w axis of frame R (radians)."""
    q = R.T @ d
    return float(np.arctan2(q[1], q[0]))


class HandKinematics:
    """Solves finger angles for one hand. Keeps the thumb rotation across frames where it is unobservable."""

    def __init__(self, side: str = "right"):
        self.side = side
        self._tau = -np.pi / 2           # thumb bend direction in its base frame (reported rotation 0)

    def palm_coords(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(P, H): landmarks in palm coordinates (wrist at origin) and the palm frame in (mirrored) world axes."""
        w = mirror_if_left(world, self.side)
        H = palm_frame(w)
        return (w - w[0]) @ H, H

    def solve(self, world: np.ndarray) -> dict:
        """Joint angles in degrees: {"index": {...}, ..., "thumb": {...}}."""
        P, _ = self.palm_coords(world)
        out = {}
        for name, (m, p, d, t) in FINGERS4.items():
            B = _finger_base(P[m])
            q = B.T @ unit(P[p] - P[m])
            flex, abd = np.arctan2(q[1], q[0]), np.arcsin(np.clip(q[2], -1.0, 1.0))
            R1 = B @ rot_z(flex) @ rot_y(-abd)
            pip = _hinge(R1, unit(P[d] - P[p]))
            R2 = R1 @ rot_z(pip)
            dip = _hinge(R2, unit(P[t] - P[d]))
            out[name] = dict(zip(FINGER_KEYS, np.degrees([flex, abd, pip, dip]).tolist()))

        c, m, i, t = THUMB
        q = _THUMB_BASE.T @ unit(P[m] - P[c])
        palmar, radial = np.arctan2(q[1], q[0]), np.arcsin(np.clip(q[2], -1.0, 1.0))
        R1p = _THUMB_BASE @ rot_z(palmar) @ rot_y(-radial)
        bend = R1p.T @ unit(P[t] - P[m])
        if np.hypot(bend[1], bend[2]) > _BEND_MIN:
            self._tau = float(np.arctan2(bend[2], bend[1]))
        R1 = R1p @ rot_x(self._tau)
        mcp = _hinge(R1, unit(P[i] - P[m]))
        R2 = R1 @ rot_z(mcp)
        ip = _hinge(R2, unit(P[t] - P[i]))
        rotation = (np.degrees(self._tau) + 90.0 + 180.0) % 360.0 - 180.0
        out["thumb"] = dict(zip(THUMB_KEYS, [float(np.degrees(palmar)), float(np.degrees(radial)),
                                             float(rotation), float(np.degrees(mcp)), float(np.degrees(ip))]))
        return out


def hand_geometry(world: np.ndarray, side: str = "right") -> dict:
    """Measured palm-frame geometry for forward kinematics: knuckle positions and bone lengths."""
    P, _ = HandKinematics(side).palm_coords(world)
    geo = {"P": P}
    for name, (m, p, d, t) in {**FINGERS4, "thumb": THUMB}.items():
        geo[name] = dict(base=P[m] if name != "thumb" else P[THUMB[0]],
                         lengths=[float(np.linalg.norm(P[b] - P[a])) for a, b in ((m, p), (p, d), (d, t))]
                         if name != "thumb" else
                         [float(np.linalg.norm(P[b] - P[a])) for a, b in ((1, 2), (2, 3), (3, 4))])
    return geo


def hand_fk(angles: dict, geometry: dict) -> np.ndarray:
    """(21, 3) landmark positions in palm coordinates rebuilt from joint angles and measured geometry."""
    out = np.zeros((21, 3))
    for name, (m, p, d, t) in FINGERS4.items():
        a, g = angles[name], geometry[name]
        L1, L2, L3 = g["lengths"]
        B = _finger_base(g["base"])
        R1 = B @ rot_z(np.radians(a["mcp_flex"])) @ rot_y(-np.radians(a["mcp_abd"]))
        R2 = R1 @ rot_z(np.radians(a["pip"]))
        R3 = R2 @ rot_z(np.radians(a["dip"]))
        out[m] = g["base"]
        out[p] = out[m] + L1 * (R1 @ _EX)
        out[d] = out[p] + L2 * (R2 @ _EX)
        out[t] = out[d] + L3 * (R3 @ _EX)
    a, g = angles["thumb"], geometry["thumb"]
    L1, L2, L3 = g["lengths"]
    tau = np.radians(a["cmc_rotation"] - 90.0)
    R1 = _THUMB_BASE @ rot_z(np.radians(a["cmc_palmar"])) @ rot_y(-np.radians(a["cmc_radial"])) @ rot_x(tau)
    R2 = R1 @ rot_z(np.radians(a["mcp"]))
    R3 = R2 @ rot_z(np.radians(a["ip"]))
    c, m, i, t = THUMB
    out[c] = g["base"]
    out[m] = out[c] + L1 * (R1 @ _EX)
    out[i] = out[m] + L2 * (R2 @ _EX)
    out[t] = out[i] + L3 * (R3 @ _EX)
    return out


def palm_to_world(P: np.ndarray, world: np.ndarray, side: str = "right") -> np.ndarray:
    """Map palm-coordinate points back into the original (unmirrored) world axes of `world`."""
    w = mirror_if_left(world, side)
    H = palm_frame(w)
    pts = w[0] + P @ H.T
    return mirror_if_left(pts, side)


def flat(angles: dict) -> dict:
    """{"index_mcp_flex": ..., "thumb_ip": ...} for CSV columns."""
    return {f"{n}_{k}": v for n, d in angles.items() for k, v in d.items()}

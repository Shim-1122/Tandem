"""Seven joint angles of one arm from pose and hand landmarks.

Shoulder (3 DOF), elbow (1 DOF) and wrist (3 DOF), all in degrees, solved from segment
orientations rather than single-plane projections.

Frames (columns are unit vectors, order X, Y, Z, right handed, X = Y x Z):
  thorax    X forward, Y up, Z right.
  humerus   Y from elbow to shoulder, Z elbow flexion axis (right when the arm hangs and the
            forearm points forward), X = Y x Z. Identity relative to the thorax at the neutral
            pose: arm hanging, elbow bent 90 degrees, forearm pointing forward.
  forearm   humerus frame rotated by the elbow flexion about Z. This is the forearm WITHOUT
            pronation, so pronation shows up as the hand frame rotating about the forearm Y.
  hand      Y from middle knuckle to wrist (proximal), X toward the thumb side
            (index knuckle minus pinky knuckle), Z = X x Y.

The shoulder is three rotations applied in this order, each about an axis of the previous result
(an intrinsic Z-X-Y sequence, like a pitch, roll, yaw shoulder on a humanoid robot):
  shoulder_flexion   sagittal plane, about the lateral axis: arm raised forward is positive,
                     range -180 to 180 (180 = arm overhead)
  shoulder_abduction coronal plane, about the forward axis: arm raised outward is positive
                     (both arms), range -90 to 90
  shoulder_rotation  transverse plane, about the arm's long axis: internal rotation positive
                     (hand moves toward the midline with the elbow bent 90 degrees)
An arm raised past 90 degrees to the side reads as flexion near 180 with a smaller abduction.
Wrist angles use the Y-X-Z Euler sequence of the hand relative to the forearm:
  forearm_rotation   pronation positive, supination negative (about the forearm long axis)
  wrist_flexion      flexion positive, extension negative
  wrist_deviation    radial deviation positive, ulnar negative

A left arm is mirrored across the torso's sagittal plane and solved as a right arm, so the
sign conventions above hold for both sides.

Observability limits, reported through the `valid` flags and held at the last good value:
  * Shoulder rotation and forearm rotation cannot be separated when the elbow is nearly straight
    (the elbow flexion plane is undefined). Below `min_elbow_deg` the shoulder rotation is held
    and the hand's total twist is attributed to the forearm.
  * With the arm held horizontally out to the side (abduction near +-90 degrees) the sagittal and
    transverse axes line up (gimbal lock): only flexion + rotation is observable. Within `pole_deg`
    of that direction the rotation is held at its last good value and flexion takes the rest.
  * Wrist flexion of +-90 degrees is a gimbal lock of the Euler sequence, beyond the usual range.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .geometry import (L_ELBOW, cross3, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST, unit)

# Pose landmark indices for the hand points MediaPipe pose provides (pinky, index) per side
POSE_PINKY = {"left": 17, "right": 18}
POSE_INDEX = {"left": 19, "right": 20}
_SIDE = {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)}


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a to unit vector b by the shortest path (Rodrigues)."""
    v = cross3(a, b)
    c = float(np.dot(a, b))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def euler_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """Angles (a, b, c) in radians with R = Ry(a) Rx(b) Rz(c); b is limited to +-90 degrees."""
    b = np.arcsin(np.clip(-R[1, 2], -1.0, 1.0))
    return float(np.arctan2(R[0, 2], R[2, 2])), float(b), float(np.arctan2(R[1, 0], R[1, 1]))


def thorax_frame(pose: np.ndarray, up_hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(R_t, origin): columns X forward, Y up, Z right, and the shoulder midpoint.

    Up normally comes from the hips. Pass `up_hint` (an up direction in world axes, e.g. the camera's
    vertical) when the hips are out of view and their landmarks are guesses.
    """
    z = unit(pose[R_SHOULDER] - pose[L_SHOULDER])
    mid_sh = (pose[R_SHOULDER] + pose[L_SHOULDER]) / 2
    y = mid_sh - (pose[R_HIP] + pose[L_HIP]) / 2 if up_hint is None else np.asarray(up_hint, dtype=float)
    y = unit(y - np.dot(y, z) * z)
    x = cross3(y, z)
    return np.column_stack([x, y, z]), mid_sh


@dataclass
class ArmAngles:
    shoulder_flexion: float = np.nan
    shoulder_abduction: float = np.nan
    shoulder_rotation: float = np.nan
    elbow_flexion: float = np.nan
    forearm_rotation: float = np.nan
    wrist_flexion: float = np.nan
    wrist_deviation: float = np.nan
    shoulder_rotation_valid: bool = False   # False: held value, elbow too straight or arm straight up
    wrist_valid: bool = False               # False: no hand orientation available this frame
    elbow_predicted: bool = False           # True: elbow was out of view and estimated (see elbow.py)

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class ArmKinematics:
    """Stateful solver for one arm. Call `solve` once per frame, in time order."""

    def __init__(self, side: str = "right", min_elbow_deg: float = 20.0, pole_deg: float = 5.0):
        if side not in _SIDE:
            raise ValueError("side must be 'left' or 'right'")
        self.side, self.min_elbow, self.pole = side, min_elbow_deg, pole_deg
        self._tau = 0.0       # last valid shoulder rotation, radians
        self._phi = 0.0       # last valid shoulder flexion, radians

    def reset(self):
        self._tau = self._phi = 0.0

    def _to_thorax(self, v: np.ndarray, R_t: np.ndarray) -> np.ndarray:
        c = R_t.T @ v
        if self.side == "left":
            c = c * np.array([1.0, 1.0, -1.0])   # mirror the left arm into a right arm
        return c

    def solve(self, pose: np.ndarray, hand: np.ndarray | None = None,
              up_hint: np.ndarray | None = None) -> ArmAngles:
        """pose: (33, 3) world landmarks. hand: (21, 3) hand world landmarks or None.

        With hand=None the hand orientation comes from the pose's own pinky and index points,
        which is noisier. Hand landmarks and pose landmarks are assumed to share axis directions
        (both aligned with the camera); check this on real data with `frame_agreement_deg`.
        """
        R_t, origin = thorax_frame(pose, up_hint)
        s_i, e_i, w_i = _SIDE[self.side]
        S, E, W = (self._to_thorax(pose[i] - origin, R_t) for i in (s_i, e_i, w_i))

        yh = unit(S - E)                       # humerus long axis, elbow to shoulder
        fd = unit(W - E)                       # forearm direction, elbow to wrist
        # the interior angle at the elbow between yh (toward shoulder) and fd (toward wrist) is 180 - flexion
        elbow = 180.0 - np.degrees(np.arccos(np.clip(np.dot(yh, fd), -1.0, 1.0)))

        out = ArmAngles(elbow_flexion=float(elbow))
        beta = float(np.arcsin(np.clip(yh[2], -1.0, 1.0)))       # coronal rotation, = -abduction
        near_pole = abs(yh[2]) > np.cos(np.radians(self.pole))
        zh = cross3(fd, yh)
        plane_ok = elbow >= self.min_elbow and np.linalg.norm(zh) > np.sin(np.radians(self.min_elbow))
        R_h = None
        if plane_ok:
            zh = unit(zh)
            R_h = np.column_stack([cross3(yh, zh), yh, zh])

        if not near_pole:
            phi = float(np.arctan2(-yh[0], yh[1]))                # sagittal rotation from the long axis
            swing = rot_z(phi) @ rot_x(beta)
            if R_h is not None:
                T = swing.T @ R_h                                  # = Ry(tau)
                self._tau = float(np.arctan2(T[0, 2], T[0, 0]))
                out.shoulder_rotation_valid = True
            else:
                R_h = swing @ rot_y(self._tau)                     # elbow too straight: hold rotation
            self._phi = phi
        else:
            # long axis along the lateral axis: only flexion + rotation (beta > 0) or rotation - flexion
            # (beta < 0) is observable. Hold rotation, let flexion carry the rest.
            if R_h is not None:
                total = float(np.arctan2(R_h[0, 2], R_h[0, 0]))
                self._phi = total - self._tau if beta > 0 else self._tau - total
            phi = self._phi
            if R_h is None:
                R_h = rot_z(phi) @ rot_x(beta) @ rot_y(self._tau)

        out.shoulder_flexion = float(np.degrees(phi))
        out.shoulder_abduction = float(np.degrees(-beta))
        out.shoulder_rotation = float(np.degrees(self._tau))

        # forearm frame without pronation
        R_f = R_h @ rot_z(np.radians(elbow))

        # hand frame
        if hand is not None:
            hy, hx = hand[0] - hand[9], hand[5] - hand[17]
        else:
            p, ix, pk = pose[w_i], pose[POSE_INDEX[self.side]], pose[POSE_PINKY[self.side]]
            hy, hx = p - (ix + pk) / 2, ix - pk
        hy, hx = self._to_thorax(hy, R_t), self._to_thorax(hx, R_t)
        y_hand = unit(hy)
        x_hand = hx - np.dot(hx, y_hand) * y_hand
        if np.linalg.norm(y_hand) > 0 and np.linalg.norm(x_hand) > 1e-9:
            x_hand = unit(x_hand)
            R_hand = np.column_stack([x_hand, y_hand, cross3(x_hand, y_hand)])
            ps, fe, dev = euler_yxz(R_f.T @ R_hand)
            out.forearm_rotation, out.wrist_flexion, out.wrist_deviation = (float(np.degrees(v)) for v in (ps, fe, dev))
            out.wrist_valid = True
        return out


def frame_agreement_deg(pose: np.ndarray, hand: np.ndarray, side: str = "right") -> dict[str, float]:
    """Checks that hand and pose world landmarks share axis directions. Read in raw world axes.

    hand_vs_forearm_deg: angle between the hand's long axis (hand model) and the forearm (pose elbow
        to wrist). With the wrist held straight (no flexion or deviation) this should be small,
        under about 20 degrees. Elbow and wrist are among the pose model's best points, so this is
        the check to trust.
    long_axis_deg, palm_normal_deg: hand model versus the pose model's own pinky and index points.
        Those pose points are rough, so large values here are expected and not conclusive.
    """
    s_i, e_i, w_i = _SIDE[side]
    p, ix, pk = pose[w_i], pose[POSE_INDEX[side]], pose[POSE_PINKY[side]]
    py, px = p - (ix + pk) / 2, ix - pk
    hy, hx = hand[0] - hand[9], hand[5] - hand[17]

    def ang(u, v):
        return float(np.degrees(np.arccos(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))))

    return {"hand_vs_forearm_deg": ang(hand[9] - hand[0], pose[w_i] - pose[e_i]),
            "long_axis_deg": ang(py, hy), "palm_normal_deg": ang(cross3(px, py), cross3(hx, hy))}


def up_from_neutral(pose: np.ndarray, side: str = "right") -> np.ndarray:
    """Torso up direction (world axes) captured while the upper arm hangs straight down at the side.

    For desk use with the hips out of view: sit upright, let the tracked upper arm hang vertically
    beside the torso (elbow bent 90 degrees, forearm forward, thumb up), and capture this once. Pass the
    result as `up_hint` to `ArmKinematics.solve`. It assumes the camera and your torso stay put afterwards.
    side="both" averages both upper arms (both hanging), which cancels small left/right differences.
    """
    sides = ["left", "right"] if side == "both" else [side]
    return unit(sum(unit(pose[_SIDE[sd][0]] - pose[_SIDE[sd][1]]) for sd in sides))


# Approximate typical active range of motion, degrees. Readings well outside these usually mean bad tracking.
TYPICAL_RANGE = {
    "elbow_flexion": (-10.0, 155.0),
    "forearm_rotation": (-95.0, 95.0),
    "wrist_flexion": (-75.0, 85.0),
    "wrist_deviation": (-40.0, 25.0),
}


def out_of_range(angles: "ArmAngles", margin: float = 10.0) -> list[str]:
    """Names of angles more than `margin` degrees outside TYPICAL_RANGE (NaN values are skipped)."""
    bad = []
    for k, (lo, hi) in TYPICAL_RANGE.items():
        v = getattr(angles, k)
        if np.isfinite(v) and (v < lo - margin or v > hi + margin):
            bad.append(k)
    return bad


def arm_visible(visibility, side: str = "right", thresh: float = 0.6) -> list[str]:
    """Which of shoulder, elbow, wrist are NOT confidently visible (MediaPipe pose visibility)."""
    if visibility is None:
        return ["shoulder", "elbow", "wrist"]
    s_i, e_i, w_i = _SIDE[side]
    return [n for n, i in (("shoulder", s_i), ("elbow", e_i), ("wrist", w_i)) if visibility[i] < thresh]

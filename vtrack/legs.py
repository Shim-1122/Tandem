"""Legs and core: 10 degrees of freedom per leg and 3 in the core, from MediaPipe pose landmarks.

Per leg (degrees):
  hip_flexion         thigh forward +, backward (extension) -           about the pelvis lateral axis
  hip_abduction       thigh outward +, inward (adduction) -             about the forward axis
  hip_rotation        internal +, external - (about the thigh's long axis)
  pelvic_hike         this side of the pelvis raised +, dropped -       (pelvic obliquity)
  knee_flexion        0 straight, + bent
  ankle_dorsiflexion  toes up +, pointed (plantarflexion) -
  ankle_inversion     sole turned inward +, outward (eversion) -
  ankle_abduction     toes turned outward +, inward (adduction) -
  foot_pitch          foot relative to the floor: toes up +
  foot_roll           foot relative to the floor: outer edge down (sole facing inward) +
Core, the chest (shoulders) relative to the pelvis (hips):
  core_flexion        bending forward +, backward -
  core_lateral_bend   bending to your right +
  core_rotation       chest turned to your left + (same direction as head yaw)

Zero is standing upright, legs straight, feet flat and pointing forward. Left and right legs use the same
signs (the left leg is mirrored internally). Hip, ankle and core angles are intrinsic Z-X-Y sequences
(lateral axis, then forward, then long axis), the order the ISB recommends for hip, ankle and spine.

Frames (columns X forward, Y up, Z right, right handed):
  pelvis  Z from the left hip joint to the right one; Y is "up" (gravity) made perpendicular to Z.
          MediaPipe gives only the two hip joints for the pelvis, so its forward/backward tilt cannot be
          seen: the pelvis is taken as level front to back, hip flexion is measured from vertical, and
          core flexion includes any pelvic tilt. Side to side tilt (pelvic hike) and turning are measured.
  thigh   Y from knee to hip, Z the knee's hinge axis (from the bend between thigh and shank).
  shank   thigh frame turned by the knee flexion about Z.
  foot    X from heel to toe, Y toward the ankle (perpendicular to X), Z = X x Y.

"Up" is the camera's vertical unless calibrated (stand upright and press z; see `up_from_standing`).

Observability limits (reported in `hip_rotation_source`):
  * With the knee nearly straight (under `min_knee_deg`) the knee's hinge axis is undefined, so hip
    rotation is taken from the direction the foot points, assuming the ankle itself is not turned
    (ankle_abduction then reads 0). Without the foot in view it is held at its last value.
  * Ankle and foot angles need the heel and toe in view.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .geometry import cross3, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, unit
from .kinematics import rot_x, rot_y, rot_z, thorax_frame

L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32            # MediaPipe "foot index": the tip of the foot
LEG = {"left": (L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_TOE), "right": (R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_TOE)}
LEG_FIELDS = ["hip_flexion", "hip_abduction", "hip_rotation", "pelvic_hike", "knee_flexion",
              "ankle_dorsiflexion", "ankle_inversion", "ankle_abduction", "foot_pitch", "foot_roll"]
CORE_FIELDS = ["core_flexion", "core_lateral_bend", "core_rotation"]
CAMERA_UP = np.array([0.0, -1.0, 0.0])            # MediaPipe world axes: y points down


def euler_zxy(R: np.ndarray) -> tuple[float, float, float]:
    """(z, x, y) radians with R = Rz(z) Rx(x) Ry(y); x is limited to +-90 degrees."""
    x = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
    return float(np.arctan2(-R[0, 1], R[1, 1])), float(x), float(np.arctan2(-R[2, 0], R[2, 2]))


def pelvis_frame(pose: np.ndarray, up: np.ndarray = CAMERA_UP):
    """(R_p, hip midpoint, obliquity radians): columns forward, up, right; obliquity + = right hip higher."""
    z = unit(pose[R_HIP] - pose[L_HIP])
    u = unit(np.asarray(up, dtype=float))
    y = unit(u - np.dot(u, z) * z)
    obliquity = float(np.arcsin(np.clip(np.dot(z, u), -1.0, 1.0)))
    return np.column_stack([cross3(y, z), y, z]), (pose[R_HIP] + pose[L_HIP]) / 2, obliquity


@dataclass
class LegAngles:
    hip_flexion: float = np.nan
    hip_abduction: float = np.nan
    hip_rotation: float = np.nan
    pelvic_hike: float = np.nan
    knee_flexion: float = np.nan
    ankle_dorsiflexion: float = np.nan
    ankle_inversion: float = np.nan
    ankle_abduction: float = np.nan
    foot_pitch: float = np.nan
    foot_roll: float = np.nan
    hip_rotation_source: str = "none"      # "knee", "foot" (knee straight; ankle_abduction is then 0) or "held"
    ankle_valid: bool = False              # heel and toe were in view

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class CoreAngles:
    core_flexion: float = np.nan
    core_lateral_bend: float = np.nan
    core_rotation: float = np.nan


def core_angles(pose: np.ndarray, up: np.ndarray = CAMERA_UP) -> CoreAngles:
    """Chest (shoulder line and trunk axis) relative to the pelvis."""
    R_p = pelvis_frame(pose, up)[0]
    R_t = thorax_frame(pose)[0]                        # up from the hips to the shoulders
    z, x, y = euler_zxy(R_p.T @ R_t)
    return CoreAngles(float(np.degrees(-z)), float(np.degrees(x)), float(np.degrees(y)))


class LegKinematics:
    """Stateful solver for one leg. Call `solve` once per frame, in time order."""

    def __init__(self, side: str = "right", min_knee_deg: float = 20.0, pole_deg: float = 5.0):
        if side not in LEG:
            raise ValueError("side must be 'left' or 'right'")
        self.side, self.min_knee, self.pole = side, min_knee_deg, pole_deg
        self.mirror = np.array([1.0, 1.0, -1.0 if side == "left" else 1.0])
        self.foot_correction = np.eye(3)      # landmark foot frame -> anatomical foot frame (see calibrate)
        self._tau = 0.0                       # last hip rotation, radians
        self._phi = 0.0

    def reset(self):
        self._tau = self._phi = 0.0

    def _local(self, pose, up):
        R_p, origin, _ = pelvis_frame(pose, up)
        pts = [(R_p.T @ (pose[i] - origin)) * self.mirror for i in LEG[self.side]]
        return pts, (R_p.T @ unit(np.asarray(up, dtype=float))) * self.mirror

    @staticmethod
    def _foot_frame(A, Hl, T):
        x = unit(T - Hl)
        a = A - Hl
        y = a - np.dot(a, x) * x
        if np.linalg.norm(x) < 0.5 or np.linalg.norm(y) < 1e-3:
            return None
        y = unit(y)
        return np.column_stack([x, y, cross3(x, y)])

    def solve(self, pose: np.ndarray, up: np.ndarray = CAMERA_UP, foot: bool = True) -> LegAngles:
        """pose: (33, 3) world landmarks; up: up direction in the same axes; foot: heel and toe in view."""
        (H, K, A, Hl, T), U = self._local(pose, up)
        out = LegAngles(pelvic_hike=float(np.degrees(np.arcsin(np.clip(U[2], -1.0, 1.0)))))
        yt, sd = unit(H - K), unit(A - K)
        knee = 180.0 - float(np.degrees(np.arccos(np.clip(np.dot(yt, sd), -1.0, 1.0))))
        out.knee_flexion = knee

        R_foot = self._foot_frame(A, Hl, T) if foot else None
        if R_foot is not None:
            R_foot = R_foot @ self.foot_correction

        beta = float(np.arcsin(np.clip(yt[2], -1.0, 1.0)))          # = -abduction
        near_pole = abs(yt[2]) > np.cos(np.radians(self.pole))
        zt = cross3(yt, sd)                                        # knee hinge axis, right
        plane_ok = knee >= self.min_knee and np.linalg.norm(zt) > np.sin(np.radians(self.min_knee))
        R_meas = None
        if plane_ok:
            zt = unit(zt)
            R_meas = np.column_stack([cross3(yt, zt), yt, zt])

        source = "held"
        if not near_pole:
            phi = float(np.arctan2(-yt[0], yt[1]))
            swing = rot_z(phi) @ rot_x(beta)
            if R_meas is not None:
                M = swing.T @ R_meas                                 # = Ry(tau)
                self._tau, source = float(np.arctan2(M[0, 2], M[0, 0])), "knee"
            elif R_foot is not None:
                # Knee straight: the thigh's forward axis is set to the foot's heading. With no ankle turn
                # this is exact for any knee bend, ankle dorsiflexion or inversion (see tests).
                v = swing.T @ R_foot[:, 0]
                if np.hypot(v[0], v[2]) > 0.2:
                    self._tau, source = float(np.arctan2(-v[2], v[0])), "foot"
            self._phi = phi
        elif R_meas is not None:
            # thigh along the lateral axis (gimbal lock): hold rotation, flexion takes the rest
            total = float(np.arctan2(R_meas[0, 2], R_meas[0, 0]))
            self._phi = total - self._tau if beta > 0 else self._tau - total
        phi, tau = self._phi, self._tau
        out.hip_flexion, out.hip_abduction, out.hip_rotation = (float(np.degrees(v)) for v in (phi, -beta, tau))
        out.hip_rotation_source = source

        if R_foot is not None:
            R_thigh = rot_z(phi) @ rot_x(beta) @ rot_y(tau)
            R_shank = R_thigh @ rot_z(np.radians(-knee))
            d, i, a = euler_zxy(R_shank.T @ R_foot)
            out.ankle_dorsiflexion, out.ankle_inversion, out.ankle_abduction = (
                float(np.degrees(d)), float(np.degrees(i)), float(np.degrees(-a)))
            xf, yf, zf = R_foot.T
            out.foot_pitch = float(np.degrees(np.arcsin(np.clip(np.dot(xf, U), -1.0, 1.0))))
            out.foot_roll = float(np.degrees(np.arctan2(-np.dot(zf, U), np.dot(yf, U))))
            out.ankle_valid = True
        return out

    def calibrate(self, pose: np.ndarray, up: np.ndarray) -> bool:
        """Standing upright, leg straight, foot flat and pointing forward: from now on this foot reads zero.

        Removes the offset between MediaPipe's heel/toe/ankle points and the foot's real axes. Returns
        False (and changes nothing) unless the knee is nearly straight and the foot frame is defined.
        """
        (H, K, A, Hl, T), _ = self._local(pose, up)
        R_foot = self._foot_frame(A, Hl, T)
        yt, sd = unit(H - K), unit(A - K)
        knee = 180.0 - float(np.degrees(np.arccos(np.clip(np.dot(yt, sd), -1.0, 1.0))))
        if R_foot is None or knee > 15.0:
            return False
        # the shank's frame when the ankle is neutral: shank axis up, heading taken from the foot
        y = unit(K - A)
        x = R_foot[:, 0] - np.dot(R_foot[:, 0], y) * y
        if np.linalg.norm(x) < 0.3:
            return False
        x = unit(x)
        R_shank = np.column_stack([x, y, cross3(x, y)])
        self.foot_correction = R_foot.T @ R_shank
        return True

    def clear_calibration(self):
        self.foot_correction = np.eye(3)


def up_from_standing(pose: np.ndarray) -> np.ndarray:
    """Up direction (world axes) while standing upright with straight legs: ankles midpoint to hips midpoint."""
    return unit((pose[L_HIP] + pose[R_HIP]) / 2 - (pose[L_ANKLE] + pose[R_ANKLE]) / 2)


def legs_visible(visibility, side: str, thresh: float = 0.6, foot_thresh: float = 0.5) -> tuple[list[str], bool]:
    """(names of hip/knee/ankle points NOT in view, whether heel and toe are in view) for one leg."""
    if visibility is None:
        return ["hip", "knee", "ankle"], False
    h, k, a, hl, t = LEG[side]
    other_hip = L_HIP if side == "right" else R_HIP
    missing = [n for n, i in (("hip", h), ("knee", k), ("ankle", a)) if visibility[i] < thresh]
    if visibility[other_hip] < thresh and "hip" not in missing:
        missing.insert(0, "hip")
    return missing, bool(min(visibility[hl], visibility[t]) >= foot_thresh)


def core_visible(visibility, thresh: float = 0.6) -> bool:
    return visibility is not None and min(visibility[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)) >= thresh


LEG_RANGE = {                                   # typical active range, degrees; well outside = tracking problem
    "hip_flexion": (-30.0, 130.0),
    "hip_abduction": (-35.0, 50.0),
    "hip_rotation": (-50.0, 50.0),
    "knee_flexion": (-10.0, 150.0),
    "ankle_dorsiflexion": (-55.0, 30.0),
    "ankle_inversion": (-25.0, 40.0),
    "ankle_abduction": (-30.0, 30.0),
}


def leg_out_of_range(angles: LegAngles, margin: float = 10.0) -> list[str]:
    bad = []
    for k, (lo, hi) in LEG_RANGE.items():
        v = getattr(angles, k)
        if np.isfinite(v) and (v < lo - margin or v > hi + margin):
            bad.append(k)
    return bad

"""Forward kinematics: joint angles to landmarks. Used to test the solver and to drive a simulated arm."""
from __future__ import annotations

import numpy as np

from .geometry import L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST
from .kinematics import POSE_INDEX, POSE_PINKY, rot_x, rot_y, rot_z


def arm_chain(angles: dict, upper_arm=0.30, forearm=0.26, hand_len=0.09, knuckle_half=0.04) -> dict:
    """Right-arm chain in thorax axes (X forward, Y up, Z right), shoulder at (0, 0, 0.2).

    Returns points S, E, W, knuckle (middle), index, pinky and segment frames R_h (upper arm),
    R_f (forearm without pronation) and R_hand. Mirror Z for a left arm.
    """
    d = {k: np.radians(v) for k, v in angles.items()}
    R_h = rot_z(d["shoulder_flexion"]) @ rot_x(-d["shoulder_abduction"]) @ rot_y(d["shoulder_rotation"])
    R_f = R_h @ rot_z(d["elbow_flexion"])
    R_hand = R_f @ rot_y(d["forearm_rotation"]) @ rot_x(d["wrist_flexion"]) @ rot_z(d["wrist_deviation"])
    S = np.array([0.0, 0.0, 0.2])
    E = S - upper_arm * R_h[:, 1]
    W = E + forearm * (R_f @ np.array([0.0, -1.0, 0.0]))
    K = W - hand_len * R_hand[:, 1]
    return {"S": S, "E": E, "W": W, "K": K, "index": K + knuckle_half * R_hand[:, 0],
            "pinky": K - knuckle_half * R_hand[:, 0], "R_h": R_h, "R_f": R_f, "R_hand": R_hand}


def forward_arm(angles: dict, side: str = "right", body_rotation: np.ndarray | None = None,
                body_offset=(0.0, 0.0, 0.0), upper_arm=0.30, forearm=0.26, hand_len=0.09, knuckle_half=0.04):
    """Return (pose (33,3), hand (21,3)) for the given joint angles in degrees.

    angles keys: shoulder_flexion, shoulder_abduction, shoulder_rotation, elbow_flexion,
    forearm_rotation, wrist_flexion, wrist_deviation. Only landmarks the solver reads are filled.
    body_rotation maps thorax axes (forward, up, right) into the camera's world axes.
    """
    c = arm_chain(angles, upper_arm, forearm, hand_len, knuckle_half)
    E, W, mid_knuckle, idx, pnk = c["E"], c["W"], c["K"], c["index"], c["pinky"]

    mirror = np.array([1.0, 1.0, -1.0 if side == "left" else 1.0])
    R_w = np.eye(3) if body_rotation is None else body_rotation
    off = np.asarray(body_offset)

    def world(v):          # arm chain: mirrored for the left arm
        return R_w @ (v * mirror) + off

    def torso(v):          # torso landmarks are never mirrored
        return R_w @ np.asarray(v, dtype=float) + off

    pose = np.zeros((33, 3))
    pose[R_SHOULDER], pose[L_SHOULDER] = torso([0, 0, 0.2]), torso([0, 0, -0.2])
    pose[R_HIP], pose[L_HIP] = torso([0, -0.5, 0.1]), torso([0, -0.5, -0.1])
    s_i, e_i, w_i = (L_SHOULDER, L_ELBOW, L_WRIST) if side == "left" else (R_SHOULDER, R_ELBOW, R_WRIST)
    pose[e_i], pose[w_i] = world(E), world(W)   # the shoulder stays the torso point set above
    pose[POSE_INDEX[side]], pose[POSE_PINKY[side]] = world(idx), world(pnk)

    hand = np.zeros((21, 3))
    hand[0] = R_w @ ((W - W) * mirror)                   # hand-centered, wrist at the origin
    hand[9] = R_w @ ((mid_knuckle - W) * mirror)
    hand[5] = R_w @ ((idx - W) * mirror)
    hand[17] = R_w @ ((pnk - W) * mirror)
    return pose, hand


def leg_chain(angles: dict, thigh=0.42, shank=0.40, heel=(-0.06, -0.07, 0.0), toe=(0.17, -0.07, 0.0)) -> dict:
    """Right leg in pelvis axes (X forward, Y up, Z right) with the hip joint at the origin.

    angles: hip_flexion, hip_abduction, hip_rotation, knee_flexion, ankle_dorsiflexion, ankle_inversion,
    ankle_abduction (degrees; see legs.py for the signs). heel and toe are offsets from the ankle in the
    foot frame. Returns points H, K, A, heel, toe and frames R_thigh, R_shank, R_foot. Mirror Z for a left leg.
    """
    d = {k: np.radians(float(angles.get(k, 0.0))) for k in (
        "hip_flexion", "hip_abduction", "hip_rotation", "knee_flexion", "ankle_dorsiflexion", "ankle_inversion",
        "ankle_abduction")}
    R_thigh = rot_z(d["hip_flexion"]) @ rot_x(-d["hip_abduction"]) @ rot_y(d["hip_rotation"])
    R_shank = R_thigh @ rot_z(-d["knee_flexion"])
    R_foot = R_shank @ rot_z(d["ankle_dorsiflexion"]) @ rot_x(d["ankle_inversion"]) @ rot_y(-d["ankle_abduction"])
    H = np.zeros(3)
    K = H - thigh * R_thigh[:, 1]
    A = K - shank * R_shank[:, 1]
    return {"H": H, "K": K, "A": A, "heel": A + R_foot @ np.asarray(heel, float),
            "toe": A + R_foot @ np.asarray(toe, float), "R_thigh": R_thigh, "R_shank": R_shank, "R_foot": R_foot}


def core_rotation_matrix(core: dict) -> np.ndarray:
    """Chest frame in pelvis axes for core_flexion, core_lateral_bend, core_rotation (degrees)."""
    f, b, r = (np.radians(float(core.get(k, 0.0))) for k in ("core_flexion", "core_lateral_bend", "core_rotation"))
    return rot_z(-f) @ rot_x(b) @ rot_y(r)


def forward_body(legs: dict, core: dict | None = None, pelvic_hike: float = 0.0,
                 body_rotation: np.ndarray | None = None, body_offset=(0.0, 0.0, 0.0), hip_half=0.09,
                 trunk=0.50, shoulder_half=0.19, **leg_lengths) -> np.ndarray:
    """(33, 3) pose with shoulders, hips, knees, ankles, heels and toes for the given angles (degrees).

    legs: {"left": angles, "right": angles} as for `leg_chain`. pelvic_hike: right side of the pelvis
    raised (the left leg reads the opposite sign). body_rotation: a level frame (forward, up, right as
    columns) in world axes; the pelvis is that frame tilted by the hike. body_offset: hip midpoint.
    """
    from .legs import LEG
    R_level = np.eye(3) if body_rotation is None else np.asarray(body_rotation, float)
    U, right = R_level[:, 1], R_level[:, 2]
    o = np.radians(pelvic_hike)
    Z = np.cos(o) * right + np.sin(o) * U
    Y = -np.sin(o) * right + np.cos(o) * U
    R_p = np.column_stack([np.cross(Y, Z), Y, Z])
    origin = np.asarray(body_offset, float)
    pose = np.zeros((33, 3))
    for side, ang in legs.items():
        mirror = np.array([1.0, 1.0, -1.0 if side == "left" else 1.0])
        c = leg_chain(ang, **leg_lengths)
        hip = np.array([0.0, 0.0, hip_half])
        for idx, key in zip(LEG[side], ("H", "K", "A", "heel", "toe")):
            pose[idx] = origin + R_p @ ((hip + c[key]) * mirror)
    R_t = core_rotation_matrix(core or {})
    mid = origin + R_p @ (trunk * R_t[:, 1])
    pose[R_SHOULDER] = mid + shoulder_half * (R_p @ R_t[:, 2])
    pose[L_SHOULDER] = mid - shoulder_half * (R_p @ R_t[:, 2])
    pose[R_HIP] = origin + hip_half * Z
    pose[L_HIP] = origin - hip_half * Z
    return pose

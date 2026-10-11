"""Forward kinematics: joint angles to landmarks. Used to test the solver and to drive a simulated arm."""
from __future__ import annotations

import numpy as np

from .geometry import L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST
from .kinematics import POSE_INDEX, POSE_PINKY, rot_x, rot_y, rot_z


def forward_arm(angles: dict, side: str = "right", body_rotation: np.ndarray | None = None,
                body_offset=(0.0, 0.0, 0.0), upper_arm=0.30, forearm=0.26, hand_len=0.09, knuckle_half=0.04):
    """Return (pose (33,3), hand (21,3)) for the given joint angles in degrees.

    angles keys: shoulder_flexion, shoulder_abduction, shoulder_rotation, elbow_flexion,
    forearm_rotation, wrist_flexion, wrist_deviation. Only landmarks the solver reads are filled.
    body_rotation maps thorax axes (forward, up, right) into the camera's world axes.
    """
    d = {k: np.radians(v) for k, v in angles.items()}
    phi, alp, tau = d["shoulder_flexion"], d["shoulder_abduction"], d["shoulder_rotation"]
    R_h = rot_z(phi) @ rot_x(-alp) @ rot_y(tau)          # sagittal, coronal, transverse
    a = -R_h[:, 1]                                       # shoulder to elbow
    R_f = R_h @ rot_z(d["elbow_flexion"])
    R_hand = R_f @ rot_y(d["forearm_rotation"]) @ rot_x(d["wrist_flexion"]) @ rot_z(d["wrist_deviation"])

    S = np.array([0.0, 0.0, 0.2])
    E = S + upper_arm * a
    W = E + forearm * (R_f @ np.array([0.0, -1.0, 0.0]))
    y_hand, x_hand = R_hand[:, 1], R_hand[:, 0]
    mid_knuckle = W - hand_len * y_hand
    idx, pnk = mid_knuckle + knuckle_half * x_hand, mid_knuckle - knuckle_half * x_hand

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

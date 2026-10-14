"""The Tandem robot in MuJoCo: an upper body that copies your tracked joint angles.

Stage 1 (this file) moves every joint with a position servo, so the robot's shape, joint axes and ranges can be
checked against the tracker before any hardware exists. Stage 2 replaces the arm, hand and waist servos with
antagonistic air muscles (tendons pulling over joint pulleys); the head stays on electric servos, as in the build plan
(a Will Cogley style animatronic head: neck, eyes and eyelids on hobby servos, for low weight and simple wiring).

Robot layout (the hardware plan, in build order):
  1. arm: 3-axis shoulder, elbow, forearm rotation, 2-axis wrist and a 5-finger hand (21 joints per arm)
  2. upper body: T-bar spine and collarbone carrying two arms, and a neck with yaw, pitch and roll
  3. waist: rotation on a slewing bearing (core_rotation); lean and side bend are tracked but not built
  head: eyes that turn together (yaw, pitch) and eyelids that blink together

Joint names are the tracker's CSV column names (R_shoulder_flexion, L_index_pip, core_rotation, neck_yaw, ...) and
every axis is set so a positive angle means the same thing it means in the tracker. tests/test_sim.py checks this:
the simulated elbow, wrist, knuckles and fingertips land exactly where the tracker's own forward kinematics put them.

MuJoCo axes: x forward, y left, z up. The tracker's thorax axes (x forward, y up, z right) map as (x, -z, y).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fingers import FINGER_KEYS, FINGERS4, THUMB_KEYS, _THUMB_BASE, _finger_base
from .kinematics import rot_x

UPPER_ARM, FOREARM, HAND_LEN = 0.30, 0.26, 0.09     # the same lengths as vtrack.synthetic.arm_chain
SHOULDER_HALF, TRUNK, WAIST_HEIGHT, NECK = 0.20, 0.50, 0.95, 0.10
LID_CLOSED_DEG = 75.0

T2M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # thorax or palm axes to MuJoCo axes
_S = np.diag([1.0, -1.0, 1.0])                                      # mirror right to left

# Palm coordinates as in vtrack.fingers (x toward the thumb, y toward the wrist, z out of the back of the hand),
# wrist at the origin. An adult hand of about 19 cm; the middle knuckle sits HAND_LEN from the wrist.
NOMINAL_HAND = {
    "index": {"base": np.array([0.024, -0.087, 0.0]), "lengths": [0.040, 0.024, 0.020]},
    "middle": {"base": np.array([0.0, -HAND_LEN, 0.0]), "lengths": [0.045, 0.028, 0.022]},
    "ring": {"base": np.array([-0.019, -0.085, 0.0]), "lengths": [0.042, 0.027, 0.021]},
    "pinky": {"base": np.array([-0.035, -0.077, 0.0]), "lengths": [0.033, 0.019, 0.018]},
    "thumb": {"base": np.array([0.022, -0.024, -0.012]), "lengths": [0.042, 0.032, 0.026]},
}


@dataclass(frozen=True)
class Drive:
    """Servo settings for a group of joints (stage 1). Torques in N m, gains in N m per radian."""
    kp: float
    max_torque: float
    damping: float
    armature: float
    motor: str


DRIVES = {
    "waist": Drive(800.0, 120.0, 30.0, 0.05, "air muscles (stage 2)"),
    "shoulder": Drive(500.0, 80.0, 12.0, 0.02, "air muscles (stage 2)"),
    "elbow": Drive(300.0, 50.0, 6.0, 0.01, "air muscles (stage 2)"),
    "wrist": Drive(60.0, 12.0, 1.0, 0.003, "air muscles (stage 2)"),
    "finger": Drive(4.0, 1.5, 0.04, 0.0005, "air muscles via tendons (stage 2)"),
    "neck": Drive(30.0, 2.5, 0.8, 0.002, "servo, about 25 kg cm"),
    "eye": Drive(2.0, 0.2, 0.02, 0.0001, "micro servo, about 2 kg cm"),
    "lid": Drive(2.0, 0.2, 0.02, 0.0001, "micro servo, about 2 kg cm"),
}

ARM_RANGES = {"shoulder_flexion": (-60, 180), "shoulder_abduction": (-30, 180), "shoulder_rotation": (-90, 90),
              "elbow_flexion": (0, 150), "forearm_rotation": (-90, 90), "wrist_flexion": (-80, 80),
              "wrist_deviation": (-40, 40)}
FINGER_RANGES = {"mcp_flex": (-30, 95), "mcp_abd": (-30, 30), "pip": (0, 115), "dip": (-10, 90)}
THUMB_RANGES = {"cmc_palmar": (-20, 90), "cmc_radial": (-20, 90), "cmc_rotation": (-60, 150), "mcp": (-10, 80),
                "ip": (-20, 90)}
HEAD_RANGES = {"neck_yaw": (-80, 80), "neck_pitch": (-40, 40), "neck_roll": (-30, 30)}
EYE_RANGES = {"eye_yaw": (-40, 40), "eye_pitch": (-30, 30)}
CORE_RANGE = (-90, 90)
SIDES = {"right": "R_", "left": "L_"}


def quat(R: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) of a rotation matrix."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(1.0 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.ravel(v))


class _Side:
    """Writes a right-side part; for the left side every position, axis and frame is mirrored."""

    def __init__(self, side: str):
        self.left = side == "left"
        self.p = SIDES[side]

    def pos(self, v):
        return _S @ np.asarray(v, float) if self.left else np.asarray(v, float)

    def axis(self, a):
        return -_S @ np.asarray(a, float) if self.left else np.asarray(a, float)   # axes are pseudovectors

    def frame(self, R):
        return _S @ R @ _S if self.left else R


class _Mjcf:
    def __init__(self):
        self.lines, self.depth = [], 0
        self.joints = []                 # (name, group, (lo, hi) degrees)

    def add(self, text):
        self.lines.append("  " * self.depth + text)

    def open(self, text):
        self.add(text)
        self.depth += 1

    def close(self, tag):
        self.depth -= 1
        self.add(f"</{tag}>")

    def joint(self, name, axis, rng, group):
        d = DRIVES[group]
        lo, hi = np.radians(rng)
        self.add(f'<joint name="{name}" axis="{_f(axis)}" range="{lo:.5f} {hi:.5f}" damping="{d.damping}" '
                 f'armature="{d.armature}"/>')
        self.joints.append((name, group, rng))


def _capsule(m, a, b, r, rgba, name=""):
    nm = f' name="{name}"' if name else ""
    m.add(f'<geom{nm} type="capsule" fromto="{_f(a)} {_f(b)}" size="{r}" rgba="{rgba}" class="robot"/>')


BONE, JOINT_RGBA, MUSCLE, SHELL = "0.85 0.83 0.78 1", "0.25 0.27 0.30 1", "0.75 0.30 0.25 1", "0.92 0.92 0.95 0.45"


def _hand(m: _Mjcf, s: _Side):
    p = s.p
    m.add(f'<geom type="box" pos="{_f(s.pos(T2M @ [0, -0.045, 0]))}" size="0.042 0.013 0.045" rgba="{MUSCLE}" '
          f'class="robot"/>')
    m.add(f'<site name="{p}knuckle" pos="{_f(s.pos(T2M @ NOMINAL_HAND["middle"]["base"]))}" size="0.004"/>')
    for name in FINGERS4:
        g = NOMINAL_HAND[name]
        L1, L2, L3 = g["lengths"]
        B = T2M @ _finger_base(g["base"])
        m.open(f'<body name="{p}{name}" pos="{_f(s.pos(T2M @ g["base"]))}" quat="{_f(quat(s.frame(B)))}">')
        m.joint(f"{p}{name}_mcp_flex", s.axis([0, 0, 1]), FINGER_RANGES["mcp_flex"], "finger")
        m.joint(f"{p}{name}_mcp_abd", s.axis([0, -1, 0]), FINGER_RANGES["mcp_abd"], "finger")
        _capsule(m, [0, 0, 0], s.pos([L1, 0, 0]), 0.0085, BONE)
        m.open(f'<body name="{p}{name}_middle" pos="{_f(s.pos([L1, 0, 0]))}">')
        m.joint(f"{p}{name}_pip", s.axis([0, 0, 1]), FINGER_RANGES["pip"], "finger")
        _capsule(m, [0, 0, 0], s.pos([L2, 0, 0]), 0.008, BONE)
        m.open(f'<body name="{p}{name}_end" pos="{_f(s.pos([L2, 0, 0]))}">')
        m.joint(f"{p}{name}_dip", s.axis([0, 0, 1]), FINGER_RANGES["dip"], "finger")
        _capsule(m, [0, 0, 0], s.pos([L3, 0, 0]), 0.0075, BONE)
        m.add(f'<site name="{p}{name}_tip" pos="{_f(s.pos([L3, 0, 0]))}" size="0.004"/>')
        m.close("body")
        m.close("body")
        m.close("body")
    g = NOMINAL_HAND["thumb"]
    L1, L2, L3 = g["lengths"]
    m.open(f'<body name="{p}thumb" pos="{_f(s.pos(T2M @ g["base"]))}" quat="{_f(quat(s.frame(T2M @ _THUMB_BASE)))}">')
    m.joint(f"{p}thumb_cmc_palmar", s.axis([0, 0, 1]), THUMB_RANGES["cmc_palmar"], "finger")
    m.joint(f"{p}thumb_cmc_radial", s.axis([0, -1, 0]), THUMB_RANGES["cmc_radial"], "finger")
    m.add(f'<geom type="sphere" size="0.012" rgba="{JOINT_RGBA}" class="robot"/>')
    # rotation about the metacarpal; reported 0 means the thumb bends across the palm (90 degrees from the frame)
    m.open(f'<body name="{p}thumb_meta" quat="{_f(quat(s.frame(rot_x(-math.pi / 2))))}">')
    m.joint(f"{p}thumb_cmc_rotation", s.axis([1, 0, 0]), THUMB_RANGES["cmc_rotation"], "finger")
    _capsule(m, [0, 0, 0], s.pos([L1, 0, 0]), 0.010, BONE)
    m.open(f'<body name="{p}thumb_proximal" pos="{_f(s.pos([L1, 0, 0]))}">')
    m.joint(f"{p}thumb_mcp", s.axis([0, 0, 1]), THUMB_RANGES["mcp"], "finger")
    _capsule(m, [0, 0, 0], s.pos([L2, 0, 0]), 0.009, BONE)
    m.open(f'<body name="{p}thumb_end" pos="{_f(s.pos([L2, 0, 0]))}">')
    m.joint(f"{p}thumb_ip", s.axis([0, 0, 1]), THUMB_RANGES["ip"], "finger")
    _capsule(m, [0, 0, 0], s.pos([L3, 0, 0]), 0.0085, BONE)
    m.add(f'<site name="{p}thumb_tip" pos="{_f(s.pos([L3, 0, 0]))}" size="0.004"/>')
    for _ in range(4):
        m.close("body")


def _arm(m: _Mjcf, side: str):
    """Shoulder (Z-X-Y like the tracker), elbow, then forearm rotation and the wrist at the hand."""
    s = _Side(side)
    p = s.p
    m.open(f'<body name="{p}upper_arm" pos="{_f(s.pos([0, -SHOULDER_HALF, TRUNK]))}">')
    m.joint(p + "shoulder_flexion", s.axis([0, -1, 0]), ARM_RANGES["shoulder_flexion"], "shoulder")
    m.joint(p + "shoulder_abduction", s.axis([-1, 0, 0]), ARM_RANGES["shoulder_abduction"], "shoulder")
    m.joint(p + "shoulder_rotation", s.axis([0, 0, 1]), ARM_RANGES["shoulder_rotation"], "shoulder")
    m.add(f'<geom type="sphere" size="0.05" rgba="{JOINT_RGBA}" class="robot"/>')
    _capsule(m, [0, 0, -0.04], [0, 0, -UPPER_ARM + 0.03], 0.045, MUSCLE)
    m.open(f'<body name="{p}forearm" pos="0 0 {-UPPER_ARM}">')
    m.joint(p + "elbow_flexion", s.axis([0, -1, 0]), ARM_RANGES["elbow_flexion"], "elbow")
    m.add(f'<site name="{p}elbow" size="0.006"/>')
    m.add(f'<geom type="sphere" size="0.04" rgba="{JOINT_RGBA}" class="robot"/>')
    _capsule(m, [0, 0, -0.03], [0, 0, -FOREARM + 0.02], 0.036, MUSCLE)
    m.open(f'<body name="{p}hand" pos="0 0 {-FOREARM}">')
    m.joint(p + "forearm_rotation", s.axis([0, 0, 1]), ARM_RANGES["forearm_rotation"], "wrist")
    m.joint(p + "wrist_flexion", s.axis([1, 0, 0]), ARM_RANGES["wrist_flexion"], "wrist")
    m.joint(p + "wrist_deviation", s.axis([0, -1, 0]), ARM_RANGES["wrist_deviation"], "wrist")
    m.add(f'<site name="{p}wrist" size="0.006"/>')
    m.add(f'<geom type="sphere" size="0.025" rgba="{JOINT_RGBA}" class="robot"/>')
    _hand(m, s)
    m.close("body")
    m.close("body")
    m.close("body")


def _head(m: _Mjcf):
    m.open(f'<body name="neck" pos="0 0 {TRUNK + 0.04}">')
    m.joint("neck_yaw", [0, 0, 1], HEAD_RANGES["neck_yaw"], "neck")
    m.joint("neck_pitch", [0, -1, 0], HEAD_RANGES["neck_pitch"], "neck")
    m.joint("neck_roll", [1, 0, 0], HEAD_RANGES["neck_roll"], "neck")
    _capsule(m, [0, 0, 0], [0, 0, NECK], 0.025, JOINT_RGBA)
    m.open(f'<body name="head" pos="0 0 {NECK}">')
    m.add(f'<geom type="ellipsoid" pos="0.005 0 0.09" size="0.095 0.075 0.11" rgba="{SHELL}" density="250" '
          f'class="robot"/>')
    m.add('<geom type="box" pos="0.0 0 0.05" size="0.04 0.05 0.035" rgba="0.3 0.3 0.33 1" density="300" '
          'class="robot"/>')
    m.add('<camera name="eyes" pos="0.1 0 0.09" xyaxes="0 -1 0 0 0 1" fovy="70"/>')
    for side, y in (("R_", -0.032), ("L_", 0.032)):
        # both eyes keep the same axes: positive yaw turns both to the robot's left, as in the tracker
        m.open(f'<body name="{side}eye" pos="0.072 {y} 0.09">')
        m.joint(f"{side}eye_yaw", [0, 0, 1], EYE_RANGES["eye_yaw"], "eye")
        m.joint(f"{side}eye_pitch", [0, -1, 0], EYE_RANGES["eye_pitch"], "eye")
        m.add('<geom type="sphere" size="0.014" rgba="0.97 0.97 0.97 1" density="400" class="robot"/>')
        m.add('<geom type="sphere" pos="0.0115 0 0" size="0.0055" rgba="0.05 0.05 0.08 1" contype="0" '
              'conaffinity="0" density="400"/>')
        m.close("body")
        m.open(f'<body name="{side}eyelid" pos="0.072 {y} 0.09">')
        m.joint(f"{side}eyelid", [0, 1, 0], (0, LID_CLOSED_DEG), "lid")
        m.add('<geom type="ellipsoid" pos="0 0 0.0125" size="0.018 0.018 0.006" rgba="0.55 0.45 0.40 1" '
              'contype="0" conaffinity="0" density="400"/>')
        m.close("body")
    m.close("body")
    m.close("body")


def _look_camera(name, pos, target, fovy=50) -> str:
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return f'<camera name="{name}" pos="{_f(pos)}" xyaxes="{_f(x)} {_f(np.cross(x, f))}" fovy="{fovy}"/>'


# Joints that follow another joint (one servo drives both eyes and both lids through linkages)
COUPLED = {"L_eye_yaw": "R_eye_yaw", "L_eye_pitch": "R_eye_pitch", "L_eyelid": "R_eyelid"}


def build_xml(scene: bool = True, arms=("right", "left")) -> str:
    """MJCF text of the Tandem upper body (and a floor, table and tennis ball when scene is True)."""
    m = _Mjcf()
    m.add('<mujoco model="tandem">')
    m.depth = 1
    m.add('<compiler angle="radian" autolimits="true"/>')
    m.add('<option timestep="0.002" integrator="implicitfast"/>')
    m.add('<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>')
    m.open("<default>")
    m.add('<geom density="1000"/>')
    m.add('<default class="robot"><geom contype="1" conaffinity="0" friction="1 0.01 0.001"/></default>')
    m.close("default")
    m.open("<asset>")
    m.add('<texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.65 0.8" rgb2="0.1 0.12 0.16" '
          'width="256" height="256"/>')
    m.add('<texture name="grid" type="2d" builtin="checker" rgb1="0.32 0.34 0.38" rgb2="0.26 0.28 0.31" '
          'width="512" height="512"/>')
    m.add('<material name="floor" texture="grid" texrepeat="8 8" reflectance="0.1"/>')
    m.close("asset")
    m.open("<worldbody>")
    m.add('<light pos="1.5 -1 3" dir="-0.5 0.3 -1" directional="true" diffuse="0.8 0.8 0.8" castshadow="true"/>')
    m.add('<light pos="-1 1 2.5" dir="0.3 -0.3 -1" directional="true" diffuse="0.3 0.3 0.35" castshadow="false"/>')
    if scene:
        m.add('<geom name="floor" type="plane" size="3 3 0.1" material="floor"/>')
        m.add('<body name="table" pos="0.55 0 0">'
              '<geom type="box" pos="0 0 0.73" size="0.25 0.45 0.02" rgba="0.55 0.42 0.3 1"/>'
              '<geom type="box" pos="0 0 0.36" size="0.03 0.3 0.35" rgba="0.4 0.3 0.22 1"/></body>')
        m.add('<body name="tennis_ball" pos="0.45 -0.2 0.8"><freejoint/>'
              '<geom type="sphere" size="0.0335" mass="0.058" rgba="0.85 0.95 0.2 1" friction="0.6 0.01 0.001"/>'
              '</body>')
    look = [0.15, 0.0, WAIST_HEIGHT + 0.35]
    for name, pos in (("front", [1.9, -1.0, 1.75]), ("side", [0.25, -2.3, 1.45]), ("top", [1.2, 0.0, 2.7])):
        m.add(_look_camera(name, pos, look))
    m.open(f'<body name="pelvis" pos="0 0 {WAIST_HEIGHT}">')
    m.add(f'<geom type="cylinder" fromto="0 0 {-WAIST_HEIGHT} 0 0 -0.04" size="0.05" rgba="{JOINT_RGBA}"/>')
    m.add('<geom type="cylinder" fromto="0 0 -0.04 0 0 0" size="0.11" rgba="0.35 0.37 0.4 1"/>')   # slewing bearing
    m.open('<body name="chest">')
    m.joint("core_rotation", [0, 0, 1], CORE_RANGE, "waist")
    _capsule(m, [0, 0, 0.0], [0, 0, TRUNK + 0.04], 0.03, BONE, "spine")
    _capsule(m, [0, -SHOULDER_HALF, TRUNK], [0, SHOULDER_HALF, TRUNK], 0.025, BONE, "collarbone")
    m.add(f'<geom type="ellipsoid" pos="0.01 0 {TRUNK - 0.17}" size="0.10 0.15 0.17" rgba="{SHELL}" '
          'contype="0" conaffinity="0" density="150"/>')
    for side in arms:
        _arm(m, side)
    _head(m)
    m.close("body")
    m.close("body")
    m.close("worldbody")
    names = {n for n, _, _ in m.joints}
    m.open("<equality>")
    for follower, leader in COUPLED.items():
        if follower in names:
            m.add(f'<joint joint1="{follower}" joint2="{leader}"/>')
    m.close("equality")
    m.open("<actuator>")
    for name, group, rng in m.joints:
        if name in COUPLED:
            continue
        d = DRIVES[group]
        lo, hi = np.radians(rng)
        m.add(f'<position name="{name}" joint="{name}" kp="{d.kp}" ctrlrange="{lo:.5f} {hi:.5f}" '
              f'forcerange="{-d.max_torque} {d.max_torque}"/>')
    m.close("actuator")
    m.depth = 0
    m.add("</mujoco>")
    return "\n".join(m.lines) + "\n"


def joint_list(arms=("right", "left")) -> list:
    """(name, group, (lo, hi) degrees) for every joint of the robot."""
    m = _Mjcf()
    for side in arms:
        _arm(m, side)
    _head(m)
    return [("core_rotation", "waist", CORE_RANGE)] + m.joints


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def source_column(joint: str, columns, hand_side: str = "right") -> str | None:
    """The tracker value that drives a joint, or None. Recordings of one arm (or the hand view) have no L_/R_
    prefix: those drive the `hand_side` arm."""
    if joint in COUPLED:
        return None
    if joint.endswith("eye_yaw") or joint.endswith("eye_pitch"):
        name = joint[2:]
        return name if name in columns else None
    if joint.endswith("eyelid"):
        return "eyes_open" if "eyes_open" in columns else None
    if joint in columns:
        return joint
    pre = SIDES[hand_side]
    if joint.startswith(pre) and joint[2:] in columns and not any(c.startswith(("L_", "R_")) and c[2:] == joint[2:]
                                                                   for c in columns):
        return joint[2:]
    return None


class TandemSim:
    """Drives the robot from tracker values (a CSV row or the live app's values) and steps the physics."""

    def __init__(self, scene: bool = True, hand_side: str = "right"):
        import mujoco
        self.mj = mujoco
        self.xml = build_xml(scene)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.hand_side = hand_side
        self.actuators = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.groups = {n: g for n, g, _ in joint_list()}
        self._qadr = {self.model.joint(i).name: int(self.model.jnt_qposadr[i]) for i in range(self.model.njnt)}
        self._renderer = None
        self._columns, self._map = None, {}
        mujoco.mj_forward(self.model, self.data)

    # ----------------------------------------------------------------- commands
    def _mapping(self, columns):
        key = tuple(columns)
        if key != self._columns:
            self._columns = key
            cols = set(columns)
            self._map = {a: source_column(a, cols, self.hand_side) for a in self.actuators}
        return self._map

    def command(self, values: dict) -> dict:
        """Set servo targets from tracker values (degrees; missing or NaN values hold the last target).
        Returns the targets that changed, in degrees."""
        changed = {}
        face_seen = math.isfinite(_num(values.get("neck_yaw")))
        for i, a in enumerate(self.actuators):
            col = self._mapping(values.keys())[a]
            if col is None:
                continue
            v = _num(values[col])
            if not math.isfinite(v):
                continue
            if col == "eyes_open":
                if not face_seen:
                    continue
                v = 0.0 if v >= 0.5 else LID_CLOSED_DEG
            lo, hi = self.model.actuator_ctrlrange[i]
            self.data.ctrl[i] = min(hi, max(lo, math.radians(v)))
            changed[a] = math.degrees(self.data.ctrl[i])
        return changed

    def targets(self) -> dict:
        return {a: math.degrees(self.data.ctrl[i]) for i, a in enumerate(self.actuators)}

    # ----------------------------------------------------------------- physics
    def step(self, seconds: float):
        """Advance the physics by `seconds` (at most a quarter second at a time, so a stall does not freeze it)."""
        n = int(round(min(max(seconds, 0.0), 0.25) / self.model.opt.timestep))
        if n:
            self.mj.mj_step(self.model, self.data, nstep=n)

    def snap(self):
        """Put every joint exactly on its target (no physics): for checking poses."""
        for i, a in enumerate(self.actuators):
            self.data.qpos[self._qadr[a]] = self.data.ctrl[i]
        for follower, leader in COUPLED.items():
            if follower in self._qadr:
                self.data.qpos[self._qadr[follower]] = self.data.qpos[self._qadr[leader]]
        self.data.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def angle(self, joint: str) -> float:
        return math.degrees(self.data.qpos[self._qadr[joint]])

    def errors(self) -> dict:
        """Target minus actual angle per servo, degrees."""
        return {a: math.degrees(self.data.ctrl[i]) - self.angle(a) for i, a in enumerate(self.actuators)}

    def site(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def chest_frame(self):
        b = self.data.body("chest")
        return b.xpos.copy(), b.xmat.reshape(3, 3).copy()

    # ----------------------------------------------------------------- pictures
    def render(self, camera: str = "front", width: int = 640, height: int = 480) -> np.ndarray:
        """RGB image (height, width, 3) from a named camera: front, side, top or eyes (the robot's view)."""
        r = self._renderer
        if r is None or (r.width, r.height) != (width, height):
            if r is not None:
                r.close()
            r = self._renderer = self.mj.Renderer(self.model, height=height, width=width)
        r.update_scene(self.data, camera=camera)
        return r.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def read_csv(path) -> list:
    """Rows of a tracker recording as dicts (the 't' column in seconds)."""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

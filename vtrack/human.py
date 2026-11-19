"""A full human body in MuJoCo that mirrors you: arms, hands, legs, core, head, eyes and a face that makes your
expressions.

What it adds to the robot model in sim.py (which stays the hardware plan for Homunculus):
  * legs: hip flexion, abduction and rotation, knee, ankle dorsiflexion, inversion and abduction per leg, and the
    pelvis tilting side to side (pelvic hike), all with the tracker's names and signs
  * the whole core: flexion, side bend and rotation (the robot only builds rotation)
  * a face: jaw, lips, mouth corners, cheeks, nose, brows, upper and lower eyelids and eyes, driven by MediaPipe's
    52 face blendshapes and the emotion model's scores (see face_rig.py)

The arms and hands are built by the same code as the robot (sim._arm), so their joints match the tracker exactly.

Two modes:
  kinematic (default)  every joint is set straight to the tracked angle and the body is put on the floor by its
                       lowest foot. No servo lag: the avatar is as fast as the tracking itself, and each joint sits
                       exactly on its measured angle.
  physics              position servos follow the angles under gravity, like the robot sim. The pelvis is held in
                       place (a person balancing is a control problem of its own).

Mirror mode (default) swaps left and right so the avatar, facing you, moves like your reflection.

MuJoCo axes: x forward (the avatar faces +x), y left, z up.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import face_rig
from .face_rig import FACE_JOINTS, LID_CLOSED
from .filters import OneEuroFilter
from .sim import (DRIVES, EYE_RANGES, HEAD_RANGES, SIDES, Drive, _arm, _capsule, _f, _look_camera, _Mjcf, _num,
                  _Side, source_column)

# Body proportions (meters): a person of about 1.80 m, arms as in the tracker's own model
HIP_HEIGHT, HIP_HALF = 0.93, 0.09
THIGH, SHANK, ANKLE_H = 0.43, 0.42, 0.08
CHEST_H = 0.52                       # hip joint line to the shoulder joints (the core bends at the hip line, as
SHOULDER_HALF = 0.19                 # the tracker measures it: chest relative to pelvis, pelvic tilt included)
NECK_BASE, NECK = 0.035, 0.065
FOOT_BACK, FOOT_FRONT, FOOT_HALF_W = 0.06, 0.19, 0.045

SKIN, SKIN_DARK = "0.86 0.70 0.58 1", "0.78 0.60 0.50 1"
SHIRT, PANTS, SHOE, HAIR = "0.20 0.36 0.55 1", "0.18 0.19 0.24 1", "0.12 0.12 0.13 1", "0.16 0.11 0.08 1"
LIP, BROW, WHITE, IRIS, PUPIL = "0.66 0.34 0.34 1", "0.18 0.12 0.09 1", "0.97 0.97 0.96 1", "0.30 0.19 0.10 1", "0.03 0.03 0.04 1"
HUMAN_LOOK = {"finger": SKIN, "palm": SKIN, "upper": SHIRT, "fore": SKIN, "shoulder": SHIRT, "joint": SKIN}

LEG_RANGES = {"hip_flexion": (-30, 130), "hip_abduction": (-35, 60), "hip_rotation": (-50, 50),
              "knee_flexion": (0, 150), "ankle_dorsiflexion": (-55, 35), "ankle_inversion": (-30, 40),
              "ankle_abduction": (-30, 30)}
# Joint axes for the right leg (the left is mirrored), checked against the tracker's leg solver in tests/test_human.py
LEG_AXES = {"hip_flexion": [0, -1, 0], "hip_abduction": [-1, 0, 0], "hip_rotation": [0, 0, 1],
            "knee_flexion": [0, 1, 0], "ankle_dorsiflexion": [0, -1, 0], "ankle_inversion": [1, 0, 0],
            "ankle_abduction": [0, 0, -1]}
CORE_RANGES = {"core_flexion": (-40, 100), "core_lateral_bend": (-45, 45), "core_rotation": (-70, 70)}
CORE_AXES = {"core_flexion": [0, 1, 0], "core_lateral_bend": [1, 0, 0], "core_rotation": [0, 0, 1]}
PELVIS_ROLL = (-25, 25)              # + raises the right hip (the tracker's R_pelvic_hike)

HUMAN_DRIVES = {**DRIVES,
                "pelvis": Drive(1500.0, 400.0, 60.0, 0.1, "body"),
                "spine": Drive(4000.0, 600.0, 120.0, 0.1, "body"),
                "hip": Drive(3000.0, 400.0, 60.0, 0.05, "body"),
                "knee": Drive(2000.0, 300.0, 35.0, 0.03, "body"),
                "ankle": Drive(600.0, 120.0, 8.0, 0.01, "body"),
                "face": Drive(3.0, 0.5, 0.03, 0.0001, "face"),          # hinge: N m per radian
                "face_slide": Drive(3000.0, 30.0, 25.0, 0.001, "face")}  # slide: N per meter
MIDLINE_FLIP = ["core_lateral_bend", "core_rotation", "neck_yaw", "neck_roll", "eye_yaw", "head_yaw", "head_roll"]
COUPLED_EYES = {"L_eye_yaw": "R_eye_yaw", "L_eye_pitch": "R_eye_pitch"}
VIEWS = ["split", "front", "face", "side", "three_quarter"]


class _HumanMjcf(_Mjcf):
    def __init__(self):
        super().__init__()
        self.kind = {}                   # joint name -> "hinge" or "slide"

    def joint(self, name, axis, rng, group):
        d = HUMAN_DRIVES[group]
        lo, hi = np.radians(rng)
        self.add(f'<joint name="{name}" axis="{_f(axis)}" range="{lo:.5f} {hi:.5f}" damping="{d.damping}" '
                 f'armature="{d.armature}"/>')
        self.joints.append((name, group, rng))
        self.kind[name] = "hinge"

    def slide(self, name, axis, rng_m, group="face_slide"):
        d = HUMAN_DRIVES[group]
        self.add(f'<joint name="{name}" type="slide" axis="{_f(axis)}" range="{rng_m[0]:.5f} {rng_m[1]:.5f}" '
                 f'damping="{d.damping}" armature="{d.armature}"/>')
        self.joints.append((name, group, rng_m))
        self.kind[name] = "slide"


# Torso and pelvis outlines: (height above the hip line, half width, half depth, forward offset), meters.
# Each is the convex hull of rounded-rectangle sections, which MuJoCo renders smoothly.
TORSO_SECTIONS = [(0.06, 0.150, 0.092, -0.005), (0.16, 0.142, 0.090, -0.005), (0.26, 0.150, 0.095, 0.0),
                  (0.36, 0.168, 0.105, 0.005), (0.45, 0.178, 0.102, 0.005), (0.51, 0.170, 0.085, -0.005),
                  (0.555, 0.120, 0.065, -0.012), (0.575, 0.06, 0.05, -0.012)]
PELVIS_SECTIONS = [(-0.07, 0.120, 0.080, -0.01), (-0.02, 0.165, 0.098, -0.01), (0.05, 0.168, 0.100, -0.008),
                   (0.11, 0.152, 0.092, -0.005)]


def _hull(sections, n: int = 28, power: float = 2.6) -> np.ndarray:
    """Vertices of superellipse sections stacked at their heights (for a convex mesh)."""
    pts = []
    for z, half_w, half_d, dx in sections:
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            c, s = np.cos(t), np.sin(t)
            pts.append([dx + half_d * np.sign(c) * abs(c) ** (2 / power), half_w * np.sign(s) * abs(s) ** (2 / power), z])
    return np.array(pts)


def _geom(m, kind, pos, size, rgba, extra=""):
    m.add(f'<geom type="{kind}" pos="{_f(pos)}" size="{_f(size)}" rgba="{rgba}" class="robot" {extra}/>')


def _face_part(m, kind, pos, size, rgba, extra=""):
    """A light, collision-free face geom."""
    m.add(f'<geom type="{kind}" pos="{_f(pos)}" size="{_f(size)}" rgba="{rgba}" contype="0" conaffinity="0" '
          f'density="300" {extra}/>')


def _leg(m: _HumanMjcf, side: str):
    s = _Side(side)
    p = s.p
    m.open(f'<body name="{p}thigh" pos="{_f(s.pos([0, -HIP_HALF, 0]))}">')
    for k in ("hip_flexion", "hip_abduction", "hip_rotation"):
        m.joint(p + k, s.axis(LEG_AXES[k]), LEG_RANGES[k], "hip")
    _geom(m, "sphere", [0, 0, 0], [0.075], PANTS)
    _capsule(m, [0, 0, -0.03], [0, 0, -THIGH + 0.04], 0.068, PANTS)
    m.open(f'<body name="{p}shank" pos="0 0 {-THIGH}">')
    m.joint(p + "knee_flexion", s.axis(LEG_AXES["knee_flexion"]), LEG_RANGES["knee_flexion"], "knee")
    _geom(m, "sphere", [0, 0, 0], [0.058], PANTS)
    _capsule(m, [0, 0, -0.03], [0, 0, -SHANK + 0.05], 0.052, PANTS)
    m.open(f'<body name="{p}foot" pos="0 0 {-SHANK}">')
    for k in ("ankle_dorsiflexion", "ankle_inversion", "ankle_abduction"):
        m.joint(p + k, s.axis(LEG_AXES[k]), LEG_RANGES[k], "ankle")
    _geom(m, "sphere", [0, 0, 0], [0.04], SKIN)
    mid, half = (FOOT_FRONT - FOOT_BACK) / 2, (FOOT_FRONT + FOOT_BACK) / 2
    _geom(m, "box", s.pos([mid, 0, -ANKLE_H + 0.03]), [half, FOOT_HALF_W, 0.03], SHOE)
    _geom(m, "ellipsoid", s.pos([FOOT_FRONT - 0.035, 0, -ANKLE_H + 0.035]), [0.04, FOOT_HALF_W, 0.03], SHOE)
    # MediaPipe's heel and toe points (the tracker's foot frame: heel to toe is level when the foot is flat)
    m.add(f'<site name="{p}heel" pos="{_f(s.pos([-FOOT_BACK, 0, -ANKLE_H + 0.01]))}" size="0.006"/>')
    m.add(f'<site name="{p}toe" pos="{_f(s.pos([FOOT_FRONT, 0, -ANKLE_H + 0.01]))}" size="0.006"/>')
    for i, (x, y) in enumerate(((-FOOT_BACK, -1), (-FOOT_BACK, 1), (FOOT_FRONT, -1), (FOOT_FRONT, 1))):
        m.add(f'<site name="{p}sole{i}" pos="{_f(s.pos([x, y * FOOT_HALF_W, -ANKLE_H]))}" size="0.004"/>')
    for _ in range(3):
        m.close("body")


def _face_x(y, z, out=0.002):
    """x of the face surface (the cranium and midface ellipsoids) at (y, z) in the head frame, plus `out`."""
    best = -1.0
    for (cx, cy, cz), (a, b, c) in (((-0.01, 0, 0.125), (0.092, 0.077, 0.105)), ((0.025, 0, 0.078), (0.060, 0.066, 0.036))):
        r = 1 - ((y - cy) / b) ** 2 - ((z - cz) / c) ** 2
        if r > 0:
            best = max(best, cx + a * math.sqrt(r))
    return best + out


def _side_slide(m, s: _Side, name, axis, rng):
    """A slide joint on one side; `axis` is for the right side (vectors mirror like positions)."""
    m.slide(name, s.pos(axis), rng)


def _head(m: _HumanMjcf):
    m.open(f'<body name="neck" pos="0 0 {CHEST_H + NECK_BASE}">')
    m.joint("neck_yaw", [0, 0, 1], HEAD_RANGES["neck_yaw"], "neck")
    m.joint("neck_pitch", [0, -1, 0], HEAD_RANGES["neck_pitch"], "neck")
    m.joint("neck_roll", [1, 0, 0], HEAD_RANGES["neck_roll"], "neck")
    _capsule(m, [0, 0, -0.04], [0.01, 0, NECK], 0.052, SKIN)
    m.open(f'<body name="head" pos="0 0 {NECK}">')
    # skull, hair, midface (cheekbones and upper lip), nose, ears, brow ridge
    _geom(m, "ellipsoid", [-0.01, 0, 0.125], [0.092, 0.077, 0.105], SKIN, 'density="900"')
    _geom(m, "ellipsoid", [-0.022, 0, 0.152], [0.093, 0.081, 0.094], HAIR, 'density="100"')
    _face_part(m, "ellipsoid", [0.025, 0, 0.078], [0.060, 0.066, 0.036], SKIN)
    _face_part(m, "ellipsoid", [0.086, 0, 0.091], [0.011, 0.009, 0.023], SKIN)
    _face_part(m, "sphere", [0.094, 0, 0.074], [0.0095], SKIN)
    for y in (-0.077, 0.077):
        _face_part(m, "ellipsoid", [-0.012, y, 0.11], [0.013, 0.007, 0.028], SKIN_DARK)
    for sgn in (-1, 1):
        m.add(f'<geom type="capsule" fromto="0.074 0 0.130 0.060 {sgn * 0.056} 0.128" size="0.011" '
              f'rgba="{SKIN}" contype="0" conaffinity="0" density="300"/>')
    # inside of the mouth and the teeth (seen when the lips part)
    _face_part(m, "ellipsoid", [0.058, 0, 0.044], [0.016, 0.022, 0.011], "0.32 0.08 0.09 1")
    _face_part(m, "box", [0.071, 0, 0.049], [0.004, 0.017, 0.0035], WHITE)
    m.add('<camera name="eyes" pos="0.1 0 0.11" xyaxes="0 -1 0 0 0 1" fovy="70"/>')
    for side in ("right", "left"):
        s = _Side(side)
        p = s.p
        eye = s.pos([0.068, -0.032, 0.110])
        m.open(f'<body name="{p}eye" pos="{_f(eye)}">')
        m.joint(f"{p}eye_yaw", [0, 0, 1], EYE_RANGES["eye_yaw"], "eye")
        m.joint(f"{p}eye_pitch", [0, -1, 0], EYE_RANGES["eye_pitch"], "eye")
        _face_part(m, "sphere", [0, 0, 0], [0.0125], WHITE)
        _face_part(m, "sphere", [0.0085, 0, 0], [0.0055], IRIS)
        _face_part(m, "sphere", [0.0118, 0, 0], [0.0025], PUPIL)
        m.close("body")
        m.open(f'<body name="{p}eyelid_body" pos="{_f(eye)}">')
        m.joint(f"{p}eyelid", [0, 1, 0], FACE_JOINTS[p + "eyelid"][1:3], "face")
        _face_part(m, "ellipsoid", [0, 0, 0.0128], [0.0148, 0.0165, 0.0035], SKIN_DARK)
        m.add(f'<geom type="capsule" fromto="0.0145 -0.0135 0.0127 0.0145 0.0135 0.0127" size="0.0011" '
              f'rgba="{BROW}" contype="0" conaffinity="0" density="300"/>')
        m.close("body")
        m.open(f'<body name="{p}lower_lid_body" pos="{_f(eye)}">')
        m.joint(f"{p}lower_lid", [0, -1, 0], FACE_JOINTS[p + "lower_lid"][1:3], "face")
        _face_part(m, "ellipsoid", [0, 0, -0.0127], [0.0140, 0.0155, 0.0024], SKIN)
        m.close("body")
        # brow: three points that slide up and down, drawn as a tendon through them
        for part, y, z in (("inner", -0.014, 0.131), ("mid", -0.034, 0.138), ("outer", -0.054, 0.134)):
            x = 0.074 - 0.014 * (abs(y) / 0.056) + 0.0125
            m.open(f'<body name="{p}brow_{part}_body" pos="{_f(s.pos([x, y, z]))}">')
            _side_slide(m, s, f"{p}brow_{part}", [0, 0, 1], FACE_JOINTS[f"{p}brow_{part}"][1:3])
            m.add(f'<site name="{p}brow_{part}" size="0.002"/>')
            m.close("body")
        # cheek and nostril wing
        m.open(f'<body name="{p}cheek" pos="{_f(s.pos([0.046, -0.043, 0.078]))}">')
        _side_slide(m, s, f"{p}cheek_raise", [0, 0, 1], FACE_JOINTS[p + "cheek_raise"][1:3])
        _side_slide(m, s, f"{p}cheek_puff", [0.3, -1, 0], FACE_JOINTS[p + "cheek_puff"][1:3])
        _face_part(m, "ellipsoid", [0, 0, 0], [0.020, 0.017, 0.015], SKIN)
        m.close("body")
        m.open(f'<body name="{p}nostril" pos="{_f(s.pos([0.084, -0.0105, 0.072]))}">')
        _side_slide(m, s, f"{p}nose_sneer", [0, 0, 1], FACE_JOINTS[p + "nose_sneer"][1:3])
        _face_part(m, "ellipsoid", [0, 0, 0], [0.0075, 0.006, 0.005], SKIN)
        m.close("body")
        # mouth corner
        m.open(f'<body name="{p}mouth_corner" pos="{_f(s.pos([_face_x(0.025, 0.047, 0.0), -0.025, 0.047]))}">')
        _side_slide(m, s, f"{p}mouth_corner_up", [0, 0, 1], FACE_JOINTS[p + "mouth_corner_up"][1:3])
        _side_slide(m, s, f"{p}mouth_corner_out", [-0.2, -1, 0], FACE_JOINTS[p + "mouth_corner_out"][1:3])
        m.add(f'<site name="{p}mouth_corner" size="0.002"/>')
        m.close("body")
    # upper lip: raises and pushes forward; the filler between the nose and the lip moves with it
    m.open('<body name="upper_lip" pos="0.088 0 0.050">')
    m.slide("upper_lip_up", [0, 0, 1], FACE_JOINTS["upper_lip_up"][1:3])
    m.slide("upper_lip_forward", [1, 0, 0], FACE_JOINTS["upper_lip_forward"][1:3])
    _face_part(m, "ellipsoid", [-0.011, 0, 0.007], [0.012, 0.025, 0.009], SKIN)
    for name, y in (("upper_lip_c", 0.0), ("upper_lip_r", -0.013), ("upper_lip_l", 0.013)):
        m.add(f'<site name="{name}" pos="{_f([-0.0015 * (abs(y) / 0.013) ** 2, y, 0.0015 * (1 - abs(y) / 0.013)])}" '
              f'size="0.002"/>')
    m.close("body")
    # jaw: chin, lower teeth and the lower lip
    m.open('<body name="jaw" pos="-0.005 0 0.074">')
    m.joint("jaw_open", [0, 1, 0], FACE_JOINTS["jaw_open"][1:3], "face")
    m.slide("jaw_side", [0, 1, 0], FACE_JOINTS["jaw_side"][1:3])
    _face_part(m, "ellipsoid", [0.027, 0, -0.040], [0.058, 0.051, 0.040], SKIN)
    _face_part(m, "ellipsoid", [0.064, 0, -0.056], [0.017, 0.023, 0.015], SKIN)                     # chin
    _face_part(m, "box", [0.075, 0, -0.030], [0.004, 0.016, 0.0035], WHITE)
    m.open('<body name="lower_lip" pos="0.088 0 -0.0275">')
    m.slide("lower_lip_down", [0, 0, -1], FACE_JOINTS["lower_lip_down"][1:3])
    m.slide("lower_lip_forward", [1, 0, 0], FACE_JOINTS["lower_lip_forward"][1:3])
    for name, y in (("lower_lip_c", 0.0), ("lower_lip_r", -0.013), ("lower_lip_l", 0.013)):
        m.add(f'<site name="{name}" pos="{_f([-0.002 * (abs(y) / 0.013) ** 2, y, -0.002 * (1 - abs(y) / 0.013)])}" '
              f'size="0.002"/>')
    m.close("body")
    m.close("body")
    m.close("body")       # head
    m.close("body")       # neck


TENDONS = {
    "upper_lip": (["R_mouth_corner", "upper_lip_r", "upper_lip_c", "upper_lip_l", "L_mouth_corner"], LIP, 0.0034),
    "lower_lip": (["R_mouth_corner", "lower_lip_r", "lower_lip_c", "lower_lip_l", "L_mouth_corner"], LIP, 0.0042),
    "R_brow": (["R_brow_inner", "R_brow_mid", "R_brow_outer"], BROW, 0.0032),
    "L_brow": (["L_brow_inner", "L_brow_mid", "L_brow_outer"], BROW, 0.0032),
}


def build(floating: bool = True, scene: bool = True) -> tuple[str, _HumanMjcf]:
    """(MJCF text, builder) of the human. `floating`: the body can move up and down (kinematic mode puts it on
    the floor); otherwise the pelvis is fixed in place."""
    m = _HumanMjcf()
    m.add('<mujoco model="tandem_human">')
    m.depth = 1
    # face points that carry only a site (brows, mouth corners) still need a little mass to move
    m.add('<compiler angle="radian" autolimits="true" boundmass="0.002" boundinertia="1e-7"/>')
    m.add('<option timestep="0.002" integrator="implicitfast"/>')
    m.add('<statistic center="0 0 0.95" extent="1.1"/>')
    m.add('<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="8192"/>'
          '<map shadowclip="1.6" shadowscale="0.6" znear="0.005"/>'
          '<headlight ambient="0.38 0.38 0.38" diffuse="0.42 0.42 0.42"/></visual>')
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
    m.add(f'<mesh name="torso" vertex="{_f(_hull(TORSO_SECTIONS))}"/>')
    m.add(f'<mesh name="hips" vertex="{_f(_hull(PELVIS_SECTIONS))}"/>')
    m.close("asset")
    m.open("<worldbody>")
    # no cast shadows: at face scale MuJoCo's shadow map aliases into jagged patches
    m.add('<light pos="2 -1.5 3.5" dir="-0.5 0.3 -1" directional="true" diffuse="0.7 0.7 0.7" castshadow="false"/>')
    m.add('<light pos="2 1.5 2" dir="-0.6 -0.4 -0.3" directional="true" diffuse="0.25 0.25 0.28" castshadow="false"/>')
    if scene:
        m.add('<geom name="floor" type="plane" size="4 4 0.1" material="floor"/>')
    look = [0.0, 0.0, 1.0]
    for name, pos, fovy in (("front", [3.0, 0.0, 1.15], 45), ("side", [0.0, -3.2, 1.1], 45),
                            ("three_quarter", [2.4, -1.6, 1.5], 45)):
        m.add(_look_camera(name, pos, look, fovy))
    m.open(f'<body name="pelvis" pos="0 0 {HIP_HEIGHT}">')
    if floating:
        m.add('<joint name="root_z" type="slide" axis="0 0 1" range="-1 1" damping="0"/>')
    m.joint("pelvis_roll", [-1, 0, 0], PELVIS_ROLL, "pelvis")
    m.add(f'<geom type="mesh" mesh="hips" rgba="{PANTS}" class="robot"/>')
    for side in ("right", "left"):
        _leg(m, side)
    m.open('<body name="chest">')
    for k in ("core_flexion", "core_lateral_bend", "core_rotation"):
        m.joint(k, CORE_AXES[k], CORE_RANGES[k], "spine")
    m.add(f'<geom type="mesh" mesh="torso" rgba="{SHIRT}" class="robot"/>')
    _capsule(m, [0, -SHOULDER_HALF + 0.03, CHEST_H - 0.012], [0, SHOULDER_HALF - 0.03, CHEST_H - 0.012], 0.05, SHIRT)
    # a camera on the chest that keeps the face in view as the body bends: the close-up of the expressions
    m.add(_look_camera("face", [0.62, 0.0, CHEST_H + 0.22], [0.0, 0.0, CHEST_H + NECK_BASE + NECK + 0.10], 30))
    for side in ("right", "left"):
        _arm(m, side, origin=[0, -SHOULDER_HALF, CHEST_H], look=HUMAN_LOOK)
    _head(m)
    m.close("body")       # chest
    m.close("body")       # pelvis
    m.close("worldbody")
    m.open("<tendon>")
    for name, (sites, rgba, width) in TENDONS.items():
        m.open(f'<spatial name="{name}" width="{width}" rgba="{rgba}" limited="false">')
        for st in sites:
            m.add(f'<site site="{st}"/>')
        m.close("spatial")
    m.close("tendon")
    m.open("<equality>")
    for follower, leader in COUPLED_EYES.items():
        m.add(f'<joint joint1="{follower}" joint2="{leader}"/>')
    m.close("equality")
    m.open("<actuator>")
    for name, group, rng in m.joints:
        if name in COUPLED_EYES:
            continue
        d = HUMAN_DRIVES[group]
        lo, hi = (np.radians(rng) if m.kind[name] == "hinge" else rng)
        m.add(f'<position name="{name}" joint="{name}" kp="{d.kp}" ctrlrange="{lo:.5f} {hi:.5f}" '
              f'forcerange="{-d.max_torque} {d.max_torque}"/>')
    m.close("actuator")
    m.depth = 0
    m.add("</mujoco>")
    return "\n".join(m.lines) + "\n", m


def build_xml(floating: bool = True, scene: bool = True) -> str:
    return build(floating, scene)[0]


# ----------------------------------------------------------------------------------------------------- values
def mirror_values(values: dict) -> dict:
    """Your values as your reflection: left and right swapped, and the angles about the midline turned around."""
    out = {}
    for k, v in values.items():
        if k.startswith("R_"):
            k = "L_" + k[2:]
        elif k.startswith("L_"):
            k = "R_" + k[2:]
        elif k.startswith("bs_"):
            k = k.replace("Left", "\0").replace("Right", "Left").replace("\0", "Right")
        out[k] = v
    for k in MIDLINE_FLIP:
        x = _num(out.get(k))
        if math.isfinite(x):
            out[k] = -x
    return out


def body_targets(values: dict, joints, hand_side: str = "right") -> dict:
    """Tracked body angles (degrees) by joint name, for the joints the values cover (NaN entries are skipped)."""
    cols = set(values)
    out = {}
    for j in joints:
        if j in COUPLED_EYES or j in FACE_JOINTS or j == "root_z":
            continue
        if j == "pelvis_roll":
            r, lft = _num(values.get("R_pelvic_hike")), _num(values.get("L_pelvic_hike"))
            v = r if math.isfinite(r) else -lft
        else:
            col = source_column(j, cols, hand_side) if not j.endswith("eyelid") else None
            v = _num(values.get(col)) if col else math.nan
        if math.isfinite(v):
            out[j] = v
    return out


class HumanSim:
    """The full human, driven by tracker values (a CSV row or the live app's values). Same interface as
    sim.TandemSim (command, step, render, errors, close), so the live app can show either."""

    VIEWS = VIEWS

    def __init__(self, mode: str = "kinematic", scene: bool = True, hand_side: str = "right", mirror: bool = True,
                 face_source: str = "auto", emotion_gain: float = 1.25, smooth_face: bool = True):
        import mujoco
        if mode not in ("kinematic", "physics"):
            raise ValueError("mode must be 'kinematic' or 'physics'")
        self.mj, self.mode, self.mirror = mujoco, mode, mirror
        self.hand_side, self.face_source, self.emotion_gain = hand_side, face_source, emotion_gain
        self.xml, b = build(floating=mode == "kinematic", scene=scene)
        self.kind = b.kind
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.joints = [n for n, _, _ in b.joints]
        self._qadr = {self.model.joint(i).name: int(self.model.jnt_qposadr[i]) for i in range(self.model.njnt)}
        self._act = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self._soles = [self.model.site(f"{p}sole{i}").id for p in SIDES.values() for i in range(4)]
        self.targets_si = {j: self._si(j, v) for j, v in face_rig.rest_pose().items()}   # radians or meters
        self._bs = None                   # last blendshapes the face used
        self._face_filter = OneEuroFilter(min_cutoff=2.5, beta=0.8) if smooth_face else None
        self._face_keys = sorted(FACE_JOINTS)
        self._renderer = None
        self.last_update_ms = 0.0
        self._apply()

    # ----------------------------------------------------------------- commands
    def _si(self, joint, v):
        return math.radians(v) if self.kind.get(joint) == "hinge" else float(v)

    def face_targets(self, values: dict) -> dict:
        """Face joint targets (degrees or meters) from blendshapes and emotions; the last face holds when both
        are missing."""
        bs = face_rig.combine(face_rig.blendshapes_from_values(values), face_rig.emotions_from_values(values),
                              self.face_source, self.emotion_gain)
        if bs is None:
            return {}
        self._bs = bs
        pose = face_rig.face_pose(bs)
        if self._face_filter is not None:
            t = _num(values.get("t"))
            t = t if math.isfinite(t) else time.monotonic()
            vec = self._face_filter(np.array([pose[k] for k in self._face_keys]), t)
            pose = dict(zip(self._face_keys, (float(x) for x in vec)))
        return pose

    def command(self, values: dict) -> dict:
        """Take new tracker values (degrees; missing or NaN values hold the last pose). Kinematic mode sets the
        pose now; physics mode sets the servo targets. Returns the targets that changed (degrees or meters)."""
        t0 = time.perf_counter()
        v = mirror_values(values) if self.mirror else values
        # one-arm recordings have no L_/R_ prefix: in mirror mode they drive the other arm
        side = {"right": "left", "left": "right"}[self.hand_side] if self.mirror else self.hand_side
        changed = body_targets(v, self.joints, side)
        if not any(math.isfinite(_num(v.get(e))) for e in ("eye_yaw", "eye_pitch")):
            bs = face_rig.blendshapes_from_values(v)
            if bs is not None:
                changed["R_eye_yaw"], changed["R_eye_pitch"] = face_rig.gaze_from_blendshapes(bs)
        changed.update(self.face_targets(v))
        if "eyes_open" in v and face_rig.blendshapes_from_values(v) is None and self.face_source != "emotion":
            o = _num(v["eyes_open"])                 # no blendshapes: blink from the tracker's eyes_open
            if math.isfinite(o) and o < 0.5:
                changed["R_eyelid"] = changed["L_eyelid"] = LID_CLOSED
        for j, x in changed.items():
            if j in self._qadr:
                lo, hi = self.model.jnt_range[self.model.joint(j).id]
                self.targets_si[j] = min(hi, max(lo, self._si(j, x)))
        self._apply()
        self.last_update_ms = 1000.0 * (time.perf_counter() - t0)
        return changed

    def _apply(self):
        for j, x in self.targets_si.items():
            if j in self._act:
                self.data.ctrl[self._act[j]] = x
        if self.mode == "physics":
            return
        q = self.data.qpos
        for j, x in self.targets_si.items():
            q[self._qadr[j]] = x
        for follower, leader in COUPLED_EYES.items():
            q[self._qadr[follower]] = q[self._qadr[leader]]
        # stand on the lowest foot
        q[self._qadr["root_z"]] = 0.0
        self.mj.mj_kinematics(self.model, self.data)
        q[self._qadr["root_z"]] = -min(self.data.site_xpos[i][2] for i in self._soles)
        self.data.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def targets(self) -> dict:
        """Current targets in degrees (hinges) or meters (slides)."""
        return {j: (math.degrees(x) if self.kind.get(j) == "hinge" else x) for j, x in self.targets_si.items()}

    # ----------------------------------------------------------------- physics
    def step(self, seconds: float):
        if self.mode == "kinematic":
            return
        n = int(round(min(max(seconds, 0.0), 0.25) / self.model.opt.timestep))
        if n:
            self.mj.mj_step(self.model, self.data, nstep=n)

    def angle(self, joint: str) -> float:
        x = self.data.qpos[self._qadr[joint]]
        return math.degrees(x) if self.kind.get(joint) == "hinge" else float(x)

    def errors(self) -> dict:
        """Target minus actual per body joint, degrees (always 0 in kinematic mode)."""
        return {j: math.degrees(x - self.data.qpos[self._qadr[j]]) for j, x in self.targets_si.items()
                if self.kind.get(j) == "hinge" and j not in FACE_JOINTS}

    def status(self) -> str:
        if self.mode == "kinematic":
            return f"kinematic: joints on the tracked angles, update {self.last_update_ms:.2f} ms"
        err = self.errors()
        worst = max(err, key=lambda a: abs(err[a])) if err else ""
        return f"physics: largest lag {worst} {err.get(worst, 0.0):+.0f} deg"

    def site(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    # ----------------------------------------------------------------- pictures
    def _render(self, camera, width, height):
        rs = self._renderer = self._renderer or {}
        r = rs.get((width, height))
        if r is None:
            r = rs[(width, height)] = self.mj.Renderer(self.model, height=height, width=width)
        r.update_scene(self.data, camera=camera)
        return r.render()

    def render(self, camera: str = "split", width: int = 640, height: int = 480) -> np.ndarray:
        """RGB image from a named camera: front, face, side, three_quarter or eyes; "split" is the front view with
        the face close-up in the corner."""
        if camera != "split":
            return self._render(camera, width, height).copy()
        body = self._render("front", width, height).copy()
        fw, fh = width * 2 // 5, height * 2 // 5
        face = self._render("face", fw, fh)
        body[8:8 + fh, width - fw - 8:width - 8] = face
        body[6:8, width - fw - 10:width - 6] = 255
        body[8 + fh:10 + fh, width - fw - 10:width - 6] = 255
        body[6:10 + fh, width - fw - 10:width - fw - 8] = 255
        body[6:10 + fh, width - 8:width - 6] = 255
        return body

    def close(self):
        for r in (self._renderer or {}).values():
            r.close()
        self._renderer = None

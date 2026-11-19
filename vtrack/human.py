"""A full human body in MuJoCo that mirrors you: arms, hands, legs, core, head, eyes and a face that makes your
expressions.

What it adds to the robot model in sim.py (which stays the hardware plan for Homunculus):
  * legs: hip flexion, abduction and rotation, knee, ankle dorsiflexion, inversion and abduction per leg, and the
    pelvis tilting side to side (pelvic hike), all with the tracker's names and signs
  * the whole core: flexion, side bend and rotation (the robot only builds rotation)
  * a face built like an animatronic head (animatronic.py): the ICT face (an open, sculpted human face) as the
    skin over a jaw hinge, eyes on joints and the skin servos of Will Cogley's DIY robot head, driven by
    MediaPipe's 52 face blendshapes and the emotion model's scores (face_rig.py)

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

from . import animatronic, face_rig
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

SKIN = animatronic.SKIN
SHIRT, PANTS, SHOE = "0.20 0.36 0.55 1", "0.18 0.19 0.24 1", "0.12 0.12 0.13 1"
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
                "jaw": Drive(60.0, 2.0, 6.0, 0.0005, "face"),           # hinge: N m per radian
                "jaw_slide": Drive(20000.0, 200.0, 60.0, 0.01, "face"),  # slide: N per meter
                "face_slide": Drive(400.0, 4.0, 4.0, 0.0005, "face")}    # a servo pulling a wire: N per meter
MIDLINE_FLIP = ["core_lateral_bend", "core_rotation", "neck_yaw", "neck_roll", "eye_yaw", "head_yaw", "head_roll"]
COUPLED_EYES = {"L_eye_yaw": "R_eye_yaw", "L_eye_pitch": "R_eye_pitch"}
VIEWS = ["split", "front", "face", "animatronic", "side", "three_quarter"]


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


def _head(m: _HumanMjcf, face: animatronic.Face):
    m.open(f'<body name="neck" pos="0 0 {CHEST_H + NECK_BASE}">')
    m.joint("neck_yaw", [0, 0, 1], HEAD_RANGES["neck_yaw"], "neck")
    m.joint("neck_pitch", [0, -1, 0], HEAD_RANGES["neck_pitch"], "neck")
    m.joint("neck_roll", [1, 0, 0], HEAD_RANGES["neck_roll"], "neck")
    # the neck's mass and collision shape; the skin is the face model's
    m.add(f'<geom type="capsule" fromto="0 0 -0.03 0.005 0 {NECK}" size="0.045" group="{animatronic.HIDDEN_GROUP}" '
          'class="robot"/>')
    m.open(f'<body name="head" pos="0 0 {NECK}">')
    face.bodies(m, EYE_RANGES)
    m.close("body")       # head
    m.close("body")       # neck


def build(floating: bool = True, scene: bool = True, rig: str = "cogley") -> tuple[str, _HumanMjcf]:
    """(MJCF text, builder) of the human. `floating`: the body can move up and down (kinematic mode puts it on
    the floor); otherwise the pelvis is fixed in place."""
    m = _HumanMjcf()
    m.face = face = animatronic.Face(head_to_chest=CHEST_H + NECK_BASE + NECK, rig=rig)
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
    face.materials(m)
    m.add(f'<mesh name="torso" vertex="{_f(_hull(TORSO_SECTIONS))}"/>')
    m.add(f'<mesh name="hips" vertex="{_f(_hull(PELVIS_SECTIONS))}"/>')
    m.close("asset")
    m.open("<worldbody>")
    # no cast shadows: at face scale MuJoCo's shadow map aliases into jagged patches
    m.add('<light pos="2 -1.5 3.5" dir="-0.5 0.3 -1" directional="true" diffuse="0.7 0.7 0.7" castshadow="false"/>')
    m.add('<light pos="2 1.5 2" dir="-0.6 -0.4 -0.3" directional="true" diffuse="0.25 0.25 0.28" castshadow="false"/>')
    # a soft fill from the front and a little below, as in portrait lighting, so the eye sockets are not black
    m.add('<light pos="3 0 1.2" dir="-1 0 0.25" directional="true" diffuse="0.32 0.30 0.29" specular="0 0 0" '
          'castshadow="false"/>')
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
    m.add(_look_camera("face", [0.62, 0.0, CHEST_H + 0.21], [0.0, 0.0, CHEST_H + NECK_BASE + NECK + 0.105], 25))
    for side in ("right", "left"):
        _arm(m, side, origin=[0, -SHOULDER_HALF, CHEST_H], look=HUMAN_LOOK)
    _head(m, face)
    m.close("body")       # chest
    m.close("body")       # pelvis
    m.close("worldbody")
    face.deformable(m, jaw_bind=face.jaw["pivot"])
    m.open("<tendon>")
    face.tendons(m)
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


def build_xml(floating: bool = True, scene: bool = True, rig: str = "cogley") -> str:
    return build(floating, scene, rig)[0]


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


def body_targets(values: dict, joints, hand_side: str = "right", skip=()) -> dict:
    """Tracked body angles (degrees) by joint name, for the joints the values cover (NaN entries are skipped)."""
    cols = set(values)
    out = {}
    for j in joints:
        if j in COUPLED_EYES or j in skip or j == "root_z":
            continue
        if j == "pelvis_roll":
            r, lft = _num(values.get("R_pelvic_hike")), _num(values.get("L_pelvic_hike"))
            v = r if math.isfinite(r) else -lft
        else:
            col = source_column(j, cols, hand_side)
            v = _num(values.get(col)) if col else math.nan
        if math.isfinite(v):
            out[j] = v
    return out


class HumanSim:
    """The full human, driven by tracker values (a CSV row or the live app's values). Same interface as
    sim.TandemSim (command, step, render, errors, close), so the live app can show either."""

    VIEWS = VIEWS

    def __init__(self, mode: str = "kinematic", scene: bool = True, hand_side: str = "right", mirror: bool = True,
                 face_source: str = "auto", emotion_gain: float = 1.25, smooth_face: bool = True,
                 rig: str = "cogley"):
        import mujoco
        if mode not in ("kinematic", "physics"):
            raise ValueError("mode must be 'kinematic' or 'physics'")
        self.mj, self.mode, self.mirror = mujoco, mode, mirror
        self.hand_side, self.face_source, self.emotion_gain = hand_side, face_source, emotion_gain
        self.xml, b = build(floating=mode == "kinematic", scene=scene, rig=rig)
        self.kind = b.kind
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.face = b.face
        self.face.attach(self.model)
        self.face_channels = set(self.face.channels)
        self.joints = [n for n, _, _ in b.joints]
        self._qadr = {self.model.joint(i).name: int(self.model.jnt_qposadr[i]) for i in range(self.model.njnt)}
        self._act = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self._soles = [self.model.site(f"{p}sole{i}").id for p in SIDES.values() for i in range(4)]
        self.targets_si = {j: 0.0 for j in self.face.channels}  # radians or meters; 0 is the neutral face
        self._bs = None                   # last blendshapes the face used
        self.servos = {}                  # last servo values, -1..1
        self._face_filter = OneEuroFilter(min_cutoff=2.5, beta=0.8) if smooth_face else None
        self._renderer = None
        self.last_update_ms = 0.0
        self._apply()

    # ----------------------------------------------------------------- commands
    def _si(self, joint, v):
        return math.radians(v) if self.kind.get(joint) == "hinge" else float(v)

    def _servo_targets(self, servos: dict) -> dict:
        out = {}
        for n, a in servos.items():
            x = self.face.target(n, a)
            out[n] = math.degrees(x) if self.kind.get(n) == "hinge" else x
        return out

    def face_targets(self, values: dict, eyes=(0.0, 0.0)) -> dict:
        """Servo targets (the jaw hinge in degrees, the skin servos in meters of anchor travel) from blendshapes and
        emotions, with the lids following the eyes (yaw, pitch in degrees); the last face holds when both are
        missing."""
        bs = face_rig.combine(face_rig.blendshapes_from_values(values), face_rig.emotions_from_values(values),
                              self.face_source, self.emotion_gain)
        if bs is None:
            return {}
        self._bs = bs
        servos = self.face.servo_values(bs, *eyes)
        if self._face_filter is not None:
            t = _num(values.get("t"))
            t = t if math.isfinite(t) else time.monotonic()
            vec = self._face_filter(np.array([servos[k] for k in self.face.channels]), t)
            servos = dict(zip(self.face.channels, (float(x) for x in vec)))
        self.servos = servos
        return self._servo_targets(servos)

    def servo_commands(self) -> dict:
        """The face's last servo positions (-1..1, the jaw 0..1), by Will Cogley's channel names on the cogley rig."""
        return self.face.commands(self.servos) if self.face.rig == "cogley" else dict(self.servos)

    def command(self, values: dict) -> dict:
        """Take new tracker values (degrees; missing or NaN values hold the last pose). Kinematic mode sets the
        pose now; physics mode sets the servo targets. Returns the targets that changed (degrees or meters)."""
        t0 = time.perf_counter()
        v = mirror_values(values) if self.mirror else values
        # one-arm recordings have no L_/R_ prefix: in mirror mode they drive the other arm
        side = {"right": "left", "left": "right"}[self.hand_side] if self.mirror else self.hand_side
        changed = body_targets(v, self.joints, side, self.face_channels)
        if not any(math.isfinite(_num(v.get(e))) for e in ("eye_yaw", "eye_pitch")):
            bs = face_rig.blendshapes_from_values(v)
            if bs is not None:
                changed["R_eye_yaw"], changed["R_eye_pitch"] = face_rig.gaze_from_blendshapes(bs)
        eyes = tuple(changed.get(k, math.degrees(self.targets_si.get(k, 0.0))) for k in ("R_eye_yaw", "R_eye_pitch"))
        changed.update(self.face_targets(v, eyes))
        if "eyes_open" in v and face_rig.blendshapes_from_values(v) is None and self.face_source != "emotion":
            o = _num(v["eyes_open"])                 # no blendshapes: blink from the tracker's eyes_open
            if math.isfinite(o) and o < 0.5:
                bs = {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}
                shut = self.face.servo_values(bs)
                lids = [n for n in shut if "lid" in n or n.startswith("eyeBlink")]
                changed.update(self._servo_targets({n: shut[n] for n in lids}))
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
            self.face.update(self.data.qpos)
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
        self.face.update(q)

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
            self.face.update(self.data.qpos)

    def angle(self, joint: str) -> float:
        x = self.data.qpos[self._qadr[joint]]
        return math.degrees(x) if self.kind.get(joint) == "hinge" else float(x)

    def errors(self) -> dict:
        """Target minus actual per body joint, degrees (always 0 in kinematic mode)."""
        return {j: math.degrees(x - self.data.qpos[self._qadr[j]]) for j, x in self.targets_si.items()
                if self.kind.get(j) == "hinge" and j not in self.face_channels}

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
    def _render(self, camera, width, height, option=None):
        rs = self._renderer = self._renderer or {}
        r = rs.get((width, height))
        if r is None:
            r = rs[(width, height)] = self.mj.Renderer(self.model, height=height, width=width)
        r.update_scene(self.data, camera=camera, scene_option=option)
        return r.render()

    def render_mechanism(self, width: int = 640, height: int = 480, skin_alpha: float = 0.22) -> np.ndarray:
        """The face close-up with the skin see-through, showing the servos, wires and anchors under it."""
        opt = self.mj.MjvOption()
        opt.geomgroup[animatronic.MECH_GROUP] = 1
        opt.tendongroup[animatronic.MECH_GROUP] = 1
        rgba = self.model.skin_rgba.copy()
        for name in animatronic.DEFORMING:
            sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SKIN, name)
            self.model.skin_rgba[sid, 3] = skin_alpha
        try:
            return self._render("face", width, height, opt).copy()
        finally:
            self.model.skin_rgba[:] = rgba

    def render(self, camera: str = "split", width: int = 640, height: int = 480) -> np.ndarray:
        """RGB image from a named camera: front, face, side, three_quarter or eyes; "split" is the front view with
        the face close-up in the corner; "animatronic" is the face with its mechanism showing through the skin."""
        if camera == "animatronic":
            return self.render_mechanism(width, height)
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

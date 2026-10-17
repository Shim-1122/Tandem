"""The robot's body: named skills turned into joint targets for every joint, with speed limits.

Every value uses Tandem's column names and sign conventions (`R_shoulder_flexion`, `neck_yaw`, `eye_yaw`,
`eyes_open`, ...), so the same dict drives the MuJoCo robot (`vtrack.sim.TandemSim.command`) and, later, the real
arm through `robot.SerialRobot`.

The body is built from layers, applied in this order every tick:
    base       rest pose, or shadow mode (mirror the person, or copy them joint for joint as `live.py --sim` does)
    hands      a held grip or hand shape per side (grip, release)
    attention  where the head and eyes look: at the person, at a direction, or nowhere
    motion     the current skill (wave, nod, point...), which overrides only the joints it moves
    freeze     stop: hold every joint exactly where it is until the next command
Then a per-joint speed limit makes every change, including starting and stopping a skill, smooth and safe.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

ARM = ("shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion", "forearm_rotation",
       "wrist_flexion", "wrist_deviation")
FINGERS = ("index", "middle", "ring", "pinky")
FINGER_COLS = tuple(f"{f}_{j}" for f in FINGERS for j in ("mcp_flex", "mcp_abd", "pip", "dip")) + tuple(
    f"thumb_{j}" for j in ("cmc_palmar", "cmc_radial", "cmc_rotation", "mcp", "ip"))
HEAD = ("neck_yaw", "neck_pitch", "neck_roll", "eye_yaw", "eye_pitch")
PRE = {"right": "R_", "left": "L_"}

# Fastest each kind of joint may move, degrees per second. Conservative for a first pneumatic build.
MAX_SPEED = {"shoulder": 90.0, "elbow": 120.0, "wrist": 150.0, "finger": 300.0, "thumb": 300.0, "neck": 120.0,
             "eye": 400.0, "core": 45.0}

REST_ARM = {"shoulder_flexion": 0.0, "shoulder_abduction": 5.0, "shoulder_rotation": 0.0, "elbow_flexion": 15.0,
            "forearm_rotation": 0.0, "wrist_flexion": 0.0, "wrist_deviation": 0.0}


def speed_limit(col: str) -> float | None:
    """Degrees per second for a column, or None for values that are not angles (eyes_open)."""
    if col == "eyes_open":
        return None
    name = col[2:] if col[:2] in ("L_", "R_") else col
    for key in ("shoulder", "elbow", "wrist", "forearm", "thumb", "neck", "eye", "core"):
        if name.startswith(key):
            return MAX_SPEED["wrist" if key == "forearm" else key]
    return MAX_SPEED["finger"]


def hand_shape(shape: str, amount: float = 1.0) -> dict:
    """Finger angles for one hand (no side prefix). shape: open, fist, grip, point, thumbs_up, pinch, peace."""
    g = max(0.0, min(1.0, amount))
    curl = {"open": 0.0, "fist": 1.0, "grip": g, "thumbs_up": 1.0, "point": 1.0, "pinch": 0.6,
            "peace": 1.0}.get(shape, 0.0)
    v = {}
    for f in FINGERS:
        c = 0.0 if (shape == "point" and f == "index") or (shape == "peace" and f in ("index", "middle")) else curl
        if shape == "pinch" and f == "index":
            c = 0.55
        spread = {"index": 10.0, "middle": -10.0}.get(f, 0.0) if shape == "peace" else 0.0   # the V of scissors
        v.update({f"{f}_mcp_flex": 75 * c, f"{f}_mcp_abd": spread, f"{f}_pip": 95 * c, f"{f}_dip": 55 * c})
    if shape == "thumbs_up":                       # thumb straight up, out of the fist
        v.update({"thumb_cmc_palmar": 10.0, "thumb_cmc_radial": 40.0, "thumb_cmc_rotation": 0.0,
                  "thumb_mcp": 0.0, "thumb_ip": 0.0})
    else:
        t = 1.0 if shape in ("fist", "point", "peace") else (0.7 if shape == "pinch" else curl)
        v.update({"thumb_cmc_palmar": 20 + 30 * t, "thumb_cmc_radial": 30 - 20 * t,
                  "thumb_cmc_rotation": 40 + 40 * t, "thumb_mcp": 10 + 30 * t, "thumb_ip": 10 + 30 * t})
    return v


def sided(side: str, values: dict) -> dict:
    return {PRE[side] + k: v for k, v in values.items()}


@dataclass
class Motion:
    """A skill in progress: `pose(u)` gives the joints it moves `u` seconds after it starts."""
    name: str
    duration: float
    pose: object
    start: float = 0.0


# ---------------------------------------------------------------------- skills
def wave(side: str = "right", seconds: float = 3.0) -> Motion:
    def pose(u):
        w = math.sin(2 * math.pi * 1.2 * u)
        return sided(side, {"shoulder_flexion": 10, "shoulder_abduction": 95, "shoulder_rotation": -70 + 12 * w,
                            "elbow_flexion": 85 + 25 * w, "forearm_rotation": 0, "wrist_flexion": 0,
                            "wrist_deviation": 0, **hand_shape("open")})
    return Motion(f"wave {side}", seconds, pose)


def nod(times: int = 2) -> Motion:
    return Motion("nod", 0.7 * times, lambda u: {"neck_pitch": -15 * abs(math.sin(math.pi * u / 0.7))})


def shake_head(times: int = 2) -> Motion:
    return Motion("shake head", 0.8 * times, lambda u: {"neck_yaw": 25 * math.sin(2 * math.pi * u / 0.8)})


def thumbs_up(side: str = "right", seconds: float = 2.5) -> Motion:
    return Motion(f"thumbs up {side}", seconds, lambda u: sided(side, {
        "shoulder_flexion": 40, "shoulder_abduction": 10, "shoulder_rotation": 0, "elbow_flexion": 75,
        "forearm_rotation": 0, "wrist_flexion": 0, "wrist_deviation": 0, **hand_shape("thumbs_up")}))


def point(yaw: float = 0.0, pitch: float = 0.0, side: str | None = None, seconds: float = 3.0) -> Motion:
    """Point at a direction (degrees, yaw + to the robot's left): the waist turns toward it, the arm on that side
    points straight out from the chest, and the head looks along it."""
    side = side or ("left" if yaw > 0 else "right")
    waist = max(-60.0, min(60.0, yaw))
    return Motion(f"point {side}", seconds, lambda u: {
        "core_rotation": waist, "neck_yaw": max(-80.0, min(80.0, yaw - waist)), "neck_pitch": pitch * 0.6,
        **sided(side, {"shoulder_flexion": max(-30.0, min(150.0, 90 + pitch)), "shoulder_abduction": 0,
                       "shoulder_rotation": 0, "elbow_flexion": 5, "forearm_rotation": 30, "wrist_flexion": 0,
                       "wrist_deviation": 0, **hand_shape("point")})})


def reach(side: str = "right", seconds: float = 4.0) -> Motion:
    """Hold an open hand out in front, palm up, to take or offer something."""
    return Motion(f"reach {side}", seconds, lambda u: sided(side, {
        "shoulder_flexion": 55, "shoulder_abduction": 5, "shoulder_rotation": 0, "elbow_flexion": 35,
        "forearm_rotation": -80, "wrist_flexion": 0, "wrist_deviation": 0, **hand_shape("open")}))


def celebrate(seconds: float = 3.0) -> Motion:
    def pose(u):
        w = math.sin(2 * math.pi * 1.5 * u)
        v = {"neck_pitch": 10}
        for sd in ("right", "left"):
            v.update(sided(sd, {"shoulder_flexion": 160 + 10 * w, "shoulder_abduction": 20, "shoulder_rotation": 0,
                                "elbow_flexion": 20 + 15 * w, "forearm_rotation": 0, "wrist_flexion": 0,
                                "wrist_deviation": 0, **hand_shape("fist")}))
        return v
    return Motion("celebrate", seconds, pose)


def shrug(seconds: float = 2.0) -> Motion:
    def pose(u):
        v = {"neck_roll": 8}
        for sd in ("right", "left"):
            v.update(sided(sd, {"shoulder_flexion": 10, "shoulder_abduction": 25, "shoulder_rotation": 50,
                                "elbow_flexion": 90, "forearm_rotation": -80, "wrist_flexion": -20,
                                "wrist_deviation": 0, **hand_shape("open")}))
        return v
    return Motion("shrug", seconds, pose)


RPS_HAND = {"rock": "fist", "paper": "open", "scissors": "peace"}
RPS_ARM = {"shoulder_flexion": 30, "shoulder_abduction": 10, "shoulder_rotation": -25, "elbow_flexion": 95,
           "forearm_rotation": 0, "wrist_flexion": 0, "wrist_deviation": 0}


def rps_ready(side: str = "right", seconds: float = 30.0) -> Motion:
    """Rock paper scissors, get set: forearm forward and turned a little out (so the hand is clear of the chest),
    hand in a fist."""
    return Motion("rps ready", seconds, lambda u: sided(side, {**RPS_ARM, **hand_shape("fist")}))


def rps_count(beats: int = 3, beat_s: float = 0.6, side: str = "right") -> Motion:
    """Pump the fist once per beat ("rock", "paper", "scissors"): down on the beat, back up between."""
    def pose(u):
        dip = math.sin(math.pi * min(1.0, (u % beat_s) / beat_s))        # 0 at each beat, 1 halfway, 0 again
        return sided(side, {**RPS_ARM, "shoulder_flexion": 30 + 12 * dip, "elbow_flexion": 95 - 40 * dip,
                            **hand_shape("fist")})
    return Motion("rps count", beats * beat_s, pose)


def rps_throw(choice: str, side: str = "right", seconds: float = 1.6) -> Motion:
    """"Shoot!": forearm down to the front and the hand in the chosen shape, held so it can be seen."""
    return Motion(f"rps {choice}", seconds, lambda u: sided(side, {
        **RPS_ARM, "shoulder_flexion": 42, "elbow_flexion": 55, "forearm_rotation": 0 if choice == "scissors" else 70,
        **hand_shape(RPS_HAND[choice])}))


SKILLS = {"wave": wave, "nod": nod, "shake_head": shake_head, "thumbs_up": thumbs_up, "point": point,
          "reach": reach, "celebrate": celebrate, "shrug": shrug}


def rest_pose() -> dict:
    v = {"core_rotation": 0.0, **{h: 0.0 for h in HEAD}, "eyes_open": 1}
    for sd in PRE:
        v.update(sided(sd, {**REST_ARM, **hand_shape("open", 0.0), **hand_shape("grip", 0.15)}))
    return v


SHADOW_COLS = ("core_rotation", "neck_yaw", "neck_pitch", "neck_roll", "eye_yaw", "eye_pitch", "eyes_open") + tuple(
    p + c for p in PRE.values() for c in ARM + FINGER_COLS)


# Mirror image: each side copies the other side's arm (the two arms already share sign conventions, mirrored
# internally), and turning or tilting to one side becomes the other.
MIRROR_NEGATE = {"core_rotation", "neck_yaw", "neck_roll", "eye_yaw"}


def mirror_source(col: str) -> str:
    if col.startswith("R_"):
        return "L_" + col[2:]
    if col.startswith("L_"):
        return "R_" + col[2:]
    return col


# ---------------------------------------------------------------------- body
class Body:
    """Holds the robot's state and turns it into joint targets: `targets(now, tracker_values, look_dir)`."""

    def __init__(self, blink: bool = True, seed: int = 0):
        self.shadow = False
        self.mirror = True                                # shadow as a mirror image (see set_shadow)
        self.frozen = False
        self.motion: Motion | None = None
        self.queue: list[Motion] = []
        self.grips: dict[str, tuple[str, float]] = {}     # side -> (shape, amount)
        self.attention = None                             # None | "person" | (yaw, pitch)
        self.blink = blink
        self._rng = random.Random(seed)
        self._next_blink = 3.0
        self._blink_until = -1.0
        self.out: dict = rest_pose()
        self._t = None

    # ------------------------------------------------------------- commands
    def do(self, motion: Motion, now: float, queue: bool = False):
        self.frozen = False
        if queue and self.motion is not None:
            self.queue.append(motion)
            return
        motion.start = now
        self.motion, self.queue = motion, []

    def grip(self, side: str, amount: float = 1.0, shape: str = "grip"):
        self.frozen = False
        self.grips[side] = (shape, max(0.0, min(1.0, amount)))

    def release(self, side: str | None = None):
        self.frozen = False
        for sd in ([side] if side else list(PRE)):
            self.grips[sd] = ("open", 0.0)

    def look(self, target):
        """target: "person", a (yaw, pitch) direction, or None (look ahead)."""
        self.attention = target

    def rest(self):
        self.frozen = self.shadow = False
        self.motion, self.queue, self.grips, self.attention = None, [], {}, None

    def freeze(self):
        self.frozen = True
        self.motion, self.queue = None, []

    def set_shadow(self, on: bool, mirror: bool | None = None):
        """Copy the person. mirror=True (the default): like a reflection, facing you the robot moves its left arm
        when you move your right, and turns its head the same way you see yours turn. mirror=False: joint for
        joint, its right arm copies your right arm (what live.py --sim does)."""
        self.frozen = False
        self.shadow = on
        if mirror is not None:
            self.mirror = mirror
        if on:
            self.motion, self.queue = None, []

    @property
    def busy(self) -> bool:
        return self.motion is not None

    def state(self) -> str:
        if self.frozen:
            return "frozen"
        if self.shadow:
            return "mirroring you" if self.mirror else "copying you"
        return self.motion.name if self.motion else "idle"

    # ------------------------------------------------------------- targets
    def targets(self, now: float, tracker: dict | None = None, person_dir=None) -> dict:
        dt = 0.0 if self._t is None else max(0.0, now - self._t)
        self._t = now
        if self.frozen:
            return dict(self.out)
        goal = rest_pose()
        if self.shadow and tracker:
            for c in SHADOW_COLS:
                v = _num(tracker.get(mirror_source(c) if self.mirror else c))
                if math.isfinite(v):
                    goal[c] = -v if self.mirror and c in MIRROR_NEGATE else v
        for sd, (shape, amount) in self.grips.items():
            goal.update(sided(sd, hand_shape(shape, amount)))
        if not self.shadow:
            look = person_dir if self.attention == "person" else (
                self.attention if isinstance(self.attention, tuple) else None)
            if look is not None:
                yaw, pitch = look
                goal["neck_yaw"], goal["neck_pitch"] = 0.75 * yaw, 0.75 * pitch
                goal["eye_yaw"], goal["eye_pitch"] = 0.25 * yaw, 0.25 * pitch
        if self.motion is not None:
            u = now - self.motion.start
            if u >= self.motion.duration:
                self.motion = None
                if self.queue:
                    self.motion = self.queue.pop(0)
                    self.motion.start, u = now, 0.0
            if self.motion is not None:
                goal.update(self.motion.pose(u))
        if self.blink and not self.shadow:
            if now >= self._next_blink:
                self._blink_until = now + 0.15
                self._next_blink = now + self._rng.uniform(2.5, 6.0)
            goal["eyes_open"] = 0 if now < self._blink_until else 1
        self.out = self._limit(goal, dt)
        return dict(self.out)

    def _limit(self, goal: dict, dt: float) -> dict:
        out = {}
        for c, g in goal.items():
            cur = self.out.get(c)
            lim = speed_limit(c)
            if cur is None or lim is None or not math.isfinite(cur) or dt <= 0:
                out[c] = g if cur is None or lim is None or not math.isfinite(cur) else cur
                continue
            step = lim * dt
            out[c] = cur + max(-step, min(step, g - cur))
        return out


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan

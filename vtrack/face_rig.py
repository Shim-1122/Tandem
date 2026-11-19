"""Facial expressions for the simulated human: tracker values in, face joint targets out.

Two sources drive the face, and they can be used together:
  * MediaPipe's 52 face blendshapes (ARKit names, 0..1), from the same face model that already tracks the head.
    They measure what each part of the face is doing (jaw open, lip corner up, brow down, ...), every face frame,
    so they are the accurate, low-latency source.
  * The emotion model's seven scores (emotion_<name>_pct). Each emotion has a preset made of blendshapes, built from
    the facial action units of that emotion (Ekman and Friesen's FACS), so the face shows the emotion even when
    the blendshapes are unavailable, and reads clearly when the face model is unsure.

`source` "auto" takes, for each blendshape, the larger of the measured value and the emotion preset; "blendshapes"
or "emotion" uses one source only.

Left and Right in blendshape names are the person's own sides (MediaPipe follows ARKit). The face joints use the
robot's prefixes: R_ is the person's right (copy mode). vtrack.human swaps sides for mirror mode.

Units: hinge joints in degrees, slide joints in meters.
"""
from __future__ import annotations

import math

# ARKit blendshape names in MediaPipe's output order (its first entry, _neutral, is left out)
BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
BS_COLS = [f"bs_{n}" for n in BLENDSHAPES]          # tracker CSV columns


def _both(**kw) -> dict:
    """Preset helper: names ending in _ are applied to the Left and Right blendshape."""
    out = {}
    for k, v in kw.items():
        if k.endswith("_"):
            out[k[:-1] + "Left"] = out[k[:-1] + "Right"] = v
        else:
            out[k] = v
    return out


# Emotion presets from the action units of each emotion (AU numbers in the comments).
EMOTION_PRESETS = {
    # AU6 cheek raiser, AU12 lip corner puller (Duchenne smile), a little AU25/26 lips apart
    "happy": _both(mouthSmile_=0.9, cheekSquint_=0.65, eyeSquint_=0.25, mouthDimple_=0.2, jawOpen=0.12),
    # AU1 inner brow raiser, AU4 brow lowerer, AU15 lip corner depressor, AU17 chin raiser; lids droop
    "sad": _both(browInnerUp=0.9, browDown_=0.15, mouthFrown_=0.75, mouthShrugLower=0.45, eyeBlink_=0.2,
                 eyeSquint_=0.1),
    # AU4 brow lowerer, AU5 upper lid raiser, AU7 lid tightener, AU23/24 lip tightener and presser
    "angry": _both(browDown_=0.95, eyeWide_=0.25, eyeSquint_=0.55, mouthPress_=0.7, mouthFrown_=0.3,
                   noseSneer_=0.35),
    # AU1+2 brow raisers, AU4, AU5 upper lid raiser, AU20 lip stretcher, AU26 jaw drop
    "fearful": _both(browInnerUp=0.95, browOuterUp_=0.55, browDown_=0.3, eyeWide_=0.85, mouthStretch_=0.75,
                     jawOpen=0.25),
    # AU1+2, AU5, AU26
    "surprised": _both(browInnerUp=0.9, browOuterUp_=0.9, eyeWide_=0.9, jawOpen=0.6),
    # AU9 nose wrinkler, AU10 upper lip raiser, AU15, AU17; lowered brows
    "disgust": _both(noseSneer_=0.9, mouthUpperUp_=0.75, cheekSquint_=0.45, browDown_=0.55, mouthFrown_=0.35,
                     mouthShrugLower=0.35, eyeSquint_=0.35),
    "neutral": {},
}
EMOTION_NAMES = list(EMOTION_PRESETS)

# Face joints: name -> (kind, low, high, rest). Hinges in degrees, slides in meters.
FACE_JOINTS = {
    "jaw_open": ("hinge", -2.0, 30.0, 0.0),
    "jaw_side": ("slide", -0.008, 0.008, 0.0),
    "upper_lip_up": ("slide", -0.004, 0.010, 0.0),
    "upper_lip_forward": ("slide", -0.004, 0.012, 0.0),
    "lower_lip_down": ("slide", -0.020, 0.012, 0.0),
    "lower_lip_forward": ("slide", -0.004, 0.012, 0.0),
}
for _p in ("R_", "L_"):
    FACE_JOINTS.update({
        _p + "eyelid": ("hinge", -15.0, 90.0, 25.0),        # upper lid: 0 on top of the eye, about 85 closed
        _p + "lower_lid": ("hinge", 0.0, 55.0, 18.0),
        _p + "brow_inner": ("slide", -0.012, 0.014, 0.0),
        _p + "brow_mid": ("slide", -0.012, 0.014, 0.0),
        _p + "brow_outer": ("slide", -0.012, 0.014, 0.0),
        _p + "mouth_corner_up": ("slide", -0.020, 0.016, 0.0),
        _p + "mouth_corner_out": ("slide", -0.012, 0.012, 0.0),
        _p + "cheek_raise": ("slide", 0.0, 0.012, 0.0),
        _p + "cheek_puff": ("slide", 0.0, 0.012, 0.0),
        _p + "nose_sneer": ("slide", 0.0, 0.006, 0.0),
    })
LID_CLOSED = 85.0
JAW_TO_LIP = 0.087          # meters from the jaw hinge to the lips: how far the lower lip drops per radian


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def blendshapes_from_values(values: dict) -> dict | None:
    """{name: 0..1} from bs_ columns, or None when the face model gave none this frame."""
    out = {}
    for n, c in zip(BLENDSHAPES, BS_COLS):
        v = _num(values.get(c))
        if math.isfinite(v):
            out[n] = min(1.0, max(0.0, v))
    return out or None


def emotions_from_values(values: dict) -> dict | None:
    """{emotion: 0..1} from emotion_<name>_pct columns, or None."""
    out = {}
    for e in EMOTION_NAMES:
        v = _num(values.get(f"emotion_{e}_pct"))
        if math.isfinite(v):
            out[e] = max(0.0, v) / 100.0
    return out or None


def emotion_blendshapes(emotions: dict, gain: float = 1.25) -> dict:
    """Blendshapes of a mix of emotions: each preset weighted by that emotion's score (scores add up to 1)."""
    out: dict = {}
    for e, p in emotions.items():
        for k, v in EMOTION_PRESETS.get(e, {}).items():
            out[k] = out.get(k, 0.0) + gain * p * v
    return {k: min(1.0, v) for k, v in out.items()}


def combine(bs: dict | None, emo: dict | None, source: str = "auto", gain: float = 1.25) -> dict | None:
    """The blendshapes that drive the face, from either source or both (see the module notes)."""
    eb = emotion_blendshapes(emo, gain) if emo and gain > 0 else None
    if source == "blendshapes":
        return bs
    if source == "emotion":
        return eb
    if bs is None:
        return eb
    if eb is None:
        return bs
    return {k: max(bs.get(k, 0.0), eb.get(k, 0.0)) for k in set(bs) | set(eb)}


def face_pose(bs: dict) -> dict:
    """Face joint targets (degrees or meters, within FACE_JOINTS ranges) from blendshapes (missing names are 0)."""
    g = bs.get
    jaw = 26.0 * g("jawOpen", 0.0)
    jaw_drop = JAW_TO_LIP * math.sin(math.radians(jaw))
    out = {"jaw_open": jaw, "jaw_side": 0.008 * (g("jawLeft", 0.0) - g("jawRight", 0.0))}
    pucker, funnel = g("mouthPucker", 0.0), g("mouthFunnel", 0.0)
    press = 0.5 * (g("mouthPressLeft", 0.0) + g("mouthPressRight", 0.0))
    shift = g("mouthLeft", 0.0) - g("mouthRight", 0.0)            # whole mouth toward the person's left
    for p, s, toward_left in (("R_", "Right", -1.0), ("L_", "Left", 1.0)):
        blink, wide, squint = g(f"eyeBlink{s}", 0.0), g(f"eyeWide{s}", 0.0), g(f"eyeSquint{s}", 0.0)
        cheek_sq, smile = g(f"cheekSquint{s}", 0.0), g(f"mouthSmile{s}", 0.0)
        down, outer_up, inner_up = g(f"browDown{s}", 0.0), g(f"browOuterUp{s}", 0.0), g("browInnerUp", 0.0)
        inner = 0.013 * inner_up - 0.010 * down
        outer = 0.012 * outer_up - 0.005 * down
        out.update({
            p + "eyelid": 25.0 + 62.0 * blink - 32.0 * wide + 10.0 * squint,
            p + "lower_lid": 18.0 + 22.0 * squint + 16.0 * cheek_sq + 4.0 * smile,
            p + "brow_inner": inner,
            p + "brow_mid": 0.55 * inner + 0.45 * outer,
            p + "brow_outer": outer,
            p + "mouth_corner_up": (0.013 * smile - 0.011 * g(f"mouthFrown{s}", 0.0)
                                    - 0.003 * g(f"mouthStretch{s}", 0.0) - 0.45 * jaw_drop),
            p + "mouth_corner_out": (0.007 * smile + 0.010 * g(f"mouthStretch{s}", 0.0)
                                     + 0.004 * g(f"mouthDimple{s}", 0.0) - 0.011 * pucker - 0.007 * funnel
                                     - 0.002 * g(f"mouthPress{s}", 0.0) + 0.008 * toward_left * shift),
            p + "cheek_raise": 0.009 * cheek_sq + 0.004 * smile,
            p + "cheek_puff": 0.012 * g("cheekPuff", 0.0),
            p + "nose_sneer": 0.006 * g(f"noseSneer{s}", 0.0),
        })
    up_up = 0.5 * (g("mouthUpperUpLeft", 0.0) + g("mouthUpperUpRight", 0.0))
    low_down = 0.5 * (g("mouthLowerDownLeft", 0.0) + g("mouthLowerDownRight", 0.0))
    out.update({
        "upper_lip_up": 0.008 * up_up - 0.003 * g("mouthRollUpper", 0.0) + 0.002 * g("mouthShrugUpper", 0.0),
        "upper_lip_forward": 0.010 * pucker + 0.008 * funnel - 0.003 * g("mouthRollUpper", 0.0),
        # mouthClose keeps the lips together while the jaw is open: the lower lip rises by what the jaw dropped
        "lower_lip_down": (0.009 * low_down + 0.003 * g("mouthRollLower", 0.0) - 0.004 * g("mouthShrugLower", 0.0)
                           - 0.002 * press - g("mouthClose", 0.0) * jaw_drop),
        "lower_lip_forward": (0.010 * pucker + 0.008 * funnel - 0.003 * g("mouthRollLower", 0.0)
                              + 0.004 * g("mouthShrugLower", 0.0)),
    })
    return {k: min(FACE_JOINTS[k][2], max(FACE_JOINTS[k][1], v)) for k, v in out.items()}


def rest_pose() -> dict:
    return {k: v[3] for k, v in FACE_JOINTS.items()}


def gaze_from_blendshapes(bs: dict) -> tuple[float, float]:
    """(eye yaw, eye pitch) in degrees from the eyeLook blendshapes, both eyes averaged.
    Yaw + turns the eyes to the person's left, pitch + looks up (the tracker's eye_yaw and eye_pitch)."""
    g = bs.get
    yaw_r = g("eyeLookInRight", 0.0) - g("eyeLookOutRight", 0.0)     # right eye looking in = toward the left
    yaw_l = g("eyeLookOutLeft", 0.0) - g("eyeLookInLeft", 0.0)
    pitch = 0.5 * (g("eyeLookUpLeft", 0.0) + g("eyeLookUpRight", 0.0)
                   - g("eyeLookDownLeft", 0.0) - g("eyeLookDownRight", 0.0))
    return 30.0 * 0.5 * (yaw_r + yaw_l), 25.0 * pitch

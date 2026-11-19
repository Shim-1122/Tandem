"""Facial expressions for the simulated human: tracker values in, the blendshapes that drive the animatronic face out.

Two sources drive the face, and they can be used together:
  * MediaPipe's 52 face blendshapes (ARKit names, 0..1), from the same face model that already tracks the head.
    They measure what each part of the face is doing (jaw open, lip corner up, brow down, ...), every face frame,
    so they are the accurate, low-latency source.
  * The emotion model's seven scores (emotion_<name>_pct). Each emotion has a preset made of blendshapes, built from
    the facial action units of that emotion (Ekman and Friesen's FACS), so the face shows the emotion even when
    the blendshapes are unavailable, and reads clearly when the face model is unsure.

`source` "auto" takes, for each blendshape, the larger of the measured value and the emotion preset; "blendshapes"
or "emotion" uses one source only.

The face is an animatronic head (vtrack/animatronic.py, laid out like Will Cogley's DIY robot head): its servos
turn by their share of each blendshape, as on the real head.

Left and Right in blendshape names are the person's own sides (MediaPipe follows ARKit). The servos use the
robot's prefixes: R_ is the person's right (copy mode). vtrack.human swaps sides for mirror mode.
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


# MediaPipe gives one browInnerUp for both brows; the face has a servo on each side
INPUT_ALIAS = {"browInnerUpLeft": "browInnerUp", "browInnerUpRight": "browInnerUp"}
GAZE_FULL = (30.0, 25.0)        # eye yaw and pitch (degrees) at a full eyeLook movement


def gaze_from_blendshapes(bs: dict) -> tuple[float, float]:
    """(eye yaw, eye pitch) in degrees from the eyeLook blendshapes, both eyes averaged.
    Yaw + turns the eyes to the person's left, pitch + looks up (the tracker's eye_yaw and eye_pitch)."""
    g = bs.get
    yaw_r = g("eyeLookInRight", 0.0) - g("eyeLookOutRight", 0.0)     # right eye looking in = toward the left
    yaw_l = g("eyeLookOutLeft", 0.0) - g("eyeLookInLeft", 0.0)
    pitch = 0.5 * (g("eyeLookUpLeft", 0.0) + g("eyeLookUpRight", 0.0)
                   - g("eyeLookDownLeft", 0.0) - g("eyeLookDownRight", 0.0))
    return GAZE_FULL[0] * 0.5 * (yaw_r + yaw_l), GAZE_FULL[1] * pitch

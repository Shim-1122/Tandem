"""A scripted visitor, for trying Jarvis with no camera: someone walks up, gestures, holds a ball, looks sad,
says a few commands, and leaves. `scenario(t)` gives tracker values like the live app's; `SAYS` the speech."""
from __future__ import annotations

import math

NAN = float("nan")

# (seconds, what the person says)
SAYS = [(26.0, "point to your left"), (31.0, "nod for me"), (34.0, "now wave with your left hand"),
        (39.0, "grip gently with your right hand"), (42.0, "what am I holding?"), (44.0, "let go and relax")]
LENGTH = 52.0


def _face(t: float, name: str = "shyam") -> dict:
    x = 25 * math.sin(2 * math.pi * t / 9.0)           # the visitor sways from side to side
    return {"head_x_cm": x, "head_y_cm": -5.0, "head_z_cm": 80.0, "identity": name, "eyes_open": 1}


def _emotions(**pct) -> dict:
    v = {f"emotion_{e}_pct": 0.0 for e in ("angry", "disgust", "fearful", "happy", "sad", "surprised")}
    v["emotion_neutral_pct"] = 100.0 - sum(pct.values())
    v.update({f"emotion_{k}_pct": p for k, p in pct.items()})
    return v


def scenario(t: float) -> dict:
    v = {"head_x_cm": NAN, "head_y_cm": NAN, "head_z_cm": NAN, "identity": "", "L_gesture": "unknown",
         "R_gesture": "unknown", "L_grasp_state": "free", "R_grasp_state": "free", "L_grasp_object": "",
         "R_grasp_object": "", "target_locked": 0, **_emotions()}
    if 1.0 <= t < 48.0:
        v.update(_face(t))
    if 7.0 <= t < 9.5:
        v["R_gesture"] = "open_palm"                     # the visitor waves (raises an open hand)
    if 12.0 <= t < 14.0:
        v["R_gesture"] = "thumbs_up"
    if 16.0 <= t < 30.0:
        v.update({"L_grasp_state": "holding", "L_grasp_object": "tennis ball"})
    if 20.0 <= t < 23.0:
        v.update(_emotions(sad=72.0))
    return v

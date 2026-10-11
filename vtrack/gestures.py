"""Rule-based hand gesture classification from 21 hand landmarks, plus debouncing."""
from __future__ import annotations

from enum import Enum

import numpy as np

from .geometry import FINGERS, WRIST, joint_angle


class Gesture(str, Enum):
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINT = "point"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    UNKNOWN = "unknown"


def finger_states(hand: np.ndarray, straight_deg: float = 160.0) -> dict[str, bool]:
    """True per finger when extended.

    Index to pinky: the PIP joint angle must be near straight. Thumb: its IP angle must be
    near straight and the tip must sit farther from the pinky base than the thumb MCP does,
    which rejects a thumb folded across the palm.
    """
    states = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        if name == "thumb":
            straight = joint_angle(hand[mcp], hand[pip], hand[dip]) > 150.0
            tip_far = np.linalg.norm(hand[tip] - hand[17]) > np.linalg.norm(hand[mcp] - hand[17])
            states[name] = bool(straight and tip_far)
        else:
            states[name] = joint_angle(hand[mcp], hand[pip], hand[dip]) > straight_deg
    return states


def classify_hand(hand: np.ndarray) -> Gesture:
    s = finger_states(hand)
    four = [s["index"], s["middle"], s["ring"], s["pinky"]]
    if all(four) and s["thumb"]:
        return Gesture.OPEN_PALM
    if not any(four):
        return Gesture.THUMBS_UP if s["thumb"] else Gesture.FIST
    if s["index"] and not any([s["middle"], s["ring"], s["pinky"]]) and not s["thumb"]:
        return Gesture.POINT
    if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
        return Gesture.PEACE
    return Gesture.UNKNOWN


class GestureDebouncer:
    """Emit a gesture only after it has been seen `hold` consecutive frames.

    Prevents a one-frame misclassification from triggering a robot action. Returns the
    newly confirmed gesture once on change, otherwise None.
    """

    def __init__(self, hold: int = 5):
        self.hold, self._cand, self._count, self.current = hold, None, 0, Gesture.UNKNOWN

    def update(self, g: Gesture) -> Gesture | None:
        if g == self._cand:
            self._count += 1
        else:
            self._cand, self._count = g, 1
        if self._count >= self.hold and g != self.current:
            self.current = g
            return g
        return None

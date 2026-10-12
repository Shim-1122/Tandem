"""Rule-based hand gesture classification from 21 hand landmarks, plus temporal filtering.

Finger states were calibrated on real MediaPipe hand landmarks (relaxed open hands, fists, pointing,
victory and thumbs-up photos). A finger (index to pinky) counts as extended when its tip is farther from
the wrist than its middle joint (PIP) and its PIP and DIP together bend less than 90 degrees; in the
calibration photos extended fingers scored 1.19 or more on the distance ratio and 57 degrees or less of
bend, curled fingers 0.83 or less and 119 degrees or more. The thumb counts as extended when its tip is
clearly farther from the pinky knuckle than the thumb's base joint is (ratio above 0.9; calibration:
extended 1.06 or more, folded 0.73 or less).

Thumbs up and down use the thumb's direction in the hand landmarks' axes, which follow the camera
(y points down in the image). With a camera that is level this is up and down in the room; with a
head-mounted camera looking down it is up and down in the camera's view.
"""
from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np

from .geometry import FINGERS, joint_angle


class Gesture(str, Enum):
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINT = "point"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    OK = "ok"
    PINCH = "pinch"
    L_SHAPE = "l_shape"
    THREE = "three"
    THREE_THUMB = "three_thumb"
    FOUR = "four"
    ROCK = "rock"
    I_LOVE_YOU = "i_love_you"
    CALL_ME = "call_me"
    PINKY_UP = "pinky_up"
    UNKNOWN = "unknown"


# How each gesture is made, for the on-screen legend and README
GESTURE_HELP = {
    Gesture.OPEN_PALM: "all five fingers out",
    Gesture.FIST: "all fingers and thumb curled",
    Gesture.POINT: "index only",
    Gesture.PEACE: "index and middle (V), thumb tucked",
    Gesture.THUMBS_UP: "thumb out pointing up, fingers curled",
    Gesture.THUMBS_DOWN: "thumb out pointing down, fingers curled",
    Gesture.OK: "thumb and index tips touching, other three fingers out",
    Gesture.PINCH: "thumb and index tips touching, other fingers curled",
    Gesture.L_SHAPE: "thumb and index out (L or finger gun)",
    Gesture.THREE: "index, middle and ring out, thumb tucked",
    Gesture.THREE_THUMB: "thumb, index and middle out",
    Gesture.FOUR: "four fingers out, thumb tucked",
    Gesture.ROCK: "index and pinky out, thumb tucked (horns)",
    Gesture.I_LOVE_YOU: "thumb, index and pinky out",
    Gesture.CALL_ME: "thumb and pinky out (shaka)",
    Gesture.PINKY_UP: "pinky only",
}

EXTENDED_TIP_RATIO = 1.0      # tip-to-wrist / PIP-to-wrist
EXTENDED_MAX_BEND = 90.0      # PIP + DIP flexion, degrees
THUMB_OUT_RATIO = 0.9         # thumb tip-to-pinky knuckle / thumb base-to-pinky knuckle
TOUCH_RATIO = 0.2             # thumb-index tip distance / (wrist to middle knuckle) for OK and pinch
THUMB_VERTICAL = 0.5          # |vertical component| of the thumb direction for up/down (30 deg from level)


def finger_states(hand: np.ndarray) -> dict[str, bool]:
    """True per finger when extended (see module docstring for the calibrated rules)."""
    states = {}
    for name, (a, b, c, tip) in FINGERS.items():
        if name == "thumb":
            # a = thumb CMC (landmark 1); 17 = pinky knuckle
            ratio = np.linalg.norm(hand[tip] - hand[17]) / max(np.linalg.norm(hand[a] - hand[17]), 1e-9)
            states[name] = bool(ratio > THUMB_OUT_RATIO)
        else:
            bend = (180.0 - joint_angle(hand[a], hand[b], hand[c])) + (180.0 - joint_angle(hand[b], hand[c], hand[tip]))
            ratio = np.linalg.norm(hand[tip] - hand[0]) / max(np.linalg.norm(hand[b] - hand[0]), 1e-9)
            states[name] = bool(ratio > EXTENDED_TIP_RATIO and bend < EXTENDED_MAX_BEND)
    return states


def thumb_vertical(hand: np.ndarray) -> float:
    """Vertical component of the thumb direction (base joint to tip): negative = up (camera y points down)."""
    d = hand[4] - hand[2]
    return float(d[1] / max(np.linalg.norm(d), 1e-9))


def tips_touching(hand: np.ndarray) -> bool:
    scale = np.linalg.norm(hand[9] - hand[0])
    return bool(np.linalg.norm(hand[4] - hand[8]) < TOUCH_RATIO * scale)


def classify_hand(hand: np.ndarray) -> Gesture:
    s = finger_states(hand)
    t, i, m, r, p = s["thumb"], s["index"], s["middle"], s["ring"], s["pinky"]
    if tips_touching(hand):
        return Gesture.OK if (m and r and p) else Gesture.PINCH
    pattern = (i, m, r, p)
    if pattern == (True, True, True, True):
        return Gesture.OPEN_PALM if t else Gesture.FOUR
    if pattern == (False, False, False, False):
        if not t:
            return Gesture.FIST
        v = thumb_vertical(hand)
        if v < -THUMB_VERTICAL:
            return Gesture.THUMBS_UP
        if v > THUMB_VERTICAL:
            return Gesture.THUMBS_DOWN
        return Gesture.UNKNOWN
    table = {
        (True, False, False, False): (Gesture.L_SHAPE, Gesture.POINT),
        (True, True, False, False): (Gesture.THREE_THUMB, Gesture.PEACE),
        (True, True, True, False): (Gesture.UNKNOWN, Gesture.THREE),
        (True, False, False, True): (Gesture.I_LOVE_YOU, Gesture.ROCK),
        (False, False, False, True): (Gesture.CALL_ME, Gesture.PINKY_UP),
    }
    if pattern in table:
        with_thumb, without_thumb = table[pattern]
        return with_thumb if t else without_thumb
    return Gesture.UNKNOWN


class GestureFilter:
    """Majority vote over the last `window_s` seconds; the same response time at any frame rate.

    Returns the gesture held by at least `min_share` of the recent frames, otherwise UNKNOWN, so a
    gesture never stays on screen after the hand has changed shape.
    """

    def __init__(self, window_s: float = 0.2, min_share: float = 0.6):
        self.window_s, self.min_share = window_s, min_share
        self._buf: deque = deque()
        self.current = Gesture.UNKNOWN

    def update(self, g: Gesture, t: float) -> Gesture:
        self._buf.append((t, g))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        counts: dict = {}
        for _, x in self._buf:
            counts[x] = counts.get(x, 0) + 1
        best, n = max(counts.items(), key=lambda kv: kv[1])
        self.current = best if n >= self.min_share * len(self._buf) else Gesture.UNKNOWN
        return self.current

    def reset(self):
        self._buf.clear()
        self.current = Gesture.UNKNOWN


class GestureDebouncer:
    """Emit a gesture only after it has been seen `hold` consecutive frames (frame-count version).

    Returns the newly confirmed gesture once on change, otherwise None. GestureFilter is preferred for
    live use because it is frame-rate independent.
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

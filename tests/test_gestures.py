import numpy as np
import pytest

from tests.conftest import make_hand
from tests.test_fingers import GEOMETRY
from vtrack.fingers import hand_fk
from vtrack.gestures import (GESTURE_HELP, Gesture, GestureDebouncer, GestureFilter, classify_hand,
                             finger_states, tips_touching)
from vtrack.kinematics import rot_x, rot_y, rot_z, shortest_arc

OUT = dict(mcp_flex=5.0, mcp_abd=0.0, pip=5.0, dip=5.0)
CURL = dict(mcp_flex=70.0, mcp_abd=0.0, pip=100.0, dip=70.0)
THUMB_OUT = dict(cmc_palmar=15.0, cmc_radial=45.0, cmc_rotation=0.0, mcp=5.0, ip=5.0)
THUMB_IN = dict(cmc_palmar=20.0, cmc_radial=10.0, cmc_rotation=0.0, mcp=80.0, ip=60.0)   # tip across the palm
FINGERS4 = ("index", "middle", "ring", "pinky")


def hand_palm(thumb: bool, extended=()):
    """(21, 3) right hand in palm coordinates (x thumb side, y toward wrist, z back of hand)."""
    a = {f: dict(OUT if f in extended else CURL) for f in FINGERS4}
    a["thumb"] = dict(THUMB_OUT if thumb else THUMB_IN)
    return hand_fk(a, GEOMETRY)


def touching(others_out: bool):
    """Thumb tip moved onto the curled index tip (OK when the other three fingers are out)."""
    a = {f: dict(OUT if (others_out and f != "index") else CURL) for f in FINGERS4}
    a["index"] = dict(mcp_flex=40.0, mcp_abd=0.0, pip=60.0, dip=30.0)
    a["thumb"] = dict(THUMB_OUT)
    P = hand_fk(a, GEOMETRY)
    P[4] = P[8] + np.array([0.004, 0.0, 0.0])
    P[3] = (P[2] + P[4]) / 2
    return P


def place(P, rng, thumb_dir=None, mirror=False):
    """Random camera orientation; if thumb_dir is given, the thumb is turned to point that way."""
    if thumb_dir is not None:
        d = (P[4] - P[2]) / np.linalg.norm(P[4] - P[2])
        R = rot_y(rng.uniform(-np.pi, np.pi)) @ shortest_arc(d, np.asarray(thumb_dir, float))
    else:
        R = rot_z(rng.uniform(-3, 3)) @ rot_y(rng.uniform(-3, 3)) @ rot_x(rng.uniform(-3, 3))
    W = (P - P.mean(axis=0)) @ R.T
    return W * np.array([-1.0, 1.0, 1.0]) if mirror else W


CASES = {
    Gesture.OPEN_PALM: lambda: hand_palm(True, FINGERS4),
    Gesture.FIST: lambda: hand_palm(False),
    Gesture.POINT: lambda: hand_palm(False, ("index",)),
    Gesture.PEACE: lambda: hand_palm(False, ("index", "middle")),
    Gesture.L_SHAPE: lambda: hand_palm(True, ("index",)),
    Gesture.THREE: lambda: hand_palm(False, ("index", "middle", "ring")),
    Gesture.THREE_THUMB: lambda: hand_palm(True, ("index", "middle")),
    Gesture.FOUR: lambda: hand_palm(False, FINGERS4),
    Gesture.ROCK: lambda: hand_palm(False, ("index", "pinky")),
    Gesture.I_LOVE_YOU: lambda: hand_palm(True, ("index", "pinky")),
    Gesture.CALL_ME: lambda: hand_palm(True, ("pinky",)),
    Gesture.PINKY_UP: lambda: hand_palm(False, ("pinky",)),
    Gesture.OK: lambda: touching(True),
    Gesture.PINCH: lambda: touching(False),
}


def test_sixteen_gestures_have_help_text():
    named = [g for g in Gesture if g != Gesture.UNKNOWN]
    assert len(named) == 16 and set(named) == set(GESTURE_HELP)


@pytest.mark.parametrize("gesture", list(CASES), ids=lambda g: g.value)
@pytest.mark.parametrize("mirror", [False, True], ids=["right", "left"])
def test_each_gesture_any_orientation_either_hand(gesture, mirror):
    rng = np.random.default_rng(hash(gesture.value) % 1000)
    for _ in range(25):
        assert classify_hand(place(CASES[gesture](), rng, mirror=mirror)) == gesture


@pytest.mark.parametrize("mirror", [False, True], ids=["right", "left"])
def test_thumbs_up_and_down_follow_camera_vertical(mirror):
    rng = np.random.default_rng(7)
    for _ in range(25):
        assert classify_hand(place(hand_palm(True), rng, (0, -1, 0), mirror)) == Gesture.THUMBS_UP   # y down
        assert classify_hand(place(hand_palm(True), rng, (0, 1, 0), mirror)) == Gesture.THUMBS_DOWN
        assert classify_hand(place(hand_palm(True), rng, (1, 0, 0), mirror)) == Gesture.UNKNOWN       # sideways


def test_finger_states_and_touch_helpers():
    s = finger_states(hand_palm(True, ("index", "pinky")))
    assert s == {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": True}
    assert tips_touching(touching(True)) and not tips_touching(hand_palm(True, FINGERS4))


def test_simple_test_hand_still_classifies():
    assert classify_hand(make_hand(extended=("index", "middle", "ring", "pinky"), thumb=True)) == Gesture.OPEN_PALM
    assert classify_hand(make_hand()) == Gesture.FIST
    assert classify_hand(make_hand(extended=("index",))) == Gesture.POINT
    assert classify_hand(make_hand(extended=("index", "middle"))) == Gesture.PEACE


def test_gesture_filter_majority_and_no_stale_output():
    f = GestureFilter(window_s=0.2, min_share=0.6)
    t = 0.0
    for _ in range(12):                                   # 60 fps of fist
        t += 1 / 60
        assert f.update(Gesture.FIST, t) in (Gesture.FIST,)
    out = [f.update(Gesture.PEACE if i % 2 else Gesture.OK, t + (i + 1) / 60) for i in range(12)]
    assert out[-1] == Gesture.UNKNOWN                      # flickering: no gesture is shown
    for i in range(12):
        last = f.update(Gesture.PEACE, t + 0.2 + (i + 1) / 60)
    assert last == Gesture.PEACE


def test_gesture_filter_same_timing_at_30_and_120_fps():
    for fps in (30, 120):
        f = GestureFilter(window_s=0.2)
        t = 0.0
        for _ in range(int(fps * 0.5)):
            t += 1 / fps
            f.update(Gesture.FIST, t)
        frames = 0
        while f.current != Gesture.PEACE:
            t += 1 / fps
            frames += 1
            f.update(Gesture.PEACE, t)
        assert 0.08 <= frames / fps <= 0.2                 # switches within the 0.2 s window


def test_debouncer_requires_hold_and_fires_once():
    d = GestureDebouncer(hold=3)
    seq = [Gesture.FIST, Gesture.FIST, Gesture.POINT, Gesture.FIST, Gesture.FIST, Gesture.FIST, Gesture.FIST]
    fired = [d.update(g) for g in seq]
    assert fired == [None, None, None, None, None, Gesture.FIST, None]

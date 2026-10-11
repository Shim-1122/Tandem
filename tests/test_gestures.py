from tests.conftest import make_hand
from vtrack import Gesture, GestureDebouncer, classify_hand, finger_states

ALL4 = ("index", "middle", "ring", "pinky")


def test_open_palm():
    assert classify_hand(make_hand(extended=ALL4, thumb=True)) == Gesture.OPEN_PALM


def test_fist():
    assert classify_hand(make_hand()) == Gesture.FIST


def test_point():
    assert classify_hand(make_hand(extended=("index",))) == Gesture.POINT


def test_peace():
    assert classify_hand(make_hand(extended=("index", "middle"))) == Gesture.PEACE


def test_thumbs_up():
    assert classify_hand(make_hand(thumb=True)) == Gesture.THUMBS_UP


def test_unknown_for_odd_combination():
    assert classify_hand(make_hand(extended=("ring",))) == Gesture.UNKNOWN


def test_finger_states_thumb_folded_vs_out():
    assert finger_states(make_hand(thumb=True))["thumb"] is True
    assert finger_states(make_hand(thumb=False))["thumb"] is False


def test_debouncer_requires_hold_and_fires_once():
    d = GestureDebouncer(hold=3)
    seq = [Gesture.FIST, Gesture.FIST, Gesture.POINT, Gesture.FIST, Gesture.FIST, Gesture.FIST, Gesture.FIST]
    fired = [d.update(g) for g in seq]
    assert fired == [None, None, None, None, None, Gesture.FIST, None]
    assert d.current == Gesture.FIST

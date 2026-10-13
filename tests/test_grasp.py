"""Grasp tracking: hand shape, object tracking with a class vote, contact, and the holding state machine."""
import numpy as np
import pytest

from tests.test_fingers import GEOMETRY
from tests.test_gestures import CURL, OUT, THUMB_OUT, touching
from vtrack.fingers import HandKinematics, hand_fk
from vtrack.grasp import (GraspTracker, ObjectTracker, contact, hand_points, hand_roi, hand_shape)
from vtrack.grip import GripModel
from vtrack.objects import Detection

NAMES = ["tennis ball", "pen", "pencil", "water bottle", "hand"]
REPORT = [True, True, True, True, False]
WRAP = dict(mcp_flex=45.0, mcp_abd=0.0, pip=55.0, dip=25.0)


def hand(fingers=WRAP, thumb=None):
    a = {f: dict(fingers) for f in ("index", "middle", "ring", "pinky")}
    a["thumb"] = dict(thumb or dict(cmc_palmar=40.0, cmc_radial=30.0, cmc_rotation=60.0, mcp=30.0, ip=20.0))
    return hand_fk(a, GEOMETRY)


def shape_of(P):
    return hand_shape(P, HandKinematics("right").solve(P))


def px_of(P, at=(640.0, 360.0), scale=2500.0):
    """Pixel landmarks: the palm-frame hand seen straight on, palm center at `at`."""
    q = P[:, :2] * scale
    return q - q[[0, 5, 9, 13, 17]].mean(axis=0) + np.asarray(at)


def det(name, box, score=0.8, attrs=None):
    s = np.zeros(len(NAMES))
    s[NAMES.index(name)] = score
    return Detection(NAMES.index(name), name, score, np.asarray(box, float), s, attrs or {})


def test_hand_shapes():
    assert shape_of(hand(OUT, THUMB_OUT)) == "open"
    assert shape_of(hand(WRAP)) == "wrap"
    assert shape_of(hand(CURL, dict(cmc_palmar=20.0, cmc_radial=10.0, cmc_rotation=0.0, mcp=80.0, ip=60.0))) == "fist"
    assert shape_of(touching(False)) in ("pinch", "tripod") and shape_of(touching(True)) == "pinch"


def test_tracker_votes_on_the_label_and_follows_the_box():
    tr = ObjectTracker(NAMES, REPORT)
    tr.update([det("pen", [100, 100, 140, 300], 0.5), det("hand", [0, 0, 50, 50], 0.9)], 0.0)
    assert [t.name for t in tr.tracks] == ["pen"]                       # helpers are not tracked
    for k in range(1, 4):
        tr.update([det("pencil", [104, 100, 144, 300], 0.6)], 0.1 * k)
    t = tr.tracks[0]
    assert t.name == "pencil" and t.hits == 4 and t.box[0] == pytest.approx(104, abs=1.0)
    assert tr.confirmed() == [t]


def test_tracker_ages_only_what_it_searched():
    tr = ObjectTracker(NAMES, REPORT, max_age_s=0.5)
    tr.update([det("tennis ball", [100, 100, 160, 160]), det("pen", [600, 100, 640, 300])], 0.0)
    before = {t.name: t.score for t in tr.tracks}
    tr.update([], 0.2, region=(0, 0, 300, 300))                          # zoomed on the ball only
    after = {t.name: t.score for t in tr.tracks}
    assert after["tennis ball"] < before["tennis ball"] and after["pen"] == before["pen"]
    tr.prune(1.0)
    assert tr.tracks == []


def test_bottle_material_vote():
    tr = ObjectTracker(NAMES, REPORT)
    tr.update([det("water bottle", [0, 0, 60, 200], attrs={"metal": 0.9, "plastic": 0.02})], 0.0)
    b = tr.tracks[0]
    assert b.material() == "metal" and b.label == "water bottle (metal)"
    b.attributes = {"metal": 0.5, "plastic": 0.3}
    assert b.material() == "plastic"
    b.attributes = {"metal": 0.5, "plastic": 0.05}
    assert b.material() is None and b.label == "water bottle (material unsure)"
    g = GraspTracker(GripModel())
    assert g.plan_for(b).name == "water bottle" and g.plan_for(b).limited
    b.attributes = {"metal": 0.9, "plastic": 0.01}
    assert g.plan_for(b).name == "metal water bottle"


def test_contact_needs_the_thumb_and_another_finger_on_the_object():
    px = px_of(hand())
    pts = px[[4, 8, 12]]
    box = [pts[:, 0].min() - 5, pts[:, 1].min() - 5, pts[:, 0].max() + 5, pts[:, 1].max() + 5]
    assert contact(px, box) >= 0.9
    far = [b + 900 for b in box]
    assert contact(px, far) == 0.0
    thumb_only = [px[4, 0] - 3, px[4, 1] - 3, px[4, 0] + 3, px[4, 1] + 3]
    assert contact(px, thumb_only, margin_ratio=0.0) < 0.7
    roi = hand_roi(px, 1280, 720)
    assert roi[0] < px[:, 0].min() and roi[2] > px[:, 0].max() and hand_roi(px * 0 - 500, 1280, 720) is None


def _scene():
    P = hand()
    px = px_of(P)
    pts = px[[4, 8, 12, 6, 10]]
    box = [pts[:, 0].min() - 10, pts[:, 1].min() - 10, pts[:, 0].max() + 10, pts[:, 1].max() + 10]
    tr = ObjectTracker(NAMES, REPORT)
    tr.update([det("tennis ball", box)], 0.0)
    tr.update([det("tennis ball", box)], 0.05)
    return P, px, tr


def test_grasp_starts_after_touching_and_carries_the_object():
    P, px, tr = _scene()
    g = GraspTracker(GripModel(), enter_s=0.25)
    st = g.update("right", px, shape_of(P), tr, 0.1)
    assert st.state == "touching" and st.object_name == "tennis ball"
    st = g.update("right", px, shape_of(P), tr, 0.4)
    assert st.state == "holding" and st.plan.name == "tennis ball" and st.plan.force_n == pytest.approx(1.0)
    assert "holding tennis ball" in st.text() and "spherical grip 1.0 N" in st.text()
    ball = tr.get(st.object_id)
    c0 = ball.center.copy()
    for k in range(1, 11):                          # the hand moves 200 px; the detector does not see the ball
        st = g.update("right", px + [20.0 * k, 0], shape_of(P), tr, 0.4 + 0.05 * k)
    assert st.state == "holding" and ball.center - c0 == pytest.approx([200.0, 0.0])
    assert st.carried_px == pytest.approx(200.0) and ball.held_by == "right"
    tr.prune(0.9 + 2.0)                              # carried objects outlive unseen ones
    assert tr.get(st.object_id) is not None


def test_opening_the_hand_releases_it():
    P, px, tr = _scene()
    g = GraspTracker(GripModel(), enter_s=0.2, exit_s=0.3)
    g.update("right", px, "wrap", tr, 0.1)
    assert g.update("right", px, "wrap", tr, 0.35).state == "holding"
    assert g.update("right", px, "open", tr, 0.4).state == "holding"          # not yet
    st = g.update("right", px, "open", tr, 0.75)
    assert st.state == "free" and tr.tracks[0].held_by is None
    for t in (0.8, 1.2, 1.6):                                                  # an open hand touches, never holds
        assert g.update("right", px, "open", tr, t).state == "touching"


def test_losing_the_hand_briefly_keeps_the_grasp():
    P, px, tr = _scene()
    g = GraspTracker(GripModel(), enter_s=0.2, lost_s=0.6)
    g.update("right", px, "wrap", tr, 0.1)
    g.update("right", px, "wrap", tr, 0.35)
    assert g.update("right", None, "", tr, 0.6).state == "holding"
    assert g.update("right", None, "", tr, 1.2).state == "free"


def test_hidden_object_is_held_while_the_hand_stays_closed():
    P, px, tr = _scene()
    g = GraspTracker(GripModel(), enter_s=0.2, exit_s=0.3)
    g.update("right", px, "wrap", tr, 0.1)
    g.update("right", px, "wrap", tr, 0.35)
    tr.tracks.clear()                                                         # the tracker lost it
    st = g.update("right", px, "wrap", tr, 0.5)
    assert st.state == "holding" and st.hidden and "(hidden)" in st.text()
    g.update("right", px, "open", tr, 0.6)
    assert g.update("right", px, "open", tr, 1.0).state == "free"


def test_one_object_per_hand_and_not_stolen():
    P, px, tr = _scene()
    g = GraspTracker(GripModel(), enter_s=0.1)
    g.update("right", px, "wrap", tr, 0.1)
    assert g.update("right", px, "wrap", tr, 0.3).state == "holding"
    st = g.update("left", px + [3, 0], "wrap", tr, 0.3)                       # the other hand at the same place
    assert st.state == "free"


def test_hand_points_scale_normalized_landmarks():
    L = np.array([[0.5, 0.25, 0.0], [1.0, 1.0, 0.1]])
    assert hand_points(L, 1280, 720) == pytest.approx(np.array([[640, 180], [1280, 720]]))


def test_ages_stretch_when_the_detector_runs_slowly():
    tr = ObjectTracker(NAMES, REPORT, max_age_s=0.8)
    for k in range(30):                                   # one detector result every 1.2 s
        tr.update([det("pen", [600, 100, 640, 300])], 1.2 * k)
    assert tr.interval == pytest.approx(1.2, abs=0.01)
    t = 1.2 * 29
    tr.prune(t + 2.5)                                     # 2.5 s unseen: kept (2.5 intervals is 3 s)
    assert len(tr.tracks) == 1
    tr.prune(t + 3.5)
    assert tr.tracks == []

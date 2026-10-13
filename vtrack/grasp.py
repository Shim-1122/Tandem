"""Grasp tracking: which object each hand is holding, the hand's shape, and the grip to use for it.

Objects come from the open-vocabulary detector (objects.py) and are tracked over time (ObjectTracker): each track
keeps a box and a running score for every class, so its label is a vote over many frames rather than one guess.
A held object is carried along with the hand between detections, because the hand hides much of it.

Each hand (GraspTracker) then looks for an object it touches in the image: the thumb tip and at least one other
fingertip or finger joint inside the object's box (slightly enlarged), or the palm on it for objects larger than the
hand. Touching for `enter_s` seconds starts a grasp; the grasp ends when the hand opens, the object is seen away
from the hand, or the hand is lost for `exit_s` seconds. The grip to use comes from the object's entry in grip.py.

The camera gives no depth at the contact, so a hand passing in front of an object can look like a touch for a
moment; the time thresholds and the hand shape (an open hand is not holding anything) filter most of that out.
All image coordinates here are camera pixels (not mirrored).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grip import OBJECTS, GripModel, GripPlan

TIPS = (4, 8, 12, 16, 20)
PIPS = (6, 10, 14, 18)
PALM = (0, 5, 9, 13, 17)
CURL_FINGERS = ("index", "middle", "ring", "pinky")
# Water bottle material from the detector's "metal" and "plastic" phrasings: log(metal / plastic) at least METAL_LOG
# reads metal, at most PLASTIC_LOG plastic, otherwise unsure. On labeled photos metal (insulated) bottles scored 2.2
# to 4.5 and ordinary water bottles mostly 0 to 1.7 (YOLO-World leans toward "metal" for every bottle).
METAL_LOG, PLASTIC_LOG = 2.5, 1.0


# ------------------------------------------------------------------------------------------------ hand shape
def curl(fingers: dict) -> float:
    """Mean knuckle plus middle-joint flexion of the four fingers, degrees (0 flat, about 180 a tight fist)."""
    return float(np.mean([fingers[f]["mcp_flex"] + fingers[f]["pip"] for f in CURL_FINGERS]))


def hand_shape(world: np.ndarray, fingers: dict) -> str:
    """open, pinch, tripod, wrap or fist, from world landmarks (m) and finger angles (fingers.py)."""
    w = np.asarray(world, float)
    scale = max(float(np.linalg.norm(w[9] - w[0])), 1e-6)
    d = lambda a, b: float(np.linalg.norm(w[a] - w[b])) / scale    # noqa: E731
    c = curl(fingers)
    if d(4, 8) < 0.25 and d(4, 12) < 0.35 and d(8, 12) < 0.35 and fingers["ring"]["pip"] > 30:
        return "tripod"
    if d(4, 8) < 0.25:
        return "pinch"
    if c > 150 and d(12, 0) < 0.9:
        return "fist"
    if c > 45:
        return "wrap"
    return "open"


# ------------------------------------------------------------------------------------------------ objects
@dataclass
class TrackedObject:
    id: int
    box: np.ndarray                      # x0, y0, x1, y1 camera pixels
    evidence: np.ndarray                 # running score per class
    names: list
    t_seen: float
    hits: int = 1
    held_by: str | None = None
    t_moved: float = -1.0                # last time it was carried along with a hand
    attributes: dict = field(default_factory=dict)      # running attribute scores (the bottle's material)

    @property
    def cls(self) -> int:
        return int(np.argmax(self.evidence))

    @property
    def name(self) -> str:
        return self.names[self.cls]

    @property
    def score(self) -> float:
        return float(self.evidence[self.cls])

    @property
    def center(self) -> np.ndarray:
        return (self.box[:2] + self.box[2:]) / 2

    def material(self) -> str | None:
        """metal, plastic, or None (unsure, or no material phrasings)."""
        m, p = self.attributes.get("metal"), self.attributes.get("plastic")
        if m is None or p is None:
            return None
        r = np.log(max(m, 1e-6) / max(p, 1e-6))
        return "metal" if r >= METAL_LOG else "plastic" if r <= PLASTIC_LOG else None

    @property
    def label(self) -> str:
        if self.name == "water bottle":
            return f"water bottle ({self.material() or 'material unsure'})"
        return self.name


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(box, region) -> bool:
    c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return region[0] <= c[0] <= region[2] and region[1] <= c[1] <= region[3]


class ObjectTracker:
    """Tracks detected objects across detector runs (whole frame or zoomed around the hands)."""

    def __init__(self, names: list, report=None, match_iou: float = 0.25, new_score: float = 0.25,
                 rate: float = 0.4, max_age_s: float = 0.8, held_age_s: float = 2.5, box_rate: float = 0.6):
        self.names = list(names)
        self.report = np.ones(len(names), bool) if report is None else np.asarray(report, bool)
        self.match_iou, self.new_score, self.rate = match_iou, new_score, rate
        self.max_age_s, self.held_age_s, self.box_rate = max_age_s, held_age_s, box_rate
        self.tracks: list[TrackedObject] = []
        self._next = 1
        self.interval, self._t_last = 0.3, None      # seconds between detector runs (ages scale with it)

    def update(self, detections, t: float, region=None):
        """detections: objects.Detection list from one run; region: the zoomed area (None for the whole frame).

        Tracks inside the searched area that were not found lose evidence; tracks elsewhere are left alone.
        """
        if self._t_last is not None and t > self._t_last + 1e-6:
            self.interval = 0.8 * self.interval + 0.2 * min(5.0, t - self._t_last)
        self._t_last = t if self._t_last is None else max(t, self._t_last)
        dets = [d for d in detections if self.report[d.cls]]
        pairs = sorted(((box_iou(tr.box, d.box), i, j) for i, tr in enumerate(self.tracks)
                        for j, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for o, i, j in pairs:
            if o < self.match_iou or i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            tr, d = self.tracks[i], dets[j]
            tr.box = (1 - self.box_rate) * tr.box + self.box_rate * np.asarray(d.box, float)
            tr.evidence = (1 - self.rate) * tr.evidence + self.rate * _class_scores(d, len(self.names))
            for a, v in getattr(d, "attributes", {}).items():
                tr.attributes[a] = (1 - self.rate) * tr.attributes.get(a, v) + self.rate * v
            tr.t_seen, tr.hits = t, tr.hits + 1
        for i, tr in enumerate(self.tracks):
            if i not in used_t and (region is None or inside(tr.box, region)):
                tr.evidence = (1 - self.rate) * tr.evidence
        for j, d in enumerate(dets):
            if j not in used_d and d.score >= self.new_score:
                self.tracks.append(TrackedObject(self._next, np.asarray(d.box, float).copy(),
                                                 _class_scores(d, len(self.names)), self.names, t,
                                                 attributes=dict(getattr(d, "attributes", {}))))
                self._next += 1
        self.prune(t)

    def prune(self, t: float):
        """Forget objects not seen for a while: longer when held, and longer when the detector runs slowly."""
        def keep(tr):
            age = t - max(tr.t_seen, tr.t_moved if tr.held_by else -1e9)
            limit = max(self.held_age_s, 4 * self.interval) if tr.held_by else max(self.max_age_s, 2.5 * self.interval)
            return age <= limit and tr.score >= 0.05
        self.tracks = [tr for tr in self.tracks if keep(tr)]

    def carry(self, track_id: int, delta, t: float):
        """Move a held object with the hand (the detector often cannot see it inside the hand)."""
        tr = self.get(track_id)
        if tr is not None:
            tr.box = tr.box + np.tile(np.asarray(delta, float), 2)
            tr.t_moved = t

    def get(self, track_id):
        return next((tr for tr in self.tracks if tr.id == track_id), None)

    def confirmed(self, min_score: float = 0.2) -> list:
        return [tr for tr in self.tracks if tr.score >= min_score and (tr.hits >= 2 or tr.score >= 0.4)]


def _class_scores(d, k: int) -> np.ndarray:
    s = getattr(d, "class_scores", None)
    if s is not None:
        return np.asarray(s, float)
    v = np.zeros(k)
    v[d.cls] = d.score
    return v


# ------------------------------------------------------------------------------------------------ grasps
def hand_points(image_landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """MediaPipe normalized hand landmarks to camera pixels (21, 2)."""
    return np.asarray(image_landmarks, float)[:, :2] * np.array([width, height])


def contact(px: np.ndarray, box, margin_ratio: float = 0.25) -> float:
    """0..1: how much the hand at pixel landmarks `px` touches the object box (thumb, other fingers, palm)."""
    palm_len = max(float(np.linalg.norm(px[9] - px[0])), 1.0)
    m = margin_ratio * palm_len
    x0, y0, x1, y1 = box[0] - m, box[1] - m, box[2] + m, box[3] + m
    inb = lambda p: x0 <= p[0] <= x1 and y0 <= p[1] <= y1     # noqa: E731
    thumb = 1.0 if inb(px[4]) else 0.0
    others = sum(inb(px[i]) for i in TIPS[1:] + PIPS)
    palm_c = px[list(PALM)].mean(axis=0)
    big = min(box[2] - box[0], box[3] - box[1]) > 1.2 * palm_len
    palm = 1.0 if (big and inb(palm_c)) else 0.0
    return min(1.0, 0.45 * thumb + 0.45 * min(1.0, others / 2.0) + 0.1 * (1.0 if inb(palm_c) else 0.0) + 0.45 * palm)


@dataclass
class GraspState:
    side: str
    state: str = "free"                 # free, touching, holding
    object_id: int | None = None
    object_name: str = ""
    contact: float = 0.0
    shape: str = ""                     # the hand's shape (open, pinch, tripod, wrap, fist)
    plan: GripPlan | None = None
    since: float = 0.0
    carried_px: float = 0.0             # how far the hand has moved the object, pixels
    hidden: bool = False                # holding, but the object has not been seen for a while

    def text(self) -> str:
        if self.state == "free":
            return f"{self.side}: free" + (f" ({self.shape})" if self.shape else "")
        what = self.object_name + (" (hidden)" if self.hidden else "")
        if self.state == "touching":
            return f"{self.side}: touching {what}"
        s = f"{self.side}: holding {what}"
        if self.plan is not None:
            s += f" | {self.plan.grasp} grip {self.plan.force_n:.1f} N ({self.plan.command:.2f})"
            if self.plan.limited:
                s += " !"
        return s


@dataclass
class GraspTracker:
    model: GripModel = field(default_factory=GripModel)
    enter_s: float = 0.25               # touching this long: holding
    exit_s: float = 0.4                 # open hand, or the object seen away from the hand, this long: released
    lost_s: float = 0.6                 # hand out of view this long: released
    states: dict = field(default_factory=dict)
    _t_touch: dict = field(default_factory=dict)
    _t_off: dict = field(default_factory=dict)
    _t_hand: dict = field(default_factory=dict)
    _palm: dict = field(default_factory=dict)         # side: (palm center px, time)

    def plan_for(self, tr: TrackedObject) -> GripPlan | None:
        if tr.name == "water bottle":
            m = tr.material()
            return self.model.plan_for(f"{m} water bottle") if m else self.model.uncertain_bottle()
        return self.model.plan_for(tr.name) if tr.name in OBJECTS else None

    def update(self, side: str, px, shape: str, tracker: ObjectTracker, t: float) -> GraspState:
        """One hand's update. px: (21, 2) camera pixels or None when the hand is not in view."""
        st = self.states.setdefault(side, GraspState(side))
        if px is None:
            if st.state == "holding" and t - self._t_hand.get(side, t) <= self.lost_s:
                return st                                        # briefly lost: keep holding
            self._reset(side, st, tracker)
            st.shape = ""
            self._palm.pop(side, None)
            return st
        self._t_hand[side] = t
        st.shape = shape
        palm = np.asarray(px, float)[list(PALM)].mean(axis=0)
        prev = self._palm.get(side)
        self._palm[side] = (palm, t)

        if st.state == "holding":
            tr = tracker.get(st.object_id)
            if tr is None:                                       # tracker dropped it: keep while the hand is closed
                st.hidden = True
                closed = shape in ("wrap", "fist", "pinch", "tripod")
                if closed:
                    self._t_off.pop(side, None)
                elif self._off(side, t):
                    self._reset(side, st, tracker)
                return st
            if prev is not None:
                if tr.t_seen <= prev[1]:                          # no newer detection: carry it with the hand
                    tracker.carry(tr.id, palm - prev[0], t)
                st.carried_px += float(np.linalg.norm(palm - prev[0]))
            tr.held_by = side
            st.contact = contact(px, tr.box)
            st.hidden = t - tr.t_seen > max(1.0, 3 * tracker.interval)
            st.object_name = tr.label
            st.plan = self.plan_for(tr)
            released = shape == "open" or st.contact < 0.3
            if released and self._off(side, t):
                self._reset(side, st, tracker)
            elif not released:
                self._t_off.pop(side, None)
            return st

        best, best_c = None, 0.0
        for tr in tracker.confirmed():
            if tr.held_by not in (None, side):
                continue
            c = contact(px, tr.box)
            if c > best_c:
                best, best_c = tr, c
        st.contact = best_c
        if best is None or best_c < 0.7 or shape == "open":
            self._reset(side, st, tracker, keep_shape=True)
            if best is not None and best_c >= 0.45:
                st.state, st.object_id, st.object_name = "touching", best.id, best.label
            return st
        if st.object_id != best.id or side not in self._t_touch:
            self._t_touch[side] = t
        st.object_id, st.object_name = best.id, best.label
        if t - self._t_touch[side] >= self.enter_s:
            st.state, st.since, st.carried_px = "holding", t, 0.0
            st.plan = self.plan_for(best)
            best.held_by = side
            self._t_off.pop(side, None)
        else:
            st.state = "touching"
        return st

    def _off(self, side, t) -> bool:
        t0 = self._t_off.setdefault(side, t)
        return t - t0 >= self.exit_s

    def _reset(self, side, st: GraspState, tracker: ObjectTracker, keep_shape=False):
        if st.object_id is not None:
            tr = tracker.get(st.object_id)
            if tr is not None and tr.held_by == side:
                tr.held_by = None
        shape = st.shape
        st.__init__(side)
        if keep_shape:
            st.shape = shape
        self._t_touch.pop(side, None)
        self._t_off.pop(side, None)


def hand_roi(px: np.ndarray, width: int, height: int, scale: float = 2.6, min_side: float = 160.0):
    """Square region around a hand (camera pixels) for the zoomed object search: the hand and what it holds."""
    px = np.asarray(px, float)
    c = px[list(PALM)].mean(axis=0)
    size = max(float(np.ptp(px[:, 0])), float(np.ptp(px[:, 1])), 2.0 * float(np.linalg.norm(px[9] - px[0])))
    half = max(min_side, scale * size) / 2
    x0, y0 = max(0.0, c[0] - half), max(0.0, c[1] - half)
    x1, y1 = min(float(width), c[0] + half), min(float(height), c[1] + half)
    return (x0, y0, x1, y1) if x1 - x0 > 16 and y1 - y0 > 16 else None

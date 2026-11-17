"""Turn Tandem's tracker values into events Jarvis can react to, and a short description of the scene.

Input is the dict the live app reports every update (`App.values()`, the same columns as a CSV recording).
Everything is debounced in time so a single noisy frame never triggers anything:
    person_arrived / person_left   a face is seen for 0.5 s / not seen for 2 s (name from face recognition)
    gesture                        a hand holds one of Tandem's 16 gestures for 0.3 s (once per gesture)
    emotion                        one emotion (not neutral) above 50% for 1 s, reported when it changes
    grasp / release                a hand starts or stops holding an object
    target_locked                  the person locks a gaze target (pinch or `t` in Tandem)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

EMOTIONS = ("angry", "disgust", "fearful", "happy", "sad", "surprised")
SIDES = {"L_": "left", "R_": "right", "": "right"}


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


@dataclass
class Event:
    kind: str
    t: float
    data: dict = field(default_factory=dict)

    def __str__(self):
        extra = " ".join(f"{k}={v}" for k, v in self.data.items())
        return f"{self.kind} {extra}".strip()


class _Held:
    """A value that has to stay the same for `hold_s` before it counts."""

    def __init__(self, hold_s: float):
        self.hold_s, self.candidate, self.since, self.value = hold_s, None, None, None

    def update(self, v, t: float) -> bool:
        """True when the confirmed value changes."""
        if v != self.candidate:
            self.candidate, self.since = v, t
        if self.candidate != self.value and t - self.since >= self.hold_s:
            self.value = self.candidate
            return True
        return False


class Perception:
    """Feed it `update(values, t)` every tracker update; it returns the new events."""

    def __init__(self, arrive_s=0.5, leave_s=2.0, gesture_s=0.3, emotion_s=1.0, emotion_min=0.5):
        self.arrive_s, self.leave_s = arrive_s, leave_s
        self.emotion_min = emotion_min
        self._gesture_s, self._emotion_s = gesture_s, emotion_s
        self.present = False
        self.name = None
        self._seen_since = self._last_seen = None
        self._gestures: dict[str, _Held] = {}
        self._grasps: dict[str, str] = {}
        self._emotion = _Held(emotion_s)
        self._locked = False
        self.values: dict = {}
        self.t = 0.0

    # ------------------------------------------------------------------ state
    def head_direction(self):
        """(yaw, pitch) in degrees from the camera to the person's head, in the robot's conventions
        (yaw + to the robot's left, pitch + up), or None. Assumes the robot's eyes sit at the camera."""
        x, y, z = (num(self.values.get(k)) for k in ("head_x_cm", "head_y_cm", "head_z_cm"))
        if not all(map(math.isfinite, (x, y, z))) or z <= 0:
            return None
        return -math.degrees(math.atan2(x, z)), -math.degrees(math.atan2(y, math.hypot(x, z)))

    def target_direction(self):
        """(yaw, pitch) toward the locked gaze target (a ray from the camera), or None."""
        if num(self.values.get("target_locked")) != 1:
            return None
        x, y, z = (num(self.values.get(k)) for k in ("target_ray_x", "target_ray_y", "target_ray_z"))
        if not all(map(math.isfinite, (x, y, z))) or z <= 0:
            return None
        return -math.degrees(math.atan2(x, z)), -math.degrees(math.atan2(y, math.hypot(x, z)))

    def emotion(self):
        return self._emotion.value

    def holding(self) -> dict:
        return dict(self._grasps)

    def gestures(self) -> dict:
        return {sd: h.value for sd, h in self._gestures.items() if h.value}

    def describe(self) -> str:
        """One or two sentences about the scene, for the language model."""
        if not self.present:
            return "Nobody is in front of the camera."
        who = self.name if self.name and self.name != "unknown" else "someone you do not recognize"
        parts = [f"You can see {who}."]
        if self._emotion.value:
            parts.append(f"Their expression looks {self._emotion.value}.")
        g = self.gestures()
        if g:
            parts.append("Gestures: " + ", ".join(f"{sd} hand {v.replace('_', ' ')}" for sd, v in g.items()) + ".")
        if self._grasps:
            parts.append("Holding: " + ", ".join(f"a {o} in the {sd} hand" for sd, o in self._grasps.items()) + ".")
        if self._locked:
            parts.append("They have locked a gaze target.")
        return " ".join(parts)

    # ------------------------------------------------------------------ update
    def update(self, values: dict, t: float) -> list[Event]:
        self.values, self.t = values, t
        ev: list[Event] = []
        self._update_presence(values, t, ev)
        if self.present:
            self._update_gestures(values, t, ev)
            self._update_emotion(values, t, ev)
        self._update_grasps(values, t, ev)
        self._update_target(values, t, ev)
        return ev

    def _update_presence(self, v, t, ev):
        seen = math.isfinite(num(v.get("head_z_cm")))
        if seen:
            self._last_seen = t
            self._seen_since = t if self._seen_since is None else self._seen_since
            ident = v.get("identity") or ""
            if ident and ident != self.name and self.present:
                self.name = ident
                ev.append(Event("person_recognized", t, {"name": ident}))
            if not self.present and t - self._seen_since >= self.arrive_s:
                self.present, self.name = True, ident or "unknown"
                ev.append(Event("person_arrived", t, {"name": self.name}))
        else:
            self._seen_since = None
            if self.present and self._last_seen is not None and t - self._last_seen >= self.leave_s:
                self.present = False
                ev.append(Event("person_left", t, {"name": self.name}))
                self.name = None
                self._emotion = _Held(self._emotion_s)
                for h in self._gestures.values():
                    h.value = h.candidate = None

    def _update_gestures(self, v, t, ev):
        for pre, sd in SIDES.items():
            col = f"{pre}gesture"
            if col not in v:
                continue
            g = v.get(col) or "unknown"
            g = None if g in ("unknown", "", "nan") else g
            held = self._gestures.setdefault(sd, _Held(self._gesture_s))
            if held.update(g, t) and held.value:
                ev.append(Event("gesture", t, {"side": sd, "gesture": held.value}))

    def _update_emotion(self, v, t, ev):
        scores = {e: num(v.get(f"emotion_{e}_pct")) / 100 for e in EMOTIONS}
        scores = {e: s for e, s in scores.items() if math.isfinite(s)}
        if not scores:
            return
        best = max(scores, key=scores.get)
        cur = best if scores[best] >= self.emotion_min else None
        if self._emotion.update(cur, t) and self._emotion.value:
            ev.append(Event("emotion", t, {"emotion": self._emotion.value, "pct": round(100 * scores[best])}))

    def _update_grasps(self, v, t, ev):
        for pre, sd in SIDES.items():
            col = f"{pre}grasp_state"
            if col not in v:
                continue
            obj = v.get(f"{pre}grasp_object") or ""
            holding = v.get(col) == "holding" and obj not in ("", "nan")
            before = self._grasps.get(sd)
            if holding and before != obj:
                self._grasps[sd] = obj
                ev.append(Event("grasp", t, {"side": sd, "object": obj}))
            elif not holding and before:
                del self._grasps[sd]
                ev.append(Event("release", t, {"side": sd, "object": before}))

    def _update_target(self, v, t, ev):
        locked = num(v.get("target_locked")) == 1
        if locked and not self._locked:
            d = self.target_direction()
            ev.append(Event("target_locked", t, {"yaw": round(d[0], 1), "pitch": round(d[1], 1)} if d else {}))
        elif self._locked and not locked:
            ev.append(Event("target_released", t))
        self._locked = locked

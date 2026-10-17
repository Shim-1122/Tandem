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


class Visitor:
    """The visitor above, with the same `step(t, jarvis)` interface as GameNight."""
    LENGTH = LENGTH

    def __init__(self):
        self._i = 0

    def step(self, t: float, jarvis):
        say = None
        if self._i < len(SAYS) and SAYS[self._i][0] <= t:
            say = SAYS[self._i][1]
            self._i += 1
        return scenario(t), say


# ---------------------------------------------------------------------- game night
class GameNight:
    """A scripted player for trying voice commands in simulation: "rock paper scissors?", three rounds (throwing
    paper, scissors, rock, with a fist pump during each count, like a person), then "shadow me" while waving the
    right arm (the robot's left arm should follow, like a mirror), then "stop mirroring me".

    `step(t, jarvis)` returns (tracker values, words said now or None). The player watches the game's state to
    time its throws the way a person listens for "shoot"; it never looks at the robot's throw."""

    THROWS = ("paper", "scissors", "rock")
    GESTURE = {"rock": "fist", "paper": "open_palm", "scissors": "peace"}
    LENGTH = 44.0

    def __init__(self):
        self.round = 0
        self.shoot_t = None
        self.result_t = None
        self.phase = "start"
        self.shadow_t = None
        self.said = set()

    def _say_once(self, key, text):
        if key in self.said:
            return None
        self.said.add(key)
        return text

    def step(self, t: float, jarvis):
        v = {**scenario(-1.0), **_face(t, "shyam")}
        v["head_x_cm"] = 0.0                                   # sits still in front of the robot
        game = jarvis.game
        say = None
        if t >= 2.0 and self.phase == "start":
            say, self.phase = "Hey Jarvis, rock paper scissors?", "ready"
        elif self.phase == "ready" and game is not None:
            if game.state == "ready" and t >= 4.0:
                v["R_gesture"] = "fist"                        # show a fist: ready
            elif game.state == "count":
                v["R_gesture"] = "fist"                        # pumping along with the count
            elif game.state == "read":
                if self.shoot_t is None:
                    self.shoot_t = t
                throw = self.THROWS[self.round % len(self.THROWS)]
                v["R_gesture"] = "fist" if t - self.shoot_t < 0.25 else self.GESTURE[throw]
            elif game.state == "again":
                if self.result_t is None:
                    self.result_t, self.shoot_t = t, None
                    self.round += 1
                v["R_gesture"] = "unknown"
                if t - self.result_t >= 3.5:              # watch the robot react, then answer
                    self.result_t = None
                    say = "again" if self.round < len(self.THROWS) else "no, that's enough"
                    if self.round >= len(self.THROWS):
                        self.phase = "after_game"
                        self.shadow_t = t + 4.0
        elif self.phase == "after_game" and self.shadow_t is not None and t >= self.shadow_t:
            say, self.phase = "Hey Jarvis, shadow me", "shadow"
        if self.phase == "shadow":
            u = t - self.shadow_t
            w = math.sin(2 * math.pi * 0.5 * u)
            v.update({"R_shoulder_flexion": 20.0, "R_shoulder_abduction": 60 + 30 * w, "R_shoulder_rotation": -30.0,
                      "R_elbow_flexion": 70 + 30 * w, "R_forearm_rotation": 0.0, "R_wrist_flexion": 0.0,
                      "R_wrist_deviation": 0.0, "L_shoulder_flexion": 0.0, "L_shoulder_abduction": 5.0,
                      "L_shoulder_rotation": 0.0, "L_elbow_flexion": 15.0, "neck_yaw": 20 * w, "neck_pitch": 0.0,
                      "neck_roll": 0.0})
            if u >= 8.0:
                say = self._say_once("unshadow", "Hey Jarvis, stop mirroring me")
        return v, say


SCRIPTS = {"visitor": Visitor, "games": GameNight}

"""Rock paper scissors with the robot.

    "Hey Jarvis, rock paper scissors?"  -> Jarvis gets ready (forearm up, fist) and asks you to show a fist
    you show a fist, or say "ready"     -> "Rock", "paper", "scissors" with a fist pump on each, then "shoot!"
    on "shoot" both of you throw        -> Jarvis reads your hand through Tandem's gestures, says who won
    "again" / "one more"                -> next round straight away; "no" or "stop" ends the game

Your throw is read from Tandem's gestures: fist = rock, open palm (or four fingers) = paper, peace (V) =
scissors. Hands pump as fists during the count and people throw a little late, so your hand is read 0.6 s after
"shoot" (the latest throw shape seen by then), or up to 1.2 s after if no shape has shown yet.
Without a camera (or if the camera misses it) you can also just say or type "rock", "paper" or "scissors" then.

Fair play: the robot picks its throw before the count starts, from a random generator that never sees yours.
"""
from __future__ import annotations

import random
import re

from . import skills

BEAT_S = 0.6                                  # time between "rock", "paper", "scissors" and "shoot"
WINDOW = (-0.2, 1.2)                          # when your throw counts, seconds around "shoot"
DECIDE_S = 0.6                                # read your hand this long after "shoot" (it has changed by then)
READY_TIMEOUT_S = 12.0                        # start anyway after this long without a fist or "ready"
AGAIN_TIMEOUT_S = 20.0                        # end the game if nobody asks for another round
GESTURE_THROW = {"fist": "rock", "open_palm": "paper", "four": "paper", "peace": "scissors"}
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}     # each key beats its value
WORD_THROW = {"rock": "rock", "rocks": "rock", "paper": "paper", "scissors": "scissors", "scissor": "scissors"}


def winner(robot: str, person: str) -> str:
    """'robot', 'person' or 'tie'."""
    if robot == person:
        return "tie"
    return "robot" if BEATS[robot] == person else "person"


READY = re.compile(r"\b(ready|go|let'?s go|start|yes|yeah|ok|okay)\b")
AGAIN = re.compile(r"\b(again|one more|another|yes|yeah|sure|rematch|play|ok|okay)\b")
DONE = re.compile(r"\b(no|nope|done|quit|enough|that'?s it|i'?m good)\b")


def throw_from_words(text: str):
    words = [w.strip(".,!?") for w in (text or "").lower().split()]
    found = [WORD_THROW[w] for w in words if w in WORD_THROW]
    return found[-1] if len(found) == 1 or (found and len(set(found)) == 1) else None


class RockPaperScissors:
    """Driven by Jarvis.tick: `update(now)` moves through the game; `hear(text)` takes words while it runs."""

    def __init__(self, jarvis, now: float, rng: random.Random | None = None, side: str = "right"):
        self.j, self.side = jarvis, side
        self.rng = rng or random.Random()
        self.score = {"robot": 0, "person": 0, "tie": 0}
        self.state, self.t0 = "ready", now
        self.robot_throw = self.person_throw = None
        self.said_throw = None
        self.done = False
        self.result = None
        self._beat = 0
        self.j.body.do(skills.rps_ready(self.side), now)
        self.j.body.look("person")
        self.j.say("You're on! Show me a fist when you're ready.")

    # ------------------------------------------------------------------ input
    def hear(self, text: str, now: float) -> bool:
        """Words for the game. True if the game used them (so the brain does not also answer)."""
        t = (text or "").lower()
        if self.state == "ready" and READY.search(t):
            self._start_count(now)
            return True
        if self.state in ("count", "read"):
            throw = throw_from_words(t)
            if throw:
                self.said_throw = throw
                return True
        if self.state == "again":
            if DONE.search(t):
                self.finish(now)
                return True
            if AGAIN.search(t):
                self.j.say("Okay. Rock...")
                self._start_count(now, said_first=True)
                return True
        return False

    def _person_gesture(self):
        """Your current throw from Tandem, or None. Either hand counts (the one making a throw shape)."""
        v = self.j.perception.values
        for col in ("R_gesture", "L_gesture", "gesture"):
            throw = GESTURE_THROW.get(v.get(col) or "")
            if throw:
                return throw
        return None

    # ------------------------------------------------------------------ the game
    def _start_count(self, now: float, said_first: bool = False):
        self.robot_throw = self.rng.choice(sorted(BEATS))           # decided before seeing anything of yours
        self.person_throw = self.said_throw = None
        self.state, self.t0, self._beat = "count", now, 1 if said_first else 0
        self.j.body.do(skills.rps_count(3, BEAT_S, self.side), now)

    def update(self, now: float):
        if self.done:
            return
        u = now - self.t0
        if self.state == "ready":
            if self._person_gesture() == "rock":                    # a fist: they're set
                self._start_count(now)
            elif u > READY_TIMEOUT_S:
                self.j.say("Let's just go.")
                self._start_count(now)
        elif self.state == "count":
            words = ("Rock", "Paper", "Scissors")
            while self._beat < 3 and u >= self._beat * BEAT_S:
                self.j.say(words[self._beat] + "...")
                self._beat += 1
            if u >= 3 * BEAT_S:
                self.j.say("Shoot!")
                self.j.body.do(skills.rps_throw(self.robot_throw, self.side), now)
                self.state, self.t0 = "read", now
                self._latest = self._person_gesture()
        elif self.state == "read":
            # Hands pump as fists while counting, so a fist right at "shoot" proves nothing: keep the latest
            # throw shape and decide once the hand has had time to change (DECIDE_S), or later if none yet.
            self._latest = self._person_gesture() or self._latest
            if u >= DECIDE_S and (self.said_throw or self._latest):
                self.person_throw = self.said_throw or self._latest
                self._score(now)
            elif u >= WINDOW[1]:
                self.j.say("I didn't catch your throw. Say again to retry.")
                self.state, self.t0 = "again", now
        elif self.state == "again" and u > AGAIN_TIMEOUT_S:
            self.finish(now)

    def _score(self, now: float):
        r, p = self.robot_throw, self.person_throw
        w = winner(r, p)
        self.score[w] += 1
        self.result = (r, p, w)
        s = self.score
        tally = f"{s['person']} to {s['robot']}" + (f", {s['tie']} tied" if s["tie"] else "")
        if w == "robot":
            self.j.say(f"{r.capitalize()} beats {p}. I win! That's {tally}. Again?")
            self.j.body.do(skills.celebrate(2.0), now, queue=True)
        elif w == "person":
            self.j.say(f"{p.capitalize()} beats {r}. You got me. That's {tally}. Again?")
            self.j.body.do(skills.shrug(1.5), now, queue=True)
        else:
            self.j.say(f"We both threw {r}. Tie! That's {tally}. Again?")
            self.j.body.do(skills.nod(1), now, queue=True)
        self.state, self.t0 = "again", now

    def finish(self, now: float):
        s = self.score
        if s["person"] + s["robot"] + s["tie"]:
            verdict = ("You win the match" if s["person"] > s["robot"] else
                       "I win the match" if s["robot"] > s["person"] else "It's a draw")
            self.j.say(f"Good game. {verdict}, {s['person']} to {s['robot']}.")
        else:
            self.j.say("Maybe later.")
        self.j.body.rest()
        self.done = True

    def status(self) -> str:
        s = self.score
        return f"rock paper scissors ({self.state}), you {s['person']}, me {s['robot']}"

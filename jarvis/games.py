"""Rock paper scissors with the robot.

    "Hey Jarvis, rock paper scissors?"  -> Jarvis gets ready (forearm up, fist) and asks you to show a fist
    you show a fist, or say "ready"     -> "Rock", "paper", "scissors" with a fist pump on each, then "shoot!"
    on "shoot" both of you throw        -> Jarvis reads your hand through Tandem's gestures, says who won
    "again" / "one more"                -> next round straight away; "no" or "stop" ends the game

Your throw is read from Tandem's gestures: fist = rock, open palm (or four fingers) = paper, peace (V) =
scissors. Hands pump as fists during the count and people throw a little late, so your hand is read 0.6 s after
"shoot" (the latest throw shape seen by then), or up to 1.2 s after if no shape has shown yet.
Without a camera (or if the camera misses it) you can also just say or type "rock", "paper" or "scissors" then,
in English or in the language you are playing in ("tijera", "グー", "камень"...). Jarvis counts in that language
too, with its own chant where it has one (じゃんけんぽん, 가위 바위 보, 石头剪刀布).

Fair play: the robot picks its throw before the count starts, from a random generator that never sees yours.
"""
from __future__ import annotations

import random

from . import skills
from .i18n import tr
from .languages import contains_any, game_words, throw_words

BEAT_S = 0.6                                  # time between "rock", "paper", "scissors" and "shoot"
WINDOW = (-0.2, 1.2)                          # when your throw counts, seconds around "shoot"
DECIDE_S = 0.6                                # read your hand this long after "shoot" (it has changed by then)
READY_TIMEOUT_S = 12.0                        # start anyway after this long without a fist or "ready"
AGAIN_TIMEOUT_S = 20.0                        # end the game if nobody asks for another round
GESTURE_THROW = {"fist": "rock", "open_palm": "paper", "four": "paper", "peace": "scissors"}
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}     # each key beats its value


def winner(robot: str, person: str) -> str:
    """'robot', 'person' or 'tie'."""
    if robot == person:
        return "tie"
    return "robot" if BEATS[robot] == person else "person"


def throw_from_words(text: str, lang: str = "en"):
    """The throw named in what was said (in English or the language spoken), or None if none or several."""
    t = (text or "").lower()
    found = {throw for word, throw in throw_words(lang).items() if contains_any(t, [word], lang)}
    return found.pop() if len(found) == 1 else None


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

    def _words(self):
        return game_words(self.j.lang)

    # ------------------------------------------------------------------ input
    def hear(self, text: str, now: float) -> bool:
        """Words for the game. True if the game used them (so the brain does not also answer)."""
        t = (text or "").lower()
        lang = self.j.lang
        ready, again, done = self._words()
        if self.state == "ready" and contains_any(t, ready, lang):
            self._start_count(now)
            return True
        if self.state in ("count", "read"):
            throw = throw_from_words(t, lang)
            if throw:
                self.said_throw = throw
                return True
        if self.state == "again":
            if contains_any(t, done, lang):
                self.finish(now)
                return True
            if contains_any(t, again, lang):
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
        s, L = self.score, self.j.local
        tally = L("{you} to {me}, {ties} tied" if s["tie"] else "{you} to {me}", you=s["person"], me=s["robot"],
                  ties=s["tie"])
        rn, pn = L(r), L(p)
        cap = lambda x: x[:1].upper() + x[1:]                      # noqa: E731 (start of the sentence)
        if w == "robot":
            self.j.say("{robot} beats {person}. I win! That's {tally}. Again?", robot=cap(rn), person=pn, tally=tally)
            self.j.body.do(skills.celebrate(2.0), now, queue=True)
        elif w == "person":
            self.j.say("{person} beats {robot}. You got me. That's {tally}. Again?", robot=rn, person=cap(pn),
                       tally=tally)
            self.j.body.do(skills.shrug(1.5), now, queue=True)
        else:
            self.j.say("We both threw {throw}. Tie! That's {tally}. Again?", throw=rn, tally=tally)
            self.j.body.do(skills.nod(1), now, queue=True)
        self.state, self.t0 = "again", now

    def finish(self, now: float):
        s = self.score
        if s["person"] + s["robot"] + s["tie"]:
            line = ("Good game. You win the match, {you} to {me}." if s["person"] > s["robot"] else
                    "Good game. I win the match, {you} to {me}." if s["robot"] > s["person"] else
                    "Good game. It's a draw, {you} to {me}.")
            self.j.say(line, you=s["person"], me=s["robot"])
        else:
            self.j.say("Maybe later.")
        self.j.body.rest()
        self.done = True

    def status(self) -> str:
        s = self.score
        return f"rock paper scissors ({self.state}), you {s['person']}, me {s['robot']}"

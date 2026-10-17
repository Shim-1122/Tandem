"""Jarvis itself: perception, reflexes, brain and body in one loop.

Call `tick(now, tracker_values)` at the tracker's rate; it returns the joint targets for the robot. Speech comes in
through `hear(text)` from any thread (the microphone or the keyboard); the brain answers on a background thread
so a slow model never stalls the robot, and "stop" is handled on the spot.
"""
from __future__ import annotations

import queue
import threading
import time

from . import actions
from .brain import Brain, RuleBrain, is_stop
from .perception import Perception
from .reflexes import WAKE_GESTURES, Reflexes
from .skills import Body


class Jarvis:
    def __init__(self, brain: Brain | None = None, robot=None, speak=None, reflexes: bool = True,
                 log=print, clock=time.monotonic):
        self.brain = brain or RuleBrain()
        self.robot = robot
        self.perception = Perception()
        self.reflexes = Reflexes() if reflexes else None
        self.body = Body()
        self.speak_fn = speak
        self.log = log
        self.clock = clock
        self.on_wake = None                      # set by the voice front end: called to start listening
        self._lock = threading.Lock()
        self._todo: "queue.Queue[tuple]" = queue.Queue()
        self._thinking = 0
        self.transcript: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ actions and speech
    def act(self, name: str, args: dict | None = None, now: float | None = None, queue: bool = False) -> str | None:
        now = self.clock() if now is None else now
        with self._lock:
            try:
                done = actions.apply(self.body, name, args or {}, now, self.perception, queue=queue)
            except ValueError as e:
                self.log(f"[jarvis] can't {name}: {e}")
                return None
        self.log(f"[jarvis] {done}")
        return done

    def say(self, text: str):
        if not text:
            return
        self.transcript.append(("jarvis", text))
        self.log(f"Jarvis: {text}")
        if self.speak_fn:
            try:
                self.speak_fn(text)
            except Exception as e:                    # a broken speaker must never stop the robot
                self.log(f"[jarvis] speech failed: {e}")

    # ------------------------------------------------------------------ hearing
    def hear(self, text: str, wait: bool = False):
        """Something the person said or typed. Returns at once (or, with wait, after the reply is acted on)."""
        text = (text or "").strip()
        if not text:
            return
        self.transcript.append(("you", text))
        self.log(f"You: {text}")
        if is_stop(text):                              # safety: no queue, no model, right now
            self.act("freeze")
            self.say("Stopping.")
            return
        with self._lock:
            scene, state = self.perception.describe(), self.body.state()
            self._thinking += 1
        th = threading.Thread(target=self._think, args=(text, scene, state), daemon=True)
        th.start()
        if wait:
            th.join()
            self._drain(self.clock())

    def _think(self, text, scene, state):
        try:
            reply, calls = self.brain.respond(text, scene, state)
        except Exception as e:
            self.log(f"[jarvis] brain error: {e}")
            reply, calls = RuleBrain().respond(text, scene, state)
        self._todo.put((reply, calls))
        with self._lock:
            self._thinking -= 1

    @property
    def thinking(self) -> bool:
        return self._thinking > 0

    def _drain(self, now):
        while True:
            try:
                reply, calls = self._todo.get_nowait()
            except queue.Empty:
                return
            for i, (name, args) in enumerate(calls):
                self.act(name, args, now, queue=i > 0)   # several moves in one request run one after another
            self.say(reply)

    # ------------------------------------------------------------------ the loop
    def tick(self, now: float | None = None, tracker: dict | None = None) -> dict:
        now = self.clock() if now is None else now
        tracker = tracker or {}
        with self._lock:
            events = self.perception.update(tracker, now)
        for ev in events:
            self.log(f"[sense] {ev}")
            if ev.kind == "gesture" and ev.data.get("gesture") in WAKE_GESTURES and self.on_wake:
                self.on_wake()
            if self.reflexes is None:
                continue
            with self._lock:
                if self.body.shadow or self.body.frozen:
                    continue                           # copying or stopped: no surprise moves
                if self.body.busy and ev.kind not in ("person_left",):
                    acts, speech = [], None            # finishing a requested skill comes first
                else:
                    acts, speech = self.reflexes.react(ev, now)
            for name, args in acts:
                self.act(name, args, now)
            self.say(speech)
        self._drain(now)
        with self._lock:
            targets = self.body.targets(now, tracker, self.perception.head_direction())
        if self.robot is not None:
            self.robot.send(targets, now)
        return targets

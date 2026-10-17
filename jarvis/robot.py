"""Where Jarvis's joint targets go: the MuJoCo robot today, a microcontroller over USB serial later.

    SimRobot     drives vtrack.sim.TandemSim (physics, gravity, torque limits) and renders it
    SerialRobot  sends 0..1 commands per joint as JSON lines to a microcontroller (ESP32, Arduino, Teensy) that
                 sets the valves or servos. Not yet tested on hardware; read the notes below before using it.
    NullRobot    keeps the last targets, for tests and running without a robot
"""
from __future__ import annotations

import json
import math

from vtrack.mapping import JointChannel, SlewLimiter


class NullRobot:
    def __init__(self):
        self.last: dict = {}

    def send(self, targets: dict, now: float):
        self.last = dict(targets)

    def close(self):
        pass


class SimRobot:
    def __init__(self, scene: bool = True):
        from vtrack.sim import TandemSim
        self.sim = TandemSim(scene=scene)
        self._t = None

    def send(self, targets: dict, now: float):
        self.sim.command(targets)
        if self._t is not None:
            self.sim.step(now - self._t)
        self._t = now

    def render(self, camera: str = "front", width: int = 640, height: int = 480):
        return self.sim.render(camera, width, height)

    def close(self):
        self.sim.close()


class SerialRobot:
    """One JSON object per line, e.g. {"t": 12.345, "R_elbow_flexion": 0.42, ...}, values 0..1.

    Each joint you list in `channels` is mapped from its angle range to 0..1 (vtrack.mapping.JointChannel) and
    limited to `max_rate` units per second (SlewLimiter), so a bad target can never slam a joint. Only listed
    joints are sent: start with one joint, check its direction and range by hand, then add the next.
    The microcontroller turns 0..1 into a pressure setpoint (per muscle calibration) or a servo angle.
    """

    def __init__(self, port: str, channels: list[JointChannel], baud: int = 115200, max_rate: float = 1.0,
                 hz: float = 50.0):
        import serial                                   # pip install pyserial
        self.port = serial.Serial(port, baud, timeout=0)
        self.channels = {c.name: c for c in channels}
        self.limiters = {c.name: SlewLimiter(max_rate) for c in channels}
        self.period = 1.0 / hz
        self._last_sent = -1e9

    def send(self, targets: dict, now: float):
        if now - self._last_sent < self.period:
            return
        self._last_sent = now
        msg = {"t": round(now, 3)}
        for name, ch in self.channels.items():
            v = targets.get(name)
            if v is None or not math.isfinite(v):
                continue
            msg[name] = round(self.limiters[name](ch.command(v), now), 4)
        self.port.write((json.dumps(msg) + "\n").encode())

    def close(self):
        self.port.close()

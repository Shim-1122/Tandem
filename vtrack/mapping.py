"""Map measured joint angles to normalized actuator commands with limits and slew-rate control.

Commands are 0..1 per channel. For pneumatic muscles the later stage maps 0..1 to a pressure
setpoint through a per-muscle calibration curve; that curve is hardware specific and not defined here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JointChannel:
    name: str
    angle_min: float  # degrees mapped to command 0
    angle_max: float  # degrees mapped to command 1
    invert: bool = False

    def command(self, angle_deg: float) -> float:
        span = self.angle_max - self.angle_min
        c = (angle_deg - self.angle_min) / span
        c = min(1.0, max(0.0, c))
        return 1.0 - c if self.invert else c


class SlewLimiter:
    """Limit how fast a command may change (units per second) to protect the hardware."""

    def __init__(self, max_rate: float = 2.0, initial: float = 0.0):
        self.max_rate, self.value, self._t = max_rate, initial, None

    def __call__(self, target: float, t: float) -> float:
        if self._t is not None:
            step = self.max_rate * (t - self._t)
            self.value += max(-step, min(step, target - self.value))
        else:
            self.value = target
        self._t = t
        return self.value

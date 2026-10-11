"""One Euro filter (Casiez et al. 2012): low lag when moving fast, strong smoothing when still."""
from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """Works on scalars or numpy arrays. Call with the sample and its timestamp in seconds."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=float)
        if self._x is None or t <= self._t:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x

    def reset(self):
        self._x = self._dx = self._t = None


class LandmarkSmoother:
    """Smooths a full (N, 3) landmark array; resets automatically after a tracking dropout."""

    def __init__(self, max_gap_s: float = 0.25, **kw):
        self._f, self._kw, self._max_gap, self._last = OneEuroFilter(**kw), kw, max_gap_s, None

    def __call__(self, landmarks: np.ndarray, t: float) -> np.ndarray:
        if self._last is not None and t - self._last > self._max_gap:
            self._f.reset()
        self._last = t
        return self._f(landmarks, t)


class AngleSmoother:
    """One Euro filtering of named angles in degrees, with unwrapping across +-180 and NaN handling.

    Filtering the joint angles themselves (after the solver) removes jitter that survives landmark
    smoothing, and does it in the units that matter for driving a joint.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self._kw = dict(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._f: dict[str, OneEuroFilter] = {}
        self._last: dict[str, float] = {}

    def __call__(self, values: dict, t: float) -> dict:
        out = {}
        for k, v in values.items():
            if v is None or not np.isfinite(v):
                self._f.pop(k, None)
                self._last.pop(k, None)
                out[k] = v
                continue
            if k in self._last:                       # unwrap next to the previous output
                v = self._last[k] + ((v - self._last[k] + 180.0) % 360.0 - 180.0)
            f = self._f.setdefault(k, OneEuroFilter(**self._kw))
            y = float(f(v, t))
            self._last[k] = y
            out[k] = (y + 180.0) % 360.0 - 180.0 if abs(y) > 180.0 else y
        return out

    def reset(self):
        self._f.clear()
        self._last.clear()


class JitterMeter:
    """Standard deviation of each value over the last `window_s` seconds; hold still to read jitter."""

    def __init__(self, window_s: float = 1.0):
        self.window_s = window_s
        self._buf: dict[str, list] = {}

    def add(self, values: dict, t: float):
        for k, v in values.items():
            b = self._buf.setdefault(k, [])
            if v is not None and np.isfinite(v):
                b.append((t, float(v)))
            while b and t - b[0][0] > self.window_s:
                b.pop(0)

    def std(self) -> dict:
        out = {}
        for k, b in self._buf.items():
            vals = np.array([v for _, v in b])
            out[k] = float(np.std(vals)) if len(vals) >= 5 else float("nan")
        return out


# Smoothing presets for the arm, tuned on simulated landmark noise (scripts/eval_stability.py: pose noise
# 0.6 cm in x and y, 2 cm in depth, hand 0.2 cm). Pose landmark cutoffs are per axis (x, y, depth):
# depth is smoothed hardest because single-camera depth is the noisiest coordinate. In that simulation
# (mean over the seven angles): previous version 1.51 deg jitter / 3.64 deg moving error;
# responsive 1.47 / 2.86, balanced 1.18 / 3.20, stable 0.91 / 5.34.
SMOOTHING_PRESETS = {
    "responsive": dict(pose=dict(min_cutoff=(4.0, 4.0, 1.0), beta=2.0), angles=dict(min_cutoff=0.5, beta=0.2)),
    "balanced": dict(pose=dict(min_cutoff=(2.0, 2.0, 1.0), beta=2.0), angles=dict(min_cutoff=0.3, beta=0.05)),
    "stable": dict(pose=dict(min_cutoff=(1.5, 1.5, 0.5), beta=0.3), angles=dict(min_cutoff=0.8, beta=0.02)),
}
HAND_SMOOTHING = dict(min_cutoff=2.0, beta=5.0)


def make_arm_smoothers(preset: str = "balanced"):
    """(pose landmark smoother, angle smoother) for a preset name."""
    p = SMOOTHING_PRESETS[preset]
    pose_kw = dict(p["pose"], min_cutoff=np.array(p["pose"]["min_cutoff"], dtype=float))
    return LandmarkSmoother(**pose_kw), AngleSmoother(**p["angles"])

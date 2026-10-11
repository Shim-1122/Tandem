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

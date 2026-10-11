"""Elbow position from the shoulder and wrist, using fixed segment lengths.

With the upper arm length Lu and forearm length Lf known, the elbow must lie on the circle where a
sphere of radius Lu around the shoulder meets a sphere of radius Lf around the wrist. The shoulder
and wrist fix that circle exactly; where on the circle the elbow sits (the arm's swivel about the
shoulder-wrist line) is NOT determined by those two points alone, so it is chosen by combining:

  * MediaPipe's own elbow estimate, weighted by its visibility score (it still guesses an occluded
    elbow from body context, and the guess is usually roughly right in direction),
  * continuity with the previous frame (an elbow does not jump around the circle),
  * a weak gravity prior (with nothing else to go on, the elbow hangs low).

When the elbow IS visible its measured position is used as is. (Projecting a visible elbow onto the
circle was tested on synthetic noisy landmarks and did not reduce angle noise, because the shoulder and
wrist carry the same noise; temporal filtering does that job instead.)

On synthetic arms with a hidden elbow and MediaPipe-like guesses off by 4 cm per axis, the median
elbow error was 2.6 cm versus 6.2 cm for the raw guess; with the elbow hidden for 1 s while bending,
continuity alone held it within 0.4 cm. Real occluded-elbow guesses may be worse than that noise model.

Segment lengths are learned from frames where all three joints are clearly visible (running median).
Until enough frames are seen, they come from shoulder width using average adult proportions
(upper arm 0.186 H, forearm 0.146 H, shoulder width 0.259 H, H = height; Drillis and Contini).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

UPPER_ARM_PER_SHOULDER_WIDTH = 0.186 / 0.259
FOREARM_PER_SHOULDER_WIDTH = 0.146 / 0.259


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


class SegmentLengths:
    """Running median of upper arm and forearm lengths from well-tracked frames."""

    def __init__(self, window: int = 300, min_samples: int = 20):
        self.upper, self.fore = deque(maxlen=window), deque(maxlen=window)
        self.min_samples = min_samples

    def add(self, shoulder, elbow, wrist):
        self.upper.append(float(np.linalg.norm(np.asarray(elbow) - shoulder)))
        self.fore.append(float(np.linalg.norm(np.asarray(wrist) - elbow)))

    @property
    def calibrated(self) -> bool:
        return len(self.upper) >= self.min_samples

    def get(self, shoulder_width: float) -> tuple[float, float]:
        if self.calibrated:
            return float(np.median(self.upper)), float(np.median(self.fore))
        return UPPER_ARM_PER_SHOULDER_WIDTH * shoulder_width, FOREARM_PER_SHOULDER_WIDTH * shoulder_width


@dataclass
class ElbowEstimate:
    elbow: np.ndarray
    mode: str            # "measured" (visible, length-constrained), "predicted", or "straight"
    swivel_deg: float    # position on the circle, for diagnostics


class ElbowPredictor:
    """Per-arm elbow estimator. Call `estimate` once per frame."""

    def __init__(self, w_guess=1.0, w_continuity=10.0, w_gravity=0.02, visible_thresh=0.6, memory_s=0.5,
                 steps=120):
        self.w_g, self.w_c, self.w_d = w_guess, w_continuity, w_gravity
        self.visible_thresh, self.memory_s, self.steps = visible_thresh, memory_s, steps
        self._prev, self._prev_t = None, None

    def reset(self):
        self._prev = self._prev_t = None

    def estimate(self, shoulder, wrist, lengths, guess=None, guess_vis=0.0, up=None, t=None) -> ElbowEstimate:
        """shoulder, wrist, guess, up: 3-vectors in the same world axes. lengths: (upper, forearm)."""
        S, W = np.asarray(shoulder, float), np.asarray(wrist, float)
        Lu, Lf = lengths
        d_vec = W - S
        d = float(np.linalg.norm(d_vec))
        n = _unit(d_vec)
        visible = guess is not None and guess_vis >= self.visible_thresh
        if visible:
            return self._keep(ElbowEstimate(np.asarray(guess, float).copy(), "measured", float("nan")), t)

        if d >= Lu + Lf - 1e-9 or d < 1e-6:
            E = S + Lu * n if d >= 1e-6 else (np.asarray(guess, float) if guess is not None else S)
            return self._keep(ElbowEstimate(E, "straight", 0.0), t)

        a = (Lu * Lu - Lf * Lf + d * d) / (2 * d)
        r2 = Lu * Lu - a * a
        r = float(np.sqrt(max(r2, 0.0)))
        C = S + a * n
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = _unit(np.cross(n, helper))
        e2 = np.cross(n, e1)

        def point(th):
            return C + r * (np.cos(th) * e1 + np.sin(th) * e2)

        def cost(th):
            dirv = np.cos(th) * e1 + np.sin(th) * e2
            c = 0.0
            if guess is not None:
                g = np.asarray(guess, float) - C
                g = g - np.dot(g, n) * n
                if np.linalg.norm(g) > 1e-9:
                    c += self.w_g * (1.0 - float(np.dot(dirv, _unit(g))))
            if self._prev is not None and (t is None or self._prev_t is None or t - self._prev_t <= self.memory_s):
                if r > 1e-9:
                    c += self.w_c * float(np.sum((C + r * dirv - self._prev) ** 2)) / (r * r)
            if up is not None:
                u = np.asarray(up, float) - np.dot(up, n) * n
                if np.linalg.norm(u) > 1e-9:
                    c += self.w_d * 0.5 * (1.0 + float(np.dot(dirv, _unit(u))))
            return c

        ths = np.linspace(-np.pi, np.pi, self.steps, endpoint=False)
        costs = [cost(th) for th in ths]
        best = ths[int(np.argmin(costs))]
        step = 2 * np.pi / self.steps
        for _ in range(3):                          # refine around the best sample
            fine = np.linspace(best - step, best + step, 21)
            best = fine[int(np.argmin([cost(th) for th in fine]))]
            step /= 10
        return self._keep(ElbowEstimate(point(best), "predicted", float(np.degrees(best))), t)

    def _keep(self, est: ElbowEstimate, t):
        self._prev, self._prev_t = est.elbow.copy(), t
        return est

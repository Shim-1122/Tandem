"""Per-person emotion calibration.

Expression models are less accurate for some faces than others: a resting face can read as sad or angry
because of its shape rather than its expression, and how strongly each person shows each emotion differs.
Calibration corrects this for one person with that person's own data. You make each of the seven
expressions on cue for a couple of seconds; a small correction of the model's raw scores is fitted (one
overall scale and one offset per emotion, 8 numbers: z' = scale * z + bias, the "temperature plus bias"
form of calibration) and saved under your name in faces/. It is used whenever you are recognized, or for
--name when nobody is enrolled.

The accuracy shown afterwards is measured on frames the fit did not use: the correction is fitted on the
first half of each expression's frames and scored on the second half, before and after. The saved
correction is then fitted on all frames. Posed expressions are easier than spontaneous ones, so treat the
numbers as a best case for this session's lighting and camera.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .face import EMOTIONS, softmax

CAL_ORDER = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgust"]
PROMPT = {"neutral": "a relaxed NEUTRAL face", "happy": "a HAPPY face (smile)", "sad": "a SAD face",
          "angry": "an ANGRY face", "fearful": "a FEARFUL face", "surprised": "a SURPRISED face",
          "disgust": "a DISGUSTED face"}
MIN_PER_EXPRESSION = 2


def _loss_grad_hess(theta, Z, Y, reg):
    """Mean cross entropy of softmax(s * z + b) plus reg * ((s - 1)^2 + |b|^2), with gradient and Hessian."""
    s, b = theta[0], theta[1:]
    L = s * Z + b
    L = L - L.max(axis=1, keepdims=True)
    P = np.exp(L)
    P /= P.sum(axis=1, keepdims=True)
    n, k = Z.shape
    loss = -np.mean(np.log(np.maximum(P[np.arange(n), Y], 1e-300))) + reg * ((s - 1.0) ** 2 + b @ b)
    onehot = np.zeros_like(P)
    onehot[np.arange(n), Y] = 1.0
    D = P - onehot
    grad = np.concatenate([[np.mean(np.sum(D * Z, axis=1))], D.mean(axis=0)])
    grad += 2 * reg * np.concatenate([[s - 1.0], b])
    # Hessian of the mean of J^T (diag p - p p^T) J with J = [z, I] (d logits / d (s, b)), vectorized
    pz = np.sum(P * Z, axis=1)
    H = np.empty((k + 1, k + 1))
    H[0, 0] = np.mean(np.sum(P * Z * Z, axis=1) - pz ** 2)
    H[0, 1:] = H[1:, 0] = np.mean(P * Z - P * pz[:, None], axis=0)
    H[1:, 1:] = np.diag(P.mean(axis=0)) - P.T @ P / n
    H += 2 * reg * np.eye(k + 1)
    return loss, grad, H


def fit_scale_bias(logits, labels, reg: float = 0.02, iters: int = 50) -> tuple[float, np.ndarray]:
    """Scale and per-class bias minimizing cross entropy (convex; damped Newton with a line search)."""
    Z, Y = np.asarray(logits, dtype=float), np.asarray(labels, dtype=int)
    theta = np.concatenate([[1.0], np.zeros(Z.shape[1])])
    for _ in range(iters):
        loss, g, H = _loss_grad_hess(theta, Z, Y, reg)
        step = np.linalg.solve(H, g)
        t, cand = 1.0, None
        while t > 1e-6:
            trial = theta - t * step
            if trial[0] > 0.05 and _loss_grad_hess(trial, Z, Y, reg)[0] <= loss - 1e-4 * t * (g @ step):
                cand = trial
                break
            t *= 0.5
        if cand is None:                                     # no further decrease: converged
            break
        theta = cand
        if np.linalg.norm(t * step) < 1e-8:
            break
    return float(theta[0]), theta[1:].copy()


@dataclass
class EmotionCalibration:
    scale: float = 1.0
    bias: list = field(default_factory=lambda: [0.0] * len(EMOTIONS))
    accuracy_before: float = float("nan")        # held-out, uncalibrated
    accuracy_after: float = float("nan")         # held-out, calibrated
    frames: int = 0
    created: str = ""
    model: str = "opencv"                        # emotion model the scores came from (files made before this field: opencv)

    def logits(self, z) -> np.ndarray:
        return self.scale * np.asarray(z, dtype=float) + np.asarray(self.bias, dtype=float)

    def probabilities(self, z) -> np.ndarray:
        return softmax(self.logits(z))

    def summary(self) -> str:
        return f"held-out accuracy {100 * self.accuracy_before:.0f}% -> {100 * self.accuracy_after:.0f}%"

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["bias"] = [float(v) for v in self.bias]
        d["emotions"] = EMOTIONS
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EmotionCalibration":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.pop("emotions", EMOTIONS) != EMOTIONS or len(d.get("bias", [])) != len(EMOTIONS):
            raise ValueError(f"{path}: made for a different emotion model")
        return cls(**d)


def calibration_path(faces_dir, name: str) -> Path:
    return Path(faces_dir) / f"{name}.emotion.json"


def load_all(faces_dir, model: str | None = None) -> dict:
    """{name: EmotionCalibration} for every calibration file in faces_dir made for `model` (any if None).

    Unreadable files, and calibrations of a different emotion model (their scores mean something else),
    are skipped.
    """
    out = {}
    for f in sorted(Path(faces_dir).glob("*.emotion.json")) if Path(faces_dir).exists() else []:
        try:
            cal = EmotionCalibration.load(f)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if model is None or cal.model == model:
            out[f.name[:-len(".emotion.json")]] = cal
    return out


def calibrate(logits, labels, times, model: str = "hsemotion") -> EmotionCalibration:
    """Fit on all frames; accuracy before/after from fitting the first half of each expression, testing the rest."""
    Z, Y, T = np.asarray(logits, float), np.asarray(labels, int), np.asarray(times, float)
    fit_idx, test_idx = [], []
    for c in np.unique(Y):
        idx = np.flatnonzero(Y == c)
        idx = idx[np.argsort(T[idx])]
        half = len(idx) // 2
        fit_idx += list(idx[:half])
        test_idx += list(idx[half:])
    s, b = fit_scale_bias(Z[fit_idx], Y[fit_idx])
    before = float(np.mean(np.argmax(Z[test_idx], axis=1) == Y[test_idx]))
    after = float(np.mean(np.argmax(s * Z[test_idx] + b, axis=1) == Y[test_idx]))
    s, b = fit_scale_bias(Z, Y)
    return EmotionCalibration(s, [float(v) for v in b], before, after, len(Y), time.strftime("%Y-%m-%d %H:%M"), model)


class CalibrationSession:
    """Guided capture: for each expression, `prepare_s` seconds to make it, then `capture_s` seconds recorded.

    Feed every new emotion result with `add(raw logits, capture time)`; when `finished(now)`, call `result()`.
    """

    def __init__(self, name: str, t0: float, prepare_s: float = 1.5, capture_s: float = 2.5, order=CAL_ORDER,
                 model: str = "hsemotion"):
        self.name, self.t0, self.prep, self.cap, self.order = name, t0, prepare_s, capture_s, list(order)
        self.model = model
        self.samples: list[tuple[np.ndarray, int, float]] = []

    @property
    def duration(self) -> float:
        return len(self.order) * (self.prep + self.cap)

    def phase(self, t: float) -> tuple[int, str, float]:
        """(expression index, "prepare" | "capture" | "done", seconds left in this phase)."""
        dt = t - self.t0
        if dt < 0:                                      # a frame captured before the calibration started
            return 0, "prepare", self.prep - dt
        if dt >= self.duration:
            return len(self.order), "done", 0.0
        i = int(dt // (self.prep + self.cap))
        within = dt - i * (self.prep + self.cap)
        return (i, "prepare", self.prep - within) if within < self.prep else (i, "capture", self.prep + self.cap - within)

    def prompt(self, t: float) -> str:
        i, ph, left = self.phase(t)
        if ph == "done":
            return "emotion calibration: fitting ..."
        what = PROMPT[self.order[i]]
        if ph == "prepare":
            return f"Emotion calibration {i + 1}/{len(self.order)}: make {what}   (recording in {left:.1f} s)"
        return f"Emotion calibration {i + 1}/{len(self.order)}: hold {what}   (recording, {left:.1f} s)"

    def add(self, logits, t_capture: float) -> bool:
        i, ph, _ = self.phase(t_capture)
        if ph != "capture":
            return False
        self.samples.append((np.asarray(logits, dtype=float).copy(), EMOTIONS.index(self.order[i]), t_capture))
        return True

    def finished(self, t: float) -> bool:
        return self.phase(t)[1] == "done"

    def counts(self) -> dict:
        return {e: sum(1 for _, y, _ in self.samples if EMOTIONS[y] == e) for e in self.order}

    def result(self) -> EmotionCalibration:
        short = [e for e, n in self.counts().items() if n < MIN_PER_EXPRESSION]
        if short:
            raise ValueError("not enough frames with a face for: " + ", ".join(short))
        Z, Y, T = zip(*self.samples)
        return calibrate(np.array(Z), np.array(Y), np.array(T), self.model)

"""Eye tracking (where you are looking) and target locking.

Each frame, from the face mesh (478 landmarks, which include the two iris centers) and the head pose:
  1. The eye corners and iris centers are placed in 3D with the same depth model as the head pose (head.py)
     and expressed in the head's own frame, which removes head rotation and perspective.
  2. The iris offset from the midpoint of its eye's corners gives the eye's rotation in the head (eyeball
     radius 12 mm), averaged over both eyes; that also cancels each eye's natural sideways offset. Frames
     with a closed eye (a blink) are skipped.
  3. The gaze ray, from between the eyes along the head axes turned by the eye angles, is intersected with
     the plane of the camera: the screen the webcam sits on.
  4. Calibration (key g) maps that point onto the video: you look at 9 dots, and an affine map (6 numbers)
     is fitted from the plane to the video's pixels. Because it works on the geometric intersection point,
     it stays valid when you move or turn your head. Its accuracy is measured by leaving each dot out of
     the fit in turn and predicting it, and reported in pixels and degrees.

Target locking: with gaze calibrated, the point you look at can be locked (key t, a pinch gesture, or
optionally by holding your gaze still). The locked target is reported in camera pixels and as a ray from
the camera, ready to hand to a robot.

Angles: eye yaw + is toward your left, eye pitch + is up (the same directions as the head's yaw and pitch).
Webcam eye tracking is coarse: expect a few degrees after calibration, and redo it if the camera, window
or your seat moves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .head import ANCHOR_IDS, HEAD_AXES_IN_MODEL, MODEL_POINTS_M

EYES = {"right": dict(outer=33, inner=133, iris=468, upper=159, lower=145),     # your right eye
        "left": dict(outer=263, inner=362, iris=473, upper=386, lower=374)}
EYEBALL_RADIUS_M = 0.012
BLINK_SCORE = 0.5             # MediaPipe blendshape eyeBlink score above this: the eye is closed
CLOSED_RATIO = 0.13            # lid gap / eye width below this: the eye is closed
MIN_SPREAD_M = 0.005           # calibration: the gaze points must spread at least this much (std, each way)
NEUTRAL_DY_M = 0.0             # iris height above the eye-corner line when looking straight ahead (head frame);
                               # measured near 0 (+-1 mm) on a real face; calibration absorbs the rest


def landmarks_in_3d(L: np.ndarray, idx, pose, width: int, height: int, K: np.ndarray):
    """(camera coordinates, head model coordinates) in meters of face landmarks `idx`.

    Uses the head pose's depth model: MediaPipe's relative depth, scaled at the anchors' distance. Model
    axes: x toward your left, y up, z forward (out of the face).
    """
    L = np.asarray(L, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R_model = pose.axes @ HEAD_AXES_IN_MODEL.T
    t = np.asarray(pose.position_m, dtype=float)
    z_mean = float(np.mean((R_model @ MODEL_POINTS_M.T)[2] + t[2]))
    z_ref = float(np.mean(L[ANCHOR_IDS, 2])) * width
    P = L[list(idx)]
    Z = z_mean + (P[:, 2] * width - z_ref) * z_mean / fx
    cam = np.column_stack([(P[:, 0] * width - cx) * Z / fx, (P[:, 1] * height - cy) * Z / fy, Z])
    return cam, (R_model.T @ (cam - t).T).T


@dataclass
class EyeState:
    valid: bool                                  # both eyes open and the face found
    yaw: float = float("nan")                    # eye in head, degrees, + toward your left
    pitch: float = float("nan")                  # + up
    origin: np.ndarray | None = None             # between the eyes, camera coordinates (m)
    direction: np.ndarray | None = None          # gaze direction, camera coordinates (unit)
    plane_point: np.ndarray | None = None        # where the gaze meets the camera's plane (x right, y down, m)
    openness: tuple = (float("nan"), float("nan"))   # lid gap / eye width, (right, left)


def eye_state(L, pose, width: int, height: int, K: np.ndarray, neutral_dy: float = NEUTRAL_DY_M,
              blink=None) -> EyeState:
    """blink: optional (right, left) blink scores 0..1 from MediaPipe's face blendshapes; either eye above
    BLINK_SCORE counts as closed, as does a lid gap under CLOSED_RATIO of the eye's width."""
    if L is None or pose is None or len(L) < 478:
        return EyeState(False)
    L = np.asarray(L, dtype=float)
    px = L[:, :2] * np.array([width, height])
    opened, offsets, mids = [], [], []
    for eye in ("right", "left"):
        e = EYES[eye]
        w = np.linalg.norm(px[e["outer"]] - px[e["inner"]])
        opened.append(np.linalg.norm(px[e["upper"]] - px[e["lower"]]) / w if w > 1e-6 else 0.0)
        cam, model = landmarks_in_3d(L, [e["outer"], e["inner"], e["iris"]], pose, width, height, K)
        mid_m = (model[0] + model[1]) / 2
        offsets.append(model[2] - mid_m)
        mids.append((cam[0] + cam[1]) / 2)
    state = EyeState(False, openness=(float(opened[0]), float(opened[1])))
    if min(opened) < CLOSED_RATIO or (blink is not None and max(blink) > BLINK_SCORE):
        return state
    dx, dy = float(np.mean([o[0] for o in offsets])), float(np.mean([o[1] for o in offsets])) - neutral_dy
    R = EYEBALL_RADIUS_M
    pitch = math.asin(max(-1.0, min(1.0, dy / R)))
    yaw = math.asin(max(-1.0, min(1.0, dx / (R * math.cos(pitch)))))
    d_model = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
    R_model = pose.axes @ HEAD_AXES_IN_MODEL.T
    d = R_model @ d_model
    o = (mids[0] + mids[1]) / 2
    state.valid, state.yaw, state.pitch = True, math.degrees(yaw), math.degrees(pitch)
    state.origin, state.direction = o, d
    if d[2] < -0.05:                                     # looking back toward the camera's plane
        state.plane_point = (o - (o[2] / d[2]) * d)[:2]
    return state


def gaze_direction_in_head(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Gaze direction in head axes (forward, up, right) for eye yaw (+ left) and pitch (+ up)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.cos(y) * math.cos(p), math.sin(p), -math.sin(y) * math.cos(p)])


# ---------------------------------------------------------------------------------------------- calibration
def calibration_targets(width: int, height: int, margin: float = 0.12) -> list:
    """Nine dots (3 x 3) in video pixels."""
    xs = [margin * width, width / 2, (1 - margin) * width]
    ys = [margin * height, height / 2, (1 - margin) * height]
    order = [(1, 1), (0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1)]   # center first
    return [np.array([xs[i], ys[j]]) for i, j in order]


def fit_affine(X, Y) -> np.ndarray:
    """2 x 3 matrix A minimizing |A [x, 1] - y|^2 over the pairs (least squares)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    A, *_ = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), Y, rcond=None)
    return A.T


@dataclass
class GazeMapping:
    A: np.ndarray                                # 2 x 3: camera-plane meters -> video pixels
    error_px: float                              # mean leave-one-dot-out error
    error_deg: float                             # the same as an angle at your eyes
    dots: int = 0

    def predict(self, plane_point) -> np.ndarray:
        return self.A @ np.append(np.asarray(plane_point, float), 1.0)

    def summary(self) -> str:
        return f"accuracy about {self.error_deg:.1f} deg ({self.error_px:.0f} px), from {self.dots} dots"


class GazeCalibrationSession:
    """Shows each dot for `settle_s` seconds (eyes move and settle), then records `capture_s` seconds."""

    def __init__(self, targets, t0: float, settle_s: float = 1.0, capture_s: float = 1.2):
        self.targets, self.t0, self.settle, self.capture = [np.asarray(p, float) for p in targets], t0, settle_s, capture_s
        self.samples: list[tuple[int, np.ndarray, float]] = []        # (dot, plane point, distance to eyes)

    @property
    def duration(self) -> float:
        return len(self.targets) * (self.settle + self.capture)

    def phase(self, t: float) -> tuple[int, str, float]:
        dt = t - self.t0
        if dt < 0:                                      # a frame captured before the calibration started
            return 0, "settle", self.settle - dt
        if dt >= self.duration:
            return len(self.targets), "done", 0.0
        i = int(dt // (self.settle + self.capture))
        within = dt - i * (self.settle + self.capture)
        return (i, "settle", self.settle - within) if within < self.settle else (i, "capture", self.settle + self.capture - within)

    def add(self, plane_point, distance_m: float, t_capture: float) -> bool:
        i, ph, _ = self.phase(t_capture)
        if ph != "capture" or plane_point is None:
            return False
        self.samples.append((i, np.asarray(plane_point, float).copy(), float(distance_m)))
        return True

    def finished(self, t: float) -> bool:
        return self.phase(t)[1] == "done"

    def result(self, min_dots: int = 6) -> GazeMapping:
        dots = sorted({i for i, _, _ in self.samples if sum(1 for j, _, _ in self.samples if j == i) >= 2})
        if len(dots) < min_dots:
            raise ValueError(f"eyes found on only {len(dots)} of {len(self.targets)} dots (blinking or face lost)")
        X = np.array([np.mean([p for j, p, _ in self.samples if j == i], axis=0) for i in dots])
        Y = np.array([self.targets[i] for i in dots])
        spread = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)[-1] / math.sqrt(len(dots))
        if spread < MIN_SPREAD_M:                       # the gaze hardly moved: the map would only fit noise
            raise ValueError("gaze hardly moved: follow each dot with your eyes")
        dist = float(np.mean([d for _, _, d in self.samples]))
        A = fit_affine(X, Y)
        errs_px, errs_m = [], []
        for k in range(len(dots)):                      # leave each dot out, predict it
            keep = [j for j in range(len(dots)) if j != k]
            Ak = fit_affine(X[keep], Y[keep])
            err = Ak @ np.append(X[k], 1.0) - Y[k]
            errs_px.append(float(np.linalg.norm(err)))
            errs_m.append(float(np.linalg.norm(np.linalg.pinv(Ak[:, :2]) @ err)))
        err_deg = math.degrees(math.atan2(float(np.mean(errs_m)), dist)) if dist > 0 else float("nan")
        return GazeMapping(A, float(np.mean(errs_px)), err_deg, len(dots))


# ------------------------------------------------------------------------------------------- target lock
@dataclass
class TargetLock:
    """Locks the gaze point as a target. Display points are in the (possibly mirrored) video as shown."""

    width: int
    height: int
    K: np.ndarray
    mirrored: bool = True
    dwell_s: float = 0.0                           # > 0: also lock by holding the gaze still this long
    radius_px: float = 50.0
    locked: bool = False
    display_point: np.ndarray | None = None
    t_locked: float = float("nan")
    _dwell: list = field(default_factory=list)

    def camera_point(self, display_point=None) -> np.ndarray | None:
        p = self.display_point if display_point is None else display_point
        if p is None:
            return None
        return np.array([self.width - 1 - p[0], p[1]]) if self.mirrored else np.asarray(p, float).copy()

    def ray(self) -> np.ndarray | None:
        """Unit direction from the camera toward the locked target (camera axes: x right, y down, z away)."""
        c = self.camera_point()
        if not self.locked or c is None:
            return None
        v = np.linalg.solve(self.K, np.array([c[0], c[1], 1.0]))
        return v / np.linalg.norm(v)

    def lock(self, gaze_point, t: float) -> bool:
        if gaze_point is None:
            return False
        self.locked, self.display_point, self.t_locked = True, np.asarray(gaze_point, float).copy(), t
        self._dwell.clear()
        return True

    def release(self):
        self.locked, self.display_point = False, None
        self._dwell.clear()

    def toggle(self, gaze_point, t: float) -> str:
        if self.locked:
            self.release()
            return "released"
        return "locked" if self.lock(gaze_point, t) else "no gaze"

    def dwell_progress(self, t: float) -> float:
        if self.dwell_s <= 0 or not self._dwell:
            return 0.0
        return min(1.0, (t - self._dwell[0][0]) / self.dwell_s)

    def update(self, gaze_point, t: float) -> bool:
        """Dwell locking; True when it just locked."""
        if self.dwell_s <= 0 or self.locked:
            return False
        if gaze_point is None:
            self._dwell.clear()
            return False
        p = np.asarray(gaze_point, float)
        if self._dwell and np.linalg.norm(p - np.mean([q for _, q in self._dwell], axis=0)) > self.radius_px:
            self._dwell.clear()
        self._dwell.append((t, p))
        if t - self._dwell[0][0] >= self.dwell_s:
            return self.lock(np.mean([q for _, q in self._dwell], axis=0), t)
        return False

"""MediaPipe Tasks wrappers: hand, pose and face landmarks per video frame.

HandModel, PoseModel and FaceModel each own one MediaPipe task (VIDEO running mode, which tracks from frame to
frame instead of detecting from scratch) and are meant to be used from one worker thread each (see
pipeline.py). They share one base class for the task options, timestamps and image conversion. FaceModel also
asks MediaPipe for its face blendshapes, of which the eye blink scores are used to skip blinks in eye tracking.
Imports mediapipe lazily so the math modules and tests run without it installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "hand": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_lite": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_full": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "pose_heavy": f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "face": f"{BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def ensure_model(key: str, folder: str | Path = "models") -> Path:
    """Download a model file on first use and return its local path."""
    from .face import download                      # certificate-safe download (macOS python.org Python)
    return download(MODELS[key], Path(folder) / Path(MODELS[key]).name)


@dataclass
class HandResult:
    handedness: str          # "Left" or "Right" as labeled by MediaPipe (assumes un-mirrored input)
    world: np.ndarray        # (21, 3) meters, origin at hand center
    image: np.ndarray        # (21, 3) normalized image coordinates, for drawing only


@dataclass
class FrameResult:
    hands: list[HandResult] = field(default_factory=list)
    pose_world: np.ndarray | None = None   # (33, 3) meters, origin at hip midpoint
    pose_image: np.ndarray | None = None   # (33, 3) normalized, for drawing only
    pose_visibility: np.ndarray | None = None  # (33,)


def _arr(landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)


def pick_hand(res: "FrameResult", side: str, max_dist: float = 0.15) -> "HandResult | None":
    """The detected hand closest (in the image) to the pose's wrist for `side` ("left" or "right").

    Uses position, not MediaPipe's handedness label, because the label assumes a mirrored image.
    Returns None when no hand is within `max_dist` (normalized image units) of that wrist.
    """
    if res.pose_image is None or not res.hands:
        return None
    wrist = res.pose_image[15 if side == "left" else 16][:2]
    best = min(res.hands, key=lambda h: np.linalg.norm(h.image[0][:2] - wrist))
    return best if np.linalg.norm(best.image[0][:2] - wrist) <= max_dist else None


def assign_hands(res: "FrameResult", sides, max_dist: float = 0.15) -> dict:
    """{side: HandResult or None}, each detected hand given to at most one side.

    Pairs are taken closest first, so with arms crossed or hands close together one hand is never
    counted as both hands (pick_hand, which looks at one side at a time, can do that).
    """
    out = {sd: None for sd in sides}
    if res.pose_image is None or not res.hands:
        return out
    pairs = []
    for sd in sides:
        wrist = res.pose_image[15 if sd == "left" else 16][:2]
        for i, h in enumerate(res.hands):
            d = float(np.linalg.norm(h.image[0][:2] - wrist))
            if d <= max_dist:
                pairs.append((d, sd, i))
    used = set()
    for d, sd, i in sorted(pairs):
        if out[sd] is None and i not in used:
            out[sd] = res.hands[i]
            used.add(i)
    return out


class _Timestamps:
    """MediaPipe VIDEO mode needs strictly increasing millisecond timestamps per task."""

    def __init__(self):
        self.last = -1

    def __call__(self, t_seconds: float) -> int:
        ts = max(int(t_seconds * 1000), self.last + 1)
        self.last = ts
        return ts


def _image(rgb):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class _LandmarkTask:
    """One MediaPipe landmarker task in VIDEO mode: builds it from a model key and options, converts frames,
    keeps the timestamps increasing, and closes it."""

    task_name = ""                      # HandLandmarker, PoseLandmarker or FaceLandmarker
    gpu_failed = False                  # set once the graphics processor could not be used (then CPU for all)

    def __init__(self, model_key: str, model_dir: str = "models", gpu: bool = False, **options):
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        task = getattr(vision, self.task_name)
        path = str(ensure_model(model_key, model_dir))

        def build(delegate):
            base = mpt.BaseOptions(model_asset_path=path, delegate=delegate)
            opts = getattr(vision, self.task_name + "Options")(base_options=base, running_mode=vision.RunningMode.VIDEO,
                                                               **options)
            return task.create_from_options(opts)

        self.on_gpu = False
        if gpu and not _LandmarkTask.gpu_failed:
            try:                                 # MediaPipe's GPU delegate: Linux and macOS, not Windows
                self.task, self.on_gpu = build(mpt.BaseOptions.Delegate.GPU), True
            except Exception as exc:             # noqa: BLE001 (any failure: use the CPU instead)
                _LandmarkTask.gpu_failed = True
                print(f"{self.task_name}: the graphics processor is not available ({type(exc).__name__}); using the CPU")
        if not self.on_gpu:
            self.task = build(mpt.BaseOptions.Delegate.CPU)
        self.ts = _Timestamps()

    def detect(self, rgb: np.ndarray, t_seconds: float):
        return self.task.detect_for_video(_image(rgb), self.ts(t_seconds))

    def close(self):
        self.task.close()


class HandModel(_LandmarkTask):
    task_name = "HandLandmarker"

    def __init__(self, num_hands: int = 2, model_dir: str = "models", gpu: bool = False):
        super().__init__("hand", model_dir, gpu, num_hands=num_hands)

    def process(self, rgb: np.ndarray, t_seconds: float) -> list[HandResult]:
        h = self.detect(rgb, t_seconds)
        return [HandResult(hd[0].category_name, _arr(wl), _arr(lm))
                for lm, wl, hd in zip(h.hand_landmarks, h.hand_world_landmarks, h.handedness)]


@dataclass
class PoseResult:
    world: np.ndarray        # (33, 3) meters, origin at hip midpoint
    image: np.ndarray        # (33, 3) normalized
    visibility: np.ndarray   # (33,)


class PoseModel(_LandmarkTask):
    task_name = "PoseLandmarker"

    def __init__(self, variant: str = "pose_full", model_dir: str = "models", gpu: bool = False):
        super().__init__(variant, model_dir, gpu, num_poses=1)

    def process(self, rgb: np.ndarray, t_seconds: float) -> PoseResult | None:
        p = self.detect(rgb, t_seconds)
        if not p.pose_world_landmarks:
            return None
        return PoseResult(_arr(p.pose_world_landmarks[0]), _arr(p.pose_landmarks[0]),
                          np.array([q.visibility for q in p.pose_landmarks[0]]))


class FaceModel(_LandmarkTask):
    """Face mesh: 478 landmarks (normalized x, y and relative depth z) for the most prominent face, plus
    MediaPipe's eye blink scores (0 open .. 1 closed) from its blendshape model."""

    task_name = "FaceLandmarker"

    def __init__(self, model_dir: str = "models", gpu: bool = False):
        super().__init__("face", model_dir, gpu, num_faces=1, output_face_blendshapes=True)
        self.blink = None                       # (right eye, left eye) of the person, from the last frame

    def process(self, rgb: np.ndarray, t_seconds: float) -> np.ndarray | None:
        r = self.detect(rgb, t_seconds)
        self.blink = blink_scores(r.face_blendshapes[0]) if r.face_blendshapes else None
        return _arr(r.face_landmarks[0]) if r.face_landmarks else None


def blink_scores(blendshapes) -> tuple | None:
    """(right eye, left eye) blink scores of the person from MediaPipe's face blendshapes.

    The blendshapes follow ARKit's naming, where eyeBlinkLeft is the person's own left eye.
    """
    d = {c.category_name: float(c.score) for c in blendshapes}
    if "eyeBlinkRight" not in d or "eyeBlinkLeft" not in d:
        return None
    return d["eyeBlinkRight"], d["eyeBlinkLeft"]

"""MediaPipe Tasks wrapper: hand and pose landmarks per video frame.

Imports mediapipe lazily so the math modules and tests run without it installed.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "hand": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_lite": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_full": f"{BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "pose_heavy": f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def ensure_model(key: str, folder: str | Path = "models") -> Path:
    """Download a model file on first use and return its local path."""
    path = Path(folder) / Path(MODELS[key]).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODELS[key], path)
    return path


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


class Tracker:
    def __init__(self, num_hands: int = 2, pose: str | None = "pose_lite", model_dir: str = "models"):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        self._mp = mp
        run = vision.RunningMode.VIDEO
        self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(ensure_model("hand", model_dir))),
            running_mode=run, num_hands=num_hands))
        self._pose = None
        if pose:
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(ensure_model(pose, model_dir))),
                running_mode=run, num_poses=1))

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> FrameResult:
        """frame_rgb: uint8 HxWx3 RGB. timestamp_ms must increase on every call."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        out = FrameResult()
        h = self._hands.detect_for_video(img, timestamp_ms)
        for lm, wl, hd in zip(h.hand_landmarks, h.hand_world_landmarks, h.handedness):
            out.hands.append(HandResult(hd[0].category_name, _arr(wl), _arr(lm)))
        if self._pose is not None:
            p = self._pose.detect_for_video(img, timestamp_ms)
            if p.pose_world_landmarks:
                out.pose_world = _arr(p.pose_world_landmarks[0])
                out.pose_image = _arr(p.pose_landmarks[0])
                out.pose_visibility = np.array([q.visibility for q in p.pose_landmarks[0]])
        return out

    def close(self):
        """Release the MediaPipe tasks; call before exit to avoid shutdown-order warnings."""
        for t in (self._hands, self._pose):
            if t is not None:
                t.close()
        self._hands = self._pose = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


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

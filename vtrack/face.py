"""Face recognition and emotion tracking.

Emotion follows the same approach as the reference project (atulapra/Emotion-detection: detect the face,
crop it, classify it with a neural network trained on facial expressions), with stronger models that run
on OpenCV alone (no TensorFlow):
  * face detection and 5 landmarks: YuNet (OpenCV Zoo, face_detection_yunet_2023mar)
  * emotion, default "hsemotion": HSEmotion's EfficientNet-B0 (enet_b0_8_best_vgaf, A. Savchenko,
    github.com/av-savchenko/face-emotion-recognition, Apache 2.0 code), trained on AffectNet. It scores
    eight classes; contempt is dropped, leaving the six cardinal (Ekman) emotions plus neutral.
  * emotion, "opencv": OpenCV Zoo's Progressive Teacher MobileFaceNet (2022july), trained on RAF-DB.
    It was the first default and is kept for comparison (--emotion-model opencv).
  * recognition: SFace (OpenCV Zoo, face_recognition_sface_2021dec), 99.4% on LFW. Faces are matched to
    people you enroll (press e) by cosine similarity; 0.363 or more counts as the same person.
Scores are turned into percentages with softmax; the six emotions and neutral add up to 100%.

Why HSEmotion: on a class-balanced sample of the AffectNet validation set (1048 faces, YuNet boxes, this
code), the OpenCV Zoo model recognized fear in 9% of fear faces (it called most of them surprise) and
averaged 43% over the seven classes; HSEmotion recognized fear in 78% and averaged 69%. AffectNet is
HSEmotion's own benchmark (its validation set was not used for training), so part of that gap is home
advantage, but the fear gap is far larger than that. Its weakest class is neutral (39%: relaxed faces
often read as mild anger or surprise), which per-person calibration (emotion_cal.py) corrects.

Models download on first use. Enrolled faces are stored only on this computer, in the faces/ folder
(128 numbers per sample, no images).
"""
from __future__ import annotations

import math
import ssl
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HF = "https://huggingface.co/opencv"
HSE = "https://raw.githubusercontent.com/av-savchenko/face-emotion-recognition/main/models/affectnet_emotions/onnx"
FACE_MODELS = {
    "yunet": f"{HF}/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "fer": f"{HF}/facial_expression_recognition/resolve/main/facial_expression_recognition_mobilefacenet_2022july.onnx",
    "sface": f"{HF}/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    "hsemotion": f"{HSE}/enet_b0_8_best_vgaf.onnx",
}
# Model output order
EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
# The six cardinal emotions, in display order, with display names
CARDINAL = [("happy", "Happiness"), ("sad", "Sadness"), ("angry", "Anger"), ("fearful", "Fear"),
            ("surprised", "Surprise"), ("disgust", "Disgust")]
SAME_PERSON_COSINE = 0.363
# FER alignment template (112 x 112): right eye, left eye, nose tip, right and left mouth corners
FER_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def ssl_context() -> ssl.SSLContext:
    """Uses certifi when installed; Python from python.org on macOS lacks the system certificates."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url: str, path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name} ...")
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, context=ssl_context()) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(path)
    return path


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 non-reflective similarity (rotation, uniform scale, shift) mapping src points onto dst.

    As in the OpenCV Zoo emotion demo's FaceAlignment (same result): the dst-to-src similarity is fitted
    by least squares, then inverted.
    """
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    n = len(src)
    A = np.zeros((2 * n, 4))
    A[:n] = np.column_stack([dst[:, 0], dst[:, 1], np.ones(n), np.zeros(n)])
    A[n:] = np.column_stack([dst[:, 1], -dst[:, 0], np.zeros(n), np.ones(n)])
    b = np.concatenate([src[:, 0], src[:, 1]])
    sc, ss, tx, ty = np.linalg.lstsq(A, b, rcond=-1)[0]
    Tinv = np.array([[sc, -ss, 0.0], [ss, sc, 0.0], [tx, ty, 1.0]])   # maps dst -> src
    T = np.linalg.inv(Tinv)
    T[:, 2] = [0.0, 0.0, 1.0]
    return T[:, :2].T


@dataclass
class FaceDetection:
    box: np.ndarray            # x, y, w, h in pixels
    landmarks5: np.ndarray     # (5, 2): right eye, left eye, nose tip, right and left mouth corners
    score: float
    row: np.ndarray = field(repr=False, default=None)   # the raw YuNet row, for SFace alignment


class FaceDetector:
    """YuNet. Runs on a frame scaled so its width is at most `max_width` (faster), returns full-size pixels."""

    def __init__(self, model_dir="models", max_width=640, score_thresh=0.7):
        import cv2
        path = download(FACE_MODELS["yunet"], Path(model_dir) / "face_detection_yunet_2023mar.onnx")
        self.det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_thresh, 0.3, 5000)
        self.max_width = max_width

    def detect(self, bgr: np.ndarray) -> list[FaceDetection]:
        import cv2
        h, w = bgr.shape[:2]
        s = min(1.0, self.max_width / w)
        img = cv2.resize(bgr, (int(round(w * s)), int(round(h * s)))) if s < 1.0 else bgr
        self.det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.det.detect(img)
        out = []
        for f in ([] if faces is None else faces):
            row = f.copy()
            row[:14] /= s
            out.append(FaceDetection(row[:4].copy(), row[4:14].reshape(5, 2).copy(), float(row[14]), row))
        return sorted(out, key=lambda d: -d.box[2] * d.box[3])           # largest face first


class EmotionModel:
    """OpenCV Zoo facial expression model (RAF-DB), on the face aligned by its 5 YuNet landmarks."""
    name = "opencv"

    def __init__(self, model_dir="models"):
        import cv2
        path = download(FACE_MODELS["fer"], Path(model_dir) / "facial_expression_recognition_mobilefacenet_2022july.onnx")
        self.net = cv2.dnn.readNet(str(path))

    def aligned(self, bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.warpAffine(bgr, similarity_transform(landmarks5, FER_TEMPLATE), (112, 112))

    def logits(self, bgr: np.ndarray, det: "FaceDetection") -> np.ndarray:
        """The model's seven raw scores in EMOTIONS order (softmax of these gives the probabilities)."""
        import cv2
        img = cv2.cvtColor(self.aligned(bgr, det.landmarks5), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        self.net.setInput(cv2.dnn.blobFromImage(img), "data")
        return self.net.forward("label").reshape(-1).astype(np.float64)

    def probabilities(self, bgr: np.ndarray, det: "FaceDetection") -> np.ndarray:
        """Seven probabilities in EMOTIONS order (sum 1)."""
        return softmax(self.logits(bgr, det))


class HSEmotionModel:
    """HSEmotion EfficientNet-B0 (AffectNet, 8 classes) on the square face crop around the YuNet box.

    Preprocessing as in the hsemotion package: RGB, resized to 224 x 224, ImageNet mean and standard
    deviation. Its classes are anger, contempt, disgust, fear, happiness, neutral, sadness, surprise;
    contempt is dropped.
    """
    name = "hsemotion"
    SIZE = 224
    TO_EMOTIONS = [0, 2, 3, 4, 5, 6, 7]                   # model index for each entry of EMOTIONS
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, model_dir="models"):
        import cv2
        path = download(FACE_MODELS["hsemotion"], Path(model_dir) / "enet_b0_8_best_vgaf.onnx")
        self.net = cv2.dnn.readNetFromONNX(str(path))

    @staticmethod
    def crop(bgr: np.ndarray, det: "FaceDetection") -> np.ndarray:
        """Square crop (side = the box's longer side) centered on the face box, clipped to the image."""
        x, y, w, h = (float(v) for v in det.box[:4])
        cx, cy, s = x + w / 2, y + h / 2, max(w, h)
        X0, Y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
        X1, Y1 = int(min(cx + s / 2, bgr.shape[1])), int(min(cy + s / 2, bgr.shape[0]))
        return bgr[Y0:max(Y1, Y0 + 1), X0:max(X1, X0 + 1)]

    def logits(self, bgr: np.ndarray, det: "FaceDetection") -> np.ndarray:
        import cv2
        rgb = cv2.cvtColor(self.crop(bgr, det), cv2.COLOR_BGR2RGB)
        x = (cv2.resize(rgb, (self.SIZE, self.SIZE)) / 255.0 - self.MEAN) / self.STD
        self.net.setInput(x.transpose(2, 0, 1)[None].astype(np.float32))
        return self.net.forward().reshape(-1)[self.TO_EMOTIONS].astype(np.float64)

    def probabilities(self, bgr: np.ndarray, det: "FaceDetection") -> np.ndarray:
        return softmax(self.logits(bgr, det))


EMOTION_MODELS = {"hsemotion": HSEmotionModel, "opencv": EmotionModel}
DEFAULT_EMOTION_MODEL = "hsemotion"


class FaceRecognizer:
    """SFace embeddings and a small gallery of enrolled people stored in `faces_dir`."""

    def __init__(self, model_dir="models", faces_dir="faces"):
        import cv2
        path = download(FACE_MODELS["sface"], Path(model_dir) / "face_recognition_sface_2021dec.onnx")
        self.rec = cv2.FaceRecognizerSF.create(str(path), "")
        self.faces_dir = Path(faces_dir)
        self.gallery: dict[str, np.ndarray] = {}
        if self.faces_dir.exists():
            for f in sorted(self.faces_dir.glob("*.npy")):
                self.gallery[f.stem] = np.load(f)

    def embedding(self, bgr: np.ndarray, det: FaceDetection) -> np.ndarray:
        crop = self.rec.alignCrop(bgr, det.row.reshape(1, -1).astype(np.float32))
        v = self.rec.feature(crop).reshape(-1).astype(np.float64)
        return v / max(np.linalg.norm(v), 1e-12)

    def enroll(self, name: str, emb: np.ndarray) -> int:
        """Add one sample for `name` and save it; returns the number of samples for that person."""
        samples = np.vstack([self.gallery[name], emb]) if name in self.gallery else emb[None, :]
        self.gallery[name] = samples
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.faces_dir / f"{name}.npy", samples)
        return len(samples)

    def identify(self, emb: np.ndarray) -> tuple[str | None, float]:
        """(name, cosine similarity) of the best match, name None when below the same-person threshold."""
        best, score = None, -1.0
        for name, samples in self.gallery.items():
            s = float(np.max(samples @ emb))
            if s > score:
                best, score = name, s
        return (best if score >= SAME_PERSON_COSINE else None), score


class EmotionSmoother:
    """Exponential smoothing of the probability vector with time constant `tau_s` (frame-rate independent)."""

    def __init__(self, tau_s: float = 0.35):
        self.tau, self.p, self.t = tau_s, None, None

    def __call__(self, p: np.ndarray, t: float) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if self.p is None or t <= self.t:
            self.p = p.copy()
        else:
            a = 1.0 - math.exp(-(t - self.t) / self.tau)
            self.p = self.p + a * (p - self.p)
        self.t = t
        return self.p.copy()

    def reset(self):
        self.p = self.t = None


def cardinal_percentages(emotions: dict) -> list[tuple[str, float]]:
    """[(display name, percent)] for the six cardinal emotions, highest first."""
    return sorted(((label, 100.0 * emotions[key]) for key, label in CARDINAL), key=lambda kv: -kv[1])


class EmotionTask:
    """Worker function for the live app: face detection, emotion and, once someone is enrolled, recognition.

    Called with a frame (attributes t and bgr); returns None without a face, else a dict with the face box,
    the model's seven raw emotion scores and their (uncalibrated) probabilities, the recognized name (or
    None) and its match score. Recognition runs at
    most every `recognize_every_s` seconds; set `enroll_request` to a name to add the current face.
    """

    def __init__(self, model_dir: str, faces_dir: str, recognize_every_s: float = 0.5,
                 model: str = DEFAULT_EMOTION_MODEL):
        self.model_dir, self.faces_dir = model_dir, faces_dir
        self.detector, self.emotion = FaceDetector(model_dir), EMOTION_MODELS[model](model_dir)
        self.recognizer = None
        if Path(faces_dir).exists() and any(Path(faces_dir).glob("*.npy")):
            self._load_recognizer()
        self.every = recognize_every_s
        self._last_id = (None, -1.0, -1e9)
        self.enroll_request: str | None = None
        self.message = ""

    def _load_recognizer(self):
        self.recognizer = FaceRecognizer(self.model_dir, self.faces_dir)

    def __call__(self, frame):
        dets = self.detector.detect(frame.bgr)
        if not dets:
            self._last_id = (None, -1.0, -1e9)        # a new face is identified from scratch
            return None
        d = dets[0]
        logits = self.emotion.logits(frame.bgr, d)
        name, score, t_id = self._last_id
        if self.enroll_request:
            who, self.enroll_request = self.enroll_request, None
            try:
                if self.recognizer is None:
                    self._load_recognizer()              # downloads the recognition model on first use
                n = self.recognizer.enroll(who, self.recognizer.embedding(frame.bgr, d))
                self.message = f"enrolled {who} ({n} sample{'s' if n > 1 else ''})"
            except Exception as exc:
                self.message = f"enrolling failed: {type(exc).__name__}: {exc}"[:120]
            t_id = -1e9
        if self.recognizer is not None and self.recognizer.gallery and frame.t - t_id >= self.every:
            name, score = self.recognizer.identify(self.recognizer.embedding(frame.bgr, d))
            t_id = frame.t
            self._last_id = (name, score, t_id)
        return dict(box=d.box, logits=logits, probs=softmax(logits), identity=name, match=score,
                    enrolled=bool(self.recognizer and self.recognizer.gallery))

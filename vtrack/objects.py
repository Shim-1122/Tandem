"""Open-vocabulary object detection: find objects named in plain English.

The detector is YOLO-World v2 (medium, or small for slower computers), exported to ONNX with the class names' text embeddings as an input, so the
list of objects is data (vtrack/data/object_vocab.npz, built from object_prompts.json), not part of the model. A
class can have several phrasings ("teddy bear", "stuffed animal"); its score is the best of them.

    det = ObjectDetector(model_path, vocab)
    for d in det.detect(bgr):                       # or det.detect(bgr, roi=(x0, y0, x1, y1)) to zoom in
        print(d.name, d.score, d.box)

Runs with onnxruntime on the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

STRIDES = (8, 16, 32)
REG_MAX = 16
DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS = {"medium": "yoloworld_v2m", "small": "yoloworld_v2s"}     # more accurate / about 3 times faster


@dataclass
class Detection:
    cls: int                        # index into the vocabulary's classes
    name: str
    score: float                    # 0..1
    box: np.ndarray                 # x0, y0, x1, y1 in image pixels
    class_scores: np.ndarray | None = None      # every class's score for this box
    attributes: dict = field(default_factory=dict)   # attribute name: score (for example the bottle's material)


@dataclass
class Vocabulary:
    """Classes and their phrasings. A phrasing with class -1 only scores an attribute (prompt_attr)."""
    names: list                     # class names, in order
    embeddings: np.ndarray          # (P, 512) CLIP text embeddings, one per phrasing
    prompt_class: np.ndarray        # (P,) the class of each phrasing, or -1
    report: np.ndarray              # (K,) bool: False for helper classes (hands, cups...) that are detected only so
                                    # they are not mistaken for a tracked object
    prompts: list = field(default_factory=list)
    attributes: list = field(default_factory=list)                    # attribute names
    prompt_attr: np.ndarray | None = None                             # (P,) attribute of each phrasing, or -1

    def __post_init__(self):
        self.prompt_class = np.asarray(self.prompt_class, int)
        self.report = np.asarray(self.report, bool)
        if self.prompt_attr is None:
            self.prompt_attr = np.full(len(self.prompt_class), -1)
        self.prompt_attr = np.asarray(self.prompt_attr, int)

    @staticmethod
    def load(path=DATA_DIR / "object_vocab.npz") -> "Vocabulary":
        z = np.load(path, allow_pickle=False)
        return Vocabulary([str(n) for n in z["names"]], z["embeddings"].astype(np.float32), z["prompt_class"],
                          z["report"], [str(p) for p in z["prompts"]], [str(a) for a in z["attributes"]],
                          z["prompt_attr"])

    @staticmethod
    def simple(names, embeddings, report=None) -> "Vocabulary":
        """One phrasing per class."""
        k = len(names)
        return Vocabulary(list(names), np.asarray(embeddings, np.float32), np.arange(k),
                          np.ones(k, bool) if report is None else report, list(names))

    def save(self, path):
        np.savez_compressed(path, names=np.array(self.names), embeddings=self.embeddings.astype(np.float16),
                            prompt_class=self.prompt_class.astype(np.int16), report=self.report,
                            prompts=np.array(self.prompts), attributes=np.array(self.attributes, dtype=str),
                            prompt_attr=self.prompt_attr.astype(np.int16))

    def index(self, name: str) -> int:
        return self.names.index(name)

    def class_scores(self, prompt_scores: np.ndarray) -> np.ndarray:
        """(P, N) phrasing scores to (K, N) class scores: the best phrasing of each class."""
        out = np.zeros((len(self.names), prompt_scores.shape[1]), prompt_scores.dtype)
        for k in range(len(self.names)):
            out[k] = prompt_scores[self.prompt_class == k].max(axis=0)
        return out

    def attribute_scores(self, prompt_scores: np.ndarray) -> np.ndarray:
        out = np.zeros((len(self.attributes), prompt_scores.shape[1]), prompt_scores.dtype)
        for a in range(len(self.attributes)):
            out[a] = prompt_scores[self.prompt_attr == a].max(axis=0)
        return out


def unpack_weights(parts: list, manifest: list, total: int) -> bytes:
    """The model's fp32 weight file from its packed form (fp16 copies of the large tensors, the rest as is)."""
    buf = bytearray(total)
    for e in manifest:
        a = parts[e.get("part", 0)][f"t{e['key']}"]
        if a.size != int(np.prod(e["shape"])):
            raise ValueError(f"packed weight {e['name']} has the wrong size")
        b = np.ascontiguousarray(a, dtype=np.dtype(e["dtype"])).tobytes()
        if len(b) != e["length"]:
            raise ValueError(f"packed weight {e['name']} has the wrong size")
        buf[e["offset"]:e["offset"] + e["length"]] = b
    return bytes(buf)


def ensure_model(name: str = "medium", model_dir="models") -> Path:
    """Unpack a detector from vtrack/data into model_dir (once, about a second); returns the .onnx path.

    The repository keeps the weights in half precision, split into files under 50 MB; the runtime needs them
    in full precision next to the graph file.
    """
    import hashlib
    import json
    import shutil
    stem = MODELS.get(name, name)
    out_dir = Path(model_dir)
    onnx_path, weights = out_dir / f"{stem}.onnx", out_dir / f"{stem}.weights"
    stamp = out_dir / f"{stem}.sha256"
    first = np.load(DATA_DIR / f"{stem}.npz", allow_pickle=False)
    info = json.loads(bytes(first["manifest"]).decode())
    if onnx_path.exists() and weights.exists() and stamp.exists() and stamp.read_text().strip() == info["sha256"]:
        return onnx_path
    parts = [first] + [np.load(DATA_DIR / f"{stem}.part{i + 1}.npz", allow_pickle=False) for i in range(1, info["parts"])]
    data = unpack_weights(parts, info["tensors"], info["total"])
    if hashlib.sha256(data).hexdigest() != info["sha256"]:
        raise ValueError(f"{stem}: unpacked weights do not match their checksum")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = weights.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(weights)
    shutil.copyfile(DATA_DIR / f"{stem}.onnx", onnx_path)
    stamp.write_text(info["sha256"])
    return onnx_path


def letterbox(bgr: np.ndarray, size: tuple):
    """Resize keeping the aspect ratio into `size` (h, w), padded with grey; returns image, scale, offset."""
    h, w = bgr.shape[:2]
    H, W = size
    s = min(H / h, W / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    out = np.full((H, W, 3), 114, np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
    return out, s, np.array([x0, y0], float)


def input_size_for(h: int, w: int, long_side: int) -> tuple:
    """Network input (h, w), multiples of 32, long side `long_side`, keeping the image's aspect ratio."""
    s = long_side / max(h, w)
    return (max(32, int(np.ceil(h * s / 32)) * 32), max(32, int(np.ceil(w * s / 32)) * 32))


def anchors_for(H: int, W: int):
    pts, strides = [], []
    for st in STRIDES:
        h, w = H // st, W // st
        yy, xx = np.mgrid[0:h, 0:w]
        pts.append(np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5]) * st)
        strides.append(np.full(h * w, st, float))
    return np.vstack(pts), np.concatenate(strides)


def decode(box_logits: np.ndarray, score_logits: np.ndarray, H: int, W: int, conf: float):
    """Raw outputs (64, A) and (P, A) to boxes (N, 4) in network pixels and phrasing scores (P, N) for the anchors
    where any phrasing scores at least `conf`."""
    keep = np.nonzero(score_logits.max(axis=0) >= np.log(conf / (1.0 - conf)))[0]
    scores = 1.0 / (1.0 + np.exp(-score_logits[:, keep]))
    if len(keep) == 0:
        return np.zeros((0, 4)), scores
    pts, strides = anchors_for(H, W)
    d = box_logits[:, keep].reshape(4, REG_MAX, -1)
    d = np.exp(d - d.max(axis=1, keepdims=True))
    d = (d / d.sum(axis=1, keepdims=True) * np.arange(REG_MAX)[None, :, None]).sum(axis=1)   # (4, N): l, t, r, b
    p, st = pts[keep], strides[keep]
    boxes = np.column_stack([p[:, 0] - d[0] * st, p[:, 1] - d[1] * st, p[:, 0] + d[2] * st, p[:, 1] + d[3] * st])
    return boxes, scores


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list:
    """Greedy non-maximum suppression over all classes together (one label per object)."""
    order = list(np.argsort(-scores))
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = np.array(order, int)
        if not len(rest):
            break
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        o = inter / (area[i] + area[rest] - inter + 1e-9)
        order = [j for j, oj in zip(rest, o) if oj < iou]
    return keep


class ObjectDetector:
    def __init__(self, model_path, vocab: Vocabulary, long_side: int = 640, conf: float = 0.2, iou: float = 0.5,
                 threads: int = 0):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.vocab, self.long_side, self.conf, self.iou = vocab, long_side, conf, iou
        self.text = vocab.embeddings[None].astype(np.float32)

    def raw(self, img: np.ndarray):
        """Network outputs for a letterboxed BGR image whose sides are multiples of 32."""
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        b, s = self.session.run(None, {"images": x, "text": self.text})
        return b[0], s[0]

    def detect(self, bgr: np.ndarray, roi=None, long_side: int | None = None, conf: float | None = None) -> list:
        """Objects in the image, or only inside roi = (x0, y0, x1, y1) (zoomed to the network's input size)."""
        off = np.zeros(2)
        img = bgr
        if roi is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in roi)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(bgr.shape[1], x1), min(bgr.shape[0], y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                return []
            img, off = bgr[y0:y1, x0:x1], np.array([x0, y0], float)
        H, W = input_size_for(*img.shape[:2], long_side or self.long_side)
        lb, s, pad = letterbox(img, (H, W))
        bl, sl = self.raw(lb)
        c = self.conf if conf is None else conf
        boxes, ps = decode(bl, sl, H, W, c)
        cs, at = self.vocab.class_scores(ps), self.vocab.attribute_scores(ps)
        cls = cs.argmax(axis=0)
        best = cs[cls, np.arange(cs.shape[1])]
        sel = np.nonzero(best >= c)[0]
        out = []
        for i in (sel[j] for j in nms(boxes[sel], best[sel], self.iou)):
            b = (boxes[i] - np.tile(pad, 2)) / s + np.tile(off, 2)
            b[[0, 2]] = np.clip(b[[0, 2]], 0, bgr.shape[1] - 1)
            b[[1, 3]] = np.clip(b[[1, 3]], 0, bgr.shape[0] - 1)
            k = int(cls[i])
            out.append(Detection(k, self.vocab.names[k], float(best[i]), b, cs[:, i].copy(),
                                 {a: float(at[j, i]) for j, a in enumerate(self.vocab.attributes)}))
        return out


class ObjectTask:
    """The object model's worker function: the whole frame every `full_every_s` seconds (to find new objects
    anywhere), and in between only zoomed regions around the hands (`rois`, set by the app), where a held object
    is seen large enough to recognize."""

    def __init__(self, model: str = "medium", model_dir="models", full_every_s: float = 0.6, long_side: int = 640,
                 roi_long_side: int = 320, threads: int = 0):
        self.det = ObjectDetector(ensure_model(model, model_dir), Vocabulary.load(), long_side=long_side,
                                  threads=threads)
        self.full_every_s, self.roi_long_side = full_every_s, roi_long_side
        self.rois: list = []                        # [(side, (x0, y0, x1, y1))] camera pixels
        self._t_full = -1e9

    def __call__(self, frame) -> dict:
        rois = [r for _, r in list(self.rois) if r is not None]
        if not rois or frame.t - self._t_full >= self.full_every_s:
            self._t_full = frame.t
            return {"kind": "full", "runs": [(None, self.det.detect(frame.bgr))]}
        return {"kind": "hands", "runs": [(r, self.det.detect(frame.bgr, roi=r, long_side=self.roi_long_side))
                                          for r in rois]}

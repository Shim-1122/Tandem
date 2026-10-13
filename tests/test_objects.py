"""Object detector: output decoding, phrasings and attributes, zoomed regions, the packed models, the vocabulary."""
import json

import numpy as np
import pytest

pytest.importorskip("cv2")

from vtrack.clip_text import ClipTokenizer, bytes_to_unicode
from vtrack.grip import OBJECTS
from vtrack.objects import (DATA_DIR, MODELS, REG_MAX, Detection, ObjectDetector, ObjectTask, Vocabulary, anchors_for,
                            decode, ensure_model, input_size_for, letterbox, nms, unpack_weights)


def raw_outputs(H, W, objects, n_prompts):
    """Network outputs with one confident anchor per object: (cx, cy, l, t, r, b in network pixels, prompt, logit)."""
    pts, strides = anchors_for(H, W)
    A = len(pts)
    box = np.full((4 * REG_MAX, A), -20.0, np.float32)
    box[0::REG_MAX] = 0.0                              # every side at bin 0 unless set below
    score = np.full((n_prompts, A), -12.0, np.float32)
    for cx, cy, l, t, r, b, prompt, logit in objects:
        i = int(np.argmin(np.where(strides == 8, np.hypot(pts[:, 0] - cx, pts[:, 1] - cy), np.inf)))
        for side, dist in enumerate((l, t, r, b)):
            box[side * REG_MAX:(side + 1) * REG_MAX, i] = -20.0
            box[side * REG_MAX + int(round(dist / 8)), i] = 20.0
        score[prompt, i] = logit
    return box, score, pts


def test_anchor_grid_and_input_sizes():
    pts, st = anchors_for(64, 96)
    assert len(pts) == 8 * 12 + 4 * 6 + 2 * 3 and set(st) == {8.0, 16.0, 32.0}
    assert pts[0] == pytest.approx([4, 4]) and pts[-1] == pytest.approx([96 - 16, 64 - 16])
    assert input_size_for(720, 1280, 640) == (384, 640) and input_size_for(300, 300, 320) == (320, 320)


def test_decode_recovers_boxes_and_scores():
    box, score, pts = raw_outputs(128, 128, [(60, 60, 16, 24, 32, 8, 1, 3.0)], 3)
    boxes, ps = decode(box, score, 128, 128, 0.2)
    assert boxes.shape == (1, 4) and ps.shape == (3, 1)
    i = int(np.argmin(np.hypot(pts[:, 0] - 60, pts[:, 1] - 60)))
    c = pts[i]
    assert boxes[0] == pytest.approx([c[0] - 16, c[1] - 24, c[0] + 32, c[1] + 8], abs=1e-3)
    assert ps[1, 0] == pytest.approx(1 / (1 + np.exp(-3.0))) and ps[0, 0] < 1e-4


def test_vocabulary_takes_the_best_phrasing_and_scores_attributes():
    v = Vocabulary(["bottle", "cup"], np.eye(5, 512, dtype=np.float32), np.array([0, 0, 1, -1, -1]), [True, False],
                   ["a", "b", "c", "m", "p"], ["metal", "plastic"], np.array([-1, -1, -1, 0, 1]))
    ps = np.array([[0.1, 0.9], [0.6, 0.2], [0.3, 0.1], [0.8, 0.0], [0.2, 0.7]])
    assert v.class_scores(ps) == pytest.approx(np.array([[0.6, 0.9], [0.3, 0.1]]))
    assert v.attribute_scores(ps) == pytest.approx(np.array([[0.8, 0.0], [0.2, 0.7]]))


def test_nms_keeps_one_box_per_object():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float)
    assert nms(boxes, np.array([0.5, 0.9, 0.3]), 0.5) == [1, 2]


class _Session:
    def __init__(self, objects, n_prompts):
        self.objects, self.n = objects, n_prompts
        self.shapes = []

    def run(self, _, feeds):
        H, W = feeds["images"].shape[2:]
        self.shapes.append((H, W))
        box, score, _ = raw_outputs(H, W, self.objects, self.n)
        return [box[None], score[None]]


def _detector(objects, vocab):
    det = object.__new__(ObjectDetector)
    det.session, det.vocab, det.long_side, det.conf, det.iou = _Session(objects, len(vocab.prompts)), vocab, 640, 0.2, 0.5
    det.text = vocab.embeddings[None]
    return det


def test_detect_maps_boxes_back_to_the_image_and_into_a_zoomed_region():
    v = Vocabulary.simple(["ball", "hand"], np.eye(2, 512, dtype=np.float32), [True, False])
    # 1280x720 frame -> 640x384 input (scale 0.5, 12 px of padding at the top)
    det = _detector([(164, 196, 40, 40, 40, 40, 0, 4.0)], v)          # on an anchor center
    d = det.detect(np.zeros((720, 1280, 3), np.uint8))
    assert det.session.shapes[-1] == (384, 640) and len(d) == 1 and d[0].name == "ball"
    cx, cy = (d[0].box[:2] + d[0].box[2:]) / 2
    assert (cx, cy) == pytest.approx((164 * 2, (196 - 12) * 2), abs=1e-6) and d[0].box[2] - d[0].box[0] == pytest.approx(160)
    # a 300x300 region at (500, 200), zoomed to 320x320: network pixels / (320/300) + offset
    d = det.detect(np.zeros((720, 1280, 3), np.uint8), roi=(500, 200, 800, 500), long_side=320)
    assert det.session.shapes[-1] == (320, 320)
    s = 320 / 300
    assert d[0].box == pytest.approx([500 + 124 / s, 200 + 156 / s, 500 + 204 / s, 200 + 236 / s], abs=1e-6)
    assert det.detect(np.zeros((720, 1280, 3), np.uint8), roi=(10, 10, 12, 500)) == []


def test_letterbox_keeps_the_aspect_ratio():
    img, s, pad = letterbox(np.full((100, 200, 3), 255, np.uint8), (128, 128))
    assert s == pytest.approx(0.64) and pad == pytest.approx([0, 32]) and img[0, 0, 0] == 114 and img[64, 64, 0] == 255


def test_object_task_searches_everything_then_around_the_hands():
    v = Vocabulary.simple(["ball"], np.eye(1, 512, dtype=np.float32))
    task = object.__new__(ObjectTask)
    task.det, task.full_every_s, task.roi_long_side, task.rois, task._t_full = _detector([], v), 0.5, 320, [], -1e9

    class F:
        def __init__(self, t):
            self.t, self.bgr = t, np.zeros((480, 640, 3), np.uint8)
    assert task(F(0.0))["kind"] == "full"
    task.rois = [("right", (100, 100, 300, 300)), ("left", None)]
    out = task(F(0.1))
    assert out["kind"] == "hands" and [r for r, _ in out["runs"]] == [(100, 100, 300, 300)]
    assert task(F(0.6))["kind"] == "full"


def test_unpack_weights_rebuilds_full_precision_bytes():
    a, b = np.arange(6, dtype=np.float32).reshape(2, 3), np.array([7, 8], np.int64)
    parts = [{"t0": a.astype(np.float16)}, {"t1": b}]
    manifest = [dict(name="a", key=0, part=0, offset=16, length=24, dtype="float32", shape=[2, 3]),
                dict(name="b", key=1, part=1, offset=0, length=16, dtype="int64", shape=[2])]
    data = unpack_weights(parts, manifest, 40)
    assert np.frombuffer(data[16:40], np.float32).tolist() == a.ravel().tolist()
    assert np.frombuffer(data[:16], np.int64).tolist() == [7, 8]
    manifest[0]["shape"] = [2, 2]
    with pytest.raises(ValueError, match="wrong size"):
        unpack_weights(parts, manifest, 40)


def test_packed_models_unpack_and_run(tmp_path):
    """The small model from the repository, unpacked once and reused, gives outputs for any size and vocabulary."""
    path = ensure_model("small", tmp_path)
    weights = tmp_path / f"{MODELS['small']}.weights"
    stamp = weights.stat().st_mtime_ns
    assert ensure_model("small", tmp_path) == path and weights.stat().st_mtime_ns == stamp     # not rebuilt
    vocab = Vocabulary.load()
    det = ObjectDetector(path, vocab)
    b, s = det.raw(np.full((320, 480, 3), 127, np.uint8))
    A = len(anchors_for(320, 480)[0])
    assert b.shape == (64, A) and s.shape == (len(vocab.prompts), A)
    assert det.detect(np.full((240, 320, 3), 127, np.uint8)) == [] or all(isinstance(d, Detection) for d in det.detect(
        np.full((240, 320, 3), 127, np.uint8)))
    for name in MODELS.values():                       # every packed file is under GitHub's 50 MB recommendation
        for f in DATA_DIR.glob(f"{name}*"):
            assert f.stat().st_size < 50e6


def test_vocabulary_covers_every_requested_object():
    v = Vocabulary.load()
    spec = json.loads((DATA_DIR / "object_prompts.json").read_text())
    tracked = [n for n, r in zip(v.names, v.report) if r]
    assert tracked == list(spec["track"]) and v.attributes == ["metal", "plastic"]
    for n in tracked:
        assert n in OBJECTS or n == "water bottle"
    assert np.allclose(np.linalg.norm(v.embeddings, axis=1), 1.0, atol=2e-3)
    assert len(v.prompts) == len(v.embeddings) == len(v.prompt_class)
    assert {"hand", "person"} <= set(v.names) - set(tracked)


def test_clip_bpe_on_a_tiny_vocabulary(tmp_path):
    bu = bytes_to_unicode()
    assert len(bu) == 256 and len(set(bu.values())) == 256 and bu[ord("a")] == "a"
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1, "p": 2, "e": 3, "n</w>": 4, "pe": 5, "pen</w>": 6, "a</w>": 7}
    merges = ["p e", "pe n</w>"]
    f = tmp_path / "tok.json"
    f.write_text(json.dumps({"model": {"vocab": vocab, "merges": merges}}))
    tok = ClipTokenizer(f)
    assert tok.encode("  PEN  ") == [0, 6, 1] and tok.encode("a pen") == [0, 7, 6, 1]

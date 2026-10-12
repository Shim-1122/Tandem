"""Face module logic that runs without the network or model files."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from vtrack.face import (CARDINAL, EMOTIONS, FER_TEMPLATE, SAME_PERSON_COSINE, EmotionSmoother, EmotionTask,
                         FaceDetection, FaceRecognizer, cardinal_percentages, download, similarity_transform, softmax)


def test_softmax_sums_to_one_and_ignores_shifts_and_large_values():
    z = np.array([1.0, 2.0, 3.0, -1.0, 0.0, 0.5, 2.5])
    p = softmax(z)
    assert p.sum() == pytest.approx(1.0) and np.all(p > 0)
    assert softmax(z + 1000.0) == pytest.approx(p)
    assert np.argmax(p) == 2


def _similarity(scale, deg, shift):
    a = np.radians(deg)
    return scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]), np.asarray(shift, float)


def test_alignment_maps_a_face_exactly_onto_the_template():
    R, t = _similarity(2.7, 23.0, (310.0, 140.0))
    src = FER_TEMPLATE @ R.T + t                                   # a face 2.7x larger, rotated and shifted
    M = similarity_transform(src, FER_TEMPLATE)
    assert np.hstack([src, np.ones((5, 1))]) @ M.T == pytest.approx(FER_TEMPLATE, abs=1e-9)


def _umeyama(src, dst):
    """Least-squares similarity (no reflection), Umeyama 1991; independent reference."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return np.hstack([s * R, (md - s * R @ ms)[:, None]])


def test_alignment_fits_template_to_image_by_least_squares_then_inverts():
    """With noisy landmarks the fit direction matters; like the emotion model's reference code, the
    template-to-image similarity is fitted (errors measured in image pixels) and then inverted."""
    rng = np.random.default_rng(3)
    R, t = _similarity(1.8, -12.0, (200.0, 90.0))
    src = FER_TEMPLATE @ R.T + t + rng.normal(scale=3.0, size=(5, 2))
    fwd = np.vstack([_umeyama(FER_TEMPLATE, src), [0.0, 0.0, 1.0]])     # template -> image
    assert similarity_transform(src, FER_TEMPLATE) == pytest.approx(np.linalg.inv(fwd)[:2], abs=1e-6)


def test_cardinal_percentages_are_the_six_emotions_sorted():
    p = dict(zip(EMOTIONS, [0.05, 0.01, 0.02, 0.6, 0.2, 0.1, 0.02]))
    out = cardinal_percentages(p)
    assert [name for name, _ in out] == ["Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust"]
    assert out[0][1] == pytest.approx(60.0)
    assert len(out) == 6 and {k for k, _ in CARDINAL} == set(EMOTIONS) - {"neutral"}
    assert sum(v for _, v in out) + 100 * p["neutral"] == pytest.approx(100.0)


def test_emotion_smoother_time_constant_and_reset():
    s = EmotionSmoother(tau_s=0.5)
    a, b = np.eye(7)[3], np.eye(7)[0]
    assert s(a, 1.0) == pytest.approx(a)                          # first sample passes through
    out = s(b, 1.5)                                               # one time constant later
    assert out[0] == pytest.approx(1 - np.exp(-1))
    assert out.sum() == pytest.approx(1.0)
    assert s(b, 1.5) == pytest.approx(b)                          # same time: restart, not divide by zero
    s.reset()
    assert s(a, 0.1) == pytest.approx(a)


def _recognizer(tmp_path):
    r = object.__new__(FaceRecognizer)                            # gallery logic only, no model file
    r.faces_dir, r.gallery = tmp_path / "faces", {}
    return r


def test_enroll_saves_samples_and_identify_uses_the_threshold(tmp_path):
    r = _recognizer(tmp_path)
    e1 = np.zeros(128); e1[0] = 1.0
    e2 = np.zeros(128); e2[:2] = (0.8, 0.6)
    assert r.enroll("shyam", e1) == 1 and r.enroll("shyam", e2) == 2
    assert np.load(tmp_path / "faces" / "shyam.npy").shape == (2, 128)
    assert r.identify(e2) == ("shyam", pytest.approx(1.0))       # best of the samples
    other = np.zeros(128); other[5] = 1.0
    assert r.identify(other) == (None, pytest.approx(0.0))
    just_below = np.zeros(128)
    just_below[0], just_below[7] = SAME_PERSON_COSINE - 0.01, np.sqrt(1 - (SAME_PERSON_COSINE - 0.01) ** 2)
    assert r.identify(just_below)[0] is None
    # a new recognizer finds the saved gallery
    r2 = _recognizer(tmp_path)
    r2.gallery = {f.stem: np.load(f) for f in (tmp_path / "faces").glob("*.npy")}
    assert r2.identify(e1)[0] == "shyam"


def test_download_copies_once_without_leaving_a_partial_file(tmp_path):
    src = tmp_path / "model.onnx"
    src.write_bytes(b"model-bytes")
    dst = tmp_path / "models" / "copy.onnx"
    download(src.as_uri(), dst)
    assert dst.read_bytes() == b"model-bytes"
    assert not list(dst.parent.glob("*.part"))
    src.write_bytes(b"changed")
    download(src.as_uri(), dst)                                   # already present: not fetched again
    assert dst.read_bytes() == b"model-bytes"


class _Detector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, bgr):
        return self.faces


class _Emotion:
    def logits(self, bgr, landmarks5):
        return np.eye(7)[3] * 30.0


class _Recognizer:
    def __init__(self):
        self.gallery, self.calls = {}, 0

    def embedding(self, bgr, det):
        self.calls += 1
        return np.ones(128) / np.sqrt(128)

    def enroll(self, name, emb):
        self.gallery[name] = emb[None]
        return 1

    def identify(self, emb):
        return ("me", 0.97) if self.gallery else (None, -1.0)


def _task(faces, recognizer=None):
    t = object.__new__(EmotionTask)
    t.model_dir, t.faces_dir = "models", "faces"
    t.detector, t.emotion, t.recognizer = _Detector(faces), _Emotion(), recognizer
    t.every, t._last_id, t.enroll_request, t.message = 0.5, (None, -1.0, -1e9), None, ""
    return t


class _Frame:
    def __init__(self, t):
        self.t, self.bgr = t, np.zeros((10, 10, 3), np.uint8)


FACE = FaceDetection(np.array([1.0, 2.0, 3.0, 4.0]), np.zeros((5, 2)), 0.9, np.zeros(15, np.float32))


def test_emotion_task_enrolls_then_recognizes_at_a_limited_rate():
    rec = _Recognizer()
    t = _task([FACE], rec)
    out = t(_Frame(0.0))
    assert out["identity"] is None and not out["enrolled"] and out["probs"][3] == pytest.approx(1.0)
    assert out["logits"][3] == 30.0
    t.enroll_request = "me"
    out = t(_Frame(0.1))
    assert t.message == "enrolled me (1 sample)" and t.enroll_request is None
    assert out["identity"] == "me" and out["enrolled"]
    calls = rec.calls
    t(_Frame(0.2))                                               # within 0.5 s: identity reused
    assert rec.calls == calls
    t(_Frame(0.7))
    assert rec.calls == calls + 1
    t.detector.faces = []
    assert t(_Frame(0.8)) is None and t._last_id[0] is None     # face lost: identity forgotten


def test_emotion_task_reports_a_failed_enrollment():
    t = _task([FACE])

    def fail():
        raise OSError("no network")

    t._load_recognizer = fail
    t.enroll_request = "me"
    out = t(_Frame(0.0))
    assert t.message.startswith("enrolling failed: OSError") and out["identity"] is None


def test_emotion_task_waits_for_a_face_before_enrolling():
    t = _task([], _Recognizer())
    t.enroll_request = "me"
    assert t(_Frame(0.0)) is None and t.enroll_request == "me"

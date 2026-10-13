import json

import numpy as np
import pytest

from vtrack.emotion_cal import (CAL_ORDER, CalibrationSession, EmotionCalibration, calibrate, calibration_path,
                                fit_scale_bias, load_all)
from vtrack.face import EMOTIONS, softmax


def test_fit_recovers_a_known_scale_and_bias():
    rng = np.random.default_rng(0)
    Z = rng.normal(scale=2.0, size=(20000, 7))
    s_true, b_true = 1.6, np.array([0.5, -1.0, 0.2, 1.0, -0.3, 0.0, -0.4])
    P = np.array([softmax(s_true * z + b_true) for z in Z])
    Y = np.array([rng.choice(7, p=p) for p in P])
    s, b = fit_scale_bias(Z, Y, reg=1e-6)
    assert s == pytest.approx(s_true, abs=0.05)
    assert b - b.mean() == pytest.approx(b_true - b_true.mean(), abs=0.06)    # bias is defined up to a constant


def biased_person(rng, n=20, bias_toward="sad", strength=4.0):
    """Raw scores for one person whose face the model reads as `bias_toward` whatever the expression."""
    Z, Y, T = [], [], []
    for i, e in enumerate(CAL_ORDER):
        c = EMOTIONS.index(e)
        for j in range(n):
            z = rng.normal(scale=1.0, size=7)
            z[c] += 3.0
            z[EMOTIONS.index(bias_toward)] += strength
            Z.append(z), Y.append(c), T.append(i * 10 + j * 0.1)
    return np.array(Z), np.array(Y), np.array(T)


def test_calibration_fixes_a_face_the_model_misreads():
    rng = np.random.default_rng(1)
    Z, Y, T = biased_person(rng)
    cal = calibrate(Z, Y, T)
    assert cal.accuracy_before < 0.4 and cal.accuracy_after > 0.8 and cal.frames == len(Y)
    # on new frames from the same person
    Z2, Y2, _ = biased_person(np.random.default_rng(2))
    raw = np.mean(np.argmax(Z2, axis=1) == Y2)
    fixed = np.mean([np.argmax(cal.probabilities(z)) == y for z, y in zip(Z2, Y2)])
    assert raw < 0.4 and fixed > 0.8
    assert "held-out accuracy" in cal.summary()


def test_calibration_leaves_an_unbiased_face_about_the_same():
    rng = np.random.default_rng(3)
    Z, Y, T = biased_person(rng, strength=0.0)
    cal = calibrate(Z, Y, T)
    assert cal.accuracy_after >= cal.accuracy_before - 0.05
    assert cal.scale > 0.3 and np.max(np.abs(cal.bias)) < 3


def test_session_timing_prompts_and_capture():
    s = CalibrationSession("shyam", t0=100.0, prepare_s=1.0, capture_s=2.0)
    assert s.duration == pytest.approx(21.0)
    assert s.phase(100.5)[:2] == (0, "prepare") and "NEUTRAL" in s.prompt(100.5) and "1/7" in s.prompt(100.5)
    assert s.phase(101.5)[:2] == (0, "capture") and "recording" in s.prompt(101.5)
    assert s.phase(103.5)[:2] == (1, "prepare") and "HAPPY" in s.prompt(103.5)
    assert not s.add(np.zeros(7), 100.2)                     # getting ready: not recorded
    assert s.add(np.zeros(7), 101.2) and s.add(np.zeros(7), 102.9)
    assert s.counts()["neutral"] == 2 and s.counts()["happy"] == 0
    assert not s.finished(120.9) and s.finished(121.0) and s.phase(130)[1] == "done"
    with pytest.raises(ValueError, match="happy"):
        s.result()


def test_session_end_to_end_produces_a_calibration():
    rng = np.random.default_rng(4)
    s = CalibrationSession("shyam", t0=0.0, prepare_s=1.0, capture_s=2.0)
    t = 0.0
    while not s.finished(t):
        i, phase, _ = s.phase(t)
        if phase == "capture":
            z = rng.normal(size=7)
            z[EMOTIONS.index(CAL_ORDER[i])] += 3.0
            z[EMOTIONS.index("angry")] += 3.5                    # this person reads angry
            s.add(z, t)
        t += 0.1
    cal = s.result()
    assert cal.accuracy_after > cal.accuracy_before + 0.3


def test_save_load_and_load_all(tmp_path):
    cal = EmotionCalibration(1.3, [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2], 0.4, 0.9, 140, "2026-10-12 18:30")
    path = calibration_path(tmp_path, "shyam")
    cal.save(path)
    back = EmotionCalibration.load(path)
    assert back == cal and back.logits(np.zeros(7)) == pytest.approx(cal.bias)
    (tmp_path / "broken.emotion.json").write_text("{not json")
    other = json.loads(path.read_text())
    other["emotions"] = ["a", "b"]
    (tmp_path / "old.emotion.json").write_text(json.dumps(other))
    assert set(load_all(tmp_path)) == {"shyam"}
    assert load_all(tmp_path / "missing") == {}


def test_gradient_and_hessian_match_finite_differences():
    from vtrack.emotion_cal import _loss_grad_hess
    rng = np.random.default_rng(5)
    Z, Y = rng.normal(size=(50, 7)), rng.integers(0, 7, 50)
    th = np.concatenate([[1.3], rng.normal(size=7)])
    _, g, H = _loss_grad_hess(th, Z, Y, 0.02)
    eps = 1e-6
    for i in range(8):
        d = np.zeros(8)
        d[i] = eps
        lp, gp, _ = _loss_grad_hess(th + d, Z, Y, 0.02)
        lm, gm, _ = _loss_grad_hess(th - d, Z, Y, 0.02)
        assert g[i] == pytest.approx((lp - lm) / (2 * eps), abs=1e-7)
        assert H[:, i] == pytest.approx((gp - gm) / (2 * eps), abs=1e-7)


def test_calibrations_are_kept_per_emotion_model(tmp_path):
    EmotionCalibration(1.0, [0.0] * 7, model="hsemotion").save(calibration_path(tmp_path, "shyam"))
    legacy = calibration_path(tmp_path, "old")
    d = json.loads(calibration_path(tmp_path, "shyam").read_text())
    del d["model"]                                          # a file written before the model field existed
    legacy.write_text(json.dumps(d))
    assert set(load_all(tmp_path, "hsemotion")) == {"shyam"}
    assert set(load_all(tmp_path, "opencv")) == {"old"}
    assert set(load_all(tmp_path)) == {"shyam", "old"}
    s = CalibrationSession("shyam", 0.0, 1.0, 2.0, model="opencv")
    assert s.model == "opencv"

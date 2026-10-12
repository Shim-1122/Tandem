"""The live app end to end (threads, kinematics, head, emotion, layout, CSV) with fake camera and models."""
import csv
import time

import numpy as np
import pytest

pytest.importorskip("cv2")

from tests.test_gestures import hand_palm
from tests.test_head import landmarks_for
from vtrack.app import App, Config, num, pick_hand_by_label
from vtrack.face import EMOTIONS
from vtrack.handpose import camera_matrix
from vtrack.head import FACING_CAMERA, HeadPoseEstimator
from vtrack.emotion_cal import CAL_ORDER, calibration_path
from vtrack.legs import LEG_FIELDS
from vtrack.synthetic import forward_arm, forward_body
from vtrack.tracker import HandResult, PoseResult

W, H = 1280, 720
K = camera_matrix(W, H, 70)
ARM = dict(shoulder_flexion=35, shoulder_abduction=25, shoulder_rotation=30, elbow_flexion=75,
           forearm_rotation=20, wrist_flexion=10, wrist_deviation=5)
HEAD = (12.0, -8.0, 5.0)                       # yaw, pitch, roll
PROBS = np.array([0.02, 0.01, 0.03, 0.70, 0.14, 0.05, 0.05])
OFFSET = np.array([0.0, 0.1, 2.0])             # body origin (hips) in front of the camera, meters
LEGS = {"right": dict(hip_flexion=30, hip_abduction=10, hip_rotation=15, knee_flexion=50, ankle_dorsiflexion=10,
                      ankle_inversion=5, ankle_abduction=5),
        "left": dict(hip_flexion=-5, hip_abduction=8, hip_rotation=-12, knee_flexion=6, ankle_dorsiflexion=-8,
                     ankle_inversion=-4, ankle_abduction=0)}      # knee straight: hip rotation from the foot


def project(points):
    q = (K @ np.asarray(points, float).T).T
    return np.column_stack([q[:, 0] / q[:, 2] / W, q[:, 1] / q[:, 2] / H, np.zeros(len(q))])


class FakeCap:
    def __init__(self, fps=60.0):
        self.dt = 1.0 / fps

    def read(self):
        time.sleep(self.dt)
        return True, np.full((H, W, 3), 80, np.uint8)


def scene(legs_angles=None, R=FACING_CAMERA):
    """R: body frame (forward, up, right) in camera axes; FACING_CAMERA is a person facing a level camera."""
    pose_w, _ = forward_arm(ARM, "right", body_rotation=R)
    # legs on the same torso (forward_arm puts the hips 0.5 m below the shoulders, 0.1 m to each side)
    legs = forward_body(legs_angles or LEGS, None, 0.0, R, R @ np.array([0.0, -0.5, 0.0]), hip_half=0.1,
                        trunk=0.5, shoulder_half=0.2)
    assert np.allclose(legs[[11, 12, 23, 24]], pose_w[[11, 12, 23, 24]])
    pose_w[25:33] = legs[25:33]
    hand_w = hand_palm(False, ("index", "middle"))                          # peace sign
    hand_w = hand_w - hand_w.mean(axis=0)
    hand_img = project(hand_w - hand_w[0] + pose_w[16] + OFFSET)            # wrist on the pose wrist
    return pose_w, project(pose_w + OFFSET), hand_w, hand_img


class FakeEmotion:
    """Model outputs PROBS; with `expression` set, raw scores for a face the model misreads as sad."""

    def __init__(self):
        self.enroll_request, self.message, self.enrolled, self.expression = None, "", False, None
        self.rng = np.random.default_rng(0)

    def __call__(self, frame):
        if self.enroll_request:
            self.enroll_request, self.message, self.enrolled = None, "enrolled me (1 sample)", True
        if self.expression is None:
            logits = np.log(PROBS)
        else:
            logits = self.rng.normal(scale=0.5, size=7)
            logits[EMOTIONS.index(self.expression(frame.t))] += 3.0
            logits[EMOTIONS.index("sad")] += 4.0
        return dict(box=np.array([600.0, 100.0, 120.0, 140.0]), logits=logits, probs=np.exp(logits) / np.exp(logits).sum(),
                    identity="me" if self.enrolled else None, match=0.95 if self.enrolled else -1.0,
                    enrolled=self.enrolled)


def fake_models(mode="body", legs_angles=None, R=FACING_CAMERA):
    pose_w, pose_img, hand_w, hand_img = scene(legs_angles, R)
    vis = np.ones(33)
    L, _ = landmarks_for(*HEAD, pos=(0.0, -0.45, 1.9))
    est = HeadPoseEstimator()
    models = {"hand": lambda: (lambda f: [HandResult("Left", hand_w.copy(), hand_img.copy())])}
    if mode == "body":
        models["pose"] = lambda: (lambda f: PoseResult(pose_w.copy(), pose_img.copy(), vis))
        models["face"] = lambda: (lambda f: (L, est.solve(L, W, H, K)))
        models["emotion"] = FakeEmotion
    return models


def run_app(app, seconds, keys=()):
    keys = dict(keys)
    last, canvas, t0, i = 0, None, time.monotonic(), 0
    try:
        while time.monotonic() - t0 < seconds:
            f = app.camera.wait_newer(last, 2.0)
            last = f.index
            canvas = app.step(f)
            if i in keys:
                app.handle_key(ord(keys[i]))
            i += 1
    finally:
        app.close()
    return canvas


def test_body_mode_end_to_end(tmp_path):
    rec = tmp_path / "run.csv"
    app = App(Config(side="right", record=rec, cores=8), FakeCap(), fake_models())
    arm_times = []
    solve = app._update_arms
    app._update_arms = lambda t: (arm_times.append(t), solve(t))
    canvas = run_app(app, 1.5, keys={5: "e", 40: "h"})

    # arm: shoulder and elbow from the pose match the joint angles that built it
    a = app.angs["right"]
    for k in ("shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion"):
        assert getattr(a, k) == pytest.approx(ARM[k], abs=0.5), k
    assert np.all(np.diff(arm_times) > 0)                        # filters always see increasing time
    assert app.gestures["right"] == "peace"
    # head, smoothed, settles on the true pose; the neck (head relative to torso) equals it here
    assert (app.head.yaw, app.head.pitch, app.head.roll) == pytest.approx(HEAD, abs=0.2)
    neck = app.neck_axes()
    assert np.allclose(neck, FACING_CAMERA.T @ app.head.axes, atol=1e-6)
    assert app.head_zero.is_set
    # emotion: smoothed probabilities settle on the model output; enrollment went through the worker
    assert [app.emotions[e] for e in EMOTIONS] == pytest.approx(PROBS, abs=1e-3)
    assert app.emotion_res["identity"] == "me"
    # layout: video plus a right column, scaled to the window width
    assert canvas.shape[1] == 1600 and canvas.shape[0] == round(H * 1600 / (W + H // 2))
    col_x = round(W * 1600 / (W + H // 2))
    half = canvas.shape[0] // 2
    assert canvas[:half, col_x + 2:].mean() < 60 < canvas[half:, 900:col_x - 20].mean()   # dark face panel top right
    # status line has every worker and no errors
    status = app.status_line(time.monotonic())
    assert all(n in status for n in ("cam", "hand", "pose", "face", "emotion")) and "ERROR" not in status
    # legs and core
    for sd in ("left", "right"):
        for k, v in LEGS[sd].items():
            assert getattr(app.legs[sd], k) == pytest.approx(v, abs=0.5), (sd, k)
        assert app.legs[sd].pelvic_hike == pytest.approx(0, abs=0.5)
    assert app.legs["left"].hip_rotation_source == "foot" and app.legs["right"].hip_rotation_source == "knee"
    assert (app.core.core_flexion, app.core.core_lateral_bend, app.core.core_rotation) == pytest.approx((0, 0, 0), abs=0.5)
    app.show_legs = True
    text = "\n".join(app.text_lines(time.monotonic()))
    assert "Knee flexion\t6.0\t50.0" in text and "-12.0f" in text and "Core rotation\t0.0" in text

    rows = list(csv.reader(open(rec, encoding="utf-8")))
    header, data = rows[0], rows[1:]
    assert len(data) > 20 and all(len(r) == len(header) for r in data)
    ts = [float(r[0]) for r in data]
    assert ts == sorted(ts)
    last = dict(zip(header, data[-1]))
    assert float(last["R_knee_flexion"]) == pytest.approx(50, abs=0.5) and last["L_hip_rotation_source"] == "foot"
    assert all("L_" + k in header and "R_" + k in header for k in LEG_FIELDS) and "core_rotation" in header
    assert last["gesture"] == "peace" and last["identity"] == "me"
    assert float(last["neck_yaw"]) == pytest.approx(HEAD[0], abs=0.3)
    assert float(last["neck_pitch"]) == pytest.approx(HEAD[1], abs=0.3)
    assert float(last["neck_roll"]) == pytest.approx(HEAD[2], abs=0.3)
    assert float(last["emotion_happy_pct"]) == pytest.approx(70.0, abs=0.2)


def test_both_arms_desk_mode_and_calibration():
    app = App(Config(side="both", torso="desk"), FakeCap(), fake_models())
    run_app(app, 0.8, keys={10: "z"})
    assert app.torso_src == "calibrated"
    assert set(app.angs) == {"left", "right"}
    # the one hand belongs to the right wrist only
    assert app.gestures["right"] == "peace" and app.gestures["left"] == "unknown"


def test_pov_mode_hand_position_and_zero(tmp_path):
    app = App(Config(mode="pov", hand="right", record=tmp_path / "pov.csv"), FakeCap(), fake_models("pov"))
    run_app(app, 0.8, keys={5: "z"})
    assert app.gestures["right"] == "peace"
    info = app.pov_info
    assert info is not None and info["rel"] is not None
    assert info["rel"].dx_cm == pytest.approx(0, abs=0.2) and info["rel"].twist == pytest.approx(0, abs=0.5)
    assert 1.5 < info["t"][2] < 2.6                               # the hand is about 2 m away
    assert list(app.workers) == ["hand"]


def test_text_and_labels():
    assert num(-0.04) == "0.0" and num(-0.6, 0) == "-1" and num(3.14159, 2) == "3.14"
    left, right = HandResult("Right", None, None), HandResult("Left", None, None)
    assert pick_hand_by_label([left, right], "right") is right       # labels are for a mirrored image
    assert pick_hand_by_label([left], "right") is left                # lone hand: used anyway
    assert pick_hand_by_label([left], "right", fallback=False) is None


def test_model_load_failure_stops_the_camera():
    def broken():
        raise OSError("download failed")

    cap = FakeCap()
    with pytest.raises(OSError):
        App(Config(), cap, {"hand": broken})


def test_emotion_calibration_through_the_app(tmp_path):
    cfg = Config(side="right", faces_dir=str(tmp_path), name="shyam", cores=8, emotion_hz=30,
                 cal_prepare_s=0.15, cal_capture_s=0.35)
    app = App(cfg, FakeCap(), fake_models())
    fake = app.fns["emotion"]
    fake.expression = lambda t: "neutral"                    # a resting face the model reads as sad
    last, t0 = 0, time.monotonic()
    while time.monotonic() - t0 < 0.5:
        f = app.camera.wait_newer(last, 2.0)
        last = f.index
        app.step(f)
    assert max(app.emotions, key=app.emotions.get) == "sad" and app.cal_person is None
    app.handle_key(ord("k"))
    session = app.cal_session

    def prompted(t):                                         # the person follows the on-screen prompt
        return CAL_ORDER[min(session.phase(t)[0], len(CAL_ORDER) - 1)]

    fake.expression = prompted
    try:
        while app.cal_session is not None and time.monotonic() - t0 < 15:
            f = app.camera.wait_newer(last, 2.0)
            last = f.index
            app.step(f)
        fake.expression = lambda t: "neutral"
        t1 = time.monotonic()
        while time.monotonic() - t1 < 1.0:
            f = app.camera.wait_newer(last, 2.0)
            last = f.index
            app.step(f)
    finally:
        app.close()
    assert "emotion calibrated for shyam" in app.message, app.message
    assert calibration_path(tmp_path, "shyam").exists() and "shyam" in app.calibrations
    cal = app.calibrations["shyam"]
    assert cal.accuracy_after > cal.accuracy_before
    assert app.cal_person == "shyam" and max(app.emotions, key=app.emotions.get) == "neutral"


def test_standing_calibration_corrects_a_tilted_camera():
    from vtrack.kinematics import rot_x
    straight = {k: 0.0 for k in LEGS["right"]}
    R = rot_x(np.radians(15)) @ FACING_CAMERA              # camera pitched 15 degrees
    app = App(Config(side="right", cores=8), FakeCap(), fake_models(legs_angles={"left": straight, "right": straight}, R=R))
    run_app(app, 0.4)
    before = app.legs["right"].hip_flexion
    app2 = App(Config(side="right", cores=8), FakeCap(), fake_models(legs_angles={"left": straight, "right": straight}, R=R))
    run_app(app2, 1.0, keys={5: "z"})
    assert abs(before) > 10                                  # the camera's tilt reads as hip flexion
    assert app2.leg_up_src == "standing calibration" and "standing calibration done" in app2.message
    for sd in ("left", "right"):
        a = app2.legs[sd]
        for k in ("hip_flexion", "hip_abduction", "knee_flexion", "ankle_dorsiflexion", "foot_pitch", "foot_roll"):
            assert getattr(a, k) == pytest.approx(0.0, abs=0.5), (sd, k)
    assert app2.core.core_flexion == pytest.approx(0.0, abs=0.5)

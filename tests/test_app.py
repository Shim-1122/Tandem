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
    # one readout table: arms and core on the left, legs on the right (single arm mode: left arm column empty)
    rows = [r.split("\t") for r in app.body_table()]
    assert all(len(r) == 6 for r in rows) and rows[0] == ["", "L", "R", "", "L", "R"]
    table = {r[0]: r[1:3] for r in rows[1:] if r[0]} | {r[3]: r[4:6] for r in rows[1:] if r[3]}
    assert table["Knee flexion"] == ["6.0", "50.0"] and table["Hip rotation"][0] == "-12.0f"
    assert table["Core rotation"] == ["0.0", ""] and table["Elbow flexion"] == ["", "75.0"]
    assert table["Shoulder flex/ext"][1] == "35.0"

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


def test_eye_tracking_calibration_and_target_lock_through_the_app(tmp_path):
    """A simulated person looks at each calibration dot, then at a spot, and locks it (key t and pinch)."""
    from tests.test_gaze import face as gaze_face, plane_of
    from tests.test_gestures import touching
    look = {"at": np.array([640.0, 360.0]), "pinch": False}
    est = HeadPoseEstimator()

    def face_fn(f):
        L, _, _ = gaze_face((0.0, -5.0, 0.0), (0.0, 0.08, 0.6), target=plane_of(look["at"]))
        return L, est.solve(L, W, H, K)

    pose_w, pose_img, hand_w, hand_img = scene()
    pinch_w = touching(False)
    pinch_w = pinch_w - pinch_w.mean(axis=0)
    models = fake_models()
    models["face"] = lambda: face_fn
    models["hand"] = lambda: (lambda f: [HandResult("Left", (pinch_w if look["pinch"] else hand_w).copy(), hand_img.copy())])
    cfg = Config(cores=8, face_hz=60, gaze_settle_s=0.15, gaze_capture_s=0.25, record=tmp_path / "gaze.csv")
    app = App(cfg, FakeCap(), models)
    last, t0 = 0, time.monotonic()

    def run(seconds, until=None):
        nonlocal last
        t1 = time.monotonic()
        while time.monotonic() - t1 < seconds and not (until and until()):
            f = app.camera.wait_newer(last, 2.0)
            last = f.index
            app.step(f)

    try:
        run(0.3)
        assert app.eye is not None and app.eye.valid and app.gaze_map is None
        app.handle_key(ord("t"))
        assert "gaze calibration first" in app.message
        app.handle_key(ord("g"))
        session = app.gaze_session
        assert session is not None and len(session.targets) == 9

        def follow():                                         # look at the dot being shown
            i, _, _ = session.phase(time.monotonic())
            look["at"] = session.targets[min(i, 8)]
            return app.gaze_session is None

        while app.gaze_session is not None and time.monotonic() - t0 < 15:
            follow()
            run(0.02)
        assert app.gaze_map is not None, app.message
        assert app.gaze_map.dots == 9 and app.gaze_map.error_px < 1.0 and "gaze calibrated" in app.message
        look["at"] = np.array([300.0, 200.0])
        run(1.0)
        assert app.gaze_pt == pytest.approx([300.0, 200.0], abs=2.0)
        app.handle_key(ord("t"))
        assert app.target.locked and app.target.display_point == pytest.approx([300.0, 200.0], abs=2.0)
        assert app.target.camera_point() == pytest.approx([W - 1 - 300.0, 200.0], abs=2.0)   # video is mirrored
        assert "LOCKED" in app.gaze_line()
        look["pinch"] = True                                  # a pinch releases it
        run(0.6)
        assert not app.target.locked and "released" in app.message
        look["pinch"] = False
        run(0.4)
        look["pinch"] = True                                  # and another pinch locks again
        run(0.6)
        assert app.target.locked
    finally:
        app.close()
    rows = list(csv.reader(open(tmp_path / "gaze.csv", encoding="utf-8")))
    last_row = dict(zip(rows[0], rows[-1]))
    assert last_row["target_locked"] == "1" and float(last_row["target_ray_z"]) > 0.5
    assert abs(float(last_row["gaze_x_px"]) - (W - 1 - 300)) < 3


class FakeObjects:
    """Sees a tennis ball wherever the hand is (as if held) and records the zoom regions the app asks for."""

    def __init__(self, box):
        from vtrack.objects import Detection, Vocabulary
        v = Vocabulary.load()
        k = v.index("tennis ball")
        scores = np.zeros(len(v.names))
        scores[k] = 0.8
        self.det = Detection(k, "tennis ball", 0.8, np.asarray(box, float), scores, {"metal": 0.1, "plastic": 0.1})
        self.rois, self.seen_rois = [], []

    def __call__(self, frame):
        self.seen_rois.append(list(self.rois))
        return {"kind": "full", "runs": [(None, [self.det])]}


def test_grasp_tracking_through_the_app(tmp_path):
    models = fake_models()
    _, _, _, hand_img = scene()
    px = hand_img[:, :2] * np.array([W, H])
    box = [*(px.min(axis=0) - 10), *(px.max(axis=0) + 10)]
    fake = FakeObjects(box)
    models["objects"] = lambda: fake
    app = App(Config(cores=8, record=tmp_path / "grasp.csv"), FakeCap(), models)
    try:
        last, t0 = 0, time.monotonic()
        while time.monotonic() - t0 < 1.5:
            f = app.camera.wait_newer(last, 2.0)
            last = f.index
            canvas = app.step(f)
        g = app.grasp["right"]
        assert g.state == "holding" and g.object_name == "tennis ball" and g.plan.force_n == pytest.approx(1.0)
        assert app.grasp["left"].state == "free"
        lines = app.text_lines(time.monotonic())
        assert any("right: holding tennis ball" in ln for ln in lines) and any(ln.startswith("objects: tennis ball") for ln in lines)
        rois = [r for r in fake.seen_rois if r]
        assert rois and rois[-1][0][0] == "right"
        x0, y0, x1, y1 = rois[-1][0][1]
        assert x0 < px[:, 0].min() and x1 > px[:, 0].max() and y0 < px[:, 1].min() and y1 > px[:, 1].max()
        assert canvas.shape[1] <= 1600
        app.handle_key(ord("o"))
        assert not app.show_objects
    finally:
        app.close()
    rows = list(csv.reader(open(tmp_path / "grasp.csv", encoding="utf-8")))
    last_row = dict(zip(rows[0], rows[-1]))
    assert last_row["R_grasp_state"] == "holding" and last_row["R_grasp_object"] == "tennis ball"
    assert last_row["R_grip_type"] == "spherical" and float(last_row["R_grip_n"]) == pytest.approx(1.0)
    assert last_row["L_grasp_state"] == "free" and last_row["objects_seen"].startswith("tennis ball@")


def test_grasp_tracking_in_pov_mode(tmp_path):
    models = fake_models("pov")
    hand_img = models["hand"]()(None)[0].image
    px = hand_img[:, :2] * np.array([W, H])
    models["objects"] = lambda: FakeObjects([*(px.min(axis=0) - 10), *(px.max(axis=0) + 10)])
    app = App(Config(mode="pov", hand="right", cores=8, record=tmp_path / "pov.csv"), FakeCap(), models)
    try:
        last, t0 = 0, time.monotonic()
        while time.monotonic() - t0 < 1.2:
            f = app.camera.wait_newer(last, 2.0)
            last = f.index
            app.step(f)
        assert app.grasp["right"].state == "holding"
    finally:
        app.close()
    rows = list(csv.reader(open(tmp_path / "pov.csv", encoding="utf-8")))
    assert dict(zip(rows[0], rows[-1]))["grasp_object"] == "tennis ball"

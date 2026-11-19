"""The full human in MuJoCo: its legs, pelvis and core read back through the tracker's own solvers as the angles that
posed them, the arms match the tracker's forward kinematics, mirror mode swaps sides, the feet stay on the floor,
and the face follows blendshapes and emotions."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vtrack import face_rig                                            # noqa: E402
from vtrack.fingers import FINGERS4, THUMB, hand_fk                    # noqa: E402
from vtrack.human import CHEST_H, SHOULDER_HALF, HumanSim, mirror_values  # noqa: E402
from vtrack.legs import LEG, LegKinematics, core_angles                # noqa: E402
from vtrack.sim import NOMINAL_HAND, T2M                               # noqa: E402
from vtrack.synthetic import arm_chain                                 # noqa: E402

UP = np.array([0.0, 0.0, 1.0])
LEG_KEYS = ["hip_flexion", "hip_abduction", "hip_rotation", "knee_flexion", "ankle_dorsiflexion", "ankle_inversion",
            "ankle_abduction"]


@pytest.fixture(scope="module")
def human():
    h = HumanSim(mirror=False, smooth_face=False)
    yield h
    h.close()


def landmarks(h: HumanSim) -> np.ndarray:
    """MediaPipe-style pose points (33, 3) taken from the simulated body."""
    pose = np.zeros((33, 3))
    pose[11], pose[12] = h.body_pos("L_upper_arm"), h.body_pos("R_upper_arm")
    for side, (hip, knee, ankle, heel, toe) in LEG.items():
        p = side[0].upper() + "_"
        pose[hip], pose[knee], pose[ankle] = h.body_pos(p + "thigh"), h.body_pos(p + "shank"), h.body_pos(p + "foot")
        pose[heel], pose[toe] = h.site(p + "heel"), h.site(p + "toe")
    return pose


def random_body(rng):
    v = {"core_flexion": rng.uniform(-20, 60), "core_lateral_bend": rng.uniform(-25, 25),
         "core_rotation": rng.uniform(-45, 45)}
    hike = rng.uniform(-12, 12)
    v["R_pelvic_hike"], v["L_pelvic_hike"] = hike, -hike
    for p in ("R_", "L_"):
        v.update({p + "hip_flexion": rng.uniform(-20, 100), p + "hip_abduction": rng.uniform(-20, 40),
                  p + "hip_rotation": rng.uniform(-35, 35), p + "knee_flexion": rng.uniform(30, 120),
                  p + "ankle_dorsiflexion": rng.uniform(-40, 25), p + "ankle_inversion": rng.uniform(-20, 30),
                  p + "ankle_abduction": rng.uniform(-20, 20)})
    return v


def test_legs_pelvis_and_core_read_back_as_the_tracked_angles(human):
    rng = np.random.default_rng(7)
    for _ in range(25):
        v = random_body(rng)
        human.command(v)
        pose = landmarks(human)
        for side in ("right", "left"):
            p = side[0].upper() + "_"
            got = LegKinematics(side).solve(pose, UP)
            for k in LEG_KEYS:
                assert getattr(got, k) == pytest.approx(v[p + k], abs=0.05), (p + k)
            assert got.pelvic_hike == pytest.approx(v[p + "pelvic_hike"], abs=0.05), p + "pelvic_hike"
        core = core_angles(pose, UP)
        for k in ("core_flexion", "core_lateral_bend", "core_rotation"):
            assert getattr(core, k) == pytest.approx(v[k], abs=0.05), k


@pytest.mark.parametrize("side", ["right", "left"])
def test_arms_match_the_tracker_kinematics(human, side):
    from tests.test_sim import random_pose
    p = side[0].upper() + "_"
    rng = np.random.default_rng(3)
    mirror = np.array([1.0, -1.0, 1.0]) if side == "left" else np.ones(3)
    for _ in range(10):
        arm, fingers = random_pose(rng)
        human.command({"core_flexion": rng.uniform(-10, 40), **{p + k: x for k, x in arm.items()},
                       **{f"{p}{n}_{k}": x for n, d in fingers.items() for k, x in d.items()}})
        b = human.data.body("chest")
        origin, R = b.xpos, b.xmat.reshape(3, 3)
        c = arm_chain(arm)

        def world(x):
            return origin + R @ ((T2M @ (x - [0, 0, 0.2]) + [0, -SHOULDER_HALF, CHEST_H]) * mirror)

        for site, key in (("elbow", "E"), ("wrist", "W"), ("knuckle", "K")):
            assert np.linalg.norm(human.site(p + site) - world(c[key])) < 1e-5, site
        tips = hand_fk(fingers, NOMINAL_HAND)
        for name, idx in {**FINGERS4, "thumb": THUMB}.items():
            assert np.linalg.norm(human.site(f"{p}{name}_tip") - world(c["W"] + c["R_hand"] @ tips[idx[3]])) < 1e-5


def test_the_lowest_foot_stays_on_the_floor(human):
    rng = np.random.default_rng(11)
    for _ in range(10):
        human.command(random_body(rng))
        lowest = min(human.site(f"{p}sole{i}")[2] for p in ("R_", "L_") for i in range(4))
        assert lowest == pytest.approx(0.0, abs=1e-9)


def test_mirror_swaps_sides_and_turns_midline_angles():
    m = mirror_values({"R_elbow_flexion": 90, "L_knee_flexion": 10, "neck_yaw": 20, "core_flexion": 30,
                       "bs_mouthSmileLeft": 0.8, "bs_eyeBlinkRight": 1.0})
    assert m == {"L_elbow_flexion": 90, "R_knee_flexion": 10, "neck_yaw": -20, "core_flexion": 30,
                 "bs_mouthSmileRight": 0.8, "bs_eyeBlinkLeft": 1.0}


def test_mirror_mode_moves_the_reflected_arm():
    h = HumanSim(mirror=True, smooth_face=False)
    try:
        h.command({"R_shoulder_abduction": 90, "neck_yaw": 30})
        assert h.angle("L_shoulder_abduction") == pytest.approx(90, abs=1e-6)
        assert h.angle("R_shoulder_abduction") == pytest.approx(0, abs=1e-6)
        assert h.angle("neck_yaw") == pytest.approx(-30, abs=1e-6)
    finally:
        h.close()


def skin_landmarks(h: HumanSim) -> np.ndarray:
    """The face's 68 landmarks on the skin as rendered, in the head's frame (x forward, y left, z up), so the
    body's pose does not matter."""
    f = h.face
    pts = f.skin_points(h.model, h.data, "skin")
    first = {int(v): i for i, v in reversed(list(enumerate(f.d["skin_src"])))}
    head = h.data.body("head")
    return (pts[[first[int(i)] for i in f.d["landmarks"]]] - head.xpos) @ head.xmat.reshape(3, 3)


def face_only(h: HumanSim, bs: dict, **extra) -> np.ndarray:
    h.command({**{f"bs_{n}": 0.0 for n in face_rig.BLENDSHAPES}, **{f"bs_{k}": v for k, v in bs.items()}, **extra})
    return skin_landmarks(h)


@pytest.fixture(scope="module")
def full():
    h = HumanSim(mirror=False, smooth_face=False, rig="full", scene=False)
    yield h
    h.close()


def test_the_full_rig_reproduces_each_sculpted_movement(full):
    """One actuator per movement: at full travel the skin sits exactly on ICT's sculpt (rest plus its field)."""
    f, d = full.face, full.face.d
    sid = full.mj.mj_name2id(full.model, full.mj.mjtObj.mjOBJ_SKIN, "skin")
    adr = int(full.model.skin_vertadr[sid])
    rows = f.row[d["skin_src"]]
    for name in ("mouthSmileLeft", "browDownRight", "eyeBlinkLeft", "mouthPucker"):
        face_only(full, {name: 1.0})
        i = f.inputs.index(name)
        want = d["base"][d["skin_src"]].astype(np.float64)
        on = rows >= 0
        want[on] += d["move_fields"][i][rows[on]]
        got = full.model.skin_vert[adr:adr + len(want)]
        assert np.abs(got - want).max() < 2e-6, name


def test_the_jaw_hinge_carries_the_teeth_and_opens_the_mouth(human):
    rest = face_only(human, {})
    teeth = human.face.skin_points(human.model, human.data, "teeth_lower")
    opened = face_only(human, {"jawOpen": 1.0})
    assert human.angle("jaw_open") == pytest.approx(np.degrees(human.face.jaw["open"]), abs=1e-6)
    gap = np.linalg.norm(opened[66] - opened[62]) - np.linalg.norm(rest[66] - rest[62])
    assert gap > 0.02                                                    # lips part by over 2 cm
    moved = np.linalg.norm(human.face.skin_points(human.model, human.data, "teeth_lower") - teeth, axis=1)
    assert moved.max() > 0.02 and opened[8][2] < rest[8][2] - 0.02       # lower teeth and chin drop


def lid_gap(lm, side):
    up, low = ((37, 38), (41, 40)) if side == "R" else ((43, 44), (47, 46))
    return float(np.mean([np.linalg.norm(lm[u] - lm[v]) for u, v in zip(up, low)]))


@pytest.mark.parametrize("rig", ["cogley", "full"])
def test_blink_smile_and_hold(rig, human, full):
    h = human if rig == "cogley" else full
    rest = face_only(h, {})
    blink = face_only(h, {"eyeBlinkRight": 1.0})
    assert lid_gap(blink, "R") < 0.45 * lid_gap(rest, "R")               # the right eye closes
    assert lid_gap(blink, "L") > 0.7 * lid_gap(rest, "L")                # the left stays open
    smile = face_only(h, {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0})
    for corner in (48, 54):
        assert smile[corner][2] > rest[corner][2] + 0.003                # both corners rise
    assert abs((smile[48] - rest[48])[2] - (smile[54] - rest[54])[2]) < 5e-4   # evenly
    held = skin_landmarks(h)
    h.command({"bs_jawOpen": float("nan")})                              # face lost: hold
    assert np.allclose(skin_landmarks(h), held)


def test_the_cogley_rig_is_symmetric_and_names_his_channels(human):
    names = {"MID_RIGHT_BROW", "FAR_LEFT_BROW", "TOP_RIGHT_LID", "LOW_LEFT_LID", "TOP_RIGHT_LIP", "LOW_LEFT_LIP",
             "TOP_LEFT_CHEEK", "LOW_RIGHT_CHEEK"}
    face_only(human, {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8, "browInnerUp": 0.6, "jawOpen": 0.3})
    cmd = human.servo_commands()
    assert names <= set(cmd) and len(cmd) == 17 and cmd["jaw_open"] == pytest.approx(0.3)
    for k, v in cmd.items():
        if "RIGHT" in k:
            assert v == pytest.approx(cmd[k.replace("RIGHT", "LEFT")], abs=1e-4), k
        assert -1.0 <= v <= 1.0
    travel = [s["travel"] for s in human.face.servos]
    assert 0.005 < min(travel) and max(travel) < 0.03                    # 5 to 30 mm, like real linkages


def test_mirror_mode_smiles_with_the_reflected_side():
    h = HumanSim(mirror=True, smooth_face=False, rig="full", scene=False)
    try:
        rest = face_only(h, {})
        lm = face_only(h, {"mouthSmileLeft": 1.0})                       # your left corner ...
        assert lm[48][2] - rest[48][2] > 0.005                           # ... is the avatar's right
        assert lm[54][2] - rest[54][2] < 0.002
    finally:
        h.close()


def test_emotion_presets_drive_the_face(human):
    def face(emotion):
        human.command({**{f"bs_{n}": 0.0 for n in face_rig.BLENDSHAPES},
                       **{f"emotion_{e}_pct": 100.0 * (e == emotion) for e in face_rig.EMOTION_NAMES}})
        return skin_landmarks(human)

    neutral, happy, surprised, angry = face("neutral"), face("happy"), face("surprised"), face("angry")
    assert happy[48][2] > neutral[48][2] + 0.003
    assert np.linalg.norm(surprised[66] - surprised[62]) > np.linalg.norm(neutral[66] - neutral[62]) + 0.01
    assert surprised[19][2] > neutral[19][2] + 0.003                     # brows up
    assert angry[21][2] < neutral[21][2] - 0.001                         # inner brows down


def test_face_sources():
    bs = {"mouthSmileLeft": 0.2}
    emo = {"happy": 1.0}
    assert face_rig.combine(bs, emo, "blendshapes") == bs
    assert face_rig.combine(bs, emo, "emotion")["mouthSmileLeft"] == pytest.approx(min(1.0, 1.25 * 0.9))
    assert face_rig.combine(bs, None, "auto") == bs
    assert face_rig.combine(bs, emo, "auto")["mouthSmileLeft"] == pytest.approx(1.0)


def test_lids_follow_the_eyes(human):
    rest = face_only(human, {})
    up = face_only(human, {}, eye_pitch=20.0)
    assert human.angle("R_eye_pitch") == pytest.approx(20.0, abs=1e-6)
    assert up[37][2] > rest[37][2] + 0.0005                              # the upper lid rises with the gaze


def test_physics_mode_holds_a_pose():
    h = HumanSim(mode="physics", scene=False, smooth_face=False, mirror=False)
    try:
        h.command({"R_hip_flexion": 60, "R_knee_flexion": 70, "R_shoulder_flexion": 90, "core_rotation": 20})
        h.step(0.25)
        h.step(0.25)
        for j, want in (("R_hip_flexion", 60), ("R_knee_flexion", 70), ("R_shoulder_flexion", 90),
                        ("core_rotation", 20)):
            assert h.angle(j) == pytest.approx(want, abs=4.0), j
    finally:
        h.close()


def test_render_views(human):
    for cam in human.VIEWS:
        img = human.render(cam, 160, 120)
        assert img.shape == (120, 160, 3) and img.std() > 5


def test_live_app_drives_the_human_with_blendshapes(tmp_path):
    pytest.importorskip("cv2")
    import csv

    from tests.test_app import ARM, HEAD, FakeCap, fake_models, run_app
    from vtrack.app import App, Config
    models = fake_models()
    inner = models["face"]

    def face_with_blendshapes():
        run = inner()
        return lambda f: (*run(f), None, {"jawOpen": 0.8, "mouthSmileLeft": 0.9, "mouthSmileRight": 0.9})
    models["face"] = face_with_blendshapes
    rec = tmp_path / "run.csv"
    app = App(Config(side="right", cores=8, sim="human", record=rec), FakeCap(), models)
    run_app(app, 1.0)
    values = app.values()
    assert values["bs_jawOpen"] == pytest.approx(0.8)
    assert values["elbow_flexion"] == pytest.approx(ARM["elbow_flexion"], abs=0.5)
    sim = app.sim
    sim.command(values)
    # mirror mode: the tracked right arm moves the avatar's left arm, and head yaw turns the other way
    assert sim.angle("L_elbow_flexion") == pytest.approx(ARM["elbow_flexion"], abs=0.5)
    assert sim.angle("neck_yaw") == pytest.approx(-HEAD[0], abs=1.0)
    assert sim.angle("jaw_open") > 15 and sim.servo_commands()["jaw_open"] == pytest.approx(0.8, abs=0.05)
    img = app.sim_frame(0.0)
    assert img.shape[2] == 3
    app.close()
    with open(rec, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert set(face_rig.BS_COLS) <= set(header)


def test_camera_specs():
    from vtrack.pipeline import parse_camera
    assert parse_camera("0") == 0 and parse_camera(2) == 2
    assert parse_camera("http://192.168.1.20:8080/video") == "http://192.168.1.20:8080/video"


def test_live_options_for_the_human_sim():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("live", Path(__file__).resolve().parents[1] / "scripts" / "live.py")
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    cfg = live.parse_args(["--sim", "human", "--camera", "rtsp://10.0.0.5:8554/live", "--no-mirror",
                           "--face-source", "emotion", "--face-rig", "full", "--sim-camera", "animatronic"])
    assert cfg.face_rig == "full" and cfg.sim_camera == "animatronic"
    assert live.parse_args([]).face_rig == "cogley"
    assert cfg.sim == "human" and cfg.camera == "rtsp://10.0.0.5:8554/live" and not cfg.sim_mirror
    assert cfg.face_source == "emotion"
    assert live.parse_args(["--sim"]).sim == "robot" and live.parse_args([]).sim is False
    assert live.parse_args(["--camera", "1"]).camera == 1

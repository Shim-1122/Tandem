"""The MuJoCo robot: its joints match the tracker's kinematics, every tracked angle drives a joint, and the servos
hold the pose under gravity."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vtrack.app import ANGLE_FIELDS, FINGER_COLS                 # noqa: E402
from vtrack.fingers import FINGERS4, THUMB, hand_fk             # noqa: E402
from vtrack.sim import (COUPLED, LID_CLOSED_DEG, NOMINAL_HAND, SHOULDER_HALF, T2M, TRUNK, TandemSim,  # noqa: E402
                        build_xml, joint_list, source_column)
from vtrack.synthetic import arm_chain                          # noqa: E402

HEAD_COLS = ["core_flexion", "core_lateral_bend", "core_rotation", "neck_yaw", "neck_pitch", "neck_roll",
             "eyes_open", "eye_yaw", "eye_pitch"]


@pytest.fixture(scope="module")
def sim():
    s = TandemSim()
    yield s
    s.close()


def random_pose(rng):
    arm = {"shoulder_flexion": rng.uniform(-40, 150), "shoulder_abduction": rng.uniform(-20, 120),
           "shoulder_rotation": rng.uniform(-70, 70), "elbow_flexion": rng.uniform(0, 140),
           "forearm_rotation": rng.uniform(-80, 80), "wrist_flexion": rng.uniform(-60, 60),
           "wrist_deviation": rng.uniform(-30, 30)}
    fingers = {n: {"mcp_flex": rng.uniform(-20, 80), "mcp_abd": rng.uniform(-20, 20), "pip": rng.uniform(0, 100),
                   "dip": rng.uniform(0, 80)} for n in FINGERS4}
    fingers["thumb"] = {"cmc_palmar": rng.uniform(0, 60), "cmc_radial": rng.uniform(0, 60),
                        "cmc_rotation": rng.uniform(-30, 120), "mcp": rng.uniform(0, 60), "ip": rng.uniform(0, 60)}
    return arm, fingers


@pytest.mark.parametrize("side", ["right", "left"])
def test_robot_joints_match_the_tracker_kinematics(sim, side):
    """Elbow, wrist, knuckle and all five fingertips land where vtrack's forward kinematics put them."""
    p = side[0].upper() + "_"
    rng = np.random.default_rng(1 if side == "right" else 2)
    mirror = np.array([1.0, -1.0, 1.0]) if side == "left" else np.ones(3)
    for _ in range(15):
        arm, fingers = random_pose(rng)
        values = {"core_rotation": rng.uniform(-60, 60), **{p + k: v for k, v in arm.items()},
                  **{f"{p}{n}_{k}": v for n, d in fingers.items() for k, v in d.items()}}
        sim.command(values)
        sim.snap()
        c = arm_chain(arm)
        origin, R = sim.chest_frame()

        def world(v):                                   # tracker thorax point (right arm) to MuJoCo world
            return origin + R @ ((T2M @ (v - [0, 0, 0.2]) + [0, -SHOULDER_HALF, TRUNK]) * mirror)

        for site, key in (("elbow", "E"), ("wrist", "W"), ("knuckle", "K")):
            assert np.linalg.norm(sim.site(p + site) - world(c[key])) < 1e-5, site
        tips = hand_fk(fingers, NOMINAL_HAND)
        for name, idx in {**FINGERS4, "thumb": THUMB}.items():
            want = world(c["W"] + c["R_hand"] @ tips[idx[3]])
            assert np.linalg.norm(sim.site(f"{p}{name}_tip") - want) < 1e-5, name


def test_every_tracked_angle_drives_a_joint():
    two_arms = [f"{s}_{k}" for s in "LR" for k in ANGLE_FIELDS + FINGER_COLS] + HEAD_COLS
    joints = [n for n, _, _ in joint_list()]
    assert len(joints) == 1 + 2 * (7 + 4 * 4 + 5) + 3 + 2 * 3          # waist, two arms, neck, eyes and lids
    driven = {j: source_column(j, two_arms) for j in joints if j not in COUPLED}
    assert all(driven.values()), [j for j, c in driven.items() if c is None]
    assert driven["R_eye_yaw"] == "eye_yaw" and driven["R_eyelid"] == "eyes_open"
    # one-arm and hand-view recordings have no prefix: they drive the chosen side only
    one_arm = list(ANGLE_FIELDS) + FINGER_COLS
    assert source_column("R_elbow_flexion", one_arm, "right") == "elbow_flexion"
    assert source_column("L_elbow_flexion", one_arm, "right") is None
    assert source_column("L_index_pip", FINGER_COLS, "left") == "index_pip"
    # core lean and side bend are tracked but the robot's waist only turns
    assert "core_flexion" not in joints and "core_rotation" in joints


def test_commands_clip_hold_and_blink():
    s = TandemSim(scene=False)
    s.command({"R_elbow_flexion": 500, "neck_yaw": 10, "eyes_open": 1, "eye_yaw": 12})
    assert s.targets()["R_elbow_flexion"] == pytest.approx(150, abs=0.01)     # clipped to the joint range
    s.command({"R_elbow_flexion": "nan", "neck_yaw": "", "eyes_open": 0, "eye_yaw": 5})
    t = s.targets()
    assert t["R_elbow_flexion"] == pytest.approx(150, abs=0.01) and t["neck_yaw"] == pytest.approx(10)   # held
    assert t["R_eyelid"] == pytest.approx(0)          # no face this row: a missing face is not a blink
    s.command({"neck_yaw": 10, "eyes_open": 0})
    assert s.targets()["R_eyelid"] == pytest.approx(LID_CLOSED_DEG)
    s.snap()
    assert s.angle("L_eyelid") == pytest.approx(LID_CLOSED_DEG) and s.angle("L_eye_yaw") == pytest.approx(5)


def test_servos_reach_and_hold_a_pose_under_gravity():
    s = TandemSim()
    pose = {"R_shoulder_flexion": 90, "R_elbow_flexion": 60, "L_shoulder_abduction": 90, "core_rotation": 25,
            "neck_yaw": 20, "neck_pitch": -10, "eye_yaw": 15, "R_index_pip": 60, "R_thumb_mcp": 40}
    s.command(pose)
    for _ in range(4):
        s.step(0.25)                                                     # one step is at most a quarter second
    errors = s.errors()
    assert max(abs(e) for e in errors.values()) < 2.0                    # within 2 degrees after a second
    assert np.abs(s.data.qvel[:-6]).max() < 0.05                         # and settled (the ball is the last 6)
    assert s.angle("L_eye_yaw") == pytest.approx(15, abs=0.5)           # the left eye follows through the link


def test_model_text_and_scene_options():
    xml = build_xml(scene=False)
    assert "<freejoint" not in xml and 'name="R_index_tip"' in xml and "\u2014" not in xml
    assert TandemSim(scene=False).model.nu == len(joint_list()) - len(COUPLED)


def load_script():
    spec = importlib.util.spec_from_file_location("sim_script", Path(__file__).parents[1] / "scripts" / "sim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_moves_every_servo_and_replay_keeps_the_timing():
    mod = load_script()
    s = TandemSim(scene=False)
    driven = s.command(mod.demo_values(1.3))
    assert set(driven) == set(s.actuators)
    rows = [{"t": "10.0"}, {"t": "10.5"}, {"t": "bad"}, {"t": "11.0"}]
    assert [t for t, _ in mod.frames(rows, 2.0)] == [0.0, 0.25, 0.5]


def test_live_app_drives_the_robot():
    pytest.importorskip("cv2")
    from tests.test_app import ARM, HEAD, FakeCap, fake_models, run_app
    from vtrack.app import App, Config
    app = App(Config(side="right", cores=8, sim=True), FakeCap(), fake_models())
    run_app(app, 1.0)
    values = app.values()
    assert values["elbow_flexion"] == pytest.approx(ARM["elbow_flexion"], abs=0.5)
    app.sim.command(values)
    for _ in range(4):
        app.sim.step(0.25)
    assert app.sim.angle("R_elbow_flexion") == pytest.approx(ARM["elbow_flexion"], abs=2.0)
    assert app.sim.angle("neck_yaw") == pytest.approx(HEAD[0], abs=1.0)
    assert app.sim.angle("L_elbow_flexion") == pytest.approx(0, abs=2.0)   # the untracked arm stays at rest


def test_render_views():
    s = TandemSim()
    try:
        img = s.render("front", 160, 120)
    except Exception as e:                       # no OpenGL on this machine (headless without EGL or OSMesa)
        pytest.skip(f"no offscreen OpenGL: {e}")
    assert img.shape == (120, 160, 3) and img.std() > 5
    assert s.render("eyes", 160, 120).shape == (120, 160, 3)
    s.close()

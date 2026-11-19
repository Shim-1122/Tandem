"""ROS 2 output with an in-memory stand-in for rclpy and sensor_msgs (ROS 2 is not needed to run the tests):
what the tracker publishes on each topic, with which QoS and stamps, and the avatar node applying it."""
import math
import sys
import time
import types
from dataclasses import dataclass, field

import numpy as np
import pytest

from vtrack import face_rig


# ------------------------------------------------------------------ a tiny in-process ROS 2
@dataclass
class _Stamp:
    sec: int = 0
    nanosec: int = 0


@dataclass
class _Header:
    stamp: _Stamp = field(default_factory=_Stamp)
    frame_id: str = ""


@dataclass
class JointState:
    header: _Header = field(default_factory=_Header)
    name: list = field(default_factory=list)
    position: list = field(default_factory=list)


class _Time:
    def __init__(self, ns):
        self.nanoseconds = int(ns)

    def __sub__(self, d):
        return _Time(self.nanoseconds - d.nanoseconds)

    def to_msg(self):
        return _Stamp(self.nanoseconds // 1_000_000_000, self.nanoseconds % 1_000_000_000)


class _Duration:
    def __init__(self, nanoseconds=0):
        self.nanoseconds = nanoseconds


class _Bus:
    def __init__(self):
        self.subs, self.log = {}, []


BUS = _Bus()


class _Publisher:
    def __init__(self, topic, qos):
        self.topic, self.qos = topic, qos

    def publish(self, msg):
        BUS.log.append((self.topic, msg))
        for cb in BUS.subs.get(self.topic, []):
            cb(msg)


class _Node:
    def __init__(self, name):
        self.name = name

    def create_publisher(self, msg_type, topic, qos):
        return _Publisher(topic, qos)

    def create_subscription(self, msg_type, topic, cb, qos):
        BUS.subs.setdefault(topic, []).append(cb)

    def get_clock(self):
        return types.SimpleNamespace(now=lambda: _Time(time.time_ns()))

    def destroy_node(self):
        pass


@pytest.fixture()
def ros(monkeypatch):
    BUS.subs.clear()
    BUS.log.clear()
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.init = lambda: None
    rclpy.create_node = _Node
    qos = types.ModuleType("rclpy.qos")

    class QoSProfile:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    qos.QoSProfile = QoSProfile
    qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="keep_last")
    qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT="best_effort")
    qos.DurabilityPolicy = types.SimpleNamespace(VOLATILE="volatile")
    duration = types.ModuleType("rclpy.duration")
    duration.Duration = _Duration
    msgs = types.ModuleType("sensor_msgs.msg")
    msgs.JointState = JointState
    for name, mod in {"rclpy": rclpy, "rclpy.qos": qos, "rclpy.duration": duration,
                      "sensor_msgs": types.ModuleType("sensor_msgs"), "sensor_msgs.msg": msgs}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return BUS


def test_the_tracker_publishes_joints_face_and_emotion(ros):
    from tests.test_app import ARM, FakeCap, fake_models, run_app
    from vtrack.app import App, Config
    models = fake_models()
    inner = models["face"]

    def face_with_blendshapes():
        run = inner()
        return lambda f: (*run(f), None, {"jawOpen": 0.6, "mouthSmileLeft": 0.8, "mouthSmileRight": 0.8})
    models["face"] = face_with_blendshapes
    app = App(Config(side="right", cores=8, ros2=True), FakeCap(), models)
    run_app(app, 1.5)
    topics = {t for t, _ in ros.log}
    assert topics == {"/tandem/joint_states", "/tandem/face/blendshapes", "/tandem/face/servos", "/tandem/emotion"}
    pubs = app.ros.pubs
    assert all(p.qos.depth == 1 and p.qos.reliability == "best_effort" for p in pubs.values())
    joints = [m for t, m in ros.log if t == "/tandem/joint_states"]
    assert len(joints) == app.ros.sent and len(joints) > 20           # one per result
    last = dict(zip(joints[-1].name, joints[-1].position))
    assert last["elbow_flexion"] == pytest.approx(math.radians(ARM["elbow_flexion"]), abs=0.01)
    assert all(math.isfinite(p) for p in joints[-1].position)
    age = time.time() - (joints[-1].header.stamp.sec + joints[-1].header.stamp.nanosec / 1e9)
    assert 0 <= age < 1.0                                             # stamped with the capture time
    bs = [m for t, m in ros.log if t == "/tandem/face/blendshapes"][-1]
    assert dict(zip(bs.name, bs.position))["jawOpen"] == pytest.approx(0.6)
    servos = [m for t, m in ros.log if t == "/tandem/face/servos"][-1]
    cmd = dict(zip(servos.name, servos.position))
    assert {"TOP_LEFT_LIP", "MID_RIGHT_BROW", "jaw_open"} <= set(cmd) and cmd["jaw_open"] == pytest.approx(0.6)
    assert cmd["TOP_LEFT_CHEEK"] == pytest.approx(cmd["TOP_RIGHT_CHEEK"])      # a symmetric smile
    emo = [m for t, m in ros.log if t == "/tandem/emotion"][-1]
    assert sum(emo.position) == pytest.approx(1.0, abs=0.02) and set(emo.name) <= set(face_rig.EMOTION_NAMES)


def test_the_avatar_node_drives_the_human_and_forwards_to_isaac(ros):
    pytest.importorskip("mujoco")
    import importlib.util
    from pathlib import Path

    from vtrack.human import HumanSim
    from vtrack.ros2 import TandemPublisher
    spec = importlib.util.spec_from_file_location("ros2_avatar",
                                                  Path(__file__).resolve().parents[1] / "scripts" / "ros2_avatar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sim = HumanSim(mirror=False, smooth_face=False, scene=False)
    try:
        node = _Node("tandem_avatar")
        av = mod.AvatarNode(node, sim, isaac_topic="/joint_command", rename={"R_knee_flexion": "right_knee"})
        pub = TandemPublisher()
        values = {"R_knee_flexion": 40.0, "R_elbow_flexion": 70.0, "core_rotation": -15.0, "eye_yaw": 0.0,
                  "bs_jawOpen": 0.5, "emotion_happy_pct": 80.0, "emotion_neutral_pct": 20.0, "R_index_pip": np.nan}
        pub.publish(values, time.monotonic(), kind="face")
        av.apply()
        assert sim.angle("R_knee_flexion") == pytest.approx(40.0, abs=1e-6)
        assert sim.angle("R_elbow_flexion") == pytest.approx(70.0, abs=1e-6)
        assert sim.angle("core_rotation") == pytest.approx(-15.0, abs=1e-6)
        assert sim.servos["jaw_open"] == pytest.approx(0.5)
        fwd = [m for t, m in ros.log if t == "/joint_command"][-1]
        assert "right_knee" in fwd.name and "R_index_pip" not in fwd.name   # NaN angles are left out
        assert 0 <= av.latency_ms < 1000
    finally:
        sim.close()

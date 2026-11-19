"""ROS 2 output: Tandem's tracking as topics, for the Jetson, Isaac Sim and the real robot.

    python scripts/live.py --ros2            # in a terminal where ROS 2 is sourced (source /opt/ros/<distro>/setup.bash)
    python scripts/ros2_avatar.py            # another process (or computer): the avatar, or a bridge to Isaac Sim

Topics (all sensor_msgs/JointState, which carries named values; best effort, depth 1, so a subscriber always
gets the newest pose and nothing queues up):
  /tandem/joint_states     every tracked joint angle, in radians (names as in the recording's CSV columns:
                           R_elbow_flexion, L_knee_flexion, core_rotation, neck_yaw, R_index_pip, ...)
  /tandem/face/blendshapes MediaPipe's 52 face blendshapes, 0..1 (ARKit names)
  /tandem/face/servos      the animatronic face's servo positions, -1..1 (Will Cogley's channel names), and jaw
  /tandem/emotion          the seven emotion scores, 0..1
Each message is stamped with the time the camera captured the frame it came from, so a subscriber can see the
whole latency. Joint states go out the moment each result is ready (on the tracking thread); the face topics
when a face or emotion result arrives.
"""
from __future__ import annotations

import math
import time

from .face import EMOTIONS
from .face_rig import BLENDSHAPES, BS_COLS
from .fingers import FINGER_KEYS, FINGERS4, THUMB_KEYS
from .kinematics import ArmAngles
from .legs import CORE_FIELDS, LEG_FIELDS

TOPICS = {"joints": "/tandem/joint_states", "blendshapes": "/tandem/face/blendshapes",
          "servos": "/tandem/face/servos", "emotion": "/tandem/emotion"}
_ARM = [f for f in ArmAngles.__dataclass_fields__ if not f.endswith(("_valid", "_predicted"))]
_FINGERS = [f"{n}_{k}" for n in FINGERS4 for k in FINGER_KEYS] + [f"thumb_{k}" for k in THUMB_KEYS]
_ANGLES = set(_ARM + _FINGERS + [f for f in LEG_FIELDS] + CORE_FIELDS
              + ["neck_yaw", "neck_pitch", "neck_roll", "eye_yaw", "eye_pitch", "head_yaw", "head_pitch", "head_roll"])


def is_joint_angle(name: str) -> bool:
    """A joint angle column (with or without its R_/L_ side)."""
    return name in _ANGLES or (name[:2] in ("R_", "L_") and name[2:] in _ANGLES)


def _finite(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def joint_positions(values: dict) -> tuple[list, list]:
    """(names, radians) of the finite joint angles in a values dict (degrees)."""
    names, pos = [], []
    for k, v in values.items():
        if is_joint_angle(k):
            x = _finite(v)
            if x is not None:
                names.append(k)
                pos.append(math.radians(x))
    return names, pos


def blendshape_positions(values: dict) -> tuple[list, list]:
    names, pos = [], []
    for n, c in zip(BLENDSHAPES, BS_COLS):
        x = _finite(values.get(c))
        if x is not None:
            names.append(n)
            pos.append(x)
    return names, pos


def emotion_positions(values: dict) -> tuple[list, list]:
    names, pos = [], []
    for e in EMOTIONS:
        x = _finite(values.get(f"emotion_{e}_pct"))
        if x is not None:
            names.append(e)
            pos.append(x / 100.0)
    return names, pos


class TandemPublisher:
    """Publishes the app's values; `publish` is called on the tracking thread after each result."""

    def __init__(self, node_name: str = "tandem", rig: str = "cogley"):
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        self._JointState = JointState
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.pubs = {k: self.node.create_publisher(JointState, topic, qos) for k, topic in TOPICS.items()}
        self._face = None
        self._rig = rig
        self.sent = 0

    def _servo_face(self):
        if self._face is None:
            from .animatronic import Face
            self._face = Face(head_to_chest=0.62, rig=self._rig)
        return self._face

    def _msg(self, names, positions, stamp):
        m = self._JointState()
        m.header.stamp = stamp
        m.header.frame_id = "tandem"
        m.name = list(names)
        m.position = [float(p) for p in positions]
        return m

    def _stamp(self, t_capture: float):
        """ROS time of the capture: now, minus how long ago (on the monotonic clock) the frame was captured."""
        from rclpy.duration import Duration
        age = max(0.0, time.monotonic() - t_capture)
        return (self.node.get_clock().now() - Duration(nanoseconds=int(age * 1e9))).to_msg()

    def publish(self, values: dict, t_capture: float, kind: str = "hand"):
        """kind: the model whose result this is (face and emotion results also send the face topics)."""
        stamp = self._stamp(t_capture)
        self.pubs["joints"].publish(self._msg(*joint_positions(values), stamp))
        if kind in ("face", "emotion"):
            names, pos = blendshape_positions(values)
            if names:
                self.pubs["blendshapes"].publish(self._msg(names, pos, stamp))
                face = self._servo_face()
                bs = dict(zip(names, pos))
                eyes = (_finite(values.get(k)) for k in ("eye_yaw", "eye_pitch"))     # degrees
                servos = face.servo_values(bs, *(0.0 if v is None else v for v in eyes))
                cmds = face.commands(servos) if face.rig == "cogley" else servos
                self.pubs["servos"].publish(self._msg(list(cmds), list(cmds.values()), stamp))
            names, pos = emotion_positions(values)
            if names:
                self.pubs["emotion"].publish(self._msg(names, pos, stamp))
        self.sent += 1

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass

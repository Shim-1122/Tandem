"""The other end of `python scripts/live.py --ros2`: a ROS 2 node that subscribes to Tandem's topics and drives
the simulated human (in its own process, so the tracker never waits on it), or forwards the joints to Isaac Sim.

    python scripts/ros2_avatar.py                       # the avatar in MuJoCo's viewer (mjpython on macOS)
    python scripts/ros2_avatar.py --video avatar.mp4 --seconds 20      # no window: record what it receives
    python scripts/ros2_avatar.py --isaac /joint_command --no-avatar   # joints to an Isaac Sim articulation

Run it where ROS 2 is sourced (source /opt/ros/<distro>/setup.bash). See vtrack/ros2.py for the topics.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtrack.face import EMOTIONS                  # noqa: E402
from vtrack.face_rig import BLENDSHAPES, BS_COLS  # noqa: E402
from vtrack.ros2 import TOPICS                    # noqa: E402

_BS_COL = dict(zip(BLENDSHAPES, BS_COLS))


class AvatarNode:
    """Keeps the newest values received (degrees, bs_ and emotion_ columns, as the tracker's values) and
    applies them to a HumanSim; optionally forwards the joints to another topic."""

    def __init__(self, node, sim=None, isaac_topic: str | None = None, rename: dict | None = None):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        self.node, self.sim, self.values, self.rename = node, sim, {}, rename or {}
        self.received = 0
        self.latency_ms = float("nan")
        self._JointState = JointState
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(JointState, TOPICS["joints"], self.on_joints, qos)
        node.create_subscription(JointState, TOPICS["blendshapes"], self.on_blendshapes, qos)
        node.create_subscription(JointState, TOPICS["emotion"], self.on_emotion, qos)
        self.isaac = node.create_publisher(JointState, isaac_topic, qos) if isaac_topic else None

    def _age(self, msg):
        now = self.node.get_clock().now().nanoseconds
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.latency_ms = (now - stamp) / 1e6

    def on_joints(self, msg):
        self._age(msg)
        self.received += 1
        for n, p in zip(msg.name, msg.position):
            self.values[n] = math.degrees(p)
        if self.isaac is not None:
            out = self._JointState()
            out.header = msg.header
            out.name = [self.rename.get(n, n) for n in msg.name]
            out.position = list(msg.position)
            self.isaac.publish(out)

    def on_blendshapes(self, msg):
        for n, p in zip(msg.name, msg.position):
            if n in _BS_COL:
                self.values[_BS_COL[n]] = p

    def on_emotion(self, msg):
        for n, p in zip(msg.name, msg.position):
            if n in EMOTIONS:
                self.values[f"emotion_{n}_pct"] = 100.0 * p

    def apply(self):
        if self.sim is not None and self.values:
            self.sim.command(dict(self.values))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-avatar", action="store_true", help="no simulated human (just the forwarding)")
    ap.add_argument("--isaac", metavar="TOPIC", help="also republish the joints (radians) on this topic")
    ap.add_argument("--rename", metavar="FILE", type=Path, help="JSON {tandem name: robot joint name} for --isaac")
    ap.add_argument("--face-rig", choices=["cogley", "full"], default="cogley")
    ap.add_argument("--no-mirror", action="store_true", help="copy left and right instead of mirroring")
    ap.add_argument("--video", type=Path, help="record the avatar to this MP4 instead of opening a viewer")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0: run until closed)")
    ap.add_argument("--fps", type=float, default=60.0)
    a = ap.parse_args(argv)
    import json

    import rclpy
    rclpy.init()
    node = rclpy.create_node("tandem_avatar")
    sim = None
    if not a.no_avatar:
        from vtrack.human import HumanSim
        sim = HumanSim(mirror=not a.no_mirror, rig=a.face_rig)
    rename = json.loads(a.rename.read_text()) if a.rename else None
    av = AvatarNode(node, sim, a.isaac, rename)
    t0, period = time.monotonic(), 1.0 / a.fps
    viewer = writer = None
    try:
        if sim is not None and a.video is None:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        elif sim is not None:
            import cv2
            writer = cv2.VideoWriter(str(a.video), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (960, 720))
        while rclpy.ok() and (a.seconds <= 0 or time.monotonic() - t0 < a.seconds):
            tick = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.0)
            if viewer is not None:
                if not viewer.is_running():
                    break
                with viewer.lock():
                    av.apply()
                viewer.sync()
            elif writer is not None:
                av.apply()
                writer.write(sim.render("split", 960, 720)[:, :, ::-1])
            rest = period - (time.monotonic() - tick)
            if rest > 0:
                rclpy.spin_once(node, timeout_sec=rest)
        print(f"received {av.received} joint messages; last latency from capture {av.latency_ms:.1f} ms")
    finally:
        if viewer is not None:
            viewer.close()
        if writer is not None:
            writer.release()
        if sim is not None:
            sim.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

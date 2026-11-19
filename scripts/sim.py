"""The Tandem robot in MuJoCo: play back a recording, or a built-in demo, on the simulated upper body.

    python scripts/sim.py                                  # demo motion in the interactive 3D viewer
    python scripts/sim.py --csv recordings/session.csv     # replay a recording made with live.py --record
    python scripts/sim.py --csv rec.csv --speed 0.5        # half speed
    python scripts/sim.py --video robot.mp4                # save a video instead of opening the viewer
    python scripts/sim.py --video robot.mp4 --camera eyes  # what the robot sees
    python scripts/sim.py --save-xml tandem.xml            # write the robot model (MJCF) to open elsewhere

    python scripts/sim.py --human                          # the full human body: legs, core and facial expressions
    python scripts/sim.py --human --csv rec.csv --video me.mp4 --camera split

For the robot copying you live, run the tracker with --sim instead: python scripts/live.py --sim
(and python scripts/live.py --sim human for the human body)

On a Mac the interactive viewer needs MuJoCo's own launcher: mjpython scripts/sim.py (videos work with python).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # let `python scripts/sim.py` find vtrack


def demo_values(t: float) -> dict:
    """A short loop of motion that uses every part: wave, reach, grip, turn the waist, look around and blink."""
    s = math.sin
    wave = s(2 * math.pi * 0.8 * t)
    grip = 0.5 - 0.5 * math.cos(2 * math.pi * 0.25 * t)          # 0 open .. 1 closed, every 4 s
    v = {"core_rotation": 25 * s(2 * math.pi * 0.1 * t),
         "neck_yaw": 30 * s(2 * math.pi * 0.13 * t), "neck_pitch": 8 * s(2 * math.pi * 0.21 * t),
         "neck_roll": 6 * s(2 * math.pi * 0.17 * t),
         "eye_yaw": 20 * s(2 * math.pi * 0.37 * t), "eye_pitch": 10 * s(2 * math.pi * 0.29 * t),
         "eyes_open": 0 if (t % 3.0) < 0.15 else 1,
         # right arm: wave
         "R_shoulder_flexion": 10, "R_shoulder_abduction": 95, "R_shoulder_rotation": -70 + 10 * wave,
         "R_elbow_flexion": 80 + 25 * wave, "R_forearm_rotation": 0, "R_wrist_flexion": 0, "R_wrist_deviation": 0,
         # left arm: reach forward and grip
         "L_shoulder_flexion": 50 + 15 * s(2 * math.pi * 0.25 * t), "L_shoulder_abduction": 10,
         "L_shoulder_rotation": 0, "L_elbow_flexion": 40, "L_forearm_rotation": 60, "L_wrist_flexion": 10,
         "L_wrist_deviation": 0}
    for side, g in (("R_", 0.0), ("L_", grip)):
        for f in ("index", "middle", "ring", "pinky"):
            v.update({f"{side}{f}_mcp_flex": 75 * g, f"{side}{f}_mcp_abd": 0, f"{side}{f}_pip": 95 * g,
                      f"{side}{f}_dip": 55 * g})
        v.update({f"{side}thumb_cmc_palmar": 20 + 30 * g, f"{side}thumb_cmc_radial": 30 - 20 * g,
                  f"{side}thumb_cmc_rotation": 40 + 40 * g, f"{side}thumb_mcp": 10 + 30 * g,
                  f"{side}thumb_ip": 10 + 30 * g})
    return v


def human_demo_values(t: float) -> dict:
    """The robot demo plus a step and a squat with the legs, a lean, and a face that goes through the emotions."""
    from vtrack.face_rig import EMOTION_NAMES
    s = math.sin
    v = demo_values(t)
    step = 0.5 - 0.5 * math.cos(2 * math.pi * 0.2 * t)
    squat = max(0.0, s(2 * math.pi * 0.07 * t)) ** 2
    v.update({"core_flexion": 10 + 25 * squat, "core_lateral_bend": 8 * s(2 * math.pi * 0.11 * t)})
    for p, k in (("R_", step), ("L_", 0.0)):
        v.update({p + "hip_flexion": 50 * k + 75 * squat, p + "hip_abduction": 4 + 6 * squat,
                  p + "hip_rotation": 0.0, p + "knee_flexion": 70 * k + 110 * squat,
                  p + "ankle_dorsiflexion": -10 * k + 25 * squat, p + "ankle_inversion": 0.0,
                  p + "ankle_abduction": 0.0})
    order = [e for e in EMOTION_NAMES if e != "neutral"]
    phase = (t / 2.5) % len(order)                 # 2.5 s per emotion, blending into the next
    i, f = int(phase), phase - int(phase)
    blend = min(1.0, max(0.0, (f - 0.7) / 0.3))
    for e in EMOTION_NAMES:
        v[f"emotion_{e}_pct"] = 0.0
    v[f"emotion_{order[i]}_pct"] = 100 * (1 - blend)
    v[f"emotion_{order[(i + 1) % len(order)]}_pct"] += 100 * blend
    v["eyes_open"] = 1
    return v


def frames(rows, speed: float):
    """(time, values) from CSV rows, with the recording's own timing."""
    t0 = None
    for r in rows:
        try:
            t = float(r["t"])
        except (KeyError, ValueError):
            continue
        t0 = t if t0 is None else t0
        yield (t - t0) / speed, r


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tandem robot simulation (MuJoCo)")
    ap.add_argument("--csv", type=Path, help="recording from live.py --record (default: the demo motion)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    ap.add_argument("--hand-side", choices=["right", "left"], default="right",
                    help="arm driven by one-arm or hand-view recordings (their columns have no L_ or R_)")
    ap.add_argument("--seconds", type=float, default=20.0, help="demo length (the viewer loops it until closed)")
    ap.add_argument("--video", type=Path, help="render to this video file instead of opening the viewer")
    ap.add_argument("--camera", choices=["front", "side", "top", "eyes", "split", "face", "three_quarter"],
                    default=None, help="video camera (robot: front; human: split, the body with the face close-up)")
    ap.add_argument("--human", action="store_true", help="the full human body instead of the robot")
    ap.add_argument("--mode", choices=["kinematic", "physics"], default="kinematic",
                    help="human: joints straight on the angles (kinematic) or servos under gravity (physics)")
    ap.add_argument("--no-mirror", action="store_true", help="human: copy left and right instead of mirroring")
    ap.add_argument("--size", default="1280x720", help="video size, WIDTHxHEIGHT")
    ap.add_argument("--fps", type=int, default=30, help="video frame rate")
    ap.add_argument("--no-scene", action="store_true", help="the robot alone (no floor, table or ball)")
    ap.add_argument("--save-xml", type=Path, help="write the robot model (MJCF) and exit")
    a = ap.parse_args(argv)
    if a.speed <= 0:
        ap.error("--speed must be more than 0")

    from vtrack.sim import TandemSim, build_xml, read_csv
    if a.save_xml:
        if a.human:
            from vtrack.human import build_xml as human_xml
            text = human_xml(floating=a.mode == "kinematic", scene=not a.no_scene)
        else:
            text = build_xml(scene=not a.no_scene)
        a.save_xml.write_text(text, encoding="utf-8")
        print(f"wrote {a.save_xml}")
        return
    if a.human:
        from vtrack.human import HumanSim
        sim = HumanSim(mode=a.mode, scene=not a.no_scene, hand_side=a.hand_side, mirror=not a.no_mirror)
        demo = human_demo_values
    else:
        sim = TandemSim(scene=not a.no_scene, hand_side=a.hand_side)
        demo = demo_values
    camera = a.camera or ("split" if a.human else "front")
    if not a.human and camera not in ("front", "side", "top", "eyes"):
        ap.error(f"--camera {camera} is a human view; add --human")
    if a.csv:
        rows = read_csv(a.csv)
        timeline = list(frames(rows, a.speed))
        if not timeline:
            raise SystemExit(f"{a.csv} has no rows with a time column")
        length = timeline[-1][0]
        print(f"{a.csv}: {len(timeline)} rows, {length:.1f} s at speed {a.speed}")
    else:
        timeline, length = None, a.seconds

    i = 0

    def values_at(t):
        nonlocal i
        if timeline is None:
            return demo(t)
        if t < (timeline[i][0] if i < len(timeline) else 0):
            i = 0                                     # looped back to the start
        while i + 1 < len(timeline) and timeline[i + 1][0] <= t:
            i += 1
        return timeline[i][1]

    if a.video:
        import cv2
        w, h = (int(x) for x in a.size.lower().split("x"))
        out = cv2.VideoWriter(str(a.video), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
        n = int(length * a.fps) + 1
        for k in range(n):
            t = k / a.fps
            sim.command(values_at(t))
            sim.step(1.0 / a.fps)
            out.write(sim.render(camera, w, h)[:, :, ::-1])
        out.release()
        sim.close()
        print(f"wrote {a.video}: {n} frames")
        return

    import mujoco.viewer
    try:
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    except RuntimeError as e:
        raise SystemExit(f"{e}\nOn a Mac, run it with mjpython: mjpython scripts/sim.py ...")
    print("viewer open: drag to orbit, right-drag to pan, scroll to zoom; close the window to stop")
    start = last = time.monotonic()
    with viewer:
        while viewer.is_running():
            now = time.monotonic()
            t = (now - start) % max(length, 1e-3)
            sim.command(values_at(t))
            sim.step(now - last)
            last = now
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - now)))


if __name__ == "__main__":
    main()

"""Live webcam demo: 7 arm joint angles (shoulder 3, elbow 1, wrist 3), gestures, optional CSV recording.

    python scripts/live.py --side right --record recordings/run1.csv

Stand 1.5 to 2.5 m from the camera with the tracked arm and torso in view, facing the camera.
Keys: q quit, c print the axis-convention and hand/pose frame agreement checks to the terminal.
The image is mirrored for display only; tracking uses the raw frame and the person's own left/right.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # let `python scripts/live.py` find vtrack

import cv2
import numpy as np

from vtrack import (ArmKinematics, Gesture, GestureDebouncer, JointChannel, LandmarkSmoother, SlewLimiter,
                    classify_hand, frame_agreement_deg)
from vtrack.kinematics import ArmAngles, thorax_frame
from vtrack.tracker import Tracker, pick_hand

ANGLE_FIELDS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion",
                "forearm_rotation", "wrist_flexion", "wrist_deviation"]
LABELS = {"shoulder_flexion": "Shoulder flex(+)/ext(-)  sagittal", "shoulder_abduction": "Shoulder abd(+)/add(-)  coronal",
          "shoulder_rotation": "Shoulder int(+)/ext(-) rot  transverse", "elbow_flexion": "Elbow flexion",
          "forearm_rotation": "Forearm pronation(+)/supination(-)", "wrist_flexion": "Wrist flex(+)/ext(-)",
          "wrist_deviation": "Wrist radial(+)/ulnar(-) dev"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--side", choices=["left", "right"], default="right", help="the person's own arm to track")
    ap.add_argument("--pose-only-wrist", action="store_true", help="ignore the hand model; noisier wrist angles")
    ap.add_argument("--record", type=Path, help="write per-frame angles to CSV")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.camera}")
    tracker = Tracker()
    kin = ArmKinematics(a.side)
    pose_smooth = LandmarkSmoother(min_cutoff=1.5, beta=0.1)
    hand_smooth = LandmarkSmoother(min_cutoff=2.0, beta=0.2)
    debounce = GestureDebouncer(hold=5)
    elbow_ch, elbow_slew = JointChannel("elbow", 0, 140), SlewLimiter(max_rate=2.0)
    writer = None
    if a.record:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        f = open(a.record, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["t", "gesture", *ANGLE_FIELDS, "shoulder_rotation_valid", "wrist_valid", "elbow_cmd"])
    t0, last_ts, want_check = time.monotonic(), -1, False
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = time.monotonic() - t0
        ts = max(int(t * 1000), last_ts + 1)
        last_ts = ts
        res = tracker.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ts)
        lines, gesture, ang, cmd = [], Gesture.UNKNOWN.value, ArmAngles(), float("nan")
        hips_seen, up_hint = True, None
        hand = pick_hand(res, a.side)
        hand_w = hand_smooth(hand.world, t) if hand is not None else None
        if hand_w is not None:
            debounce.update(classify_hand(hand_w))
            gesture = debounce.current.value
        if res.pose_world is not None:
            pose_w = pose_smooth(res.pose_world, t)
            hips_seen = res.pose_visibility is not None and min(res.pose_visibility[23], res.pose_visibility[24]) > 0.5
            up_hint = None if hips_seen else np.array([0.0, -1.0, 0.0])   # camera vertical (world y points down)
            ang = kin.solve(pose_w, None if a.pose_only_wrist else hand_w, up_hint)
            cmd = elbow_slew(elbow_ch.command(ang.elbow_flexion), t)
            if want_check:
                fwd = thorax_frame(pose_w, up_hint)[0][:, 0]
                print("hips visible:", hips_seen, "(if False, torso up is taken from the camera's vertical)")
                print("torso forward in camera axes:", np.round(fwd, 2),
                      "(facing the camera, z should be clearly negative; if positive, the world axes are mirrored and"
                      " every sagittal sign is flipped)")
                if hand_w is not None:
                    print("frame agreement (deg):", {k: round(v, 1) for k, v in frame_agreement_deg(pose_w, hand_w, a.side).items()})
                want_check = False
            for k in ANGLE_FIELDS:
                v = getattr(ang, k)
                mark = ""
                if k == "shoulder_rotation" and not ang.shoulder_rotation_valid:
                    mark = "  (held)"
                if k in ("forearm_rotation", "wrist_flexion", "wrist_deviation") and not ang.wrist_valid:
                    mark = "  (no hand)"
                lines.append(f"{LABELS[k]}: {v:7.1f}{mark}")
        lines.append(f"gesture: {gesture}")
        if res.pose_world is not None and not hips_seen:
            lines.append("hips not visible: torso up = camera vertical, keep the camera level")
        if writer:
            writer.writerow([f"{t:.3f}", gesture, *[round(getattr(ang, k), 2) for k in ANGLE_FIELDS],
                             int(ang.shoulder_rotation_valid), int(ang.wrist_valid), round(cmd, 3)])
        view = cv2.flip(bgr, 1)                    # mirror first, then draw text so it reads normally
        scale = max(0.6, view.shape[1] / 1280 * 0.9)
        step = int(34 * scale)
        cv2.rectangle(view, (0, 0), (int(620 * scale), step * len(lines) + 12), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            cv2.putText(view, text, (10, step * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)
        cv2.imshow("vision-joint-tracker", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            want_check = True
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()

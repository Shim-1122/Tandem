"""Live tracking with a wireframe overlay and a 3D kinematics panel.

Body mode (default): seven arm joint angles from a webcam that sees your shoulders and arm.
    python scripts/live.py --side right                 # standing, hips in view
    python scripts/live.py --side right --torso desk    # seated at a desk, hips hidden
POV mode: hands only, for a head-mounted camera (goggles). Hand position and orientation, finger joints.
    python scripts/live.py --mode pov --hand right --fov 70

Keys (click the video window first):
    q  quit
    z  body: calibrate torso up from the neutral pose (needed for --torso desk)
       pov:  set the current hand pose as zero for relative motion
    c  body: print axis and hand/pose agreement checks to the terminal
    w  toggle the wireframe overlay
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

from vtrack import (ArmKinematics, Gesture, GestureDebouncer, LandmarkSmoother, classify_hand,
                    frame_agreement_deg)
from vtrack.draw import (CYAN, HAND_EDGES, POSE_ARM_EDGES, YELLOW, draw_wireframe, render_arm_panel,
                         render_hand_panel, text_block)
from vtrack.handpose import HandPoseZero, all_finger_angles, camera_matrix, hand_frame, hand_pose_camera
from vtrack.kinematics import ArmAngles, arm_visible, out_of_range, thorax_frame, up_from_neutral
from vtrack.tracker import Tracker, pick_hand

ANGLE_FIELDS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion",
                "forearm_rotation", "wrist_flexion", "wrist_deviation"]
LABELS = {"shoulder_flexion": "Shoulder flex+/ext-  (sagittal)", "shoulder_abduction": "Shoulder abd+/add-  (coronal)",
          "shoulder_rotation": "Shoulder int+/ext- rot (transverse)", "elbow_flexion": "Elbow flexion",
          "forearm_rotation": "Forearm pronation+/supination-", "wrist_flexion": "Wrist flex+/ext-",
          "wrist_deviation": "Wrist radial+/ulnar-"}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def compose(video, panel, max_width=1600):
    h = video.shape[0]
    panel = cv2.resize(panel, (h, h))
    canvas = np.hstack([video, panel])
    if canvas.shape[1] > max_width:
        f = max_width / canvas.shape[1]
        canvas = cv2.resize(canvas, (max_width, int(canvas.shape[0] * f)))
    return canvas


def pick_hand_pov(res, hand: str, swap: bool):
    """MediaPipe labels handedness for a mirrored image; raw camera frames are not mirrored, so swap."""
    if not res.hands:
        return None
    want = {"right": "Left", "left": "Right"}[hand]
    if swap:
        want = "Right" if want == "Left" else "Left"
    matches = [h for h in res.hands if h.handedness == want]
    return matches[0] if matches else (res.hands[0] if len(res.hands) == 1 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["body", "pov"], default="body")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--side", choices=["left", "right"], default="right", help="body mode: your arm to track")
    ap.add_argument("--torso", choices=["auto", "hips", "desk"], default="auto",
                    help="body mode: where torso 'up' comes from. auto = hips if visible, else calibration (z), "
                         "else camera vertical. desk = calibration (z), else camera vertical")
    ap.add_argument("--pose-only-wrist", action="store_true", help="body mode: ignore the hand model for the wrist")
    ap.add_argument("--hand", choices=["left", "right"], default="right", help="pov mode: which hand to follow")
    ap.add_argument("--swap-hands", action="store_true", help="pov mode: use if it follows the wrong hand")
    ap.add_argument("--fov", type=float, default=70.0, help="pov mode: camera horizontal field of view, degrees")
    ap.add_argument("--record", type=Path, help="write per-frame values to CSV")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.camera}")
    pov = a.mode == "pov"
    tracker = Tracker(num_hands=2, pose=None if pov else "pose_lite")
    kin = ArmKinematics(a.side)
    pose_smooth = LandmarkSmoother(min_cutoff=1.5, beta=0.1)
    hand_smooth = LandmarkSmoother(min_cutoff=2.0, beta=0.2)
    debounce = GestureDebouncer(hold=5)
    zero = HandPoseZero()
    up_cal, show_wire, want_check, want_zero = None, True, False, False
    K = None

    writer = None
    if a.record:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        writer = csv.writer(open(a.record, "w", newline=""))
        if pov:
            writer.writerow(["t", "gesture", "x_m", "y_m", "z_m", "dx_cm", "dy_cm", "dz_cm", "twist", "pitch", "yaw"]
                            + [f"{f}_{j}" for f in FINGER_ORDER for j in ("mcp", "pip", "dip")])
        else:
            writer.writerow(["t", "gesture", *ANGLE_FIELDS, "shoulder_rotation_valid", "wrist_valid", "torso_source"])

    t0, last_ts = time.monotonic(), -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if K is None:
            K = camera_matrix(bgr.shape[1], bgr.shape[0], a.fov)
        t = time.monotonic() - t0
        ts = max(int(t * 1000), last_ts + 1)
        last_ts = ts
        res = tracker.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ts)
        lines, gesture = [], Gesture.UNKNOWN.value

        if show_wire:
            if res.pose_image is not None:
                draw_wireframe(bgr, res.pose_image, POSE_ARM_EDGES, CYAN, res.pose_visibility)
            for h in res.hands:
                draw_wireframe(bgr, h.image, HAND_EDGES, YELLOW, radius=3)

        if not pov:
            # ---------------- body mode ----------------
            ang, torso_src = ArmAngles(), "-"
            hand = pick_hand(res, a.side)
            hand_w = hand_smooth(hand.world, t) if hand is not None else None
            if hand_w is not None:
                g = classify_hand(hand_w)
                debounce.update(g)
                gesture = debounce.current.value if g == debounce.current else "unknown"
            if res.pose_world is not None:
                pose_w = pose_smooth(res.pose_world, t)
                vis = res.pose_visibility
                hips_seen = vis is not None and min(vis[23], vis[24]) > 0.5
                missing = arm_visible(vis, a.side)
                if want_zero:
                    up_cal = up_from_neutral(pose_w, a.side)
                    kin.reset()
                    print("calibrated torso up from neutral pose:", np.round(up_cal, 3))
                    want_zero = False
                if a.torso == "hips" or (a.torso == "auto" and hips_seen):
                    up_hint, torso_src = None, "hips"
                elif up_cal is not None:
                    up_hint, torso_src = up_cal, "calibrated"
                else:
                    up_hint, torso_src = np.array([0.0, -1.0, 0.0]), "camera vertical"
                if missing:
                    ang = ArmAngles()          # do not show angles built on guessed landmarks
                else:
                    ang = kin.solve(pose_w, None if a.pose_only_wrist else hand_w, up_hint)
                if want_check:
                    fwd = thorax_frame(pose_w, up_hint)[0][:, 0]
                    print(f"torso up from: {torso_src}; forward in camera axes: {np.round(fwd, 2)} "
                          "(facing the camera, z should be clearly negative)")
                    if hand_w is not None:
                        print("agreement (deg):", {k: round(v, 1) for k, v in
                                                   frame_agreement_deg(pose_w, hand_w, a.side).items()},
                              "(hand_vs_forearm under ~20 with a straight wrist is good)")
                    want_check = False
                if missing:
                    lines.append(f"{', '.join(missing)} not in view: move back so the whole {a.side} arm is visible")
                bad = set(out_of_range(ang))
                for k in ANGLE_FIELDS if not missing else []:
                    v, mark = getattr(ang, k), ""
                    if k in bad:
                        mark = " (outside normal range: check tracking)"
                    if k == "shoulder_rotation" and not ang.shoulder_rotation_valid:
                        mark = " (held)"
                    if k in ("forearm_rotation", "wrist_flexion", "wrist_deviation") and not ang.wrist_valid:
                        mark = " (no hand)"
                    lines.append(f"{LABELS[k]}: {v:6.1f}{mark}")
                lines.append(f"torso up: {torso_src}" + ("   press z in neutral pose to calibrate"
                                                         if torso_src == "camera vertical" else ""))
            else:
                lines.append("no body detected: shoulders and tracked arm must be in view")
            lines.append(f"gesture: {gesture}")
            if res.pose_world is not None and missing:
                note = "arm not fully visible: no angles"
            elif not ang.shoulder_rotation_valid and np.isfinite(ang.shoulder_flexion):
                note = "shoulder rotation held (elbow straight or arm out sideways)"
            else:
                note = ""
            panel = render_arm_panel(ang, a.side, size=480, note=note)
            if writer:
                writer.writerow([f"{t:.3f}", gesture, *[round(getattr(ang, k), 2) for k in ANGLE_FIELDS],
                                 int(ang.shoulder_rotation_valid), int(ang.wrist_valid), torso_src])
            view = cv2.flip(bgr, 1)          # mirror view for a camera facing you
        else:
            # ---------------- POV mode ----------------
            hand = pick_hand_pov(res, a.hand, a.swap_hands)
            world = hand_smooth(hand.world, t) if hand is not None else None
            row = None
            if world is not None:
                g = classify_hand(world)
                debounce.update(g)
                gesture = debounce.current.value if g == debounce.current else "unknown"
                R, tvec = hand_pose_camera(hand.world, hand.image, K)
                if R is not None:
                    R_hand = R @ hand_frame(world)
                    if want_zero:
                        zero.zero(R_hand, tvec)
                        print("zeroed hand pose at", np.round(tvec, 3), "m")
                        want_zero = False
                    lines.append(f"hand at x {tvec[0]*100:6.1f}  y {tvec[1]*100:6.1f}  z {tvec[2]*100:6.1f} cm "
                                 "(right, down, away)")
                    m = zero.relative(R_hand, tvec)
                    if m is not None:
                        lines.append(f"moved dx {m.dx_cm:6.1f}  dy {m.dy_cm:6.1f}  dz {m.dz_cm:6.1f} cm")
                        lines.append(f"turned twist {m.twist:6.1f}  pitch {m.pitch:6.1f}  yaw {m.yaw:6.1f} deg")
                    else:
                        lines.append("press z to set the zero pose for relative motion")
                    fa = all_finger_angles(world)
                    for f in FINGER_ORDER:
                        j = fa[f]
                        lines.append(f"{f:6s} {j['mcp']:5.0f} {j['pip']:5.0f} {j['dip']:5.0f}")
                    row = [tvec, m, fa]
            else:
                lines.append(f"no {a.hand} hand in view")
            lines.append(f"gesture: {gesture}")
            panel = render_hand_panel(world, size=480, note="finger rows: MCP PIP DIP flexion (deg)")
            if writer and row is not None:
                tvec, m, fa = row
                rel = [m.dx_cm, m.dy_cm, m.dz_cm, m.twist, m.pitch, m.yaw] if m else [np.nan] * 6
                writer.writerow([f"{t:.3f}", gesture, *np.round(tvec, 4), *np.round(rel, 2),
                                 *[round(fa[f][j], 1) for f in FINGER_ORDER for j in ("mcp", "pip", "dip")]])
            view = bgr                       # a POV camera is not mirrored

        text_block(view, lines)
        cv2.imshow("Homunculus Shadowing", compose(view, panel))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            want_check = True
        if key == ord("z"):
            want_zero = True
        if key == ord("w"):
            show_wire = not show_wire
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()

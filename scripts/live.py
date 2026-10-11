"""Live tracking with a wireframe overlay and a 3D kinematics panel.

Body mode (default): seven arm joint angles per arm, plus finger joints, from a webcam that sees you.
    python scripts/live.py --side right                 # standing, hips in view
    python scripts/live.py --side right --torso desk    # seated at a desk, hips hidden
    python scripts/live.py --side both --torso desk     # both arms at once
POV mode: hands only, for a head-mounted camera (goggles). Hand position and orientation, finger joints.
    python scripts/live.py --mode pov --hand right --fov 70

Options: --pose-model lite|full|heavy (accuracy vs speed, default full),
         --smoothing responsive|balanced|stable (lag vs jitter, default balanced).

Keys (click the video window first):
    q  quit
    z  body: calibrate torso up from the neutral pose (needed for --torso desk)
       pov:  set the current hand pose as zero for relative motion
    f  body: cycle the finger joint table (off, then each tracked hand)
    c  body: print axis and hand/pose agreement checks to the terminal
    w  toggle the wireframe overlay
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # let `python scripts/live.py` find vtrack

import cv2
import numpy as np

from vtrack import ArmKinematics, Gesture, GestureDebouncer, LandmarkSmoother, classify_hand, frame_agreement_deg
from vtrack.draw import (CYAN, HAND_EDGES, POSE_ARM_EDGES, YELLOW, draw_wireframe, render_arms_panel,
                         render_hand_panel, text_block)
from vtrack.elbow import ElbowPredictor, SegmentLengths
from vtrack.filters import HAND_SMOOTHING, SMOOTHING_PRESETS, JitterMeter, make_arm_smoothers
from vtrack.fingers import FINGER_KEYS, THUMB_KEYS, HandKinematics, flat, hand_fk, hand_geometry, palm_to_world
from vtrack.handpose import HandPoseZero, camera_matrix, hand_frame, hand_pose_camera
from vtrack.kinematics import _SIDE, ArmAngles, out_of_range, thorax_frame, up_from_neutral
from vtrack.tracker import Tracker, pick_hand

ANGLE_FIELDS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion",
                "forearm_rotation", "wrist_flexion", "wrist_deviation"]
LABELS = {"shoulder_flexion": "Shoulder flex+/ext-  (sagittal)", "shoulder_abduction": "Shoulder abd+/add-  (coronal)",
          "shoulder_rotation": "Shoulder int+/ext- rot (transverse)", "elbow_flexion": "Elbow flexion",
          "forearm_rotation": "Forearm pronation+/supination-", "wrist_flexion": "Wrist flex+/ext-",
          "wrist_deviation": "Wrist radial+/ulnar-"}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
VIS = 0.6


def compose(video, panel, max_width=1600):
    h = video.shape[0]
    canvas = np.hstack([video, cv2.resize(panel, (h, h))])
    if canvas.shape[1] > max_width:
        f = max_width / canvas.shape[1]
        canvas = cv2.resize(canvas, (max_width, int(canvas.shape[0] * f)))
    return canvas


def finger_table(angles: dict, title: str) -> list[str]:
    lines = [f"{title}\tMCP/CMC\tside\tPIP/MCP\tDIP/IP"]
    th = angles["thumb"]
    lines.append(f"thumb (rot {th['cmc_rotation']:.0f})\t{th['cmc_palmar']:.0f}\t{th['cmc_radial']:.0f}"
                 f"\t{th['mcp']:.0f}\t{th['ip']:.0f}")
    for f in FINGER_ORDER[1:]:
        a = angles[f]
        lines.append(f"{f}\t{a['mcp_flex']:.0f}\t{a['mcp_abd']:.0f}\t{a['pip']:.0f}\t{a['dip']:.0f}")
    return lines


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
    ap.add_argument("--side", choices=["left", "right", "both"], default="right", help="body mode: arm(s) to track")
    ap.add_argument("--torso", choices=["auto", "hips", "desk"], default="auto",
                    help="body mode: where torso 'up' comes from. auto = hips if visible, else calibration (z), "
                         "else camera vertical. desk = calibration (z), else camera vertical")
    ap.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="full",
                    help="body mode: pose model size; heavier is more accurate and slower")
    ap.add_argument("--smoothing", choices=list(SMOOTHING_PRESETS), default="balanced",
                    help="body mode: responsive = least lag, stable = least jitter")
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
    tracker = Tracker(num_hands=2, pose=None if pov else f"pose_{a.pose_model}")

    sides = ["left", "right"] if a.side == "both" else [a.side]
    pose_smooth, _ = make_arm_smoothers(a.smoothing)
    kins = {sd: ArmKinematics(sd) for sd in sides}
    elbows = {sd: ElbowPredictor() for sd in sides}
    seglens = {sd: SegmentLengths() for sd in sides}
    ang_smooth = {sd: make_arm_smoothers(a.smoothing)[1] for sd in sides}
    jitter = {sd: JitterMeter(1.0) for sd in sides}
    hand_smooths = {sd: LandmarkSmoother(**HAND_SMOOTHING) for sd in sides}
    hand_kins = {sd: HandKinematics(sd) for sd in sides}
    debounces = {sd: GestureDebouncer(hold=5) for sd in sides}
    pov_smooth, pov_debounce, pov_hk = LandmarkSmoother(**HAND_SMOOTHING), GestureDebouncer(hold=5), HandKinematics(a.hand)
    zero = HandPoseZero()
    up_cal, show_wire, want_check, want_zero, K = None, True, False, False, None
    table_cycle = [None] + sides
    table_i = 0

    writer = None
    if a.record:
        a.record.parent.mkdir(parents=True, exist_ok=True)
        writer = csv.writer(open(a.record, "w", newline=""))
        fcols = [f"{f}_{k}" for f in ["index", "middle", "ring", "pinky"] for k in FINGER_KEYS] + \
                [f"thumb_{k}" for k in THUMB_KEYS]
        if pov:
            writer.writerow(["t", "gesture", "x_m", "y_m", "z_m", "dx_cm", "dy_cm", "dz_cm", "twist", "pitch", "yaw"]
                            + fcols)
        else:
            cols = []
            for sd in sides:
                p = sd[0].upper() + "_" if len(sides) > 1 else ""
                cols += [p + "gesture", *[p + k for k in ANGLE_FIELDS], p + "shoulder_rotation_valid",
                         p + "wrist_valid", p + "elbow_predicted", *[p + c for c in fcols]]
            writer.writerow(["t", *cols, "torso_source"])

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
        lines = []

        if show_wire:
            if res.pose_image is not None:
                draw_wireframe(bgr, res.pose_image, POSE_ARM_EDGES, CYAN, res.pose_visibility)
            for h in res.hands:
                draw_wireframe(bgr, h.image, HAND_EDGES, YELLOW, radius=3)

        if not pov:
            # ---------------- body mode ----------------
            angs = {sd: ArmAngles() for sd in sides}
            gestures = {sd: Gesture.UNKNOWN.value for sd in sides}
            missing = {sd: [] for sd in sides}
            fingers, finger_geo, hands_w = {}, {}, {}
            torso_src = "-"
            for sd in sides:
                hand = pick_hand(res, sd)
                hands_w[sd] = hand_smooths[sd](hand.world, t) if hand is not None else None
                if hands_w[sd] is not None:
                    g = classify_hand(hands_w[sd])
                    debounces[sd].update(g)
                    gestures[sd] = debounces[sd].current.value if g == debounces[sd].current else "unknown"
                    fingers[sd] = hand_kins[sd].solve(hands_w[sd])
                    finger_geo[sd] = hand_geometry(hands_w[sd], sd)
            if res.pose_world is not None:
                pose_w = pose_smooth(res.pose_world, t)
                vis = res.pose_visibility
                hips_seen = vis is not None and min(vis[23], vis[24]) > 0.5
                if a.torso == "hips" or (a.torso == "auto" and hips_seen):
                    up_hint, torso_src = None, "hips"
                elif up_cal is not None:
                    up_hint, torso_src = up_cal, "calibrated"
                else:
                    up_hint, torso_src = np.array([0.0, -1.0, 0.0]), "camera vertical"
                torso_up = thorax_frame(pose_w, up_hint)[0][:, 1]
                shoulder_width = float(np.linalg.norm(pose_w[11] - pose_w[12]))
                for sd in sides:
                    s_i, e_i, w_i = _SIDE[sd]
                    missing[sd] = [n for n, i in (("shoulder", s_i), ("wrist", w_i)) if vis[i] < VIS]
                    if missing[sd]:
                        ang_smooth[sd].reset()
                        continue
                    if min(vis[s_i], vis[e_i], vis[w_i]) >= 0.8:
                        seglens[sd].add(pose_w[s_i], pose_w[e_i], pose_w[w_i])
                    est = elbows[sd].estimate(pose_w[s_i], pose_w[w_i], seglens[sd].get(shoulder_width),
                                              guess=pose_w[e_i], guess_vis=float(vis[e_i]), up=torso_up, t=t)
                    pose_sd = pose_w.copy()
                    pose_sd[e_i] = est.elbow
                    raw = kins[sd].solve(pose_sd, None if a.pose_only_wrist else hands_w[sd], up_hint)
                    sm = ang_smooth[sd]({k: getattr(raw, k) for k in ANGLE_FIELDS}, t)
                    angs[sd] = dataclasses.replace(raw, **sm, elbow_predicted=est.mode != "measured")
                    jitter[sd].add(sm, t)
                if want_zero:
                    usable = [sd for sd in sides if not missing[sd] and not angs[sd].elbow_predicted]
                    if usable:
                        up_cal = up_from_neutral(pose_w, "both" if len(usable) == 2 else usable[0])
                        for k_ in kins.values():
                            k_.reset()
                        for s_ in ang_smooth.values():
                            s_.reset()
                        print("calibrated torso up from neutral pose using", "+".join(usable), "arm:",
                              np.round(up_cal, 3))
                    else:
                        print("calibration skipped: shoulders, elbows and wrists must all be in view")
                    want_zero = False
                if want_check:
                    fwd = thorax_frame(pose_w, up_hint)[0][:, 0]
                    print(f"torso up from: {torso_src}; forward in camera axes: {np.round(fwd, 2)} "
                          "(facing the camera, z should be clearly negative)")
                    for sd in sides:
                        print(f"{sd} segment lengths (m): {np.round(seglens[sd].get(shoulder_width), 3)}",
                              "(measured)" if seglens[sd].calibrated else "(estimated from shoulder width)")
                        if hands_w[sd] is not None:
                            print(f"{sd} agreement (deg):", {k: round(v, 1) for k, v in
                                                             frame_agreement_deg(pose_w, hands_w[sd], sd).items()},
                                  "(hand_vs_forearm under ~20 with a straight wrist is good)")
                    want_check = False

                for sd in sides:
                    if missing[sd]:
                        lines.append(f"{sd}: {' and '.join(missing[sd])} not in view")
                if len(sides) == 1:
                    sd = sides[0]
                    bad, jit = set(out_of_range(angs[sd])), jitter[sd].std()
                    for k in ANGLE_FIELDS if not missing[sd] else []:
                        v, mark = getattr(angs[sd], k), ""
                        if k == "shoulder_rotation" and not angs[sd].shoulder_rotation_valid:
                            mark = " held"
                        if k in ("forearm_rotation", "wrist_flexion", "wrist_deviation") and not angs[sd].wrist_valid:
                            mark = " no hand"
                        if k in bad:
                            mark = " check tracking"
                        j = jit.get(k, float("nan"))
                        lines.append(f"{LABELS[k]}\t{v:.1f}\t{'+-%.1f' % j if np.isfinite(j) else ''}{mark}")
                    if not missing[sd] and angs[sd].elbow_predicted:
                        lines.append("elbow out of view: predicted from shoulder, wrist and arm lengths")
                    lines.append("+- = jitter over the last second; hold still to read it")
                else:
                    bads = {sd: set(out_of_range(angs[sd])) for sd in sides}
                    lines.append("\tLEFT\tRIGHT")
                    for k in ANGLE_FIELDS:
                        cells = []
                        for sd in sides:
                            v = getattr(angs[sd], k)
                            flag = "!" if k in bads[sd] else ("h" if k == "shoulder_rotation" and np.isfinite(v)
                                                               and not angs[sd].shoulder_rotation_valid else "")
                            cells.append(f"{v:.1f}{flag}" if np.isfinite(v) else "--")
                        lines.append(f"{LABELS[k]}\t{cells[0]}\t{cells[1]}")
                    pred = [sd for sd in sides if angs[sd].elbow_predicted]
                    jl = []
                    for sd in sides:
                        js = [v for v in jitter[sd].std().values() if np.isfinite(v)]
                        jl.append(f"{sd} {max(js):.1f}" if js and not missing[sd] else f"{sd} --")
                    lines.append("jitter, worst joint over 1 s (deg): " + "   ".join(jl))
                    lines.append("h = shoulder rotation held, ! = outside normal range"
                                 + (f", elbow predicted: {', '.join(pred)}" if pred else ""))
                lines.append(f"torso up: {torso_src}" + ("   press z in neutral pose to calibrate"
                                                         if torso_src == "camera vertical" else ""))
            else:
                lines.append("no body detected: shoulders and tracked arm(s) must be in view")
            lines.append("gesture: " + "   ".join(f"{sd} {gestures[sd]}" for sd in sides)
                         if len(sides) > 1 else f"gesture: {gestures[sides[0]]}")
            tsd = table_cycle[table_i]
            if tsd is not None:
                lines += finger_table(fingers[tsd], f"{tsd} fingers (deg)") if tsd in fingers \
                    else [f"{tsd} hand not detected"]
            else:
                lines.append("press f for finger joint angles")
            held = [sd for sd in sides if np.isfinite(angs[sd].shoulder_flexion) and not angs[sd].shoulder_rotation_valid]
            note = ("shoulder rotation held: " + ", ".join(held)) if held else ""
            panel = render_arms_panel(angs, size=480, note=note,
                                      fingers_by_side={sd: (fingers[sd], finger_geo[sd]) for sd in fingers})
            if writer:
                row = []
                for sd in sides:
                    fv = flat(fingers[sd]) if sd in fingers else {}
                    row += [gestures[sd], *[round(getattr(angs[sd], k), 2) for k in ANGLE_FIELDS],
                            int(angs[sd].shoulder_rotation_valid), int(angs[sd].wrist_valid),
                            int(angs[sd].elbow_predicted),
                            *[round(fv.get(c, float("nan")), 1) for c in
                              [f"{f}_{k}" for f in ["index", "middle", "ring", "pinky"] for k in FINGER_KEYS]
                              + [f"thumb_{k}" for k in THUMB_KEYS]]]
                writer.writerow([f"{t:.3f}", *row, torso_src])
            view = cv2.flip(bgr, 1)          # mirror view for a camera facing you
        else:
            # ---------------- POV mode ----------------
            hand = pick_hand_pov(res, a.hand, a.swap_hands)
            world = pov_smooth(hand.world, t) if hand is not None else None
            gesture, solved, row = Gesture.UNKNOWN.value, None, None
            if world is not None:
                g = classify_hand(world)
                pov_debounce.update(g)
                gesture = pov_debounce.current.value if g == pov_debounce.current else "unknown"
                fa = pov_hk.solve(world)
                solved = palm_to_world(hand_fk(fa, hand_geometry(world, a.hand)), world, a.hand)
                R, tvec = hand_pose_camera(hand.world, hand.image, K)
                m = None
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
                lines += finger_table(fa, f"{a.hand} fingers (deg)")
                row = [tvec, m, fa]
            else:
                lines.append(f"no {a.hand} hand in view")
            lines.append(f"gesture: {gesture}")
            panel = render_hand_panel(world, size=480, solved=solved,
                                      note="grey: measured   yellow: rebuilt from solved joint angles")
            if writer and row is not None and row[0] is not None:
                tvec, m, fa = row
                rel = [m.dx_cm, m.dy_cm, m.dz_cm, m.twist, m.pitch, m.yaw] if m else [np.nan] * 6
                fv = flat(fa)
                writer.writerow([f"{t:.3f}", gesture, *np.round(tvec, 4), *np.round(rel, 2),
                                 *[round(fv[c], 1) for c in
                                   [f"{f}_{k}" for f in ["index", "middle", "ring", "pinky"] for k in FINGER_KEYS]
                                   + [f"thumb_{k}" for k in THUMB_KEYS]]])
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
        if key == ord("f"):
            table_i = (table_i + 1) % len(table_cycle)
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()

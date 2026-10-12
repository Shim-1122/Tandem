"""The live application: threaded capture and models, kinematics, and the on-screen layout.

scripts/live.py parses the command line and runs App. App takes a model factory so the same code can be
driven headless in tests (real models on still images, or fake models).
"""
from __future__ import annotations

import csv
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .draw import (CYAN, HAND_EDGES, POSE_ARM_EDGES, YELLOW, compose_layout, draw_head_on_video, draw_wireframe,
                   render_arms_panel, render_face_panel, render_hand_panel, text_block)
from .elbow import ElbowPredictor, SegmentLengths
from .face import EMOTIONS, EmotionSmoother, EmotionTask
from .filters import HAND_SMOOTHING, AngleSmoother, JitterMeter, LandmarkSmoother, OneEuroFilter, make_arm_smoothers
from .fingers import FINGER_KEYS, THUMB_KEYS, HandKinematics, flat, hand_fk, hand_geometry, palm_to_world
from .gestures import Gesture, GestureFilter, classify_hand
from .handpose import HandPoseZero, camera_matrix, hand_frame, hand_pose_camera
from .head import HeadPoseEstimator, HeadPoseZero, angles_from_axes, head_axes_from_angles
from .kinematics import (_SIDE, ArmAngles, ArmKinematics, frame_agreement_deg, out_of_range, thorax_frame,
                         up_from_neutral)
from .pipeline import BackgroundGate, CameraReader, ModelWorker, stop_all
from .tracker import FrameResult, assign_hands

ANGLE_FIELDS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion",
                "forearm_rotation", "wrist_flexion", "wrist_deviation"]
LABELS = {"shoulder_flexion": "Shoulder flex/ext", "shoulder_abduction": "Shoulder abd/add",
          "shoulder_rotation": "Shoulder rotation", "elbow_flexion": "Elbow flexion",
          "forearm_rotation": "Forearm pro/sup", "wrist_flexion": "Wrist flex/ext",
          "wrist_deviation": "Wrist rad/uln"}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_COLS = [f"{f}_{k}" for f in ["index", "middle", "ring", "pinky"] for k in FINGER_KEYS] + \
              [f"thumb_{k}" for k in THUMB_KEYS]
VIS = 0.6


@dataclass
class Config:
    mode: str = "body"                 # body | pov
    camera: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    side: str = "right"                # left | right | both
    torso: str = "auto"                # auto | hips | desk
    pose_model: str = "full"           # lite | full | heavy
    smoothing: str = "balanced"
    pose_only_wrist: bool = False
    hand: str = "right"                # pov mode
    swap_hands: bool = False
    fov: float = 70.0
    face: bool = True                  # head pose (face mesh)
    emotion: bool = True               # emotion + recognition
    face_hz: float = 30.0              # head pose: at most this often
    face_min_hz: float = 10.0          #   and at least this often when the CPU is busy with hands and body
    emotion_hz: float = 10.0
    emotion_min_hz: float = 3.0
    cores: int | None = None           # CPU cores to plan for (default: all)
    name: str = "me"                   # name stored when enrolling a face
    record: Path | None = None
    model_dir: str = "models"
    faces_dir: str = "faces"
    window_width: int = 1600


def num(v: float, nd: int = 1) -> str:
    """Fixed-point text without a minus sign on zero (-0.0 shows as 0.0)."""
    return f"{round(float(v), nd) + 0.0:.{nd}f}"


def finger_table(angles: dict, title: str) -> list[str]:
    lines = [f"{title}\tMCP/CMC\tside\tPIP/MCP\tDIP/IP"]
    th = angles["thumb"]
    lines.append(f"thumb (rot {num(th['cmc_rotation'], 0)})\t{num(th['cmc_palmar'], 0)}\t{num(th['cmc_radial'], 0)}"
                 f"\t{num(th['mcp'], 0)}\t{num(th['ip'], 0)}")
    for f in FINGER_ORDER[1:]:
        a = angles[f]
        lines.append(f"{f}\t{num(a['mcp_flex'], 0)}\t{num(a['mcp_abd'], 0)}\t{num(a['pip'], 0)}\t{num(a['dip'], 0)}")
    return lines


def pick_hand_by_label(hands, side: str, swap: bool = False, fallback: bool = True):
    """MediaPipe labels handedness as if the image were mirrored; raw camera frames are not, so swap.

    With `fallback`, a single detected hand is used even when its label does not match.
    """
    if not hands:
        return None
    want = {"right": "Left", "left": "Right"}[side]
    if swap:
        want = "Right" if want == "Left" else "Left"
    matches = [h for h in hands if h.handedness == want]
    if matches:
        return matches[0]
    return hands[0] if (fallback and len(hands) == 1) else None


def default_models(cfg: Config, size):
    """Factories for the real models; tests pass their own."""
    from .tracker import FaceModel, HandModel, PoseModel
    w, h = size
    K = camera_matrix(w, h, cfg.fov)
    makers = {"hand": lambda: _call(HandModel(num_hands=2, model_dir=cfg.model_dir))}
    if cfg.mode == "body":
        makers["pose"] = lambda: _call(PoseModel(f"pose_{cfg.pose_model}", cfg.model_dir))
        if cfg.face:
            def face():
                fm, est = FaceModel(cfg.model_dir), HeadPoseEstimator()

                def run(frame):
                    L = fm.process(frame.rgb, frame.t)
                    return None if L is None else (L, est.solve(L, w, h, K))
                run.close = fm.close
                return run
            makers["face"] = face
        if cfg.emotion:
            makers["emotion"] = lambda: EmotionTask(cfg.model_dir, cfg.faces_dir)
    return makers


def _call(model):
    def run(frame):
        return model.process(frame.rgb, frame.t)
    run.close = model.close
    return run


@dataclass
class _Cache:
    index: int = -1


class App:
    def __init__(self, cfg: Config, cap, models: dict | None = None):
        import cv2
        self.cv2 = cv2
        self.cfg = cfg
        self.pov = cfg.mode == "pov"
        self.camera = CameraReader(cap)
        self.camera.start()
        first = self.camera.wait_newer(0, timeout=5.0)
        if first is None:
            self.camera.stop()
            raise RuntimeError("the camera did not deliver any frames")
        self.size = (first.bgr.shape[1], first.bgr.shape[0])
        self.K = camera_matrix(self.size[0], self.size[1], cfg.fov)
        makers = models if models is not None else default_models(cfg, self.size)
        self.fns = {}
        try:
            for name, make in makers.items():
                self.fns[name] = make()
        except BaseException:
            self.camera.stop()
            for fn in self.fns.values():
                getattr(fn, "close", lambda: None)()
            raise
        # Hands and body are the priority and run as fast as they can; face and emotion use the cores left
        # free, with a guaranteed minimum rate (see pipeline.py), and are capped so they never need more.
        priority = [n for n in self.fns if n in ("hand", "pose")]
        self.gate = BackgroundGate(len(priority), cfg.cores)
        caps = {"face": cfg.face_hz, "emotion": cfg.emotion_hz}
        floors = {"face": cfg.face_min_hz, "emotion": cfg.emotion_min_hz}
        self.workers = {name: ModelWorker(name, self.camera, fn, max_hz=caps.get(name) or None,
                                          gate=None if name in priority else self.gate,
                                          min_hz=floors.get(name, 5.0))
                        for name, fn in self.fns.items()}
        self._seen = {name: _Cache() for name in self.workers}
        self._arm_t = self._hand_t = -1.0
        self.data_t = None            # capture time of the newest result used (for the CSV)

        # body state
        self.sides = ["left", "right"] if cfg.side == "both" else [cfg.side]
        self.pose_smooth, _ = make_arm_smoothers(cfg.smoothing)
        self.kins = {sd: ArmKinematics(sd) for sd in self.sides}
        self.elbows = {sd: ElbowPredictor() for sd in self.sides}
        self.seglens = {sd: SegmentLengths() for sd in self.sides}
        self.ang_smooth = {sd: make_arm_smoothers(cfg.smoothing)[1] for sd in self.sides}
        self.jitter = {sd: JitterMeter(1.0) for sd in self.sides}
        hand_sides = [cfg.hand] if self.pov else self.sides
        self.hand_smooths = {sd: LandmarkSmoother(**HAND_SMOOTHING) for sd in hand_sides}
        self.hand_kins = {sd: HandKinematics(sd) for sd in hand_sides}
        self.gfilters = {sd: GestureFilter() for sd in hand_sides}
        self.hands_w, self.hands_raw = {}, {}
        self.fingers, self.finger_geo = {}, {}
        self.gestures = {sd: Gesture.UNKNOWN.value for sd in hand_sides}
        self.angs = {sd: ArmAngles() for sd in self.sides}
        self.missing = {sd: [] for sd in self.sides}
        self.pose_w, self.pose_res, self.up_hint, self.up_cal, self.torso_src = None, None, None, None, "-"
        self.thorax = None
        # head and face state
        self.head, self.head_raw, self.face_L = None, None, None
        self.head_ang = AngleSmoother(min_cutoff=1.0, beta=0.05)
        self.head_pos = OneEuroFilter(min_cutoff=1.0, beta=0.3)     # filters centimeters
        self.head_zero = HeadPoseZero()
        self.emotions, self.emotion_res = None, None
        self.emo_smooth = EmotionSmoother()
        # pov state
        self.pov_zero, self.pov_info = HandPoseZero(), None
        # ui state
        self.show_wire = True
        self.table_cycle = [None] + hand_sides
        self.table_i = 1 if self.pov else 0
        self.want_check = self.want_calib = False
        self.message, self.message_t = "", 0.0
        self.writer, self._csv_file = None, None
        try:
            if cfg.record:
                self._open_csv(cfg.record)
        except BaseException:
            self.close()
            raise
        for w in self.workers.values():          # start last, once everything they feed is ready
            w.start()

    # ------------------------------------------------------------------ results
    def _new(self, name):
        w = self.workers.get(name)
        if w is None:
            return None
        r = w.get()
        if r is None or r.frame_index == self._seen[name].index:
            return None
        self._seen[name].index = r.frame_index
        return r

    def update(self) -> bool:
        """Consume new worker results; True when any arrived."""
        hand_r, pose_r, face_r, emo_r = (self._new(n) for n in ("hand", "pose", "face", "emotion"))
        if pose_r is not None:
            self.pose_res = pose_r
            if pose_r.value is not None:
                self.pose_w = self.pose_smooth(pose_r.value.world, pose_r.t_capture)
        if hand_r is not None:
            self._hand_t = hand_r.t_capture
            self._update_hands(hand_r)
        if not self.pov and (pose_r is not None or hand_r is not None) and self.pose_res is not None:
            # The arm combines the newest body and the newest hand result; its time is the newer of the two,
            # and it is solved only when that time advances (the filters need increasing timestamps).
            t = max(self.pose_res.t_capture, self._hand_t)
            if t > self._arm_t:
                self._arm_t = t
                self._update_arms(t)
        if face_r is not None:
            self._update_head(face_r)
        if emo_r is not None:
            self.emotion_res = emo_r.value
            if emo_r.value is None:
                self.emotions = None
                self.emo_smooth.reset()
            else:
                self.emotions = dict(zip(EMOTIONS, self.emo_smooth(emo_r.value["probs"], emo_r.t_capture)))
        task = self.fns.get("emotion")
        if getattr(task, "message", ""):
            self.say(task.message)
            task.message = ""
        new = [r for r in (hand_r, pose_r, face_r, emo_r) if r is not None]
        if new:
            self.data_t = max([r.t_capture for r in new] + ([self.data_t] if self.data_t is not None else []))
        return bool(new)

    def _update_hands(self, r):
        hands = r.value or []
        if self.pov:
            chosen = {self.cfg.hand: pick_hand_by_label(hands, self.cfg.hand, self.cfg.swap_hands)}
        else:
            pose_img = self.pose_res.value.image if (self.pose_res and self.pose_res.value is not None) else None
            if pose_img is not None:
                chosen = assign_hands(FrameResult(hands=hands, pose_image=pose_img), self.sides)
            else:
                both = len(self.sides) == 2          # then a lone hand must carry its own label
                chosen = {sd: pick_hand_by_label(hands, sd, fallback=not both) for sd in self.sides}
        for sd, hand in chosen.items():
            if hand is None:
                self.hands_w.pop(sd, None)
                self.hands_raw.pop(sd, None)
                self.fingers.pop(sd, None)
                self.gestures[sd] = self.gfilters[sd].update(Gesture.UNKNOWN, r.t_capture).value
                continue
            w = self.hand_smooths[sd](hand.world, r.t_capture)
            self.hands_w[sd], self.hands_raw[sd] = w, hand
            self.gestures[sd] = self.gfilters[sd].update(classify_hand(w), r.t_capture).value
            self.fingers[sd] = self.hand_kins[sd].solve(w)
            self.finger_geo[sd] = hand_geometry(w, sd)
        if self.pov:
            self._update_pov()

    def _update_arms(self, t):
        cfg = self.cfg
        if self.pose_res is None or self.pose_res.value is None or self.pose_w is None:
            self.angs = {sd: ArmAngles() for sd in self.sides}
            self.missing = {sd: [] for sd in self.sides}
            self.thorax = None
            return
        pose_w, vis = self.pose_w, self.pose_res.value.visibility
        hips_seen = min(vis[23], vis[24]) > 0.5
        if cfg.torso == "hips" or (cfg.torso == "auto" and hips_seen):
            self.up_hint, self.torso_src = None, "hips"
        elif self.up_cal is not None:
            self.up_hint, self.torso_src = self.up_cal, "calibrated"
        else:
            self.up_hint, self.torso_src = np.array([0.0, -1.0, 0.0]), "camera vertical"
        self.thorax = thorax_frame(pose_w, self.up_hint)[0]
        torso_up = self.thorax[:, 1]
        shoulder_width = float(np.linalg.norm(pose_w[11] - pose_w[12]))
        for sd in self.sides:
            s_i, e_i, w_i = _SIDE[sd]
            self.missing[sd] = [n for n, i in (("shoulder", s_i), ("wrist", w_i)) if vis[i] < VIS]
            if self.missing[sd]:
                self.ang_smooth[sd].reset()
                self.angs[sd] = ArmAngles()
                continue
            if min(vis[s_i], vis[e_i], vis[w_i]) >= 0.8:
                self.seglens[sd].add(pose_w[s_i], pose_w[e_i], pose_w[w_i])
            est = self.elbows[sd].estimate(pose_w[s_i], pose_w[w_i], self.seglens[sd].get(shoulder_width),
                                           guess=pose_w[e_i], guess_vis=float(vis[e_i]), up=torso_up, t=t)
            pose_sd = pose_w.copy()
            pose_sd[e_i] = est.elbow
            raw = self.kins[sd].solve(pose_sd, None if cfg.pose_only_wrist else self.hands_w.get(sd), self.up_hint)
            sm = self.ang_smooth[sd]({k: getattr(raw, k) for k in ANGLE_FIELDS}, t)
            self.angs[sd] = dataclasses.replace(raw, **sm, elbow_predicted=est.mode != "measured")
            self.jitter[sd].add(sm, t)
        if self.want_calib:
            self.want_calib = False
            usable = [sd for sd in self.sides if not self.missing[sd] and not self.angs[sd].elbow_predicted]
            if usable:
                self.up_cal = up_from_neutral(pose_w, "both" if len(usable) == 2 else usable[0])
                for k in self.kins.values():
                    k.reset()
                for s in self.ang_smooth.values():
                    s.reset()
                self.say(f"torso calibrated from the neutral pose ({'+'.join(usable)} arm)")
            else:
                self.say("calibration skipped: shoulders, elbows and wrists must be in view")
        if self.want_check:
            self.want_check = False
            fwd = self.thorax[:, 0]
            print(f"torso up from: {self.torso_src}; forward in camera axes: {np.round(fwd, 2)} "
                  "(facing the camera, z should be clearly negative)")
            for sd in self.sides:
                print(f"{sd} segment lengths (m): {np.round(self.seglens[sd].get(shoulder_width), 3)}",
                      "(measured)" if self.seglens[sd].calibrated else "(estimated from shoulder width)")
                if sd in self.hands_w:
                    print(f"{sd} agreement (deg):", {k: round(v, 1) for k, v in
                                                     frame_agreement_deg(pose_w, self.hands_w[sd], sd).items()},
                          "(hand_vs_forearm under ~20 with a straight wrist is good)")

    def _update_head(self, r):
        if r.value is None or r.value[1] is None:
            self.head = self.head_raw = self.face_L = None
            self.head_ang.reset()
            self.head_pos.reset()
            return
        L, hp = r.value
        self.face_L, self.head_raw = L, hp
        a = self.head_ang({"yaw": hp.yaw, "pitch": hp.pitch, "roll": hp.roll}, r.t_capture)
        pos = self.head_pos(hp.position_cm, r.t_capture) / 100.0
        axes = head_axes_from_angles(a["yaw"], a["pitch"], a["roll"])
        self.head = dataclasses.replace(hp, axes=axes, position_m=np.asarray(pos, float), yaw=a["yaw"],
                                        pitch=a["pitch"], roll=a["roll"])

    def _update_pov(self):
        sd = self.cfg.hand
        hand, w = self.hands_raw.get(sd), self.hands_w.get(sd)
        self.pov_info = None
        if hand is None:
            return
        R, tvec = hand_pose_camera(hand.world, hand.image, self.K)
        if R is None:
            return
        R_hand = R @ hand_frame(w)
        if self.want_calib:
            self.want_calib = False
            self.pov_zero.zero(R_hand, tvec)
            self.say("hand pose zeroed")
        solved = palm_to_world(hand_fk(self.fingers[sd], self.finger_geo[sd]), w, sd)
        self.pov_info = dict(t=tvec, rel=self.pov_zero.relative(R_hand, tvec), solved=solved)

    # ------------------------------------------------------------------ ui
    def say(self, text):
        self.message, self.message_t = text, time.monotonic()
        print(text)

    def neck_axes(self):
        """Head axes in the torso frame (for the 3D panel), when both are known."""
        if self.head is None or self.thorax is None:
            return None
        return self.thorax.T @ self.head.axes

    def status_line(self, now):
        parts = [f"cam {self.camera.meter.rate(now):.0f} fps"]
        for name, w in self.workers.items():
            if w.error is not None:
                parts.append(f"{name}: ERROR {type(w.error).__name__}")
                continue
            lat = w.meter.latency_ms()
            parts.append(f"{name} {w.meter.rate(now):.0f} Hz" + (f" {lat:.0f} ms" if np.isfinite(lat) else ""))
        return " | ".join(parts)

    def text_lines(self, now):
        lines = []
        if self.pov:
            sd = self.cfg.hand
            if self.pov_info is not None:
                tv, rel = self.pov_info["t"] * 100, self.pov_info["rel"]
                lines.append(f"hand x {num(tv[0])}  y {num(tv[1])}  z {num(tv[2])} cm (right, down, away)")
                if rel is not None:
                    lines.append(f"moved dx {num(rel.dx_cm)}  dy {num(rel.dy_cm)}  dz {num(rel.dz_cm)} cm")
                    lines.append(f"turned twist {num(rel.twist, 0)}  pitch {num(rel.pitch, 0)}  yaw {num(rel.yaw, 0)} deg")
                else:
                    lines.append("press z to set the zero pose for relative motion")
            else:
                lines.append(f"no {sd} hand in view")
            lines.append(f"gesture: {self.gestures[sd]}")
        else:
            for sd in self.sides:
                if self.missing[sd]:
                    lines.append(f"{sd}: {' and '.join(self.missing[sd])} not in view")
            if self.pose_res is None or self.pose_res.value is None:
                lines.append("no body: shoulders and the tracked arm must be in view")
            elif len(self.sides) == 1:
                sd = self.sides[0]
                a, bad, jit = self.angs[sd], set(out_of_range(self.angs[sd])), self.jitter[sd].std()
                for k in ANGLE_FIELDS if not self.missing[sd] else []:
                    mark = ""
                    if k == "shoulder_rotation" and not a.shoulder_rotation_valid:
                        mark = " held"
                    if k in ("forearm_rotation", "wrist_flexion", "wrist_deviation") and not a.wrist_valid:
                        mark = " no hand"
                    if k in bad:
                        mark = " check"
                    j = jit.get(k, float("nan"))
                    v = getattr(a, k)
                    lines.append(f"{LABELS[k]}\t{num(v) if np.isfinite(v) else '--'}\t"
                                 f"{'+-' + num(j) if np.isfinite(j) else ''}{mark}")
                if not self.missing[sd] and a.elbow_predicted:
                    lines.append("elbow predicted (out of view)")
            else:
                lines.append("\tLEFT\tRIGHT")
                for k in ANGLE_FIELDS:
                    cells = []
                    for sd in self.sides:
                        v = getattr(self.angs[sd], k)
                        flag = "!" if k in out_of_range(self.angs[sd]) else (
                            "h" if k == "shoulder_rotation" and np.isfinite(v) and not self.angs[sd].shoulder_rotation_valid else "")
                        cells.append(f"{num(v)}{flag}" if np.isfinite(v) else "--")
                    lines.append(f"{LABELS[k]}\t{cells[0]}\t{cells[1]}")
                pred = [sd for sd in self.sides if self.angs[sd].elbow_predicted]
                if pred:
                    lines.append("elbow predicted: " + ", ".join(pred))
            lines.append(f"torso: {self.torso_src}" + ("  (z: calibrate)" if self.torso_src == "camera vertical" else ""))
            lines.append("gesture: " + "  ".join(f"{sd} {self.gestures[sd]}" for sd in self.sides))
        tsd = self.table_cycle[self.table_i]
        if tsd is not None:
            lines += finger_table(self.fingers[tsd], f"{tsd} fingers") if tsd in self.fingers else [f"{tsd} hand not detected"]
        if self.message and now - self.message_t < 4.0:
            lines.append(self.message)
        lines.append(self.status_line(now))
        keys = "q quit  w wireframe  f fingers  " + ("z zero hand" if self.pov else "z calibrate  h zero head  e enroll face  c checks")
        lines.append(keys)
        return lines

    def face_panel(self, frame, size):
        if self.pov:
            return render_face_panel(size, note="face tracking is off in POV mode")
        crop = None
        box = None
        if self.face_L is not None:
            xy = self.face_L[:, :2] * np.array(self.size)
            (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
            box = (x0, y0, x1 - x0, y1 - y0)
        elif self.emotion_res is not None:
            box = tuple(self.emotion_res["box"])
        if box is not None:
            x, y, w, h = box
            c, s = (x + w / 2, y + h / 2), max(w, h) * 1.3
            X0, Y0 = int(max(c[0] - s / 2, 0)), int(max(c[1] - s / 2, 0))
            X1, Y1 = int(min(c[0] + s / 2, self.size[0])), int(min(c[1] + s / 2, self.size[1]))
            if X1 > X0 + 4 and Y1 > Y0 + 4:
                crop = self.cv2.flip(frame.bgr[Y0:Y1, X0:X1], 1)
        ident = ""
        er = self.emotion_res
        if "emotion" in self.workers:
            if er is None:
                ident = ""
            elif er["identity"]:
                ident = f"{er['identity']}  (match {er['match']:.2f})"
            elif er["enrolled"]:
                ident = f"unknown face (best {er['match']:.2f})"
            else:
                ident = f"press e to enroll as '{self.cfg.name}'"
        rel = self.head_zero.relative(self.head) if (self.head is not None and self.head_zero.is_set) else None
        note = ""
        if "emotion" not in self.workers:
            note = "emotion off (--no-emotion)"
        return render_face_panel(size, crop, self.emotions, ident, self.head, rel, note)

    def render(self, frame):
        cv2 = self.cv2
        now = time.monotonic()
        video = frame.bgr.copy()
        if self.show_wire:
            pr = self.pose_res
            if pr is not None and pr.value is not None:
                draw_wireframe(video, pr.value.image, POSE_ARM_EDGES, CYAN, pr.value.visibility)
            hw = self.workers.get("hand")
            hr = hw.get() if hw is not None else None
            for h in (hr.value or []) if hr is not None else []:
                draw_wireframe(video, h.image, HAND_EDGES, YELLOW, radius=3)
            if self.head is not None:
                draw_head_on_video(video, self.head, self.K)
        if not self.pov:
            video = cv2.flip(video, 1)            # mirror view for a camera facing you
        text_block(video, self.text_lines(now))
        size = video.shape[0] // 2
        if self.pov:
            info = self.pov_info
            kin = render_hand_panel(self.hands_w.get(self.cfg.hand), size=size,
                                    solved=None if info is None else info["solved"],
                                    note="grey: measured  yellow: from solved joint angles")
        else:
            held = [sd for sd in self.sides if np.isfinite(self.angs[sd].shoulder_flexion)
                    and not self.angs[sd].shoulder_rotation_valid]
            kin = render_arms_panel(self.angs, size=size, note=("shoulder rotation held: " + ", ".join(held)) if held else "",
                                    fingers_by_side={sd: (self.fingers[sd], self.finger_geo[sd]) for sd in self.fingers
                                                     if sd in self.sides},
                                    head_axes=self.neck_axes())
        return compose_layout(video, self.face_panel(frame, size), kin, self.cfg.window_width)

    def handle_key(self, key):
        k = chr(key) if 0 <= key < 256 else ""
        if k == "q":
            return False
        if k == "w":
            self.show_wire = not self.show_wire
        elif k == "f":
            self.table_i = (self.table_i + 1) % len(self.table_cycle)
        elif k == "c":
            self.want_check = True
        elif k == "z":
            self.want_calib = True
        elif k == "h":
            if self.head is not None:
                self.head_zero.zero(self.head)
                self.say("head pose zeroed")
            else:
                self.say("no face: cannot zero the head pose")
        elif k == "e":
            task = self.fns.get("emotion")
            if task is None:
                self.say("face recognition is off")
            else:
                task.enroll_request = self.cfg.name
                self.say(f"enrolling your face as '{self.cfg.name}' (look at the camera)")
        return True

    # ------------------------------------------------------------------ csv
    def _open_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self._csv_file)
        if self.pov:
            cols = ["t", "gesture", "x_m", "y_m", "z_m", "dx_cm", "dy_cm", "dz_cm", "twist", "pitch", "yaw"] + FINGER_COLS
        else:
            cols = ["t"]
            for sd in self.sides:
                p = sd[0].upper() + "_" if len(self.sides) > 1 else ""
                cols += [p + "gesture", *[p + k for k in ANGLE_FIELDS], p + "shoulder_rotation_valid",
                         p + "wrist_valid", p + "elbow_predicted", *[p + c for c in FINGER_COLS]]
            cols += ["torso_source", "head_x_cm", "head_y_cm", "head_z_cm", "head_yaw", "head_pitch", "head_roll",
                     "neck_yaw", "neck_pitch", "neck_roll", *[f"emotion_{e}_pct" for e in EMOTIONS], "identity", "match"]
        self.writer.writerow(cols)

    def write_row(self, t):
        if self.writer is None:
            return
        nan = float("nan")
        if self.pov:
            sd = self.cfg.hand
            info = self.pov_info
            if info is None:
                return
            rel = info["rel"]
            fv = flat(self.fingers[sd]) if sd in self.fingers else {}
            self.writer.writerow([f"{t:.3f}", self.gestures[sd], *np.round(info["t"], 4),
                                  *([round(v, 2) for v in (rel.dx_cm, rel.dy_cm, rel.dz_cm, rel.twist, rel.pitch, rel.yaw)]
                                    if rel else [nan] * 6), *[round(fv.get(c, nan), 1) for c in FINGER_COLS]])
            return
        row = [f"{t:.3f}"]
        for sd in self.sides:
            a = self.angs[sd]
            fv = flat(self.fingers[sd]) if sd in self.fingers else {}
            row += [self.gestures[sd], *[round(getattr(a, k), 2) for k in ANGLE_FIELDS], int(a.shoulder_rotation_valid),
                    int(a.wrist_valid), int(a.elbow_predicted), *[round(fv.get(c, nan), 1) for c in FINGER_COLS]]
        h = self.head
        neck = angles_from_axes(self.neck_axes(), np.eye(3)) if self.neck_axes() is not None else (nan, nan, nan)
        emo = [round(100 * self.emotions[e], 1) for e in EMOTIONS] if self.emotions else [nan] * len(EMOTIONS)
        er = self.emotion_res
        row += [self.torso_src, *(np.round(h.position_cm, 2) if h else [nan] * 3),
                *([round(h.yaw, 2), round(h.pitch, 2), round(h.roll, 2)] if h else [nan] * 3),
                *[round(v, 2) for v in neck], *emo, (er or {}).get("identity") or "",
                round(er["match"], 3) if er else nan]
        self.writer.writerow(row)

    # ------------------------------------------------------------------ loop
    def step(self, frame):
        if self.update() and self.data_t is not None:
            self.write_row(self.data_t)
        return self.render(frame)

    def run(self, show=None, waitkey=None):
        cv2 = self.cv2
        show = show or (lambda img: cv2.imshow("Homunculus Shadowing", img))
        waitkey = waitkey or (lambda: cv2.waitKey(1) & 0xFF)
        last = 0
        try:
            while True:
                frame = self.camera.wait_newer(last, timeout=2.0)
                if frame is None:
                    if self.camera.failed:
                        print("The camera stopped delivering frames.")
                        break
                    continue
                last = frame.index
                show(self.step(frame))
                if not self.handle_key(waitkey()):
                    break
        finally:
            self.close()

    def close(self):
        stop_all(list(self.workers.values()) + [self.camera])
        for fn in self.fns.values():
            close = getattr(fn, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

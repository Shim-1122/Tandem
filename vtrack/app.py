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

from .draw import (CYAN, HAND_EDGES, POSE_ARM_EDGES, POSE_LEG_EDGES, YELLOW, compose_layout, draw_head_on_video,
                   draw_wireframe, render_arms_panel, render_face_panel, render_hand_panel, text_block)
from .elbow import ElbowPredictor, SegmentLengths
from .emotion_cal import CalibrationSession, calibration_path, load_all
from .face import EMOTIONS, EmotionSmoother, EmotionTask, softmax
from .filters import HAND_SMOOTHING, AngleSmoother, JitterMeter, LandmarkSmoother, OneEuroFilter, make_arm_smoothers
from .gaze import GazeCalibrationSession, TargetLock, calibration_targets, eye_state
from .fingers import FINGER_KEYS, THUMB_KEYS, HandKinematics, flat, hand_fk, hand_geometry, palm_to_world
from .gestures import Gesture, GestureFilter, classify_hand
from .grasp import GraspState, GraspTracker, ObjectTracker, hand_points, hand_roi, hand_shape
from .grip import GripModel
from .handpose import HandPoseZero, camera_matrix, hand_frame, hand_pose_camera
from .head import HeadPoseEstimator, HeadPoseZero, angles_from_axes, head_axes_from_angles
from .legs import (CAMERA_UP, CORE_FIELDS, LEG_FIELDS, CoreAngles, LegAngles, LegKinematics, core_angles,
                   core_visible, leg_out_of_range, legs_visible, up_from_standing)
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
LEG_LABELS = {"hip_flexion": "Hip flex/ext", "hip_abduction": "Hip abd/add", "hip_rotation": "Hip rotation",
              "pelvic_hike": "Pelvic hike", "knee_flexion": "Knee flexion", "ankle_dorsiflexion": "Ankle dorsi/plantar",
              "ankle_inversion": "Ankle inv/ev", "ankle_abduction": "Ankle abd/add", "foot_pitch": "Foot pitch",
              "foot_roll": "Foot roll"}
CORE_LABELS = {"core_flexion": "Core flex/ext", "core_lateral_bend": "Core side bend", "core_rotation": "Core rotation"}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
GRASP_COLS = ["grasp_state", "grasp_object", "hand_shape", "grip_type", "grip_n", "grip_cmd", "grip_open_cm"]
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
    side: str = "both"                 # left | right | both (arms; legs, core and head are always tracked)
    torso: str = "auto"                # auto | hips | desk
    pose_model: str = "full"           # lite | full | heavy
    smoothing: str = "balanced"
    pose_only_wrist: bool = False
    hand: str = "right"                # pov mode
    swap_hands: bool = False
    fov: float = 70.0
    face: bool = True                  # head pose (face mesh)
    emotion: bool = True               # emotion + recognition
    emotion_model: str = "hsemotion"   # hsemotion | opencv (see face.py)
    face_hz: float = 30.0              # head pose: at most this often
    face_min_hz: float = 10.0          #   and at least this often when the CPU is busy with hands and body
    emotion_hz: float = 10.0
    emotion_min_hz: float = 3.0
    cal_prepare_s: float = 1.5         # emotion calibration: seconds to get each expression ready
    cal_capture_s: float = 2.5         #   and seconds recorded
    gaze_dwell: float = 0.0            # > 0: also lock a target by holding your gaze still this many seconds
    gaze_settle_s: float = 1.0         # gaze calibration: seconds to look at each dot before recording
    gaze_capture_s: float = 1.2        #   and seconds recorded per dot
    objects: bool = True               # object detection and grasp tracking
    object_model: str = "medium"       # medium | small (about 3 times faster, less accurate)
    object_hz: float = 10.0            # object model: at most this often
    object_min_hz: float = 2.0         #   and at least this often when the CPU is busy
    object_every_s: float = 0.6        # search the whole frame this often; otherwise only around the hands
    max_grip_n: float = 50.0           # the robot hand's strongest grip (scales the 0..1 grip command)
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
            makers["emotion"] = lambda: EmotionTask(cfg.model_dir, cfg.faces_dir, model=cfg.emotion_model)
    if cfg.objects:
        def objects():
            from .objects import ObjectTask
            import os
            threads = max(1, min(4, (cfg.cores or os.cpu_count() or 2) - 2))   # leave cores for hands and body
            return ObjectTask(cfg.object_model, cfg.model_dir, cfg.object_every_s, threads=threads)
        makers["objects"] = objects
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
        caps = {"face": cfg.face_hz, "emotion": cfg.emotion_hz, "objects": cfg.object_hz}
        floors = {"face": cfg.face_min_hz, "emotion": cfg.emotion_min_hz, "objects": cfg.object_min_hz}
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
        # legs and core (both legs always; they are solved whenever they are in view)
        self.leg_kins = {sd: LegKinematics(sd) for sd in ("left", "right")}
        self.leg_smooth = {sd: make_arm_smoothers(cfg.smoothing)[1] for sd in ("left", "right")}
        self.core_smooth = make_arm_smoothers(cfg.smoothing)[1]
        self.legs = {sd: LegAngles() for sd in ("left", "right")}
        self.leg_missing = {sd: ["hip", "knee", "ankle"] for sd in ("left", "right")}
        self.core = CoreAngles()
        self.leg_up_cal, self.leg_up_src = None, "camera vertical"
        self.pose_w, self.pose_res, self.up_hint, self.up_cal, self.torso_src = None, None, None, None, "-"
        self.thorax = None
        # head and face state
        self.head, self.head_raw, self.face_L = None, None, None
        self.head_ang = AngleSmoother(min_cutoff=1.0, beta=0.05)
        self.head_pos = OneEuroFilter(min_cutoff=1.0, beta=0.3)     # filters centimeters
        self.head_zero = HeadPoseZero()
        # eyes: gaze, calibration onto the video, target lock
        self.eye, self.gaze_map, self.gaze_session, self.gaze_pt = None, None, None, None
        self.gaze_filter = OneEuroFilter(min_cutoff=0.6, beta=0.01)       # pixels
        self.target = TargetLock(self.size[0], self.size[1], self.K, mirrored=not self.pov, dwell_s=cfg.gaze_dwell)
        self.prev_gestures = {}
        self.emotions, self.emotion_res = None, None
        self.emo_smooth = EmotionSmoother()
        self.calibrations = load_all(cfg.faces_dir, cfg.emotion_model) if cfg.emotion else {}
        self.cal_session, self.cal_person = None, None
        # objects and grasps (vocabulary from the object model, or the repository's when the model is faked)
        vocab = getattr(getattr(self.fns.get("objects"), "det", None), "vocab", None)
        if vocab is None:
            from .objects import Vocabulary
            vocab = Vocabulary.load()
        self.obj_tracker = ObjectTracker(vocab.names, vocab.report)
        self.grasps = GraspTracker(GripModel(arm_max_n=cfg.max_grip_n))
        self.grasp = {sd: GraspState(sd) for sd in hand_sides}
        self.show_objects = True
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
        hand_r, pose_r, face_r, emo_r, obj_r = (self._new(n) for n in ("hand", "pose", "face", "emotion", "objects"))
        if obj_r is not None:
            for region, dets in obj_r.value["runs"]:
                self.obj_tracker.update(dets, obj_r.t_capture, region=region)
        if pose_r is not None:
            self.pose_res = pose_r
            if pose_r.value is not None:
                self.pose_w = self.pose_smooth(pose_r.value.world, pose_r.t_capture)
        if hand_r is not None:
            self._hand_t = hand_r.t_capture
            self._update_hands(hand_r)
        if not self.pov and pose_r is not None:
            self._update_legs(pose_r.t_capture)
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
            self._update_emotion(emo_r)
        if self.want_calib and not self.pov and pose_r is not None:
            self.want_calib = False
            self._calibrate_body()
        if self.cal_session is not None and self.cal_session.finished(time.monotonic() - 0.3):
            self._finish_emotion_calibration()
        if self.gaze_session is not None and self.gaze_session.finished(time.monotonic() - 0.2):
            self._finish_gaze_calibration()
        task = self.fns.get("emotion")
        if getattr(task, "message", ""):
            self.say(task.message)
            task.message = ""
        new = [r for r in (hand_r, pose_r, face_r, emo_r, obj_r) if r is not None]
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
        rois = []
        for sd, hand in chosen.items():
            if hand is None:
                self.hands_w.pop(sd, None)
                self.hands_raw.pop(sd, None)
                self.fingers.pop(sd, None)
                self.gestures[sd] = self.gfilters[sd].update(Gesture.UNKNOWN, r.t_capture).value
                self.grasp[sd] = self.grasps.update(sd, None, "", self.obj_tracker, r.t_capture)
                continue
            w = self.hand_smooths[sd](hand.world, r.t_capture)
            self.hands_w[sd], self.hands_raw[sd] = w, hand
            self.gestures[sd] = self.gfilters[sd].update(classify_hand(w), r.t_capture).value
            self.fingers[sd] = self.hand_kins[sd].solve(w)
            self.finger_geo[sd] = hand_geometry(w, sd)
            px = hand_points(hand.image, *self.size)
            self.grasp[sd] = self.grasps.update(sd, px, hand_shape(w, self.fingers[sd]), self.obj_tracker, r.t_capture)
            rois.append((sd, hand_roi(px, *self.size)))
        task = self.fns.get("objects")
        if task is not None:
            task.rois = rois                        # the object model zooms in on the hands next
        if not self.pov:                            # a new pinch (either hand) locks or releases the target
            for sd, g in self.gestures.items():
                if g == Gesture.PINCH.value and self.prev_gestures.get(sd) != Gesture.PINCH.value:
                    self._toggle_target(f"{sd} pinch")
            self.prev_gestures = dict(self.gestures)
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
                jit = {k: round(v, 2) for k, v in self.jitter[sd].std().items() if np.isfinite(v)}
                if jit:
                    print(f"{sd} arm jitter, standard deviation over the last second (deg; hold still):", jit)

    def _leg_up(self):
        # Not the arm (desk) calibration: that one assumes the upper arms hang straight down when z is pressed,
        # which a standing person need not be doing, and the legs need the true vertical.
        if self.leg_up_cal is not None:
            return self.leg_up_cal, "standing calibration"
        return CAMERA_UP, "camera vertical"

    def _update_legs(self, t):
        pr = self.pose_res
        if pr is None or pr.value is None or self.pose_w is None:
            self.legs = {sd: LegAngles() for sd in self.legs}
            self.leg_missing = {sd: ["hip", "knee", "ankle"] for sd in self.legs}
            self.core = CoreAngles()
            self.core_smooth.reset()
            for sm in self.leg_smooth.values():
                sm.reset()
            return
        vis, pose_w = pr.value.visibility, self.pose_w
        up, self.leg_up_src = self._leg_up()
        for sd in self.legs:
            missing, foot = legs_visible(vis, sd)
            self.leg_missing[sd] = missing
            if missing:
                self.leg_smooth[sd].reset()
                self.legs[sd] = LegAngles()
                continue
            raw = self.leg_kins[sd].solve(pose_w, up, foot=foot)
            sm = self.leg_smooth[sd]({k: getattr(raw, k) for k in LEG_FIELDS}, t)
            self.legs[sd] = dataclasses.replace(raw, **sm)
        if core_visible(vis):
            c = core_angles(pose_w, up)
            self.core = CoreAngles(**self.core_smooth({k: getattr(c, k) for k in CORE_FIELDS}, t))
        else:
            self.core_smooth.reset()
            self.core = CoreAngles()

    def _calibrate_body(self):
        """Key z: arms from the neutral arm pose (desk use) and, when standing in full view, legs and feet."""
        pr = self.pose_res
        if pr is None or pr.value is None or self.pose_w is None:
            self.say("calibration skipped: no body in view")
            return
        pose_w, vis = self.pose_w, pr.value.visibility
        msgs = []
        usable = [sd for sd in self.sides if not self.missing[sd] and not self.angs[sd].elbow_predicted]
        if usable:
            self.up_cal = up_from_neutral(pose_w, "both" if len(usable) == 2 else usable[0])
            for k in self.kins.values():
                k.reset()
            for sm in self.ang_smooth.values():
                sm.reset()
            msgs.append(f"torso calibrated from the neutral pose ({'+'.join(usable)} arm)")
        else:
            msgs.append("arm calibration skipped: shoulders, elbows and wrists must be in view")
        seen = {sd: legs_visible(vis, sd) for sd in ("left", "right")}
        if all(not miss and foot for miss, foot in seen.values()):
            up = up_from_standing(pose_w)
            ok = [sd for sd in ("left", "right") if self.leg_kins[sd].calibrate(pose_w, up)]
            if len(ok) == 2:
                self.leg_up_cal = up
                for sm in list(self.leg_smooth.values()) + [self.core_smooth]:
                    sm.reset()
                msgs.append("standing calibration done (legs, feet, core)")
            else:
                for sd in ok:
                    self.leg_kins[sd].clear_calibration()
                msgs.append("standing calibration skipped: stand upright with straight legs, feet flat")
        elif any(not miss for miss, _ in seen.values()):
            msgs.append("standing calibration skipped: both legs and feet must be in view")
        self.say("; ".join(msgs))

    def _update_emotion(self, r):
        v = r.value
        self.emotion_res = v
        if v is None:
            self.emotions = None
            self.emo_smooth.reset()
            return
        if self.cal_session is not None:
            self.cal_session.add(v["logits"], r.t_capture)
        who = v["identity"] or (None if v["enrolled"] else self.cfg.name)
        cal = self.calibrations.get(who) if who else None
        person = who if cal is not None else None
        if person != self.cal_person:              # a different correction: do not blend across it
            self.cal_person = person
            self.emo_smooth.reset()
        probs = cal.probabilities(v["logits"]) if cal is not None else softmax(v["logits"])
        self.emotions = dict(zip(EMOTIONS, self.emo_smooth(probs, r.t_capture)))

    def _finish_emotion_calibration(self):
        session, self.cal_session = self.cal_session, None
        try:
            cal = session.result()
        except ValueError as exc:
            self.say(f"emotion calibration failed: {exc}; press k to try again")
            return
        cal.save(calibration_path(self.cfg.faces_dir, session.name))
        self.calibrations[session.name] = cal
        self.cal_person = None
        self.say(f"emotion calibrated for {session.name}: {cal.summary()} ({cal.frames} frames)")

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
        self._update_gaze(L, hp, r.t_capture)

    def _update_gaze(self, L, pose, t):
        """Eye angles, the calibrated gaze point on the video (as displayed), dwell locking."""
        self.eye = eye_state(L, pose, self.size[0], self.size[1], self.K)
        e = self.eye
        if not e.valid or e.plane_point is None:
            return                                   # blink or looking away: keep the last gaze point
        if self.gaze_session is not None:
            self.gaze_session.add(e.plane_point, float(np.linalg.norm(np.append(e.plane_point, 0.0) - e.origin)), t)
        if self.gaze_map is not None:
            self.gaze_pt = self.gaze_filter(self.gaze_map.predict(e.plane_point), t)
            if self.target.update(self.gaze_pt, t):
                self.say(f"target locked at {self._target_text()} (held gaze)")

    def _target_text(self):
        c = self.target.camera_point()
        return f"({c[0]:.0f}, {c[1]:.0f}) px" if c is not None else "--"

    def _toggle_target(self, how):
        if self.gaze_map is None:
            self.say("target lock needs gaze calibration first: press g")
            return
        res = self.target.toggle(self.gaze_pt, time.monotonic())
        if res == "locked":
            self.say(f"target locked at {self._target_text()} ({how})")
        elif res == "released":
            self.say(f"target released ({how})")
        else:
            self.say("no gaze point yet: look at the video")

    def _finish_gaze_calibration(self):
        session, self.gaze_session = self.gaze_session, None
        try:
            self.gaze_map = session.result()
        except ValueError as exc:
            self.say(f"gaze calibration failed: {exc}; press g to try again")
            return
        self.gaze_filter.reset()
        self.gaze_pt = None
        self.say(f"gaze calibrated: {self.gaze_map.summary()}")

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
        """The readout. Body mode: one table with both arms and the core (left) and both legs (right)."""
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
            lines += self.grasp_lines()
        else:
            lines += self.body_table()
            if self.pose_res is None or self.pose_res.value is None:
                lines.append("no body in view")
            else:
                gone = []
                for sd in ("left", "right"):
                    if sd in self.sides and self.missing[sd]:
                        gone.append(f"{sd} arm ({', '.join(self.missing[sd])})")
                    if self.leg_missing[sd]:
                        gone.append(f"{sd} leg ({', '.join(self.leg_missing[sd])})")
                if gone:
                    lines.append("not in view: " + "; ".join(gone))
                pred = [sd for sd in self.sides if self.angs[sd].elbow_predicted]
                if pred:
                    lines.append("elbow predicted (out of view): " + ", ".join(pred))
            lines.append(f"torso up: {self.torso_src}   legs up: {self.leg_up_src}"
                         + ("   (z: calibrate)" if "camera vertical" in (self.torso_src, self.leg_up_src) else ""))
            lines.append("gesture: " + "  ".join(f"{sd} {self.gestures[sd]}" for sd in self.sides))
            lines += self.grasp_lines()
            lines.append(self.gaze_line())
        if self.message and now - self.message_t < 4.0:
            lines.append(self.message)
        lines.append(self.status_line(now))
        keys = "q quit  w wire  f fingers  o objects  " + ("z zero hand" if self.pov else
                                                             "z calibrate  h head zero  e enroll  k emotion cal  g gaze cal  t lock  c checks")
        lines.append(keys)
        return lines

    def body_table(self):
        """Rows of "arm/core label, L, R, leg label, L, R" (tab separated; draw with align "lrrlrr").

        Flags after a value: ! outside the normal range, h held (shoulder rotation with a straight elbow, or hip
        rotation without the foot in view), f hip rotation from the foot (knee straight; ankle abd/add is then 0).
        """
        def cell(v, flag=""):
            return f"{num(v)}{flag}" if np.isfinite(v) else "--"

        arm_bad = {sd: set(out_of_range(self.angs[sd])) for sd in self.sides}
        leg_bad = {sd: set(leg_out_of_range(self.legs[sd])) for sd in self.legs}
        left_rows = []
        for k in ANGLE_FIELDS:
            cells = []
            for sd in ("left", "right"):
                if sd not in self.sides:
                    cells.append("")
                    continue
                a = self.angs[sd]
                v = getattr(a, k)
                flag = "!" if k in arm_bad[sd] else ("h" if k == "shoulder_rotation" and np.isfinite(v)
                                                     and not a.shoulder_rotation_valid else "")
                cells.append(cell(v, flag))
            left_rows.append([LABELS[k], *cells])
        for k in CORE_FIELDS:
            left_rows.append([CORE_LABELS[k], cell(getattr(self.core, k)), ""])
        right_rows = []
        for k in LEG_FIELDS:
            cells = []
            for sd in ("left", "right"):
                a = self.legs[sd]
                v = getattr(a, k)
                flag = "!" if k in leg_bad[sd] else ""
                if k == "hip_rotation" and np.isfinite(v):
                    flag += {"foot": "f", "held": "h"}.get(a.hip_rotation_source, "")
                cells.append(cell(v, flag))
            right_rows.append([LEG_LABELS[k], *cells])
        n = max(len(left_rows), len(right_rows))
        left_rows += [["", "", ""]] * (n - len(left_rows))
        right_rows += [["", "", ""]] * (n - len(right_rows))
        return ["\tL\tR\t\tL\tR"] + ["\t".join(a + b) for a, b in zip(left_rows, right_rows)]

    def grasp_lines(self):
        """One line per hand (what it holds and the grip for it) and the objects in view."""
        if "objects" not in self.workers:
            return ["grasp: off (--no-objects)"]
        lines = ["grasp " + g.text() for g in self.grasp.values()]
        seen = [tr.label for tr in self.obj_tracker.confirmed()]
        lines.append("objects: " + (", ".join(seen[:6]) + (f" (+{len(seen) - 6})" if len(seen) > 6 else "") if seen else "none"))
        return lines

    def gaze_line(self):
        e = self.eye
        if "face" not in self.workers:
            return "eyes: off (needs the face model)"
        if e is None:
            return "eyes: no face"
        if not e.valid:
            return "eyes: closed or not found"
        eyes = f"eyes yaw {num(e.yaw, 0)} pitch {num(e.pitch, 0)} (approx.)"
        if self.gaze_map is None:
            return eyes + "   gaze: press g to calibrate"
        target = f"target LOCKED {self._target_text()}" if self.target.locked else "target: t or pinch to lock"
        return f"{eyes}   gaze {self.gaze_map.error_deg:.1f} deg   {target}"

    def finger_lines(self):
        tsd = self.table_cycle[self.table_i]
        if tsd is None:
            return []
        return finger_table(self.fingers[tsd], f"{tsd} fingers") if tsd in self.fingers else [f"{tsd} hand not detected"]

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
        elif self.cal_person is not None:
            note = f"emotion calibrated for {self.cal_person}"
        elif er is not None:
            note = "emotion not calibrated for this face (k)"
        return render_face_panel(size, crop, self.emotions, ident, self.head, rel, note)

    def render(self, frame):
        cv2 = self.cv2
        now = time.monotonic()
        video = frame.bgr.copy()
        if self.show_wire:
            pr = self.pose_res
            if pr is not None and pr.value is not None:
                draw_wireframe(video, pr.value.image, POSE_ARM_EDGES + POSE_LEG_EDGES, CYAN, pr.value.visibility)
            hw = self.workers.get("hand")
            hr = hw.get() if hw is not None else None
            for h in (hr.value or []) if hr is not None else []:
                draw_wireframe(video, h.image, HAND_EDGES, YELLOW, radius=3)
            if self.head is not None:
                draw_head_on_video(video, self.head, self.K)
        if not self.pov:
            video = cv2.flip(video, 1)            # mirror view for a camera facing you
        if self.gaze_session is not None:              # dots only: nothing else on the video to look at
            self._draw_gaze_calibration(video, now)
        else:
            if self.show_objects:
                self._draw_objects(video)
            _, h1 = text_block(video, self.text_lines(now), align=None if self.pov else "lrrlrr")
            text_block(video, self.finger_lines(), origin=(0, h1 + 4))
            self._draw_gaze(video, now)
        if self.cal_session is not None:
            self._banner(video, self.cal_session.prompt(now))
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
                                    head_axes=self.neck_axes(),
                                    legs_by_side={sd: self.legs[sd] for sd in self.legs if not self.leg_missing[sd]},
                                    core=self.core, full_body=True, gaze_dir=self.gaze_in_torso())
        return compose_layout(video, self.face_panel(frame, size), kin, self.cfg.window_width)

    def _banner(self, img, text, top=False):
        """Large instruction across the bottom (or top) of the video (calibrations)."""
        cv2 = self.cv2
        h, w = img.shape[:2]
        scale = max(0.6, w / 1280 * 0.85)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        y0, y1 = (10, th + 44) if top else (h - th - 34, h - 10)
        roi = img[y0:y1, 10:min(w - 10, tw + 40)]
        roi[:] = (roi * 0.35).astype(img.dtype)
        cv2.putText(img, text, (25, y1 - 16), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 230, 255), 2, cv2.LINE_AA)

    def _draw_gaze_calibration(self, img, now):
        cv2 = self.cv2
        s = self.gaze_session
        i, phase, left = s.phase(now)
        if phase == "done":
            self._banner(img, "Gaze calibration: fitting ...", top=True)
            return
        x, y = (int(round(v)) for v in s.targets[i])
        if phase == "settle":                          # a ring closing in on the dot draws the eyes to it
            r = int(14 + 40 * left / s.settle)
            cv2.circle(img, (x, y), r, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 11, (40, 40, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        self._banner(img, f"Gaze calibration {i + 1}/{len(s.targets)}: look at the red dot, keep your head still",
                     top=True)

    def _draw_gaze(self, img, now):
        """Gaze cursor and the locked target, on the video as displayed."""
        cv2 = self.cv2
        if self.gaze_pt is not None and self.gaze_map is not None:
            g = tuple(int(round(v)) for v in self.gaze_pt)
            cv2.circle(img, g, 16, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(img, g, 2, (255, 255, 0), -1, cv2.LINE_AA)
            prog = self.target.dwell_progress(now)
            if prog > 0 and not self.target.locked:
                cv2.ellipse(img, g, (22, 22), -90, 0, int(360 * prog), (0, 200, 255), 3, cv2.LINE_AA)
        if self.target.locked:
            x, y = (int(round(v)) for v in self.target.display_point)
            red = (40, 40, 255)
            cv2.rectangle(img, (x - 28, y - 28), (x + 28, y + 28), red, 2, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(img, (x + 12 * dx, y + 12 * dy), (x + 40 * dx, y + 40 * dy), red, 2, cv2.LINE_AA)
            cv2.putText(img, "LOCKED", (x - 28, y - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2, cv2.LINE_AA)

    def _draw_objects(self, img):
        """Object boxes on the video as displayed; a held object in its hand's color with the grip to use."""
        cv2 = self.cv2
        w = img.shape[1]
        holder = {g.object_id: g for g in self.grasp.values() if g.state == "holding"}
        colors = {"right": (255, 255, 0), "left": (203, 112, 255)}
        for tr in self.obj_tracker.confirmed():
            x0, y0, x1, y1 = tr.box
            if not self.pov:                            # the video is mirrored
                x0, x1 = w - 1 - x1, w - 1 - x0
            p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
            g = holder.get(tr.id)
            color = colors[g.side] if g else (230, 230, 230)
            cv2.rectangle(img, p0, p1, color, 3 if g else 1, cv2.LINE_AA)
            text = f"{tr.label} {tr.score:.2f}"
            if g is not None and g.plan is not None:
                text += f"  grip {g.plan.force_n:.1f} N"
            cv2.putText(img, text, (p0[0], max(14, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (p0[0], max(14, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def gaze_in_torso(self):
        """Gaze direction in the torso frame (for the 3D panel), when the eyes and the torso are known."""
        e = self.eye
        if e is None or not e.valid or self.thorax is None:
            return None
        return self.thorax.T @ e.direction

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
        elif k == "k":
            if "emotion" not in self.workers:
                self.say("emotion tracking is off")
            elif self.cal_session is not None:
                self.cal_session = None
                self.say("emotion calibration cancelled")
            else:
                er = self.emotion_res
                who = (er or {}).get("identity") or self.cfg.name
                self.cal_session = CalibrationSession(who, time.monotonic(), self.cfg.cal_prepare_s,
                                                      self.cfg.cal_capture_s, model=self.cfg.emotion_model)
                self.say(f"emotion calibration for {who}: follow the prompts ({self.cal_session.duration:.0f} s)")
        elif k == "g":
            if "face" not in self.workers:
                self.say("eye tracking needs the face model (it is off)")
            elif self.gaze_session is not None:
                self.gaze_session = None
                self.say("gaze calibration cancelled")
            else:
                self.gaze_session = GazeCalibrationSession(calibration_targets(*self.size), time.monotonic(),
                                                           self.cfg.gaze_settle_s, self.cfg.gaze_capture_s)
                self.say(f"gaze calibration: look at each dot ({self.gaze_session.duration:.0f} s)")
        elif k == "t":
            self._toggle_target("key t")
        elif k == "o":
            self.show_objects = not self.show_objects
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
            cols += GRASP_COLS + ["objects_seen"]
        else:
            cols = ["t"]
            for sd in self.sides:
                p = sd[0].upper() + "_" if len(self.sides) > 1 else ""
                cols += [p + "gesture", *[p + k for k in ANGLE_FIELDS], p + "shoulder_rotation_valid",
                         p + "wrist_valid", p + "elbow_predicted", *[p + c for c in FINGER_COLS]]
            for sd in ("left", "right"):
                p = sd[0].upper() + "_"
                cols += [p + k for k in LEG_FIELDS] + [p + "hip_rotation_source"]
            cols += CORE_FIELDS + ["leg_up_source"]
            cols += ["torso_source", "head_x_cm", "head_y_cm", "head_z_cm", "head_yaw", "head_pitch", "head_roll",
                     "neck_yaw", "neck_pitch", "neck_roll", *[f"emotion_{e}_pct" for e in EMOTIONS], "identity", "match",
                     "emotion_calibrated_for", "eyes_open", "eye_yaw", "eye_pitch", "gaze_x_px", "gaze_y_px",
                     "target_locked", "target_x_px", "target_y_px", "target_ray_x", "target_ray_y", "target_ray_z"]
            for sd in self.sides:
                p = sd[0].upper() + "_" if len(self.sides) > 1 else ""
                cols += [p + c for c in GRASP_COLS]
            cols.append("objects_seen")
        self.writer.writerow(cols)

    def grasp_row(self, sd):
        nan = float("nan")
        g = self.grasp.get(sd) or GraspState(sd)
        pl = g.plan if g.state == "holding" else None
        return [g.state, g.object_name if g.state != "free" else "", g.shape, pl.grasp if pl else "",
                round(pl.force_n, 2) if pl else nan, round(pl.command, 3) if pl else nan,
                round(100 * pl.open_m, 1) if pl else nan]

    def objects_seen(self):
        return ";".join(f"{tr.label}@{tr.id}" for tr in self.obj_tracker.confirmed())

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
                                    if rel else [nan] * 6), *[round(fv.get(c, nan), 1) for c in FINGER_COLS],
                                  *self.grasp_row(sd), self.objects_seen()])
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
        for sd in ("left", "right"):
            a = self.legs[sd]
            row += [round(getattr(a, k), 2) for k in LEG_FIELDS] + [a.hip_rotation_source]
        row += [round(getattr(self.core, k), 2) for k in CORE_FIELDS] + [self.leg_up_src]
        row += [self.torso_src, *(np.round(h.position_cm, 2) if h else [nan] * 3),
                *([round(h.yaw, 2), round(h.pitch, 2), round(h.roll, 2)] if h else [nan] * 3),
                *[round(v, 2) for v in neck], *emo, (er or {}).get("identity") or "",
                round(er["match"], 3) if er else nan, self.cal_person or ""]
        e = self.eye
        g = self.target.camera_point(self.gaze_pt) if self.gaze_pt is not None else None
        tp, ray = self.target.camera_point() if self.target.locked else None, self.target.ray()
        row += [int(bool(e and e.valid)), *([round(e.yaw, 2), round(e.pitch, 2)] if e and e.valid else [nan, nan]),
                *(np.round(g, 1) if g is not None else [nan, nan]), int(self.target.locked),
                *(np.round(tp, 1) if tp is not None else [nan, nan]), *(np.round(ray, 4) if ray is not None else [nan] * 3)]
        for sd in self.sides:
            row += self.grasp_row(sd)
        row.append(self.objects_seen())
        self.writer.writerow(row)

    # ------------------------------------------------------------------ loop
    def step(self, frame):
        if self.update() and self.data_t is not None:
            self.write_row(self.data_t)
        return self.render(frame)

    def run(self, show=None, waitkey=None):
        cv2 = self.cv2
        show = show or (lambda img: cv2.imshow("Tandem", img))
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

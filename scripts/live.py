"""Live tracking: arms, fingers, gestures, head pose, face recognition and emotion, from a webcam.

Body mode (default): seven arm joint angles per arm, finger joints and gestures, 10 joint angles per leg
and 3 in the core (when they are in view), the head's 6 degrees of freedom, and emotion and face
recognition, from a webcam that sees you.
    python scripts/live.py                              # everything: both arms, hands, legs, core, head, face
    python scripts/live.py --torso desk                 # seated at a desk, hips hidden
    python scripts/live.py --side right                 # one arm only
    python scripts/live.py --gaze-dwell 1.0             # also lock a target by looking at it for 1 s
POV mode: hands only, for a head-mounted camera (goggles). Hand position and orientation, finger joints.
    python scripts/live.py --mode pov --hand right --fov 70

Each model runs in its own thread on the newest camera frame, so hand tracking never waits for the body,
face or emotion models. The status line shows the camera rate and each model's rate and latency.

Keys (click the video window first):
    q  quit
    z  body: calibrate. Arms: torso up from the neutral arm pose (needed for --torso desk). Legs: standing
             upright with straight legs and feet flat, in full view, sets "up" and zeroes the feet.
       pov:  set the current hand pose as zero for relative motion
    h  body: set the current head pose as zero for relative head motion
    e  body: enroll your face for recognition, under --name (press a few times from different angles)
    k  body: calibrate the emotion tracker to your face: make each expression when asked (about 30 s)
    g  body: calibrate eye tracking: look at each red dot until it moves (9 dots, about 20 s)
    t  body: lock onto (or release) the spot you are looking at; a pinch with either hand does the same
    f  cycle the finger joint table (off, then each tracked hand)
    c  body: print axis and hand/pose agreement checks to the terminal
    w  toggle the wireframe overlay
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # let `python scripts/live.py` find vtrack

from vtrack.app import App, Config                              # noqa: E402
from vtrack.filters import SMOOTHING_PRESETS                    # noqa: E402
from vtrack.pipeline import open_camera                         # noqa: E402


def parse_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description="Homunculus Shadowing live tracker")
    ap.add_argument("--mode", choices=["body", "pov"], default="body")
    ap.add_argument("--camera", type=int, default=0, help="camera number (0 is the built-in webcam)")
    ap.add_argument("--width", type=int, default=1280, help="requested camera width")
    ap.add_argument("--height", type=int, default=720, help="requested camera height")
    ap.add_argument("--fps", type=int, default=60, help="requested camera frame rate (the camera may give less)")
    ap.add_argument("--side", choices=["left", "right", "both"], default="both",
                    help="body mode: arm(s) to track (legs, core and head are always tracked)")
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
    ap.add_argument("--fov", type=float, default=70.0,
                    help="camera horizontal field of view in degrees (head and POV hand distance)")
    ap.add_argument("--no-face", action="store_true", help="body mode: turn off head pose tracking")
    ap.add_argument("--no-emotion", action="store_true", help="body mode: turn off emotion and face recognition")
    ap.add_argument("--emotion-model", choices=["hsemotion", "opencv"], default="hsemotion",
                    help="emotion model: hsemotion (default, much better at fear) or opencv (the first one)")
    ap.add_argument("--gaze-dwell", type=float, default=0.0,
                    help="body mode: lock a target by looking at one spot this many seconds (0 = off; "
                         "use key t or a pinch)")
    ap.add_argument("--name", default="me", help="name stored when you enroll your face (key e)")
    ap.add_argument("--faces-dir", default="faces", help="where enrolled faces are stored")
    ap.add_argument("--model-dir", default="models", help="where model files are downloaded")
    ap.add_argument("--window-width", type=int, default=1600, help="maximum window width in pixels")
    ap.add_argument("--record", type=Path, help="write per-update values to CSV")
    a = ap.parse_args(argv)
    if a.gaze_dwell < 0:
        ap.error("--gaze-dwell must be 0 or more")
    if not a.name.replace("-", "").replace("_", "").isalnum():
        ap.error("--name may contain only letters, digits, - and _")
    return Config(mode=a.mode, camera=a.camera, width=a.width, height=a.height, fps=a.fps, side=a.side,
                  torso=a.torso, pose_model=a.pose_model, smoothing=a.smoothing, pose_only_wrist=a.pose_only_wrist,
                  hand=a.hand, swap_hands=a.swap_hands, fov=a.fov, face=not a.no_face, emotion=not a.no_emotion,
                  emotion_model=a.emotion_model, gaze_dwell=a.gaze_dwell,
                  name=a.name, faces_dir=a.faces_dir, model_dir=a.model_dir, window_width=a.window_width,
                  record=a.record)


def main(argv=None):
    cfg = parse_args(argv)
    import cv2
    cap, actual = open_camera(cfg.camera, cfg.width, cfg.height, cfg.fps)
    if cap is None:
        raise SystemExit(f"cannot open camera {cfg.camera}. Close other apps that use the camera, or try "
                         "--camera 1. On a Mac, allow camera access for your terminal in System Settings, "
                         "Privacy & Security, Camera.")
    w, h, fps = actual
    print(f"camera {cfg.camera}: {w}x{h}, asked for {cfg.fps} fps, driver reports {fps:.0f} fps")
    print("loading models (the first run downloads them) ...")
    app = App(cfg, cap)
    print("running; click the video window, then press q to quit")
    try:
        app.run()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

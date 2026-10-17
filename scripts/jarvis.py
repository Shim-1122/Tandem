"""Jarvis: the locally hosted brain for the Tandem robot. Sees you through Tandem, hears you, drives the robot.

    python scripts/jarvis.py                          # camera + simulated robot, type commands in this terminal
    python scripts/jarvis.py --voice                  # also "Hey Jarvis" by voice, replies spoken (Piper)
    python scripts/jarvis.py --brain rules            # no language model: keyword commands only
    python scripts/jarvis.py --model llama3.1:8b      # a different Ollama model (default qwen2.5:7b)
    python scripts/jarvis.py --no-camera              # robot and commands only, no tracking
    python scripts/jarvis.py --demo                   # a scripted visitor instead of the camera
    python scripts/jarvis.py --demo games             # scripted voice commands: rock paper scissors, then shadow
    python scripts/jarvis.py --demo games --video g.mp4   # save a demo as a video (no windows needed)

Keys in the video windows: q quit, space stop (freeze), r rest, s shadow mode on and off, plus Tandem's own keys.
Type in the terminal at any time; "stop" or "freeze" always stops the robot at once.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # let `python scripts/jarvis.py` find the packages

from jarvis import Jarvis, OllamaBrain, RuleBrain                # noqa: E402
from jarvis.voice import Keyboard                                # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Jarvis for the Tandem robot (runs locally)")
    ap.add_argument("--brain", choices=["ollama", "rules"], default="ollama",
                    help="ollama: a local language model (falls back to rules if Ollama is not running)")
    ap.add_argument("--model", default="qwen2.5:7b", help="Ollama model with tool support")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--voice", action="store_true", help="listen for 'Hey Jarvis' and speak replies")
    ap.add_argument("--voice-model", type=Path, default=Path("models/en_GB-alan-medium.onnx"),
                    help="Piper voice (.onnx); download with: python -m piper.download_voices en_GB-alan-medium")
    ap.add_argument("--whisper", default="base.en", help="faster-whisper model: tiny.en, base.en, small.en")
    ap.add_argument("--mic", default=None, help="microphone device number or name (default: system default)")
    ap.add_argument("--no-reflexes", action="store_true", help="only act when asked")
    ap.add_argument("--no-sim", action="store_true", help="do not open the simulated robot")
    ap.add_argument("--sim-camera", choices=["front", "side", "top", "eyes"], default="front")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--no-camera", action="store_true", help="no tracking (commands only)")
    src.add_argument("--demo", nargs="?", const="visitor", choices=["visitor", "games"],
                     help="a scripted person instead of the camera: visitor (reflexes and commands) or games "
                          "(rock paper scissors, then shadow mode)")
    ap.add_argument("--video", type=Path, help="with --demo: write the robot to this video instead of a window")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--torso", choices=["auto", "hips", "desk"], default="desk")
    ap.add_argument("--name", default="me", help="your enrolled face name in Tandem (press e to enroll)")
    ap.add_argument("--object-model", choices=["medium", "small"], default="small")
    return ap.parse_args(argv)


def make_brain(a):
    if a.brain == "rules":
        return RuleBrain()
    b = OllamaBrain(model=a.model, host=a.ollama_host)
    if b.available():
        print(f"brain: {a.model} through Ollama")
    else:
        print(f"brain: Ollama not ready ({b.last_error}); using keyword rules until it is")
    return b


def make_voice(a, jarvis):
    if not a.voice:
        return None
    try:
        from jarvis.voice import LocalVoice
        mic = int(a.mic) if a.mic and a.mic.isdigit() else a.mic
        voice = LocalVoice(jarvis.hear, whisper_model=a.whisper, device=mic,
                           voice_model=str(a.voice_model) if a.voice_model.exists() else None)
    except ImportError as e:
        print(f"voice off: {e}. Install it with: pip install -r requirements-jarvis.txt")
        return None
    if voice.tts is None:
        print(f"no Piper voice at {a.voice_model}: replies are printed only")
    jarvis.speak_fn = voice.speak
    jarvis.on_wake = voice.wake_up
    voice.start()
    return voice


def run_demo_video(a, jarvis, robot, fps=30, size=(960, 540)):
    import cv2
    import random
    from jarvis import demo
    script = demo.SCRIPTS[a.demo]()
    jarvis.rng = random.Random(6)                    # the same robot throws every time, for a repeatable video
    out = cv2.VideoWriter(str(a.video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    clock = [0.0]
    jarvis.clock = lambda: clock[0]                  # the video's own time, not the wall clock
    for k in range(int(script.LENGTH * fps)):
        t = clock[0] = k / fps
        values, words = script.step(t, jarvis)
        if words:
            jarvis.hear(words, wait=True)
        jarvis.tick(t, values)
        img = robot.render(a.sim_camera, *size)[:, :, ::-1].copy()
        state = jarvis.game.status() if jarvis.game else jarvis.body.state()
        lines = [f"t {t:4.1f}s   {state}"] + [f"{who}: {txt}" for who, txt in jarvis.transcript[-4:]]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (14, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                        cv2.LINE_AA)
        out.write(img)
    out.release()
    print(f"wrote {a.video}")


def main(argv=None):
    a = parse_args(argv)
    robot = None
    if not a.no_sim:
        from jarvis.robot import SimRobot
        robot = SimRobot()
    jarvis = Jarvis(brain=make_brain(a), robot=robot, reflexes=not a.no_reflexes)
    if a.demo and a.video:
        run_demo_video(a, jarvis, robot)
        return
    make_voice(a, jarvis)
    Keyboard(jarvis.hear).start()
    print("Jarvis is up. Type a command (or say 'Hey Jarvis' with --voice). Ctrl+C or q in a window quits.")

    app = None
    if not (a.no_camera or a.demo):
        from vtrack.app import App, Config
        from vtrack.pipeline import open_camera
        cfg = Config(camera=a.camera, torso=a.torso, name=a.name, object_model=a.object_model)
        cap, _ = open_camera(cfg.camera, cfg.width, cfg.height, cfg.fps)
        if cap is None:
            raise SystemExit(f"cannot open camera {a.camera}; try --camera 1, or --no-camera")
        print("loading Tandem's models ...")
        app = App(cfg, cap)

    import cv2
    from jarvis import demo
    start = time.monotonic()
    script = demo.SCRIPTS[a.demo]() if a.demo else None
    try:
        while True:
            now = time.monotonic()
            if app is not None:
                frame = app.camera.wait_newer(getattr(app, "_jarvis_last", 0), timeout=0.05)
                if frame is not None:
                    app._jarvis_last = frame.index
                    app.update()
                    cv2.imshow("Tandem", app.render(frame))
                values = app.values()
            elif script is not None:
                values, words = script.step(now - start, jarvis)
                if words:
                    jarvis.hear(words)
            else:
                values = {}
                time.sleep(1 / 60)
            jarvis.tick(now, values)
            if robot is not None:
                img = robot.render(a.sim_camera, 800, 600)[:, :, ::-1].copy()
                state = jarvis.game.status() if jarvis.game else jarvis.body.state()
                cv2.putText(img, f"Jarvis: {state}" + ("  (thinking)" if jarvis.thinking else ""),
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Jarvis robot", img)
            key = cv2.waitKey(1) & 0xFF if (robot is not None or app is not None) else 255
            if key == ord("q"):
                break
            if key == ord(" "):
                jarvis.hear("stop")
            elif key == ord("r"):
                jarvis.act("rest")
            elif key == ord("s"):
                jarvis.act("shadow", {"on": not jarvis.body.shadow})
            elif app is not None and key != 255:
                app.handle_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        if app is not None:
            app.close()
        if robot is not None:
            robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

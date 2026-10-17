"""Jarvis: perception events, skills and speed limits, reflexes, both brains, and the loop driving the robot."""
import http.server
import json
import math
import threading

import pytest

from jarvis import Jarvis, OllamaBrain, Perception, Reflexes, RuleBrain
from jarvis import demo
from jarvis.brain import is_stop
from jarvis.perception import Event
from jarvis.robot import NullRobot
from jarvis.skills import Body, rest_pose, speed_limit

NAN = float("nan")


def face(x=0.0, y=0.0, z=80.0, name="shyam", **more):
    return {"head_x_cm": x, "head_y_cm": y, "head_z_cm": z, "identity": name, **more}


def run(p, values, t0, t1, dt=0.05):
    ev, t = [], t0
    while t <= t1 + 1e-9:
        ev += p.update(values(t) if callable(values) else values, t)
        t += dt
    return ev


# ------------------------------------------------------------------ perception
def test_person_arrives_after_half_a_second_and_leaves_after_two():
    p = Perception()
    assert run(p, face(), 0.0, 0.4) == []
    ev = run(p, face(), 0.45, 0.6)
    assert [e.kind for e in ev] == ["person_arrived"] and ev[0].data["name"] == "shyam"
    assert run(p, {"head_z_cm": NAN}, 0.65, 2.5) == []                # a short dropout is not leaving
    ev = run(p, {"head_z_cm": NAN}, 2.55, 3.0)
    assert [e.kind for e in ev] == ["person_left"] and not p.present


def test_gesture_fires_once_while_held_and_again_after_a_change():
    p = Perception()
    run(p, face(), 0.0, 1.0)
    ev = run(p, face(R_gesture="thumbs_up", L_gesture="unknown"), 1.05, 3.0)
    assert [(e.data["side"], e.data["gesture"]) for e in ev] == [("right", "thumbs_up")]
    run(p, face(R_gesture="unknown", L_gesture="unknown"), 3.05, 3.5)
    ev = run(p, face(R_gesture="thumbs_up", L_gesture="unknown"), 3.55, 4.0)
    assert len(ev) == 1


def test_a_one_frame_gesture_is_ignored():
    p = Perception()
    run(p, face(), 0.0, 1.0)
    assert p.update(face(R_gesture="fist"), 1.05) == []
    assert run(p, face(R_gesture="unknown"), 1.1, 2.0) == []


def test_emotion_needs_a_strong_held_score():
    p = Perception()
    run(p, face(), 0.0, 1.0)
    weak = face(emotion_sad_pct=40.0, emotion_happy_pct=5.0)
    assert run(p, weak, 1.05, 3.0) == []
    strong = face(emotion_sad_pct=70.0)
    ev = run(p, strong, 3.05, 4.5)
    assert [e.data["emotion"] for e in ev] == ["sad"]
    assert "sad" in p.describe()


def test_grasp_and_release():
    p = Perception()
    held = face(L_grasp_state="holding", L_grasp_object="tennis ball")
    ev = p.update(held, 0.0)
    assert [(e.kind, e.data["object"]) for e in ev] == [("grasp", "tennis ball")]
    ev = p.update(face(L_grasp_state="free", L_grasp_object=""), 0.1)
    assert [(e.kind, e.data["object"]) for e in ev] == [("release", "tennis ball")]


def test_head_direction_uses_the_robot_conventions():
    p = Perception()
    p.update(face(x=80.0, y=0.0, z=80.0), 0.0)                        # to the camera's right
    yaw, pitch = p.head_direction()
    assert yaw == pytest.approx(-45.0) and pitch == pytest.approx(0.0)  # the robot turns right: yaw negative
    p.update(face(x=0.0, y=-80.0, z=80.0), 0.1)                       # above the camera (y is down)
    assert p.head_direction()[1] == pytest.approx(45.0)               # look up: pitch positive


# ------------------------------------------------------------------ body
def test_every_joint_respects_its_speed_limit():
    b = Body(blink=False)
    from jarvis import skills
    b.do(skills.celebrate(), 0.0)
    prev = b.targets(0.0)
    for k in range(1, 60):
        t = k / 30
        cur = b.targets(t)
        for c, v in cur.items():
            lim = speed_limit(c)
            if lim is not None and c in prev:
                assert abs(v - prev[c]) <= lim / 30 + 1e-6, c
        prev = cur
    assert b.out["R_shoulder_flexion"] > 100                           # it did get the arms up


def test_wave_moves_only_its_arm_then_returns_to_rest():
    from jarvis import skills
    b = Body(blink=False)
    b.do(skills.wave("right"), 0.0)
    for k in range(60):
        out = b.targets(k / 30)
    assert out["R_shoulder_abduction"] > 50 and abs(out["L_shoulder_abduction"] - 5) < 1e-6
    for k in range(60, 300):
        out = b.targets(k / 30)
    assert not b.busy and out["R_shoulder_abduction"] == pytest.approx(rest_pose()["R_shoulder_abduction"])


def test_freeze_holds_the_pose_exactly():
    from jarvis import skills
    b = Body(blink=False)
    b.do(skills.wave("right"), 0.0)
    for k in range(20):
        b.targets(k / 30)
    b.freeze()
    held = b.targets(1.0)
    assert b.targets(3.0) == held and b.state() == "frozen"


def test_copy_mode_copies_the_tracker_joint_for_joint():
    b = Body(blink=False)
    b.set_shadow(True, mirror=False)
    tracker = {"R_elbow_flexion": 100.0, "neck_yaw": 20.0, "L_index_pip": NAN}
    for k in range(120):
        out = b.targets(k / 30, tracker)
    assert out["R_elbow_flexion"] == pytest.approx(100.0) and out["neck_yaw"] == pytest.approx(20.0)
    assert math.isfinite(out["L_index_pip"])                           # a missing value keeps the rest pose


def test_look_at_the_person():
    b = Body(blink=False)
    b.look("person")
    for k in range(90):
        out = b.targets(k / 30, None, (-40.0, 10.0))
    assert out["neck_yaw"] + out["eye_yaw"] == pytest.approx(-40.0)
    assert out["neck_pitch"] + out["eye_pitch"] == pytest.approx(10.0)


# ------------------------------------------------------------------ reflexes
def test_reflex_mirrors_the_hand_and_has_a_cooldown():
    r = Reflexes()
    acts, _ = r.react(Event("gesture", 0.0, {"side": "right", "gesture": "open_palm"}), 0.0)
    assert acts == [("wave", {"side": "left"})]                       # your right hand is across from its left
    assert r.react(Event("gesture", 1.0, {"side": "right", "gesture": "open_palm"}), 1.0) == ([], None)
    assert r.react(Event("gesture", 9.0, {"side": "right", "gesture": "open_palm"}), 9.0)[0]


def test_greeting_uses_the_name():
    acts, speech = Reflexes().react(Event("person_arrived", 0.0, {"name": "shyam"}), 0.0)
    assert speech == "Hello, shyam." and ("look", {"at": "me"}) in acts
    _, speech = Reflexes().react(Event("person_arrived", 0.0, {"name": "unknown"}), 0.0)
    assert speech == "Hello."


# ------------------------------------------------------------------ brains
@pytest.mark.parametrize("text,action,args", [
    ("wave with your left hand", "wave", {"side": "left"}),
    ("give me a thumbs up", "thumbs_up", {}),
    ("point to your left", "point", {"at": "left"}),
    ("look at me", "look", {"at": "me"}),
    ("grip gently with your right hand", "grip", {"side": "right", "strength": 0.5}),
    ("let go", "release", {}),
    ("copy me", "shadow", {"on": True, "mirror": False}),
    ("shadow me", "shadow", {"on": True, "mirror": True}),
    ("mirror my movements", "shadow", {"on": True, "mirror": True}),
    ("rock paper scissors?", "play_rock_paper_scissors", {}),
    ("stop mirroring me", "shadow", {"on": False}),
    ("stop copying me", "shadow", {"on": False}),
    ("FREEZE", "freeze", {}),
    ("relax", "rest", {}),
])
def test_rule_brain_commands(text, action, args):
    _, acts = RuleBrain().respond(text)
    assert acts[0] == (action, args)


def test_rule_brain_compound_command():
    reply, acts = RuleBrain().respond("let go and relax")
    assert acts == [("release", {}), ("rest", {})] and reply == "Relaxing."
    assert [a for a, _ in RuleBrain().respond("nod and then wave")[1]] == ["nod", "wave"]


def test_stop_words():
    assert is_stop("stop!") and is_stop("please hold still") and is_stop("halt")
    assert not is_stop("stop copying me") and not is_stop("wave")


def test_rule_brain_answers_from_the_scene():
    p = Perception()
    run(p, face(L_grasp_state="holding", L_grasp_object="tennis ball"), 0.0, 1.0)
    reply, acts = RuleBrain().respond("what am I holding?", p.describe())
    assert reply == "A tennis ball in the left hand." and acts == []


class _FakeOllama(http.server.BaseHTTPRequestHandler):
    reply = {"message": {"role": "assistant", "content": "**Sure!** Waving now \U0001F44B",
                         "tool_calls": [{"function": {"name": "wave", "arguments": {"side": "left"}}},
                                        {"function": {"name": "nod", "arguments": "{}"}},
                                        {"function": {"name": "self_destruct", "arguments": {}}}]}}
    seen = []

    def do_GET(self):
        self._send({"models": [{"name": "qwen2.5:7b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.seen.append(body)
        self._send(self.reply)

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def ollama():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_brain_tool_calls(ollama):
    b = OllamaBrain(host=ollama)
    assert b.available()
    reply, calls = b.respond("say hi with your left hand and nod", "You can see shyam.")
    assert calls == [("wave", {"side": "left"}), ("nod", {})]         # unknown tools are dropped
    assert reply == "Sure! Waving now"                                 # no markdown or emoji to read aloud
    sent = _FakeOllama.seen[-1]
    assert sent["model"] == "qwen2.5:7b" and "shyam" in sent["messages"][0]["content"]
    assert {t["function"]["name"] for t in sent["tools"]} >= {"wave", "point", "freeze", "shadow"}


def test_ollama_brain_falls_back_when_ollama_is_down():
    b = OllamaBrain(host="http://127.0.0.1:9", timeout=1)
    assert not b.available()
    assert b.respond("nod")[1] == [("nod", {})]


# ------------------------------------------------------------------ the loop
def test_stop_is_immediate_even_with_a_slow_brain():
    class Slow(RuleBrain):
        def respond(self, *a):
            raise AssertionError("stop must not reach the brain")
    j = Jarvis(brain=Slow(), log=lambda *a: None, clock=lambda: 0.0)
    j.act("wave")
    j.hear("stop")
    assert j.body.frozen and j.body.motion is None


def test_several_moves_from_one_request_run_in_order():
    class Two(RuleBrain):
        def respond(self, *a):
            return "Okay.", [("wave", {}), ("nod", {})]
    j = Jarvis(brain=Two(), log=lambda *a: None, clock=lambda: 0.0)
    j.hear("wave then nod", wait=True)
    assert j.body.motion.name == "wave right" and [m.name for m in j.body.queue] == ["nod"]
    assert j.transcript[-1] == ("jarvis", "Okay.")


def test_demo_visitor_triggers_reflexes_and_drives_the_robot():
    robot, log = NullRobot(), []
    j = Jarvis(brain=RuleBrain(), robot=robot, log=log.append)
    t = 0.0
    while t < 15.0:
        j.tick(t, demo.scenario(t))
        t += 1 / 30
    text = "\n".join(map(str, log))
    assert "person_arrived name=shyam" in text and "Hello, shyam." in text
    assert "gesture side=right gesture=open_palm" in text and "wave (left)" in text
    assert "thumbs up (left)" in text
    assert robot.last and all(math.isfinite(v) for v in robot.last.values())


def test_jarvis_drives_the_mujoco_robot():
    pytest.importorskip("mujoco")
    from jarvis.robot import SimRobot
    robot = SimRobot(scene=False)
    j = Jarvis(robot=robot, log=lambda *a: None)
    j.act("wave", {"side": "right"}, now=0.0)
    for k in range(60):
        j.tick(k / 30, {})
    assert robot.sim.angle("R_shoulder_abduction") > 40               # the simulated arm really went up
    robot.close()


# ------------------------------------------------------------------ mirror, wake word, rock paper scissors
def test_mirror_mode_moves_the_opposite_arm_and_turns_the_other_way():
    b = Body(blink=False)
    b.set_shadow(True)                                                  # mirror is the default
    tracker = {"R_shoulder_abduction": 80.0, "L_shoulder_abduction": 5.0, "neck_yaw": 20.0, "neck_pitch": 10.0}
    for k in range(120):
        out = b.targets(k / 30, tracker)
    assert out["L_shoulder_abduction"] == pytest.approx(80.0)            # your right arm, its left arm
    assert out["R_shoulder_abduction"] == pytest.approx(5.0)
    assert out["neck_yaw"] == pytest.approx(-20.0) and out["neck_pitch"] == pytest.approx(10.0)
    assert b.state() == "mirroring you"


@pytest.mark.parametrize("heard,command", [
    ("Hey Jarvis, shadow me.", "shadow me."), ("hey jarvis rock paper scissors?", "rock paper scissors?"),
    ("Jarvis, wave", "wave"), ("Okay Jarvis... nod", "nod"), ("wave at jarvis", "wave at jarvis"),
])
def test_wake_phrase_is_removed(heard, command):
    from jarvis.voice import strip_wake
    assert strip_wake(heard) == command


def test_hey_jarvis_shadow_me_starts_mirroring():
    j = Jarvis(log=lambda *a: None, clock=lambda: 0.0)
    j.hear("Hey Jarvis, shadow me", wait=True)
    assert j.body.shadow and j.body.mirror
    j.hear("Hey Jarvis, stop mirroring me", wait=True)
    assert not j.body.shadow and not j.body.frozen


def _play(j, person_throw, t0, by="gesture", dt=1 / 30, limit=6.0):
    """Run one round from `t0` with a person who pumps a fist, then throws 0.25 s after "shoot"."""
    t, shoot = t0, None
    while t < t0 + limit:
        g, game = "unknown", j.game
        if game is None or game.state == "again" and shoot is not None:
            return t
        if game.state in ("ready", "count"):
            g = "fist"
        elif game.state == "read":
            shoot = t if shoot is None else shoot
            if by == "words" and t - shoot >= 0.2 and not getattr(j, "_said", False):
                j._said = True
                j.hear(person_throw, wait=True)
            g = "fist" if t - shoot < 0.25 or by == "words" else {"rock": "fist", "paper": "open_palm",
                                                                "scissors": "peace"}[person_throw]
        j.tick(t, face(R_gesture=g, L_gesture="unknown"))
        t += dt
    raise AssertionError("round did not finish")


def test_rock_paper_scissors_round_by_voice_and_gesture():
    import random
    from jarvis.games import winner
    j = Jarvis(log=lambda *a: None, clock=lambda: 0.0)
    j.rng = random.Random(3)
    j.hear("Hey Jarvis, rock paper scissors?", wait=True)
    assert j.game is not None and j.game.state == "ready" and j.body.motion.name == "rps ready"
    t = _play(j, "paper", 0.0)
    g = j.game
    robot, person, w = g.result
    assert person == "paper"                       # read as paper, not the fist pumped during the count
    assert w == winner(robot, "paper") and sum(g.score.values()) == 1
    said = [txt for who, txt in j.transcript if who == "jarvis"]
    assert said[1:5] == ["Rock...", "Paper...", "Scissors...", "Shoot!"] and said[-1].endswith("Again?")
    assert j.body.motion is not None and j.body.motion.name == f"rps {robot}"   # its throw is showing
    j.clock = lambda: t
    j.hear("again", wait=True)
    assert j.game.state == "count"
    _play(j, "scissors", t)
    assert j.game.result[1] == "scissors" and sum(j.game.score.values()) == 2
    j.hear("no, that's enough", wait=True)
    j.tick(t + 10, face())
    assert j.game is None and "Good game" in j.transcript[-1][1]


def test_rock_paper_scissors_without_a_camera_by_saying_the_throw():
    import random
    j = Jarvis(log=lambda *a: None, clock=lambda: 0.0)
    j.rng = random.Random(1)
    j.hear("rock paper scissors", wait=True)
    j.hear("ready", wait=True)
    assert j.game.state == "count"
    _play(j, "scissors", 0.0, by="words")
    assert j.game.result[1] == "scissors"


def test_mid_game_gestures_do_not_trigger_reflexes_and_stop_ends_the_game():
    j = Jarvis(log=lambda *a: None, clock=lambda: 0.0)
    for k in range(30):
        j.tick(k / 30, face())                                           # arrives: greeting reflex
    j.hear("rock paper scissors", wait=True)
    for k in range(30, 90):
        j.tick(k / 30, face(R_gesture="open_palm"))                       # would be a wave back outside a game
    assert j.game is not None and "wave" not in j.body.state()
    j.hear("stop")
    assert j.game is None and j.body.frozen


def test_robot_throws_are_fair_and_varied():
    import random
    from jarvis.games import RockPaperScissors
    j = Jarvis(log=lambda *a: None, clock=lambda: 0.0)
    g = RockPaperScissors(j, 0.0, rng=random.Random(0))
    seen = []
    for _ in range(300):
        g._start_count(0.0)
        seen.append(g.robot_throw)
    assert {s: seen.count(s) for s in set(seen)}.keys() == {"rock", "paper", "scissors"}
    assert min(seen.count(s) for s in ("rock", "paper", "scissors")) > 70


def test_game_night_demo_runs_on_the_mujoco_robot():
    pytest.importorskip("mujoco")
    import random
    from jarvis.robot import SimRobot
    robot = SimRobot(scene=False)
    clock = [0.0]
    j = Jarvis(robot=robot, log=lambda *a: None, clock=lambda: clock[0])
    j.rng = random.Random(7)
    script = demo.GameNight()
    left_abd = []
    for k in range(int(script.LENGTH * 30)):
        t = clock[0] = k / 30
        values, words = script.step(t, j)
        if words:
            j.hear(words, wait=True)
        j.tick(t, values)
        if script.phase == "shadow" and t - script.shadow_t > 1.5 and j.body.shadow:
            left_abd.append(robot.sim.angle("L_shoulder_abduction"))
    said = " ".join(txt for who, txt in j.transcript if who == "jarvis")
    assert said.count("Shoot!") == 3 and "Good game" in said and "Mirroring you." in said
    assert max(left_abd) - min(left_abd) > 30                          # its left arm followed your right
    assert not j.body.shadow                                           # "stop mirroring me" ended it
    robot.close()

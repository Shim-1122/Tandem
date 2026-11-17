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
    acts, (template, values) = Reflexes().react(Event("person_arrived", 0.0, {"name": "shyam"}), 0.0)
    assert template.format(**values) == "Hello, shyam." and ("look", {"at": "me"}) in acts
    _, (template, values) = Reflexes().react(Event("person_arrived", 0.0, {"name": "unknown"}), 0.0)
    assert template == "Hello."


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


# ------------------------------------------------------------------ languages
from jarvis import i18n, languages  # noqa: E402


@pytest.mark.parametrize("heard,whisper,command,lang", [
    ("Hey Jarvis, shadow me", "en", "shadow me", "en"),
    ("Wake up Jarvis, wave", "en", "wave", "en"),
    ("Oye Jarvis, imítame", "es", "imítame", "es"),
    ("Despierta Jarvis", "es", "", "es"),
    ("Hey Jarvis, piedra papel o tijera", "es", "piedra papel o tijera", "es"),     # shared "hey": Whisper decides
    ("Réveille-toi Jarvis, salue", "fr", "salue", "fr"),
    ("Hallo Jarvis, mach mich nach", "de", "mach mich nach", "de"),
    ("Jarvis, wach auf! Wink mal", "de", "Wink mal", "de"),
    ("Ehi Jarvis, saluta", "it", "saluta", "it"),
    ("Acorda Jarvis, me imita", "pt", "me imita", "pt"),
    ("अरे जार्विस, हाथ हिलाओ", "hi", "हाथ हिलाओ", "hi"),
    ("嘿贾维斯，石头剪刀布", "zh", "石头剪刀布", "zh"),
    ("ジャービス、起きて。じゃんけんしよう", "ja", "じゃんけんしよう", "ja"),
    ("자비스 일어나, 따라해", "ko", "따라해", "ko"),
    ("Джарвис, проснись, помаши", "ru", "помаши", "ru"),
    ("يا جارفيس، قلدني", "ar", "قلدني", "ar"),
    ("Hey Jarvis, zwaai eens", "nl", "zwaai eens", "nl"),                         # not built in: Whisper decides
])
def test_wake_phrase_sets_the_language(heard, whisper, command, lang):
    found = languages.find_wake(heard)
    assert found is not None and found[0] == command
    assert languages.wake_language(heard, whisper, 0.9, previous="en") == lang


def test_wake_words_beat_a_wrong_whisper_guess_and_only_lead_a_sentence():
    assert languages.wake_language("Despierta Jarvis", "pt", 0.6) == "es"       # the wake word decides
    assert languages.wake_language("Hey Jarvis", "de", 0.3, previous="ja") == "ja"   # unsure: keep the last one
    assert languages.find_wake("wave at Jarvis") is None
    assert languages.find_wake("I told Jarvis to wave") is None
    assert languages.find_wake("no wake word here") is None


@pytest.mark.parametrize("text", ["stop", "FREEZE!", "alto", "detente", "arrête", "stopp", "fermati", "pare", "रुको",
                                  "停下", "止まれ", "멈춰", "стоп", "توقف", "stoppen", "dur", "stanna", "berhenti",
                                  "σταμάτα", "עצור", "dừng lại"])
def test_stop_in_many_languages(text):
    from jarvis.brain import is_stop
    assert is_stop(text)


def test_these_are_not_stops():
    from jarvis.brain import is_stop
    for text in ("stop copying me", "para mí, saluda", "wave", "hola", "parle", "camera"):
        assert not is_stop(text), text


@pytest.mark.parametrize("lang,text,action", [
    ("es", "imítame", ("shadow", {"on": True, "mirror": True})),
    ("es", "saluda con la mano izquierda", ("wave", {"side": "left"})),
    ("es", "piedra, papel o tijera", ("play_rock_paper_scissors", {})),
    ("fr", "imite-moi", ("shadow", {"on": True, "mirror": True})),
    ("fr", "pierre feuille ciseaux", ("play_rock_paper_scissors", {})),
    ("de", "mach mich nach", ("shadow", {"on": True, "mirror": True})),
    ("de", "schere stein papier", ("play_rock_paper_scissors", {})),
    ("it", "imitami", ("shadow", {"on": True, "mirror": True})),
    ("it", "sasso carta forbice", ("play_rock_paper_scissors", {})),
    ("pt", "me imita", ("shadow", {"on": True, "mirror": True})),
    ("pt", "pedra papel tesoura", ("play_rock_paper_scissors", {})),
    ("hi", "मेरी नकल करो", ("shadow", {"on": True, "mirror": True})),
    ("hi", "पत्थर कागज़ कैंची", ("play_rock_paper_scissors", {})),
    ("zh", "模仿我", ("shadow", {"on": True, "mirror": True})),
    ("zh", "石头剪刀布", ("play_rock_paper_scissors", {})),
    ("ja", "真似して", ("shadow", {"on": True, "mirror": True})),
    ("ja", "じゃんけんしよう", ("play_rock_paper_scissors", {})),
    ("ko", "따라해", ("shadow", {"on": True, "mirror": True})),
    ("ko", "가위바위보", ("play_rock_paper_scissors", {})),
    ("ru", "повторяй за мной", ("shadow", {"on": True, "mirror": True})),
    ("ru", "камень ножницы бумага", ("play_rock_paper_scissors", {})),
    ("ar", "قلدني", ("shadow", {"on": True, "mirror": True})),
    ("ar", "حجر ورقة مقص", ("play_rock_paper_scissors", {})),
    ("ru", "помаши левой рукой", ("wave", {"side": "left"})),
    ("ja", "左手で手を振って", ("wave", {"side": "left"})),
])
def test_commands_in_the_built_in_languages_need_no_model(lang, text, action):
    _, acts = RuleBrain().respond(text, lang=lang)
    assert acts[0] == action


def test_every_built_in_language_has_every_line_with_its_placeholders():
    import string
    fields = lambda s: sorted(f for _, f, _, _ in string.Formatter().parse(s) if f)   # noqa: E731
    assert set(i18n.CATALOG_LANGS) | {"en"} == set(languages.BUILT_IN)
    for lang, rows in i18n.CATALOG_LANGS.items():
        assert len(rows) == len(i18n.TEMPLATES), lang
        for en, local in zip(i18n.TEMPLATES, rows):
            assert fields(en) == fields(local), (lang, en, local)
            assert "—" not in local
    for lang in languages.BUILT_IN:
        assert languages.voice_for(lang) and lang in languages.THROWS and lang in languages.GAME_WORDS


def test_every_piper_voice_name_is_well_formed():
    import re
    for lang, v in languages.VOICES.items():
        assert re.fullmatch(rf"{lang}_[A-Z]{{2}}-[\w]+-(x_low|low|medium|high)", v), v


def test_jarvis_answers_in_the_language_of_the_wake_phrase():
    spoken = []
    j = Jarvis(log=lambda *a: None, clock=lambda: 0.0, speak=lambda text, lang: spoken.append((text, lang)))
    j.hear("Oye Jarvis, imítame", wait=True)
    assert j.lang == "es" and j.body.shadow and j.transcript[-1] == ("jarvis", "Te imito como un espejo.")
    j.hear("Hey Jarvis, stop mirroring me", wait=True, lang="en")
    assert j.lang == "en" and not j.body.shadow and j.transcript[-1] == ("jarvis", "Shadow mode off.")
    import time as _t
    for _ in range(50):
        if len(spoken) >= 2:
            break
        _t.sleep(0.02)
    assert spoken[0] == ("Te imito como un espejo.", "es") and spoken[1] == ("Shadow mode off.", "en")


def test_rock_paper_scissors_in_japanese_by_voice():
    import random
    j = Jarvis(log=lambda *a: None, clock=lambda: 0.0)
    j.rng = random.Random(2)
    j.hear("ジャービス、起きて。じゃんけんしよう", wait=True)
    assert j.lang == "ja" and j.game is not None
    j.hear("いいよ", wait=True)                                          # "ready" in Japanese
    assert j.game.state == "count"
    t = 0.0
    while j.game.state != "again":
        if j.game.state == "read" and j.game.said_throw is None:
            j.hear("チョキ", wait=True)                                   # scissors
        j.tick(t, {})
        t += 1 / 30
    assert j.game.result[1] == "scissors"
    said = [txt for who, txt in j.transcript if who == "jarvis"]
    assert said[1:5] == ["最初は...", "グー...", "じゃんけん...", "ぽん！"]
    assert said[-1].endswith("もう一回？")
    j.hear("もういい", wait=True)                                         # "that's enough"
    j.tick(t + 1, {})
    assert j.game is None and j.transcript[-1][1].startswith("いい勝負でした")


def test_other_languages_use_the_model_translation_once():
    calls = []

    def fake_translate(text, lang):
        calls.append(text)
        return {"Mirroring you.": "Ik spiegel je.", "Hello, {name}.": "Hallo, {name}."}.get(text)
    import tempfile
    t = i18n.Translator(fake_translate, path=tempfile.mkdtemp() + "/translations.json")
    assert i18n.tr("Mirroring you.", "nl", t) == "Ik spiegel je."
    assert i18n.tr("Mirroring you.", "nl", t) == "Ik spiegel je." and calls == ["Mirroring you."]   # cached
    assert i18n.tr("Hello, {name}.", "nl", t, name="Shyam") == "Hallo, Shyam."
    assert i18n.Translator(path=t.path).get("Mirroring you.", "nl") == "Ik spiegel je."          # kept on disk
    assert i18n.tr("Relaxing.", "nl") == "Relaxing."                                             # no model: English


def test_ollama_is_told_the_language_and_its_reply_is_not_translated_again(ollama):
    _FakeOllama.reply = {"message": {"role": "assistant", "content": "¡Claro! Te saludo.",
                                     "tool_calls": [{"function": {"name": "wave", "arguments": {}}}]}}
    try:
        j = Jarvis(brain=OllamaBrain(host=ollama), log=lambda *a: None, clock=lambda: 0.0)
        j.hear("Oye Jarvis, salúdame por favor", wait=True)
        assert "Spanish" in _FakeOllama.seen[-1]["messages"][0]["content"]
        assert j.transcript[-1] == ("jarvis", "¡Claro! Te saludo.") and j.body.motion.name == "wave right"
    finally:
        del _FakeOllama.reply


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeInfo:
    def __init__(self, lang, prob):
        self.language, self.language_probability = lang, prob


class _FakeWhisper:
    """Pretends to transcribe: the 'audio' is a dict of what each language setting would hear."""
    def transcribe(self, audio, language=None, **kw):
        text = audio.get(language, audio[None])
        return [_FakeSegment(text)], _FakeInfo(audio["detected"], audio.get("prob", 0.9))


def _listener(got):
    from jarvis.voice import Listener
    lst = Listener.__new__(Listener)
    lst.stt, lst.lang, lst.log = _FakeWhisper(), "en", (lambda *a: None)
    lst.on_text = lambda text, lang: got.append((text, lang))
    lst.follow_up_s = 0.1
    return lst


def test_voice_finds_the_command_and_language():
    got = []
    lst = _listener(got)
    lst._handle({None: "Oye Jarvis, imítame", "detected": "es"}, mic=None)
    lst._handle({None: "nice weather today", "detected": "en"}, mic=None)          # no wake word: ignored
    lst._handle({None: "Despierta Jarvis, saluda", "detected": "pt", "prob": 0.55,
                 "es": "Despierta Jarvis, saluda"}, mic=None)                       # re-heard in Spanish
    assert got == [("imítame", "es"), ("saluda", "es")]


def test_voice_after_a_wake_gesture_takes_the_whole_phrase():
    got = []
    lst = _listener(got)
    lst.lang = "fr"
    lst._handle({None: "salue", "detected": "fr"}, mic=None, forced=True)
    assert got == [("salue", "fr")]


@pytest.mark.parametrize("heard,lang", [
    # what Whisper (small, with the Jarvis hint) wrote for Piper speaking each wake phrase, in testing
    ("o de Jarvis imítame", "es"), ("Hallo Jarvis, mach mich nach.", "de"), ("औरे Jarvis, हाद हिलाओ", "hi"),
    ("Жарvis, проснись по маширукой.", "ru"), ("JARVIS動きて、じゃんけんしよう", "ja"), ("يا جار فيس قلدني", "ar"),
    ("Hey Harvis, wave", "en"), ("じゃあビス、手を振って", "ja"), ("जारवेस, नकल करो", "hi"),
    ("o dejarvis imítame", "es"), ("A Cora de Jarvis me imita.", "pt"),
])
def test_speech_near_misses_still_wake(heard, lang):
    assert languages.wake_language(heard, lang, 0.9) == lang


def test_fuzzy_name_does_not_match_ordinary_words():
    for text in ("it behaves well", "the jars", "Travis Scott", "chavez", "Harvard"):
        assert languages.JARVIS_RE.search(text) is None, text


def test_two_of_three_throws_start_the_game():
    assert RuleBrain().respond("pierre feuillis-zo", lang="fr")[1] == [("shrug", {})]   # one throw: not enough
    assert RuleBrain().respond("pierre feuille ciseaux", lang="fr")[1][0][0] == "play_rock_paper_scissors"
    assert RuleBrain().respond("piedra y papel", lang="es")[1][0][0] == "play_rock_paper_scissors"
    assert RuleBrain().respond("paper paper", lang="en")[1] != [("play_rock_paper_scissors", {})]

"""The brain: turn what the person says into a spoken reply and robot actions.

Two brains, same interface (`respond(text, scene) -> (reply, [(action, args), ...])`):
    OllamaBrain   a local language model through Ollama (http://localhost:11434) with tool calling. Free, private,
                  works offline once the model is downloaded (`ollama pull qwen2.5:7b`).
    RuleBrain     keyword rules, no model at all. Always available; the OllamaBrain falls back to it when Ollama
                  is not running, and it handles "stop" instantly so safety never waits on a model.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import urllib.error
import urllib.request

from .actions import ACTIONS, ollama_tools

STOP_WORDS = re.compile(r"\b(stop|freeze|halt|hold still|don'?t move|emergency)\b", re.I)


NOT_STOP = re.compile(r"\bstop (shadow|copy|copying|mirror|follow)", re.I)


def is_stop(text: str) -> bool:
    """An emergency stop: "stop", "freeze", "halt"... but not "stop copying me", which only ends shadow mode."""
    return bool(STOP_WORDS.search(text or "")) and not NOT_STOP.search(text or "")


class Brain:
    name = "brain"

    def respond(self, text: str, scene: str = "", state: str = "idle") -> tuple[str, list]:
        raise NotImplementedError


# ------------------------------------------------------------------ rules
def _side(t: str) -> dict:
    if re.search(r"\bleft\b", t):
        return {"side": "left"}
    if re.search(r"\bright\b", t):
        return {"side": "right"}
    return {}


def _where(t: str) -> str:
    if re.search(r"\b(me|at me|us|my face)\b", t):
        return "me"
    if re.search(r"\b(that|there|target|this)\b", t):
        return "target"
    for d in ("left", "right", "up", "down"):
        if re.search(rf"\b{d}\b", t):
            return d
    return "forward"


class RuleBrain(Brain):
    """Understands short commands and a few questions about the scene, with no model."""
    name = "rules"

    def respond(self, text: str, scene: str = "", state: str = "idle"):
        t = (text or "").lower().strip()
        if not t:
            return "", []
        if is_stop(t):
            return "Stopping.", [("freeze", {})]
        rules = [
            (NOT_STOP.pattern + r"|\bshadow mode off\b", "Shadow mode off.", [("shadow", {"on": False})]),
            (r"\b(shadow|copy|mirror|follow) me\b|\bshadow mode\b|\bstart shadow", "On it. Copying you.",
             [("shadow", {"on": True})]),
            (r"\b(wave|say (hi|hello|bye|goodbye))\b", "", [("wave", _side(t))]),
            (r"\bthumbs? up\b", "", [("thumbs_up", _side(t))]),
            (r"\bnod\b|\bsay yes\b", "", [("nod", {})]),
            (r"\bshake (your )?head\b|\bsay no\b", "", [("shake_head", {})]),
            (r"\bpoint\b", "", [("point", {"at": _where(t)})]),
            (r"\b(look|turn)\b", "", [("look", {"at": _where(t)})]),
            (r"\b(give|hand) (it|that) (to )?me\b|\breach\b|\btake (this|it)\b", "Here you go.",
             [("reach", _side(t))]),
            (r"\b(let go|release|drop|open (your )?hand)\b", "", [("release", _side(t))]),
            (r"\b(grip|grab|close (your )?hand|squeeze|hold (it|this|that))\b", "",
             [("grip", {**_side(t), "strength": 0.5 if "gent" in t or "soft" in t else 1.0})]),
            (r"\b(celebrate|party|hooray|we did it)\b", "Woo!", [("celebrate", {})]),
            (r"\bshrug\b|\bi don'?t know\b", "", [("shrug", {})]),
            (r"\b(rest|relax|at ease|go home|reset)\b", "Relaxing.", [("rest", {})]),
        ]
        hits = sorted((m.start(), i) for i, (pat, _, _) in enumerate(rules) if (m := re.search(pat, t)))
        if hits:                       # "let go and relax": every command found, in the order it was said
            acts = [a for _, i in hits for a in rules[i][2]]
            return next((rules[i][1] for _, i in hits if rules[i][1]), ""), acts
        if re.search(r"\bwhat (do|can) you see\b|\bwho('?s| is) (there|here)\b", t):
            return scene or "Nothing yet.", []
        if re.search(r"\bwho am i\b", t):
            m = re.search(r"You can see (.+?)\.", scene or "")
            return (f"You're {m.group(1)}." if m else "I can't see you right now."), []
        if re.search(r"\bhow (do i look|am i feeling)\b", t):
            m = re.search(r"expression looks (\w+)", scene or "")
            return (f"You look {m.group(1)}." if m else "You look neutral to me."), []
        if re.search(r"\bwhat am i holding\b", t):
            m = re.search(r"Holding: (.+?)\.", scene or "")
            return (f"{m.group(1)[0].upper()}{m.group(1)[1:]}." if m else "Nothing that I can see."), []
        if re.search(r"\bwhat time\b", t):
            return _dt.datetime.now().strftime("It's %I:%M %p.").replace(" 0", " "), []
        if re.search(r"\bwhat are you doing\b|\bstatus\b", t):
            return f"I'm {state}.", []
        if re.search(r"\b(hello|hi|hey)\b", t):
            return "Hello.", [("wave", {})]
        return "Sorry, I didn't catch a command in that.", [("shrug", {})]


# ------------------------------------------------------------------ ollama
SYSTEM = """You are Jarvis, the mind of Tandem, a humanoid robot upper body (two arms with hands, a turning waist, \
and a head with eyes) built by Shyam. You see through a camera and act by calling the tools.

Rules:
- When asked to move, call the matching tools. You may call several in order.
- Reply in one or two short spoken sentences. Plain words only, no lists, no markdown, no emoji.
- Directions are your own: your left is the person's right when you face them.
- If unsure what they want, ask a short question instead of moving.
- Safety first: if anything sounds unsafe, or they say stop, call freeze.

What you can see right now: {scene}
What your body is doing: {state}"""


class OllamaBrain(Brain):
    """A local model through Ollama. Pick a model that supports tools: qwen2.5:7b (default), llama3.1:8b,
    or qwen2.5:3b / llama3.2:3b on a slower computer."""
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434", history: int = 8,
                 timeout: float = 60.0, fallback: Brain | None = None):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.history_len = history
        self.history: list[dict] = []
        self.fallback = fallback or RuleBrain()
        self.last_error = None

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                tags = json.load(r)
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.last_error = str(e)
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            self.last_error = f"model {self.model} not downloaded (run: ollama pull {self.model})"
            return False
        return True

    def _chat(self, messages: list) -> dict:
        body = json.dumps({"model": self.model, "messages": messages, "tools": ollama_tools(), "stream": False,
                           "options": {"temperature": 0.3}}).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def respond(self, text: str, scene: str = "", state: str = "idle"):
        if is_stop(text):                      # never wait on a model to stop
            return self.fallback.respond(text, scene, state)
        system = {"role": "system", "content": SYSTEM.format(scene=scene or "nothing yet", state=state)}
        messages = [system, *self.history, {"role": "user", "content": text}]
        try:
            out = self._chat(messages)
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.last_error = str(e)
            reply, acts = self.fallback.respond(text, scene, state)
            return reply, acts
        msg = out.get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            name, args = fn.get("name"), fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if name in ACTIONS:
                calls.append((name, args))
        reply = _clean(msg.get("content", ""))
        if not reply and calls:
            reply = "Okay."
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.history = self.history[-2 * self.history_len:]
        return reply, calls


def _clean(s: str) -> str:
    """Strip markdown, emoji and any thinking text a model leaves in, so the reply reads well aloud."""
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S)
    s = re.sub(r"[*_`#>]+", "", s)
    s = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", s)
    return re.sub(r"\s+", " ", s).replace("—", ", ").strip()

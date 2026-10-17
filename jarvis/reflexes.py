"""Reflexes: what Jarvis does on its own when something happens, without asking the language model.

They are fast (the same tick the event arrives) and predictable. Each rule has a cooldown so a held gesture
or a lingering expression does not repeat the reaction. Edit RULES to change Jarvis's personality.

A rule: (event kind, match, actions, speech, cooldown seconds)
    match    dict of event data that must be equal (e.g. {"gesture": "thumbs_up"}), or {}
    actions  list of (action name, args) from actions.ACTIONS; "{side}" in args is the event's hand
    speech   what to say ("{name}" and "{object}" filled from the event), or None
"""
from __future__ import annotations

from .perception import Event

RULES = [
    ("person_arrived", {}, [("look", {"at": "me"}), ("wave", {"side": "right"})], "Hello{name_part}.", 30.0),
    ("person_recognized", {}, [("nod", {})], "Oh, it's you, {name}.", 60.0),
    ("person_left", {}, [("rest", {})], None, 5.0),
    ("gesture", {"gesture": "open_palm"}, [("wave", {"side": "mirror"})], None, 6.0),
    ("gesture", {"gesture": "thumbs_up"}, [("thumbs_up", {"side": "mirror"})], None, 6.0),
    ("gesture", {"gesture": "thumbs_down"}, [("shake_head", {})], "Not good? Tell me what to change.", 10.0),
    ("gesture", {"gesture": "i_love_you"}, [("celebrate", {})], None, 10.0),
    ("gesture", {"gesture": "call_me"}, [("look", {"at": "me"}), ("nod", {})], "Yes? I'm listening.", 4.0),
    ("emotion", {"emotion": "sad"}, [("look", {"at": "me"})], "You look a little down. Anything I can do?", 120.0),
    ("emotion", {"emotion": "fearful"}, [("rest", {})], "I'll hold still.", 30.0),
    ("emotion", {"emotion": "surprised"}, [("nod", {})], None, 30.0),
    ("grasp", {}, [("look", {"at": "me"})], "Nice {object}.", 20.0),
    ("target_locked", {}, [("look", {"at": "target"}), ("point", {"at": "target"})], None, 3.0),
]

# Gestures that start listening for a command, like saying the wake word
WAKE_GESTURES = {"call_me"}


class Reflexes:
    def __init__(self, rules=None):
        self.rules = list(RULES if rules is None else rules)
        self._last: dict[int, float] = {}

    def react(self, event: Event, now: float):
        """The actions and speech for an event: (list of (name, args), speech or None)."""
        for i, (kind, match, acts, speech, cooldown) in enumerate(self.rules):
            if kind != event.kind or any(event.data.get(k) != v for k, v in match.items()):
                continue
            if now - self._last.get(i, -1e9) < cooldown:
                return [], None
            self._last[i] = now
            side = event.data.get("side", "right")
            mirror = "left" if side == "right" else "right"   # facing you, your right hand is across from its left
            out = []
            for name, args in acts:
                a = {k: (mirror if v == "mirror" else side if v == "{side}" else v) for k, v in args.items()}
                out.append((name, a))
            text = None
            if speech:
                name = event.data.get("name") or ""
                data = {"name": name, "object": event.data.get("object", "").replace("_", " "),
                        "name_part": f", {name}" if name and name != "unknown" else ""}
                text = speech.format(**data)
            return out, text
        return [], None

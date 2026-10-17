"""The actions Jarvis can take, shared by the reflexes, the rule-based parser and the language model's tools.

Directions are from the robot's own point of view: "left" is the robot's left (your right when you face it).
"me" is the person's head, "target" the gaze target they locked in Tandem.
"""
from __future__ import annotations

from . import skills

DIRECTIONS = {"forward": (0.0, 0.0), "ahead": (0.0, 0.0), "left": (45.0, 0.0), "right": (-45.0, 0.0),
              "up": (0.0, 30.0), "down": (0.0, -25.0)}
SIDE = {"type": "string", "enum": ["right", "left"], "description": "which arm or hand (the robot's own)"}
WHERE = {"type": "string", "enum": ["me", "target", "forward", "left", "right", "up", "down"],
         "description": "me = the person, target = their locked gaze target, else a direction"}

# name -> (description, JSON schema properties, required)
ACTIONS = {
    "wave": ("Wave hello or goodbye with one arm.", {"side": SIDE}, []),
    "nod": ("Nod yes.", {}, []),
    "shake_head": ("Shake the head no.", {}, []),
    "thumbs_up": ("Give a thumbs up.", {"side": SIDE}, []),
    "point": ("Point at something.", {"at": WHERE}, ["at"]),
    "look": ("Turn the head and eyes toward something and keep looking.", {"at": WHERE}, ["at"]),
    "reach": ("Hold an open hand out, palm up, to take or offer something.", {"side": SIDE}, []),
    "grip": ("Close a hand to grip. strength 0 (open) to 1 (full grip).",
             {"side": SIDE, "strength": {"type": "number", "minimum": 0, "maximum": 1}}, []),
    "release": ("Open a hand (or both) to let go.", {"side": SIDE}, []),
    "celebrate": ("Throw both arms up in celebration.", {}, []),
    "shrug": ("Shrug, for 'I don't know'.", {}, []),
    "rest": ("Go back to the relaxed rest pose and stop looking at anything.", {}, []),
    "freeze": ("Stop all motion immediately and hold still.", {}, []),
    "shadow": ("Start or stop copying the person's movements (shadow mode).",
               {"on": {"type": "boolean"}}, ["on"]),
}


def ollama_tools() -> list:
    """The actions in the tool format Ollama's /api/chat expects."""
    return [{"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req}}}
        for name, (desc, props, req) in ACTIONS.items()]


def resolve(where: str, perception) -> tuple | None:
    where = (where or "forward").lower()
    if where in ("me", "person", "you", "user"):
        return perception.head_direction() if perception else None
    if where == "target":
        return perception.target_direction() if perception else None
    return DIRECTIONS.get(where)


def apply(body: skills.Body, name: str, args: dict, now: float, perception=None, queue: bool = False) -> str:
    """Carry out one action on the body. Returns a short description, or raises ValueError.
    With queue, a skill waits for the one in progress instead of replacing it."""
    args = args or {}
    side = args.get("side") if args.get("side") in ("left", "right") else "right"
    if name in ("wave", "thumbs_up", "reach"):
        body.do(skills.SKILLS[name](side), now, queue=queue)
        return f"{name.replace('_', ' ')} ({side})"
    if name in ("nod", "shake_head", "celebrate", "shrug"):
        body.do(skills.SKILLS[name](), now, queue=queue)
        return name.replace("_", " ")
    if name in ("point", "look"):
        where = args.get("at", "forward")
        d = resolve(where, perception)
        if d is None:
            raise ValueError(f"cannot see '{where}' to {name} at")
        if name == "point":
            body.do(skills.point(*d), now, queue=queue)
        else:
            body.look("person" if where in ("me", "person", "you", "user") else d)
        return f"{name} at {where}"
    if name == "grip":
        s = float(args.get("strength", 1.0))
        body.grip(side, s)
        return f"grip {side} {s:.0%}"
    if name == "release":
        body.release(args.get("side") if args.get("side") in ("left", "right") else None)
        return "release"
    if name == "rest":
        body.rest()
        return "rest"
    if name == "freeze":
        body.freeze()
        return "freeze"
    if name == "shadow":
        on = args.get("on", True)
        on = on if isinstance(on, bool) else str(on).lower() in ("1", "true", "yes", "on")
        body.set_shadow(on)
        return "shadow on" if on else "shadow off"
    raise ValueError(f"unknown action '{name}'")

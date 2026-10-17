"""Jarvis: a locally hosted brain for the Tandem robot.

Jarvis watches what Tandem tracks (who is there, their gestures, expression, gaze and what they hold), listens
for spoken or typed commands, and drives the robot through a library of named skills. Everything runs on your
own computer: the language model through Ollama, speech through openWakeWord, faster-whisper and Piper.

Layers, from fastest to slowest:
    perception.Perception   tracker values -> events (person arrived, gesture, emotion, grasp, target)
    reflexes.Reflexes       events -> immediate skills, no language model (wave back, look at you)
    brain.Brain             words -> skills and a reply (a local LLM with tool calling, or a rule-based parser)
    skills.Body             skills -> joint targets for every robot joint, speed-limited for the hardware
    robot.*                 joint targets -> the MuJoCo robot now, a serial microcontroller later
"""
from .brain import Brain, RuleBrain, OllamaBrain
from .core import Jarvis
from .perception import Event, Perception
from .reflexes import Reflexes
from .skills import Body

__all__ = ["Body", "Brain", "Event", "Jarvis", "OllamaBrain", "Perception", "Reflexes", "RuleBrain"]

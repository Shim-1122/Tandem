"""Hearing and speaking in many languages, all on your own computer.

    Listener   waits for "Jarvis" in any language, works out which language you used, and passes on the command.
               Two wake engines:
                 whisper (default)  every phrase you say is transcribed by Whisper's multilingual model, which
                                    finds "Jarvis" in any language and script ("Hey Jarvis", "Despierta Jarvis",
                                    "ジャービス、起きて", "Джарвис, проснись") and tells the language apart
                 oww                openWakeWord's "Hey Jarvis" model: lighter on the processor, but trained on
                                    English, so other wake phrases may be missed (the command can still be in any
                                    language)
    Speaker    Piper voices, one per language, downloaded the first time each language is spoken
    Keyboard   type commands in the terminal instead (always available)

The language comes from the wake phrase (languages.wake_language), and Jarvis answers in it.
pip install -r requirements-jarvis.txt
"""
from __future__ import annotations

import collections
import sys
import threading
import time
from pathlib import Path

from .languages import find_wake, voice_for, wake_language

RATE = 16000
CHUNK = 1280                       # 80 ms, the frame size openWakeWord expects
PREROLL_S = 1.2                    # audio kept from before speech is noticed (so no word is cut off)


def strip_wake(text: str) -> str:
    """'Hey Jarvis, shadow me.' -> 'shadow me.'; text without a wake phrase is returned as it is."""
    found = find_wake(text)
    return found[0] if found is not None else (text or "").strip()


class Keyboard(threading.Thread):
    def __init__(self, on_text, prompt: str = ""):
        super().__init__(daemon=True)
        self.on_text, self.prompt = on_text, prompt

    def run(self):
        for line in sys.stdin:
            if line.strip():
                self.on_text(line.strip())


class Speaker:
    """Speaks with a Piper voice in the language asked for; voices are downloaded to `voice_dir` on first use.
    `fixed_voice` (a .onnx file) is used for `fixed_lang` instead of the default voice for it."""

    def __init__(self, voice_dir: str | Path = "models/piper", device=None, fixed_voice: str | None = None,
                 fixed_lang: str = "en", download: bool = True, log=print):
        import numpy as np
        import sounddevice as sd
        self.np, self.sd = np, sd
        self.dir, self.device, self.download, self.log = Path(voice_dir), device, download, log
        self.voices = {}
        self.fixed = (fixed_lang, fixed_voice) if fixed_voice else None
        self.speaking = threading.Event()

    def voice(self, lang: str):
        if lang in self.voices:
            return self.voices[lang]
        from piper import PiperVoice
        path = None
        if self.fixed and self.fixed[0] == lang and Path(self.fixed[1]).exists():
            path = Path(self.fixed[1])
        else:
            name = voice_for(lang)
            if name:
                path = self.dir / f"{name}.onnx"
                if not path.exists() and self.download:
                    try:
                        from piper.download_voices import download_voice
                        self.log(f"[voice] downloading the {lang} voice {name} (once) ...")
                        self.dir.mkdir(parents=True, exist_ok=True)
                        download_voice(name, self.dir)
                    except Exception as e:
                        self.log(f"[voice] could not download {name}: {e}")
        v = PiperVoice.load(str(path)) if path is not None and path.exists() else None
        if v is None:
            self.log(f"[voice] no voice for '{lang}': replies in that language are shown, not spoken")
        self.voices[lang] = v
        return v

    def speak(self, text: str, lang: str = "en"):
        v = self.voice(lang)
        if v is None:
            return
        np = self.np
        self.speaking.set()
        try:
            if hasattr(v, "synthesize_wav"):                          # piper-tts 1.3 and later
                audio = [c.audio_int16_array for c in v.synthesize(text)]
            else:                                                      # piper-tts 1.2
                audio = [np.frombuffer(b, dtype=np.int16) for b in v.synthesize_stream_raw(text)]
            if audio:
                self.sd.play(np.concatenate(audio), v.config.sample_rate, device=self.device)
                self.sd.wait()
        finally:
            time.sleep(0.2)                                            # let the room go quiet before listening
            self.speaking.clear()


class Listener(threading.Thread):
    """Calls `on_text(command, lang)` for each command said after "Jarvis"."""

    def __init__(self, on_text, engine: str = "whisper", whisper_model: str = "small", device=None,
                 speaking: threading.Event | None = None, threshold: float = 0.5, silence_s: float = 0.8,
                 max_s: float = 10.0, follow_up_s: float = 5.0, log=print):
        super().__init__(daemon=True)
        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel
        self.np, self.sd = np, sd
        self.on_text, self.log, self.engine = on_text, log, engine
        self.threshold, self.silence_s, self.max_s, self.follow_up_s = threshold, silence_s, max_s, follow_up_s
        self.device = device
        self.speaking = speaking or threading.Event()
        self.lang = "en"
        self.stt = WhisperModel(whisper_model, device="auto", compute_type="int8")
        self.wake = None
        if engine == "oww":
            from openwakeword.model import Model
            try:
                self.wake = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            except Exception:
                import openwakeword.utils
                openwakeword.utils.download_models()                   # first run: fetch the wake word models
                self.wake = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        self._wake_now = threading.Event()
        self.stopped = threading.Event()
        self._noise = 300.0

    def wake_up(self):
        """Listen for a command without the wake word (for example after the call-me gesture)."""
        self._wake_now.set()

    # ------------------------------------------------------------------ audio
    def _level(self, block) -> float:
        return float(self.np.sqrt(self.np.mean(block.astype(self.np.float32) ** 2)))

    def _phrase(self, mic, preroll=None, wait_s: float | None = None):
        """Record one phrase: from when speech starts (with the audio just before it) until a pause.
        Returns float audio, or None if nobody spoke within `wait_s`."""
        np = self.np
        before = collections.deque(preroll or [], maxlen=int(PREROLL_S * RATE / CHUNK))
        frames, heard, quiet = [], False, 0.0
        t0 = time.monotonic()
        while not self.stopped.is_set():
            block, _ = mic.read(CHUNK)
            x = block[:, 0].copy()
            if self.speaking.is_set():
                before.clear()
                continue
            level = self._level(x)
            loud = level > max(300.0, 3.0 * self._noise)
            if not heard:
                if not loud:
                    self._noise = 0.95 * self._noise + 0.05 * level
                before.append(x)
                if loud:
                    heard, frames, t0 = True, list(before), time.monotonic()
                elif wait_s is not None and time.monotonic() - t0 > wait_s:
                    return None
                continue
            frames.append(x)
            quiet = 0.0 if loud else quiet + CHUNK / RATE
            if quiet >= self.silence_s or time.monotonic() - t0 > self.max_s:
                break
        return np.concatenate(frames).astype(np.float32) / 32768.0 if frames else None

    def _transcribe(self, audio, lang=None):
        # hotwords nudges Whisper to spell the name "Jarvis" whatever the language (in testing it turned
        # "Hallo Jahres" into "Hallo Jarvis" and "じゃあビス" into "Jarvis")
        segments, info = self.stt.transcribe(audio, language=lang, beam_size=1, vad_filter=True, hotwords="Jarvis")
        return " ".join(s.text for s in segments).strip(), info.language, info.language_probability

    # ------------------------------------------------------------------ handling a phrase
    def _handle(self, audio, mic, forced: bool = False):
        text, wl, wp = self._transcribe(audio)
        if forced and find_wake(text, strict=False) is None:          # already woken (gesture, or openWakeWord heard it):
            lang = wl if wp >= 0.5 else self.lang       # everything said is the command
            command = text if lang == wl else self._transcribe(audio, lang)[0]
        else:
            lang = wake_language(text, wl, wp, previous=self.lang)
            if lang is None:
                return                                   # no "Jarvis" at the start: not for us
            if lang != wl:                               # the wake phrase settled it: hear it in that language
                again = self._transcribe(audio, lang)[0]
                text = again if find_wake(again, strict=False) is not None else text
            command = (find_wake(text, strict=False) or ("", None))[0]
        self.lang = lang
        if not command:                                  # just "Hey Jarvis": the command comes next
            self.log(f"[voice] listening ({lang})...")
            follow = self._phrase(mic, wait_s=self.follow_up_s)
            if follow is None:
                return
            command = self._transcribe(follow, lang)[0]
        if command:
            self.on_text(command, lang)

    # ------------------------------------------------------------------ wake engines
    def run(self):
        with self.sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK,
                                 device=self.device) as mic:
            self.log("[voice] say \"Hey Jarvis\" (or Jarvis in your language) and a command")
            if self.engine == "oww":
                self._run_oww(mic)
                return
            while not self.stopped.is_set():
                audio = self._phrase(mic)
                if audio is not None:
                    forced = self._wake_now.is_set()
                    self._wake_now.clear()
                    self._handle(audio, mic, forced=forced)

    def _run_oww(self, mic):
        np = self.np
        preroll = collections.deque(maxlen=int(PREROLL_S * RATE / CHUNK))
        while not self.stopped.is_set():
            block, _ = mic.read(CHUNK)
            if self.speaking.is_set():
                preroll.clear()
                continue
            preroll.append(block[:, 0].copy())
            scores = self.wake.predict(block[:, 0])
            if self._wake_now.is_set() or max(scores.values()) >= self.threshold:
                self._wake_now.clear()
                self.wake.reset()
                # The wake phrase is in the pre-roll; the command usually follows in the same breath.
                rest = self._phrase(mic, preroll=list(preroll), wait_s=1.0)
                if rest is None:                         # "Hey Jarvis" and a pause: _handle waits for the command
                    rest = np.concatenate(list(preroll)).astype(np.float32) / 32768.0
                preroll.clear()
                self._handle(rest, mic, forced=True)

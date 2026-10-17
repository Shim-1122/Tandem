"""Hearing and speaking, all on your own computer.

    LocalVoice   "Hey Jarvis" wake word (openWakeWord) -> record until you stop talking -> faster-whisper
                 speech to text; replies spoken with a Piper voice. pip install -r requirements-jarvis.txt
    Keyboard     type commands in the terminal instead (always available, also handy next to LocalVoice)

Both call `on_text(text)` from their own thread.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

RATE = 16000
CHUNK = 1280                       # 80 ms, the frame size openWakeWord expects


class Keyboard(threading.Thread):
    def __init__(self, on_text, prompt: str = ""):
        super().__init__(daemon=True)
        self.on_text, self.prompt = on_text, prompt

    def run(self):
        for line in sys.stdin:
            if line.strip():
                self.on_text(line.strip())


class LocalVoice(threading.Thread):
    def __init__(self, on_text, wake_word: str = "hey_jarvis", threshold: float = 0.5,
                 whisper_model: str = "base.en", voice_model: str | None = None, device=None,
                 silence_s: float = 0.8, max_s: float = 10.0, log=print):
        super().__init__(daemon=True)
        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel
        from openwakeword.model import Model
        self.np, self.sd = np, sd
        self.on_text, self.log = on_text, log
        self.threshold, self.silence_s, self.max_s = threshold, silence_s, max_s
        self.device = device
        try:
            self.wake = Model(wakeword_models=[wake_word], inference_framework="onnx")
        except Exception:
            import openwakeword.utils
            openwakeword.utils.download_models()     # first run: fetch the wake word models
            self.wake = Model(wakeword_models=[wake_word], inference_framework="onnx")
        self.stt = WhisperModel(whisper_model, device="auto", compute_type="int8")
        self.tts = None
        if voice_model:
            from piper import PiperVoice
            self.tts = PiperVoice.load(str(Path(voice_model)))
        self.speaking = threading.Event()
        self._wake_now = threading.Event()
        self.stopped = threading.Event()

    # ------------------------------------------------------------------ speaking
    def speak(self, text: str):
        if self.tts is None:
            return
        np = self.np
        self.speaking.set()
        try:
            audio = []
            if hasattr(self.tts, "synthesize_wav"):            # piper-tts 1.3 and later
                for chunk in self.tts.synthesize(text):
                    audio.append(chunk.audio_int16_array)
                rate = self.tts.config.sample_rate
            else:                                                # piper-tts 1.2
                audio = [np.frombuffer(b, dtype=np.int16) for b in self.tts.synthesize_stream_raw(text)]
                rate = self.tts.config.sample_rate
            if audio:
                self.sd.play(np.concatenate(audio), rate, device=self.device)
                self.sd.wait()
        finally:
            time.sleep(0.2)                                       # let the room go quiet before listening again
            self.speaking.clear()

    def wake_up(self):
        """Start listening without the wake word (for example after the call-me gesture)."""
        self._wake_now.set()

    # ------------------------------------------------------------------ listening
    def run(self):
        np = self.np
        with self.sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK,
                                 device=self.device) as mic:
            self.log("[voice] say \"Hey Jarvis\"")
            while not self.stopped.is_set():
                block, _ = mic.read(CHUNK)
                if self.speaking.is_set():
                    continue
                scores = self.wake.predict(block[:, 0])
                if self._wake_now.is_set() or max(scores.values()) >= self.threshold:
                    self._wake_now.clear()
                    self.wake.reset()
                    self.log("[voice] listening...")
                    text = self._transcribe(self._record(mic))
                    if text:
                        self.on_text(text)

    def _record(self, mic):
        """Record until `silence_s` of quiet after speech, or `max_s`."""
        np = self.np
        frames, quiet, heard, noise = [], 0.0, False, None
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.max_s:
            block, _ = mic.read(CHUNK)
            frames.append(block[:, 0].copy())
            level = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
            if not heard:                                         # background level before you start talking
                noise = level if noise is None else 0.9 * noise + 0.1 * level
            if level > max(400.0, 3 * (noise or 0)):
                heard, quiet = True, 0.0
            elif heard:
                quiet += CHUNK / RATE
                if quiet >= self.silence_s:
                    break
            elif time.monotonic() - t0 > 4.0:
                break                                             # woke but nobody spoke
        return np.concatenate(frames).astype(np.float32) / 32768.0

    def _transcribe(self, audio) -> str:
        segments, _ = self.stt.transcribe(audio, language="en", beam_size=1, vad_filter=True)
        return " ".join(s.text for s in segments).strip()

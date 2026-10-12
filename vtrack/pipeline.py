"""Low-latency capture and inference pipeline.

Latency comes from three places, and each is handled here:
  1. Camera buffering: webcams queue frames, so a slow loop reads old ones. A dedicated thread reads the
     camera continuously and keeps only the newest frame.
  2. Running models one after another: each model (hands, body, face, emotion) gets its own worker
     thread that always takes the newest frame. MediaPipe releases Python's lock while it runs, so the
     models run in parallel on separate cores, and hand tracking updates as fast as the hand model can
     go, never waiting for the slower body or face models.
  3. Low camera frame rate: the camera is asked for 60 fps (MJPG on Windows and Linux, which many
     webcams need for 60 fps). The rate actually delivered is shown on screen.
  4. CPU contention: the hand and body models have priority. The face and emotion models (BackgroundGate)
     use only the processor cores the priority models leave free, and on a computer without spare cores
     they still run at a guaranteed minimum rate. On a computer with enough cores everything runs at
     full speed; on a small one the hands stay smooth and the face updates less often.
Each worker records its update rate and its latency (capture to result).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    index: int
    t: float              # capture time, time.monotonic() seconds
    bgr: np.ndarray
    rgb: np.ndarray


class RateMeter:
    """Events per second and mean latency over the last `window_s` seconds."""

    def __init__(self, window_s: float = 1.0):
        self.window_s = window_s
        self._ev: deque = deque()
        self._lock = threading.Lock()

    def add(self, t_done: float, latency_s: float = 0.0):
        with self._lock:
            self._ev.append((t_done, latency_s))
            while self._ev and t_done - self._ev[0][0] > self.window_s:
                self._ev.popleft()

    def rate(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        with self._lock:
            ev = [e for e in self._ev if now - e[0] <= self.window_s]
        if len(ev) < 2:
            return 0.0
        span = ev[-1][0] - ev[0][0]
        return (len(ev) - 1) / span if span > 0 else 0.0

    def latency_ms(self) -> float:
        with self._lock:
            return 1000.0 * float(np.mean([e[1] for e in self._ev])) if self._ev else float("nan")


def open_camera(index: int = 0, width: int = 1280, height: int = 720, fps: int = 60):
    """Open a webcam with low-latency settings; returns (capture, (actual width, height, fps))."""
    import cv2
    backends = {"win32": [cv2.CAP_DSHOW, cv2.CAP_MSMF], "darwin": [cv2.CAP_AVFOUNDATION]}.get(sys.platform, [cv2.CAP_V4L2])
    cap = None
    for be in backends + [cv2.CAP_ANY]:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            break
    if cap is None or not cap.isOpened():
        return None, None
    if sys.platform != "darwin":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS))
    return cap, actual


class CameraReader(threading.Thread):
    """Reads frames continuously and keeps only the newest one."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.cond = threading.Condition()
        self.latest: Frame | None = None
        self.meter = RateMeter()
        self.stopped = threading.Event()
        self.failed = False

    def run(self):
        import cv2
        index = 0
        misses = 0
        while not self.stopped.is_set():
            ok, bgr = self.cap.read()
            t = time.monotonic()
            if not ok or bgr is None:
                misses += 1
                if misses > 50:
                    self.failed = True
                    break
                time.sleep(0.01)
                continue
            misses = 0
            index += 1
            frame = Frame(index, t, bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            with self.cond:
                self.latest = frame
                self.cond.notify_all()
            self.meter.add(t)
        with self.cond:
            self.cond.notify_all()

    def newest(self) -> Frame | None:
        with self.cond:
            return self.latest

    def wait_newer(self, after_index: int, timeout: float = 1.0) -> Frame | None:
        with self.cond:
            self.cond.wait_for(lambda: (self.latest is not None and self.latest.index > after_index)
                               or self.stopped.is_set() or self.failed, timeout=timeout)
            f = self.latest
        return f if f is not None and f.index > after_index else None

    def stop(self):
        self.stopped.set()


class BackgroundGate:
    """CPU cores left for low-priority workers once the priority workers each have one."""

    def __init__(self, priority_workers: int, cores: int | None = None):
        cores = cores if cores is not None else (os.cpu_count() or 2)
        self.free = max(0, int(cores) - int(priority_workers))
        self._sem = threading.Semaphore(self.free) if self.free > 0 else None

    def acquire(self, timeout: float, stopped: threading.Event) -> bool:
        """Wait up to `timeout` s for a free core; False when none came free (the caller runs anyway)."""
        if self._sem is None:
            stopped.wait(max(0.0, timeout))
            return False
        return self._sem.acquire(timeout=max(0.0, timeout))

    def release(self):
        self._sem.release()


@dataclass
class Result:
    value: object
    frame_index: int
    t_capture: float
    t_done: float


class ModelWorker(threading.Thread):
    """Runs fn(frame) on the newest frame, as often as it can (or at most `max_hz` times a second).

    With a `gate` the worker is low priority: it runs when a core is free, and otherwise waits, but never
    longer than it takes to keep `min_hz`. fn is called only from this thread, so a MediaPipe task created
    for this worker is used safely.
    """

    def __init__(self, name: str, camera: CameraReader, fn, max_hz: float | None = None,
                 gate: BackgroundGate | None = None, min_hz: float = 5.0):
        super().__init__(daemon=True, name=name)
        self.camera, self.fn, self.max_hz = camera, fn, max_hz
        self.gate, self.min_hz = gate, min_hz
        self.latest: Result | None = None
        self.meter = RateMeter()
        self.error: BaseException | None = None
        self.stopped = threading.Event()
        self._lock = threading.Lock()

    def run(self):
        last, t_start = 0, -1e9
        while not self.stopped.is_set():
            frame = self.camera.wait_newer(last, timeout=0.5)
            if frame is None:
                if self.camera.failed:
                    break
                continue
            got = False
            if self.gate is not None:
                got = self.gate.acquire(t_start + 1.0 / self.min_hz - time.monotonic(), self.stopped)
                if self.stopped.is_set():
                    if got:
                        self.gate.release()
                    break
                frame = self.camera.newest() or frame     # the frame may have aged while waiting
            last = frame.index
            t0 = t_start = time.monotonic()
            try:
                value = self.fn(frame)
            except Exception as exc:                       # report on screen instead of dying silently
                self.error = exc
                time.sleep(0.2)
                continue
            finally:
                if got:
                    self.gate.release()
            done = time.monotonic()
            self.error = None                              # recovered (for example after a failed download)
            with self._lock:
                self.latest = Result(value, frame.index, frame.t, done)
            self.meter.add(done, done - frame.t)
            if self.max_hz:
                rest = 1.0 / self.max_hz - (time.monotonic() - t0)
                if rest > 0:
                    self.stopped.wait(rest)

    def get(self) -> Result | None:
        with self._lock:
            return self.latest

    def stop(self):
        self.stopped.set()


def stop_all(threads, timeout: float = 2.0):
    for th in threads:
        th.stop()
    for th in threads:
        if th.ident is not None:                  # joining a thread that never started is an error
            th.join(timeout)

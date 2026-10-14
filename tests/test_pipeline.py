import threading
import time

import numpy as np
import pytest

pytest.importorskip("cv2")

from vtrack.pipeline import BackgroundGate, CameraReader, FrameClaim, ModelWorker, RateMeter, WorkerGroup, stop_all


class FakeCap:
    def __init__(self, fps=100.0, fail_after=None):
        self.dt, self.n, self.fail_after = 1.0 / fps, 0, fail_after

    def read(self):
        time.sleep(self.dt)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        return True, np.full((48, 64, 3), self.n % 255, np.uint8)


def test_slow_worker_always_takes_the_newest_frame():
    cam = CameraReader(FakeCap(fps=100))
    seen = []

    def slow(frame):
        seen.append(frame.index)
        time.sleep(0.03)                       # 3 camera frames per call
        return frame.index

    w = ModelWorker("slow", cam, slow)
    cam.start(); w.start()
    time.sleep(0.6)
    stop_all([w, cam])
    gaps = np.diff(seen)
    assert len(seen) >= 8
    assert np.median(gaps) >= 2                # skipped stale frames instead of queueing them
    assert w.meter.latency_ms() < 80           # capture-to-result stays near one processing time


def test_fast_worker_keeps_up_with_the_camera_and_reports_rates():
    cam = CameraReader(FakeCap(fps=60))
    w = ModelWorker("fast", cam, lambda f: f.index)
    cam.start(); w.start()
    time.sleep(1.2)
    rate_cam, rate_w = cam.meter.rate(), w.meter.rate()
    res = w.get()
    stop_all([w, cam])
    assert 40 < rate_cam < 70 and 40 < rate_w < 70
    assert res is not None and res.value == res.frame_index and res.t_done >= res.t_capture


def test_max_hz_limits_a_worker():
    cam = CameraReader(FakeCap(fps=100))
    w = ModelWorker("limited", cam, lambda f: 1, max_hz=10)
    cam.start(); w.start()
    time.sleep(1.2)
    rate = w.meter.rate()
    stop_all([w, cam])
    assert 6 < rate < 12


def test_worker_errors_are_kept_not_raised():
    cam = CameraReader(FakeCap(fps=100))

    def bad(frame):
        raise ValueError("model failed")

    w = ModelWorker("bad", cam, bad)
    cam.start(); w.start()
    time.sleep(0.3)
    stop_all([w, cam])
    assert isinstance(w.error, ValueError) and w.get() is None


def test_dead_camera_is_detected_and_threads_exit():
    cam = CameraReader(FakeCap(fps=200, fail_after=5))
    w = ModelWorker("w", cam, lambda f: 1)
    cam.start(); w.start()
    cam.join(3)
    w.join(3)
    assert cam.failed and not cam.is_alive() and not w.is_alive()


def test_rate_meter():
    m = RateMeter(window_s=1.0)
    for i in range(31):
        m.add(i / 30, 0.02)
    assert m.rate(now=1.0) == pytest.approx(30, rel=0.05)
    assert m.latency_ms() == pytest.approx(20)


def test_background_gate_counts_the_cores_left_free():
    assert BackgroundGate(2, cores=2).free == 0
    assert BackgroundGate(2, cores=8).free == 6
    assert BackgroundGate(3, cores=1).free == 0
    gate, stopped = BackgroundGate(2, cores=2), threading.Event()
    t0 = time.monotonic()
    assert gate.acquire(0.1, stopped) is False            # no free core: waits out the timeout
    assert time.monotonic() - t0 == pytest.approx(0.1, abs=0.05)


def test_low_priority_worker_without_free_cores_keeps_its_minimum_rate():
    cam = CameraReader(FakeCap(fps=200))
    w = ModelWorker("face", cam, lambda f: f.index, gate=BackgroundGate(2, cores=2), min_hz=10)
    cam.start(); w.start()
    time.sleep(1.2)
    rate = w.meter.rate()
    stop_all([w, cam])
    assert 7 < rate < 13


def test_low_priority_worker_with_free_cores_runs_freely_and_releases_its_core():
    cam = CameraReader(FakeCap(fps=200))
    gate = BackgroundGate(1, cores=2)                        # one free core
    w = ModelWorker("face", cam, lambda f: f.index, gate=gate, min_hz=5)
    cam.start(); w.start()
    time.sleep(1.0)
    rate = w.meter.rate()
    stop_all([w, cam])
    assert rate > 30                                        # far above the 5 Hz floor
    assert gate._sem.acquire(timeout=0.5)                  # the core was given back


def test_low_priority_worker_releases_its_core_when_the_model_fails():
    cam = CameraReader(FakeCap(fps=200))
    gate = BackgroundGate(1, cores=2)

    def bad(frame):
        raise RuntimeError("no model")

    w = ModelWorker("face", cam, bad, gate=gate)
    cam.start(); w.start()
    time.sleep(0.5)
    stop_all([w, cam])
    assert isinstance(w.error, RuntimeError)
    assert gate._sem.acquire(timeout=0.5)


def test_worker_error_clears_after_recovery():
    cam = CameraReader(FakeCap(fps=100))
    calls = []

    def flaky(frame):
        calls.append(1)
        if len(calls) < 3:
            raise OSError("download failed")
        return 1

    w = ModelWorker("w", cam, flaky)
    cam.start(); w.start()
    time.sleep(1.0)
    stop_all([w, cam])
    assert w.error is None and w.get().value == 1


def test_stop_all_accepts_threads_that_never_started():
    cam = CameraReader(FakeCap(fps=100))
    stop_all([ModelWorker("w", cam, lambda f: 1), cam])     # must not raise


def _rate_with(count, seconds=1.2):
    """A 25 ms model (40 Hz alone) on a 200 fps camera, run as `count` instances."""
    cam = CameraReader(FakeCap(fps=200))
    seen, lock = [], threading.Lock()

    def model(frame):
        time.sleep(0.025)                      # sleeping, like MediaPipe, lets the other instances run
        with lock:
            seen.append(frame.index)
        return frame.index

    g = WorkerGroup("hand", cam, [model] * count)
    cam.start(); g.start()
    time.sleep(seconds)
    rate, results = g.meter.rate(), g.new_results(0)
    stop_all([g, cam])
    return rate, seen, results


def test_a_worker_group_multiplies_the_rate_and_never_repeats_a_frame():
    r1, _, _ = _rate_with(1)
    r3, seen, results = _rate_with(3)
    assert 30 < r1 < 45 and r3 > 2.4 * r1                 # three instances: about three times the results
    assert len(seen) == len(set(seen))                     # each frame went to exactly one instance
    assert [r.frame_index for r in results] == sorted(r.frame_index for r in results)
    assert len(results) <= 3


def test_frame_claim_takes_each_frame_once():
    c = FrameClaim()
    assert c.take(1) and not c.take(1) and c.take(3) and not c.take(2) and c.last == 3

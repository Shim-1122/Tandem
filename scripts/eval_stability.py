"""Measure jitter and lag of the arm pipeline on simulated noisy landmarks (no camera needed).

    python scripts/eval_stability.py

Landmark noise model (an assumption, not measured): pose world landmarks get independent Gaussian
noise of `--xy-cm` along the image axes and `--z-cm` along depth per frame; hand landmarks get `--hand-cm`.
Reports, per joint angle, the standard deviation while holding still (jitter) and the RMS error while
moving (which includes lag), for raw angles and for the smoothed pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vtrack import ArmKinematics, LandmarkSmoother
from vtrack.filters import AngleSmoother
from vtrack.synthetic import forward_arm

KEYS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion",
        "forearm_rotation", "wrist_flexion", "wrist_deviation"]
CAM = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], float)
BASE = dict(shoulder_flexion=30.0, shoulder_abduction=20.0, shoulder_rotation=20.0, elbow_flexion=80.0,
            forearm_rotation=10.0, wrist_flexion=10.0, wrist_deviation=5.0)


def run(pipeline, motion, args, seed=0, fps=30.0, seconds=6.0):
    rng = np.random.default_rng(seed)
    pose_noise = np.array([args.xy_cm, args.xy_cm, args.z_cm]) / 100
    errs = {k: [] for k in KEYS}
    outs = {k: [] for k in KEYS}
    kin = ArmKinematics("right")
    ps = LandmarkSmoother(**pipeline["pose"]) if pipeline["pose"] else None
    hs = LandmarkSmoother(**pipeline["hand"]) if pipeline["hand"] else None
    ang_f = AngleSmoother(**pipeline["angles"]) if pipeline["angles"] else None
    for i in range(int(fps * seconds)):
        t = i / fps
        truth = dict(BASE)
        if motion:
            truth["elbow_flexion"] = 80 + 40 * np.sin(2 * np.pi * 0.5 * t)
            truth["shoulder_flexion"] = 30 + 25 * np.sin(2 * np.pi * 0.3 * t)
            truth["forearm_rotation"] = 10 + 40 * np.sin(2 * np.pi * 0.4 * t)
        pose, hand = forward_arm(truth, "right", body_rotation=CAM, body_offset=(0, 0, 2.0))
        pose = pose + rng.normal(size=pose.shape) * pose_noise
        hand = hand + rng.normal(size=hand.shape) * args.hand_cm / 100
        if ps:
            pose = ps(pose, t)
        if hs:
            hand = hs(hand, t)
        a = kin.solve(pose, hand).as_dict()
        vals = {k: a[k] for k in KEYS}
        if ang_f:
            vals = ang_f(vals, t)
        if t >= 1.0:                      # let filters settle
            for k in KEYS:
                outs[k].append(vals[k])
                errs[k].append(((vals[k] - truth[k] + 180) % 360) - 180)
    jitter = {k: float(np.std(outs[k])) for k in KEYS}
    rms = {k: float(np.sqrt(np.mean(np.square(errs[k])))) for k in KEYS}
    return jitter, rms


from vtrack.filters import HAND_SMOOTHING, SMOOTHING_PRESETS  # noqa: E402

PIPELINES = {"raw": dict(pose=None, hand=None, angles=None),
             "previous": dict(pose=dict(min_cutoff=1.5, beta=0.1), hand=dict(min_cutoff=2.0, beta=0.2), angles=None)}
for _name, _p in SMOOTHING_PRESETS.items():
    PIPELINES[_name] = dict(pose=dict(_p["pose"], min_cutoff=np.array(_p["pose"]["min_cutoff"])),
                            hand=HAND_SMOOTHING, angles=_p["angles"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xy-cm", type=float, default=0.6)
    ap.add_argument("--z-cm", type=float, default=2.0)
    ap.add_argument("--hand-cm", type=float, default=0.2)
    args = ap.parse_args()
    print(f"noise model: pose {args.xy_cm} cm (x, y), {args.z_cm} cm (depth); hand {args.hand_cm} cm\n")
    print(f"{'angle':22s}" + "".join(f"{n + ' jitter':>16s}" for n in PIPELINES) + "".join(
        f"{n + ' moving RMS':>18s}" for n in PIPELINES))
    res = {n: (run(p, False, args)[0], run(p, True, args)[1]) for n, p in PIPELINES.items()}
    for k in KEYS:
        print(f"{k:22s}" + "".join(f"{res[n][0][k]:16.2f}" for n in PIPELINES)
              + "".join(f"{res[n][1][k]:18.2f}" for n in PIPELINES))
    for n in PIPELINES:
        print(f"\n{n}: mean jitter {np.mean(list(res[n][0].values())):.2f} deg, "
              f"mean moving RMS error {np.mean(list(res[n][1].values())):.2f} deg")


if __name__ == "__main__":
    main()

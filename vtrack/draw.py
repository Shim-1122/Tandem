"""Wireframe overlays and a 3D kinematics panel, drawn with OpenCV.

The 3D arm panel is drawn from the SOLVED joint angles through forward kinematics, not from the raw
landmarks, so it shows what the solver believes the arm is doing. If it moves like your arm, the
kinematics are working; if it disagrees with the video, the angles are wrong.
"""
from __future__ import annotations

import math

import numpy as np

from .kinematics import rot_x, rot_y, rot_z
from .synthetic import arm_chain

POSE_ARM_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
                  (15, 17), (15, 19), (17, 19), (15, 21), (16, 18), (16, 20), (18, 20), (16, 22)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

GREEN, CYAN, WHITE, GREY, YELLOW = (0, 255, 0), (255, 255, 0), (255, 255, 255), (110, 110, 110), (0, 220, 255)
AXIS_COLORS = [(60, 60, 255), (60, 255, 60), (255, 120, 60)]   # X red, Y green, Z blue (BGR)


def _cv2():
    import cv2
    return cv2


def draw_wireframe(img, pts_norm, edges, color, visibility=None, thresh=0.5, radius=4):
    """Draw normalized landmarks (N, 2 or 3) and their connections onto a BGR image in place."""
    cv2 = _cv2()
    h, w = img.shape[:2]
    pts = np.asarray(pts_norm, dtype=float)
    finite = np.isfinite(pts[:, :2]).all(axis=1) & (np.abs(pts[:, :2]) < 10).all(axis=1)
    px = [(int(p[0] * w), int(p[1] * h)) if f else (0, 0) for p, f in zip(pts, finite)]
    used = {i for e in edges for i in e}
    ok = [bool(finite[i]) and (visibility is None or visibility[i] >= thresh) for i in range(len(px))]
    for a, b in edges:
        if ok[a] and ok[b]:
            cv2.line(img, px[a], px[b], color, 2, cv2.LINE_AA)
    for i in used:
        if ok[i]:
            cv2.circle(img, px[i], radius, WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, px[i], radius, color, 1, cv2.LINE_AA)
    return img


def _view(yaw_deg, pitch_deg, pitch_axis="z"):
    """Turn about the vertical axis, then tilt about the screen's horizontal axis."""
    tilt = rot_z if pitch_axis == "z" else rot_x
    return tilt(math.radians(pitch_deg)) @ rot_y(math.radians(yaw_deg))


def _project(points, view, center, scale, horiz_axis=2, vert_axis=1, vert_sign=-1.0):
    q = (view @ np.asarray(points, dtype=float).T).T
    return [(int(center[0] + scale * p[horiz_axis]), int(center[1] + vert_sign * scale * p[vert_axis])) for p in q]


ARM_COLORS = {"right": CYAN, "left": (255, 140, 255)}   # BGR: cyan right, pink left
KEYS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion",
        "forearm_rotation", "wrist_flexion", "wrist_deviation"]


def render_arm_panel(angles, side="right", size=480, yaw_deg=-35.0, pitch_deg=-20.0, note=""):
    """3D view of the torso and one arm rebuilt from solved angles. See `render_arms_panel`."""
    return render_arms_panel({side: angles}, size, yaw_deg, pitch_deg, note)


def render_arms_panel(angles_by_side, size=480, yaw_deg=-30.0, pitch_deg=-20.0, note=""):
    """3D view of the torso and arms rebuilt from solved angles. Returns a BGR image.

    angles_by_side: {"right": ArmAngles or dict, "left": ...}; either side may be omitted or all NaN.
    Drawn as a mirror view (your right arm on the right), matching the mirrored video. Small
    red/green/blue axes show each segment's frame: upper arm at the elbow, forearm at the wrist,
    hand at the knuckles.
    """
    cv2 = _cv2()
    img = np.full((size, size, 3), 25, np.uint8)
    view = _view(yaw_deg, pitch_deg, "z")
    center, scale = (size // 2, int(size * 0.30)), size * 0.75

    def pr(*pts):
        return _project(pts, view, center, scale)

    for g in np.linspace(-0.4, 0.4, 5):                 # floor grid for depth cues
        a, b = pr(np.array([g, -0.5, -0.4]), np.array([g, -0.5, 0.4]))
        cv2.line(img, a, b, (45, 45, 45), 1)
        a, b = pr(np.array([-0.4, -0.5, g]), np.array([0.4, -0.5, g]))
        cv2.line(img, a, b, (45, 45, 45), 1)
    rs, ls, rh, lh = pr(np.array([0, 0, 0.2]), np.array([0, 0, -0.2]),
                        np.array([0, -0.5, 0.1]), np.array([0, -0.5, -0.1]))
    for a, b in ((rs, ls), (rs, rh), (ls, lh), (rh, lh)):
        cv2.line(img, a, b, GREY, 2, cv2.LINE_AA)
    cv2.putText(img, "R", (rs[0] + 8, rs[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ARM_COLORS["right"], 2)
    cv2.putText(img, "L", (ls[0] - 22, ls[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ARM_COLORS["left"], 2)

    drawn = 0
    for side, angles in angles_by_side.items():
        if angles is None:
            continue
        get = (lambda k: angles[k]) if isinstance(angles, dict) else (lambda k: getattr(angles, k))
        vals = {k: get(k) for k in KEYS}
        if any(not np.isfinite(vals[k]) for k in KEYS[:4]):
            continue
        drawn += 1
        wrist_ok = all(np.isfinite(vals[k]) for k in KEYS[4:])
        for k in KEYS[4:]:
            if not np.isfinite(vals[k]):
                vals[k] = 0.0
        c = arm_chain(vals)
        mirror = np.array([1.0, 1.0, -1.0 if side == "left" else 1.0])
        P = {k: c[k] * mirror for k in ("S", "E", "W", "K", "index", "pinky")}
        s, e, w, k, ix, pk = pr(P["S"], P["E"], P["W"], P["K"], P["index"], P["pinky"])
        col = ARM_COLORS[side]
        cv2.line(img, s, e, col, 4, cv2.LINE_AA)
        cv2.line(img, e, w, col, 4, cv2.LINE_AA)
        hand_col = YELLOW if wrist_ok else GREY
        for a, b in ((w, ix), (w, pk), (ix, pk), (w, k)):
            cv2.line(img, a, b, hand_col, 2, cv2.LINE_AA)
        for p in (s, e, w):
            cv2.circle(img, p, 6, WHITE, -1, cv2.LINE_AA)
        for origin, R in ((P["E"], c["R_h"]), (P["W"], c["R_f"]), (P["K"], c["R_hand"])):
            R = R * mirror[:, None]
            for i in range(3):
                a, b = pr(origin, origin + 0.08 * R[:, i])
                cv2.arrowedLine(img, a, b, AXIS_COLORS[i], 2, cv2.LINE_AA, tipLength=0.25)
    if drawn == 0:
        cv2.putText(img, "no arm pose", (20, size // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
    cv2.putText(img, "solved kinematics (mirror view)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    if note:
        cv2.putText(img, note, (10, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
    return img


def render_hand_panel(world, size=480, yaw_deg=-30.0, pitch_deg=-20.0, note=""):
    """3D view of a hand from its world landmarks (camera axes: x right, y down, z away). BGR image."""
    cv2 = _cv2()
    img = np.full((size, size, 3), 25, np.uint8)
    if world is None:
        cv2.putText(img, "no hand", (20, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
        return img
    pts = np.asarray(world, dtype=float) - np.asarray(world, dtype=float).mean(axis=0)
    view = _view(yaw_deg, pitch_deg, "x")
    px = _project(pts, view, (size // 2, size // 2), size * 2.2, horiz_axis=0, vert_axis=1, vert_sign=1.0)
    for a, b in HAND_EDGES:
        cv2.line(img, px[a], px[b], YELLOW, 2, cv2.LINE_AA)
    for p in px:
        cv2.circle(img, p, 4, WHITE, -1, cv2.LINE_AA)
    cv2.putText(img, "hand, 3D (camera view, rotated)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    if note:
        cv2.putText(img, note, (10, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
    return img


def text_block(img, lines, scale=None, color=GREEN):
    """Draw lines of text on a dark box in the top-left corner, in place.

    A line containing tab characters is drawn as columns: the first cell left aligned, later cells
    right aligned at fixed positions, so numbers line up even with OpenCV's proportional font.
    """
    cv2 = _cv2()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = scale or max(0.5, img.shape[1] / 1280 * 0.8)
    step = int(30 * scale)

    def width(t):
        return cv2.getTextSize(t, font, scale, 2)[0][0]

    cells = [ln.split("\t") for ln in lines]
    first_w = max([width(c[0]) for c in cells if len(c) > 1] + [0])
    col_w = max([width(x) for c in cells if len(c) > 1 for x in c[1:]] + [0]) + int(18 * scale)
    ncols = max(len(c) for c in cells)
    total = max([width(c[0]) for c in cells if len(c) == 1] + [first_w + col_w * (ncols - 1)] + [10])
    cv2.rectangle(img, (0, 0), (total + 20, step * len(lines) + 12), (0, 0, 0), -1)
    for i, c in enumerate(cells):
        y = step * (i + 1)
        cv2.putText(img, c[0], (10, y), font, scale, color, 2, cv2.LINE_AA)
        for j, x in enumerate(c[1:], start=1):
            right = 10 + first_w + col_w * j
            cv2.putText(img, x, (right - width(x), y), font, scale, color, 2, cv2.LINE_AA)
    return img

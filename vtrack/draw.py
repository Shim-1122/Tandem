"""Wireframe overlays and a 3D kinematics panel, drawn with OpenCV.

The 3D arm panel is drawn from the SOLVED joint angles through forward kinematics, not from the raw
landmarks, so it shows what the solver believes the arm is doing. If it moves like your arm, the
kinematics are working; if it disagrees with the video, the angles are wrong.
"""
from __future__ import annotations

import math

import numpy as np

from .kinematics import rot_x, rot_y, rot_z
from .fingers import hand_fk
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


def render_arms_panel(angles_by_side, size=480, yaw_deg=-30.0, pitch_deg=-20.0, note="", fingers_by_side=None,
                      head_axes=None):
    """3D view of the torso and arms rebuilt from solved angles. Returns a BGR image.

    angles_by_side: {"right": ArmAngles or dict, "left": ...}; either side may be omitted or all NaN.
    fingers_by_side: optional {side: (finger angles, hand geometry)} from fingers.py; when given, that
    hand is drawn from the solved finger angles through forward kinematics instead of as a simple palm.
    An arm whose elbow was predicted (out of view) is drawn in a darker shade.
    head_axes: optional (3, 3) head axes (forward, up, right) in the torso frame; draws the head with its
    four anchor points above the shoulders.
    Drawn as a mirror view (your right arm on the right), matching the mirrored video. Small
    red/green/blue axes show each segment's frame: upper arm at the elbow, forearm at the wrist,
    hand at the knuckles.
    """
    cv2 = _cv2()
    img = np.full((size, size, 3), 25, np.uint8)
    view = _view(yaw_deg, pitch_deg, "z")
    center, scale = (size // 2, int(size * 0.36)), size * 0.72

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
        if not isinstance(angles, dict) and getattr(angles, "elbow_predicted", False):
            col = tuple(int(x * 0.5) for x in col)
        cv2.line(img, s, e, col, 4, cv2.LINE_AA)
        cv2.line(img, e, w, col, 4, cv2.LINE_AA)
        hand_col = YELLOW if wrist_ok else GREY
        fing = (fingers_by_side or {}).get(side)
        if fing is not None and wrist_ok:
            Pf = hand_fk(*fing)
            pts = (c["W"] + Pf @ c["R_hand"].T) * mirror
            pp = pr(*pts)
            for a, b in HAND_EDGES:
                cv2.line(img, pp[a], pp[b], hand_col, 2, cv2.LINE_AA)
        else:
            for a, b in ((w, ix), (w, pk), (ix, pk), (w, k)):
                cv2.line(img, a, b, hand_col, 2, cv2.LINE_AA)
        for p in (s, e, w):
            cv2.circle(img, p, 6, WHITE, -1, cv2.LINE_AA)
        for origin, R in ((P["E"], c["R_h"]), (P["W"], c["R_f"]), (P["K"], c["R_hand"])):
            R = R * mirror[:, None]
            for i in range(3):
                a, b = pr(origin, origin + 0.08 * R[:, i])
                cv2.arrowedLine(img, a, b, AXIS_COLORS[i], 2, cv2.LINE_AA, tipLength=0.25)
    if head_axes is not None:
        from .head import HEAD_AXES_IN_MODEL, MODEL_POINTS_M
        head_c = np.array([0.0, 0.24, 0.0])
        anchors = [head_c + 1.6 * head_axes @ (HEAD_AXES_IN_MODEL.T @ p) for p in MODEL_POINTS_M]   # drawn 1.6x size
        f, n, r, l = pr(*anchors)
        c = pr(head_c, np.array([0.0, 0.0, 0.0]))
        cv2.line(img, c[1], c[0], GREY, 2, cv2.LINE_AA)                       # neck
        cv2.circle(img, c[0], int(0.13 * scale), (90, 90, 90), 1, cv2.LINE_AA)
        for a, b in ((r, l), (r, f), (l, f), (r, n), (l, n)):
            cv2.line(img, a, b, YELLOW, 1, cv2.LINE_AA)
        for q, col in ((f, WHITE), (n, (0, 220, 255)), (r, (255, 140, 255)), (l, (255, 140, 255))):
            cv2.circle(img, q, 4, col, -1, cv2.LINE_AA)
        for i in range(3):
            a, b = pr(head_c, head_c + 0.13 * head_axes[:, i])
            cv2.arrowedLine(img, a, b, AXIS_COLORS[i], 2, cv2.LINE_AA, tipLength=0.25)
    if drawn == 0:
        cv2.putText(img, "no arm pose", (20, size // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
    cv2.putText(img, "solved kinematics (mirror view)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    if note:
        cv2.putText(img, note, (10, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
    return img


def render_hand_panel(world, size=480, yaw_deg=-30.0, pitch_deg=-20.0, note="", solved=None):
    """3D view of a hand from its world landmarks (camera axes: x right, y down, z away). BGR image.

    solved: optional (21, 3) hand rebuilt from the solved finger angles (same axes); drawn in yellow over
    the raw landmarks (grey), so any mismatch between measurement and model is visible.
    """
    cv2 = _cv2()
    img = np.full((size, size, 3), 25, np.uint8)
    if world is None:
        cv2.putText(img, "no hand", (20, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2)
        return img
    center = np.asarray(world, dtype=float).mean(axis=0)
    view = _view(yaw_deg, pitch_deg, "x")

    def proj(points):
        return _project(np.asarray(points, float) - center, view, (size // 2, size // 2), size * 2.2,
                        horiz_axis=0, vert_axis=1, vert_sign=1.0)

    px = proj(world)
    raw_col = GREY if solved is not None else YELLOW
    for a, b in HAND_EDGES:
        cv2.line(img, px[a], px[b], raw_col, 2 if solved is None else 1, cv2.LINE_AA)
    if solved is not None:
        sp = proj(solved)
        for a, b in HAND_EDGES:
            cv2.line(img, sp[a], sp[b], YELLOW, 2, cv2.LINE_AA)
        px = sp
    for p in px:
        cv2.circle(img, p, 4, WHITE, -1, cv2.LINE_AA)
    cv2.putText(img, "hand, 3D (camera view, rotated)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    if note:
        cv2.putText(img, note, (10, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
    return img


def text_block(img, lines, scale=None, color=GREEN, origin=(0, 0), alpha=0.55):
    """Draw lines of text on a semi-transparent dark box (top-left corner at `origin`), in place.

    A line containing tab characters is drawn as columns: the first cell left aligned, later cells
    right aligned at fixed positions, so numbers line up even with OpenCV's proportional font.
    Returns the (width, height) of the box.
    """
    cv2 = _cv2()
    if not lines:
        return 0, 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = scale or max(0.38, img.shape[1] / 1280 * 0.46)
    thick = 1 if scale < 0.6 else 2
    step = int(round(26 * scale)) + 4

    def width(t):
        return cv2.getTextSize(t, font, scale, thick)[0][0]

    cells = [ln.split("\t") for ln in lines]
    first_w = max([width(c[0]) for c in cells if len(c) > 1] + [0])
    col_w = max([width(x) for c in cells if len(c) > 1 for x in c[1:]] + [0]) + int(16 * scale)
    ncols = max(len(c) for c in cells)
    total = max([width(c[0]) for c in cells if len(c) == 1] + [first_w + col_w * (ncols - 1)] + [10])
    x0, y0 = origin
    x1, y1 = min(img.shape[1], x0 + total + 16), min(img.shape[0], y0 + step * len(lines) + 10)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1.0 - alpha)).astype(img.dtype)
    for i, c in enumerate(cells):
        y = y0 + step * (i + 1)
        cv2.putText(img, c[0], (x0 + 8, y), font, scale, color, thick, cv2.LINE_AA)
        for j, x in enumerate(c[1:], start=1):
            right = x0 + 8 + first_w + col_w * j
            cv2.putText(img, x, (right - width(x), y), font, scale, color, thick, cv2.LINE_AA)
    return x1 - x0, y1 - y0


def compose_layout(video, face_panel, kin_panel, max_width=1600):
    """Video on the left; a right column with the face panel on top and the 3D panel below."""
    cv2 = _cv2()
    h = video.shape[0]
    top = h // 2
    col = np.vstack([cv2.resize(face_panel, (top, top)), cv2.resize(kin_panel, (top, h - top))])
    canvas = np.hstack([video, col])
    if canvas.shape[1] > max_width:
        f = max_width / canvas.shape[1]
        # linear: about 6 times faster than area averaging for this small reduction, same look
        canvas = cv2.resize(canvas, (max_width, int(round(canvas.shape[0] * f))), interpolation=cv2.INTER_LINEAR)
    return canvas


def draw_head_on_video(img, head, K, axis_len_m=0.08):
    """Anchor dots and the head's forward/up/right axes, drawn on the raw (unmirrored) camera frame."""
    cv2 = _cv2()
    if head is None:
        return img
    for (u, v), col in zip(head.anchors_px, [(255, 255, 255), (0, 220, 255), (255, 140, 255), (255, 140, 255)]):
        cv2.circle(img, (int(u), int(v)), 6, col, -1, cv2.LINE_AA)
        cv2.circle(img, (int(u), int(v)), 6, (0, 0, 0), 1, cv2.LINE_AA)
    origin = head.position_m
    ends = [origin + axis_len_m * head.axes[:, i] for i in range(3)]
    pts = [_project_cam(p, K) for p in [origin] + ends]
    if all(p is not None for p in pts):
        for i in range(3):
            cv2.arrowedLine(img, pts[0], pts[i + 1], AXIS_COLORS[i], 3, cv2.LINE_AA, tipLength=0.2)
    return img


def _project_cam(p, K):
    if p[2] <= 1e-6:
        return None
    u = K @ p
    return int(u[0] / u[2]), int(u[1] / u[2])


def _bar(img, x, y, w, h, frac, color, bg=(60, 60, 60)):
    cv2 = _cv2()
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + int(round(w * max(0.0, min(1.0, frac)))), y + h), color, -1)


def render_face_panel(size=360, face_crop=None, emotions=None, identity_line="", head=None, head_rel=None,
                      note=""):
    """Face, identity, the six cardinal emotions with percentages, and the head's 6 degrees of freedom.

    emotions: dict of the seven model classes -> probability (smoothed), or None when no face.
    head: HeadPose-like object (position_cm, yaw, pitch, roll) or None; head_rel: (dx, dy, dz, dyaw,
    dpitch, droll) relative to the zero pose, or None.
    """
    cv2 = _cv2()
    from .face import CARDINAL
    img = np.full((size, size, 3), 25, np.uint8)
    s = size / 360.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(text, x, y, sc=0.45, col=WHITE, th=1):
        cv2.putText(img, text, (int(x * s), int(y * s)), font, sc * s, col, th, cv2.LINE_AA)

    put("Face", 10, 20, 0.55, WHITE, 1)
    thumb = int(110 * s)
    if face_crop is not None and face_crop.size:
        img[int(28 * s):int(28 * s) + thumb, int(10 * s):int(10 * s) + thumb] = cv2.resize(face_crop, (thumb, thumb))
    else:
        cv2.rectangle(img, (int(10 * s), int(28 * s)), (int(10 * s) + thumb, int(28 * s) + thumb), (60, 60, 60), 1)
        put("no face", 30, 88, 0.45, GREY)
    put(identity_line[:28], 128, 44, 0.45, YELLOW if identity_line else GREY)
    if emotions is not None:
        y = 62
        top = max(CARDINAL, key=lambda kv: emotions[kv[0]])[0]
        for key, label in CARDINAL:
            p = emotions[key]
            col = (0, 220, 120) if key == top else (0, 160, 255)
            put(label, 128, y + 9, 0.38, WHITE)
            _bar(img, int(205 * s), int(y * s), int(105 * s), int(10 * s), p, col)
            put(f"{100 * p:3.0f}%", 314, y + 9, 0.38, WHITE)
            y += 15
        put(f"Neutral {100 * emotions['neutral']:.0f}%  (all seven add to 100%)", 128, y + 9, 0.34, GREY)
    else:
        put("emotion: waiting for a face", 128, 75, 0.4, GREY)
    y0 = 190
    put("Head (6 DOF, camera frame)", 10, y0, 0.45, WHITE)
    if head is not None:
        x, yy, z, yaw, pitch, roll = [float(v) + 0.0 for v in np.round([*head.position_cm, head.yaw, head.pitch,
                                                                         head.roll], 1)]   # no "-0.0"
        put(f"x {x:6.1f}  y {yy:6.1f}  z {z:6.1f} cm", 10, y0 + 22, 0.42, GREEN)
        put(f"yaw {yaw:6.1f}  pitch {pitch:6.1f}  roll {roll:6.1f} deg", 10, y0 + 42, 0.42, GREEN)
        if head_rel is not None:
            head_rel = [float(v) + 0.0 for v in np.round(head_rel, 1)]     # + 0.0 turns -0.0 into 0.0
            put("since zero (h):", 10, y0 + 64, 0.38, GREY)
            put(f"dx {head_rel[0]:5.1f} dy {head_rel[1]:5.1f} dz {head_rel[2]:5.1f} cm", 10, y0 + 82, 0.4, CYAN)
            put(f"dyaw {head_rel[3]:5.1f} dpitch {head_rel[4]:5.1f} droll {head_rel[5]:5.1f} deg", 10, y0 + 100, 0.4, CYAN)
        else:
            put("press h to zero the head pose", 10, y0 + 64, 0.38, GREY)
        put("yaw +: turn left  pitch +: look up  roll +: tilt right", 10, y0 + 120, 0.34, GREY)
        put("anchors: forehead, nose tip, outer eye corners", 10, y0 + 136, 0.34, GREY)
    else:
        put("no face landmarks", 10, y0 + 22, 0.42, GREY)
    if note:
        put(note[:52], 10, 352, 0.36, YELLOW)
    return img


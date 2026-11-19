"""Bake the ICT face (ICT-FaceKit, MIT license, USC Institute for Creative Technologies) into vtrack/data/face_ict.npz,
the face skin, teeth and eyes of the simulated human (vtrack/human.py).

    python scripts/bake_face.py                          # downloads the 57 files it needs (about 145 MB) once
    python scripts/bake_face.py --ict path/to/ICT-FaceKit  # from a clone of github.com/USC-ICT/ICT-FaceKit

You do not need to run this: the baked file is in the repository. It is here so the bake can be checked and redone.

What it computes, all in the head frame of the simulated human (meters; x forward, y to the person's left, z up):
  * the skin (face, head and neck, welded where the parts meet), the inside of the mouth and the eyelashes, which
    deform; the teeth, gums and tongue, split into the upper set (on the skull) and the lower set (on the jaw);
    each eyeball (white, iris and pupil), on its own eye joint
  * the servos of Will Cogley's DIY robot head as the reference layout (16 skin servos: brows, lids, lips and
    cheeks): each servo's skin field (how its patch of skin moves at full travel) and how far each servo turns for
    each ARKit movement, fitted together so the servos reproduce ICT's 47 sculpted movements as closely as 16 servos
    can, with each servo moving only the skin around it
  * the jaw: its hinge axis and opening angle, its sideways and forward travel, fitted as rigid motions of the
    lower teeth in ICT's jaw expressions, and how much each skin point turns with the jaw (its bone weight). The
    stored jaw fields are what is left after the jaw's rotation, so hinge plus field gives back the sculpt exactly.
  * the eyeball centers (spheres fitted to the whites)
  * the skin's texture (face_ict_skin.png): skin, lips, brows and lash lines placed from ICT's 68 landmarks, and
    hair over the scalp, painted in the face's own texture layout
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ICT_COMMIT = "da5f95a607f5e6b37755b38d3385d7f2853732e5"
ICT_URL = f"https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/{ICT_COMMIT}/FaceXModel/"
OUT = Path(__file__).resolve().parents[1] / "vtrack" / "data" / "face_ict.npz"

# ICT's vertex ranges (its README): [start, end)
PARTS = {"face": (0, 9409), "head": (9409, 11248), "mouth": (11248, 13294), "gums": (14062, 17039),
         "teeth": (17039, 21451), "sclera_L": (21451, 22221), "iris_L": (22221, 23021), "sclera_R": (23021, 23791),
         "iris_R": (23791, 24591), "lash_L": (25351, 26035), "lash_R": (26035, 26719)}
# ICT expressions used (its _L and _R are the person's own sides, like ARKit's Left and Right)
SIDED = ["browDown", "browInnerUp", "browOuterUp", "cheekSquint", "eyeBlink", "eyeLookDown", "eyeLookIn", "eyeLookOut",
         "eyeLookUp", "eyeSquint", "eyeWide", "mouthDimple", "mouthFrown", "mouthLowerDown", "mouthPress", "mouthSmile",
         "mouthStretch", "mouthUpperUp", "noseSneer"]
SINGLE = ["jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthFunnel", "mouthLeft", "mouthPucker",
          "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper"]
# Where the head frame's origin sits in ICT's coordinates (cm): on the midline, 9.4 cm below ICT's origin (about
# level with the bottom of the jaw), over the middle of the neck (its z is found from the neck)
HEAD_ORIGIN_Y = -9.4
TO_M = 0.01
IRIS_DEG, PUPIL_DEG = 31.0, 12.0     # half-angles of the iris and the pupil on the eyeball
# Colors painted into the skin texture (linear 0..1 RGB)
SKIN_TONE = (0.80, 0.60, 0.48)
LIP, BROW, LASH, HAIR = (0.66, 0.40, 0.40), (0.17, 0.12, 0.10), (0.10, 0.08, 0.07), (0.13, 0.10, 0.085)


def ict_names() -> list[str]:
    return (["generic_neutral_mesh"] + [f"{n}_{s}" for n in SIDED for s in "LR"] + ["cheekPuff_L", "cheekPuff_R"]
            + SINGLE)


def fetch(ict: Path | None, cache: Path) -> Path:
    if ict is not None:
        return ict / "FaceXModel"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ict_names() + ["vertex_indices"]:
        ext = ".json" if name == "vertex_indices" else ".obj"
        f = cache / (name + ext)
        if not f.exists():
            print("downloading", f.name)
            urllib.request.urlretrieve(ICT_URL + f.name, f)
    return cache


def read_obj(path: Path, faces: bool = False):
    v, f = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif faces and line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                f.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))   # fan into triangles
    v = np.array(v, dtype=np.float64)
    return (v, np.array(f, dtype=np.int64)) if faces else v


def to_head(p_ict: np.ndarray, origin_ict: np.ndarray) -> np.ndarray:
    """ICT cm (x the person's left, y up, z forward) to head-frame meters (x forward, y left, z up)."""
    q = (p_ict - origin_ict) * TO_M
    return q[..., [2, 0, 1]]


def vec_to_head(d_ict: np.ndarray) -> np.ndarray:
    return (d_ict * TO_M)[..., [2, 0, 1]]


def kabsch(P: np.ndarray, Q: np.ndarray):
    """R, t with Q ~ R P + t."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def axis_angle(R: np.ndarray):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return ax / np.linalg.norm(ax), ang


def fit_sphere(P: np.ndarray):
    A = np.c_[2 * P, np.ones(len(P))]
    sol = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + c @ c))


def in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Points (n, 2) inside a polygon (m, 2), by ray casting."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cross
    return inside


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def read_obj_uv(path: Path):
    """Positions, texture coordinates and triangles as (vertex, uv) corner pairs."""
    v, vt, corners = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("vt "):
                vt.append(line.split()[1:3])
            elif line.startswith("f "):
                c = [tuple(int(x) - 1 for x in t.split("/")[:2]) for t in line.split()[1:]]
                corners.extend([c[0], c[i], c[i + 1]] for i in range(1, len(c) - 1))
    return np.array(v, float), np.array(vt, float), np.array(corners, dtype=np.int64)   # (T, 3, 2)


def catmull_rom(pts: np.ndarray, closed: bool, n: int = 12) -> np.ndarray:
    """A smooth curve through the points."""
    P = np.vstack([pts[-1], pts, pts[0], pts[1]]) if closed else np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(P[-2])
    return np.array(out)


def rasterize(uv_px: np.ndarray, tris: np.ndarray, attr: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a per-vertex attribute over triangles in texture space: (image of the attribute, coverage)."""
    H, W = shape
    img = np.zeros((H, W, attr.shape[1]), np.float32)
    cov = np.zeros((H, W), bool)
    for t in tris:
        p = uv_px[t]
        x0, y0 = np.floor(p.min(0)).astype(int)
        x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / den
        l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / den
        l3 = 1 - l1 - l2
        m = (l1 >= -0.02) & (l2 >= -0.02) & (l3 >= -0.02)
        if not m.any():
            continue
        val = l1[m, None] * attr[t[0]] + l2[m, None] * attr[t[1]] + l3[m, None] * attr[t[2]]
        yy, xx = np.nonzero(m)
        img[yy + y0, xx + x0] = val
        cov[yy + y0, xx + x0] = True
    return img, cov


def paint_skin(pos: np.ndarray, cov: np.ndarray, L: np.ndarray, seed: int = 7) -> np.ndarray:
    """The skin texture from a map of head-frame positions: skin, lips, brows, lash lines and hair."""
    import cv2
    rng = np.random.default_rng(seed)
    H, W = cov.shape
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    # frontal masks on a grid over the face: 0.1 mm per pixel, y from -0.1 to 0.1 m, z from 0 to 0.26 m
    res, y_lo, z_hi = 1e-4, -0.1, 0.26
    FW, FH = 2000, 2600

    def to_px(pts_yz):
        return np.c_[(pts_yz[:, 0] - y_lo) / res, (z_hi - pts_yz[:, 1]) / res].astype(np.int32)

    def sample(mask):
        px = np.clip(((y - y_lo) / res).astype(int), 0, FW - 1)
        pz = np.clip(((z_hi - z) / res).astype(int), 0, FH - 1)
        return mask[pz, px]

    lips = np.zeros((FH, FW), np.float32)
    cv2.fillPoly(lips, [to_px(catmull_rom(L[48:60, 1:], closed=True))], 1.0, lineType=cv2.LINE_AA)
    lips = cv2.GaussianBlur(lips, (0, 0), 4)
    brows = np.zeros((FH, FW), np.float32)
    for idx in (list(range(17, 22)), list(range(22, 27))):
        pts = L[idx][:, 1:]
        if abs(pts[0, 0]) > abs(pts[-1, 0]):
            pts = pts[::-1]                                      # inner end first
        curve = catmull_rom(pts, closed=False, n=10)
        k = np.linspace(0, 1, len(curve))
        thick = 0.0052 * (1 - k) + 0.0019 * k                    # full at the inner end, a thin tail
        top = curve + np.c_[np.zeros(len(k)), 0.0009 + 0.25 * thick]
        bot = curve - np.c_[np.zeros(len(k)), 0.75 * thick]
        shape = np.zeros((FH, FW), np.uint8)
        cv2.fillPoly(shape, [to_px(np.vstack([top, bot[::-1]]))], 255, lineType=cv2.LINE_AA)
        strokes = np.zeros((FH, FW), np.uint8)
        for _ in range(900):                                     # short hairs, combed up and out
            i = rng.integers(len(curve))
            c = curve[i] + np.c_[0, rng.uniform(-0.75, 0.25) * thick[i]][0]
            ang = np.radians(rng.normal(20 + 35 * k[i], 12)) * (1 if curve[-1, 0] > curve[0, 0] else -1)
            d = np.array([np.cos(ang) * np.sign(curve[-1, 0] - curve[0, 0]) * 0.8 + 0.0, np.sin(abs(ang))])
            d = d / np.linalg.norm(d) * rng.uniform(0.0025, 0.0045)
            a, b = to_px(np.vstack([c, c + d]))
            cv2.line(strokes, tuple(int(v) for v in a), tuple(int(v) for v in b), int(rng.integers(150, 256)), 2,
                     cv2.LINE_AA)
        hairs = cv2.GaussianBlur(shape, (0, 0), 3).astype(np.float32) / 255
        brows = np.maximum(brows, hairs * (0.45 + 0.55 * strokes / 255.0))
    lash = np.zeros((FH, FW), np.float32)
    for upper, lower in ((L[[36, 37, 38, 39]], L[[39, 40, 41, 36]]), (L[[42, 43, 44, 45]], L[[45, 46, 47, 42]])):
        cv2.polylines(lash, [to_px(catmull_rom(upper[:, 1:], closed=False))], False, 1.0, 9, cv2.LINE_AA)
        cv2.polylines(lash, [to_px(catmull_rom(lower[:, 1:], closed=False))], False, 0.55, 4, cv2.LINE_AA)
    lash = cv2.GaussianBlur(lash, (0, 0), 2)

    eyes_x = L[[36, 45], 0].mean()
    front = np.clip((x - (eyes_x - 0.035)) / 0.01, 0, 1)       # only the front of the face takes the masks
    w_lips = sample(lips) * np.clip((x - (L[48, 0] - 0.012)) / 0.004, 0, 1)
    w_brow = sample(brows) * front * (z > L[36, 2])
    near_eye = np.minimum(np.linalg.norm(pos - L[[36, 39]].mean(0), axis=-1),
                          np.linalg.norm(pos - L[[42, 45]].mean(0), axis=-1))
    w_lash = sample(lash) * front * (near_eye < 0.022)
    # hair: above a hairline over the forehead, the temples and down to the nape (heights above the eyes, in m)
    rel_z = z - L[36:48, 2].mean()
    back = x - eyes_x                                          # 0 at the eyes, negative toward the back
    hairline = np.interp(back, [-0.25, -0.17, -0.12, -0.07, -0.035, -0.015, 0.05],
                         [-0.13, -0.115, -0.05, 0.035, 0.052, 0.058, 0.058])
    wobble = 0.0012 * cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 5) / 0.056
    ear = (np.abs(y) > 0.06) & (rel_z < 0.035) & (rel_z > -0.06) & (back > -0.13) & (back < -0.05)
    w_hair = np.clip((rel_z - hairline - wobble) / 0.004 + 0.5, 0, 1) * ~ear
    # colors (linear blend of flat colors with a little noise)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.2)
    noise2 = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 6)
    skin = np.array(SKIN_TONE) * (1 + 0.025 * noise[..., None] + 0.03 * noise2[..., None])
    blush = np.zeros((H, W), np.float32)
    for cpt, r in ((L[[1, 31]].mean(0), 0.022), (L[[15, 35]].mean(0), 0.022), (L[30], 0.013)):
        blush += np.exp(-np.linalg.norm(pos - cpt, axis=-1) ** 2 / (2 * r * r))
    tint = np.dstack([1 + 0.0 * blush, 1 - 0.07 * blush, 1 - 0.07 * blush])
    col = skin * tint
    lip_c = np.array(LIP)
    col = col * (1 - w_lips[..., None]) + lip_c * (1 + 0.03 * noise[..., None]) * w_lips[..., None]
    brow_c = np.array(BROW)
    col = col * (1 - w_brow[..., None] * 0.92) + brow_c * w_brow[..., None] * 0.92
    col = col * (1 - w_lash[..., None] * 0.9) + np.array(LASH) * w_lash[..., None] * 0.9
    strands = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=4)
    hair_c = np.array(HAIR) * (1 + 0.18 * strands[..., None])
    col = col * (1 - w_hair[..., None]) + hair_c * w_hair[..., None]
    img = np.clip(col, 0, 1)
    img[~cov] = 0
    # pad past the island edges so filtering never pulls in the background
    known = cov.astype(np.uint8)
    for _ in range(12):
        grown = cv2.dilate(img, np.ones((3, 3), np.uint8))
        grown_k = cv2.dilate(known, np.ones((3, 3), np.uint8))
        fill = (grown_k > 0) & (known == 0)
        img[fill] = grown[fill]
        known = grown_k
    return (img * 255 + 0.5).astype(np.uint8)


# The skin servos of Will Cogley's DIY robot head (github.com/will-cogley/RobotHead), used as the reference layout:
# per side a middle and an outer brow servo, an upper and a lower lid, an upper and a lower lip, an upper and a
# lower cheek. (His head also has two jaw servos, a tongue servo and two eye servos shared by both eyes; here the
# jaw is the fitted hinge and the eyes are their own joints.) Each servo moves a patch of skin around where it pulls:
# (name, his channel name, landmarks whose mean is the patch center, patch radius in m). Sides are the person's own.
SERVO_LAYOUT = []
for _p, _side, (brow_in, brow_out, lid_up, lid_lo, lip_up, lip_lo, chk_up, chk_lo) in (
        ("R_", "RIGHT", ([19, 20, 21], [17, 18], [37, 38], [40, 41], [49, 50], [58, 59], [2, 31, 48], [48, 4, 6])),
        ("L_", "LEFT", ([22, 23, 24], [25, 26], [43, 44], [46, 47], [52, 53], [55, 56], [14, 35, 54], [54, 12, 10]))):
    SERVO_LAYOUT += [(_p + "brow_inner", f"MID_{_side}_BROW", brow_in, 0.026),
                     (_p + "brow_outer", f"FAR_{_side}_BROW", brow_out, 0.026),
                     (_p + "upper_lid", f"TOP_{_side}_LID", lid_up, 0.015),
                     (_p + "lower_lid", f"LOW_{_side}_LID", lid_lo, 0.015),
                     (_p + "upper_lip", f"TOP_{_side}_LIP", lip_up, 0.030),
                     (_p + "lower_lip", f"LOW_{_side}_LIP", lip_lo, 0.030),
                     (_p + "upper_cheek", f"TOP_{_side}_CHEEK", chk_up, 0.038),
                     (_p + "lower_cheek", f"LOW_{_side}_CHEEK", chk_lo, 0.036)]
NOT_SERVOED = ("jawOpen", "jawLeft", "jawRight", "jawForward")   # the jaw hinge does these (and has no side slide)


def mirror_name(n: str) -> str:
    """The movement or servo on the other side (Left <-> Right, R_ <-> L_)."""
    if n.startswith(("R_", "L_")):
        return {"R": "L", "L": "R"}[n[0]] + n[1:]
    return n.replace("Left", "\0").replace("Right", "Left").replace("\0", "Right")


def fit_servos(F: np.ndarray, pts: np.ndarray, L: np.ndarray, inputs: list, part: np.ndarray, lam: float = 0.005,
               stiff: float = 0.1, iters: int = 40):
    """Re-express the sculpted movements F (B, m, 3) with the servos of SERVO_LAYOUT: servo skin fields Phi (S, m, 3)
    and a blend matrix Theta (B, S) with F ~ Theta Phi, by alternating least squares. Each servo's field is kept to
    its own patch of skin (a penalty that grows away from the patch center, plus a little stiffness everywhere so
    neighboring patches cannot cancel each other with large opposite motions), and the rig is kept mirror-symmetric
    (each left servo is its right twin reflected). Phi is scaled so every servo spans -1..1."""
    from scipy.spatial import cKDTree
    names = [n for n, _, _, _ in SERVO_LAYOUT]
    S = len(names)
    smir = np.array([names.index(mirror_name(n)) for n in names])
    bmir = np.array([inputs.index(mirror_name(n)) for n in inputs])
    vmir = np.empty(len(pts), int)                                         # each point's mirror twin, in its part
    for pid in np.unique(part):
        ids = np.flatnonzero(part == pid)
        vmir[ids] = ids[cKDTree(pts[ids]).query(pts[ids] * [1, -1, 1])[1]]
    flip = np.array([1.0, -1.0, 1.0])

    def symmetric(Phi, Theta):
        Phi = 0.5 * (Phi + Phi[smir][:, vmir] * flip)
        Theta = 0.5 * (Theta + Theta[bmir][:, smir])
        return Phi, Theta

    g = np.stack([np.exp(-np.sum((pts - L[idx].mean(0)) ** 2, 1) / (2 * r * r)) for _, _, idx, r in SERVO_LAYOUT], 1)
    pen = lam / np.maximum(g, 1e-4) + stiff                                # (m, S)
    B, m, _ = F.shape
    Ff = F.reshape(B, -1)
    Theta = np.sqrt(np.einsum("bmk,ms->bs", F ** 2, g))
    Theta /= Theta.max(0, keepdims=True) + 1e-12
    for _ in range(iters):
        G = Theta.T @ Theta                                                # (S, S)
        M = G[None] + np.einsum("ms,st->mst", pen, np.eye(S))              # (m, S, S)
        rhs = np.einsum("bs,bmk->msk", Theta, F)                           # (m, S, 3)
        Phi = np.linalg.solve(M, rhs).transpose(1, 0, 2)                   # (S, m, 3)
        A = Phi.reshape(S, -1).T                                           # (3m, S)
        Theta = np.linalg.solve(A.T @ A + 1e-12 * np.eye(S), A.T @ Ff.T).T   # (B, S)
        Phi, Theta = symmetric(Phi, Theta)
        c = np.abs(Theta).max(0) + 1e-12
        c = 0.5 * (c + c[smir])
        Theta /= c
        Phi *= c[:, None, None]
    err = np.linalg.norm(Ff - Theta @ Phi.reshape(S, -1), axis=1) / (np.linalg.norm(Ff, axis=1) + 1e-12)
    return Phi, Theta, err


def bake(src: Path, tex_size: int = 1024) -> tuple[dict, dict, np.ndarray]:
    N, VT, C = read_obj_uv(src / "generic_neutral_mesh.obj")
    F = C[..., 0]
    meta = json.loads((src / "vertex_indices.json").read_text(encoding="utf-8"))
    lm_ict = np.array(meta["idx_to_landmark_verts"])
    shapes = {name: read_obj(src / f"{name}.obj") - N for name in ict_names()[1:]}
    fields_ict = {}
    for n in SIDED:
        fields_ict[n + "Left"], fields_ict[n + "Right"] = shapes[n + "_L"], shapes[n + "_R"]
    fields_ict["cheekPuff"] = shapes["cheekPuff_L"] + shapes["cheekPuff_R"]
    for n in SINGLE:
        fields_ict[n] = shapes[n]

    a, b = PARTS["head"]
    neck = N[a:b][N[a:b, 1] < -17.0]
    origin = np.array([0.0, HEAD_ORIGIN_Y, neck[:, 2].mean()])
    V = to_head(N, origin)

    def faces_of(lo, hi):
        return F[np.all((F >= lo) & (F < hi), axis=1)]

    # ---- deformable base: the skin (face and head, ICT 0..11247) and the inside of the mouth (11248..13293)
    n_base = PARTS["mouth"][1]
    base = V[:n_base]
    part_id = np.zeros(n_base, np.int8)
    part_id[PARTS["mouth"][0]:] = 1
    # the skin's vertices are (position, texture coordinate) pairs: seams get a copy per side
    skin_c = C[np.all(F < PARTS["head"][1], axis=1)]
    pairs, inv = np.unique(skin_c.reshape(-1, 2), axis=0, return_inverse=True)
    skin_src, skin_uv = pairs[:, 0], VT[pairs[:, 1]]
    skin_f = inv.reshape(-1, 3)
    mouth_f = faces_of(*PARTS["mouth"])

    # ---- jaw: rigid fit of the lower teeth
    t_lo, t_hi = PARTS["teeth"]
    open_ict = fields_ict["jawOpen"]
    mov = np.linalg.norm(open_ict, axis=1)
    lower_teeth = np.arange(t_lo, t_hi)[mov[t_lo:t_hi] > 0.3]
    P = V[lower_teeth]
    R, t = kabsch(P, P + vec_to_head(open_ict[lower_teeth]))
    axis, angle = axis_angle(R)
    if axis[1] < 0:
        axis = -axis
    c = np.linalg.lstsq(np.eye(3) - R, t, rcond=None)[0]
    c = c - axis * (axis @ c)
    jaw_fit_err = np.abs(P @ R.T + t - (P + vec_to_head(open_ict[lower_teeth]))).max()

    def teeth_shift(name):
        return vec_to_head(fields_ict[name][lower_teeth]).mean(0)

    side_l, side_r, fwd = teeth_shift("jawLeft"), teeth_shift("jawRight"), teeth_shift("jawForward")
    side_axis = side_l / np.linalg.norm(side_l)
    fwd_axis = fwd / np.linalg.norm(fwd)

    # ---- fields, jaw bone weights and the jaw residuals
    fields = {k: vec_to_head(v[:n_base]) for k, v in fields_ict.items()}
    r_open = base @ R.T + t - base
    rr = (r_open ** 2).sum(1)
    w_jaw = np.where(rr > 1e-8, (fields["jawOpen"] * r_open).sum(1) / np.maximum(rr, 1e-12), 0.0)
    w_jaw = np.clip(w_jaw, 0.0, 1.0)
    fields["jawOpen"] = fields["jawOpen"] - w_jaw[:, None] * r_open
    fields["jawLeft"] = fields["jawLeft"] - w_jaw[:, None] * (side_axis * (side_l @ side_axis))
    fields["jawRight"] = fields["jawRight"] - w_jaw[:, None] * (side_axis * (side_r @ side_axis))
    fields["jawForward"] = fields["jawForward"] - w_jaw[:, None] * (fwd_axis * np.linalg.norm(fwd))
    w_chest = np.where(part_id == 0, smoothstep(-9.5, -17.5, N[:n_base, 1]), 0.0)
    w_jaw = np.minimum(w_jaw, 1.0 - w_chest)
    names = sorted(fields)
    stack = np.stack([fields[k] for k in names])
    stack[np.abs(stack) < 2e-5] = 0.0
    moving = np.flatnonzero(np.abs(stack).max(axis=(0, 2)) > 0)
    # the servos: fit to every movement except the jaw's
    L = V[lm_ict]
    inputs = [n for n in names if n not in NOT_SERVOED]
    Fm = stack[[names.index(n) for n in inputs]][:, moving]
    Phi, Theta, fit_err = fit_servos(Fm, base[moving], L, inputs, part_id[moving])
    Phi[np.abs(Phi) < 2e-5] = 0.0

    # ---- rigid parts
    def split_jaw(lo, hi):
        tris = faces_of(lo, hi)
        lower_t = (mov > 0.3)[tris].mean(1) > 0.5
        return tris[~lower_t], tris[lower_t]

    def compact(tris):
        ids, inv_ = np.unique(tris, return_inverse=True)
        return V[ids].astype(np.float32), inv_.reshape(-1, 3).astype(np.int32)

    out = {}
    for part, (lo, hi) in (("teeth", PARTS["teeth"]), ("gums", PARTS["gums"])):
        up, low = split_jaw(lo, hi)
        out[f"{part}_upper_v"], out[f"{part}_upper_f"] = compact(up)
        out[f"{part}_lower_v"], out[f"{part}_lower_f"] = compact(low)
    eye_center, eye_radius = {}, {}
    for s in "LR":
        lo, hi = PARTS[f"sclera_{s}"]
        cen, rad = fit_sphere(V[lo:hi])
        eye_center[s], eye_radius[s] = cen, rad
        ilo, ihi = PARTS[f"iris_{s}"]
        d = V[ilo:ihi] - cen
        fwd_dir = d.mean(0) / np.linalg.norm(d.mean(0))

        def ang_of(pts):
            q = pts - cen
            return np.degrees(np.arccos(np.clip((q @ fwd_dir) / np.linalg.norm(q, axis=1), -1, 1)))

        # ICT's "sclera" is the clear, wet outer shell and its "iris" the eyeball under it (the iris is painted
        # on in ICT's textures). Here the shell is left out and the eyeball is split into white, iris and pupil.
        tris = faces_of(ilo, ihi)
        ang = ang_of(V[tris].mean(1))
        for name, keep in (("sclera", ang >= IRIS_DEG), ("iris", (ang < IRIS_DEG) & (ang >= PUPIL_DEG)),
                           ("pupil", ang < PUPIL_DEG)):
            out[f"{name}_{s}_v"], out[f"{name}_{s}_f"] = compact(tris[keep])

    # ---- the skin's texture: two UV tiles side by side
    uv_px = np.c_[skin_uv[:, 0] / 2 * (2 * tex_size), (1 - skin_uv[:, 1]) * tex_size]
    pos_map, cov = rasterize(uv_px, skin_f, base[skin_src], (tex_size, 2 * tex_size))
    texture = paint_skin(pos_map, cov, L)

    out.update({
        "version": np.array(2), "ict_commit": np.array(ICT_COMMIT), "origin_ict_cm": origin,
        "base": base.astype(np.float32), "part": part_id,
        "skin_src": skin_src.astype(np.int32), "skin_uv": np.c_[skin_uv[:, 0] / 2, 1 - skin_uv[:, 1]].astype(np.float32),   # MuJoCo: v = 0 is the top row
        "skin_f": skin_f.astype(np.int32), "mouth_f": mouth_f.astype(np.int32),
        "moving": moving.astype(np.int32), "jaw_field": stack[names.index("jawOpen")][moving].astype(np.float16),
        "servo_names": np.array([n for n, _, _, _ in SERVO_LAYOUT]),
        "servo_cogley": np.array([c for _, c, _, _ in SERVO_LAYOUT]),
        "servo_fields": Phi.astype(np.float16), "servo_inputs": np.array(inputs),
        "servo_matrix": Theta.astype(np.float32), "servo_fit_error": fit_err.astype(np.float32),
        "move_fields": Fm.astype(np.float16),            # the "full" rig: one actuator per movement
        "w_jaw": w_jaw.astype(np.float32), "w_chest": w_chest.astype(np.float32),
        "jaw_pivot": c, "jaw_axis": axis, "jaw_open_rad": np.array(angle),
        "jaw_side_axis": side_axis, "jaw_side_left_m": np.array(side_l @ side_axis),
        "jaw_side_right_m": np.array(side_r @ side_axis),
        "jaw_fwd_axis": fwd_axis, "jaw_fwd_m": np.array(np.linalg.norm(fwd)),
        "eye_center": np.array([eye_center["R"], eye_center["L"]]),
        "eye_radius": np.array([eye_radius["R"], eye_radius["L"]]),
        "landmarks": lm_ict.astype(np.int32),
    })
    worst = np.argsort(fit_err)[::-1][:6]
    info = {"skin vertices": len(skin_src), "moving": len(moving),
            "servo fit (relative error per movement)": f"median {np.median(fit_err):.2f}, worst "
            + ", ".join(f"{inputs[i]} {fit_err[i]:.2f}" for i in worst),
            "jaw opening": f"{np.degrees(angle):.1f} deg about {np.round(axis, 3)} at {np.round(c, 4)} m "
                           f"(rigid fit error {jaw_fit_err * 1000:.2f} mm)",
            "jaw side": f"left {side_l @ side_axis * 1000:.1f} mm, right {side_r @ side_axis * 1000:.1f} mm",
            "jaw forward": f"{np.linalg.norm(fwd) * 1000:.1f} mm",
            "eyes": {s: (np.round(eye_center[s], 4).tolist(), round(eye_radius[s] * 1000, 1)) for s in "RL"}}
    return out, info, texture


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ict", type=Path, help="a clone of github.com/USC-ICT/ICT-FaceKit (else the files download)")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "tandem" / "ict-facekit")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    data, info, texture = bake(fetch(a.ict, a.cache))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **data)
    import cv2
    tex_path = a.out.with_name(a.out.stem + "_skin.png")
    cv2.imwrite(str(tex_path), texture[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"wrote {tex_path} ({tex_path.stat().st_size / 1e6:.1f} MB)")
    for k, v in info.items():
        print(f"{k}: {v}")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())

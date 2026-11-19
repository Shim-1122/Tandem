"""The simulated human's face, built like an animatronic head: a rigid skull, a jaw on a hinge, eyes on their own
joints, and servos moving a skin.

The skin is the ICT face (ICT-FaceKit, MIT license, USC Institute for Creative Technologies; baked by
scripts/bake_face.py into data/face_ict.npz and face_ict_skin.png): a sculpted human face with eyeballs, teeth,
gums and tongue.

The servo layout follows Will Cogley's DIY robot head (github.com/will-cogley/RobotHead), which is driven by the same
MediaPipe face blendshapes as Tandem: per side a middle and an outer brow servo, an upper and a lower lid, an upper
and a lower lip, an upper and a lower cheek (16), one servo turning both eyes side to side and one up and down, and
the jaw. (His tongue servo is left out; his two jaw servos drive the one hinge here.) Each skin servo is modeled at
its linkage: a slide joint for the anchor's travel, with a position actuator as the servo.

Two rigs (`rig`):
  cogley  the 16 skin servos above. Each moves its own patch of skin; the bake fitted the patches and how far each
          servo turns for each ARKit movement so the servos reproduce ICT's sculpted movements as closely as 16
          servos can (they cannot make every movement on its own, as on the real head).
  full    one actuator per ARKit movement (47): every measured movement exactly as ICT sculpted it.

How the skin moves: rigid parts ride their bones (skull, jaw, eyes; the neck blends into the chest). On top of that,
each servo moves the skin by its field, scaled by how far its anchor has travelled (a linear skin model: the
servos' effects add up), and the jaw adds what its hinge alone does not explain. The skin is computed from the
joints' actual positions, so in physics mode servo lag shows on the face.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from .face_rig import GAZE_FULL, INPUT_ALIAS

DATA = Path(__file__).resolve().parent / "data" / "face_ict.npz"
TEXTURE = DATA.with_name("face_ict_skin.png")       # skin, lips, brows, lash lines and hair, painted by the bake

SKIN = "0.80 0.60 0.48 1"                            # the bake's skin tone, for the hands
COLORS = {"skin": "1 1 1 1", "mouth": "0.36 0.12 0.13 1", "teeth_upper": "0.94 0.92 0.86 1",
          "teeth_lower": "0.94 0.92 0.86 1", "gums_upper": "0.78 0.43 0.45 1", "gums_lower": "0.78 0.43 0.45 1",
          "sclera": "0.96 0.95 0.93 1", "iris": "0.33 0.21 0.12 1", "pupil": "0.03 0.03 0.03 1"}
WIRE, ANCHOR, SERVO_BOX = "0.75 0.77 0.80 1", "0.95 0.75 0.10 1", "0.12 0.35 0.75 1"
MECH_GROUP = 3                    # geoms and tendons of the mechanism (shown in the "animatronic" view)
HIDDEN_GROUP = 4                  # mass and collision shapes under the skin
DEFORMING = ("skin", "mouth")
RIGS = ("cogley", "full")


@functools.lru_cache(maxsize=1)
def load(path: str | None = None) -> dict:
    with np.load(path or DATA, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    for k in ("servo_fields", "move_fields", "jaw_field"):
        d[k] = d[k].astype(np.float32)
    return d


def _fmt(a) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(a, dtype=float).ravel())


class Face:
    """The face's geometry, rig and live skin. Build the MJCF with `bodies` (inside the head body) and `deformable`
    (at the model's top level), then `attach` it to the compiled model."""

    def __init__(self, head_to_chest: float, rig: str = "cogley", path: str | None = None):
        if rig not in RIGS:
            raise ValueError(f"rig must be one of {RIGS}")
        self.rig = rig
        self.d = d = load(path)
        self.head_to_chest = head_to_chest       # height of the head frame above the chest frame at rest
        base, part = d["base"].astype(np.float64), d["part"]
        # skins: name -> (rest vertices in the head frame, faces, base index of each vertex or None, bones)
        self.skins: dict = {}
        self.texcoord = {"skin": d["skin_uv"]}
        wj, wc = d["w_jaw"].astype(np.float64), d["w_chest"].astype(np.float64)

        def deforming(name, ids, faces):
            self.skins[name] = (base[ids], faces, ids, {"jaw": wj[ids], "chest": wc[ids]})

        deforming("skin", d["skin_src"], d["skin_f"])
        ids = np.flatnonzero(part == 1)
        deforming("mouth", ids, np.searchsorted(ids, d["mouth_f"]))
        for name in ("teeth_upper", "gums_upper", "teeth_lower", "gums_lower"):
            bone = "jaw" if name.endswith("lower") else "head"
            v = d[f"{name}_v"].astype(np.float64)
            self.skins[name] = (v, d[f"{name}_f"], None, {bone: np.ones(len(v))})
        for side, s in (("R_", "R"), ("L_", "L")):
            for part_name in ("sclera", "iris", "pupil"):
                v = d[f"{part_name}_{s}_v"].astype(np.float64)
                self.skins[f"{side}{part_name}"] = (v, d[f"{part_name}_{s}_f"], None, {f"{side}eye": np.ones(len(v))})
        self._rig()

    # ------------------------------------------------------------------ rig geometry
    def _rig(self):
        d = self.d
        base = d["base"].astype(np.float64)
        skin = d["part"] == 0
        self.eye_center = {"R_": d["eye_center"][0], "L_": d["eye_center"][1]}
        eyes = d["eye_center"].mean(0)
        self.skull_center = np.array([eyes[0] - 0.075, 0.0, eyes[2] - 0.005])
        moving = d["moving"]
        row = np.full(len(base), -1)
        row[moving] = np.arange(len(moving))
        self.row = row
        self.inputs = [str(n) for n in d["servo_inputs"]]           # ARKit movements, in matrix row order
        if self.rig == "cogley":
            names, fields, matrix = [str(n) for n in d["servo_names"]], d["servo_fields"], d["servo_matrix"]
            self.cogley = dict(zip(names, (str(c) for c in d["servo_cogley"])))
        else:
            names, fields, matrix = list(self.inputs), d["move_fields"], np.eye(len(self.inputs), dtype=np.float32)
            self.cogley = {}
        self.matrix = matrix                                        # (movements, servos)
        self.fields = np.ascontiguousarray(fields.reshape(len(fields), -1))
        # each servo's anchor: where its field moves the skin most; the anchor slides along that motion
        self.servos = []
        for name, f in zip(names, fields):
            mag = np.where(skin[moving], np.linalg.norm(f, axis=1), 0.0)
            r = int(np.argmax(mag))
            anchor = base[moving[r]]
            inward = self.skull_center - anchor
            inward /= np.linalg.norm(inward)
            signed = bool((matrix[:, len(self.servos)] < -0.05).any())   # turns both ways (antagonist pair)
            self.servos.append({"name": name, "axis": f[r] / mag[r], "travel": float(mag[r]), "signed": signed,
                                "anchor": anchor + 0.002 * inward, "exit": anchor + 0.024 * inward,
                                "servo": anchor + 0.036 * inward})
        self.servo_index = {s["name"]: i for i, s in enumerate(self.servos)}
        self.channels = [s["name"] for s in self.servos] + ["jaw_open"]
        self.jaw = {"pivot": d["jaw_pivot"], "axis": d["jaw_axis"], "open": float(d["jaw_open_rad"])}
        self.jaw_field = d["jaw_field"].reshape(-1)
        self._gaze_rows = {f"eyeLook{k}{s}": self.inputs.index(f"eyeLook{k}{s}")
                           for k in ("Up", "Down", "In", "Out") for s in ("Left", "Right")}

    def servo_values(self, bs: dict, eye_yaw: float = 0.0, eye_pitch: float = 0.0) -> dict:
        """Servo positions, -1..1 (the jaw 0..1), from blendshapes (ARKit names, 0..1) and the eye angles in degrees,
        which make the lids follow the eyes. Like the real head: each servo is the sum of its share of every
        movement."""
        w = np.array([float(bs.get(n, bs.get(INPUT_ALIAS.get(n, ""), 0.0))) for n in self.inputs])
        yaw, pitch = eye_yaw / GAZE_FULL[0], eye_pitch / GAZE_FULL[1]
        gaze = {"eyeLookUpLeft": pitch, "eyeLookUpRight": pitch, "eyeLookDownLeft": -pitch,
                "eyeLookDownRight": -pitch, "eyeLookInRight": yaw, "eyeLookOutRight": -yaw,
                "eyeLookOutLeft": yaw, "eyeLookInLeft": -yaw}               # + yaw turns toward the person's left
        for n, v in gaze.items():
            w[self._gaze_rows[n]] = min(1.0, max(0.0, v))
        a = np.clip(w @ self.matrix, -1.0, 1.0)
        out = {s["name"]: float(x) if s["signed"] else max(0.0, float(x)) for s, x in zip(self.servos, a)}
        jaw = float(bs.get("jawOpen", 0.0))
        out["jaw_open"] = min(1.0, max(0.0, jaw))
        return out

    def commands(self, values: dict) -> dict:
        """Servo positions by the channel names of Will Cogley's head (cogley rig), for driving a real one."""
        return {self.cogley.get(n, n): v for n, v in values.items() if n in self.cogley or n == "jaw_open"}

    # ------------------------------------------------------------------ MJCF
    def bodies(self, m, eye_ranges: dict):
        """Inside the head body: skull mass, eyes, jaw, and the wires with their servos. `m` is the human's MJCF
        builder (its joint and slide add joints with their servos)."""
        c = self.skull_center + [0.025, 0, 0.02]
        m.add(f'<geom type="ellipsoid" pos="{_fmt(c)}" size="0.085 0.072 0.1" group="{HIDDEN_GROUP}" '
              f'rgba="0.8 0.8 0.8 0.3" class="robot" density="1050"/>')
        m.add('<camera name="eyes" pos="0.1 0 0.13" xyaxes="0 -1 0 0 0 1" fovy="70"/>')
        for p in ("R_", "L_"):
            m.open(f'<body name="{p}eye" pos="{_fmt(self.eye_center[p])}">')
            m.joint(f"{p}eye_yaw", [0, 0, 1], eye_ranges["eye_yaw"], "eye")
            m.joint(f"{p}eye_pitch", [0, -1, 0], eye_ranges["eye_pitch"], "eye")
            m.add(f'<geom type="sphere" size="0.012" group="{HIDDEN_GROUP}" contype="0" conaffinity="0" density="1000"/>')
            m.close("body")
        j = self.jaw
        m.open(f'<body name="jaw" pos="{_fmt(j["pivot"])}">')
        m.joint("jaw_open", j["axis"], (-2.0, np.degrees(j["open"]) * 1.15), "jaw")
        m.add(f'<geom type="ellipsoid" pos="0.07 0 -0.045" size="0.04 0.05 0.022" group="{HIDDEN_GROUP}" '
              'contype="0" conaffinity="0" density="1050"/>')
        m.close("body")
        for w in self.servos:
            n = w["name"]
            m.add(f'<site name="exit_{n}" pos="{_fmt(w["exit"])}" size="0.0015" group="{MECH_GROUP}" rgba="{WIRE}"/>')
            m.add(f'<geom type="box" pos="{_fmt(w["servo"])}" size="0.006 0.0045 0.0035" group="{MECH_GROUP}" '
                  f'rgba="{SERVO_BOX}" contype="0" conaffinity="0" density="0" mass="0.002"/>')
            m.open(f'<body name="anchor_{n}" pos="{_fmt(w["anchor"])}">')
            m.slide(n, w["axis"], (-w["travel"] * 1.05 if w["signed"] else -0.0005, w["travel"] * 1.05))
            m.add(f'<geom type="sphere" size="0.0018" group="{MECH_GROUP}" rgba="{ANCHOR}" contype="0" '
                  'conaffinity="0" mass="0.002"/>')
            m.add(f'<site name="anchor_{n}" size="0.001" group="{MECH_GROUP}"/>')
            m.close("body")

    def tendons(self, m):
        """The linkage from each servo to its anchor."""
        for w in self.servos:
            n = w["name"]
            m.open(f'<spatial name="link_{n}" width="0.0005" rgba="{WIRE}" group="{MECH_GROUP}" limited="false">')
            m.add(f'<site site="exit_{n}"/>')
            m.add(f'<site site="anchor_{n}"/>')
            m.close("spatial")

    def deformable(self, m, jaw_bind: np.ndarray):
        """<skin> elements: vertices in the head frame, bound to the head, jaw, chest and eye bodies."""
        bind = {"head": np.zeros(3), "jaw": np.asarray(jaw_bind, float), "chest": np.array([0, 0, -self.head_to_chest]),
                "R_eye": self.eye_center["R_"], "L_eye": self.eye_center["L_"]}
        m.open("<deformable>")
        for name, (rest, faces, _ids, weights) in self.skins.items():
            color = COLORS.get(name, COLORS.get(name[2:], SKIN))
            w = dict(weights)
            if name in DEFORMING:
                w["head"] = 1.0 - w["jaw"] - w["chest"]
            mat = "face_skin" if name == "skin" else "face_wet"
            uv = f' texcoord="{_fmt(self.texcoord[name])}"' if name in self.texcoord else ""
            m.open(f'<skin name="{name}" rgba="{color}" material="{mat}" inflate="0"{uv} '
                   f'vertex="{_fmt(rest)}" face="{" ".join(map(str, np.asarray(faces).ravel()))}">')
            for bone, wv in w.items():
                ids = np.flatnonzero(wv > 1e-4)
                if len(ids) == 0:
                    continue
                m.add(f'<bone body="{bone}" bindpos="{_fmt(bind[bone])}" bindquat="1 0 0 0" '
                      f'vertid="{" ".join(map(str, ids))}" vertweight="{_fmt(wv[ids])}"/>')
            m.close("skin")
        m.close("deformable")

    @staticmethod
    def materials(m):
        m.add(f'<texture name="face_skin" type="2d" file="{TEXTURE.as_posix()}"/>')
        m.add('<material name="face_skin" texture="face_skin" specular="0.18" shininess="0.3" reflectance="0"/>')
        m.add('<material name="face_wet" specular="0.6" shininess="0.8" reflectance="0"/>')

    # ------------------------------------------------------------------ live skin
    def attach(self, model):
        """Index the compiled model's skins and joints for `update`."""
        import mujoco
        self.model = model
        tgt, src, rest = [], [], []
        for name in DEFORMING:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SKIN, name)
            adr = int(model.skin_vertadr[sid])
            r, _, ids, _ = self.skins[name]
            rows = self.row[ids]
            keep = np.flatnonzero(rows >= 0)
            tgt.append(adr + keep)
            src.append(rows[keep])
            rest.append(r[keep])
        self._tgt = np.concatenate(tgt)
        self._src = np.concatenate(src)
        self._rest = np.concatenate(rest).astype(np.float32)
        self._qadr = {n: int(model.jnt_qposadr[model.joint(n).id]) for n in self.channels}
        self._travel = np.array([w["travel"] for w in self.servos])
        self._q_ids = np.array([self._qadr[w["name"]] for w in self.servos])
        self.activations = np.zeros(len(self.servos) + 1)

    def target(self, name: str, a: float) -> float:
        """Joint position (meters, or radians for the jaw hinge) for a servo value a in -1..1."""
        if name == "jaw_open":
            return a * self.jaw["open"]
        return a * self.servos[self.servo_index[name]]["travel"]

    def field_activations(self, qpos: np.ndarray) -> np.ndarray:
        """How far each servo is turned (-1..1), and last the jaw (0..1), from the joints' actual positions."""
        act = np.empty(len(self.servos) + 1)
        act[:-1] = qpos[self._q_ids] / self._travel
        act[-1] = max(0.0, qpos[self._qadr["jaw_open"]] / self.jaw["open"])
        return act

    def update(self, qpos: np.ndarray):
        """Move the skin to match the joints (well under a millisecond)."""
        act = self.field_activations(qpos)
        self.activations = act
        nz = np.flatnonzero(np.abs(act[:-1]) > 1e-6)
        disp = act[-1] * self.jaw_field
        if len(nz):
            disp = disp + act[nz].astype(np.float32) @ self.fields[nz]
        self.model.skin_vert[self._tgt] = self._rest + disp.reshape(-1, 3)[self._src]

    def skin_points(self, model, data, name: str = "skin") -> np.ndarray:
        """World positions of a skin's vertices as rendered (bones applied), for checks."""
        rest, _, _, weights = self.skins[name]
        import mujoco
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SKIN, name)
        adr = int(model.skin_vertadr[sid])
        v = model.skin_vert[adr:adr + len(rest)].astype(np.float64)
        out = np.zeros_like(v)
        bind = {"head": np.zeros(3), "chest": np.array([0, 0, -self.head_to_chest])}
        bind["jaw"] = model.body("jaw").pos.copy()
        bind["R_eye"], bind["L_eye"] = self.eye_center["R_"], self.eye_center["L_"]
        w = dict(weights)
        w.setdefault("head", np.zeros(len(rest)))
        if name in DEFORMING:
            w["head"] = 1.0 - w.get("jaw", 0) - w.get("chest", 0)
        for bone, wv in w.items():
            b = data.body(bone)
            out += wv[:, None] * ((v - bind[bone]) @ b.xmat.reshape(3, 3).T + b.xpos)
        return out

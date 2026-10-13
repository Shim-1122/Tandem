"""Grip force for each tracked object, as a target for the planned Homunculus hand.

The arm is not built yet, so these are engineering estimates from physics, to be checked with a force sensor on the
real hand. Every number is in OBJECTS below, with where it came from, so it can be edited.

Model. An object held between opposing contacts stays put while friction carries its weight plus the extra force
of moving it. With a total squeeze (grip force) F spread over contacts on opposite sides, the friction available is
n_eff * mu * F, so the least squeeze that holds it is

    F_slip = m (g + a) / (n_eff mu)

where n_eff is 2 for fingertip grasps (two opposing pads) and 3 for whole-hand wraps (pressure all around the
object), mu is the friction coefficient between the fingertip pads and the object, and a is the acceleration of
moving it. A long object held away from its center of mass (a racket, a knife) also tips about the grip with torque
m (g + a) * lever, which the hand resists with a force couple across its width:

    F_torque = m (g + a) lever / couple

The target is safety * (F_slip + F_torque), raised to a floor the hand can control and limited to what the object
tolerates (crush_n). The 0..1 command is the target divided by the hand's maximum grip force (arm_max_n).

Default margins: people hold still objects with 40 to 50% of their grip force above the slip force (80% for very
light ones), that is about 1.7 to 2 times the slip force (Hiramatsu et al. 2015); a robot with less touch feedback
should not use less, so safety = 2 (3 for knives). Friction: silicone fingertip pads on smooth plastic measured
0.3 to 1.7, falling as the squeeze rises (Leddy et al. 2020), so smooth surfaces use 0.4 to 0.5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

G = 9.81
GRASP_CONTACTS = {"pinch": 2, "tripod": 2, "lateral": 2, "power": 3, "spherical": 3}
GRASP_TEXT = {"pinch": "fingertip pinch", "tripod": "tripod (pen grip)", "lateral": "lateral pinch",
              "power": "power wrap", "spherical": "spherical"}

SOURCES = {
    "hiramatsu2015": "Hiramatsu Y, Kimura D, Kadota K, Ito T, Kinoshita H (2015) Control of precision grip force in "
                     "lifting and holding of low-mass objects. PLOS ONE 10(9): e0138506",
    "leddy2020": "Leddy MT, Dollar AM (2020) Examining the frictional behavior of primitive contact geometries for "
                 "use as robotic finger pads. IEEE Robotics and Automation Letters (silicone pads on acrylic: "
                 "0.3 to 1.7, lower at higher squeeze)",
    "itf": "ITF Rules of Tennis, Appendix I: ball mass 56.0 to 59.4 g, diameter 6.54 to 6.86 cm",
    "ittf": "ITTF: ball 2.7 g (2.67 to 2.77 g), 40 mm; hardness test with a 50 N load indents 0.68 to 0.90 mm",
    "laws_of_the_game": "IFAB Laws of the Game, Law 2: size 5 ball 68 to 70 cm around, 410 to 450 g, 0.6 to 1.1 bar",
    "fiba": "FIBA size 7 ball: 750 to 770 mm around, 580 to 620 g (NBA ball about 22 oz, 624 g)",
    "nfl": "NFL Rule 2: ball 14 to 15 oz (400 to 430 g), 12.5 to 13.5 psi",
    "typical": "typical retail values (not regulated): weigh yours and edit the table",
}


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    mass_kg: tuple                  # (typical, heaviest); the heaviest is used for the force
    width_m: float                  # size across the grasp (the hand opens to this plus clearance)
    grasp: str                      # pinch, tripod, lateral, power, spherical
    mu: float                       # friction with rubber-like fingertip pads (conservative)
    crush_n: float = math.inf       # squeezing harder damages, dents or presses buttons on it
    lever_m: float = 0.0            # grip to center of mass, held level
    couple_m: float = 0.08          # span of the hand that resists tipping (whole hand 8 cm, fingertips 3 cm)
    safety: float = 2.0
    sources: tuple = ("typical",)
    notes: str = ""


OBJECTS = {o.name: o for o in [
    ObjectSpec("tennis ball", (0.057, 0.0594), 0.067, "spherical", 0.6, sources=("itf",),
               notes="felt over a rubber core: it squashes elastically, hand forces do not damage it"),
    ObjectSpec("ping pong ball", (0.0027, 0.0027), 0.040, "pinch", 0.4, crush_n=20.0, sources=("ittf",),
               notes="weighs almost nothing: the floor sets the force; the 50 N hardness test load only dents it "
                     "elastically, 20 N keeps a wide margin for sharp fingertip edges"),
    ObjectSpec("soccer ball", (0.43, 0.45), 0.22, "spherical", 0.6, sources=("laws_of_the_game",),
               notes="too large to wrap: fingers spread over it (palming), or hold it from below"),
    ObjectSpec("basketball", (0.60, 0.65), 0.24, "spherical", 0.7, sources=("fiba",),
               notes="pebbled surface grips well; too large to wrap, spread the fingers or hold from below"),
    ObjectSpec("american football", (0.41, 0.43), 0.17, "power", 0.7, sources=("nfl",),
               notes="fingers across the laces; 0.17 m is the thick middle"),
    ObjectSpec("tennis racket", (0.30, 0.34), 0.033, "power", 0.7, lever_m=0.26, couple_m=0.08,
               notes="held level by the handle, so tipping dominates; held head up it needs about 4 N; "
                     "hitting a ball takes far more than holding"),
    ObjectSpec("ping pong paddle", (0.17, 0.20), 0.030, "power", 0.5, lever_m=0.08, couple_m=0.06,
               notes="shakehand grip: index finger on the blade, the rest around the handle"),
    ObjectSpec("cell phone", (0.19, 0.24), 0.075, "power", 0.4, crush_n=30.0,
               notes="squeeze across the front and back, or on the edges away from the side buttons (pressing "
                     "a button takes only a few newtons)"),
    ObjectSpec("pen", (0.006, 0.02), 0.010, "tripod", 0.4, crush_n=30.0,
               notes="floor sets the force; writing adds the drag of the tip on the paper"),
    ObjectSpec("pencil", (0.005, 0.008), 0.007, "tripod", 0.5, crush_n=40.0,
               notes="floor sets the force; writing adds the drag of the tip on the paper"),
    ObjectSpec("ruler", (0.025, 0.07), 0.003, "lateral", 0.4, crush_n=30.0, lever_m=0.03, couple_m=0.03,
               notes="pick it up near the middle; held level by one end it needs about 3 times more"),
    ObjectSpec("eraser", (0.015, 0.03), 0.020, "pinch", 0.8, crush_n=15.0,
               notes="soft rubber: more squeeze only deforms it"),
    ObjectSpec("knife", (0.10, 0.25), 0.025, "power", 0.5, lever_m=0.04, couple_m=0.08, safety=3.0,
               notes="by the handle only, blade away from people; extra safety because dropping it is dangerous; "
                     "cutting needs more than holding"),
    ObjectSpec("stuffed animal", (0.20, 0.50), 0.15, "power", 0.6, crush_n=15.0,
               notes="soft: squeezing harder only flattens the stuffing"),
    ObjectSpec("plastic water bottle", (0.52, 0.55), 0.065, "power", 0.5, crush_n=15.0,
               notes="500 ml, full; thin walls: an open or half-empty bottle dents and spills when squeezed"),
    ObjectSpec("metal water bottle", (0.90, 1.20), 0.075, "power", 0.4,
               notes="insulated steel, 0.7 to 1 litre, full; condensation makes it slippery"),
]}


@dataclass
class GripPlan:
    name: str
    grasp: str
    force_n: float                  # target squeeze
    command: float                  # force_n / arm maximum, 0..1
    slip_n: float                   # least squeeze that holds it (no safety margin)
    max_n: float                    # do not exceed (crush limit, or the hand's maximum)
    open_m: float                   # open the hand this wide before closing
    limited: bool                   # the target was cut to the crush limit or the hand's maximum
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        s = f"{self.name}: {GRASP_TEXT[self.grasp]}, grip {self.force_n:.1f} N ({self.command:.2f})"
        return s + (" !" if self.limited else "")


@dataclass
class GripModel:
    arm_max_n: float = 50.0         # the hand's strongest grip; measure it once the hand is built
    accel: float = 3.0              # m/s^2 of arm motion to allow for (0.3 g)
    floor_n: float = 1.0            # least squeeze the hand controls reliably
    clearance_m: float = 0.02       # extra opening before closing on an object

    def plan(self, spec: ObjectSpec) -> GripPlan:
        load = spec.mass_kg[1] * (G + self.accel)
        slip = load / (GRASP_CONTACTS[spec.grasp] * spec.mu)
        torque = load * spec.lever_m / spec.couple_m if spec.lever_m > 0 else 0.0
        need = spec.safety * (slip + torque)
        target = max(need, self.floor_n)
        limit = min(spec.crush_n, self.arm_max_n)
        warnings = []
        if target > spec.crush_n:
            warnings.append(f"needs {need:.1f} N but it tolerates {spec.crush_n:.0f} N: move slowly or support it from below")
        if target > self.arm_max_n:
            warnings.append(f"needs {target:.1f} N, more than the hand's {self.arm_max_n:.0f} N")
        limited = target > limit
        force = min(target, limit)
        return GripPlan(spec.name, spec.grasp, force, min(1.0, force / self.arm_max_n), slip + torque, limit,
                        spec.width_m + self.clearance_m, limited, warnings)

    def plan_for(self, name: str) -> GripPlan:
        return self.plan(OBJECTS[name])

    def uncertain_bottle(self) -> GripPlan:
        """A water bottle of unknown material: the metal bottle's force, held to the plastic bottle's limit."""
        metal, plastic = self.plan_for("metal water bottle"), self.plan_for("plastic water bottle")
        force = min(metal.force_n, plastic.max_n)
        warnings = ["material unsure: limited for plastic, a full metal bottle may slip"] if force < metal.force_n else []
        return GripPlan("water bottle", "power", force, min(1.0, force / self.arm_max_n), metal.slip_n, plastic.max_n,
                        max(metal.open_m, plastic.open_m), force < metal.force_n, warnings)

    def table(self) -> list:
        return [self.plan(s) for s in OBJECTS.values()]

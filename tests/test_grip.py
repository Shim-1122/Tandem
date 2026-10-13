"""Grip force model: physics, limits, and a complete table for every requested object."""
import math

import pytest

from vtrack.grip import G, GRASP_CONTACTS, OBJECTS, SOURCES, GripModel, ObjectSpec

REQUESTED = ["tennis ball", "ping pong ball", "soccer ball", "basketball", "american football", "tennis racket",
             "ping pong paddle", "cell phone", "pen", "pencil", "ruler", "eraser", "knife", "stuffed animal",
             "metal water bottle", "plastic water bottle"]


def test_every_requested_object_has_a_complete_entry():
    assert set(REQUESTED) <= set(OBJECTS)
    for o in OBJECTS.values():
        assert o.grasp in GRASP_CONTACTS
        assert 0 < o.mass_kg[0] <= o.mass_kg[1] < 2.0
        assert 0.2 <= o.mu <= 1.0 and o.width_m > 0 and o.crush_n > 0 and o.safety >= 2.0
        assert all(s in SOURCES for s in o.sources)


def test_force_follows_the_friction_and_tipping_formula():
    m = GripModel(accel=3.0, floor_n=0.0)
    r = OBJECTS["tennis racket"]
    load = r.mass_kg[1] * (G + 3.0)
    expect = r.safety * (load / (3 * r.mu) + load * r.lever_m / r.couple_m)
    p = m.plan(r)
    assert p.force_n == pytest.approx(expect) and p.slip_n == pytest.approx(expect / r.safety)
    assert p.command == pytest.approx(expect / m.arm_max_n) and p.grasp == "power"
    assert p.open_m == pytest.approx(r.width_m + m.clearance_m)


def test_heavier_needs_more_and_grippier_needs_less():
    m = GripModel(floor_n=0.0)
    a = ObjectSpec("a", (0.2, 0.2), 0.05, "pinch", 0.5)
    assert m.plan(ObjectSpec("b", (0.4, 0.4), 0.05, "pinch", 0.5)).force_n == pytest.approx(2 * m.plan(a).force_n)
    assert m.plan(ObjectSpec("c", (0.2, 0.2), 0.05, "pinch", 1.0)).force_n == pytest.approx(m.plan(a).force_n / 2)
    assert m.plan(ObjectSpec("d", (0.2, 0.2), 0.05, "power", 0.5)).force_n == pytest.approx(m.plan(a).force_n * 2 / 3)
    faster = GripModel(floor_n=0.0, accel=6.0)
    assert faster.plan(a).force_n == pytest.approx(m.plan(a).force_n * (G + 6.0) / (G + 3.0))


def test_light_objects_get_the_floor_and_fragile_ones_are_capped():
    m = GripModel(floor_n=1.0)
    ball = m.plan_for("ping pong ball")
    assert ball.slip_n < 0.1 and ball.force_n == pytest.approx(1.0) and not ball.limited
    heavy = ObjectSpec("full jug", (2.0, 2.0), 0.1, "power", 0.4, crush_n=15.0)
    p = m.plan(heavy)
    assert p.force_n == pytest.approx(15.0) and p.limited and p.max_n == 15.0
    assert "tolerates 15 N" in p.warnings[0]


def test_command_never_exceeds_one_and_reports_a_weak_hand():
    weak = GripModel(arm_max_n=10.0)
    p = weak.plan_for("metal water bottle")
    assert p.command == pytest.approx(1.0) and p.force_n == pytest.approx(10.0) and p.limited
    assert any("hand's 10 N" in w for w in p.warnings)


def test_unknown_bottle_material_is_held_to_the_plastic_limit():
    m = GripModel()
    metal, plastic, unsure = m.plan_for("metal water bottle"), m.plan_for("plastic water bottle"), m.uncertain_bottle()
    assert plastic.force_n < unsure.force_n <= plastic.max_n < metal.force_n
    assert unsure.limited and "material unsure" in unsure.warnings[0]


def test_table_values_are_plausible():
    """Every target is within what a hand can do and inside the object's limit; the table is printed in the README."""
    for p in GripModel().table():
        assert 1.0 <= p.force_n <= min(p.max_n, 50.0) and 0 < p.command <= 1.0
        assert math.isfinite(p.slip_n) and p.summary().startswith(p.name)
    assert GripModel().plan_for("tennis racket").force_n > GripModel().plan_for("ping pong paddle").force_n

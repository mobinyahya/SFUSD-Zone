from __future__ import annotations

import pytest

from optimization.data.mid import MidProgram, MidStudent
from optimization.welfare_bounds import (
    first_choice_upper_bound,
    transport_upper_bound,
    welfare_upper_bound,
)


def student(node: int, programs, utilities) -> MidStudent:
    return MidStudent(
        node=node,
        programs=tuple(programs),
        priorities=tuple(1 for _ in programs),
        utilities=tuple(float(u) for u in utilities),
        scaled_utilities=tuple(int(u * 100) for u in utilities),
    )


def test_capacity_makes_the_transport_bound_strictly_tighter():
    """Three students chasing two single-seat programs cannot all be served."""
    programs = (
        MidProgram("A", 1, 1, False, None),
        MidProgram("B", 2, 1, False, None),
    )
    students = tuple(student(i, ("A", "B"), (10.0, 5.0)) for i in range(3))

    assert first_choice_upper_bound(students) == pytest.approx(30.0)
    # Best capacity-feasible assignment: one seat at A, one at B, one student out.
    assert transport_upper_bound(programs, students) == pytest.approx(15.0)


def test_the_bounds_agree_when_capacity_never_binds():
    """With a seat for everyone, respecting capacity costs nothing."""
    programs = (MidProgram("A", 1, 5, False, None),)
    students = tuple(student(i, ("A",), (7.0,)) for i in range(3))

    assert transport_upper_bound(programs, students) == pytest.approx(
        first_choice_upper_bound(students)
    )


def test_students_with_no_programs_contribute_nothing():
    programs = (MidProgram("A", 1, 1, False, None),)
    students = (student(0, ("A",), (4.0,)), student(1, (), ()))

    assert first_choice_upper_bound(students) == pytest.approx(4.0)
    assert transport_upper_bound(programs, students) == pytest.approx(4.0)


def test_unknown_bound_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown welfare bound"):
        welfare_upper_bound("wishful", (), ())


def test_transport_is_the_default_welfare_bound():
    """Pinned deliberately: the transport constant is weakly dominant.

    The reported SAA bound is ``min(constant, what the cuts prove)`` and the
    transport constant is never above the first-choice sum, so defaulting to it
    can only tighten the reported bound. Changing this should be a conscious
    decision, not a side effect.
    """
    from optimization.config import OptimizationConfig

    config = OptimizationConfig()
    assert config.saa_welfare_bound == "transport"
    # The strategy's own fallback must agree, or a config-free construction
    # would silently use the looser constant.
    assert config.make_strategy().options["saa_welfare_bound"] == "transport"

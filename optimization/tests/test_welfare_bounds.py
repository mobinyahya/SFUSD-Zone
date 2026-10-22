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


def test_the_zoned_neighbors_bound_is_the_default():
    """Pinned deliberately: it is the tightest and cheapest of the four.

    The reported SAA bound is ``min(constant, what the cuts prove)``, and the
    chain ``first_choice >= transport >= zoned_transport`` means the zoned
    constant can only tighten it. ``neighbors`` over ``flow`` because it
    measured both tighter and faster, and because it relaxes the very rows the
    master carries. Changing this should be a conscious decision, not a side
    effect.
    """
    from optimization.config import OptimizationConfig

    config = OptimizationConfig()
    assert config.saa_welfare_bound == "zoned_transport_neighbors"
    # The strategy's own fallback must agree, or a config-free construction
    # would silently use a looser constant.
    options = config.make_strategy().options
    assert options["saa_welfare_bound"] == "zoned_transport_neighbors"


def test_the_zoned_transport_lp_runs_single_threaded_by_default():
    """Gurobi's concurrent LP only contends on a model this size.

    Measured on Block_2: every thread count above one was slower, ``flow``
    worst at 2.4s to 4.4s between 1 and 8 threads. This is deliberately a
    separate knob from the master's ``workers``.
    """
    from optimization.config import OptimizationConfig

    config = OptimizationConfig(workers=32)
    assert config.zoned_transport_workers == 1
    assert config.make_strategy().options["zoned_transport_workers"] == 1

    with pytest.raises(ValueError, match="zoned_transport_workers"):
        OptimizationConfig(zoned_transport_workers=0)

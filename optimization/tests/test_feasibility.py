"""Tests for the solver-independent feasibility validator."""

from __future__ import annotations

import pytest

from optimization.data.feasibility import (
    InfeasibleZoningError,
    assert_feasible,
    check_zoning,
)
from optimization.data.initial_solutions import initial_solution
from optimization.tests.synthetic import make_grid_problem


def _feasible_grid():
    problem = make_grid_problem(3, 3, boundary_prop=0.5)
    hint = initial_solution(
        problem,
        "feasible",
        solver_options={"feasible_hint_time_limit": 10, "seed": 3},
    )
    return problem, hint.assignment


def test_accepts_a_solver_feasible_zoning():
    problem, assignment = _feasible_grid()

    report = check_zoning(problem, assignment)

    assert report.feasible
    assert report.describe() == "feasible"


def test_flags_an_unassigned_node():
    problem, assignment = _feasible_grid()
    partial = {
        node: zone for node, zone in assignment.items() if node != problem.nodes[-1]
    }

    report = check_zoning(problem, partial)

    assert not report.feasible
    assert [violation.kind for violation in report.violations] == ["coverage"]


def test_flags_a_node_outside_its_candidate_zones():
    problem, assignment = _feasible_grid()
    centroid = problem.centroids[0]
    # A centroid's only candidate is its own zone, so moving it is forbidden by
    # construction rather than by any numeric bound.
    misplaced = {**assignment, centroid: 1}

    report = check_zoning(problem, misplaced)

    assert "candidate" in [violation.kind for violation in report.violations]


def test_flags_a_disconnected_zone():
    """3x3 grid, centroids at the corners: zone 0 gets two opposite cells."""

    problem, _ = _feasible_grid()
    split = {node: 1 for node in problem.nodes}
    split[0] = 0
    split[2] = 0

    report = check_zoning(problem, split)

    assert [violation.kind for violation in report.violations] == ["contiguity"]


def test_flags_a_balance_violation_a_scaled_model_would_miss():
    """The exact check is what a 1/100-granular model cannot promise."""

    problem = make_grid_problem(3, 3, boundary_prop=0.5, frl_dev=0.0)
    _, assignment = _feasible_grid()
    # Uniform grid data satisfies the FRL band exactly; a single fractional
    # nudge is well below the CP-SAT coefficient scale but well above 1e-6.
    problem.G.nodes[4]["FRL"] = 0.5 + 0.004

    report = check_zoning(problem, assignment)

    kinds = [violation.kind for violation in report.violations]
    assert "frl_upper" in kinds or "frl_lower" in kinds
    assert max(violation.amount for violation in report.violations) < 0.01


def test_tolerance_can_absorb_a_violation():
    problem = make_grid_problem(3, 3, boundary_prop=0.5, frl_dev=0.0)
    _, assignment = _feasible_grid()
    problem.G.nodes[4]["FRL"] = 0.5 + 0.004

    assert check_zoning(problem, assignment, tolerance=0.01).feasible


def test_flags_a_boundary_violation():
    problem, assignment = _feasible_grid()
    # 12 edges, so a proportion of 0 caps the cut at zero edges.
    tight = make_grid_problem(3, 3, boundary_prop=0.0)

    report = check_zoning(tight, assignment)

    assert "boundary" in [violation.kind for violation in report.violations]


def test_assert_feasible_raises_with_the_reason():
    problem, assignment = _feasible_grid()
    partial = {
        node: zone for node, zone in assignment.items() if node != problem.nodes[-1]
    }

    with pytest.raises(InfeasibleZoningError, match="coverage") as excinfo:
        assert_feasible(problem, partial, label="cached hint")

    assert "cached hint" in str(excinfo.value)
    assert not excinfo.value.report.feasible


def test_rejects_a_negative_tolerance():
    problem, assignment = _feasible_grid()

    with pytest.raises(ValueError, match="non-negative"):
        check_zoning(problem, assignment, tolerance=-1.0)

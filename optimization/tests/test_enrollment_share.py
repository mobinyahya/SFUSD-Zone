"""The capacity rows count enrolling students; every other row counts students."""

from __future__ import annotations

import pytest

from optimization.data.feasibility import check_zoning
from optimization.data.initial_solutions import feasibility_fingerprint
from optimization.solvers import get_solver
from optimization.solvers.balance import balance_constraints, balance_terms
from optimization.solvers.recom import _ReComContext
from optimization.tests.synthetic import make_grid_problem

ASSIGNMENT = [0, 0, 1, 1]


def _crowded_problem(enrolling: float | None):
    """2x2 grid whose node 0 holds 100 students against one seat.

    With every student counted, zone {0, 1} needs 101 seats and holds 2; when
    node 0's students enrol at ``enrolling``, it needs ``enrolling + 1``.
    """
    problem = make_grid_problem(2, 2, shortage=0.0, overage=0.0)
    for node in problem.nodes:
        attrs = problem.G.nodes[node]
        attrs["ge_enrolling_students"] = attrs["ge_students"]
    problem.G.nodes[0]["ge_students"] = 100.0
    problem.G.nodes[0]["FRL"] = 50.0
    problem.G.nodes[0]["ge_enrolling_students"] = (
        100.0 if enrolling is None else enrolling
    )
    return problem


def _zoning(problem):
    return dict(zip(problem.nodes, ASSIGNMENT, strict=True))


def test_capacity_rows_divide_by_enrolling_students():
    problem = _crowded_problem(enrolling=1.0)
    capacity, *others = balance_constraints(problem)

    assert capacity.kind == "capacity"
    lower, upper = balance_terms(problem, capacity, 0, [0])
    assert lower == [(1.0 - 1.0, 0, 0)]
    assert upper == [(1.0 - 1.0, 0, 0)]
    # FRL and racial rows still take their ratios against raw students.
    frl = next(constraint for constraint in others if constraint.kind == "frl")
    assert frl.students(problem, 0) == 100.0


def test_validator_and_recom_use_enrolling_students():
    crowded = _crowded_problem(enrolling=None)
    report = check_zoning(crowded, _zoning(crowded))
    assert "capacity_lower" in {violation.kind for violation in report.violations}
    assert not _ReComContext(crowded).build_state(ASSIGNMENT).feasible

    scaled = _crowded_problem(enrolling=1.0)
    assert check_zoning(scaled, _zoning(scaled)).feasible
    assert _ReComContext(scaled).build_state(ASSIGNMENT).feasible


@pytest.mark.parametrize("solver_name", ["cp_int", "cp_bool"])
def test_cp_sat_finds_the_zoning_only_enrolling_counts_allow(solver_name):
    def solve(problem):
        return get_solver(solver_name, solve_time_limit=10, workers=1).solve(problem)

    assert solve(_crowded_problem(enrolling=None)).status == "INFEASIBLE"

    scaled = _crowded_problem(enrolling=1.0)
    solution = solve(scaled)
    assert solution.feasible, solution.status
    assert check_zoning(scaled, solution.assignment).feasible


def test_fingerprint_changes_only_when_enrolling_differs():
    plain = make_grid_problem(2, 2)
    baseline = feasibility_fingerprint(plain)
    for node in plain.nodes:
        attrs = plain.G.nodes[node]
        attrs["ge_enrolling_students"] = attrs["ge_students"]
    assert feasibility_fingerprint(plain) == baseline

    plain.G.nodes[0]["ge_enrolling_students"] = 0.5
    assert feasibility_fingerprint(plain) != baseline

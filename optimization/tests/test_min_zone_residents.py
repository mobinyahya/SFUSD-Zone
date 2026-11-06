"""The per-zone resident floor, ``min_zone_residents``."""

from __future__ import annotations

import pytest

from optimization.data.feasibility import check_zoning
from optimization.data.initial_solutions import feasibility_fingerprint
from optimization.objectives import solve_objective
from optimization.pipeline import checked_constraints
from optimization.solvers.recom import _ReComContext
from optimization.tests.synthetic import (
    make_grid_problem,
    make_solver_contract_problem,
)

EVEN = {0: 0, 1: 0, 2: 1, 3: 1}


def _residents(problem, assignment) -> dict[int, float]:
    totals: dict[int, float] = {}
    for node, zone in assignment.items():
        totals[zone] = totals.get(zone, 0.0) + problem.residents(node)
    return totals


def test_check_zoning_reports_zones_below_the_floor() -> None:
    off = make_solver_contract_problem(max_distance=float("inf"))
    assert off.resident_floor() is None
    assert check_zoning(off, EVEN).feasible

    met = make_solver_contract_problem(max_distance=float("inf"), min_zone_residents=2)
    assert check_zoning(met, EVEN).feasible

    missed = make_solver_contract_problem(
        max_distance=float("inf"), min_zone_residents=3
    )
    kinds = {v.kind for v in check_zoning(missed, EVEN).violations}
    assert kinds == {"residents_lower"}


def test_residents_count_every_program_whatever_the_population() -> None:
    problem = make_solver_contract_problem(program_population="GE")
    for node in problem.nodes:
        problem.G.nodes[node]["ge_students"] = 0.25
        problem.G.nodes[node]["all_prog_students"] = 2.0
    assert problem.students(0) == 0.25
    assert problem.residents(0) == 2.0


def test_recom_carries_the_floor_as_a_violation_row() -> None:
    problem = make_solver_contract_problem(min_zone_residents=3)
    context = _ReComContext(problem)
    assert context.violation_labels()[-1] == "residents_lower"

    state = context.build_state(context.validate_hint(EVEN))
    assert not state.feasible
    # Each zone holds two residents against a floor of three: one third short.
    for violations in state.zone_violations:
        assert violations[-1] == pytest.approx(100.0 / 3.0)


def test_the_floor_keys_the_feasible_hint_cache_only_when_on() -> None:
    off = make_solver_contract_problem()
    explicit_off = make_solver_contract_problem(min_zone_residents=-1)
    on = make_solver_contract_problem(min_zone_residents=2)
    assert feasibility_fingerprint(off) == feasibility_fingerprint(explicit_off)
    assert feasibility_fingerprint(off) != feasibility_fingerprint(on)


def test_min_zone_residents_is_a_step_constraint() -> None:
    assert checked_constraints({"min_zone_residents": 20}) == {"min_zone_residents": 20}


@pytest.mark.parametrize("method", ["cp_bool", "cp_int", "mip", "recom"])
def test_every_method_honors_the_floor(method) -> None:
    if method == "mip":
        pytest.importorskip("gurobipy")
    # Two corner schools on a 3x3 grid of one resident each: the cheapest cut
    # isolates a corner, which a floor of four forbids.
    problem = make_grid_problem(3, 3, min_zone_residents=4)
    solution = solve_objective(
        problem,
        method,
        "cut_edges",
        options={
            "solve_time_limit": 5,
            "recom_iterations": 60,
            "recom_burst_length": 5,
            "workers": 1,
            "seed": 11,
            # ReCom walks from a start; Voronoi, as the sweeps use, need not
            # meet the floor, so this also checks the walk repairs it.
            "hints": "voronoi" if method == "recom" else "none",
        },
    )
    assert solution.feasible
    assert check_zoning(solution.problem, solution.assignment).feasible
    assert min(_residents(problem, solution.assignment).values()) >= 4

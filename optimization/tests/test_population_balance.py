"""The per-zone population band, ``population_balance``."""

from __future__ import annotations

import pytest

from optimization.config import OptimizationConfig
from optimization.data.feasibility import check_zoning
from optimization.data.initial_solutions import feasibility_fingerprint
from optimization.objectives import solve_objective
from optimization.pipeline import checked_constraints
from optimization.problem import POPULATION_BALANCE_AUTO_SLACK
from optimization.solvers.recom import _ReComContext
from optimization.tests.synthetic import make_grid_problem

# 3x3 grid, one student a node, corner schools: zone 0 the left column.
LEFT_COLUMN = {n: int(n % 3 != 0) for n in range(9)}  # 3 and 6 students


def _students(problem, assignment) -> list[float]:
    totals = [0.0] * problem.Z
    for node, zone in assignment.items():
        totals[zone] += problem.students(node)
    return totals


def test_it_is_off_by_default() -> None:
    assert OptimizationConfig().population_balance == -1.0
    assert make_grid_problem(3, 3).population_band() is None


@pytest.mark.parametrize("value", ["auto", " AUTO ", 0.1, 0, -1])
def test_the_config_accepts_a_number_or_auto(value) -> None:
    config = OptimizationConfig(population_balance=value)
    assert config.population_balance == (
        "auto" if isinstance(value, str) else float(value)
    )


@pytest.mark.parametrize("value", ["balanced", float("nan"), float("inf"), True])
def test_the_config_rejects_anything_else(value) -> None:
    with pytest.raises(ValueError, match="population_balance"):
        OptimizationConfig(population_balance=value)


def test_it_is_a_step_constraint_and_auto_passes_through() -> None:
    assert checked_constraints({"population_balance": "auto"}) == {
        "population_balance": "auto"
    }
    assert checked_constraints({"population_balance": 0.1}) == {
        "population_balance": 0.1
    }


def test_auto_is_the_start_worst_deviation_plus_the_slack() -> None:
    problem = make_grid_problem(3, 3, population_balance="auto")
    # 3 and 6 against an average of 4.5: worst deviation 1/3.
    assert problem.population_band(LEFT_COLUMN) == pytest.approx(
        1 / 3 + POPULATION_BALANCE_AUTO_SLACK
    )
    with pytest.raises(ValueError, match="starting zoning"):
        problem.population_band()


def test_check_zoning_reports_a_numeric_band_and_skips_auto() -> None:
    loose = make_grid_problem(3, 3, population_balance=0.34)
    assert check_zoning(loose, LEFT_COLUMN).feasible

    tight = make_grid_problem(3, 3, population_balance=0.2)
    kinds = {v.kind for v in check_zoning(tight, LEFT_COLUMN).violations}
    assert kinds == {"population_lower", "population_upper"}

    auto = make_grid_problem(3, 3, population_balance="auto")
    assert check_zoning(auto, LEFT_COLUMN).feasible


def test_recom_carries_the_band_as_two_violation_rows() -> None:
    problem = make_grid_problem(3, 3, population_balance=0.2)
    context = _ReComContext(problem)
    assert context.violation_labels()[-2:] == ("population_lower", "population_upper")

    state = context.build_state(context.validate_hint(LEFT_COLUMN))
    assert not state.feasible
    # 3 students against a floor of 0.8 * 4.5 = 3.6, in percent of 4.5.
    assert state.zone_violations[0][-2] == pytest.approx(0.6 / 4.5 * 100)
    assert state.zone_violations[1][-1] == pytest.approx(0.6 / 4.5 * 100)


def test_the_band_keys_the_feasible_hint_cache_only_when_on() -> None:
    off = make_grid_problem(3, 3)
    explicit_off = make_grid_problem(3, 3, population_balance=-1)
    on = make_grid_problem(3, 3, population_balance=0.1)
    auto = make_grid_problem(3, 3, population_balance="auto")
    assert feasibility_fingerprint(off) == feasibility_fingerprint(explicit_off)
    assert len({feasibility_fingerprint(p) for p in (off, on, auto)}) == 3


@pytest.mark.parametrize("method", ["cp_bool", "cp_int", "mip", "recom"])
def test_every_method_honors_a_numeric_band(method) -> None:
    if method == "mip":
        pytest.importorskip("gurobipy")
    # On a 4x4 grid the cheapest cut isolates a corner; a 10% band forces an
    # eight-eight split.
    problem = make_grid_problem(4, 4, population_balance=0.1)
    solution = solve_objective(
        problem,
        method,
        "cut_edges",
        options={
            "solve_time_limit": 5,
            "recom_iterations": 200,
            "recom_burst_length": 5,
            "workers": 1,
            "seed": 11,
            "hints": "voronoi" if method == "recom" else "none",
        },
    )
    assert solution.feasible
    assert check_zoning(solution.problem, solution.assignment).feasible
    assert sorted(_students(problem, solution.assignment)) == [8.0, 8.0]


@pytest.mark.parametrize("method", ["cp_bool", "cp_int"])
def test_auto_is_recom_only(method) -> None:
    problem = make_grid_problem(4, 4, population_balance="auto")
    with pytest.raises(ValueError, match="only ReCom"):
        solve_objective(
            problem,
            method,
            "cut_edges",
            options={"solve_time_limit": 5, "workers": 1, "hints": "none"},
        )

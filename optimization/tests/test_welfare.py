"""Tests for stable finite-grid utilitarian-welfare optimization."""

from __future__ import annotations

import itertools

import pytest

from optimization.data import contiguity
from optimization.problem import CutoffMarket, CutoffStudent
from optimization.solvers import get_solver
from optimization.solvers.welfare import WelfareSolver
from optimization.solvers.welfare_decomposition import (
    WelfareDecompositionSolver,
    _WelfareIncumbent,
)
from optimization.tests.synthetic import make_grid_problem
from optimization.welfare_oracle import (
    MAX_EXACT_CP_SAT_OBJECTIVE,
    raw_welfare_upper_bound,
    solve_zoned_welfare,
    validate_welfare_market,
)
from optimization.verify_welfare_scenario import _float_maps_close, _integer_keyed


def _student(studentno, node, preferences, utilities):
    return CutoffStudent(
        studentno,
        node,
        preferences,
        {school: 0 for school in preferences},
        utilities,
    )


def test_welfare_oracle_integrates_lottery_assignment_mass():
    market = CutoffMarket(
        students=(
            _student(1, 0, (100,), {100: 2.0}),
            _student(2, 0, (100,), {100: 6.0}),
        ),
        school_nodes={100: 1},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )

    result = solve_zoned_welfare(market, {0: 0, 1: 0}, num_zones=1, utility_scale=100)

    assert result.cutoffs.school_cutoffs == {100: 2}
    assert result.assignments == {1: {100: 2}, 2: {100: 2}}
    assert result.outside_option_mass == {1: 2, 2: 2}
    assert result.welfare == pytest.approx(4.0)
    assert result.raw_scaled_welfare == 1600


def test_serialized_school_cutoff_keys_are_normalized():
    stored = {"100": 1.25, "200": 2.5}

    assert _integer_keyed(stored) == {100: 1.25, 200: 2.5}
    assert _float_maps_close({100: 1.25, 200: 2.5}, _integer_keyed(stored))


@pytest.mark.parametrize(
    "student, outside, message",
    [
        (
            _student(1, 0, (100, 200), {100: 1.0, 200: 2.0}),
            0.0,
            "nonincreasing",
        ),
        (_student(1, 0, (100,), {100: -1.0}), 0.0, "outside option"),
        (_student(1, 0, (100,), {100: 2.0}), 1.0, "utility zero"),
    ],
)
def test_welfare_market_rejects_invalid_cardinal_preferences(student, outside, message):
    market = CutoffMarket(
        students=(student,),
        school_nodes={
            school: index for index, school in enumerate(student.preferences)
        },
        school_capacities={school: 1 for school in student.preferences},
        zone_restricted_schools=frozenset(student.preferences),
        lottery_scale=4,
        outside_option_utility=outside,
    )

    with pytest.raises(ValueError, match=message):
        validate_welfare_market(market)


def test_welfare_market_rejects_inexact_cp_sat_objective_range():
    utility = (MAX_EXACT_CP_SAT_OBJECTIVE + 1) / 4
    market = CutoffMarket(
        students=(_student(1, 0, (100,), {100: utility}),),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )

    with pytest.raises(ValueError, match="exact CP-SAT reporting range"):
        validate_welfare_market(market, utility_scale=1)


def test_gurobi_heuristic_bound_cannot_cap_integer_master():
    pytest.importorskip("gurobipy")
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    utility = float(MAX_EXACT_CP_SAT_OBJECTIVE - 2048)
    problem.cutoff_market = CutoffMarket(
        students=(_student(1, 1, (100,), {100: utility}),),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=1,
    )
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    result = solve_zoned_welfare(
        problem.cutoff_market,
        assignment,
        num_zones=2,
        utility_scale=1,
    )
    solver = WelfareDecompositionSolver(
        get_solver("cp_bool", solve_time_limit=1, workers=1), utility_scale=1
    )

    _, bound, _ = solver._assignment_relaxation_mip(
        problem, _WelfareIncumbent(assignment, result), 1.0
    )

    assert bound == raw_welfare_upper_bound(problem.cutoff_market, 1)


def test_external_direct_cap_cannot_certify_global_optimum():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = CutoffMarket(
        students=(_student(1, 1, (100,), {100: 5.0}),),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    problem.hint = {0: 0, 1: 0, 2: 1, 3: 1}
    incumbent = solve_zoned_welfare(
        problem.cutoff_market,
        problem.hint,
        num_zones=problem.Z,
        utility_scale=10,
    )
    zoning_solver = get_solver(
        "cp_bool",
        solve_time_limit=10,
        workers=1,
        seed=42,
        welfare_raw_upper_bound=incumbent.raw_scaled_welfare,
    )

    solution = WelfareSolver(zoning_solver, utility_scale=10).solve(problem)

    assert solution.metadata["configured_raw_upper_bound"] == (
        incumbent.raw_scaled_welfare
    )
    assert not solution.metadata["global_optimum_certified"]
    assert solution.status == "FEASIBLE"


def test_gurobi_transport_bound_remains_diagnostic():
    pytest.importorskip("gurobipy")
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = CutoffMarket(
        students=(_student(1, 1, (100,), {100: 5.0}),),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    result = solve_zoned_welfare(
        problem.cutoff_market,
        assignment,
        num_zones=problem.Z,
        utility_scale=10,
    )
    solver = WelfareDecompositionSolver(
        get_solver("cp_bool", solve_time_limit=1, workers=1),
        utility_scale=10,
    )

    _, returned_bound, _ = solver._assignment_transport_mip(
        problem,
        _WelfareIncumbent(assignment, result),
        1.0,
    )

    assert returned_bound == solver._global_capacity_upper_bound(problem.cutoff_market)
    assert solver._transport_mip_details["proof_grade_raw_upper_bound"] == (
        returned_bound
    )


def test_welfare_decomposition_matches_exhaustive_tiny_zonings():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = CutoffMarket(
        students=(
            _student(1, 0, (100, 200), {100: 2.0, 200: 0.5}),
            _student(2, 1, (100, 200), {100: 6.0, 200: 1.0}),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    closer = contiguity.closer_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    brute_force = []
    for zone_1, zone_2 in itertools.product(range(2), repeat=2):
        assignment = {0: 0, 1: zone_1, 2: zone_2, 3: 1}
        if any(
            node != problem.centroids[zone]
            and not any(assignment[neighbor] == zone for neighbor in closer[node, zone])
            for node, zone in assignment.items()
        ):
            continue
        brute_force.append(
            solve_zoned_welfare(
                problem.cutoff_market,
                assignment,
                num_zones=2,
                utility_scale=100,
            ).raw_scaled_welfare
        )

    zoning_solver = get_solver("cp_bool", solve_time_limit=10, workers=1, seed=42)
    solution = WelfareDecompositionSolver(zoning_solver, utility_scale=100).solve(
        problem
    )

    assert solution.status == "OPTIMAL"
    assert solution.metadata["raw_scaled_welfare"] == max(brute_force)
    assert solution.metadata["global_optimum_certified"]
    assert solution.metadata["grid_minimal"]
    assert solution.metadata["stable"]

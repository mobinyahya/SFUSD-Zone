"""Tests for constructing the cutoff assignment market."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from ortools.sat.python import cp_model

from optimization.cutoff_oracle import (
    _coupled_zone_stability_checks,
    assignments_and_demands,
    solve_coupled_continuum_cutoffs,
    solve_coupled_cutoffs,
    solve_continuum_market_cutoffs,
    solve_market_cutoffs,
    solve_zoned_cutoffs,
    validate_market_cutoffs,
)
from optimization.data import contiguity
from optimization.data.cutoffs import (
    _exclude_citywide_school_columns,
    _zone_restricted_schools,
)
from optimization.problem import CutoffMarket, CutoffStudent
from optimization.solvers import get_solver
from optimization.solvers.cutoff_decomposition import (
    CutoffDecompositionSolver,
    _DemandInterval,
)
from optimization.tests.synthetic import make_grid_problem


def test_zone_restricted_schools_are_attendance_schools_with_ge():
    market = SimpleNamespace(
        schools=SimpleNamespace(
            school_df=pd.DataFrame(
                {
                    "category": ["Attendance", "Attendance", "Citywide"],
                },
                index=[100, 101, 200],
            )
        ),
        programs=SimpleNamespace(
            program_df=pd.DataFrame(
                {
                    "school_id": [100, 101, 200],
                    "program_type": ["GE", "CB", "GE"],
                }
            )
        ),
    )

    restricted = _zone_restricted_schools(market, [100, 101, 200])

    assert restricted == frozenset({100})


def test_remove_citywide_excludes_columns_and_restricts_every_remaining_school():
    market = SimpleNamespace(
        schools=SimpleNamespace(
            school_df=pd.DataFrame(
                {"category": ["Attendance", "Attendance", "Citywide"]},
                index=[100, 101, 200],
            )
        ),
        programs=SimpleNamespace(
            program_df=pd.DataFrame(
                {
                    "school_id": [100, 101, 200],
                    "program_type": ["GE", "CB", "GE"],
                }
            )
        ),
    )
    priorities = np.array([[1, 2, 3], [4, 5, 6]])
    utilities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    school_ids, priorities, utilities, excluded = _exclude_citywide_school_columns(
        frozenset({200}),
        [100, 101, 200],
        priorities,
        utilities,
    )
    restricted = _zone_restricted_schools(market, school_ids, restrict_all=True)

    assert school_ids == [100, 101]
    assert excluded == [200]
    np.testing.assert_array_equal(priorities, [[1, 2], [4, 5]])
    np.testing.assert_array_equal(utilities, [[0.1, 0.2], [0.4, 0.5]])
    assert restricted == frozenset({100, 101})


def test_analytical_cutoffs_clear_discretized_stb_market():
    students = (
        CutoffStudent(1, 0, (100, 200), {100: 0, 200: 0}),
        CutoffStudent(2, 0, (100, 200), {100: 0, 200: 0}),
        CutoffStudent(3, 0, (100, 200), {100: 0, 200: 0}),
    )

    result = solve_market_cutoffs(students, {100: 1, 200: 2}, lottery_scale=20)

    assert result.cutoffs == {100: 14, 200: 1}
    assert result.demands == {100: 18, 200: 39}
    assert result.objective == 15
    assert result.normalized_objective == 0.75
    assert result.grid_minimal
    assert validate_market_cutoffs(
        students, {100: 1, 200: 2}, result.cutoffs, lottery_scale=20
    )


def test_zoned_cutoff_oracle_solves_markets_independently():
    market = CutoffMarket(
        students=(
            CutoffStudent(1, 0, (100, 200), {100: 0, 200: 0}),
            CutoffStudent(2, 0, (100, 200), {100: 0, 200: 0}),
            CutoffStudent(3, 2, (100, 200), {100: 0, 200: 0}),
        ),
        school_nodes={100: 1, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=20,
    )

    result = solve_zoned_cutoffs(market, {0: 0, 1: 0, 2: 1, 3: 1}, num_zones=2)

    assert result.school_cutoffs == {100: 10, 200: 0}
    assert result.normalized_objective == 0.5
    assert result.grid_minimal


def test_continuum_cutoffs_exactly_clear_positive_cutoff_schools():
    students = tuple(
        CutoffStudent(studentno, 0, (100,), {100: 0}) for studentno in range(3)
    )

    result = solve_continuum_market_cutoffs(students, {100: 1})

    assert result.cutoffs[100] == pytest.approx(2 / 3)
    assert result.demands[100] == pytest.approx(1.0)
    assert result.stable


def test_zoned_cutoff_oracle_rejects_cross_market_schools():
    market = CutoffMarket(
        students=(),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset(),
        lottery_scale=20,
    )

    with np.testing.assert_raises_regex(ValueError, "every school"):
        solve_zoned_cutoffs(market, {0: 0}, num_zones=1)


def test_coupled_oracle_shares_citywide_capacity_across_zones():
    market = CutoffMarket(
        students=(
            CutoffStudent(1, 0, (200,), {200: 0}),
            CutoffStudent(2, 2, (200,), {200: 0}),
        ),
        school_nodes={200: 1},
        school_capacities={200: 1},
        zone_restricted_schools=frozenset(),
        lottery_scale=20,
    )
    assignment = {0: 0, 1: 0, 2: 1}

    grid = solve_coupled_cutoffs(market, assignment, num_zones=2)
    continuum = solve_coupled_continuum_cutoffs(market, assignment, num_zones=2)

    assert grid.school_cutoffs == {200: 10}
    assert grid.zone_demands == {0: {200: 10}, 1: {200: 10}}
    assert grid.grid_minimal
    assert continuum.school_cutoffs[200] == pytest.approx(0.5)
    assert continuum.zone_demands == {
        0: {200: pytest.approx(0.5)},
        1: {200: pytest.approx(0.5)},
    }
    assert continuum.stable
    assert continuum.zone_stable == {0: True, 1: True}


def test_coupled_zone_checks_detect_cross_zone_restricted_demand():
    market = CutoffMarket(
        students=(
            CutoffStudent(1, 0, (100, 200), {100: 0, 200: 0}),
            CutoffStudent(2, 2, (100, 200), {100: 0, 200: 0}),
        ),
        school_nodes={100: 1, 200: 1},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=20,
    )
    assignment = {0: 0, 1: 0, 2: 1}
    continuum = solve_coupled_continuum_cutoffs(market, assignment, num_zones=2)
    tampered_demands = {
        zone: dict(demands) for zone, demands in continuum.zone_demands.items()
    }
    tampered_demands[1][100] = 0.25

    checks = _coupled_zone_stability_checks(
        market,
        assignment,
        continuum.market,
        tampered_demands,
        (0, 1),
        2,
    )

    assert checks[0]["access_respected"]
    assert not checks[1]["access_respected"]
    assert not checks[1]["zone_demands_reconcile_globally"]


def test_grid_oracle_matches_brute_force_least_cutoffs():
    students = (
        CutoffStudent(1, 0, (100, 200), {100: 0, 200: 0}),
        CutoffStudent(2, 0, (200, 100), {100: 0, 200: 0}),
        CutoffStudent(3, 0, (100, 200), {100: 0, 200: 0}),
    )
    capacities = {100: 1, 200: 1}
    result = solve_market_cutoffs(students, capacities, lottery_scale=4)
    feasible = []
    for first in range(5):
        for second in range(5):
            cutoffs = {100: first, 200: second}
            _, demands = assignments_and_demands(students, cutoffs, 4)
            if all(demands[school] <= capacities[school] * 4 for school in capacities):
                feasible.append(cutoffs)

    assert feasible
    assert all(
        result.cutoffs[school] <= cutoffs[school]
        for cutoffs in feasible
        for school in capacities
    )


@pytest.mark.parametrize(
    ("block_zone", "school_zone", "expected_access"),
    ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)),
)
def test_block_school_access_follows_the_assigned_school_zone(
    block_zone, school_zone, expected_access
):
    model = cp_model.CpModel()
    x = {
        (zone, node): model.NewBoolVar(f"x_{zone}_{node}")
        for zone in range(2)
        for node in range(2)
    }
    for node, assigned_zone in ((0, block_zone), (1, school_zone)):
        for zone in range(2):
            model.Add(x[zone, node] == (zone == assigned_zone))
    problem = SimpleNamespace(candidate_zones=lambda _node: {0, 1})
    market = SimpleNamespace(school_nodes={100: 1})
    decomposition = CutoffDecompositionSolver(SimpleNamespace(options={}))

    access = decomposition._zone_access(model, problem, market, x, 0, 100)
    solver = cp_model.CpSolver()

    assert solver.Solve(model) == cp_model.OPTIMAL
    assert solver.Value(access) == expected_access


def test_block_school_access_does_not_imply_a_school_zone_assignment():
    model = cp_model.CpModel()
    x = {
        (zone, node): model.NewBoolVar(f"x_{zone}_{node}")
        for zone in range(2)
        for node in range(2)
    }
    problem = SimpleNamespace(candidate_zones=lambda _node: {0, 1})
    market = SimpleNamespace(school_nodes={100: 1})
    decomposition = CutoffDecompositionSolver(SimpleNamespace(options={}))
    access = decomposition._zone_access(model, problem, market, x, 0, 100)
    model.Add(access == 1)
    model.Add(x[0, 1] == 0)
    model.Add(x[1, 1] == 0)

    assert cp_model.CpSolver().Solve(model) == cp_model.OPTIMAL


@pytest.mark.parametrize(
    (
        "restricted_schools",
        "block_zone",
        "target_zone",
        "higher_zone",
        "higher_cutoff",
        "expected_status",
    ),
    (
        (frozenset({100, 200}), 1, 1, 0, 0, cp_model.INFEASIBLE),
        (frozenset({100, 200}), 1, 1, 1, 0, cp_model.OPTIMAL),
        (frozenset({100, 200}), 1, 1, 1, 4, cp_model.INFEASIBLE),
        (frozenset({100, 200}), 1, 0, 0, 4, cp_model.OPTIMAL),
        (frozenset({200}), 1, 0, 0, 0, cp_model.INFEASIBLE),
        (frozenset({100}), 1, 1, 0, 0, cp_model.OPTIMAL),
    ),
)
def test_interval_capacity_clause_uses_access_and_higher_affordability(
    restricted_schools,
    block_zone,
    target_zone,
    higher_zone,
    higher_cutoff,
    expected_status,
):
    student = CutoffStudent(1, 0, (200, 100), {100: 0, 200: 0})
    market = CutoffMarket(
        students=(student,),
        school_nodes={100: 1, 200: 2},
        school_capacities={100: 0, 200: 1},
        zone_restricted_schools=restricted_schools,
        lottery_scale=4,
    )
    problem = SimpleNamespace(candidate_zones=lambda _node: {0, 1})
    model = cp_model.CpModel()
    x = {
        (zone, node): model.NewBoolVar(f"x_{zone}_{node}")
        for zone in range(2)
        for node in range(3)
    }
    for node, assigned_zone in (
        (0, block_zone),
        (1, target_zone),
        (2, higher_zone),
    ):
        for zone in range(2):
            model.Add(x[zone, node] == (zone == assigned_zone))
    cutoffs = {
        100: model.NewIntVar(0, 4, "cutoff_100"),
        200: model.NewIntVar(0, 4, "cutoff_200"),
    }
    model.Add(cutoffs[100] == 0)
    model.Add(cutoffs[200] == higher_cutoff)
    decomposition = CutoffDecompositionSolver(SimpleNamespace(options={}))

    added = decomposition._add_interval_capacity_cut(
        model,
        problem,
        market,
        x,
        cutoffs,
        100,
        [_DemandInterval(student, 1, 4, (200,))],
        4,
    )

    assert added
    assert not decomposition._add_interval_capacity_cut(
        model,
        problem,
        market,
        x,
        cutoffs,
        100,
        [_DemandInterval(student, 1, 4, (200,))],
        4,
    )
    assert decomposition._cut_count == 1
    assert decomposition._cut_profile_count == 1
    assert cp_model.CpSolver().Solve(model) == expected_status


def test_conditional_demand_groups_identical_block_prefix_priority_profiles():
    students = (
        CutoffStudent(1, 0, (100,), {100: 0}),
        CutoffStudent(2, 0, (100,), {100: 0}),
    )
    market = CutoffMarket(
        students=students,
        school_nodes={100: 1},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    problem = SimpleNamespace(candidate_zones=lambda _node: {0, 1})
    model = cp_model.CpModel()
    x = {
        (zone, node): model.NewBoolVar(f"x_{zone}_{node}")
        for zone in range(2)
        for node in range(2)
    }
    for node in range(2):
        model.Add(x[0, node] == 1)
        model.Add(x[1, node] == 0)
    cutoffs = {100: model.NewIntVar(0, 4, "cutoff_100")}
    model.Add(cutoffs[100] == 0)
    decomposition = CutoffDecompositionSolver(
        SimpleNamespace(options={}), generate_assigned_pairs=True
    )

    pairs, profiles, cuts = decomposition._activate_conditional_demand_pairs(
        model,
        problem,
        market,
        x,
        cutoffs,
        {
            100: [
                _DemandInterval(student, 1, 4, ()) for student in students
            ]
        },
        4,
    )

    assert (pairs, profiles, cuts) == (2, 1, 1)
    assert len(decomposition._conditional_demand_vars) == 1
    # One shared demand of four has multiplicity two, exceeding capacity four.
    assert cp_model.CpSolver().Solve(model) == cp_model.INFEASIBLE


def test_cutoff_decomposition_matches_exhaustive_tiny_zonings():
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
            CutoffStudent(1, 1, (100, 200), {100: 0, 200: 0}),
            CutoffStudent(2, 2, (100, 200), {100: 0, 200: 0}),
            CutoffStudent(3, 2, (200, 100), {100: 0, 200: 0}),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 0, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    closer = contiguity.closer_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    brute_objectives = []
    for zone_1 in range(2):
        for zone_2 in range(2):
            assignment = {0: 0, 1: zone_1, 2: zone_2, 3: 1}
            if any(
                node != problem.centroids[zone]
                and not any(
                    assignment[neighbor] == zone for neighbor in closer[node, zone]
                )
                for node, zone in assignment.items()
            ):
                continue
            brute_objectives.append(
                solve_zoned_cutoffs(
                    problem.cutoff_market, assignment, num_zones=2
                ).objective
            )

    zoning_solver = get_solver("cp_bool", solve_time_limit=10, workers=1, seed=42)
    solution = CutoffDecompositionSolver(zoning_solver).solve(problem)
    pair_solution = CutoffDecompositionSolver(
        zoning_solver, generate_assigned_pairs=True
    ).solve(problem)
    direct_solution = zoning_solver.solve(problem)

    assert solution.status == "OPTIMAL"
    assert solution.metadata["raw_objective"] == min(brute_objectives)
    assert solution.metadata["global_optimum_certified"]
    assert solution.metadata["stable"]
    assert pair_solution.status == "OPTIMAL"
    assert pair_solution.metadata["raw_objective"] == min(brute_objectives)
    assert pair_solution.metadata["optimization_method"] == (
        "exact_overloaded_pair_generation"
    )
    assert pair_solution.metadata["assigned_pair_seed_count"] == 0
    assert pair_solution.metadata["assigned_pair_seed_cut_count"] == 0
    assert pair_solution.metadata["conditional_demand_pair_count"] > 0
    assert pair_solution.metadata["conditional_demand_capacity_constraint_count"] <= len(
        problem.cutoff_market.school_capacities
    )
    assert all(
        row.get("conditional_demand_pairs_added", 0) == 0
        for row in pair_solution.metadata["decomposition_rounds"]
        if row["overloaded_schools"] == 0
    )
    assert direct_solution.status == "OPTIMAL"
    assert direct_solution.metadata["raw_objective"] == min(brute_objectives)
    assert direct_solution.metadata["global_optimum_certified"]
    assert (
        solve_zoned_cutoffs(
            problem.cutoff_market, direct_solution.assignment, num_zones=2
        ).objective
        == direct_solution.metadata["raw_objective"]
    )


def test_citywide_decomposition_matches_exhaustive_tiny_zonings():
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
            CutoffStudent(1, 1, (100, 200), {100: 0, 200: 0}),
            CutoffStudent(2, 2, (100, 200), {100: 0, 200: 0}),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 0},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    closer = contiguity.closer_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    brute_objectives = []
    for zone_1 in range(2):
        for zone_2 in range(2):
            assignment = {0: 0, 1: zone_1, 2: zone_2, 3: 1}
            if any(
                node != problem.centroids[zone]
                and not any(
                    assignment[neighbor] == zone for neighbor in closer[node, zone]
                )
                for node, zone in assignment.items()
            ):
                continue
            brute_objectives.append(
                solve_coupled_cutoffs(
                    problem.cutoff_market, assignment, num_zones=2
                ).objective
            )

    zoning_solver = get_solver("cp_bool", solve_time_limit=10, workers=1, seed=42)
    solution = CutoffDecompositionSolver(zoning_solver).solve(problem)

    assert solution.status == "OPTIMAL"
    assert solution.metadata["raw_objective"] == min(brute_objectives)
    assert solution.metadata["global_optimum_certified"]
    assert solution.metadata["market_coupling"] == "global_citywide_access"
    assert solution.metadata["stable"]
    assert solution.metadata["zone_stable"] == {"0": True, "1": True}


def test_coupled_oracle_rejects_school_outside_zone_range():
    market = CutoffMarket(
        students=(CutoffStudent(1, 0, (100,), {100: 0}),),
        school_nodes={100: 1},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=20,
    )

    with pytest.raises(ValueError, match="School 100 has invalid zone 2"):
        solve_coupled_cutoffs(market, {0: 0, 1: 2}, num_zones=2)


def test_cutoff_decomposition_preserves_centroid_neighborhoods():
    problem = make_grid_problem(
        3,
        3,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = CutoffMarket(
        students=(CutoffStudent(1, 5, (100, 200), {100: 0, 200: 0}),),
        school_nodes={100: 0, 200: 8},
        school_capacities={100: 1, 200: 0},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    zoning_solver = get_solver(
        "cp_bool",
        solve_time_limit=10,
        workers=1,
        seed=42,
        centroid_neighbor_radius=1,
    )

    solution = CutoffDecompositionSolver(zoning_solver).solve(problem)

    assert solution.status == "OPTIMAL"
    assert solution.metadata["global_optimum_certified"]
    for zone, centroid in enumerate(problem.centroids):
        neighborhood = nx.single_source_shortest_path_length(
            problem.G, centroid, cutoff=1
        )
        assert all(solution.assignment[node] == zone for node in neighborhood)

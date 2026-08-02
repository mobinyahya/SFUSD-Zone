"""Tests for constructing the cutoff assignment market."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from optimization.cutoff_oracle import (
    assignments_and_demands,
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
from optimization.solvers.cutoff_decomposition import CutoffDecompositionSolver
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
        CutoffStudent(studentno, 0, (100,), {100: 0})
        for studentno in range(3)
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
                and not any(assignment[neighbor] == zone for neighbor in closer[node, zone])
                for node, zone in assignment.items()
            ):
                continue
            brute_objectives.append(
                solve_zoned_cutoffs(
                    problem.cutoff_market, assignment, num_zones=2
                ).objective
            )

    zoning_solver = get_solver(
        "cp_bool", solve_time_limit=10, workers=1, seed=42
    )
    solution = CutoffDecompositionSolver(zoning_solver).solve(problem)

    assert solution.status == "OPTIMAL"
    assert solution.metadata["raw_objective"] == min(brute_objectives)
    assert solution.metadata["global_optimum_certified"]
    assert solution.metadata["stable"]


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

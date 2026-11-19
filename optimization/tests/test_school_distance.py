"""The school-distance objective, the same centroid cost for CP/MIP and ReCom."""

from __future__ import annotations

import itertools

import pytest

from benchmark.config import optimization_config_hash
from optimization.config import OptimizationConfig
from optimization.data.feasibility import check_zoning
from optimization.objectives import heuristic_scorer
from optimization.solvers import get_solver
from optimization.tests.synthetic import make_grid_problem


def _grid(area_weight: float = 0.75, **overrides):
    """3x3 grid, schools in opposite corners; areas grow left to right."""
    problem = make_grid_problem(
        3,
        3,
        objective="school_distance",
        school_distance_area_weight=area_weight,
        school_distance_student_weight=1.0 - area_weight,
        **overrides,
    )
    for node in problem.nodes:
        problem.G.nodes[node]["land_area_m2"] = 1.0 + node % 3
    return problem


def _brute_force_cost(problem) -> float:
    """Cheapest contiguous, check_zoning-feasible two-zone split of the grid."""
    costs = problem.school_distance_costs()
    best = float("inf")
    for labels in itertools.product(range(problem.Z), repeat=problem.A):
        assignment = dict(zip(problem.nodes, labels))
        if not check_zoning(problem, assignment).feasible:
            continue
        best = min(best, sum(costs[(z, n)] for n, z in assignment.items()))
    return best


def test_weights_mix_area_and_student_shares_and_sum_to_one():
    problem = _grid(area_weight=0.75)
    weights = problem.school_distance_weights()

    assert sum(weights.values()) == pytest.approx(1.0)
    total_area = sum(1.0 + n % 3 for n in problem.nodes)
    for node, weight in weights.items():
        expected = 0.75 * (1.0 + node % 3) / total_area + 0.25 / problem.A
        assert weight == pytest.approx(expected)


def test_costs_are_weight_times_squared_distance_to_the_zone_school():
    problem = _grid()
    weights = problem.school_distance_weights()
    for (zone, node), cost in problem.school_distance_costs().items():
        distance = problem.distance(problem.centroids[zone], node)
        assert cost == pytest.approx(weights[node] * distance**2)


@pytest.mark.parametrize("solver_name", ["cp_bool", "cp_int"])
def test_cp_sat_minimizes_school_distance_exactly(solver_name):
    problem = _grid()
    solver = get_solver(solver_name, solve_time_limit=20, workers=2, hints="none")

    solution = solver.solve(problem)

    assert solution.status == "OPTIMAL"
    assert check_zoning(problem, solution.assignment).feasible
    assert solution.objective == pytest.approx(_brute_force_cost(problem), abs=1e-7)
    assert solution.metadata["objective_kind"] == "school_distance"


def test_mip_minimizes_school_distance_exactly():
    pytest.importorskip("gurobipy")
    problem = _grid()
    solver = get_solver("mip", solve_time_limit=20, workers=1, hints="none")

    solution = solver.solve(problem)

    if solution.status not in {"OPTIMAL", "FEASIBLE"}:
        pytest.skip(f"Gurobi unavailable here ({solution.status}).")
    assert solution.objective == pytest.approx(_brute_force_cost(problem), abs=1e-7)


def test_recom_scores_school_distance_to_each_zones_centroid():
    problem = _grid()
    scorer = heuristic_scorer(problem, "school_distance", {})
    costs = problem.school_distance_costs()
    assignment = {n: (0 if n in {0, 1, 3} else 1) for n in problem.nodes}

    assert scorer((assignment,)) == pytest.approx(
        (sum(costs[(z, n)] for n, z in assignment.items()),)
    )


def test_recom_school_distance_anchors_centroids_and_reports_the_cp_cost():
    problem = _grid()
    solver = get_solver("recom", solve_time_limit=5, recom_iterations=30, seed=1)

    solution = solver.solve(problem)

    assert solution.feasible
    for zone, centroid in enumerate(problem.centroids):
        assert solution.assignment[centroid] == zone
    costs = problem.school_distance_costs()
    assert solution.objective == pytest.approx(
        sum(costs[(z, n)] for n, z in solution.assignment.items())
    )
    assert solution.metadata["objective_kind"] == "school_distance"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"school_distance_area_weight": 0.5}, "sum to 1"),
        (
            {
                "school_distance_area_weight": 1.5,
                "school_distance_student_weight": -0.5,
            },
            r"\[0, 1\]",
        ),
        ({"objective": "perimeter"}, "Unknown objective"),
    ],
)
def test_config_rejects_bad_school_distance_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        OptimizationConfig(levels=["BlockGroup_0"], **overrides)


def test_config_accepts_school_distance_for_cp_and_short_bursts():
    cp = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="cp_bool",
        objective="school_distance",
        school_distance_area_weight=1.0,
        school_distance_student_weight=0.0,
    )
    assert cp.school_distance_area_weight == 1.0
    asb = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="recom",
        objective="school_distance",
    )
    assert (asb.school_distance_area_weight, asb.school_distance_student_weight) == (
        0.75,
        0.25,
    )


def test_default_objective_settings_keep_benchmark_hashes():
    manifest = {"sources": {}}
    default = OptimizationConfig(levels=["BlockGroup_0"])
    reweighted = OptimizationConfig(
        levels=["BlockGroup_0"],
        school_distance_area_weight=0.5,
        school_distance_student_weight=0.5,
    )
    distance = OptimizationConfig(
        levels=["BlockGroup_0"], objective="school_distance"
    )

    base = optimization_config_hash(default, source_manifest=manifest)
    # Weights are inert, and so hash-neutral, until school_distance is chosen.
    assert optimization_config_hash(reweighted, source_manifest=manifest) == base
    assert optimization_config_hash(distance, source_manifest=manifest) != base

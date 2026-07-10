"""Shared assertions and solver setup for solver contract tests."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from Zone_Generation.optimization.data.contiguity import boundary_edges
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers import get_solver
from Zone_Generation.optimization.solvers.balance import balance_constraints
from Zone_Generation.optimization.solvers.base import available_solvers

CONTRACT_SOLVERS = (
    "cp_int",
    "cp_bool",
    "mip",
    "recom",
    "relaxed_recom",
    "short_bursts_recom",
)
RECOM_SOLVERS = ("recom", "relaxed_recom", "short_bursts_recom")
TOLERANCE = 1e-6


def solve_contract_problem(
    name: str,
    problem: ZoneProblem,
    **option_overrides,
) -> ZoneSolution:
    """Solve with deterministic, short-running options shared by the suite."""
    if name == "mip" and name not in available_solvers():
        pytest.skip("gurobipy is not installed")

    options = {
        "solve_time_limit": 2.0,
        "workers": 1,
        "seed": 1,
        "hints": "voronoi",
        "recom_iterations": 0,
        "recom_cut_attempts": 10,
        "recom_population_epsilon": 10.0,
        "recom_balance_metric": "nodes",
        "recom_temperature": 0.0,
        "recom_repair_iterations": 0,
        "short_bursts_length": 1,
        "relaxed_recom_min_boundary_edges": 0,
    }
    options.update(option_overrides)

    try:
        return get_solver(name, **options).solve(problem)
    except Exception as exc:
        if name == "mip" and "license" in str(exc).lower():
            pytest.skip(f"Gurobi license unavailable: {exc}")
        raise


def assert_valid_solution(
    problem: ZoneProblem,
    solution: ZoneSolution,
    *,
    tolerance: float = TOLERANCE,
    check_boundary_objective: bool = True,
) -> None:
    """Assert the complete hard-constraint contract independently of a backend."""
    assert solution.status in {"OPTIMAL", "FEASIBLE"}
    assert set(solution.assignment) == set(problem.nodes)
    assert set(solution.assignment.values()) == set(range(problem.Z))

    for node, zone in solution.assignment.items():
        assert isinstance(zone, int)
        assert zone in problem.candidate_zones(node)

    for zone, centroid in enumerate(problem.centroids):
        assert solution.assignment[centroid] == zone
        zone_nodes = [
            node
            for node, assigned_zone in solution.assignment.items()
            if assigned_zone == zone
        ]
        assert centroid in zone_nodes
        assert nx.is_connected(problem.G.subgraph(zone_nodes))

        students = sum(problem.students(node) for node in zone_nodes)
        assert students > tolerance
        for constraint in balance_constraints(problem):
            value = sum(constraint.value(node) for node in zone_nodes)
            assert value >= constraint.lower_ratio * students - tolerance
            assert value <= constraint.upper_ratio * students + tolerance

    total_schools = sum(problem.num_schools(node) for node in problem.nodes)
    if total_schools:
        average = total_schools / problem.Z
        for zone in range(problem.Z):
            count = sum(
                problem.num_schools(node)
                for node, assigned_zone in solution.assignment.items()
                if assigned_zone == zone
            )
            assert count >= max(0.0, average - 1.0) - tolerance
            assert count <= average + 1.0 + tolerance

    if check_boundary_objective:
        assert solution.objective == pytest.approx(
            boundary_edges(problem.G, solution.assignment),
            abs=tolerance,
        )
    assert solution.wall_time is not None
    assert math.isfinite(solution.wall_time)
    assert solution.wall_time >= 0.0


def assert_no_feasible_solution(solution: ZoneSolution) -> None:
    assert not solution.feasible
    assert solution.assignment == {}
    assert solution.objective is None

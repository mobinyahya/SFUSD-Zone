"""Shared initial-solution helpers for solver warm starts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from gerrychain import Graph

from optimization.data import contiguity
from optimization.problem import ZoneProblem

HINT_METHODS = {"voronoi", "none"}
RECOM_BALANCE_METRICS = {"students", "nodes", "schools"}
RECOM_BALANCE_METRIC_ALIASES = {"num_schools": "schools"}
RECOM_NODE_COUNT_COL = "__recom_node_count"
RECOM_SCHOOL_COUNT_COL = "__recom_school_count"


@dataclass(frozen=True)
class InitialSolution:
    assignment: dict[int, int]
    metadata: dict[str, object]


def normalize_hints(value: object, default: str = "voronoi") -> str:
    method = str(default if value is None else value)
    if method not in HINT_METHODS:
        raise ValueError("hints must be one of: voronoi, none.")
    return method


def normalize_recom_balance_metric(value: object, default: str = "students") -> str:
    metric = str(default if value is None else value)
    metric = RECOM_BALANCE_METRIC_ALIASES.get(metric, metric)
    if metric not in RECOM_BALANCE_METRICS:
        raise ValueError(
            "recom_balance_metric must be one of: students, nodes, schools."
        )
    return metric


def initial_solution(
    problem: ZoneProblem,
    hints: object,
) -> InitialSolution | None:
    """Return a complete candidate-aware initial solution for ``hints``."""

    method = normalize_hints(hints)
    if method == "none":
        return None
    return voronoi_initial_solution(problem)


def voronoi_initial_solution(problem: ZoneProblem) -> InitialSolution:
    assignment = _nearest_centroid_assignment(problem)
    return InitialSolution(
        assignment=assignment,
        metadata={"hints": "voronoi"},
    )


def complete_assignment(
    problem: ZoneProblem,
    seed: Mapping[int, int],
) -> dict[int, int]:
    assignment: dict[int, int] = {}
    for node in problem.nodes:
        zone = seed.get(node)
        candidates = problem.candidate_zones(node)
        if zone in candidates:
            assignment[node] = int(zone)
        else:
            if not candidates:
                raise problem.no_candidate_zones_error(node)
            assignment[node] = min(
                candidates,
                key=lambda z: problem.distance(problem.centroids[z], node),
            )
    for z, centroid in enumerate(problem.centroids):
        assignment[centroid] = z
    return assignment


def _nearest_centroid_assignment(problem: ZoneProblem) -> dict[int, int]:
    assignment = complete_assignment(problem, {})
    repaired = contiguity.repair(problem.G, assignment, problem.centroids)
    return complete_assignment(problem, repaired)


def recom_gerrychain_graph(
    problem: ZoneProblem, balance_metric: object = "students"
) -> Graph:
    metric = normalize_recom_balance_metric(balance_metric)
    if metric == "students":
        return Graph.from_networkx(problem.G)
    graph = problem.G.copy()
    if metric == "nodes":
        for node in graph.nodes:
            graph.nodes[node][RECOM_NODE_COUNT_COL] = 1.0
    elif metric == "schools":
        for node in graph.nodes:
            graph.nodes[node][RECOM_SCHOOL_COUNT_COL] = float(problem.num_schools(node))
    return Graph.from_networkx(graph)


def recom_balance_pop_col(balance_metric: object = "students") -> str:
    metric = normalize_recom_balance_metric(balance_metric)
    if metric == "nodes":
        return RECOM_NODE_COUNT_COL
    if metric == "schools":
        return RECOM_SCHOOL_COUNT_COL
    return "ge_students"


def recom_balance_target(
    problem: ZoneProblem, balance_metric: object = "students"
) -> float:
    metric = normalize_recom_balance_metric(balance_metric)
    if metric == "nodes":
        return len(problem.nodes) / max(1, problem.Z)
    if metric == "schools":
        return sum(problem.num_schools(node) for node in problem.nodes) / max(
            1, problem.Z
        )
    return sum(problem.students(node) for node in problem.nodes) / max(1, problem.Z)


def recom_balance_epsilon(
    problem: ZoneProblem,
    population_epsilon: float | None = None,
    balance_metric: object = "students",
) -> float:
    metric = normalize_recom_balance_metric(balance_metric)
    if population_epsilon is not None:
        epsilon = max(0.01, float(population_epsilon))
    else:
        epsilon = _population_epsilon(problem)
    if metric == "schools":
        target = recom_balance_target(problem, metric)
        if target > 0:
            epsilon = min(epsilon, 1.0 / target)
    return max(0.01, epsilon)


def _population_epsilon(problem: ZoneProblem) -> float:
    tolerances = [problem.shortage, problem.overage, 0.05]
    finite = [float(value) for value in tolerances if math.isfinite(float(value))]
    return max(0.01, min(max(finite) if finite else 1.0, 10.0))

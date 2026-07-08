"""Shared initial-solution helpers for solver warm starts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Mapping

from gerrychain import Graph
from gerrychain.proposals.tree_proposals import MetagraphError
from gerrychain.tree import (
    BalanceError,
    PopulationBalanceError,
    ReselectException,
    bipartition_tree,
    recursive_tree_part,
)

from Zone_Generation.optimization.data import contiguity
from Zone_Generation.optimization.problem import ZoneProblem

HINT_METHODS = {"voronoi", "gerry_chain", "none"}
RECOM_BALANCE_METRICS = {"students", "nodes", "schools"}
RECOM_BALANCE_METRIC_ALIASES = {"num_schools": "schools"}
RECOM_NODE_COUNT_COL = "__recom_node_count"
RECOM_SCHOOL_COUNT_COL = "__recom_school_count"

_EPS = 1e-6
_GERRY_CHAIN_ERRORS = (
    BalanceError,
    PopulationBalanceError,
    ReselectException,
    MetagraphError,
    IndexError,
)


@dataclass(frozen=True)
class InitialSolution:
    assignment: dict[int, int]
    metadata: dict[str, object]


@dataclass(frozen=True, order=True)
class _Score:
    penalty: float
    boundary: int


def normalize_hints(value: object, default: str = "gerry_chain") -> str:
    method = str(default if value is None else value)
    if method not in HINT_METHODS:
        raise ValueError("hints must be one of: voronoi, gerry_chain, none.")
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
    *,
    cut_attempts: int = 100,
    population_epsilon: float | None = None,
    balance_metric: object = "students",
) -> InitialSolution | None:
    """Return a complete candidate-aware initial solution for ``hints``."""

    method = normalize_hints(hints)
    if method == "none":
        return None
    if method == "voronoi":
        return voronoi_initial_solution(problem)
    return gerry_chain_initial_solution(
        problem,
        cut_attempts=cut_attempts,
        population_epsilon=population_epsilon,
        balance_metric=balance_metric,
    )


def voronoi_initial_solution(problem: ZoneProblem) -> InitialSolution:
    assignment = _nearest_centroid_assignment(problem)
    return InitialSolution(
        assignment=assignment,
        metadata={"hints": "voronoi"},
    )


def gerry_chain_initial_solution(
    problem: ZoneProblem,
    *,
    cut_attempts: int = 100,
    population_epsilon: float | None = None,
    balance_metric: object = "students",
) -> InitialSolution:
    requested_metric = normalize_recom_balance_metric(balance_metric)
    metrics = _initial_balance_metrics(problem, requested_metric)
    requested_epsilon = recom_balance_epsilon(
        problem, population_epsilon, requested_metric
    )
    tree_method = partial(bipartition_tree, max_attempts=max(1, int(cut_attempts)))
    best_assignment = None
    best_score = None
    best_metadata = None
    best_epsilon = requested_epsilon
    best_metric = requested_metric
    metric_attempts: dict[str, int] = {}
    errors: list[str] = []

    for metric in metrics:
        target = recom_balance_target(problem, metric)
        if problem.Z < 2 or target <= 0:
            continue
        epsilon = recom_balance_epsilon(problem, population_epsilon, metric)
        graph = recom_gerrychain_graph(problem, metric)
        pop_col = recom_balance_pop_col(metric)
        for attempt, current_epsilon in enumerate(
            _epsilon_schedule(epsilon), start=1
        ):
            metric_attempts[metric] = attempt
            try:
                raw = recursive_tree_part(
                    graph,
                    parts=list(range(problem.Z)),
                    pop_target=target,
                    pop_col=pop_col,
                    epsilon=current_epsilon,
                    method=tree_method,
                )
            except _GERRY_CHAIN_ERRORS as exc:
                errors.append(f"{metric}:{type(exc).__name__}")
                continue

            assignment = _normalize_gerry_chain_assignment(problem, raw)
            score = _score(problem, assignment)
            metadata = _gerry_chain_metadata(
                metric=metric,
                requested_metric=requested_metric,
                metrics=metrics,
                attempts=attempt,
                metric_attempts=metric_attempts,
                target=target,
                epsilon=current_epsilon,
            )
            if best_score is None or score < best_score:
                best_assignment = assignment
                best_score = score
                best_metadata = metadata
                best_epsilon = current_epsilon
                best_metric = metric
            if _valid(score):
                return InitialSolution(assignment=assignment, metadata=metadata)

    if best_assignment is not None:
        metadata = dict(best_metadata or {})
        metadata.update(
            {
                "gerry_chain_balance_metric": best_metric,
                "gerry_chain_population_epsilon": best_epsilon,
                "gerry_chain_initial_penalty": best_score.penalty
                if best_score
                else None,
            }
        )
        if errors:
            metadata["gerry_chain_initial_errors"] = errors[-3:]
        return InitialSolution(
            assignment=best_assignment,
            metadata=metadata,
        )

    result = _gerry_chain_fallback(
        problem,
        recom_balance_target(problem, requested_metric),
        requested_epsilon,
        requested_metric,
    )
    result.metadata["gerry_chain_initial_balance_metrics"] = metrics
    result.metadata["gerry_chain_initial_metric_attempts"] = metric_attempts
    if errors:
        result.metadata["gerry_chain_initial_errors"] = errors[-3:]
    return result


def _initial_balance_metrics(problem: ZoneProblem, requested_metric: str) -> list[str]:
    metrics = [requested_metric]
    for metric in ("students", "schools"):
        if metric not in metrics and recom_balance_target(problem, metric) > 0:
            metrics.append(metric)
    return metrics


def _gerry_chain_metadata(
    *,
    metric: str,
    requested_metric: str,
    metrics: list[str],
    attempts: int,
    metric_attempts: dict[str, int],
    target: float,
    epsilon: float,
) -> dict[str, object]:
    return {
        "hints": "gerry_chain",
        "gerry_chain_initial_attempts": attempts,
        "gerry_chain_initial_balance_metrics": list(metrics),
        "gerry_chain_initial_metric_attempts": dict(metric_attempts),
        "gerry_chain_requested_balance_metric": requested_metric,
        "gerry_chain_balance_metric": metric,
        "gerry_chain_population_target": target,
        "gerry_chain_population_epsilon": epsilon,
    }


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


def _gerry_chain_fallback(
    problem: ZoneProblem,
    target: float,
    epsilon: float,
    balance_metric: str,
) -> InitialSolution:
    return InitialSolution(
        assignment=_nearest_centroid_assignment(problem),
        metadata={
            "hints": "gerry_chain",
            "hint_fallback": "voronoi",
            "gerry_chain_balance_metric": balance_metric,
            "gerry_chain_population_target": target,
            "gerry_chain_population_epsilon": epsilon,
        },
    )


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
            graph.nodes[node][RECOM_SCHOOL_COUNT_COL] = float(
                problem.num_schools(node)
            )
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


def _normalize_gerry_chain_assignment(
    problem: ZoneProblem,
    raw: Mapping[int, int],
) -> dict[int, int]:
    relabeled = _relabel_parts_by_centroids(problem, raw)
    completed = complete_assignment(problem, relabeled)
    repaired = contiguity.repair(problem.G, completed, problem.centroids)
    return complete_assignment(problem, repaired)


def _relabel_parts_by_centroids(
    problem: ZoneProblem,
    raw: Mapping[int, int],
) -> dict[int, int]:
    part_to_zone: dict[int, int] = {}
    used_zones: set[int] = set()
    for z, centroid in enumerate(problem.centroids):
        part = raw.get(centroid)
        if part is None or part in part_to_zone:
            continue
        part_to_zone[int(part)] = z
        used_zones.add(z)

    remaining_zones = [z for z in range(problem.Z) if z not in used_zones]
    remaining_parts = sorted({int(part) for part in raw.values()} - set(part_to_zone))
    for part, zone in zip(remaining_parts, remaining_zones):
        part_to_zone[part] = zone

    if not part_to_zone:
        return {}
    fallback_zone = remaining_zones[0] if remaining_zones else 0
    return {
        int(node): int(part_to_zone.get(int(part), fallback_zone))
        for node, part in raw.items()
    }


def _score(problem: ZoneProblem, assignment: Mapping[int, int]) -> _Score:
    penalty = 0.0
    hard_penalty = float(problem.A + problem.Z + 1) * 1000.0

    if set(assignment) != set(problem.nodes):
        missing = set(problem.nodes) - set(assignment)
        extra = set(assignment) - set(problem.nodes)
        penalty += hard_penalty * (len(missing) + len(extra))

    for node in problem.nodes:
        zone = assignment.get(node)
        if zone not in problem.candidate_zones(node):
            penalty += hard_penalty

    for z, centroid in enumerate(problem.centroids):
        if assignment.get(centroid) != z:
            penalty += hard_penalty

    if set(assignment) >= set(problem.nodes) and not contiguity.is_contiguous(
        problem.G, dict(assignment), problem.centroids
    ):
        penalty += hard_penalty

    penalty += _balance_penalty(problem, assignment)
    penalty += _school_count_penalty(problem, assignment)
    return _Score(
        penalty=penalty,
        boundary=contiguity.boundary_edges(problem.G, dict(assignment)),
    )


def _balance_penalty(problem: ZoneProblem, assignment: Mapping[int, int]) -> float:
    from Zone_Generation.optimization.solvers.balance import balance_constraints

    penalty = 0.0
    for z in range(problem.Z):
        nodes = [n for n in problem.nodes if assignment.get(n) == z]
        students = sum(problem.students(n) for n in nodes)
        for constraint in balance_constraints(problem):
            value = sum(constraint.value(n) for n in nodes)
            lower = constraint.lower_ratio * students
            upper = constraint.upper_ratio * students
            if value < lower:
                penalty += lower - value
            if value > upper:
                penalty += value - upper
    return penalty


def _school_count_penalty(problem: ZoneProblem, assignment: Mapping[int, int]) -> float:
    total = sum(problem.num_schools(n) for n in problem.nodes)
    if total == 0:
        return 0.0
    avg = total / problem.Z
    lower = max(0.0, avg - 1.0)
    upper = avg + 1.0
    penalty = 0.0
    for z in range(problem.Z):
        schools = sum(
            problem.num_schools(n) for n in problem.nodes if assignment.get(n) == z
        )
        if schools < lower:
            penalty += lower - schools
        if schools > upper:
            penalty += schools - upper
    return penalty


def _valid(score: _Score) -> bool:
    return score.penalty <= _EPS


def _population_target(problem: ZoneProblem) -> float:
    return sum(problem.students(node) for node in problem.nodes) / max(1, problem.Z)


def _population_epsilon(problem: ZoneProblem) -> float:
    tolerances = [problem.shortage, problem.overage, 0.05]
    finite = [float(value) for value in tolerances if math.isfinite(float(value))]
    return max(0.01, min(max(finite) if finite else 1.0, 10.0))


def _epsilon_schedule(epsilon: float) -> list[float]:
    values = [epsilon, max(epsilon, 0.10), max(epsilon, 0.25), max(epsilon, 0.50)]
    values.append(max(epsilon, 1.0))
    out = []
    for value in values:
        value = float(value)
        if value not in out:
            out.append(value)
    return out

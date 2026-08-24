"""Shared initial-solution helpers for solver warm starts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from optimization.data import contiguity
from optimization.problem import ZoneProblem

HINT_METHODS = {"feasible", "voronoi", "none"}


@dataclass(frozen=True)
class InitialSolution:
    assignment: dict[int, int]
    metadata: dict[str, object]


def normalize_hints(value: object, default: str = "voronoi") -> str:
    method = str(default if value is None else value)
    if method not in HINT_METHODS:
        raise ValueError("hints must be one of: feasible, voronoi, none.")
    return method


def initial_solution(
    problem: ZoneProblem,
    hints: object,
    *,
    solver_options: Mapping[str, object] | None = None,
) -> InitialSolution | None:
    """Return a complete candidate-aware initial solution for ``hints``."""

    method = normalize_hints(hints)
    if method == "none":
        return None
    if method == "feasible":
        return feasible_initial_solution(problem, solver_options=solver_options)
    return voronoi_initial_solution(problem)


def feasible_initial_solution(
    problem: ZoneProblem,
    *,
    solver_options: Mapping[str, object] | None = None,
) -> InitialSolution:
    """Find one zoning-feasible assignment without an optimization objective."""

    options = solver_options or {}
    time_limit = options.get("feasible_hint_time_limit", 60.0)
    if isinstance(time_limit, bool):
        raise ValueError("feasible_hint_time_limit must be positive.")
    try:
        time_limit = float(time_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("feasible_hint_time_limit must be positive.") from exc
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("feasible_hint_time_limit must be positive.")

    # Import lazily because CP-SAT also consumes this shared hint interface.
    from optimization.solvers.cpsat import CpBoolSolver

    solver = CpBoolSolver(
        solve_time_limit=time_limit,
        seed=int(options.get("seed", 42)),
        workers=int(options.get("workers", 8)),
        hints="voronoi",
        centroid_neighbor_radius=int(options.get("centroid_neighbor_radius", 0)),
        linearization_level=options.get("linearization_level"),
        cp_model_probing_level=options.get("cp_model_probing_level"),
        symmetry_level=options.get("symmetry_level"),
        cp_sat_search_strategy=options.get("cp_sat_search_strategy"),
    )
    solution = solver.find_feasible_solution(problem)
    if not solution.feasible:
        raise RuntimeError(
            "Could not find a zoning-feasible hint within "
            f"{time_limit:g} seconds (status={solution.status})."
        )
    return InitialSolution(
        assignment=solution.assignment,
        metadata={
            "hints": "feasible",
            "hint_solver": "cp_bool",
            "hint_solver_status": solution.status,
            "hint_solver_wall_time_seconds": solution.wall_time,
        },
    )


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

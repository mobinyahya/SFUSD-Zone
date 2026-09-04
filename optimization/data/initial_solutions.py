"""Shared initial-solution helpers for solver warm starts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

from loaders import CacheNamespace, CacheStore, DataScenario
from optimization.data import contiguity
from optimization.data.closer_neighbors import CLOSER_NEIGHBORS_GRAPH_KEY
from optimization.problem import ZoneProblem

HINT_METHODS = {"feasible", "voronoi", "none"}

FEASIBLE_HINT_CACHE_SCHEMA_VERSION = 1
FEASIBLE_HINT_ARTIFACT = "feasible_hint"
FEASIBLE_HINT_PAYLOAD = "hint.pickle"


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
    """Find one zoning-feasible assignment without an optimization objective.

    The solve is reused through the shared content-addressed cache, keyed by
    the feasibility model itself (:func:`feasibility_fingerprint`) plus every
    CP-SAT search setting that shapes the search, including the seed. Problems
    built without an originating config carry no scenario and are never cached.
    """

    options = solver_options or {}
    time_limit = _feasible_hint_time_limit(options)

    namespace = _feasible_hint_namespace(problem, options, time_limit=time_limit)
    if namespace is not None:
        cached = _cached_hint(problem, namespace.load_pickle(FEASIBLE_HINT_PAYLOAD))
        if cached is not None:
            return InitialSolution(
                assignment=cached["assignment"],
                metadata={
                    "hints": "feasible",
                    "hint_solver": "cp_bool",
                    "hint_solver_status": cached["status"],
                    "hint_solver_wall_time_seconds": cached["wall_time"],
                    "hint_cache": "hit",
                    "hint_cache_key": namespace.key,
                },
            )

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

    metadata: dict[str, object] = {
        "hints": "feasible",
        "hint_solver": "cp_bool",
        "hint_solver_status": solution.status,
        "hint_solver_wall_time_seconds": solution.wall_time,
    }
    if namespace is not None:
        namespace.save_pickle(
            FEASIBLE_HINT_PAYLOAD,
            {
                "assignment": {
                    int(node): int(zone) for node, zone in solution.assignment.items()
                },
                "status": solution.status,
                "wall_time": solution.wall_time,
            },
        )
        metadata["hint_cache"] = "miss"
        metadata["hint_cache_key"] = namespace.key

    return InitialSolution(assignment=solution.assignment, metadata=metadata)


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


# ---------------------------------------------------------------------- #
# feasible-hint cache
# ---------------------------------------------------------------------- #
def feasibility_fingerprint(problem: ZoneProblem) -> str:
    """Hash every problem input the objective-free zoning model reads.

    Two problems sharing a fingerprint have the same feasible set, so a hint
    found for one is a valid hint for the other. The objective is excluded
    because the hint solve ignores it.
    """

    digest = hashlib.sha256()
    ethnicities = problem.ethnicities
    racial = problem.district_racial

    _write(digest, ["level", problem.level.name, problem.program_population])
    _write(
        digest,
        [
            "limits",
            float(problem.frl_dev),
            float(problem.racial_dev),
            float(problem.overage),
            float(problem.shortage),
            float(problem.max_distance),
            float(problem.boundary_prop),
            bool(problem.weight_edges),
        ],
    )
    _write(
        digest,
        [
            "district",
            float(problem.district_frl),
            [float(racial[ethnicity]) for ethnicity in ethnicities],
        ],
    )
    _write(
        digest,
        [
            "centroids",
            [int(node) for node in problem.centroids],
            [int(school_id) for school_id in problem.centroid_school_ids],
        ],
    )

    relation = problem.G.graph.get(CLOSER_NEIGHBORS_GRAPH_KEY) or {}
    for node in sorted(int(node) for node in problem.G.nodes()):
        supports = relation.get(node) or {}
        _write(
            digest,
            [
                "node",
                node,
                sorted(int(zone) for zone in problem.candidate_zones(node)),
                float(problem.students(node)),
                float(problem.capacity(node)),
                float(problem.frl(node)),
                int(problem.num_schools(node)),
                [float(problem.ethnicity(node, e)) for e in ethnicities],
                [
                    _closer_support(supports, school_id)
                    for school_id in problem.centroid_school_ids
                ],
            ],
        )

    edges = sorted(tuple(sorted((int(u), int(v)))) for u, v in problem.G.edges())
    _write(
        digest,
        ["edges", [[u, v, int(problem.boundary_weight(u, v))] for u, v in edges]],
    )
    _write(digest, ["fixed", _sorted_assignment(problem.fixed)])
    _write(digest, ["hint", _sorted_assignment(problem.hint)])
    return digest.hexdigest()


def _closer_support(supports: Mapping[int, object], school_id: object) -> object:
    neighbors = supports.get(int(school_id))
    if neighbors is None:
        # Missing relation entries fail with a precise message in the solver.
        return None
    return sorted(int(neighbor) for neighbor in neighbors)


def _sorted_assignment(value: Mapping[int, int] | None) -> list[list[int]]:
    if not value:
        return []
    return sorted([int(node), int(zone)] for node, zone in value.items())


def _write(digest: "hashlib._Hash", value: object) -> None:
    """Feed one unambiguously encoded value into ``digest``."""

    if value is None:
        digest.update(b"n")
    elif isinstance(value, bool):
        digest.update(b"b1" if value else b"b0")
    elif isinstance(value, int):
        digest.update(b"i" + str(value).encode("ascii"))
    elif isinstance(value, float):
        digest.update(b"f" + format(value, ".17g").encode("ascii"))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"s" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, (list, tuple)):
        digest.update(b"[" + len(value).to_bytes(8, "big"))
        for item in value:
            _write(digest, item)
    else:
        raise TypeError(f"Cannot fingerprint {value!r} for the feasible-hint cache.")
    digest.update(b";")


def _feasible_hint_time_limit(options: Mapping[str, object]) -> float:
    time_limit = options.get("feasible_hint_time_limit", 60.0)
    if isinstance(time_limit, bool):
        raise ValueError("feasible_hint_time_limit must be positive.")
    try:
        time_limit = float(time_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("feasible_hint_time_limit must be positive.") from exc
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("feasible_hint_time_limit must be positive.")
    return time_limit


def _feasible_hint_namespace(
    problem: ZoneProblem,
    options: Mapping[str, object],
    *,
    time_limit: float,
) -> CacheNamespace | None:
    """Resolve the cache namespace for one hint solve, or ``None`` if unusable."""

    scenario = _hint_scenario(problem)
    if scenario is None:
        return None
    return CacheStore(scenario).namespace(
        FEASIBLE_HINT_ARTIFACT,
        {
            "problem": feasibility_fingerprint(problem),
            "hint_solver": "cp_bool",
            "hint_solver_hints": "voronoi",
            "feasible_hint_time_limit": time_limit,
            "seed": int(options.get("seed", 42)),
            "workers": int(options.get("workers", 8)),
            "centroid_neighbor_radius": int(options.get("centroid_neighbor_radius", 0)),
            "linearization_level": options.get("linearization_level"),
            "cp_model_probing_level": options.get("cp_model_probing_level"),
            "symmetry_level": options.get("symmetry_level"),
            "cp_sat_search_strategy": options.get("cp_sat_search_strategy"),
        },
        schema_version=FEASIBLE_HINT_CACHE_SCHEMA_VERSION,
        # The fingerprint already pins every source-derived model input, so the
        # namespace only needs the scenario's identity and selectors.
        roles=(),
    )


def _hint_scenario(problem: ZoneProblem) -> DataScenario | None:
    config = problem.optimization_config
    if config is None:
        return None
    scenario = getattr(config, "data_scenario", None)
    return scenario if isinstance(scenario, DataScenario) else None


def _cached_hint(problem: ZoneProblem, payload: object) -> dict[str, Any] | None:
    """Return a cached hint only when it is still a valid assignment."""

    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("assignment")
    if not isinstance(raw, Mapping):
        return None
    try:
        assignment = {int(node): int(zone) for node, zone in raw.items()}
    except (TypeError, ValueError):
        return None
    if set(assignment) != {int(node) for node in problem.nodes}:
        return None
    if any(
        zone not in problem.candidate_zones(node) for node, zone in assignment.items()
    ):
        return None
    if any(
        assignment.get(int(centroid)) != zone
        for zone, centroid in enumerate(problem.centroids)
    ):
        return None
    if not contiguity.is_contiguous(problem.G, assignment, problem.centroids):
        return None
    return {
        "assignment": assignment,
        "status": payload.get("status"),
        "wall_time": payload.get("wall_time"),
    }

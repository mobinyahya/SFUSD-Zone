"""Shared initial-solution helpers for solver warm starts.

Cache-key policy for the feasible hint
--------------------------------------
The hint solve is expensive and its result is interchangeable: any point inside
the feasible set is as good a warm start as any other. So the cache key names
the *feasible set* and nothing else, and a run is meant to lean on a hint some
other run found -- including one that searched with far more resources than it
has. A 1-worker Slurm task reusing what a 16-worker task proved is the point of
the cache, not a bug in it.

That splits every input to the hint solve into two lists, and the split is
enforced rather than documented: :func:`_hint_solver_options` builds the solver
and raises if it is about to pass a name neither list classifies, and
:func:`_hint_model_identity` raises if a feasibility-affecting name is missing
from the key.

``_HINT_FEASIBILITY_OPTIONS``
    Changes *which* assignments are feasible. Must be in the key -- two runs
    that disagree here are asking different questions and cannot share an
    answer. ``centroid_neighbor_radius`` is the only one today: a positive
    radius pins each centroid's graph-hop neighbourhood, shrinking the feasible
    set in a way the problem fingerprint cannot see.

``_HINT_SEARCH_OPTIONS``
    Steers the search for a point inside an unchanged feasible set, so it must
    stay *out* of the key. ``workers``, ``seed`` and
    ``feasible_hint_time_limit`` are the obvious ones -- a run that searched
    harder proves the same thing, only more often. The CP-SAT tuning knobs
    (``linearization_level``, ``cp_model_probing_level``, ``symmetry_level``,
    ``cp_sat_search_strategy``) belong here too -- they change the encoding
    CP-SAT searches over and which solution it lands on, never the set of
    assignments that satisfy the model.

Options are not the whole story, so the key also carries
:func:`_hint_model_identity`: the encoding, the coefficient scale and the
rounding direction, read from the solver module rather than restated here.
Those are code-level choices that move the feasible set just as surely as an
option would, and a hint found under one of them is not valid under another.
Adding a new option to the hint solver means putting its name in one of the two
lists above; changing the model's arithmetic means the identity changes with it.

Nothing here can make a stored hint trustworthy on its own, so every hint is
checked against :func:`optimization.data.feasibility.check_zoning` before it is
stored and again after it is read back.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

from loaders import CacheNamespace, CacheStore, DataScenario
from optimization.data import contiguity
from optimization.data.closer_neighbors import CLOSER_NEIGHBORS_GRAPH_KEY
from optimization.data.feasibility import check_zoning
from optimization.problem import ZoneProblem

HINT_METHODS = {"feasible", "voronoi", "none"}


class FeasibleHintError(RuntimeError):
    """The bounded feasibility search found no zoning hint."""


# Bumped to 3 when the CP-SAT balance rows moved to conservative rounding: the
# v2 hints were solutions of a slightly looser model and some of them break the
# exact constraints.
FEASIBLE_HINT_CACHE_SCHEMA_VERSION = 3
FEASIBLE_HINT_ARTIFACT = "feasible_hint"
FEASIBLE_HINT_PAYLOAD = "hint.pickle"

# See the module docstring. Every option the hint solver reads must be in one.
_HINT_FEASIBILITY_OPTIONS = ("centroid_neighbor_radius",)
_HINT_SEARCH_OPTIONS = (
    "feasible_hint_time_limit",
    "seed",
    "workers",
    "linearization_level",
    "cp_model_probing_level",
    "symmetry_level",
    "cp_sat_search_strategy",
)
_HINT_SOLVER_NAME = "cp_bool"


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

    The solve is reused through the shared content-addressed cache, keyed by the
    feasibility model itself and nothing that only steers the search -- see this
    module's docstring for the policy and for why ``workers`` in particular is
    excluded. Problems built without an originating config carry no scenario and
    are never cached.

    Every hint is checked against
    :func:`optimization.data.feasibility.check_zoning` before it is stored and
    again after it is read back, so a solver that is feasible only in its own
    arithmetic cannot poison the cache for later runs.
    """

    options = solver_options or {}
    time_limit = _feasible_hint_time_limit(options)

    namespace = _feasible_hint_namespace(problem, options)
    rejected: str | None = None
    if namespace is not None:
        cached = _cached_hint(namespace.load_pickle(FEASIBLE_HINT_PAYLOAD))
        if cached is not None:
            report = check_zoning(problem, cached["assignment"])
            if report.feasible:
                return InitialSolution(
                    assignment=cached["assignment"],
                    metadata={
                        "hints": "feasible",
                        "hint_solver": _HINT_SOLVER_NAME,
                        "hint_solver_status": cached["status"],
                        "hint_solver_wall_time_seconds": cached["wall_time"],
                        "hint_cache": "hit",
                        "hint_cache_key": namespace.key,
                    },
                )
            # Solve again rather than hand on a hint that does not survive the
            # exact check; the fresh one overwrites this payload below.
            rejected = report.describe()

    # Import lazily because CP-SAT also consumes this shared hint interface.
    from optimization.solvers.cpsat import CpBoolSolver

    solver = CpBoolSolver(
        solve_time_limit=time_limit,
        hints="voronoi",
        **_hint_solver_options(options),
    )
    solution = solver.find_feasible_solution(problem)
    if not solution.feasible:
        raise FeasibleHintError(
            "Could not find a zoning-feasible hint within "
            f"{time_limit:g} seconds (status={solution.status})."
        )

    report = check_zoning(problem, solution.assignment)
    if not report.feasible:
        raise FeasibleHintError(
            f"{_HINT_SOLVER_NAME} reported a feasible hint that fails the exact "
            f"feasibility check -- {report.describe()}."
        )

    metadata: dict[str, object] = {
        "hints": "feasible",
        "hint_solver": _HINT_SOLVER_NAME,
        "hint_solver_status": solution.status,
        "hint_solver_wall_time_seconds": solution.wall_time,
    }
    if rejected is not None:
        metadata["hint_cache_rejected_reason"] = rejected
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
        metadata["hint_cache"] = "rejected" if rejected else "miss"
        metadata["hint_cache_key"] = namespace.key

    return InitialSolution(assignment=solution.assignment, metadata=metadata)


def _hint_solver_options(options: Mapping[str, object]) -> dict[str, object]:
    """Solver keyword arguments for the hint solve, with every name classified.

    Raises if a name is in neither list, so a new hint-solver option cannot be
    added without deciding whether it belongs in the cache key.
    """

    classified = set(_HINT_FEASIBILITY_OPTIONS) | set(_HINT_SEARCH_OPTIONS)
    passed = {
        "seed": int(options.get("seed", 42)),
        "workers": int(options.get("workers", 8)),
        "centroid_neighbor_radius": int(options.get("centroid_neighbor_radius", 0)),
        "linearization_level": options.get("linearization_level"),
        "cp_model_probing_level": options.get("cp_model_probing_level"),
        "symmetry_level": options.get("symmetry_level"),
        "cp_sat_search_strategy": options.get("cp_sat_search_strategy"),
    }
    unclassified = sorted(set(passed) - classified)
    if unclassified:
        raise ValueError(
            "Feasible-hint solver options must be classified as feasibility- or "
            f"search-affecting before use: {unclassified}. See the cache-key "
            "policy in optimization.data.initial_solutions."
        )
    return passed


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
def feasibility_fingerprint(problem: ZoneProblem, *, include_hint: bool = True) -> str:
    """Hash every problem input the objective-free zoning model reads.

    Two problems sharing a fingerprint have the same feasible set, so a hint
    found for one is a valid hint for the other. The objective is excluded
    because the hint solve ignores it.

    ``include_hint`` keeps the warm start in the digest, which is what the hint
    cache wants -- a hint solve is steered by the incoming hint. Callers that
    only need the *feasible set* identified, such as the zoning relaxation in
    :mod:`optimization.zoned_transport`, pass ``False``.
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
    if include_hint:
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
) -> CacheNamespace | None:
    """Resolve the cache namespace for one hint solve, or ``None`` if unusable.

    The key is the feasible set and nothing else: the problem fingerprint for
    everything the model reads off the problem, plus
    :func:`_hint_model_identity` for what it reads off the code and the
    feasibility-affecting options. Search settings are absent by construction --
    see the module docstring.
    """

    scenario = _hint_scenario(problem)
    if scenario is None:
        return None
    return CacheStore(scenario).namespace(
        FEASIBLE_HINT_ARTIFACT,
        {
            "problem": feasibility_fingerprint(problem),
            **_hint_model_identity(options),
        },
        schema_version=FEASIBLE_HINT_CACHE_SCHEMA_VERSION,
        # The fingerprint already pins every source-derived model input, so the
        # namespace only needs the scenario's identity and selectors.
        roles=(),
    )


def _hint_model_identity(options: Mapping[str, object]) -> dict[str, object]:
    """Everything outside the problem that decides the hint's feasible set.

    The scale and rounding come from the solver module rather than being
    restated here, so a change to either invalidates the hints the old
    arithmetic produced instead of silently re-serving them.
    """

    from optimization.solvers.cpsat import COEFFICIENT_ROUNDING, CP_SAT_SCALE

    identity: dict[str, object] = {
        "solver": _HINT_SOLVER_NAME,
        "coefficient_scale": CP_SAT_SCALE,
        "coefficient_rounding": COEFFICIENT_ROUNDING,
        "centroid_neighbor_radius": int(options.get("centroid_neighbor_radius", 0)),
    }
    unkeyed = [name for name in _HINT_FEASIBILITY_OPTIONS if name not in identity]
    if unkeyed:
        raise ValueError(
            f"Feasibility-affecting hint options {unkeyed} are missing from the "
            "cache key; two runs that disagree on them would share one hint."
        )
    return identity


def _hint_scenario(problem: ZoneProblem) -> DataScenario | None:
    config = problem.optimization_config
    if config is None:
        return None
    scenario = getattr(config, "data_scenario", None)
    return scenario if isinstance(scenario, DataScenario) else None


def _cached_hint(payload: object) -> dict[str, Any] | None:
    """Parse one cache payload into a hint, or ``None`` if it is not one.

    Only the payload's shape is judged here. Whether the assignment is still a
    feasible zoning is :func:`~optimization.data.feasibility.check_zoning`'s
    call, which keeps a malformed payload distinguishable from a well-formed
    one that no longer holds.
    """

    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("assignment")
    if not isinstance(raw, Mapping) or not raw:
        return None
    try:
        assignment = {int(node): int(zone) for node, zone in raw.items()}
    except (TypeError, ValueError):
        return None
    return {
        "assignment": assignment,
        "status": payload.get("status"),
        "wall_time": payload.get("wall_time"),
    }

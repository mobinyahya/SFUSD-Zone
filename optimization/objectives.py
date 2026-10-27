"""Independent method/objective capabilities and single-problem dispatch.

The pipeline supplies a fully constrained ``ZoneProblem``. This module chooses
an implementation without changing its feasible set. Objective-dependent market
code is imported only for stages that actually need it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optimization.problem import ZoneProblem
    from optimization.solution import ZoneSolution


METHODS = (
    "cp_bool",
    "cp_int",
    "mip",
    "recom",
)
_HEURISTICS = frozenset({"recom"})


@dataclass(frozen=True)
class Objective:
    name: str
    methods: frozenset[str]
    maximize: bool = False
    reads_assignment_market: bool = False
    compactness_objective: str = "cut_edges"
    unit: str | None = None


_OBJECTIVES = {
    "cut_edges": Objective("cut_edges", frozenset(METHODS)),
    "boundary_perimeter": Objective(
        "boundary_perimeter", frozenset(METHODS), unit="meter"
    ),
    "average_distance": Objective(
        "average_distance",
        frozenset(METHODS),
        compactness_objective="average_distance",
        unit="mile",
    ),
    "average_squared_distance": Objective(
        "average_squared_distance",
        frozenset(METHODS),
        compactness_objective="school_distance",
        unit="mile^2",
    ),
    "school_distance": Objective(
        "school_distance",
        frozenset(METHODS),
        compactness_objective="school_distance",
        unit="mile^2",
    ),
    "matching_welfare": Objective(
        "matching_welfare", _HEURISTICS | {"cp_bool", "cp_int"}, True, True
    ),
    "saa_welfare": Objective("saa_welfare", frozenset(METHODS), True, True),
    "choice_welfare": Objective("choice_welfare", frozenset(METHODS), True, True),
    "capacity_match": Objective("capacity_match", _HEURISTICS, False, True),
    "aa_capacity_match": Objective("aa_capacity_match", _HEURISTICS, False, True),
    "fill_polarization": Objective("fill_polarization", _HEURISTICS, False, True),
    # A weighted sum of the objectives named in ``objective_weights``.
    "weighted_sum": Objective("weighted_sum", _HEURISTICS, False, True),
}
OBJECTIVES = tuple(_OBJECTIVES)


def get_objective(name: str) -> Objective:
    name = "matching_welfare" if name == "welfare" else name
    try:
        return _OBJECTIVES[name]
    except KeyError:
        raise ValueError(
            f"Unknown objective {name!r}. Choose one of: {', '.join(OBJECTIVES)}."
        ) from None


def supported_objectives(method: str) -> tuple[str, ...]:
    if method not in METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Choose one of: {', '.join(METHODS)}."
        )
    return tuple(name for name, spec in _OBJECTIVES.items() if method in spec.methods)


def validate_method_objective(method: str, objective: str) -> None:
    """Fail before loading data when a requested formulation is unavailable."""
    supported = supported_objectives(method)
    spec = get_objective(objective)
    if spec.name not in supported:
        raise ValueError(
            f"Method {method!r} does not support objective {objective!r}. "
            f"Supported objectives: {', '.join(supported)}."
        )


# Terms a weighted sum may combine: every batch-scored heuristic objective.
# The boundary objectives are scored inside the walk, not in batches.
WEIGHTED_SUM_TERMS = tuple(
    name
    for name, spec in _OBJECTIVES.items()
    if _HEURISTICS <= spec.methods
    and name not in {"cut_edges", "boundary_perimeter", "weighted_sum"}
)
_MID_TERMS = frozenset(
    {"matching_welfare", "capacity_match", "aa_capacity_match", "fill_polarization"}
)


def checked_objective_weights(weights) -> dict[str, float] | None:
    """``objective_weights`` as ``{term: weight}``, or ``None`` when unset."""
    import math

    if weights is None:
        return None
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("objective_weights must be a nonempty mapping of term: weight.")
    checked = {}
    for name, weight in weights.items():
        term = get_objective(name).name
        if term not in WEIGHTED_SUM_TERMS:
            raise ValueError(
                f"objective_weights term {name!r} is not one of: "
                f"{', '.join(WEIGHTED_SUM_TERMS)}."
            )
        if term in checked:
            raise ValueError(f"objective_weights names {term!r} twice.")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                f"objective_weights[{name!r}] must be finite and non-negative."
            )
        checked[term] = float(weight)
    if not any(checked.values()):
        raise ValueError("objective_weights needs a positive weight.")
    return checked


class _WeightedSumScorer:
    """``sum_t w_t * cost_t``, a maximized term entering with its sign flipped.

    MID terms share one market and outcome cache, so a weighted sum of several
    costs one matching per zoning.
    """

    def __init__(self, terms):
        self.terms = tuple(terms)  # (name, weight, sign, scorer)

    def __call__(self, assignments, base=None) -> tuple[float, ...]:
        totals = [0.0] * len(assignments)
        for _, weight, sign, scorer in self.terms:
            for index, value in enumerate(scorer(assignments, base)):
                totals[index] += weight * sign * float(value)
        return tuple(totals)

    def describe(self, assignment) -> dict[str, float]:
        details: dict[str, float] = {}
        for name, weight, _, scorer in self.terms:
            details[f"objective_term_{name}"] = float(scorer((assignment,), None)[0])
            details[f"objective_weight_{name}"] = weight
            describe = getattr(scorer, "describe", None)
            if describe is not None:
                details.update(describe(assignment))
        return details

    def close(self) -> None:
        for _, _, _, scorer in self.terms:
            close = getattr(scorer, "close", None)
            if close is not None:
                close()


def _weighted_sum_scorer(problem: ZoneProblem, options: Mapping[str, Any]):
    weights = checked_objective_weights(options.get("objective_weights"))
    if weights is None:
        raise ValueError("The weighted_sum objective needs objective_weights.")
    weights = {name: weight for name, weight in weights.items() if weight > 0}
    mid_terms = tuple(name for name in weights if name in _MID_TERMS)
    scorers = {}
    if mid_terms:
        from optimization.solvers.burst_objectives import mid_burst_scorers

        metrics = {
            name: "welfare" if name == "matching_welfare" else name
            for name in mid_terms
        }
        shared = mid_burst_scorers(problem, tuple(metrics.values()), options)
        scorers.update({name: shared[metric] for name, metric in metrics.items()})
    try:
        for name in weights:
            if name not in scorers:
                scorers[name] = heuristic_scorer(problem, name, options)
    except Exception:
        for scorer in scorers.values():
            close = getattr(scorer, "close", None)
            if close is not None:
                close()
        raise
    return _WeightedSumScorer(
        (name, weight, -1.0 if get_objective(name).maximize else 1.0, scorers[name])
        for name, weight in weights.items()
    )


def heuristic_scorer(problem: ZoneProblem, objective: str, options: Mapping[str, Any]):
    """A batch scorer in the objective's own units; cut edges need no scorer."""
    spec = get_objective(objective)
    objective = spec.name
    if objective in {"cut_edges", "boundary_perimeter"}:
        return None
    if objective == "weighted_sum":
        return _weighted_sum_scorer(problem, options)
    if spec.compactness_objective in {
        "school_distance",
        "average_distance",
    }:
        return _CentroidDistanceScorer(
            replace(problem, compactness_objective=spec.compactness_objective)
        )
    if objective in {"choice_welfare", "saa_welfare"}:
        from optimization.solvers.welfare_scorers import (
            ChoiceWelfareScorer,
            SaaWelfareScorer,
        )

        scorer_class = (
            ChoiceWelfareScorer if objective == "choice_welfare" else SaaWelfareScorer
        )
        return scorer_class(problem, options)
    from optimization.solvers.burst_objectives import mid_burst_scorer

    return mid_burst_scorer(
        problem, "welfare" if objective == "matching_welfare" else objective, options
    )


class _CentroidDistanceScorer:
    """The exact CP/MIP school-distance cost, evaluated over whole partitions."""

    def __init__(self, problem: ZoneProblem):
        self.costs = problem.distance_objective_costs()

    def __call__(self, assignments, base=None) -> tuple[float, ...]:
        return tuple(
            sum(self.costs[zone, node] for node, zone in assignment.items())
            for assignment in assignments
        )


def _heuristic_domains(
    problem: ZoneProblem, options: Mapping[str, Any]
) -> dict[int, set[int]]:
    """Materialize every domain row, including centroid-neighborhood ownership."""
    from optimization.solvers.base import (
        checked_centroid_neighbor_radius,
        centroid_neighborhoods,
    )

    domains = {node: set(problem.candidate_zones(node)) for node in problem.nodes}
    for node, zone in (problem.fixed or {}).items():
        domains[node] &= {zone}
    neighborhoods = centroid_neighborhoods(
        problem,
        checked_centroid_neighbor_radius(options.get("centroid_neighbor_radius", 0)),
    )
    for zone, nodes in neighborhoods.items():
        for node in nodes:
            domains[node] &= {zone}
    for node, domain in domains.items():
        if not domain:
            raise ValueError(
                f"Node {node} has no legal zone after centroid-neighborhood constraints."
            )
    return domains


def _welfare_solver(problem: ZoneProblem, method: str, options: Mapping[str, Any]):
    from optimization.data.mid import build_mid_market
    from optimization.solvers.mid import MidCpIntSolver, MidCpSatSolver

    if problem.program_population != "All":
        raise ValueError("The welfare objective requires program_population='All'.")
    if problem.optimization_config is None:
        raise ValueError(
            "The welfare objective requires the originating optimization_config."
        )
    start = time.perf_counter()
    market = build_mid_market(problem, problem.optimization_config)
    preprocessing_seconds = time.perf_counter() - start
    if market.utility_student_count == 0:
        raise ValueError(
            "The welfare objective's MID market has no choice-utility rows."
        )
    solver_class = MidCpIntSolver if method == "cp_int" else MidCpSatSolver
    return solver_class(
        market,
        options.get("mid_lottery_scale", 20),
        preprocessing_seconds=preprocessing_seconds,
        transport_bounds=options.get("mid_transport_bounds", True),
        complementary_slackness=options.get("mid_complementary_slackness", False),
        complementary_slackness_slack=options.get(
            "mid_complementary_slackness_slack", "auto"
        ),
        **options,
    )


def solve_objective(
    problem: ZoneProblem,
    method: str,
    objective: str,
    *,
    options: Mapping[str, Any] | None = None,
) -> ZoneSolution:
    """Solve one stage, preserving its constraints and explicit method choice.

    Matching welfare means discrete MID program welfare for every supported method.
    CP methods use the joint finite-grid model; heuristics evaluate the same
    market with the finite-grid oracle. No generic method silently falls back
    to a different mathematical formulation.
    """
    validate_method_objective(method, objective)
    spec = get_objective(objective)
    objective = spec.name
    options = dict(options or {})
    config = problem.optimization_config
    for name in (
        "choice_model_method",
        "saa_num_seeds",
        "saa_tie_breaking_method",
        "mid_transport_bounds",
        "mid_complementary_slackness",
        "mid_complementary_slackness_slack",
    ):
        if config is not None and hasattr(config, name):
            options.setdefault(name, getattr(config, name))
    # Copy the dataclass so callers can reuse the problem under another
    # objective; the graph, candidate domains and source config stay shared.
    problem = replace(
        problem,
        weight_edges=problem.weight_edges or objective == "boundary_perimeter",
        compactness_objective=spec.compactness_objective,
        candidates=(
            _heuristic_domains(problem, options)
            if method in _HEURISTICS
            else problem.candidates
        ),
    )
    if problem.choice_objective is not None:
        raise ValueError("Pipeline objectives require choice_objective=None.")
    from optimization.solvers import get_solver

    if objective in {"saa_welfare", "choice_welfare"} and method not in _HEURISTICS:
        from optimization.welfare_objectives import solve_welfare_objective

        solution = solve_welfare_objective(problem, method, objective, options)
    elif objective == "matching_welfare" and method in {"cp_bool", "cp_int"}:
        solution = _welfare_solver(problem, method, options).solve(problem)
    elif method in _HEURISTICS and objective in {"cut_edges", "boundary_perimeter"}:
        solution = get_solver(method, **{**options, "recom_objective": "cut_edges"}).solve(
            problem
        )
    elif method in _HEURISTICS:
        scorer = heuristic_scorer(problem, objective, options)
        try:
            solution = get_solver(method, **options).solve_with_scorer(
                problem,
                scorer,
                objective_kind=objective,
                maximize=spec.maximize,
            )
            describe = getattr(scorer, "describe", None)
            if solution.feasible and describe is not None:
                solution.metadata.update(describe(solution.assignment))
        finally:
            close = getattr(scorer, "close", None)
            if close is not None:
                close()
    else:
        solution = get_solver(method, **options).solve(problem)
    solution.metadata.update(
        method=method, objective=objective, objective_maximize=spec.maximize
    )
    if spec.compactness_objective in {"school_distance", "average_distance"}:
        solution.metadata.update(problem.distance_objective_metadata())
    if spec.unit is not None:
        solution.metadata.update(objective_kind=objective, objective_unit=spec.unit)
    if spec.reads_assignment_market:
        solution.metadata.update(
            aggregate_capacity_overage_disabled=problem.overage < 0,
            aggregate_capacity_shortage_disabled=problem.shortage < 0,
        )
    return solution

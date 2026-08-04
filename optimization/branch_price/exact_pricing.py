"""Exact one-zone finite-grid recurrence pricing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral

from ortools.sat.python import cp_model

from optimization.branch_price.access_pricing import (
    analytic_access_pricing_bound,
    zone_raw_welfare,
)
from optimization.branch_price.certificate import PricingMultipliers
from optimization.branch_price.geography import add_zone_pattern_constraints
from optimization.branch_price.patterns import (
    ZonePattern,
    validate_zone_pattern,
    zone_perimeter,
)
from optimization.problem import ZoneProblem
from optimization.solvers.welfare import add_finite_grid_recurrence
from optimization.welfare_oracle import (
    MAX_EXACT_CP_SAT_OBJECTIVE,
    raw_welfare_upper_bound,
    solve_zoned_welfare,
    validate_welfare_market,
)


@dataclass(frozen=True, slots=True)
class ExactPricingResult:
    """An exact recurrence candidate and a safe bound on one label's price."""

    label: int
    status: str
    candidate: ZonePattern | None
    candidate_lagrangian_value: int | None
    pricing_lagrangian_upper_bound: int
    analytic_lagrangian_upper_bound: int
    build_time: float
    solve_time: float
    wall_time: float


def solve_exact_pricing(
    problem: ZoneProblem,
    label: int,
    *,
    utility_scale: int,
    multipliers: PricingMultipliers | None = None,
    centroid_neighbor_radius: int = 0,
    time_limit: float = 60.0,
    workers: int = 1,
    random_seed: int = 0,
    seed_pattern: ZonePattern | None = None,
) -> ExactPricingResult:
    """Maximize exact stable welfare minus integer node and perimeter prices."""
    started = time.monotonic()
    market = problem.cutoff_market
    if market is None:
        raise ValueError("Exact pricing requires a cutoff market.")
    if set(market.school_capacities) != set(market.zone_restricted_schools):
        raise ValueError("Exact pricing requires fully isolated zone markets.")
    if label not in range(problem.Z):
        raise ValueError(f"Unknown zone label {label}.")
    if (
        isinstance(utility_scale, bool)
        or not isinstance(utility_scale, Integral)
        or utility_scale <= 0
    ):
        raise ValueError("utility_scale must be a positive integer.")
    if time_limit < 0:
        raise ValueError("Exact-pricing time_limit must be nonnegative.")
    if (
        isinstance(centroid_neighbor_radius, bool)
        or not isinstance(centroid_neighbor_radius, int)
        or centroid_neighbor_radius < 0
    ):
        raise ValueError("centroid_neighbor_radius must be a non-negative integer.")
    validate_welfare_market(market, utility_scale=int(utility_scale))
    _validate_pricing_market(problem)
    multipliers = multipliers or PricingMultipliers(node={}, boundary=0)
    coverage_nodes = set(problem.nodes) - set(problem.centroids)
    unknown_multipliers = set(multipliers.node) - coverage_nodes
    if unknown_multipliers:
        raise ValueError(
            "Node multipliers must correspond to noncentroid coverage rows: "
            f"{sorted(unknown_multipliers)}."
        )
    analytic_bound = analytic_access_pricing_bound(
        problem,
        label,
        utility_scale=int(utility_scale),
        multipliers=multipliers,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    if time_limit == 0:
        return ExactPricingResult(
            label=label,
            status="NOT_SOLVED",
            candidate=None,
            candidate_lagrangian_value=None,
            pricing_lagrangian_upper_bound=analytic_bound,
            analytic_lagrangian_upper_bound=analytic_bound,
            build_time=0.0,
            solve_time=0.0,
            wall_time=time.monotonic() - started,
        )
    cutoff_hints = None
    same_zone_hints = None
    if seed_pattern is not None:
        if seed_pattern.label != label:
            raise ValueError("Exact-pricing seed pattern has the wrong label.")
        validate_zone_pattern(
            problem,
            seed_pattern,
            centroid_neighbor_radius=centroid_neighbor_radius,
        )
        exact_seed_welfare = zone_raw_welfare(
            market,
            seed_pattern.nodes,
            utility_scale=int(utility_scale),
        )
        if exact_seed_welfare != seed_pattern.raw_welfare:
            raise ValueError(
                "Exact-pricing seed pattern has an incorrect welfare value."
            )
        seed_assignment = {
            node: int(node not in seed_pattern.nodes) for node in problem.nodes
        }
        seed_grid = solve_zoned_welfare(
            market,
            seed_assignment,
            num_zones=2,
            utility_scale=int(utility_scale),
        )
        cutoff_hints = seed_grid.cutoffs.school_cutoffs
        same_zone_hints = {
            (node, school): (
                node in seed_pattern.nodes
                and market.school_nodes[school] in seed_pattern.nodes
            )
            for node, school in {
                (student.node, school)
                for student in market.students
                for school in student.preferences
            }
        }

    model = cp_model.CpModel()
    selected = {
        node: model.NewBoolVar(f"exact_selected_{label}_{node}")
        for node in problem.nodes
    }
    if seed_pattern is not None:
        for node, variable in selected.items():
            model.AddHint(variable, int(node in seed_pattern.nodes))
    perimeter = add_zone_pattern_constraints(
        model,
        problem,
        label,
        selected,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    pairs = {
        (student.node, school)
        for student in market.students
        for school in student.preferences
    }
    same_zone = {}
    for node, school in sorted(pairs):
        school_node = market.school_nodes[school]
        together = model.NewBoolVar(f"exact_together_{node}_{school}")
        model.Add(together <= selected[node])
        model.Add(together <= selected[school_node])
        model.Add(together >= selected[node] + selected[school_node] - 1)
        if same_zone_hints is not None:
            model.AddHint(together, int(same_zone_hints[node, school]))
        same_zone[node, school] = together
    _, raw_welfare = add_finite_grid_recurrence(
        model,
        market,
        same_zone,
        utility_scale=int(utility_scale),
        cutoff_hints=cutoff_hints,
        same_zone_hints=same_zone_hints,
    )
    node_cost = sum(
        multiplier * selected[node] for node, multiplier in multipliers.node.items()
    )
    model.Maximize(raw_welfare - node_cost - multipliers.boundary * perimeter)
    objective_range = (
        raw_welfare_upper_bound(market, int(utility_scale))
        + sum(abs(value) for value in multipliers.node.values())
        + multipliers.boundary * problem.G.number_of_edges()
    )
    if objective_range > MAX_EXACT_CP_SAT_OBJECTIVE:
        raise ValueError("Exact-pricing objective exceeds exact reporting range.")
    remaining = time_limit - (time.monotonic() - started)
    build_time = time.monotonic() - started
    if remaining <= 0:
        return ExactPricingResult(
            label=label,
            status="NOT_SOLVED",
            candidate=None,
            candidate_lagrangian_value=None,
            pricing_lagrangian_upper_bound=analytic_bound,
            analytic_lagrangian_upper_bound=analytic_bound,
            build_time=build_time,
            solve_time=0.0,
            wall_time=time.monotonic() - started,
        )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = remaining
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(random_seed)
    solve_started = time.monotonic()
    status = solver.Solve(model)
    solve_time = time.monotonic() - solve_started
    status_name = solver.StatusName(status)
    candidate = seed_pattern
    candidate_value = (
        seed_pattern.raw_welfare
        - sum(multipliers.node.get(node, 0) for node in seed_pattern.nodes)
        - multipliers.boundary * seed_pattern.perimeter
        if seed_pattern is not None
        else None
    )
    pricing_upper_bound = analytic_bound
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        candidate_nodes = frozenset(
            node for node, variable in selected.items() if solver.Value(variable) == 1
        )
        solver_candidate = ZonePattern(
            label=label,
            nodes=candidate_nodes,
            raw_welfare=zone_raw_welfare(
                market,
                candidate_nodes,
                utility_scale=int(utility_scale),
            ),
            perimeter=zone_perimeter(problem.G, candidate_nodes),
        )
        validate_zone_pattern(
            problem,
            solver_candidate,
            centroid_neighbor_radius=centroid_neighbor_radius,
        )
        solver_candidate_value = (
            solver_candidate.raw_welfare
            - sum(multipliers.node.get(node, 0) for node in solver_candidate.nodes)
            - multipliers.boundary * solver_candidate.perimeter
        )
        if candidate_value is None or solver_candidate_value > candidate_value:
            candidate = solver_candidate
            candidate_value = solver_candidate_value
    if status == cp_model.OPTIMAL:
        pricing_upper_bound = min(
            analytic_bound,
            int(round(solver.ObjectiveValue())),
        )
    elif status == cp_model.FEASIBLE:
        reported_bound = solver.BestObjectiveBound()
        if math.isfinite(reported_bound):
            outward_bound = math.ceil(math.nextafter(reported_bound, math.inf))
            if candidate_value is None or outward_bound >= candidate_value:
                pricing_upper_bound = min(analytic_bound, outward_bound)
    if candidate_value is not None:
        pricing_upper_bound = max(pricing_upper_bound, candidate_value)
    return ExactPricingResult(
        label=label,
        status=status_name,
        candidate=candidate,
        candidate_lagrangian_value=candidate_value,
        pricing_lagrangian_upper_bound=pricing_upper_bound,
        analytic_lagrangian_upper_bound=analytic_bound,
        build_time=build_time,
        solve_time=solve_time,
        wall_time=time.monotonic() - started,
    )


def _validate_pricing_market(problem: ZoneProblem) -> None:
    market = problem.cutoff_market
    if (
        isinstance(market.lottery_scale, bool)
        or not isinstance(market.lottery_scale, int)
        or market.lottery_scale <= 0
    ):
        raise ValueError("Exact pricing requires a positive integer lottery scale.")
    if set(market.school_nodes) != set(market.school_capacities):
        raise ValueError("Exact pricing requires one graph node per market school.")
    if not set(market.school_nodes.values()) <= set(problem.nodes):
        raise ValueError("Exact pricing contains a school outside the problem graph.")
    if any(
        isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0
        for capacity in market.school_capacities.values()
    ):
        raise ValueError("Exact pricing requires nonnegative integer capacities.")
    schools = set(market.school_capacities)
    for student in market.students:
        if len(student.preferences) != len(set(student.preferences)):
            raise ValueError("Exact pricing requires unique student preferences.")
        if not set(student.preferences) <= schools:
            raise ValueError("Exact pricing contains an unknown preferred school.")
        if not set(student.preferences) <= set(student.priorities):
            raise ValueError("Exact pricing requires every listed-school priority.")
        if any(
            isinstance(student.priorities[school], bool)
            or not isinstance(student.priorities[school], int)
            or student.priorities[school] < 0
            for school in student.preferences
        ):
            raise ValueError("Exact pricing requires nonnegative integer priorities.")

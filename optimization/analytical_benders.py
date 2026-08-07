"""Logic-based Benders optimization for zoned analytical Shi welfare."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import gurobipy as gp
from gurobipy import GRB

from optimization.analytical_bounds import (
    prepare_shi_attractions,
    shi_dual_potentials,
)
from optimization.branch_price.analytical_patterns import (
    AnalyticalPatternValuator,
    AnalyticalZonePattern,
)
from optimization.column_generation_seeds import validate_complete_seed
from optimization.problem import ZoneProblem
from optimization.solvers.mip import add_gurobi_zoning_geography


@dataclass(frozen=True, slots=True)
class ShiPriceCut:
    """One globally dual-feasible affine upper bound on zone recourse."""

    school_prices: tuple[tuple[int, float], ...]
    node_coefficients: tuple[tuple[int, float], ...]

    def rhs(self, nodes: frozenset[int] | set[int]) -> float:
        selected = set(nodes)
        return sum(
            coefficient
            for node, coefficient in self.node_coefficients
            if node in selected
        )


@dataclass(frozen=True, slots=True)
class ZonedBendersResult:
    assignment: dict[int, int]
    selected_patterns: tuple[AnalyticalZonePattern, ...]
    incumbent_objective: float
    upper_bound: float
    additive_gap: float
    closed: bool
    status: str
    termination_reason: str
    rounds: int
    master_solves: int
    subproblem_calls: int
    price_cuts_added: int
    point_cuts_added: int
    master_status: str
    timing_seconds: float
    seed_fallback_used: bool
    max_recourse_gap: float
    price_cuts: tuple[ShiPriceCut, ...] = ()


def make_shi_price_cut(
    problem: ZoneProblem,
    school_prices: Mapping[int, float],
    *,
    protect_missing_schools: bool = True,
) -> ShiPriceCut:
    """Lift any nonnegative school prices into a full-universe Benders cut."""
    market = problem.analytical_welfare_market
    if market is None:
        raise ValueError("Analytical Benders cuts require an attached Shi market.")
    unknown = set(school_prices) - set(market.school_capacities)
    if unknown:
        raise ValueError(f"Shi prices contain unknown schools {sorted(unknown)}.")
    prices = {}
    for school, raw_price in school_prices.items():
        price = float(raw_price)
        if not math.isfinite(price) or price < -1e-9:
            raise ValueError("Shi school prices must be finite and nonnegative.")
        prices[school] = max(0.0, price)
    if protect_missing_schools:
        prices.update(_safe_missing_school_prices(problem, set(prices)))
    prices = {
        school: float(prices.get(school, 0.0))
        for school in market.school_capacities
    }
    potentials = shi_dual_potentials(
        market.segments,
        prices,
        beta=market.beta,
    )
    coefficients = {node: 0.0 for node in problem.nodes}
    for segment in market.segments:
        coefficients[segment.node] += (
            float(segment.mass) * potentials[segment.segment_id]
        )
    for school, capacity in market.school_capacities.items():
        coefficients[market.school_nodes[school]] += float(capacity) * prices[school]
    return ShiPriceCut(
        school_prices=tuple(sorted(prices.items())),
        node_coefficients=tuple(
            (node, math.nextafter(coefficient, math.inf))
            for node, coefficient in sorted(coefficients.items())
            if coefficient > 0.0
        ),
    )


def solve_zoned_shi_benders(
    problem: ZoneProblem,
    seed_patterns: Sequence[AnalyticalZonePattern],
    seed_assignment: Mapping[int, int],
    *,
    valuator: AnalyticalPatternValuator | None = None,
    wall_time_limit: float = 2700.0,
    max_rounds: int = 100,
    master_time_limit: float = 180.0,
    feasibility_tolerance: float = 1e-8,
    optimality_tolerance: float = 1e-6,
    centroid_neighbor_radius: int = 0,
    workers: int = 1,
    random_seed: int = 0,
    deadline: float | None = None,
) -> ZonedBendersResult:
    """Optimize a direct zoning master with Shi price and logic cuts."""
    started = time.monotonic()
    local_deadline = started + float(wall_time_limit)
    if deadline is not None:
        local_deadline = min(local_deadline, deadline)
    if isinstance(max_rounds, bool) or max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative.")
    if not math.isfinite(master_time_limit) or master_time_limit <= 0:
        raise ValueError("master_time_limit must be positive and finite.")
    if feasibility_tolerance <= 0 or not math.isfinite(feasibility_tolerance):
        raise ValueError("feasibility_tolerance must be positive and finite.")
    if optimality_tolerance <= 0 or not math.isfinite(optimality_tolerance):
        raise ValueError("optimality_tolerance must be positive and finite.")
    market = problem.analytical_welfare_market
    if market is None:
        raise ValueError("Analytical Benders requires an attached Shi market.")
    valuator = valuator or AnalyticalPatternValuator(problem)
    seed_assignment = validate_complete_seed(
        problem,
        seed_assignment,
        centroid_neighbor_radius=centroid_neighbor_radius,
        validator=valuator.validator,
    )
    pattern_by_key = {pattern.key: pattern for pattern in seed_patterns}
    selected_patterns = _patterns_for_assignment(
        problem,
        seed_assignment,
        pattern_by_key,
    )
    for pattern in seed_patterns:
        valuator.validate_pattern(pattern)

    incumbent_assignment = dict(seed_assignment)
    incumbent_patterns = selected_patterns
    incumbent_objective = sum(pattern.shi_welfare for pattern in selected_patterns)

    model = gp.Model("analytical_shi_benders")
    model.Params.OutputFlag = 0
    model.Params.Threads = max(1, int(workers))
    model.Params.Seed = int(random_seed)
    model.Params.MIPGap = 0.0
    model.Params.FeasibilityTol = max(1e-9, float(feasibility_tolerance))
    model.Params.OptimalityTol = max(1e-9, float(feasibility_tolerance))
    assignment_vars = add_gurobi_zoning_geography(
        model,
        problem,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    theta = {
        label: model.addVar(lb=0.0, name=f"shi_welfare_{label}")
        for label in range(problem.Z)
    }
    model.setObjective(gp.quicksum(theta.values()), GRB.MAXIMIZE)

    price_cuts: list[ShiPriceCut] = []
    price_signatures: set[tuple[tuple[int, str], ...]] = set()
    point_signatures: set[tuple[int, frozenset[int]]] = set()
    price_cuts_added = 0
    point_cuts_added = 0

    def add_price_cut(cut: ShiPriceCut) -> bool:
        nonlocal price_cuts_added
        signature = tuple(
            (school, float(price).hex()) for school, price in cut.school_prices
        )
        if signature in price_signatures:
            return False
        price_signatures.add(signature)
        coefficients = dict(cut.node_coefficients)
        for label in range(problem.Z):
            model.addConstr(
                theta[label]
                <= gp.quicksum(
                    coefficient * assignment_vars[(label, node)]
                    for node, coefficient in coefficients.items()
                    if (label, node) in assignment_vars
                ),
                name=f"shi_price_{price_cuts_added}_{label}",
            )
        price_cuts.append(cut)
        price_cuts_added += 1
        return True

    zero_price_cut = make_shi_price_cut(
        problem,
        {},
        protect_missing_schools=False,
    )
    add_price_cut(zero_price_cut)
    district_upper = math.nextafter(
        sum(coefficient for _, coefficient in zero_price_cut.node_coefficients),
        math.inf,
    )

    def add_point_cut(pattern: AnalyticalZonePattern) -> bool:
        nonlocal point_cuts_added
        signature = pattern.key
        if signature in point_signatures:
            return False
        point_signatures.add(signature)
        source_upper = _pattern_upper_bound(pattern)
        big_m = max(0.0, district_upper - source_upper)
        distance_terms = []
        for node in problem.nodes:
            variable = assignment_vars.get((pattern.label, node))
            if variable is None:
                continue
            distance_terms.append(1.0 - variable if node in pattern.nodes else variable)
        model.addConstr(
            theta[pattern.label]
            <= source_upper + big_m * gp.quicksum(distance_terms),
            name=f"shi_point_{point_cuts_added}_{pattern.label}",
        )
        point_cuts_added += 1
        return True

    for pattern in seed_patterns:
        add_point_cut(pattern)
    for pattern in selected_patterns:
        mechanism = pattern.mechanism
        if mechanism is not None:
            add_price_cut(make_shi_price_cut(problem, mechanism.school_prices))
    add_price_cut(
        make_shi_price_cut(
            problem,
            _combined_school_prices(selected_patterns),
        )
    )

    _set_mip_start(problem, assignment_vars, seed_assignment)
    best_upper = max(district_upper, incumbent_objective)
    max_recourse_gap = max(
        (_pattern_upper_bound(pattern) - pattern.shi_welfare for pattern in seed_patterns),
        default=0.0,
    )
    master_solves = 0
    subproblem_calls = 0
    rounds = 0
    master_status = "NOT_SOLVED"
    termination_reason = "round_limit" if max_rounds == 0 else "unknown"
    closed = False
    seed_fallback_used = True

    for round_index in range(max_rounds):
        remaining = max(0.0, local_deadline - time.monotonic())
        if remaining <= 0:
            termination_reason = "global_time_limit"
            break
        model.Params.TimeLimit = min(float(master_time_limit), remaining)
        model.optimize()
        master_solves += 1
        rounds = round_index + 1
        master_status = _gurobi_status(model.Status)
        if math.isfinite(float(model.ObjBound)):
            best_upper = min(
                best_upper,
                math.nextafter(float(model.ObjBound), math.inf),
            )
        if model.SolCount <= 0:
            termination_reason = (
                "geography_infeasible"
                if model.Status == GRB.INFEASIBLE
                else "master_no_incumbent"
            )
            break

        candidate_assignment = _extract_assignment(problem, assignment_vars)
        candidate_assignment = validate_complete_seed(
            problem,
            candidate_assignment,
            centroid_neighbor_radius=centroid_neighbor_radius,
            validator=valuator.validator,
        )
        _set_mip_start(problem, assignment_vars, candidate_assignment)
        candidate_patterns = []
        recourse_interrupted = False
        try:
            for label in range(problem.Z):
                if time.monotonic() >= local_deadline:
                    raise TimeoutError
                nodes = frozenset(
                    node
                    for node, assigned in candidate_assignment.items()
                    if assigned == label
                )
                subproblem_calls += 1
                candidate_patterns.append(
                    valuator.value(label, nodes, deadline=local_deadline)
                )
        except TimeoutError:
            termination_reason = "subproblem_time_limit"
            recourse_interrupted = True
        except RuntimeError:
            termination_reason = "subproblem_numerical_nonclosure"
            recourse_interrupted = True

        point_cut_added_this_round = False
        for pattern in candidate_patterns:
            mechanism = pattern.mechanism
            if mechanism is not None:
                add_price_cut(make_shi_price_cut(problem, mechanism.school_prices))
            point_cut_added_this_round |= add_point_cut(pattern)
            max_recourse_gap = max(
                max_recourse_gap,
                _pattern_upper_bound(pattern) - pattern.shi_welfare,
            )
        if recourse_interrupted:
            break

        candidate_patterns_tuple = tuple(candidate_patterns)
        add_price_cut(
            make_shi_price_cut(
                problem,
                _combined_school_prices(candidate_patterns_tuple),
            )
        )
        candidate_objective = sum(
            pattern.shi_welfare for pattern in candidate_patterns_tuple
        )
        if candidate_objective > incumbent_objective + optimality_tolerance:
            incumbent_assignment = dict(candidate_assignment)
            incumbent_patterns = candidate_patterns_tuple
            incumbent_objective = candidate_objective
            seed_fallback_used = False

        best_upper = max(best_upper, incumbent_objective)
        additive_gap = max(0.0, best_upper - incumbent_objective)
        if additive_gap <= optimality_tolerance:
            closed = True
            termination_reason = "upper_bound_closed"
            break

        candidate_master_value = sum(float(theta[label].X) for label in theta)
        candidate_gap = max(0.0, candidate_master_value - candidate_objective)
        if model.Status == GRB.OPTIMAL and candidate_gap <= optimality_tolerance:
            best_upper = max(incumbent_objective, min(best_upper, candidate_master_value))
            closed = best_upper - incumbent_objective <= optimality_tolerance
            termination_reason = (
                "master_candidate_closed" if closed else "master_optimal_bound_gap"
            )
            if closed:
                break
        elif model.Status == GRB.OPTIMAL and not point_cut_added_this_round:
            source_upper = sum(
                _pattern_upper_bound(pattern) for pattern in candidate_patterns_tuple
            )
            if source_upper - candidate_objective > optimality_tolerance:
                termination_reason = "subproblem_numerical_gap"
                break
    else:
        termination_reason = "round_limit"

    if not closed and best_upper - incumbent_objective <= optimality_tolerance:
        closed = True
        termination_reason = "upper_bound_closed"
    if best_upper + optimality_tolerance < incumbent_objective:
        raise RuntimeError("Benders upper bound fell below the feasible incumbent.")
    best_upper = max(best_upper, incumbent_objective)
    return ZonedBendersResult(
        assignment=incumbent_assignment,
        selected_patterns=tuple(sorted(incumbent_patterns, key=lambda item: item.label)),
        incumbent_objective=float(incumbent_objective),
        upper_bound=float(best_upper),
        additive_gap=max(0.0, float(best_upper - incumbent_objective)),
        closed=closed,
        status="OPTIMAL" if closed else "FEASIBLE",
        termination_reason=termination_reason,
        rounds=rounds,
        master_solves=master_solves,
        subproblem_calls=subproblem_calls,
        price_cuts_added=price_cuts_added,
        point_cuts_added=point_cuts_added,
        master_status=master_status,
        timing_seconds=time.monotonic() - started,
        seed_fallback_used=seed_fallback_used,
        max_recourse_gap=max_recourse_gap,
        price_cuts=tuple(price_cuts),
    )


def _patterns_for_assignment(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
    pattern_by_key: Mapping[tuple[int, frozenset[int]], AnalyticalZonePattern],
) -> tuple[AnalyticalZonePattern, ...]:
    selected = []
    for label in range(problem.Z):
        key = (
            label,
            frozenset(node for node, zone in assignment.items() if int(zone) == label),
        )
        if key not in pattern_by_key:
            raise ValueError("Seed assignment patterns are absent from the Benders cache.")
        selected.append(pattern_by_key[key])
    return tuple(selected)


def _safe_missing_school_prices(
    problem: ZoneProblem,
    priced_schools: set[int],
) -> dict[int, float]:
    """Price absent schools high enough that they cannot improve any menu.

    When the required coefficient would be numerically excessive, zero is used
    instead.  The subsequent full-potential recomputation keeps the cut valid;
    only source tightness is lost for that school.
    """
    market = problem.analytical_welfare_market
    missing = set(market.school_capacities) - priced_schools
    required = {school: 0.0 for school in missing}
    unsafe = set()
    for segment in market.segments:
        relevant = missing & set(segment.eligible_schools)
        if not relevant:
            continue
        attractions = prepare_shi_attractions(segment, market.beta)
        denominator = 1.0 + sum(attractions.values())
        welfare = market.beta * math.log(denominator)
        for school in relevant:
            bound = welfare * denominator / attractions[school]
            if not math.isfinite(bound) or bound > 1e12:
                unsafe.add(school)
                continue
            required[school] = max(required[school], bound)
    return {
        school: (
            0.0
            if school in unsafe
            else math.nextafter(price * (1.0 + 1e-10), math.inf)
        )
        for school, price in required.items()
    }


def _combined_school_prices(
    patterns: Sequence[AnalyticalZonePattern],
) -> dict[int, float]:
    prices = {}
    for pattern in patterns:
        mechanism = pattern.mechanism
        if mechanism is None:
            raise ValueError("Analytical Benders requires mechanism-valued patterns.")
        overlap = set(prices) & set(mechanism.school_prices)
        if overlap:
            raise ValueError(
                f"Selected zone mechanisms repeat schools {sorted(overlap)}."
            )
        prices.update(mechanism.school_prices)
    return prices


def _pattern_upper_bound(pattern: AnalyticalZonePattern) -> float:
    mechanism = pattern.mechanism
    if mechanism is None:
        raise ValueError("Analytical Benders requires a mechanism-valued pattern.")
    return math.nextafter(
        max(float(pattern.shi_welfare), float(mechanism.repaired_upper_bound)),
        math.inf,
    )


def _set_mip_start(problem, variables, assignment: Mapping[int, int]) -> None:
    for (label, node), variable in variables.items():
        variable.Start = float(int(assignment[node]) == label)


def _extract_assignment(problem: ZoneProblem, variables) -> dict[int, int]:
    assignment = {}
    for node in problem.nodes:
        candidates = tuple(sorted(problem.candidate_zones(node)))
        if not candidates:
            raise problem.no_candidate_zones_error(node)
        assignment[node] = max(candidates, key=lambda label: variables[label, node].X)
    return assignment


def _gurobi_status(status: int) -> str:
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
    }.get(status, f"STATUS_{status}")

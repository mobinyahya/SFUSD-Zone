"""Nested all-bundle pricing for complete analytical zone patterns."""

from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from optimization.analytical_bounds import (
    prepare_shi_attractions,
    shi_menu_value,
    solve_shi_menu_bound,
)
from optimization.branch_price.analytical_master import AnalyticalMasterDuals
from optimization.branch_price.analytical_patterns import (
    AnalyticalPatternValuator,
    AnalyticalZonePattern,
    validate_zoned_shi_market,
)
from optimization.data import contiguity
from optimization.problem import AnalyticalWelfareSegment, ZoneProblem
from optimization.solvers.balance import (
    enforced_balance_constraints,
    rounded_balance_coefficient,
)


@dataclass(frozen=True, slots=True)
class PricingFallbackResult:
    kind: str
    status: str
    upper_bound: float
    timing_seconds: float


@dataclass(frozen=True, slots=True)
class AnalyticalPricingResult:
    label: int
    candidate: AnalyticalZonePattern | None
    candidate_reduced_cost: float | None
    reduced_cost_upper_bound: float
    menu_residual_bound: float
    closed: bool
    status: str
    branch_nodes: int
    menu_columns: int
    timing_seconds: float
    pricing_upper_bound: float
    candidates: tuple[AnalyticalZonePattern, ...] = ()
    fallbacks: tuple[PricingFallbackResult, ...] = ()
    geographic_branches: int = 0
    closure_reason: str = ""


@dataclass(order=True, slots=True)
class _QueueNode:
    priority: float
    sequence: int
    inherited_bound: float = field(compare=False)
    fixes: tuple[tuple[int, int], ...] = field(compare=False)


@dataclass(frozen=True, slots=True)
class _NodeLpResult:
    status: str
    upper_bound: float | None
    x_values: dict[int, float]
    menu_residual_bound: float
    menu_columns: int
    integral: bool
    numerical_nonclosure: bool = False


def solve_analytical_pricing(
    problem: ZoneProblem,
    label: int,
    duals: AnalyticalMasterDuals,
    *,
    valuator: AnalyticalPatternValuator | None = None,
    seed_patterns: Sequence[AnalyticalZonePattern] = (),
    time_limit: float = 300.0,
    node_limit: int = 10_000,
    menu_tolerance: float = 1e-9,
    reduced_cost_tolerance: float = 1e-7,
    centroid_neighbor_radius: int = 0,
    workers: int = 1,
    random_seed: int = 0,
    columns_per_label: int = 10,
    deadline: float | None = None,
) -> AnalyticalPricingResult:
    """Price one label with explicit best-bound branch-and-price."""
    started = time.monotonic()
    if label not in range(problem.Z):
        raise ValueError(f"Unknown analytical pricing label {label}.")
    market = problem.analytical_welfare_market
    if market is None:
        raise ValueError("Analytical pricing requires an attached Shi market.")
    validate_zoned_shi_market(problem, market)
    if not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError("Pricing time_limit must be finite and nonnegative.")
    if isinstance(node_limit, bool) or node_limit < 0:
        raise ValueError("Pricing node_limit must be nonnegative.")
    if menu_tolerance <= 0 or not math.isfinite(menu_tolerance):
        raise ValueError("menu_tolerance must be positive and finite.")
    if reduced_cost_tolerance <= 0 or not math.isfinite(reduced_cost_tolerance):
        raise ValueError("reduced_cost_tolerance must be positive and finite.")
    if duals.boundary < -reduced_cost_tolerance:
        raise ValueError("Analytical pricing requires a nonnegative boundary price.")
    local_deadline = started + float(time_limit)
    if deadline is not None:
        local_deadline = min(local_deadline, deadline)
    valuator = valuator or AnalyticalPatternValuator(
        problem,
        centroid_neighbor_radius=centroid_neighbor_radius,
        menu_tolerance=menu_tolerance,
    )

    attractions = tuple(
        prepare_shi_attractions(segment, market.beta) for segment in market.segments
    )
    menus: dict[int, set[tuple[int, ...]]] = {
        index: {()} for index in range(len(market.segments))
    }
    sigma = duals.convexity[label]
    immediate = model_free_analytical_pricing_bound(
        problem,
        label,
        node_prices=duals.coverage,
        boundary_price=max(0.0, duals.boundary),
        centroid_neighbor_radius=centroid_neighbor_radius,
        attractions=attractions,
    )
    fallbacks = [
        PricingFallbackResult(
            kind="model_free_access",
            status="COMPUTED",
            upper_bound=immediate,
            timing_seconds=0.0,
        )
    ]
    pricing_upper = immediate

    possible_schools = _possible_schools(problem, label)
    access_values = _access_node_values(
        problem,
        possible_schools,
        attractions,
        duals.coverage,
    )
    remaining = max(0.0, local_deadline - time.monotonic())
    if remaining > 0 and time_limit > 0:
        fallback_started = time.monotonic()
        access_bound, access_status = _solve_additive_geographic_bound(
            problem,
            label,
            access_values,
            max(0.0, duals.boundary),
            time_limit=min(remaining * 0.1, 10.0),
            workers=workers,
            random_seed=random_seed,
            centroid_neighbor_radius=centroid_neighbor_radius,
        )
        access_bound = min(immediate, access_bound)
        pricing_upper = min(pricing_upper, access_bound)
        fallbacks.append(
            PricingFallbackResult(
                kind="unconstrained_access_geography",
                status=access_status,
                upper_bound=access_bound,
                timing_seconds=time.monotonic() - fallback_started,
            )
        )

    if local_deadline - time.monotonic() > 0.1:
        try:
            price_vector = _relaxed_capacity_prices(
                problem,
                possible_schools,
                tolerance=menu_tolerance,
                deadline=local_deadline,
            )
        except TimeoutError:
            price_vector = {school: 0.0 for school in possible_schools}
    else:
        price_vector = {school: 0.0 for school in possible_schools}
    capacity_values = _capacity_price_node_values(
        problem,
        label,
        possible_schools,
        attractions,
        price_vector,
        duals.coverage,
    )
    remaining = max(0.0, local_deadline - time.monotonic())
    if remaining > 0 and time_limit > 0:
        fallback_started = time.monotonic()
        capacity_bound, capacity_status = _solve_additive_geographic_bound(
            problem,
            label,
            capacity_values,
            max(0.0, duals.boundary),
            time_limit=min(remaining * 0.1, 10.0),
            workers=workers,
            random_seed=random_seed + 1,
            centroid_neighbor_radius=centroid_neighbor_radius,
        )
        capacity_bound = min(immediate, capacity_bound)
        pricing_upper = min(pricing_upper, capacity_bound)
        fallbacks.append(
            PricingFallbackResult(
                kind="capacity_price_geography",
                status=capacity_status,
                upper_bound=capacity_bound,
                timing_seconds=time.monotonic() - fallback_started,
            )
        )

    candidates: dict[tuple[int, frozenset[int]], AnalyticalZonePattern] = {}
    incumbent_pattern = None
    incumbent_value = -math.inf
    for pattern in seed_patterns:
        if pattern.label != label:
            continue
        value = duals.pricing_value(pattern)
        candidates[pattern.key] = pattern
        if value > incumbent_value:
            incumbent_pattern = pattern
            incumbent_value = value
    if incumbent_pattern is not None:
        pricing_upper = max(pricing_upper, incumbent_value)

    queue: list[_QueueNode] = []
    sequence = 0
    heapq.heappush(
        queue,
        _QueueNode(-pricing_upper, sequence, pricing_upper, ()),
    )
    terminal_bounds: list[float] = []
    in_flight_bound: float | None = None
    branch_nodes = 0
    branches = 0
    max_menu_residual_bound = 0.0
    timed_out = False
    numerical_nonclosure = False
    closure_reason = "all_subtrees_fathomed"

    while queue and branch_nodes < node_limit:
        if time.monotonic() >= local_deadline:
            timed_out = True
            closure_reason = "pricing_time_limit"
            break
        node = heapq.heappop(queue)
        in_flight_bound = node.inherited_bound
        if node.inherited_bound <= incumbent_value + reduced_cost_tolerance:
            terminal_bounds.append(node.inherited_bound)
            in_flight_bound = None
            continue
        branch_nodes += 1
        lp = _solve_pricing_node_lp(
            problem,
            label,
            duals,
            dict(node.fixes),
            menus,
            attractions,
            menu_tolerance=menu_tolerance,
            feasibility_tolerance=min(1e-7, max(1e-9, menu_tolerance)),
            centroid_neighbor_radius=centroid_neighbor_radius,
            deadline=local_deadline,
        )
        if lp.status == "INFEASIBLE":
            in_flight_bound = None
            continue
        if lp.upper_bound is None:
            timed_out = True
            closure_reason = f"node_lp_{lp.status.lower()}"
            break
        node_bound = min(node.inherited_bound, lp.upper_bound)
        max_menu_residual_bound = max(
            max_menu_residual_bound,
            lp.menu_residual_bound,
        )
        numerical_nonclosure |= lp.numerical_nonclosure
        if node_bound <= incumbent_value + reduced_cost_tolerance:
            terminal_bounds.append(node_bound)
            in_flight_bound = None
            continue
        if lp.integral:
            selected_nodes = frozenset(
                node_id for node_id, value in lp.x_values.items() if value >= 0.5
            )
            try:
                pattern = valuator.value(
                    label,
                    selected_nodes,
                    deadline=local_deadline,
                )
            except (ValueError, RuntimeError, TimeoutError):
                terminal_bounds.append(node_bound)
                numerical_nonclosure = True
                closure_reason = "integral_candidate_revaluation_failed"
                in_flight_bound = None
                continue
            exact_value = duals.pricing_value(pattern)
            candidates[pattern.key] = pattern
            if exact_value > incumbent_value:
                incumbent_pattern = pattern
                incumbent_value = exact_value
            if lp.menu_residual_bound > 0:
                terminal_bounds.append(node_bound)
            in_flight_bound = None
            continue

        branch_node = _branching_node(problem, lp.x_values)
        if branch_node is None:
            terminal_bounds.append(node_bound)
            numerical_nonclosure = True
            closure_reason = "fractional_node_without_branch_variable"
            in_flight_bound = None
            continue
        fixes = dict(node.fixes)
        # Install both inherited child bounds before releasing the parent ledger.
        for value in (0, 1):
            child_fixes = dict(fixes)
            child_fixes[branch_node] = value
            sequence += 1
            heapq.heappush(
                queue,
                _QueueNode(
                    -node_bound,
                    sequence,
                    node_bound,
                    tuple(sorted(child_fixes.items())),
                ),
            )
        branches += 1
        in_flight_bound = None

    if queue and branch_nodes >= node_limit:
        closure_reason = "pricing_node_limit"
    open_bounds = [node.inherited_bound for node in queue]
    represented_bounds = [incumbent_value, *open_bounds, *terminal_bounds]
    if in_flight_bound is not None:
        represented_bounds.append(in_flight_bound)
    represented_bounds = [value for value in represented_bounds if math.isfinite(value)]
    global_upper = max(represented_bounds, default=pricing_upper)
    global_upper = min(pricing_upper, global_upper)
    if math.isfinite(incumbent_value):
        global_upper = max(global_upper, incumbent_value)
    global_upper = math.nextafter(global_upper, math.inf)
    closed = (
        not timed_out
        and not queue
        and in_flight_bound is None
        and global_upper <= incumbent_value + reduced_cost_tolerance
    )
    if numerical_nonclosure and not closed:
        status = "NUMERICAL_NONCLOSURE"
    elif closed:
        status = "OPTIMAL_FLOATING" if not terminal_bounds else "TOLERANCE_CLOSED"
    elif timed_out:
        status = "TIME_LIMIT"
    elif queue:
        status = "NODE_LIMIT"
    else:
        status = "BOUNDED_NOT_CLOSED"

    ordered_candidates = tuple(
        sorted(
            candidates.values(),
            key=lambda pattern: (-duals.pricing_value(pattern), pattern.key),
        )[: max(1, int(columns_per_label))]
    )
    if incumbent_pattern is None and ordered_candidates:
        incumbent_pattern = ordered_candidates[0]
        incumbent_value = duals.pricing_value(incumbent_pattern)
    candidate_reduced_cost = (
        incumbent_value - sigma if incumbent_pattern is not None else None
    )
    return AnalyticalPricingResult(
        label=label,
        candidate=incumbent_pattern,
        candidate_reduced_cost=candidate_reduced_cost,
        reduced_cost_upper_bound=global_upper - sigma,
        menu_residual_bound=max_menu_residual_bound,
        closed=closed,
        status=status,
        branch_nodes=branch_nodes,
        menu_columns=sum(len(segment_menus) for segment_menus in menus.values()),
        timing_seconds=time.monotonic() - started,
        pricing_upper_bound=global_upper,
        candidates=ordered_candidates,
        fallbacks=tuple(fallbacks),
        geographic_branches=branches,
        closure_reason=closure_reason,
    )


def model_free_analytical_pricing_bound(
    problem: ZoneProblem,
    label: int,
    *,
    node_prices: Mapping[int, float],
    boundary_price: float,
    centroid_neighbor_radius: int = 0,
    attractions: Sequence[Mapping[int, float]] | None = None,
) -> float:
    """Immediate finite relaxation that cannot lose a timed-out subtree."""
    market = problem.analytical_welfare_market
    if market is None:
        raise ValueError("Analytical pricing requires an attached Shi market.")
    if boundary_price < 0:
        raise ValueError("boundary_price must be nonnegative.")
    attractions = attractions or tuple(
        prepare_shi_attractions(segment, market.beta) for segment in market.segments
    )
    possible_schools = _possible_schools(problem, label)
    values = _access_node_values(
        problem,
        possible_schools,
        attractions,
        node_prices,
    )
    required = set(
        nx.single_source_shortest_path_length(
            problem.G,
            problem.centroids[label],
            cutoff=centroid_neighbor_radius,
        )
    )
    forbidden_centroids = set(problem.centroids) - {problem.centroids[label]}
    bound = 0.0
    for node in problem.nodes:
        if node in forbidden_centroids or label not in problem.candidate_zones(node):
            continue
        value = values.get(node, -float(node_prices.get(node, 0.0)))
        bound += value if node in required else max(0.0, value)
    # Dropping the nonpositive perimeter term is a valid relaxation.
    return math.nextafter(bound, math.inf)


def _solve_pricing_node_lp(
    problem: ZoneProblem,
    label: int,
    duals: AnalyticalMasterDuals,
    fixes: Mapping[int, int],
    menus: dict[int, set[tuple[int, ...]]],
    attractions: Sequence[Mapping[int, float]],
    *,
    menu_tolerance: float,
    feasibility_tolerance: float,
    centroid_neighbor_radius: int,
    deadline: float,
) -> _NodeLpResult:
    market = problem.analytical_welfare_market
    model = gp.Model(f"analytical_price_{label}")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Method = 1
    model.Params.FeasibilityTol = max(1e-9, feasibility_tolerance)
    model.Params.OptimalityTol = max(1e-9, feasibility_tolerance)
    x = {
        node: model.addVar(
            lb=0.0,
            ub=1.0,
            obj=-float(duals.coverage.get(node, 0.0)),
            name=f"x_{node}",
        )
        for node in problem.nodes
    }
    b = {
        (left, right): model.addVar(
            lb=0.0,
            ub=1.0,
            obj=-max(0.0, duals.boundary),
            name=f"b_{left}_{right}",
        )
        for left, right in problem.G.edges
    }
    _add_gurobi_zone_constraints(
        model,
        problem,
        label,
        x,
        b,
        centroid_neighbor_radius=centroid_neighbor_radius,
        fixes=fixes,
    )
    type_rows = {
        index: model.addConstr(
            -segment.mass * x[segment.node] == 0.0,
            name=f"type_{index}",
        )
        for index, segment in enumerate(market.segments)
    }
    capacity_rows = {
        school: model.addConstr(
            -market.school_capacities[school] * x[market.school_nodes[school]] <= 0.0,
            name=f"capacity_{school}",
        )
        for school in market.school_capacities
    }
    menu_variables: dict[tuple[int, tuple[int, ...]], gp.Var] = {}
    for type_index, segment_menus in menus.items():
        for menu in sorted(segment_menus):
            _add_menu_variable(
                model,
                type_index,
                menu,
                market.beta,
                attractions[type_index],
                type_rows,
                capacity_rows,
                menu_variables,
            )
    model.ModelSense = GRB.MAXIMIZE
    model.update()
    numerical_nonclosure = False
    residuals: list[float] = []
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            return _NodeLpResult(
                "TIME_LIMIT", None, {}, 0.0, len(menu_variables), False
            )
        model.Params.TimeLimit = remaining
        model.optimize()
        if model.Status == GRB.INFEASIBLE:
            return _NodeLpResult(
                "INFEASIBLE", None, {}, 0.0, len(menu_variables), False
            )
        if model.Status != GRB.OPTIMAL:
            return _NodeLpResult(
                _gurobi_status(model.Status),
                None,
                {},
                0.0,
                len(menu_variables),
                False,
            )
        prices = {school: float(row.Pi) for school, row in capacity_rows.items()}
        if any(price < -feasibility_tolerance for price in prices.values()):
            return _NodeLpResult(
                "NEGATIVE_CAPACITY_DUAL",
                None,
                {},
                0.0,
                len(menu_variables),
                False,
                numerical_nonclosure=True,
            )
        prices = {school: max(0.0, price) for school, price in prices.items()}
        additions: list[tuple[int, tuple[int, ...]]] = []
        residuals = []
        for type_index, segment in enumerate(market.segments):
            potential = float(type_rows[type_index].Pi)
            prefixes = _prefix_menus(
                segment,
                attractions[type_index],
                prices,
                market.beta,
            )
            best_value = max(value for _, _, _, value in prefixes)
            residuals.append(max(0.0, best_value - potential))
            for menu, _, _, value in prefixes:
                if value - potential <= menu_tolerance:
                    continue
                if menu in menus[type_index]:
                    numerical_nonclosure = True
                    continue
                additions.append((type_index, menu))
        if numerical_nonclosure:
            break
        if not additions:
            break
        for type_index, menu in additions:
            menus[type_index].add(menu)
            _add_menu_variable(
                model,
                type_index,
                menu,
                market.beta,
                attractions[type_index],
                type_rows,
                capacity_rows,
                menu_variables,
            )
        model.update()

    residual_bound = sum(
        segment.mass * residual
        for segment, residual in zip(market.segments, residuals, strict=True)
    )
    upper_bound = float(model.ObjVal) + residual_bound
    x_values = {node: float(variable.X) for node, variable in x.items()}
    integral = all(
        value <= feasibility_tolerance or value >= 1.0 - feasibility_tolerance
        for value in x_values.values()
    )
    return _NodeLpResult(
        status="OPTIMAL_FLOATING"
        if not numerical_nonclosure
        else "NUMERICAL_NONCLOSURE",
        upper_bound=math.nextafter(upper_bound, math.inf),
        x_values=x_values,
        menu_residual_bound=residual_bound,
        menu_columns=len(menu_variables),
        integral=integral,
        numerical_nonclosure=numerical_nonclosure,
    )


def _add_gurobi_zone_constraints(
    model: gp.Model,
    problem: ZoneProblem,
    label: int,
    x: Mapping[int, gp.Var],
    b: Mapping[tuple[int, int], gp.Var],
    *,
    centroid_neighbor_radius: int,
    fixes: Mapping[int, int] = {},
) -> None:
    for node in problem.nodes:
        if label not in problem.candidate_zones(node):
            model.addConstr(x[node] == 0.0)
    centroid = problem.centroids[label]
    model.addConstr(x[centroid] == 1.0)
    for other_label, other_centroid in enumerate(problem.centroids):
        if other_label != label:
            model.addConstr(x[other_centroid] == 0.0)
    for node in nx.single_source_shortest_path_length(
        problem.G,
        centroid,
        cutoff=centroid_neighbor_radius,
    ):
        model.addConstr(x[node] == 1.0)
    closer = contiguity.closer_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    supports = contiguity.contiguity_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    for node in problem.nodes:
        if node == centroid or label not in problem.candidate_zones(node):
            continue
        support_nodes = supports[(node, label)]
        if not closer[(node, label)] or not support_nodes:
            model.addConstr(x[node] == 0.0)
        else:
            model.addConstr(
                x[node] <= gp.quicksum(x[support] for support in support_nodes)
            )
    for constraint in enforced_balance_constraints(problem):
        if constraint.lower_ratio is not None:
            model.addConstr(
                gp.quicksum(
                    rounded_balance_coefficient(
                        problem,
                        constraint,
                        node,
                        constraint.lower_ratio,
                    )
                    * x[node]
                    for node in problem.nodes
                )
                >= 0.0
            )
        if constraint.upper_ratio is not None:
            model.addConstr(
                gp.quicksum(
                    rounded_balance_coefficient(
                        problem,
                        constraint,
                        node,
                        constraint.upper_ratio,
                    )
                    * x[node]
                    for node in problem.nodes
                )
                <= 0.0
            )
    total_schools = sum(problem.num_schools(node) for node in problem.nodes)
    if total_schools:
        average = total_schools / problem.Z
        school_count = gp.quicksum(
            100 * problem.num_schools(node) * x[node] for node in problem.nodes
        )
        model.addConstr(school_count >= round(100 * max(0.0, average - 1.0)))
        model.addConstr(school_count <= round(100 * (average + 1.0)))
    for (left, right), boundary in b.items():
        model.addConstr(boundary >= x[left] - x[right])
        model.addConstr(boundary >= x[right] - x[left])
        model.addConstr(boundary <= x[left] + x[right])
        model.addConstr(boundary <= 2.0 - x[left] - x[right])
    if problem.boundary_prop >= 0:
        model.addConstr(
            gp.quicksum(b.values())
            <= math.floor(problem.boundary_prop * problem.G.number_of_edges())
        )
    for node, value in fixes.items():
        model.addConstr(x[node] == int(value))


def _add_menu_variable(
    model: gp.Model,
    type_index: int,
    menu: tuple[int, ...],
    beta: float,
    attractions: Mapping[int, float],
    type_rows: Mapping[int, gp.Constr],
    capacity_rows: Mapping[int, gp.Constr],
    variables: dict[tuple[int, tuple[int, ...]], gp.Var],
) -> None:
    welfare, canonical, shares = shi_menu_value(attractions, menu, beta)
    constraints = [type_rows[type_index]]
    coefficients = [1.0]
    for school, share in zip(canonical, shares, strict=True):
        constraints.append(capacity_rows[school])
        coefficients.append(share)
    variables[type_index, canonical] = model.addVar(
        lb=0.0,
        obj=welfare,
        column=gp.Column(coefficients, constraints),
        name=f"w_{type_index}_{'_'.join(map(str, canonical)) or 'empty'}",
    )


def _prefix_menus(
    segment: AnalyticalWelfareSegment,
    attractions: Mapping[int, float],
    prices: Mapping[int, float],
    beta: float,
) -> tuple[tuple[tuple[int, ...], float, tuple[float, ...], float], ...]:
    order = sorted(
        segment.eligible_schools,
        key=lambda school: (prices.get(school, 0.0), school),
    )
    out = [((), 0.0, (), 0.0)]
    prefix: list[int] = []
    for school in order:
        prefix.append(school)
        welfare, menu, shares = shi_menu_value(attractions, prefix, beta)
        priced = welfare - sum(
            prices.get(item, 0.0) * share
            for item, share in zip(menu, shares, strict=True)
        )
        out.append((menu, welfare, shares, priced))
    return tuple(out)


def _possible_schools(problem: ZoneProblem, label: int) -> frozenset[int]:
    market = problem.analytical_welfare_market
    other_centroids = set(problem.centroids) - {problem.centroids[label]}
    return frozenset(
        school
        for school, node in market.school_nodes.items()
        if node not in other_centroids and label in problem.candidate_zones(node)
    )


def _access_node_values(
    problem: ZoneProblem,
    possible_schools: frozenset[int],
    attractions: Sequence[Mapping[int, float]],
    node_prices: Mapping[int, float],
) -> dict[int, float]:
    market = problem.analytical_welfare_market
    values = defaultdict(float)
    for segment, segment_attractions in zip(market.segments, attractions, strict=True):
        eligible = possible_schools & set(segment.eligible_schools)
        welfare, _, _ = shi_menu_value(segment_attractions, eligible, market.beta)
        values[segment.node] += segment.mass * welfare
    for node in problem.nodes:
        values[node] -= float(node_prices.get(node, 0.0))
    return dict(values)


def _relaxed_capacity_prices(
    problem: ZoneProblem,
    possible_schools: frozenset[int],
    *,
    tolerance: float,
    deadline: float,
) -> dict[int, float]:
    market = problem.analytical_welfare_market
    local_segments = []
    for segment in market.segments:
        eligible = tuple(
            school for school in segment.eligible_schools if school in possible_schools
        )
        local_segments.append(
            AnalyticalWelfareSegment(
                segment_id=segment.segment_id,
                node=segment.node,
                mass=segment.mass,
                eligible_schools=eligible,
                priorities={school: segment.priorities[school] for school in eligible},
                systematic_utilities={
                    school: segment.systematic_utilities[school] for school in eligible
                },
                outside_utility=segment.outside_utility,
            )
        )
    result = solve_shi_menu_bound(
        local_segments,
        {school: market.school_capacities[school] for school in possible_schools},
        beta=market.beta,
        tolerance=max(tolerance, 1e-8),
        max_rounds=100,
        deadline=deadline,
    )
    return {school: max(0.0, price) for school, price in result.school_prices.items()}


def _capacity_price_node_values(
    problem: ZoneProblem,
    label: int,
    possible_schools: frozenset[int],
    attractions: Sequence[Mapping[int, float]],
    prices: Mapping[int, float],
    node_prices: Mapping[int, float],
) -> dict[int, float]:
    market = problem.analytical_welfare_market
    values = defaultdict(float)
    for segment, segment_attractions in zip(market.segments, attractions, strict=True):
        restricted = AnalyticalWelfareSegment(
            segment_id=segment.segment_id,
            node=segment.node,
            mass=segment.mass,
            eligible_schools=tuple(
                school
                for school in segment.eligible_schools
                if school in possible_schools
            ),
            priorities={},
            systematic_utilities={},
            outside_utility=segment.outside_utility,
        )
        prefixes = _prefix_menus(
            restricted,
            segment_attractions,
            prices,
            market.beta,
        )
        values[segment.node] += segment.mass * max(item[3] for item in prefixes)
    for school in possible_schools:
        values[market.school_nodes[school]] += market.school_capacities[
            school
        ] * prices.get(school, 0.0)
    for node in problem.nodes:
        values[node] -= float(node_prices.get(node, 0.0))
    return dict(values)


def _solve_additive_geographic_bound(
    problem: ZoneProblem,
    label: int,
    node_values: Mapping[int, float],
    boundary_price: float,
    *,
    time_limit: float,
    workers: int,
    random_seed: int,
    centroid_neighbor_radius: int,
) -> tuple[float, str]:
    immediate = _model_free_additive_bound(
        problem,
        label,
        node_values,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    if time_limit <= 0:
        return immediate, "NOT_SOLVED"
    model = gp.Model(f"analytical_additive_bound_{label}")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit)
    model.Params.Threads = max(1, int(workers))
    model.Params.Seed = int(random_seed)
    model.Params.MIPGap = 0.0
    x = {
        node: model.addVar(vtype=GRB.BINARY, obj=float(node_values.get(node, 0.0)))
        for node in problem.nodes
    }
    b = {
        edge: model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            obj=-boundary_price,
        )
        for edge in problem.G.edges
    }
    _add_gurobi_zone_constraints(
        model,
        problem,
        label,
        x,
        b,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    model.ModelSense = GRB.MAXIMIZE
    model.optimize()
    bound = immediate
    reported = float(model.ObjBound)
    if math.isfinite(reported):
        bound = min(bound, math.nextafter(reported, math.inf))
    if model.SolCount:
        bound = max(bound, float(model.ObjVal))
    return bound, _gurobi_status(model.Status)


def _model_free_additive_bound(
    problem: ZoneProblem,
    label: int,
    node_values: Mapping[int, float],
    *,
    centroid_neighbor_radius: int,
) -> float:
    required = set(
        nx.single_source_shortest_path_length(
            problem.G,
            problem.centroids[label],
            cutoff=centroid_neighbor_radius,
        )
    )
    other_centroids = set(problem.centroids) - {problem.centroids[label]}
    bound = 0.0
    for node in problem.nodes:
        if node in other_centroids or label not in problem.candidate_zones(node):
            continue
        value = float(node_values.get(node, 0.0))
        bound += value if node in required else max(0.0, value)
    return math.nextafter(bound, math.inf)


def _branching_node(
    problem: ZoneProblem,
    values: Mapping[int, float],
    *,
    tolerance: float = 1e-7,
) -> int | None:
    school_nodes = set(problem.analytical_welfare_market.school_nodes.values())
    fractional = [
        node for node, value in values.items() if tolerance < value < 1.0 - tolerance
    ]
    if not fractional:
        return None
    return min(
        fractional,
        key=lambda node: (
            node not in school_nodes,
            abs(values[node] - 0.5),
            node,
        ),
    )


def _gurobi_status(status: int) -> str:
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
    }.get(status, f"STATUS_{status}")

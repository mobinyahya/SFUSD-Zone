"""Floating common-cutoff/common-STB conditional-loss welfare bounds.

The routines in this module use ordinary binary64 arithmetic and an ordinary
Gurobi solve.  Their results are FLOATING diagnostics, not proof-grade
certificates.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from optimization.analytical_bounds import _best_shi_cardinality
from optimization.data import contiguity
from optimization.problem import (
    AnalyticalWelfareMarket,
    AnalyticalWelfareSegment,
    ZoneProblem,
)
from optimization.solvers.balance import balance_constraints


FLOATING_DIAGNOSTIC_SCOPE = "FLOATING_DIAGNOSTIC_NOT_PROOF_GRADE"


@dataclass(frozen=True)
class AnalyticalPriorityBoundDimensions:
    """Unpresolved dimensions of one conditional-loss model."""

    graph_nodes: int
    labeled_nodes: int
    zones: int
    market_schools: int
    graph_only_school_nodes: int
    graph_only_schools: int
    matched_segments: int
    priority_values: int
    cutoff_states: int
    observed_node_school_priority_states: int
    node_label_variables: int
    school_label_variables: int
    graph_only_school_label_variables: int
    cutoff_state_variables: int
    inclusion_marginal_variables: int
    priced_value_variables: int
    omission_loss_epigraph_variables: int
    inclusion_loss_epigraph_variables: int
    boundary_variables: int
    selector_binary_variables: int
    base_cap_constraints: int
    omission_cap_constraints: int
    inclusion_cap_constraints: int
    conditional_loss_rows: int
    variables: int
    binary_variables: int
    constraints: int
    nonzeros: int


@dataclass(frozen=True)
class AnalyticalPriorityBoundResult:
    """Ordinary-floating result for the common-cutoff/common-STB bound."""

    status: str
    numerical_scope: str
    relax_integrality: bool
    enforce_geography: bool
    continuous_lp_bound: float | None
    integrality_strengthened_mip_obj_bound: float | None
    integrality_strengthened_mip_incumbent: float | None
    cardinality_shi_bound_at_prices: float
    capacity_price_constant: float
    bound_reduction_from_cardinality_shi: float | None
    school_prices: dict[int, float]
    dimensions: AnalyticalPriorityBoundDimensions
    validation_seconds: float
    conditional_pricing_seconds: float
    model_build_seconds: float
    solve_seconds: float
    total_seconds: float
    raw_solver_status: int

    @property
    def diagnostic_upper_bound(self) -> float | None:
        """Return the applicable FLOATING diagnostic bound, if available."""
        if self.relax_integrality:
            return self.continuous_lp_bound
        return self.integrality_strengthened_mip_obj_bound


@dataclass(frozen=True)
class _PreparedInputs:
    segments: tuple[AnalyticalWelfareSegment, ...]
    schools: tuple[int, ...]
    school_index: dict[int, int]
    prices: np.ndarray
    attractions: tuple[dict[int, float], ...]
    graph_only_school_counts: dict[int, int]
    priority_values: tuple[int, ...]
    state_vectors: tuple[tuple[float, ...], ...]
    maximum_market_menu_cardinality: int


@dataclass(frozen=True)
class _ConditionalPricing:
    segment: AnalyticalWelfareSegment
    attractions: dict[int, float]
    base_menu: tuple[int, ...]
    base_value: float
    omission_losses: dict[int, float]
    inclusion_losses: dict[int, float]


def best_shi_cardinality_forced_inclusion(
    eligible: frozenset[int],
    attractions: Mapping[int, float],
    prices: np.ndarray,
    school_index: Mapping[int, int],
    beta: float,
    cardinality: int,
    mandatory: int,
) -> tuple[tuple[int, ...], float, tuple[float, ...], float]:
    """Exactly price a cardinality-constrained Shi menu containing one school.

    The finite line-arrangement sweep is the forced-inclusion counterpart of
    :func:`optimization.analytical_bounds._best_shi_cardinality`.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite.")
    if (
        isinstance(cardinality, bool)
        or not isinstance(cardinality, Integral)
        or cardinality <= 0
    ):
        raise ValueError("cardinality must be a positive integer.")
    mandatory = _integer_id("mandatory", mandatory)
    eligible = frozenset(_integer_id("eligible school", school) for school in eligible)
    if mandatory not in eligible:
        raise ValueError("mandatory must be an eligible school.")
    if set(eligible) - set(attractions):
        raise ValueError("Every eligible school must have an attraction.")
    if set(eligible) - set(school_index):
        raise ValueError("Every eligible school must have a school_index entry.")
    try:
        price_vector = np.asarray(prices, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prices must be a one-dimensional numeric array.") from exc
    if price_vector.ndim != 1:
        raise ValueError("prices must be a one-dimensional numeric array.")
    if np.any(~np.isfinite(price_vector)) or np.any(price_vector < 0):
        raise ValueError("School prices must be finite and non-negative.")
    normalized_attractions = {}
    normalized_prices = {}
    used_indices = set()
    for school in eligible:
        attraction = float(attractions[school])
        if not math.isfinite(attraction) or attraction <= 0:
            raise ValueError("Attractions must be positive and finite.")
        index = school_index[school]
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("school_index values must be integers.")
        index = int(index)
        if index < 0 or index >= len(price_vector):
            raise ValueError("school_index contains an invalid price index.")
        if index in used_indices:
            raise ValueError("Eligible schools must use distinct price indices.")
        used_indices.add(index)
        price = float(price_vector[index])
        normalized_attractions[school] = attraction
        normalized_prices[school] = price

    optional = eligible - {mandatory}
    optional_cardinality = int(cardinality) - 1
    if optional_cardinality <= 0 or not optional:
        menu = (mandatory,)
        denominator = 1.0 + normalized_attractions[mandatory]
        welfare = beta * math.log(denominator)
        value = (
            welfare
            - normalized_prices[mandatory]
            * normalized_attractions[mandatory]
            / denominator
        )
        return menu, welfare, (normalized_attractions[mandatory] / denominator,), value

    items = sorted(
        optional,
        key=lambda school: (
            -normalized_attractions[school],
            normalized_prices[school],
            school,
        ),
    )
    label = {school: index for index, school in enumerate(items)}
    rank = dict(label)
    active = {school: True for school in items}
    selected = set(items[: min(optional_cardinality, len(items))])
    events = []
    for school in items:
        revenue = -normalized_prices[school]
        events.append((revenue, -label[school], 0, school, -1))
    for first in items:
        first_weight = normalized_attractions[first]
        first_revenue = -normalized_prices[first]
        for second in items:
            second_weight = normalized_attractions[second]
            if first_weight <= second_weight:
                continue
            second_revenue = -normalized_prices[second]
            crossing = (
                first_weight * first_revenue - second_weight * second_revenue
            ) / (first_weight - second_weight)
            events.append((crossing, -label[first], label[second], first, second))
    events.sort()

    attraction_sum = normalized_attractions[mandatory] + sum(
        normalized_attractions[school] for school in selected
    )
    priced_sum = normalized_prices[mandatory] * normalized_attractions[mandatory] + sum(
        normalized_prices[school] * normalized_attractions[school]
        for school in selected
    )

    def evaluate() -> tuple[float, float]:
        denominator = 1.0 + attraction_sum
        welfare = beta * math.log(denominator)
        return welfare, welfare - priced_sum / denominator

    best_welfare, best_value = evaluate()
    best_menu = tuple(sorted({mandatory, *selected}))
    for _, _, _, first, second in events:
        old_membership = {first: first in selected}
        if second >= 0:
            old_membership[second] = second in selected
            if rank[first] < rank[second]:
                rank[first], rank[second] = rank[second], rank[first]
        else:
            active[first] = False
        changed = False
        for school, was_selected in old_membership.items():
            is_selected = active[school] and rank[school] < optional_cardinality
            if is_selected == was_selected:
                continue
            changed = True
            if is_selected:
                selected.add(school)
                attraction_sum += normalized_attractions[school]
                priced_sum += normalized_prices[school] * normalized_attractions[school]
            else:
                selected.discard(school)
                attraction_sum -= normalized_attractions[school]
                priced_sum -= normalized_prices[school] * normalized_attractions[school]
        if changed:
            welfare, value = evaluate()
            if value > best_value:
                best_welfare = welfare
                best_value = value
                best_menu = tuple(sorted({mandatory, *selected}))

    denominator = 1.0 + sum(normalized_attractions[school] for school in best_menu)
    shares = tuple(normalized_attractions[school] / denominator for school in best_menu)
    return best_menu, best_welfare, shares, best_value


def solve_common_cutoff_stb_bound(
    problem: ZoneProblem,
    market: AnalyticalWelfareMarket,
    school_prices: Mapping[int, float],
    *,
    cardinality: int,
    inclusion_shortlist: int = 4,
    relax_integrality: bool = True,
    enforce_geography: bool = False,
    time_limit: float = 300.0,
    workers: int = 1,
) -> AnalyticalPriorityBoundResult:
    """Solve the FLOATING common-cutoff/common-STB conditional-loss bound.

    ``relax_integrality=True`` builds and solves the direct continuous LP.  A
    false value builds the integrality-strengthened diagnostic lift; its
    ``ObjBound``, rather than its incumbent objective, is the global diagnostic
    bound reported by this function.
    """
    started = time.monotonic()
    _validate_solve_options(
        cardinality=cardinality,
        inclusion_shortlist=inclusion_shortlist,
        relax_integrality=relax_integrality,
        enforce_geography=enforce_geography,
        time_limit=time_limit,
        workers=workers,
    )
    prepared = _prepare_inputs(problem, market, school_prices, cardinality)
    validation_seconds = time.monotonic() - started

    pricing_started = time.monotonic()
    priced = _price_conditional_losses(
        prepared,
        beta=market.beta,
        cardinality=cardinality,
        inclusion_shortlist=inclusion_shortlist,
    )
    pricing_seconds = time.monotonic() - pricing_started
    capacity_price_constant = sum(
        float(market.school_capacities[school])
        * float(prepared.prices[prepared.school_index[school]])
        for school in prepared.schools
    )
    base_bound = capacity_price_constant + sum(
        item.segment.mass * item.base_value for item in priced
    )

    build_started = time.monotonic()
    model, dimensions = _build_model(
        problem,
        market,
        prepared,
        priced,
        capacity_price_constant=capacity_price_constant,
        relax_integrality=relax_integrality,
        enforce_geography=enforce_geography,
    )
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit)
    model.Params.Threads = int(workers)
    if not relax_integrality:
        model.Params.MIPGap = 0.0
        model.Params.MIPFocus = 3
    model_build_seconds = time.monotonic() - build_started

    solve_started = time.monotonic()
    model.optimize()
    solve_seconds = time.monotonic() - solve_started
    status = _floating_status(model.Status)
    continuous_lp_bound = None
    mip_obj_bound = None
    mip_incumbent = None
    if relax_integrality:
        if model.Status == GRB.OPTIMAL and model.SolCount:
            continuous_lp_bound = float(model.ObjVal)
    else:
        mip_obj_bound = _finite_model_attribute(model, "ObjBound")
        if model.SolCount:
            mip_incumbent = float(model.ObjVal)
    diagnostic_bound = continuous_lp_bound if relax_integrality else mip_obj_bound
    reduction = base_bound - diagnostic_bound if diagnostic_bound is not None else None
    return AnalyticalPriorityBoundResult(
        status=status,
        numerical_scope=FLOATING_DIAGNOSTIC_SCOPE,
        relax_integrality=relax_integrality,
        enforce_geography=enforce_geography,
        continuous_lp_bound=continuous_lp_bound,
        integrality_strengthened_mip_obj_bound=mip_obj_bound,
        integrality_strengthened_mip_incumbent=mip_incumbent,
        cardinality_shi_bound_at_prices=base_bound,
        capacity_price_constant=capacity_price_constant,
        bound_reduction_from_cardinality_shi=reduction,
        school_prices={
            school: float(prepared.prices[prepared.school_index[school]])
            for school in prepared.schools
        },
        dimensions=dimensions,
        validation_seconds=validation_seconds,
        conditional_pricing_seconds=pricing_seconds,
        model_build_seconds=model_build_seconds,
        solve_seconds=solve_seconds,
        total_seconds=time.monotonic() - started,
        raw_solver_status=int(model.Status),
    )


def _validate_solve_options(
    *,
    cardinality: int,
    inclusion_shortlist: int,
    relax_integrality: bool,
    enforce_geography: bool,
    time_limit: float,
    workers: int,
) -> None:
    if (
        isinstance(cardinality, bool)
        or not isinstance(cardinality, Integral)
        or cardinality <= 0
    ):
        raise ValueError("cardinality must be a positive integer.")
    if (
        isinstance(inclusion_shortlist, bool)
        or not isinstance(inclusion_shortlist, Integral)
        or inclusion_shortlist < 0
    ):
        raise ValueError("inclusion_shortlist must be a non-negative integer.")
    if not isinstance(relax_integrality, bool):
        raise ValueError("relax_integrality must be Boolean.")
    if not isinstance(enforce_geography, bool):
        raise ValueError("enforce_geography must be Boolean.")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite.")
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers <= 0:
        raise ValueError("workers must be a positive integer.")


def _prepare_inputs(
    problem: ZoneProblem,
    market: AnalyticalWelfareMarket,
    school_prices: Mapping[int, float],
    cardinality: int,
) -> _PreparedInputs:
    if not isinstance(problem, ZoneProblem):
        raise TypeError("problem must be a ZoneProblem.")
    if not isinstance(market, AnalyticalWelfareMarket):
        raise TypeError("market must be an AnalyticalWelfareMarket.")
    if problem.Z <= 0 or not problem.nodes:
        raise ValueError("problem must contain at least one zone and graph node.")
    if len(set(problem.centroids)) != problem.Z:
        raise ValueError("problem centroids must be distinct.")
    if len(set(problem.centroid_school_ids)) != problem.Z:
        raise ValueError("problem centroid school IDs must be distinct.")
    if not math.isfinite(market.beta) or market.beta <= 0:
        raise ValueError("market.beta must be positive and finite.")
    if (
        isinstance(market.lottery_scale, bool)
        or not isinstance(market.lottery_scale, Integral)
        or market.lottery_scale <= 0
    ):
        raise ValueError("market.lottery_scale must be a positive integer.")
    if not isinstance(school_prices, Mapping):
        raise TypeError("school_prices must be a mapping.")

    schools = tuple(
        sorted(_integer_id("school", school) for school in market.school_capacities)
    )
    if len(set(schools)) != len(market.school_capacities):
        raise ValueError("School IDs must be unique integers.")
    if not schools:
        raise ValueError("The analytical priority bound requires at least one school.")
    for school in schools:
        capacity = float(market.school_capacities[school])
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError("School capacities must be finite and non-negative.")
    if set(market.school_nodes) != set(schools):
        raise ValueError(
            "market.school_nodes must contain every market school exactly."
        )
    graph_nodes = set(problem.nodes)
    if not set(market.school_nodes.values()) <= graph_nodes:
        raise ValueError("Every market school must map to a problem graph node.")
    if set(market.zone_restricted_schools) != set(schools):
        raise ValueError(
            "The common-zone formulation requires every market school to be "
            "zone restricted."
        )

    normalized_prices = {}
    for raw_school, raw_price in school_prices.items():
        school = _integer_id("school price key", raw_school)
        if school in normalized_prices:
            raise ValueError("school_prices contains duplicate school IDs.")
        price = float(raw_price)
        if not math.isfinite(price) or price < 0:
            raise ValueError("School prices must be finite and non-negative.")
        normalized_prices[school] = price
    if set(normalized_prices) != set(schools):
        raise ValueError("school_prices must contain exactly the market schools.")
    school_index = {school: index for index, school in enumerate(schools)}
    prices = np.asarray([normalized_prices[school] for school in schools], dtype=float)

    market_school_counts = defaultdict(int)
    for school, node in market.school_nodes.items():
        market_school_counts[node] += 1
    graph_only_school_counts = {}
    for node in problem.nodes:
        graph_count = problem.num_schools(node)
        if graph_count < 0:
            raise ValueError("Graph school counts must be non-negative.")
        residual = graph_count - market_school_counts[node]
        if residual < 0:
            raise ValueError(
                f"Node {node} has more market schools than its graph school count."
            )
        if residual:
            graph_only_school_counts[node] = residual
    total_graph_schools = sum(problem.num_schools(node) for node in problem.nodes)
    if total_graph_schools != len(schools) + sum(graph_only_school_counts.values()):
        raise ValueError(
            "Market and graph-only schools do not reconstruct graph counts."
        )

    for zone, (centroid, centroid_school) in enumerate(
        zip(problem.centroids, problem.centroid_school_ids, strict=True)
    ):
        centroid_school = _integer_id("centroid school", centroid_school)
        if centroid_school in market.school_nodes:
            if market.school_nodes[centroid_school] != centroid:
                raise ValueError(
                    f"Centroid school {centroid_school} is not at centroid node {centroid}."
                )
        else:
            node_school_ids = {
                _integer_id("graph school", school)
                for school in problem.G.nodes[centroid].get("school_ids", ())
            }
            if graph_only_school_counts.get(centroid, 0) <= 0 or (
                node_school_ids and centroid_school not in node_school_ids
            ):
                raise ValueError(
                    f"Zone {zone} centroid school is neither a market nor "
                    "graph-only school at its centroid node."
                )

    maximum_school_count = round(100 * (total_graph_schools / problem.Z + 1.0)) // 100
    market_centroids = set(problem.centroid_school_ids) & set(schools)
    maximum_market_menu_cardinality = max(
        min(
            maximum_school_count,
            len(schools) - len(market_centroids - {centroid_school}),
        )
        for centroid_school in problem.centroid_school_ids
    )
    if cardinality < maximum_market_menu_cardinality:
        raise ValueError(
            "cardinality is smaller than the maximum market-school menu allowed "
            "by the encoded school-count and centroid constraints."
        )

    segments = tuple(market.segments)
    if len({segment.segment_id for segment in segments}) != len(segments):
        raise ValueError("segment_id values must be unique.")
    matched = []
    attractions = []
    priorities = set()
    for segment in segments:
        if segment.node not in graph_nodes:
            raise ValueError("Every segment node must belong to the problem graph.")
        if not math.isfinite(segment.mass) or segment.mass <= 0:
            raise ValueError("Segment masses must be positive and finite.")
        if not math.isfinite(segment.outside_utility):
            raise ValueError("Segment outside utilities must be finite.")
        eligible = tuple(
            _integer_id("eligible school", school)
            for school in segment.eligible_schools
        )
        if len(set(eligible)) != len(eligible):
            raise ValueError("A segment has duplicate eligible schools.")
        if set(eligible) - set(schools):
            raise ValueError("A segment is eligible for an unknown school.")
        if not eligible:
            continue
        if set(eligible) - set(segment.systematic_utilities):
            raise ValueError("Every eligible school must have a systematic utility.")
        if set(eligible) - set(segment.priorities):
            raise ValueError("Every eligible school must have a priority.")
        type_attractions = {}
        for school in eligible:
            utility = float(segment.systematic_utilities[school])
            if not math.isfinite(utility):
                raise ValueError("Systematic utilities must be finite.")
            log_attraction = (utility - segment.outside_utility) / market.beta
            if not math.isfinite(log_attraction) or log_attraction > 700:
                raise ValueError(
                    "MNL attraction scale requires high-precision arithmetic."
                )
            attraction = math.exp(log_attraction)
            if not math.isfinite(attraction) or attraction <= 0:
                raise ValueError("MNL attractions must be positive and finite.")
            type_attractions[school] = attraction
            priority = float(segment.priorities[school])
            if not math.isfinite(priority) or priority < 0 or not priority.is_integer():
                raise ValueError("Priorities must be finite non-negative integers.")
            priorities.add(int(priority))
        matched.append(segment)
        attractions.append(type_attractions)

    priority_values = tuple(sorted(priorities))
    state_vectors = _cutoff_state_vectors(priority_values, int(market.lottery_scale))
    return _PreparedInputs(
        segments=tuple(matched),
        schools=schools,
        school_index=school_index,
        prices=prices,
        attractions=tuple(attractions),
        graph_only_school_counts=graph_only_school_counts,
        priority_values=priority_values,
        state_vectors=state_vectors,
        maximum_market_menu_cardinality=maximum_market_menu_cardinality,
    )


def _cutoff_state_vectors(
    priorities: tuple[int, ...], lottery_scale: int
) -> tuple[tuple[float, ...], ...]:
    if not priorities:
        return ()
    states = []
    seen = set()
    for cutoff_index in range((max(priorities) + 1) * lottery_scale + 1):
        cutoff = cutoff_index / lottery_scale
        state = tuple(
            max(0.0, min(1.0, priority + 1.0 - cutoff)) for priority in priorities
        )
        if state not in seen:
            seen.add(state)
            states.append(state)
    return tuple(states)


def _price_conditional_losses(
    prepared: _PreparedInputs,
    *,
    beta: float,
    cardinality: int,
    inclusion_shortlist: int,
) -> tuple[_ConditionalPricing, ...]:
    out = []
    for segment, attractions in zip(
        prepared.segments, prepared.attractions, strict=True
    ):
        eligible = frozenset(segment.eligible_schools)
        menu, _, _, base_value = _best_shi_cardinality(
            eligible,
            attractions,
            prepared.prices,
            prepared.school_index,
            beta,
            cardinality,
        )
        omission_losses = {}
        for school in menu:
            _, _, _, conditional_value = _best_shi_cardinality(
                eligible - {school},
                attractions,
                prepared.prices,
                prepared.school_index,
                beta,
                cardinality,
            )
            omission_losses[school] = max(0.0, base_value - conditional_value)

        denominator = 1.0 + sum(attractions[school] for school in menu)
        priced_average = (
            sum(
                prepared.prices[prepared.school_index[school]] * attractions[school]
                for school in menu
            )
            / denominator
        )
        support_parameter = -priced_average - beta
        candidates = sorted(
            eligible - set(menu),
            key=lambda school: (
                attractions[school]
                * (-prepared.prices[prepared.school_index[school]] - support_parameter),
                school,
            ),
        )[:inclusion_shortlist]
        inclusion_losses = {}
        for school in candidates:
            _, _, _, conditional_value = best_shi_cardinality_forced_inclusion(
                eligible,
                attractions,
                prepared.prices,
                prepared.school_index,
                beta,
                cardinality,
                school,
            )
            inclusion_losses[school] = max(0.0, base_value - conditional_value)
        out.append(
            _ConditionalPricing(
                segment=segment,
                attractions=attractions,
                base_menu=menu,
                base_value=base_value,
                omission_losses=omission_losses,
                inclusion_losses=inclusion_losses,
            )
        )
    return tuple(out)


def _loss_level_supports(
    losses: Mapping[int, float],
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    """Return exact positive levels and their literal greater-or-equal supports."""
    normalized = {}
    for school, raw_loss in losses.items():
        school = _integer_id("loss school", school)
        loss = float(raw_loss)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("Conditional losses must be finite and non-negative.")
        if loss > 0.0:
            normalized[school] = loss
    return tuple(
        (
            level,
            tuple(
                sorted(school for school, loss in normalized.items() if loss >= level)
            ),
        )
        for level in sorted(set(normalized.values()))
    )


def _build_model(
    problem: ZoneProblem,
    market: AnalyticalWelfareMarket,
    prepared: _PreparedInputs,
    priced: tuple[_ConditionalPricing, ...],
    *,
    capacity_price_constant: float,
    relax_integrality: bool,
    enforce_geography: bool,
) -> tuple[gp.Model, AnalyticalPriorityBoundDimensions]:
    model = gp.Model("common_cutoff_common_stb_conditional_loss")
    selector_type = GRB.CONTINUOUS if relax_integrality else GRB.BINARY
    zones = tuple(range(problem.Z))
    labeled_nodes = tuple(
        problem.nodes
        if enforce_geography
        else sorted({segment.node for segment in prepared.segments})
    )
    node_label = {
        (node, zone): model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=selector_type,
            name=f"node_label_{node}_{zone}",
        )
        for node in labeled_nodes
        for zone in zones
    }
    school_label = {
        (school, zone): model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=selector_type,
            name=f"school_label_{school}_{zone}",
        )
        for school in prepared.schools
        for zone in zones
    }
    graph_only_label = {
        (node, zone): model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=selector_type,
            name=f"graph_only_school_node_label_{node}_{zone}",
        )
        for node in prepared.graph_only_school_counts
        for zone in zones
    }
    for node in labeled_nodes:
        model.addConstr(
            gp.quicksum(node_label[node, zone] for zone in zones) == 1,
            name=f"assign_node_label_{node}",
        )
    for school in prepared.schools:
        model.addConstr(
            gp.quicksum(school_label[school, zone] for zone in zones) == 1,
            name=f"assign_school_label_{school}",
        )
    for node in prepared.graph_only_school_counts:
        model.addConstr(
            gp.quicksum(graph_only_label[node, zone] for zone in zones) == 1,
            name=f"assign_graph_only_school_node_label_{node}",
        )

    for zone, (centroid, centroid_school) in enumerate(
        zip(problem.centroids, problem.centroid_school_ids, strict=True)
    ):
        if centroid_school in prepared.school_index:
            model.addConstr(
                school_label[centroid_school, zone] == 1,
                name=f"anchor_centroid_school_{centroid_school}_{zone}",
            )
        else:
            model.addConstr(
                graph_only_label[centroid, zone] == 1,
                name=f"anchor_graph_only_centroid_{centroid}_{zone}",
            )

    total_graph_schools = len(prepared.schools) + sum(
        prepared.graph_only_school_counts.values()
    )
    average_schools = total_graph_schools / problem.Z
    for zone in zones:
        school_count = 100 * gp.quicksum(
            school_label[school, zone] for school in prepared.schools
        ) + gp.quicksum(
            100 * count * graph_only_label[node, zone]
            for node, count in prepared.graph_only_school_counts.items()
        )
        model.addConstr(
            school_count >= round(100 * max(0.0, average_schools - 1.0)),
            name=f"school_count_lower_{zone}",
        )
        model.addConstr(
            school_count <= round(100 * (average_schools + 1.0)),
            name=f"school_count_upper_{zone}",
        )

    boundary_count = 0
    if enforce_geography:
        boundary_count = _add_geography_constraints(
            model,
            problem,
            node_label,
            school_label,
            graph_only_label,
            market,
            selector_type=selector_type,
        )

    states = tuple(range(len(prepared.state_vectors)))
    cutoff_state = {
        (school, state): model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=selector_type,
            name=f"cutoff_state_{school}_{state}",
        )
        for school in prepared.schools
        for state in states
    }
    for school in prepared.schools:
        if states:
            model.addConstr(
                gp.quicksum(cutoff_state[school, state] for state in states) == 1,
                name=f"select_cutoff_state_{school}",
            )

    observed = defaultdict(set)
    for item in priced:
        for school in item.segment.eligible_schools:
            observed[item.segment.node, school].add(
                int(item.segment.priorities[school])
            )
    observed_keys = tuple(
        (node, school, priority)
        for (node, school), priorities in sorted(observed.items())
        for priority in sorted(priorities)
    )
    inclusion_marginal = {
        key: model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name=f"inclusion_marginal_{key[0]}_{key[1]}_{key[2]}",
        )
        for key in observed_keys
    }
    priority_index = {
        priority: index for index, priority in enumerate(prepared.priority_values)
    }
    for node, school, priority in observed_keys:
        marginal = inclusion_marginal[node, school, priority]
        qualification = gp.quicksum(
            prepared.state_vectors[state][priority_index[priority]]
            * cutoff_state[school, state]
            for state in states
        )
        model.addConstr(
            marginal <= qualification,
            name=f"qualification_upper_{node}_{school}_{priority}",
        )
        for zone in zones:
            model.addConstr(
                marginal <= 1.0 - node_label[node, zone] + school_label[school, zone],
                name=f"access_upper_{node}_{school}_{priority}_{zone}",
            )
            model.addConstr(
                marginal
                >= qualification
                + node_label[node, zone]
                + school_label[school, zone]
                - 2.0,
                name=f"access_lower_{node}_{school}_{priority}_{zone}",
            )

    priced_value = {
        index: model.addVar(
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"priced_type_value_{index}",
        )
        for index in range(len(priced))
    }
    omission_levels = 0
    inclusion_levels = 0
    conditional_rows = 0
    omission_caps = 0
    inclusion_caps = 0
    for index, item in enumerate(priced):
        model.addConstr(
            priced_value[index] <= item.base_value,
            name=f"base_value_cap_{index}",
        )
        omission_loss, levels, rows = _add_loss_chain(
            model,
            inclusion_marginal,
            item,
            item.omission_losses,
            prefix="omission",
        )
        omission_levels += levels
        conditional_rows += rows
        if levels:
            model.addConstr(
                priced_value[index] <= item.base_value - omission_loss,
                name=f"omission_value_cap_{index}",
            )
            omission_caps += 1
        inclusion_loss, levels, rows = _add_loss_chain(
            model,
            inclusion_marginal,
            item,
            item.inclusion_losses,
            prefix="inclusion",
        )
        inclusion_levels += levels
        conditional_rows += rows
        if levels:
            model.addConstr(
                priced_value[index] <= item.base_value - inclusion_loss,
                name=f"inclusion_value_cap_{index}",
            )
            inclusion_caps += 1

    model.setObjective(
        capacity_price_constant
        + gp.quicksum(
            item.segment.mass * priced_value[index] for index, item in enumerate(priced)
        ),
        GRB.MAXIMIZE,
    )
    model.update()
    selector_count = (
        len(node_label) + len(school_label) + len(graph_only_label) + len(cutoff_state)
    )
    dimensions = AnalyticalPriorityBoundDimensions(
        graph_nodes=problem.A,
        labeled_nodes=len(labeled_nodes),
        zones=problem.Z,
        market_schools=len(prepared.schools),
        graph_only_school_nodes=len(prepared.graph_only_school_counts),
        graph_only_schools=sum(prepared.graph_only_school_counts.values()),
        matched_segments=len(priced),
        priority_values=len(prepared.priority_values),
        cutoff_states=len(states),
        observed_node_school_priority_states=len(observed_keys),
        node_label_variables=len(node_label),
        school_label_variables=len(school_label),
        graph_only_school_label_variables=len(graph_only_label),
        cutoff_state_variables=len(cutoff_state),
        inclusion_marginal_variables=len(inclusion_marginal),
        priced_value_variables=len(priced_value),
        omission_loss_epigraph_variables=omission_levels,
        inclusion_loss_epigraph_variables=inclusion_levels,
        boundary_variables=boundary_count,
        selector_binary_variables=0 if relax_integrality else selector_count,
        base_cap_constraints=len(priced),
        omission_cap_constraints=omission_caps,
        inclusion_cap_constraints=inclusion_caps,
        conditional_loss_rows=conditional_rows,
        variables=int(model.NumVars),
        binary_variables=int(model.NumBinVars),
        constraints=int(model.NumConstrs),
        nonzeros=int(model.NumNZs),
    )
    return model, dimensions


def _add_geography_constraints(
    model: gp.Model,
    problem: ZoneProblem,
    node_label: dict[tuple[int, int], gp.Var],
    school_label: dict[tuple[int, int], gp.Var],
    graph_only_label: dict[tuple[int, int], gp.Var],
    market: AnalyticalWelfareMarket,
    *,
    selector_type: str,
) -> int:
    zones = tuple(range(problem.Z))
    for node in problem.nodes:
        candidates = problem.candidate_zones(node)
        if not candidates:
            raise problem.no_candidate_zones_error(node)
        if not candidates <= set(zones):
            raise ValueError("A problem candidate zone is outside the zone labels.")
        for zone in set(zones) - candidates:
            model.addConstr(
                node_label[node, zone] == 0,
                name=f"forbid_candidate_{node}_{zone}",
            )
    for zone, centroid in enumerate(problem.centroids):
        model.addConstr(
            node_label[centroid, zone] == 1,
            name=f"anchor_centroid_node_{centroid}_{zone}",
        )

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
        for zone in problem.candidate_zones(node):
            if node == problem.centroids[zone]:
                continue
            support_nodes = supports[(node, zone)]
            if not closer[(node, zone)] or not support_nodes:
                model.addConstr(
                    node_label[node, zone] == 0,
                    name=f"forbid_unsupported_{node}_{zone}",
                )
            else:
                model.addConstr(
                    node_label[node, zone]
                    <= gp.quicksum(
                        node_label[support, zone] for support in support_nodes
                    ),
                    name=f"monotone_support_{node}_{zone}",
                )

    for school, node in market.school_nodes.items():
        for zone in zones:
            model.addConstr(
                school_label[school, zone] == node_label[node, zone],
                name=f"link_market_school_{school}_{node}_{zone}",
            )
    for node in {key[0] for key in graph_only_label}:
        for zone in zones:
            model.addConstr(
                graph_only_label[node, zone] == node_label[node, zone],
                name=f"link_graph_only_school_node_{node}_{zone}",
            )

    for zone in zones:
        for constraint in balance_constraints(problem):
            if constraint.kind == "capacity":
                continue
            if constraint.lower_ratio is not None:
                model.addConstr(
                    gp.quicksum(
                        round(
                            100
                            * (
                                constraint.value(node)
                                - constraint.lower_ratio * problem.students(node)
                            )
                        )
                        * node_label[node, zone]
                        for node in problem.nodes
                    )
                    >= 0,
                    name=f"rounded_{constraint.kind}_lower_{zone}",
                )
            if constraint.upper_ratio is not None:
                model.addConstr(
                    gp.quicksum(
                        round(
                            100
                            * (
                                constraint.value(node)
                                - constraint.upper_ratio * problem.students(node)
                            )
                        )
                        * node_label[node, zone]
                        for node in problem.nodes
                    )
                    <= 0,
                    name=f"rounded_{constraint.kind}_upper_{zone}",
                )

    if problem.boundary_prop < 0:
        return 0
    boundary_variables = []
    for left, right in problem.G.edges:
        boundary = model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=selector_type,
            name=f"boundary_limit_{left}_{right}",
        )
        for zone in zones:
            model.addConstr(
                boundary >= node_label[left, zone] - node_label[right, zone],
                name=f"boundary_forward_{left}_{right}_{zone}",
            )
            model.addConstr(
                boundary >= node_label[right, zone] - node_label[left, zone],
                name=f"boundary_reverse_{left}_{right}_{zone}",
            )
        boundary_variables.append(boundary)
    model.addConstr(
        gp.quicksum(boundary_variables)
        <= math.floor(problem.boundary_prop * problem.G.number_of_edges()),
        name="global_boundary_limit",
    )
    return len(boundary_variables)


def _add_loss_chain(
    model: gp.Model,
    inclusion_marginal: dict[tuple[int, int, int], gp.Var],
    item: _ConditionalPricing,
    losses: Mapping[int, float],
    *,
    prefix: str,
) -> tuple[gp.LinExpr, int, int]:
    supports = _loss_level_supports(losses)
    expression = gp.LinExpr()
    previous = 0.0
    row_count = 0
    for level_index, (level, schools) in enumerate(supports):
        event = model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name=f"{prefix}_loss_event_{item.segment.segment_id}_{level_index}",
        )
        expression += (level - previous) * event
        previous = level
        for school in schools:
            priority = int(item.segment.priorities[school])
            marginal = inclusion_marginal[item.segment.node, school, priority]
            if prefix == "omission":
                model.addConstr(event >= 1.0 - marginal)
            else:
                model.addConstr(event >= marginal)
            row_count += 1
    return expression, len(supports), row_count


def _floating_status(status: int) -> str:
    name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }.get(status, f"SOLVER_STATUS_{status}")
    return f"{name}_FLOATING_DIAGNOSTIC"


def _finite_model_attribute(model: gp.Model, name: str) -> float | None:
    try:
        value = float(getattr(model, name))
    except (AttributeError, gp.GurobiError):
        return None
    return value if math.isfinite(value) else None


def _integer_id(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return int(value)

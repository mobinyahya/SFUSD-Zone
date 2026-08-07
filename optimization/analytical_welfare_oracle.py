"""Analytical expected-MNL welfare for isolated DA-STB cutoff markets."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Mapping

from optimization.problem import AnalyticalWelfareMarket, AnalyticalWelfareSegment


EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class AnalyticalIntegrationResult:
    """Expected assignments and inclusive-value welfare at fixed cutoffs."""

    demands: dict[int, float]
    assignment_measures: dict[int, dict[int, float]]
    outside_measures: dict[int, float]
    normalized_welfare: float
    max_mass_balance_residual: float


@dataclass(frozen=True)
class AnalyticalMarketResult:
    """Least-cutoff result for one isolated analytical-Gumbel market."""

    cutoffs: dict[int, float]
    cutoff_indices: dict[int, int] | None
    demands: dict[int, float]
    capacities: dict[int, int]
    assignment_measures: dict[int, dict[int, float]]
    outside_measures: dict[int, float]
    normalized_welfare: float
    raw_welfare_constant: float
    iterations: int
    cutoff_grid: int | None
    capacity_feasible: bool
    complementarity_valid: bool | None
    grid_minimal: bool | None
    least_cutoff_numerically_verified: bool
    least_cutoff_residual: float
    least_cutoff_tolerance: float
    capacity_feasibility_tolerance: float
    complementarity_tolerance: float | None
    grid_underfill: dict[int, float] | None
    grid_lowered_demand: dict[int, float] | None
    max_capacity_violation: float
    max_mass_balance_residual: float
    timing_seconds: float

    @property
    def stable(self) -> bool:
        if self.cutoff_grid is None:
            return self.capacity_feasible and self.complementarity_valid is True
        return self.capacity_feasible and self.grid_minimal is True

    @property
    def objective_kind(self) -> str:
        if self.cutoff_grid is None:
            return "analytical_gumbel_stable_welfare_continuum"
        return f"analytical_gumbel_stable_welfare_cutoff_grid_{self.cutoff_grid}"


@dataclass(frozen=True)
class ZonedAnalyticalWelfareResult:
    """Analytical welfare and diagnostics for every isolated zone."""

    zones: dict[int, AnalyticalMarketResult]
    school_cutoffs: dict[int, float]
    school_cutoff_indices: dict[int, int] | None
    school_demands: dict[int, float]
    assignment_measures: dict[int, dict[int, float]]
    outside_measures: dict[int, float]
    normalized_welfare: float
    raw_welfare_constant: float
    cutoff_grid: int | None
    timing_seconds: float

    @property
    def stable(self) -> bool:
        return all(zone.stable for zone in self.zones.values())

    @property
    def least_cutoff_numerically_verified(self) -> bool:
        return all(
            zone.least_cutoff_numerically_verified for zone in self.zones.values()
        )

    @property
    def objective_kind(self) -> str:
        if self.cutoff_grid is None:
            return "analytical_gumbel_stable_welfare_continuum"
        return f"analytical_gumbel_stable_welfare_cutoff_grid_{self.cutoff_grid}"


@dataclass(frozen=True)
class AnalyticalNodeValues:
    """Fixed-bundle, fixed-cutoff welfare and demand aggregated by node."""

    welfare: dict[int, float]
    demands: dict[int, dict[int, float]]


@dataclass(frozen=True)
class _PreparedSegment:
    segment: AnalyticalWelfareSegment
    log_attractions: dict[int, float]


def integrate_analytical_market(
    segments: Iterable[AnalyticalWelfareSegment],
    school_capacities: Mapping[int, int],
    cutoffs: Mapping[int, float],
    *,
    beta: float,
) -> AnalyticalIntegrationResult:
    """Exactly integrate constant MNL menus between STB threshold events."""
    segments = tuple(segments)
    capacities = _validate_market_inputs(segments, school_capacities, beta)
    if set(cutoffs) != set(capacities):
        raise ValueError("cutoffs must contain exactly the market schools.")
    normalized_cutoffs = {int(school): float(value) for school, value in cutoffs.items()}
    if any(not math.isfinite(value) or value < 0 for value in normalized_cutoffs.values()):
        raise ValueError("cutoffs must be finite and non-negative.")
    return _integrate_prepared(
        _prepare_segments(segments, beta), capacities, normalized_cutoffs, beta
    )


def aggregate_analytical_node_values(
    segments: Iterable[AnalyticalWelfareSegment],
    schools: Iterable[int],
    cutoffs: Mapping[int, float],
    *,
    beta: float,
) -> AnalyticalNodeValues:
    """Aggregate exact event-sweep coefficients for a fixed school/cutoff state."""
    schools = tuple(map(int, schools))
    school_set = set(schools)
    if set(cutoffs) != school_set:
        raise ValueError("cutoffs must contain exactly the fixed school bundle.")
    by_node: dict[int, list[AnalyticalWelfareSegment]] = {}
    for segment in segments:
        eligible = tuple(
            school for school in segment.eligible_schools if school in school_set
        )
        by_node.setdefault(segment.node, []).append(
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
    capacities = {school: 0 for school in schools}
    welfare = {}
    demands = {}
    for node, node_segments in by_node.items():
        integrated = integrate_analytical_market(
            node_segments, capacities, cutoffs, beta=beta
        )
        welfare[node] = integrated.normalized_welfare
        demands[node] = integrated.demands
    return AnalyticalNodeValues(welfare=welfare, demands=demands)


def solve_analytical_market(
    segments: Iterable[AnalyticalWelfareSegment],
    school_capacities: Mapping[int, int],
    *,
    beta: float,
    cutoff_grid: int | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    deadline: float | None = None,
) -> AnalyticalMarketResult:
    """Compute the componentwise-least feasible cutoff vector from zero."""
    started = time.monotonic()
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if cutoff_grid is not None:
        if isinstance(cutoff_grid, bool) or not isinstance(cutoff_grid, int):
            raise ValueError("cutoff_grid must be a positive integer or None.")
        if cutoff_grid <= 0:
            raise ValueError("cutoff_grid must be a positive integer or None.")

    segments = tuple(segments)
    capacities = _validate_market_inputs(segments, school_capacities, beta)
    prepared = _prepare_segments(segments, beta)
    schools = tuple(capacities)
    demand_tolerance = tolerance

    if cutoff_grid is None:
        cutoffs = {school: 0.0 for school in schools}
        cutoff_indices = None
    else:
        cutoff_indices = {school: 0 for school in schools}
        cutoffs = {school: 0.0 for school in schools}

    updates = 0
    converged = False
    for _ in range(max_iterations):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Analytical welfare solve reached its deadline.")
        changed = False
        max_change = 0.0
        for school in schools:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Analytical welfare solve reached its deadline.")
            profiles = _school_demand_profiles(prepared, school, cutoffs)
            if cutoff_grid is None:
                required = _minimum_continuum_cutoff(
                    profiles,
                    capacities[school],
                    tolerance=tolerance,
                    demand_tolerance=demand_tolerance,
                )
                if required > cutoffs[school]:
                    change = required - cutoffs[school]
                    cutoffs[school] = required
                    max_change = max(max_change, change)
                    if change > tolerance:
                        changed = True
                        updates += 1
            else:
                required_index = _minimum_grid_cutoff(
                    profiles,
                    capacities[school],
                    cutoff_grid,
                    demand_tolerance=demand_tolerance,
                )
                if required_index > cutoff_indices[school]:
                    change = (required_index - cutoff_indices[school]) / cutoff_grid
                    cutoff_indices[school] = required_index
                    cutoffs[school] = required_index / cutoff_grid
                    changed = True
                    updates += 1
                    max_change = max(max_change, change)

        integration = _integrate_prepared(prepared, capacities, cutoffs, beta)
        max_violation = max(
            (integration.demands[school] - capacities[school] for school in schools),
            default=0.0,
        )
        if cutoff_grid is not None:
            if not changed and max_violation <= demand_tolerance:
                converged = True
                break
        elif max_change <= tolerance and max_violation <= demand_tolerance:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            f"Analytical cutoff iteration did not converge after {max_iterations} sweeps."
        )

    integration = _integrate_prepared(prepared, capacities, cutoffs, beta)
    max_capacity_violation = max(
        0.0,
        max(
            (integration.demands[school] - capacities[school] for school in schools),
            default=0.0,
        ),
    )
    capacity_feasible = max_capacity_violation <= demand_tolerance
    if cutoff_grid is None:
        coordinate_residual = 0.0
        for school in schools:
            required = _minimum_continuum_cutoff(
                _school_demand_profiles(prepared, school, cutoffs),
                capacities[school],
                tolerance=tolerance,
                demand_tolerance=demand_tolerance,
            )
            coordinate_residual = max(
                coordinate_residual, abs(required - cutoffs[school])
            )
        least_cutoff_tolerance = 2 * tolerance
        least_cutoff_verified = (
            capacity_feasible and coordinate_residual <= least_cutoff_tolerance
        )
        complementarity_tolerance = max(demand_tolerance, 10 * tolerance)
        complementarity_valid = capacity_feasible and all(
            cutoff <= tolerance
            or abs(integration.demands[school] - capacities[school])
            <= complementarity_tolerance
            for school, cutoff in cutoffs.items()
        )
        grid_minimal = None
        grid_underfill = None
        grid_lowered_demand = None
    else:
        grid_minimal = capacity_feasible
        coordinate_residual = 0.0
        least_cutoff_tolerance = demand_tolerance
        complementarity_tolerance = None
        grid_underfill = {
            school: capacities[school] - integration.demands[school]
            for school in schools
        }
        grid_lowered_demand = {}
        for school, index in cutoff_indices.items():
            if index <= 0:
                continue
            lowered = dict(cutoffs)
            lowered[school] = (index - 1) / cutoff_grid
            lower_demand = _demand_from_profiles(
                _school_demand_profiles(prepared, school, lowered), lowered[school]
            )
            grid_lowered_demand[school] = lower_demand
            if lower_demand <= capacities[school] + demand_tolerance:
                coordinate_residual = max(
                    coordinate_residual,
                    capacities[school] + demand_tolerance - lower_demand,
                )
                grid_minimal = False
                break
        least_cutoff_verified = grid_minimal
        complementarity_valid = None

    raw_constant = sum(
        segment.mass * (segment.outside_utility + beta * EULER_GAMMA)
        for segment in segments
    )
    return AnalyticalMarketResult(
        cutoffs=cutoffs,
        cutoff_indices=cutoff_indices,
        demands=integration.demands,
        capacities=capacities,
        assignment_measures=integration.assignment_measures,
        outside_measures=integration.outside_measures,
        normalized_welfare=integration.normalized_welfare,
        raw_welfare_constant=raw_constant,
        iterations=updates,
        cutoff_grid=cutoff_grid,
        capacity_feasible=capacity_feasible,
        complementarity_valid=complementarity_valid,
        grid_minimal=grid_minimal,
        least_cutoff_numerically_verified=least_cutoff_verified,
        least_cutoff_residual=coordinate_residual,
        least_cutoff_tolerance=least_cutoff_tolerance,
        capacity_feasibility_tolerance=demand_tolerance,
        complementarity_tolerance=complementarity_tolerance,
        grid_underfill=grid_underfill,
        grid_lowered_demand=grid_lowered_demand,
        max_capacity_violation=max_capacity_violation,
        max_mass_balance_residual=integration.max_mass_balance_residual,
        timing_seconds=time.monotonic() - started,
    )


def evaluate_zoned_analytical_welfare(
    market: AnalyticalWelfareMarket,
    node_assignment: Mapping[int, int],
    *,
    num_zones: int | None = None,
    cutoff_grid: int | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    deadline: float | None = None,
) -> ZonedAnalyticalWelfareResult:
    """Evaluate independent analytical matching markets induced by a zoning."""
    started = time.monotonic()
    validate_analytical_welfare_market(market)
    unrestricted = set(market.school_capacities) - set(
        market.zone_restricted_schools
    )
    if unrestricted:
        raise ValueError(
            "Isolated analytical markets require every school to be zone restricted; "
            f"unrestricted schools: {sorted(unrestricted)}."
        )
    missing_nodes = {
        segment.node
        for segment in market.segments
        if segment.node not in node_assignment
    } | {
        node for node in market.school_nodes.values() if node not in node_assignment
    }
    if missing_nodes:
        raise ValueError(f"Zoning omits analytical market nodes: {sorted(missing_nodes)}.")
    if num_zones is None:
        num_zones = max(node_assignment.values(), default=-1) + 1
    if num_zones <= 0:
        raise ValueError("num_zones must be positive.")

    schools_by_zone = {zone: [] for zone in range(num_zones)}
    for school in market.school_capacities:
        zone = int(node_assignment[market.school_nodes[school]])
        if zone not in schools_by_zone:
            raise ValueError(f"School {school} has invalid zone {zone}.")
        schools_by_zone[zone].append(school)
    segments_by_zone = {zone: [] for zone in range(num_zones)}
    for segment in market.segments:
        zone = int(node_assignment[segment.node])
        if zone not in segments_by_zone:
            raise ValueError(f"Segment {segment.segment_id} has invalid zone {zone}.")
        school_set = set(schools_by_zone[zone])
        eligible = tuple(
            school for school in segment.eligible_schools if school in school_set
        )
        segments_by_zone[zone].append(
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

    results = {}
    all_cutoffs = {}
    all_indices = {} if cutoff_grid is not None else None
    all_demands = {}
    all_assignments = {}
    all_outside = {}
    for zone in range(num_zones):
        capacities = {
            school: market.school_capacities[school]
            for school in schools_by_zone[zone]
        }
        result = solve_analytical_market(
            segments_by_zone[zone],
            capacities,
            beta=market.beta,
            cutoff_grid=cutoff_grid,
            tolerance=tolerance,
            max_iterations=max_iterations,
            deadline=deadline,
        )
        results[zone] = result
        all_cutoffs.update(result.cutoffs)
        if all_indices is not None:
            all_indices.update(result.cutoff_indices or {})
        all_demands.update(result.demands)
        all_assignments.update(result.assignment_measures)
        all_outside.update(result.outside_measures)

    return ZonedAnalyticalWelfareResult(
        zones=results,
        school_cutoffs=all_cutoffs,
        school_cutoff_indices=all_indices,
        school_demands=all_demands,
        assignment_measures=all_assignments,
        outside_measures=all_outside,
        normalized_welfare=sum(result.normalized_welfare for result in results.values()),
        raw_welfare_constant=sum(result.raw_welfare_constant for result in results.values()),
        cutoff_grid=cutoff_grid,
        timing_seconds=time.monotonic() - started,
    )


def validate_analytical_welfare_market(market: AnalyticalWelfareMarket) -> None:
    """Reject incomplete or mixed-semantics analytical market contracts."""
    _validate_market_inputs(market.segments, market.school_capacities, market.beta)
    schools = set(market.school_capacities)
    if set(market.school_nodes) != schools:
        raise ValueError("school_nodes and school_capacities must have identical keys.")
    if not set(market.zone_restricted_schools) <= schools:
        raise ValueError("zone_restricted_schools contains an unknown school.")
    if isinstance(market.lottery_scale, bool) or not isinstance(
        market.lottery_scale, int
    ):
        raise ValueError("lottery_scale must be a positive integer.")
    if market.lottery_scale <= 0:
        raise ValueError("lottery_scale must be a positive integer.")


def _validate_market_inputs(
    segments: Iterable[AnalyticalWelfareSegment],
    school_capacities: Mapping[int, int],
    beta: float,
) -> dict[int, int]:
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite.")
    capacities = {}
    for school, capacity in school_capacities.items():
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("School capacities must be non-negative integers.")
        capacities[int(school)] = capacity
    schools = set(capacities)
    seen = set()
    for segment in segments:
        if segment.segment_id in seen:
            raise ValueError(f"Duplicate segment_id {segment.segment_id}.")
        seen.add(segment.segment_id)
        if not math.isfinite(segment.mass) or segment.mass <= 0:
            raise ValueError("Segment masses must be positive and finite.")
        if not math.isfinite(segment.outside_utility):
            raise ValueError("Outside systematic utilities must be finite.")
        eligible = tuple(map(int, segment.eligible_schools))
        if len(eligible) != len(set(eligible)):
            raise ValueError(f"Segment {segment.segment_id} repeats an eligible school.")
        unknown = set(eligible) - schools
        if unknown:
            raise ValueError(
                f"Segment {segment.segment_id} has unknown schools: {sorted(unknown)}."
            )
        if set(segment.priorities) != set(eligible):
            raise ValueError(
                f"Segment {segment.segment_id} priorities must match eligibility."
            )
        if set(segment.systematic_utilities) != set(eligible):
            raise ValueError(
                f"Segment {segment.segment_id} utilities must match eligibility."
            )
        if any(
            not math.isfinite(float(segment.priorities[school]))
            or float(segment.priorities[school]) < 0
            for school in eligible
        ):
            raise ValueError("Priorities must be non-negative and finite.")
        if any(
            not math.isfinite(float(segment.systematic_utilities[school]))
            for school in eligible
        ):
            raise ValueError("Systematic utilities must be finite.")
    return capacities


def _prepare_segments(
    segments: Iterable[AnalyticalWelfareSegment], beta: float
) -> tuple[_PreparedSegment, ...]:
    prepared = []
    for segment in segments:
        log_attractions = {
            school: (
                segment.systematic_utilities[school] - segment.outside_utility
            )
            / beta
            for school in segment.eligible_schools
        }
        if any(not math.isfinite(value) for value in log_attractions.values()):
            raise ValueError(
                "Utility differences divided by beta must remain finite."
            )
        menu_logs = (0.0, *log_attractions.values())
        if max(menu_logs) - min(menu_logs) > 700.0:
            raise ValueError(
                "Analytical MNL log-attraction range exceeds the supported "
                "double-precision range; use a high-precision evaluator."
            )
        prepared.append(_PreparedSegment(segment, log_attractions))
    return tuple(prepared)


def _integrate_prepared(
    prepared: Iterable[_PreparedSegment],
    capacities: Mapping[int, int],
    cutoffs: Mapping[int, float],
    beta: float,
) -> AnalyticalIntegrationResult:
    demands = {school: 0.0 for school in capacities}
    assignments = {}
    outside = {}
    welfare = 0.0
    max_residual = 0.0
    for item in prepared:
        segment = item.segment
        segment_assignments = {
            school: 0.0 for school in segment.eligible_schools
        }
        events: dict[float, list[int]] = {}
        for school in segment.eligible_schools:
            threshold = _threshold(cutoffs[school], segment.priorities[school])
            events.setdefault(threshold, []).append(school)
        points = sorted({0.0, 1.0, *events})
        active: list[int] = []
        outside_measure = 0.0
        segment_welfare = 0.0
        for left, right in zip(points, points[1:]):
            active.extend(events.get(left, ()))
            duration = right - left
            if duration <= 0:
                continue
            shares, outside_share, inclusive_value = _menu_statistics(
                item.log_attractions, active, beta
            )
            weighted_duration = segment.mass * duration
            for school, share in shares.items():
                measure = weighted_duration * share
                segment_assignments[school] += measure
                demands[school] += measure
            outside_measure += weighted_duration * outside_share
            segment_welfare += weighted_duration * inclusive_value
        if not all(
            math.isfinite(value)
            for value in (
                outside_measure,
                segment_welfare,
                *segment_assignments.values(),
            )
        ):
            raise FloatingPointError(
                f"Non-finite analytical integration for segment {segment.segment_id}."
            )
        assignments[segment.segment_id] = segment_assignments
        outside[segment.segment_id] = outside_measure
        welfare += segment_welfare
        residual = abs(
            math.fsum((outside_measure, *segment_assignments.values())) - segment.mass
        )
        max_residual = max(max_residual, residual)
    if not all(math.isfinite(value) for value in (*demands.values(), welfare)):
        raise FloatingPointError(
            "Non-finite aggregate analytical demand or welfare; use smaller "
            "segment masses or a high-precision evaluator."
        )
    return AnalyticalIntegrationResult(
        demands=demands,
        assignment_measures=assignments,
        outside_measures=outside,
        normalized_welfare=welfare,
        max_mass_balance_residual=max_residual,
    )


def _school_demand_profiles(
    prepared: Iterable[_PreparedSegment],
    school: int,
    cutoffs: Mapping[int, float],
) -> tuple[tuple[float, float, tuple[tuple[float, float, float], ...]], ...]:
    profiles = []
    for item in prepared:
        segment = item.segment
        if school not in item.log_attractions:
            continue
        events: dict[float, list[int]] = {}
        for competitor in segment.eligible_schools:
            if competitor == school:
                continue
            threshold = _threshold(
                cutoffs[competitor], segment.priorities[competitor]
            )
            events.setdefault(threshold, []).append(competitor)
        points = sorted({0.0, 1.0, *events})
        active: list[int] = []
        intervals = []
        for left, right in zip(points, points[1:]):
            active.extend(events.get(left, ()))
            if right <= left:
                continue
            school_share = _school_share_if_added(
                item.log_attractions, active, school
            )
            intervals.append((left, right, school_share))
        profiles.append(
            (
                float(segment.priorities[school]),
                segment.mass,
                tuple(intervals),
            )
        )
    return tuple(profiles)


def _demand_from_profiles(
    profiles: Iterable[tuple[float, float, tuple[tuple[float, float, float], ...]]],
    cutoff: float,
) -> float:
    demand = 0.0
    for priority, mass, intervals in profiles:
        threshold = _threshold(cutoff, priority)
        for left, right, share in intervals:
            duration = right - max(left, threshold)
            if duration > 0:
                demand += mass * duration * share
    return demand


def _minimum_grid_cutoff(
    profiles,
    capacity: int,
    scale: int,
    *,
    demand_tolerance: float,
) -> int:
    if _demand_from_profiles(profiles, 0.0) <= capacity + demand_tolerance:
        return 0
    max_priority = max((profile[0] for profile in profiles), default=0.0)
    low = 0
    high = math.ceil((max_priority + 1.0) * scale)
    while low < high:
        middle = (low + high) // 2
        if (
            _demand_from_profiles(profiles, middle / scale)
            <= capacity + demand_tolerance
        ):
            high = middle
        else:
            low = middle + 1
    return low


def _minimum_continuum_cutoff(
    profiles,
    capacity: int,
    *,
    tolerance: float,
    demand_tolerance: float,
) -> float:
    target = float(capacity)
    demand = _demand_from_profiles(profiles, 0.0)
    if demand <= target + demand_tolerance:
        return 0.0

    # Every profile interval contributes a clipped affine ramp. Sweeping its two
    # endpoints finds the first capacity-feasible cutoff without inner bisection.
    slope_events: dict[float, float] = {}
    for priority, mass, intervals in profiles:
        for left, right, share in intervals:
            weight = mass * share
            start = priority + left
            end = priority + right
            slope_events[start] = slope_events.get(start, 0.0) - weight
            slope_events[end] = slope_events.get(end, 0.0) + weight

    current = 0.0
    slope = sum(
        delta for point, delta in slope_events.items() if point <= tolerance
    )
    for point in sorted(point for point in slope_events if point > tolerance):
        next_demand = demand + slope * (point - current)
        if next_demand <= target and slope < 0:
            root = current + (target - demand) / slope
            return min(point, max(current, root))
        demand = next_demand
        current = point
        slope += slope_events[point]
    return max((profile[0] for profile in profiles), default=0.0) + 1.0


def _threshold(cutoff: float, priority: float) -> float:
    return min(1.0, max(0.0, cutoff - priority))


def _menu_statistics(
    log_attractions: Mapping[int, float],
    active: Iterable[int],
    beta: float,
) -> tuple[dict[int, float], float, float]:
    active = tuple(active)
    maximum = max((0.0, *(log_attractions[school] for school in active)))
    outside_weight = math.exp(-maximum)
    weights = {
        school: math.exp(log_attractions[school] - maximum) for school in active
    }
    denominator = outside_weight + sum(weights.values())
    shares = {school: weight / denominator for school, weight in weights.items()}
    return (
        shares,
        outside_weight / denominator,
        beta * (maximum + math.log(denominator)),
    )


def _school_share_if_added(
    log_attractions: Mapping[int, float], active: Iterable[int], school: int
) -> float:
    menu = (*active, school)
    maximum = max((0.0, *(log_attractions[item] for item in menu)))
    denominator = math.exp(-maximum) + sum(
        math.exp(log_attractions[item] - maximum) for item in menu
    )
    return math.exp(log_attractions[school] - maximum) / denominator

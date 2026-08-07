"""Outer complete-zone column generation for analytical Shi welfare."""

from __future__ import annotations

import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Mapping, Sequence

from optimization.analytical_bounds import prepare_shi_attractions, shi_menu_value
from optimization.branch_price.analytical_master import (
    AnalyticalMasterResult,
    RestrictedAnalyticalPatternMaster,
)
from optimization.branch_price.analytical_patterns import (
    AnalyticalPatternKey,
    AnalyticalPatternValuator,
    AnalyticalZonePattern,
)
from optimization.branch_price.analytical_pricing import (
    AnalyticalPricingResult,
    solve_analytical_pricing,
)
from optimization.problem import ZoneProblem


@dataclass(frozen=True, slots=True)
class ZonedColumnGenerationResult:
    patterns: tuple[AnalyticalZonePattern, ...]
    root_lp_objective: float
    root_lp_upper_bound: float
    root_lp_closed: bool
    root_lp_integral: bool
    root_lp_additive_gap: float
    restricted_mip_objective: float
    incumbent_upper_bound_gap: float
    assignment: dict[int, int]
    rounds: int
    pricing_calls: int
    timing_seconds: float
    selected_patterns: tuple[AnalyticalZonePattern, ...] = ()
    root_lp_status: str = "UNKNOWN"
    restricted_mip_status: str = "UNKNOWN"
    max_pricing_upper_bound_reduced_cost: float = math.inf
    pricing_status_counts: dict[str, int] | None = None
    pricing_results: tuple[AnalyticalPricingResult, ...] = ()
    seed_fallback_used: bool = False


def solve_analytical_pattern_root(
    problem: ZoneProblem,
    seed_patterns: Sequence[AnalyticalZonePattern],
    seed_assignment: Mapping[int, int],
    *,
    valuator: AnalyticalPatternValuator | None = None,
    wall_time_limit: float = 2700.0,
    max_rounds: int = 100,
    pricing_time_limit: float = 300.0,
    pricing_node_limit: int = 10_000,
    columns_per_label: int = 10,
    reduced_cost_tolerance: float = 1e-7,
    menu_tolerance: float = 1e-9,
    master_feasibility_tolerance: float = 1e-8,
    optimality_tolerance: float = 1e-6,
    mip_time_limit: float = 300.0,
    centroid_neighbor_radius: int = 0,
    workers: int = 1,
    random_seed: int = 0,
    deadline: float | None = None,
) -> ZonedColumnGenerationResult:
    """Close the full complete-zone root LP, then solve its restricted MIP."""
    started = time.monotonic()
    local_deadline = started + float(wall_time_limit)
    if deadline is not None:
        local_deadline = min(local_deadline, deadline)
    if isinstance(max_rounds, bool) or max_rounds < 0:
        raise ValueError("max_rounds must be nonnegative.")
    if columns_per_label <= 0:
        raise ValueError("columns_per_label must be positive.")
    if set(seed_assignment) != set(problem.nodes):
        raise ValueError("The analytical root seed must assign every graph node.")
    valuator = valuator or AnalyticalPatternValuator(
        problem,
        centroid_neighbor_radius=centroid_neighbor_radius,
        menu_tolerance=menu_tolerance,
    )
    patterns = _deduplicate_patterns(seed_patterns, optimality_tolerance)
    max_cut_edges = (
        math.floor(problem.boundary_prop * problem.G.number_of_edges())
        if problem.boundary_prop >= 0
        else None
    )
    # Construction checks that the seed's exact set of columns is feasible.
    initial_master = RestrictedAnalyticalPatternMaster(
        problem.G,
        problem.centroids,
        patterns,
        max_cut_edges=max_cut_edges,
        pattern_validator=valuator.validate_pattern,
        welfare_tolerance=optimality_tolerance,
    )
    initial_master.reconstruct_assignment(
        initial_master.patterns_for_assignment(seed_assignment)
    )

    pricing_calls = 0
    pricing_results: list[AnalyticalPricingResult] = []
    best_full_upper = math.inf
    max_upper_reduced_cost = math.inf
    root_closed = False
    rounds = 0
    lp: AnalyticalMasterResult | None = None
    priced_current_master = False

    round_budget = max_rounds if max_rounds > 0 else 1
    for round_index in range(round_budget):
        master = RestrictedAnalyticalPatternMaster(
            problem.G,
            problem.centroids,
            patterns,
            max_cut_edges=max_cut_edges,
            pattern_validator=valuator.validate_pattern,
            welfare_tolerance=optimality_tolerance,
        )
        lp = master.solve_lp(
            feasibility_tolerance=master_feasibility_tolerance,
            optimality_tolerance=optimality_tolerance,
            threads=1,
            time_limit=max(0.0, local_deadline - time.monotonic()),
        )
        if lp.objective is None or lp.duals is None:
            return _seed_timeout_result(
                problem,
                patterns,
                initial_master,
                seed_assignment,
                started=started,
                upper_bound=best_full_upper,
                root_status=lp.status,
                pricing_results=pricing_results,
                rounds=rounds,
            )
        rounds = round_index + 1 if max_rounds > 0 else 0
        remaining = max(0.0, local_deadline - time.monotonic())
        per_label_time = min(float(pricing_time_limit), remaining)
        if max_rounds == 0:
            per_label_time = 0.0
        current_pricing = _price_all_labels(
            problem,
            lp,
            valuator,
            patterns,
            time_limit=per_label_time,
            node_limit=pricing_node_limit,
            menu_tolerance=menu_tolerance,
            reduced_cost_tolerance=reduced_cost_tolerance,
            centroid_neighbor_radius=centroid_neighbor_radius,
            workers=workers,
            random_seed=random_seed + round_index * problem.Z,
            columns_per_label=columns_per_label,
            deadline=local_deadline,
        )
        pricing_results.extend(current_pricing)
        pricing_calls += len(current_pricing)
        priced_current_master = len(current_pricing) == problem.Z
        if not priced_current_master:
            break
        label_bounds = {
            result.label: result.pricing_upper_bound for result in current_pricing
        }
        repaired_upper = sum(lp.duals.coverage.values()) + sum(
            max(lp.duals.convexity[label], label_bounds[label])
            for label in range(problem.Z)
        )
        if max_cut_edges is not None:
            repaired_upper += 2 * max_cut_edges * lp.duals.boundary
        repaired_upper = math.nextafter(repaired_upper, math.inf)
        best_full_upper = min(best_full_upper, repaired_upper)
        max_upper_reduced_cost = max(
            result.reduced_cost_upper_bound for result in current_pricing
        )
        root_closed = max_upper_reduced_cost <= reduced_cost_tolerance

        by_key = {pattern.key: pattern for pattern in patterns}
        additions = []
        for result in current_pricing:
            for candidate in result.candidates[:columns_per_label]:
                if candidate.key in by_key:
                    continue
                reduced_cost = lp.duals.reduced_cost(candidate)
                if reduced_cost <= reduced_cost_tolerance:
                    continue
                if time.monotonic() >= local_deadline:
                    break
                try:
                    revalued = valuator.value(
                        candidate.label,
                        candidate.nodes,
                        deadline=local_deadline,
                        force_revalue=True,
                    )
                except TimeoutError:
                    return _seed_timeout_result(
                        problem,
                        patterns,
                        initial_master,
                        seed_assignment,
                        started=started,
                        upper_bound=best_full_upper,
                        root_status="VALUATION_TIME_LIMIT",
                        pricing_results=pricing_results,
                        rounds=rounds,
                    )
                additions.append(revalued)
                by_key[revalued.key] = revalued
        if additions:
            patterns = tuple((*patterns, *additions))
        if root_closed:
            break
        if not additions:
            break
        if time.monotonic() >= local_deadline:
            break

    if lp is None:
        raise RuntimeError("The analytical root did not solve its restricted LP.")
    # Resolve the final restricted LP if the last pricing round added columns.
    final_master = RestrictedAnalyticalPatternMaster(
        problem.G,
        problem.centroids,
        patterns,
        max_cut_edges=max_cut_edges,
        pattern_validator=valuator.validate_pattern,
        welfare_tolerance=optimality_tolerance,
    )
    final_lp = final_master.solve_lp(
        feasibility_tolerance=master_feasibility_tolerance,
        optimality_tolerance=optimality_tolerance,
        threads=1,
        time_limit=max(0.0, local_deadline - time.monotonic()),
    )
    if final_lp.objective is None:
        return _seed_timeout_result(
            problem,
            patterns,
            initial_master,
            seed_assignment,
            started=started,
            upper_bound=best_full_upper,
            root_status=final_lp.status,
            pricing_results=pricing_results,
            rounds=rounds,
        )
    if not math.isfinite(best_full_upper):
        best_full_upper = final_lp.objective
        root_closed = False
    best_full_upper = max(best_full_upper, final_lp.objective)
    remaining = max(0.0, local_deadline - time.monotonic())
    mip = final_master.solve_mip(
        time_limit=min(float(mip_time_limit), remaining),
        workers=max(1, int(workers)),
        random_seed=random_seed,
        seed_assignment=seed_assignment,
        feasibility_tolerance=master_feasibility_tolerance,
        optimality_tolerance=optimality_tolerance,
    )
    if mip.assignment is None or mip.objective is None:
        raise RuntimeError("Restricted analytical MIP and seed fallback both failed.")
    selected = []
    for master_pattern in mip.selected_patterns:
        if time.monotonic() >= local_deadline:
            selected.append(master_pattern)
            continue
        try:
            refreshed = valuator.value(
                master_pattern.label,
                master_pattern.nodes,
                deadline=local_deadline,
                force_revalue=True,
            )
            if not math.isclose(
                refreshed.shi_welfare,
                master_pattern.shi_welfare,
                rel_tol=optimality_tolerance,
                abs_tol=optimality_tolerance,
            ):
                raise RuntimeError(
                    "Final Shi revaluation disagrees with the master coefficient."
                )
            selected.append(refreshed)
        except TimeoutError:
            selected.append(master_pattern)
    selected = tuple(selected)
    assignment = final_master.reconstruct_assignment(selected)
    incumbent = sum(pattern.shi_welfare for pattern in selected)
    best_full_upper = max(best_full_upper, incumbent)
    root_gap = max(0.0, best_full_upper - final_lp.objective)
    incumbent_gap = max(0.0, best_full_upper - incumbent)
    return ZonedColumnGenerationResult(
        patterns=tuple(patterns),
        root_lp_objective=float(final_lp.objective),
        root_lp_upper_bound=float(best_full_upper),
        root_lp_closed=root_closed and priced_current_master,
        root_lp_integral=final_lp.assignment is not None,
        root_lp_additive_gap=root_gap,
        restricted_mip_objective=incumbent,
        incumbent_upper_bound_gap=incumbent_gap,
        assignment=assignment,
        rounds=rounds,
        pricing_calls=pricing_calls,
        timing_seconds=time.monotonic() - started,
        selected_patterns=selected,
        root_lp_status=final_lp.status,
        restricted_mip_status=mip.status,
        max_pricing_upper_bound_reduced_cost=max_upper_reduced_cost,
        pricing_status_counts=dict(
            Counter(result.status for result in pricing_results)
        ),
        pricing_results=tuple(pricing_results),
        seed_fallback_used=mip.seed_fallback_used,
    )


def _price_all_labels(
    problem: ZoneProblem,
    lp: AnalyticalMasterResult,
    valuator: AnalyticalPatternValuator,
    patterns: Sequence[AnalyticalZonePattern],
    *,
    time_limit: float,
    node_limit: int,
    menu_tolerance: float,
    reduced_cost_tolerance: float,
    centroid_neighbor_radius: int,
    workers: int,
    random_seed: int,
    columns_per_label: int,
    deadline: float,
) -> tuple[AnalyticalPricingResult, ...]:
    duals = lp.duals

    def price(label: int) -> AnalyticalPricingResult:
        return solve_analytical_pricing(
            problem,
            label,
            duals,
            valuator=valuator,
            seed_patterns=patterns,
            time_limit=time_limit,
            node_limit=node_limit,
            menu_tolerance=menu_tolerance,
            reduced_cost_tolerance=reduced_cost_tolerance,
            centroid_neighbor_radius=centroid_neighbor_radius,
            workers=1,
            random_seed=random_seed + label,
            columns_per_label=columns_per_label,
            deadline=deadline,
        )

    max_workers = min(problem.Z, max(1, int(workers)))
    if max_workers == 1:
        return tuple(price(label) for label in range(problem.Z))
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(price, label): label for label in range(problem.Z)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return tuple(results[label] for label in range(problem.Z))


def _deduplicate_patterns(
    patterns: Sequence[AnalyticalZonePattern],
    tolerance: float,
) -> tuple[AnalyticalZonePattern, ...]:
    by_key: dict[AnalyticalPatternKey, AnalyticalZonePattern] = {}
    for pattern in patterns:
        previous = by_key.get(pattern.key)
        if previous is not None and (
            previous.perimeter != pattern.perimeter
            or not math.isclose(
                previous.shi_welfare,
                pattern.shi_welfare,
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
        ):
            raise ValueError("Duplicate analytical pattern has conflicting valuation.")
        by_key.setdefault(pattern.key, pattern)
    return tuple(by_key.values())


def _seed_timeout_result(
    problem: ZoneProblem,
    patterns: Sequence[AnalyticalZonePattern],
    seed_master: RestrictedAnalyticalPatternMaster,
    seed_assignment: Mapping[int, int],
    *,
    started: float,
    upper_bound: float,
    root_status: str,
    pricing_results: Sequence[AnalyticalPricingResult],
    rounds: int,
) -> ZonedColumnGenerationResult:
    selected = seed_master.patterns_for_assignment(seed_assignment)
    assignment = seed_master.reconstruct_assignment(selected)
    incumbent = sum(pattern.shi_welfare for pattern in selected)
    if not math.isfinite(upper_bound):
        upper_bound = _unconstrained_district_upper(problem)
    upper_bound = max(upper_bound, incumbent)
    return ZonedColumnGenerationResult(
        patterns=tuple(patterns),
        root_lp_objective=incumbent,
        root_lp_upper_bound=upper_bound,
        root_lp_closed=False,
        root_lp_integral=False,
        root_lp_additive_gap=max(0.0, upper_bound - incumbent),
        restricted_mip_objective=incumbent,
        incumbent_upper_bound_gap=max(0.0, upper_bound - incumbent),
        assignment=assignment,
        rounds=rounds,
        pricing_calls=len(pricing_results),
        timing_seconds=time.monotonic() - started,
        selected_patterns=selected,
        root_lp_status=root_status,
        restricted_mip_status="NOT_SOLVED_SEED_FALLBACK",
        max_pricing_upper_bound_reduced_cost=max(
            (result.reduced_cost_upper_bound for result in pricing_results),
            default=math.inf,
        ),
        pricing_status_counts=dict(
            Counter(result.status for result in pricing_results)
        ),
        pricing_results=tuple(pricing_results),
        seed_fallback_used=True,
    )


def _unconstrained_district_upper(problem: ZoneProblem) -> float:
    market = problem.analytical_welfare_market
    bound = 0.0
    for segment in market.segments:
        attractions = prepare_shi_attractions(segment, market.beta)
        welfare, _, _ = shi_menu_value(
            attractions,
            segment.eligible_schools,
            market.beta,
        )
        bound += segment.mass * welfare
    return math.nextafter(bound, math.inf)

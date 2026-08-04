"""Root restricted master and safe access-pricing certificate."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

from optimization.branch_price.access_pricing import (
    AccessPricingResult,
    build_access_pricing_template,
    solve_access_pricing,
)
from optimization.branch_price.certificate import (
    LagrangianCertificate,
    assemble_lagrangian_certificate,
    quantize_multipliers,
)
from optimization.branch_price.exact_pricing import (
    ExactPricingResult,
    solve_exact_pricing,
)
from optimization.branch_price.master import (
    PatternMasterResult,
    RestrictedPatternMaster,
)
from optimization.branch_price.patterns import ZonePattern, ZonePatternValidator
from optimization.problem import ZoneProblem


@dataclass(frozen=True, slots=True)
class PatternRootResult:
    """One restricted-master pass and a global integer access certificate."""

    initial_lp: PatternMasterResult
    enriched_lp: PatternMasterResult
    restricted_mip: PatternMasterResult
    pricing: tuple[AccessPricingResult, ...]
    exact_pricing: tuple[ExactPricingResult, ...]
    certificate: LagrangianCertificate
    patterns: tuple[ZonePattern, ...]
    added_patterns: tuple[ZonePattern, ...]


def solve_pattern_root(
    problem: ZoneProblem,
    patterns: Sequence[ZonePattern],
    *,
    utility_scale: int,
    centroid_neighbor_radius: int = 0,
    pricing_time_limit: float = 60.0,
    mip_time_limit: float = 30.0,
    exact_pricing_time_limit: float = 0.0,
    run_access_models: bool = True,
    workers: int = 1,
    random_seed: int = 0,
) -> PatternRootResult:
    """Price one access column per label and certify a safe global root bound."""
    if not patterns:
        raise ValueError("Pattern-root solving requires at least one seed pattern.")
    if isinstance(utility_scale, bool) or not isinstance(utility_scale, Integral):
        raise TypeError("utility_scale must be an integer.")
    if utility_scale <= 0:
        raise ValueError("utility_scale must be positive.")
    if pricing_time_limit < 0 or mip_time_limit < 0 or exact_pricing_time_limit < 0:
        raise ValueError("Pattern-root time limits must be nonnegative.")
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers <= 0:
        raise ValueError("workers must be a positive integer.")

    max_cut_edges = (
        math.floor(problem.boundary_prop * problem.G.number_of_edges())
        if problem.boundary_prop >= 0
        else problem.G.number_of_edges()
    )
    validator = ZonePatternValidator(
        problem,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )
    initial_master = RestrictedPatternMaster(
        problem.G,
        problem.centroids,
        patterns,
        max_cut_edges=max_cut_edges,
        pattern_validator=validator,
    )
    initial_lp = initial_master.solve_lp()
    if initial_lp.status != "OPTIMAL" or initial_lp.duals is None:
        raise ValueError(
            f"Seed patterns do not form a feasible root LP: {initial_lp.status}."
        )

    multipliers = quantize_multipliers(
        initial_lp.duals.coverage,
        initial_lp.duals.boundary,
    )
    pricing_deadline = time.monotonic() + pricing_time_limit
    template = (
        build_access_pricing_template(
            problem,
            utility_scale=int(utility_scale),
        )
        if run_access_models and pricing_time_limit > 0
        else None
    )
    pricing_results = []
    for label in range(problem.Z):
        remaining_labels = problem.Z - label
        seconds = (
            max(0.0, pricing_deadline - time.monotonic()) / remaining_labels
            if run_access_models
            else 0.0
        )
        pricing_results.append(
            solve_access_pricing(
                problem,
                label,
                utility_scale=int(utility_scale),
                multipliers=multipliers,
                centroid_neighbor_radius=centroid_neighbor_radius,
                time_limit=seconds,
                workers=int(workers),
                random_seed=int(random_seed) + label,
                template=template,
            )
        )
    pricing = tuple(pricing_results)
    pricing_bounds = {
        result.label: result.pricing_lagrangian_upper_bound for result in pricing
    }
    exact_results = []
    if exact_pricing_time_limit > 0:
        exact_label = max(
            pricing, key=lambda result: result.pricing_lagrangian_upper_bound
        ).label
        seed_pattern = max(
            (
                pattern
                for pattern in initial_master.patterns
                if pattern.label == exact_label
            ),
            key=lambda pattern: (
                pattern.raw_welfare
                - sum(multipliers.node.get(node, 0) for node in pattern.nodes)
                - multipliers.boundary * pattern.perimeter
            ),
        )
        exact_result = solve_exact_pricing(
            problem,
            exact_label,
            utility_scale=int(utility_scale),
            multipliers=multipliers,
            centroid_neighbor_radius=centroid_neighbor_radius,
            time_limit=exact_pricing_time_limit,
            workers=int(workers),
            random_seed=int(random_seed) + 10_000 + exact_label,
            seed_pattern=seed_pattern,
        )
        exact_results.append(exact_result)
        pricing_bounds[exact_label] = min(
            pricing_bounds[exact_label],
            exact_result.pricing_lagrangian_upper_bound,
        )
    certificate = assemble_lagrangian_certificate(
        labels=range(problem.Z),
        coverage_nodes=initial_master.coverage_nodes,
        zone_perimeter_cap=initial_master.zone_perimeter_cap,
        multipliers=multipliers,
        pricing_upper_bounds=pricing_bounds,
    )

    by_key = {pattern.key: pattern for pattern in initial_master.patterns}
    added = []
    for result in pricing:
        candidate = result.candidate
        if candidate is None:
            continue
        previous = by_key.get(candidate.key)
        if previous is not None:
            if previous.raw_welfare != candidate.raw_welfare:
                raise ValueError(
                    "A priced pattern conflicts with its exact cached value."
                )
            continue
        by_key[candidate.key] = candidate
        added.append(candidate)
    for result in exact_results:
        candidate = result.candidate
        if candidate is None or candidate.key in by_key:
            continue
        by_key[candidate.key] = candidate
        added.append(candidate)

    enriched_master = RestrictedPatternMaster(
        problem.G,
        problem.centroids,
        tuple(by_key.values()),
        max_cut_edges=max_cut_edges,
        pattern_validator=validator,
    )
    enriched_lp = enriched_master.solve_lp()
    restricted_mip = enriched_master.solve_mip(
        time_limit=mip_time_limit,
        workers=int(workers),
        random_seed=int(random_seed),
    )
    return PatternRootResult(
        initial_lp=initial_lp,
        enriched_lp=enriched_lp,
        restricted_mip=restricted_mip,
        pricing=pricing,
        exact_pricing=tuple(exact_results),
        certificate=certificate,
        patterns=enriched_master.patterns,
        added_patterns=tuple(added),
    )

"""Mathematically screened graph-school bundle exchanges for analytical welfare."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from optimization.analytical_fixed_state_master import (
    FixedStateMasterResult,
    optimize_fixed_school_bundles,
    solve_fixed_analytical_states,
)
from optimization.analytical_welfare_oracle import evaluate_zoned_analytical_welfare
from optimization.problem import AnalyticalWelfareMarket, ZoneProblem


@dataclass(frozen=True)
class BundleSwapResult:
    assignment: dict[int, int]
    q20_welfare: float
    considered_swaps: int
    exactly_evaluated_swaps: int
    completed_swaps: int
    best_swap: tuple[int, int] | None
    records: tuple[dict, ...]
    timing_seconds: float


def search_graph_school_swaps(
    problem: ZoneProblem,
    market: AnalyticalWelfareMarket,
    initial_assignment: Mapping[int, int],
    *,
    exact_candidates: int = 40,
    completion_candidates: int = 10,
    completion_time_limit: float = 30.0,
) -> BundleSwapResult:
    """Screen all one-for-one school swaps, then solve strongest completions."""
    started = time.monotonic()
    if exact_candidates <= 0 or completion_candidates <= 0:
        raise ValueError("candidate counts must be positive.")
    assignment = dict(initial_assignment)
    baseline = evaluate_zoned_analytical_welfare(
        market,
        assignment,
        num_zones=problem.Z,
        cutoff_grid=market.lottery_scale,
    )
    graph_school_nodes = tuple(
        node for node in problem.nodes if problem.num_schools(node) > 0
    )
    centroid_nodes = set(problem.centroids)
    swappable = tuple(node for node in graph_school_nodes if node not in centroid_nodes)
    market_school_columns = {
        school: index for index, school in enumerate(market.school_capacities)
    }
    node_columns = {
        node: tuple(
            market_school_columns[school]
            for school, school_node in market.school_nodes.items()
            if school_node == node
        )
        for node in graph_school_nodes
    }
    segment_zones = np.asarray(
        [assignment[segment.node] for segment in market.segments], dtype=int
    )
    attraction = np.zeros((len(market.segments), len(market_school_columns)))
    for row, segment in enumerate(market.segments):
        for school in segment.eligible_schools:
            attraction[row, market_school_columns[school]] = math.exp(
                (
                    segment.systematic_utilities[school]
                    - segment.outside_utility
                )
                / market.beta
            )
    bundle_columns = {
        label: tuple(
            market_school_columns[school]
            for school, node in market.school_nodes.items()
            if assignment[node] == label
        )
        for label in range(problem.Z)
    }
    denominators = {
        label: 1.0 + attraction[:, bundle_columns[label]].sum(axis=1)
        for label in range(problem.Z)
    }
    ranked = []
    for first_index, first in enumerate(swappable):
        first_zone = assignment[first]
        for second in swappable[first_index + 1 :]:
            second_zone = assignment[second]
            if first_zone == second_zone:
                continue
            first_remove = attraction[:, node_columns[first]].sum(axis=1)
            second_remove = attraction[:, node_columns[second]].sum(axis=1)
            first_mask = segment_zones == first_zone
            second_mask = segment_zones == second_zone
            first_new = (
                denominators[first_zone] - first_remove + second_remove
            )
            second_new = (
                denominators[second_zone] - second_remove + first_remove
            )
            delta = market.beta * (
                np.log(first_new[first_mask] / denominators[first_zone][first_mask]).sum()
                + np.log(
                    second_new[second_mask] / denominators[second_zone][second_mask]
                ).sum()
            )
            ranked.append((float(delta), first, second))
    ranked.sort(reverse=True)

    exact_records = []
    for access_delta, first, second in ranked[:exact_candidates]:
        provisional = dict(assignment)
        provisional[first], provisional[second] = (
            provisional[second],
            provisional[first],
        )
        q20 = evaluate_zoned_analytical_welfare(
            market,
            provisional,
            num_zones=problem.Z,
            cutoff_grid=market.lottery_scale,
        )
        exact_records.append(
            {
                "swap": (first, second),
                "access_delta": access_delta,
                "provisional_q20": q20.normalized_welfare,
                "provisional_assignment": provisional,
            }
        )
    exact_records.sort(key=lambda record: record["provisional_q20"], reverse=True)

    best_assignment = assignment
    best_welfare = baseline.normalized_welfare
    best_swap = None
    completed = 0
    output_records = []
    for record in exact_records[:completion_candidates]:
        try:
            completed_result: FixedStateMasterResult = solve_fixed_analytical_states(
                problem,
                market,
                record["provisional_assignment"],
                time_limit=completion_time_limit,
                enforce_market_capacities=False,
            )
        except RuntimeError as exc:
            output_records.append(
                {
                    key: value
                    for key, value in record.items()
                    if key != "provisional_assignment"
                }
                | {"completion_status": "INFEASIBLE_OR_FAILED", "error": str(exc)}
            )
            continue
        completed += 1
        refined = optimize_fixed_school_bundles(
            problem,
            market,
            completed_result.assignment,
            time_limit=completion_time_limit,
            max_iterations=5,
        )
        welfare = refined.q20_result.normalized_welfare
        output_records.append(
            {
                key: value
                for key, value in record.items()
                if key != "provisional_assignment"
            }
            | {
                "completion_status": completed_result.status,
                "completed_q20": welfare,
                "completion_seconds": (
                    completed_result.total_seconds + refined.timing_seconds
                ),
            }
        )
        if welfare > best_welfare:
            best_welfare = welfare
            best_assignment = refined.assignment
            best_swap = tuple(record["swap"])
    return BundleSwapResult(
        assignment=best_assignment,
        q20_welfare=best_welfare,
        considered_swaps=len(ranked),
        exactly_evaluated_swaps=min(exact_candidates, len(ranked)),
        completed_swaps=completed,
        best_swap=best_swap,
        records=tuple(output_records),
        timing_seconds=time.monotonic() - started,
    )

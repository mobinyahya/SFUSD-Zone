"""Expected cardinal welfare at isolated DA-STB score-limit equilibria."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

from optimization.cutoff_oracle import (
    ZonedContinuumCutoffResult,
    ZonedCutoffResult,
    solve_zoned_continuum_cutoffs,
    solve_zoned_cutoffs,
)
from optimization.problem import CutoffMarket


MAX_EXACT_CP_SAT_OBJECTIVE = 2**53 - 1


@dataclass(frozen=True)
class WelfareResult:
    """Exact finite-grid assignment measures and their cardinal welfare."""

    cutoffs: ZonedCutoffResult
    welfare: float
    scaled_welfare: int
    utility_scale: int
    assignments: dict[int, dict[int, int]]
    outside_option_mass: dict[int, int]

    @property
    def raw_scaled_welfare(self) -> int:
        return self.scaled_welfare


@dataclass(frozen=True)
class ContinuumWelfareResult:
    """Continuous-lottery welfare and market-clearing validation."""

    cutoffs: ZonedContinuumCutoffResult
    welfare: float

    @property
    def stable(self) -> bool:
        return self.cutoffs.stable


def solve_zoned_welfare(
    market: CutoffMarket,
    node_assignment: Mapping[int, int],
    *,
    num_zones: int | None = None,
    utility_scale: int = 1_000_000,
) -> WelfareResult:
    """Evaluate least-cutoff finite-grid welfare without drawing tie breakers."""
    if isinstance(utility_scale, bool) or not isinstance(utility_scale, int):
        raise ValueError("utility_scale must be a positive integer.")
    if utility_scale <= 0:
        raise ValueError("utility_scale must be a positive integer.")
    validate_welfare_market(market, utility_scale=utility_scale)
    cutoffs = solve_zoned_cutoffs(market, node_assignment, num_zones=num_zones)
    assignments: dict[int, dict[int, int]] = {}
    outside_option_mass: dict[int, int] = {}
    welfare_numerator = 0.0
    scaled_welfare = 0
    scale = market.lottery_scale

    for student in market.students:
        zone = int(node_assignment[student.node])
        remaining = scale
        student_assignment = {}
        for school in student.preferences:
            if int(node_assignment[market.school_nodes[school]]) != zone:
                continue
            if school not in student.utilities:
                raise ValueError(
                    f"Student {student.studentno} lacks utility for school {school}."
                )
            threshold = min(
                scale,
                max(
                    0,
                    cutoffs.school_cutoffs[school]
                    - student.priorities[school] * scale,
                ),
            )
            mass = max(0, remaining - threshold)
            if mass:
                utility = student.utilities[school]
                student_assignment[school] = mass
                welfare_numerator += mass * utility
                scaled_welfare += mass * round(utility * utility_scale)
            remaining = min(remaining, threshold)
        assignments[student.studentno] = student_assignment
        outside_option_mass[student.studentno] = remaining

    return WelfareResult(
        cutoffs=cutoffs,
        welfare=welfare_numerator / scale,
        scaled_welfare=scaled_welfare,
        utility_scale=utility_scale,
        assignments=assignments,
        outside_option_mass=outside_option_mass,
    )


def solve_zoned_continuum_welfare(
    market: CutoffMarket,
    node_assignment: Mapping[int, int],
    *,
    num_zones: int | None = None,
) -> ContinuumWelfareResult:
    """Evaluate cardinal welfare at continuous isolated-market cutoffs."""
    validate_welfare_market(market)
    cutoffs = solve_zoned_continuum_cutoffs(
        market, node_assignment, num_zones=num_zones
    )
    welfare = 0.0
    for student in market.students:
        zone = int(node_assignment[student.node])
        remaining = 1.0
        for school in student.preferences:
            if int(node_assignment[market.school_nodes[school]]) != zone:
                continue
            if school not in student.utilities:
                raise ValueError(
                    f"Student {student.studentno} lacks utility for school {school}."
                )
            threshold = min(
                1.0,
                max(
                    0.0,
                    cutoffs.school_cutoffs[school] - student.priorities[school],
                ),
            )
            welfare += max(0.0, remaining - threshold) * student.utilities[school]
            remaining = min(remaining, threshold)
    return ContinuumWelfareResult(cutoffs=cutoffs, welfare=welfare)


def validate_welfare_market(
    market: CutoffMarket, *, utility_scale: int | None = None
) -> None:
    """Validate assumptions required by the stable-welfare formulation."""
    if not math.isfinite(market.outside_option_utility):
        raise ValueError("Welfare requires a finite outside-option utility.")
    if market.outside_option_utility != 0.0:
        raise ValueError("Welfare currently requires outside-option utility zero.")
    for student in market.students:
        missing = set(student.preferences) - set(student.utilities)
        if missing:
            raise ValueError(
                f"Student {student.studentno} lacks utilities for {sorted(missing)}."
            )
        ordered = [student.utilities[school] for school in student.preferences]
        if any(not math.isfinite(utility) for utility in ordered):
            raise ValueError("Welfare utilities must be finite.")
        if any(utility <= market.outside_option_utility for utility in ordered):
            raise ValueError(
                "Welfare preferences may only contain schools strictly preferred "
                "to the outside option."
            )
        if any(left < right for left, right in zip(ordered, ordered[1:])):
            raise ValueError(
                "Welfare utilities must be nonincreasing in preference order."
            )
    if utility_scale is not None:
        raw_welfare_upper_bound(market, utility_scale)


def raw_welfare_upper_bound(market: CutoffMarket, utility_scale: int) -> int:
    """Return a safely float-representable integer upper bound."""
    upper_bound = sum(
        market.lottery_scale
        * max(
            (
                round(student.utilities[school] * utility_scale)
                for school in student.preferences
            ),
            default=0,
        )
        for student in market.students
    )
    if upper_bound > MAX_EXACT_CP_SAT_OBJECTIVE:
        raise ValueError(
            "The scaled welfare objective exceeds the exact CP-SAT reporting "
            f"range ({MAX_EXACT_CP_SAT_OBJECTIVE}); lower welfare_utility_scale."
        )
    return upper_bound


def outward_true_welfare_upper_bound(
    raw_upper_bound: int,
    market: CutoffMarket,
    utility_scale: int,
) -> float:
    """Round the fixed-point plus coefficient-error bound upward to float."""
    exact = Fraction(
        raw_upper_bound, market.lottery_scale * utility_scale
    ) + Fraction(len(market.students), 2 * utility_scale)
    value = float(exact)
    while Fraction.from_float(value) < exact:
        value = math.nextafter(value, math.inf)
    return value

"""Floating upper bounds for analytical expected-MNL stable welfare."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from optimization.problem import AnalyticalWelfareSegment


@dataclass(frozen=True)
class ShiBoundResult:
    """Complete ordinary-floating Shi menu-LP upper bound."""

    upper_bound: float
    school_prices: dict[int, float]
    type_potentials: dict[int, float]
    rounds: int
    constraint_count: int
    max_pricing_violation: float
    timing_seconds: float
    status: str


def solve_shi_menu_bound(
    segments: Iterable[AnalyticalWelfareSegment],
    school_capacities: Mapping[int, int],
    *,
    beta: float,
    tolerance: float = 1e-8,
    max_rounds: int = 100,
    cardinality: int | None = None,
) -> ShiBoundResult:
    """Solve Shi's complete menu LP through exact MNL prefix separation.

    This routine uses ordinary IEEE-754 LP arithmetic. It is a diagnostic upper
    bound and price generator, not a directed-rounding proof artifact.
    """
    started = time.monotonic()
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive.")
    if cardinality is not None and (
        isinstance(cardinality, bool)
        or not isinstance(cardinality, int)
        or cardinality < 0
    ):
        raise ValueError("cardinality must be a non-negative integer or None.")
    segments = tuple(segments)
    schools = tuple(int(school) for school in school_capacities)
    school_index = {school: index for index, school in enumerate(schools)}
    capacities = np.asarray(
        [float(school_capacities[school]) for school in schools], dtype=float
    )
    if np.any(~np.isfinite(capacities)) or np.any(capacities < 0):
        raise ValueError("School capacities must be finite and non-negative.")
    if len({segment.segment_id for segment in segments}) != len(segments):
        raise ValueError("segment_id values must be unique.")

    attractions = []
    eligible = []
    for segment in segments:
        if not math.isfinite(segment.mass) or segment.mass <= 0:
            raise ValueError("Segment masses must be positive and finite.")
        if set(segment.eligible_schools) - set(schools):
            raise ValueError("A segment is eligible for an unknown school.")
        logs = {
            school: (
                segment.systematic_utilities[school] - segment.outside_utility
            )
            / beta
            for school in segment.eligible_schools
        }
        if any(not math.isfinite(value) for value in logs.values()):
            raise ValueError("MNL log attractions must be finite.")
        maximum = max((0.0, *logs.values()))
        if maximum > 700:
            raise ValueError("MNL attraction scale requires high-precision arithmetic.")
        attractions.append(
            {school: math.exp(value) for school, value in logs.items()}
        )
        eligible.append(frozenset(segment.eligible_schools))

    type_count = len(segments)
    variable_count = type_count + len(schools)
    objective = np.concatenate(
        (
            np.asarray([segment.mass for segment in segments], dtype=float),
            capacities,
        )
    )
    generated: list[tuple[int, tuple[int, ...], float, tuple[float, ...]]] = []
    seen = set()
    solution = None
    max_violation = math.inf
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        if generated:
            rows = []
            columns = []
            values = []
            rhs = []
            for row, (type_index, menu, welfare, shares) in enumerate(generated):
                rows.append(row)
                columns.append(type_index)
                values.append(-1.0)
                for school, share in zip(menu, shares, strict=True):
                    rows.append(row)
                    columns.append(type_count + school_index[school])
                    values.append(-share)
                rhs.append(-welfare)
            constraints = coo_matrix(
                (values, (rows, columns)),
                shape=(len(generated), variable_count),
            ).tocsr()
            upper_rhs = np.asarray(rhs, dtype=float)
        else:
            constraints = None
            upper_rhs = None
        solution = linprog(
            objective,
            A_ub=constraints,
            b_ub=upper_rhs,
            bounds=(0.0, None),
            method="highs",
            options={
                "dual_feasibility_tolerance": max(1e-10, min(1e-7, tolerance / 10)),
                "primal_feasibility_tolerance": max(
                    1e-10, min(1e-7, tolerance / 10)
                ),
            },
        )
        if not solution.success:
            raise RuntimeError(f"Shi dual LP failed: {solution.message}")
        potentials = solution.x[:type_count]
        prices = solution.x[type_count:]
        price_order = tuple(
            sorted(schools, key=lambda school: (prices[school_index[school]], school))
        )
        violations = []
        max_violation = 0.0
        for type_index, segment in enumerate(segments):
            if cardinality is None or cardinality >= len(eligible[type_index]):
                menu, welfare, shares, priced_value = _best_shi_prefix(
                    price_order,
                    eligible[type_index],
                    attractions[type_index],
                    prices,
                    school_index,
                    beta,
                )
            else:
                menu, welfare, shares, priced_value = _best_shi_cardinality(
                    eligible[type_index],
                    attractions[type_index],
                    prices,
                    school_index,
                    beta,
                    cardinality,
                )
            violation = priced_value - potentials[type_index]
            max_violation = max(max_violation, violation)
            key = (type_index, menu)
            if violation > tolerance and key not in seen:
                violations.append((violation, key, welfare, shares))
        if not violations:
            break
        for _, key, welfare, shares in violations:
            type_index, menu = key
            seen.add(key)
            generated.append((type_index, menu, welfare, shares))
    else:
        raise RuntimeError(
            f"Shi menu pricing did not close after {max_rounds} rounds; "
            f"maximum violation {max_violation}."
        )

    return ShiBoundResult(
        upper_bound=float(solution.fun),
        school_prices={
            school: float(solution.x[type_count + school_index[school]])
            for school in schools
        },
        type_potentials={
            segment.segment_id: float(solution.x[index])
            for index, segment in enumerate(segments)
        },
        rounds=rounds,
        constraint_count=len(generated),
        max_pricing_violation=max_violation,
        timing_seconds=time.monotonic() - started,
        status="OPTIMAL_FLOATING",
    )


def _best_shi_prefix(
    price_order: tuple[int, ...],
    eligible: frozenset[int],
    attractions: Mapping[int, float],
    prices: np.ndarray,
    school_index: Mapping[int, int],
    beta: float,
) -> tuple[tuple[int, ...], float, tuple[float, ...], float]:
    best_menu: tuple[int, ...] = ()
    best_welfare = 0.0
    best_shares: tuple[float, ...] = ()
    best_value = 0.0
    menu = []
    attraction_sum = 0.0
    priced_attraction_sum = 0.0
    for school in price_order:
        if school not in eligible:
            continue
        attraction = attractions[school]
        menu.append(school)
        attraction_sum += attraction
        priced_attraction_sum += prices[school_index[school]] * attraction
        denominator = 1.0 + attraction_sum
        welfare = beta * math.log(denominator)
        value = welfare - priced_attraction_sum / denominator
        if value > best_value:
            best_menu = tuple(menu)
            best_welfare = welfare
            best_shares = tuple(attractions[item] / denominator for item in menu)
            best_value = value
    return best_menu, best_welfare, best_shares, best_value


def _best_shi_cardinality(
    eligible: frozenset[int],
    attractions: Mapping[int, float],
    prices: np.ndarray,
    school_index: Mapping[int, int],
    beta: float,
    cardinality: int,
) -> tuple[tuple[int, ...], float, tuple[float, ...], float]:
    """Price Shi's MNL assortment with at most ``cardinality`` schools."""
    if cardinality <= 0 or not eligible:
        return (), 0.0, (), 0.0
    items = sorted(
        eligible,
        key=lambda school: (
            -attractions[school],
            prices[school_index[school]],
            school,
        ),
    )
    label = {school: index for index, school in enumerate(items)}
    rank = {school: index for index, school in enumerate(items)}
    active = {school: True for school in items}
    selected = set(items[: min(cardinality, len(items))])
    events = []
    for school in items:
        revenue = -float(prices[school_index[school]])
        events.append((revenue, -label[school], 0, 0, school, -1))
    for first in items:
        first_weight = attractions[first]
        first_revenue = -float(prices[school_index[first]])
        for second in items:
            second_weight = attractions[second]
            if first_weight <= second_weight:
                continue
            second_revenue = -float(prices[school_index[second]])
            crossing = (
                first_weight * first_revenue
                - second_weight * second_revenue
            ) / (first_weight - second_weight)
            events.append(
                (
                    crossing,
                    -label[first],
                    label[second],
                    1,
                    first,
                    second,
                )
            )
    events.sort()

    attraction_sum = sum(attractions[school] for school in selected)
    priced_sum = sum(
        prices[school_index[school]] * attractions[school] for school in selected
    )

    def evaluate():
        denominator = 1.0 + attraction_sum
        welfare = beta * math.log(denominator)
        return welfare, welfare - priced_sum / denominator

    best_welfare, best_value = evaluate()
    best_menu = tuple(sorted(selected))
    for _, _, _, event_kind, first, second in events:
        old_membership = {first: first in selected}
        if second >= 0:
            old_membership[second] = second in selected
            if rank[first] < rank[second]:
                rank[first], rank[second] = rank[second], rank[first]
        else:
            active[first] = False
        changed = False
        for school, was_selected in old_membership.items():
            is_selected = active[school] and rank[school] < cardinality
            if is_selected == was_selected:
                continue
            changed = True
            if is_selected:
                selected.add(school)
                attraction_sum += attractions[school]
                priced_sum += prices[school_index[school]] * attractions[school]
            else:
                selected.discard(school)
                attraction_sum -= attractions[school]
                priced_sum -= prices[school_index[school]] * attractions[school]
        if changed:
            welfare, value = evaluate()
            if value > best_value:
                best_welfare = welfare
                best_value = value
                best_menu = tuple(sorted(selected))
    denominator = 1.0 + sum(attractions[school] for school in best_menu)
    shares = tuple(attractions[school] / denominator for school in best_menu)
    return best_menu, best_welfare, shares, best_value

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


@dataclass(frozen=True, slots=True)
class ShiBoundResult:
    """Backward-compatible ordinary-floating Shi bound contract."""

    upper_bound: float
    school_prices: dict[int, float]
    type_potentials: dict[int, float]
    rounds: int
    constraint_count: int
    max_pricing_violation: float
    timing_seconds: float
    status: str


@dataclass(frozen=True, slots=True)
class ShiMechanismResult(ShiBoundResult):
    """Closed ordinary-floating Shi menu LP and sparse primal witness."""

    primal_objective: float
    dual_objective: float
    repaired_upper_bound: float
    menu_probabilities: dict[
        int, tuple[tuple[tuple[int, ...], float], ...]
    ]
    quotas: dict[int, float]
    closed: bool
    max_primal_capacity_violation: float = 0.0
    max_primal_probability_residual: float = 0.0
    max_dual_feasibility_violation: float = 0.0
    primal_dual_gap: float = 0.0


def solve_shi_menu_bound(
    segments: Iterable[AnalyticalWelfareSegment],
    school_capacities: Mapping[int, int],
    *,
    beta: float,
    tolerance: float = 1e-8,
    max_rounds: int = 100,
    cardinality: int | None = None,
    deadline: float | None = None,
) -> ShiMechanismResult:
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

    attractions: list[dict[int, float]] = []
    eligible = []
    for segment in segments:
        if not math.isfinite(segment.mass) or segment.mass <= 0:
            raise ValueError("Segment masses must be positive and finite.")
        if not math.isfinite(segment.outside_utility):
            raise ValueError("Outside systematic utilities must be finite.")
        if len(segment.eligible_schools) != len(set(segment.eligible_schools)):
            raise ValueError("A segment repeats an eligible school.")
        if set(segment.eligible_schools) - set(schools):
            raise ValueError("A segment is eligible for an unknown school.")
        if set(segment.systematic_utilities) != set(segment.eligible_schools):
            raise ValueError("Segment utilities must match eligibility.")
        logs = {
            school: (
                segment.systematic_utilities[school] - segment.outside_utility
            )
            / beta
            for school in segment.eligible_schools
        }
        if any(not math.isfinite(value) for value in logs.values()):
            raise ValueError("MNL log attractions must be finite.")
        menu_logs = (0.0, *logs.values())
        if max(menu_logs) - min(menu_logs) > 700.0:
            raise ValueError(
                "Analytical MNL log-attraction range exceeds the supported "
                "double-precision range; use a high-precision evaluator."
            )
        segment_attractions = {
            school: math.exp(value) for school, value in logs.items()
        }
        if any(
            not math.isfinite(value) or value <= 0
            for value in segment_attractions.values()
        ):
            raise ValueError("MNL attractions must be finite and strictly positive.")
        attractions.append(segment_attractions)
        eligible.append(frozenset(segment.eligible_schools))

    if not segments:
        return ShiMechanismResult(
            upper_bound=0.0,
            primal_objective=0.0,
            dual_objective=0.0,
            repaired_upper_bound=0.0,
            menu_probabilities={},
            quotas={school: 0.0 for school in schools},
            school_prices={school: 0.0 for school in schools},
            type_potentials={},
            max_pricing_violation=0.0,
            closed=True,
            status="OPTIMAL_FLOATING",
            rounds=0,
            constraint_count=0,
            timing_seconds=time.monotonic() - started,
        )

    type_count = len(segments)
    variable_count = type_count + len(schools)
    objective = np.concatenate(
        (
            np.asarray([segment.mass for segment in segments], dtype=float),
            capacities,
        )
    )
    generated: list[tuple[int, tuple[int, ...], float, tuple[float, ...]]] = [
        (type_index, (), 0.0, ()) for type_index in range(type_count)
    ]
    seen = {(type_index, ()) for type_index in range(type_count)}
    solution = None
    max_violation = math.inf
    rounds = 0
    closed = False
    status = "ROUND_LIMIT"
    for rounds in range(1, max_rounds + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Shi menu pricing reached its global deadline.")
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
        lp_options = {
            "dual_feasibility_tolerance": max(1e-10, min(1e-7, tolerance / 10)),
            "primal_feasibility_tolerance": max(
                1e-10, min(1e-7, tolerance / 10)
            ),
        }
        if deadline is not None:
            lp_options["time_limit"] = max(1e-6, deadline - time.monotonic())
        solution = linprog(
            objective,
            A_ub=constraints,
            b_ub=upper_rhs,
            bounds=(0.0, None),
            method="highs",
            options=lp_options,
        )
        if not solution.success:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Shi dual LP reached its global deadline.")
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
            canonical_menu, canonical_shares = _canonical_menu_shares(
                menu, attractions[type_index]
            )
            key = (type_index, canonical_menu)
            if violation > tolerance:
                if key in seen:
                    status = "NUMERICAL_NONCLOSURE_REPEATED_MENU"
                    violations = []
                    break
                violations.append((violation, key, welfare, canonical_shares))
        if status == "NUMERICAL_NONCLOSURE_REPEATED_MENU":
            break
        if not violations:
            closed = max_violation <= tolerance
            status = "OPTIMAL_FLOATING" if closed else "NUMERICAL_NONCLOSURE"
            break
        for _, key, welfare, shares in violations:
            type_index, menu = key
            seen.add(key)
            generated.append((type_index, menu, welfare, shares))
    prices = np.asarray(solution.x[type_count:], dtype=float)
    potentials = np.asarray(solution.x[:type_count], dtype=float)
    residuals = []
    for type_index in range(type_count):
        price_order = tuple(
            sorted(schools, key=lambda school: (prices[school_index[school]], school))
        )
        if cardinality is None or cardinality >= len(eligible[type_index]):
            _, _, _, priced_value = _best_shi_prefix(
                price_order,
                eligible[type_index],
                attractions[type_index],
                prices,
                school_index,
                beta,
            )
        else:
            _, _, _, priced_value = _best_shi_cardinality(
                eligible[type_index],
                attractions[type_index],
                prices,
                school_index,
                beta,
                cardinality,
            )
        residuals.append(max(0.0, priced_value - potentials[type_index]))
    max_violation = max(residuals, default=0.0)
    dual_objective = float(solution.fun)
    repaired_upper_bound = dual_objective + sum(
        segment.mass * residual
        for segment, residual in zip(segments, residuals, strict=True)
    )
    primal = _solve_restricted_shi_primal(
        segments,
        schools,
        school_index,
        capacities,
        generated,
        deadline=deadline,
    )
    primal_objective = -float(primal.fun)
    menu_probabilities, quotas = _extract_shi_primal_witness(
        segments,
        schools,
        generated,
        primal.x,
    )
    probability_residual = max(
        (
            abs(sum(probability for _, probability in menu_probabilities[segment.segment_id]) - 1.0)
            for segment in segments
        ),
        default=0.0,
    )
    capacity_violation = max(
        (
            max(0.0, quotas[school] - float(school_capacities[school]))
            for school in schools
        ),
        default=0.0,
    )
    primal_dual_gap = max(0.0, repaired_upper_bound - primal_objective)
    dual_violation = max_violation
    objective_tolerance = max(1e-8, min(tolerance, 1e-6)) * max(
        1.0, abs(primal_objective)
    )
    closed = (
        closed
        and probability_residual <= max(tolerance, 1e-8)
        and capacity_violation <= max(tolerance, 1e-8)
        and dual_objective + max(tolerance, 1e-8) >= primal_objective
        and primal_dual_gap <= objective_tolerance
    )
    if not closed and status == "OPTIMAL_FLOATING":
        status = "NUMERICAL_NONCLOSURE"

    return ShiMechanismResult(
        upper_bound=repaired_upper_bound,
        primal_objective=primal_objective,
        dual_objective=dual_objective,
        repaired_upper_bound=repaired_upper_bound,
        menu_probabilities=menu_probabilities,
        quotas=quotas,
        school_prices={
            school: float(prices[school_index[school]]) for school in schools
        },
        type_potentials={
            segment.segment_id: float(potentials[index])
            for index, segment in enumerate(segments)
        },
        max_pricing_violation=max_violation,
        closed=closed,
        status=status,
        rounds=rounds,
        constraint_count=len(generated),
        timing_seconds=time.monotonic() - started,
        max_primal_capacity_violation=capacity_violation,
        max_primal_probability_residual=probability_residual,
        max_dual_feasibility_violation=dual_violation,
        primal_dual_gap=primal_dual_gap,
    )


def shi_menu_value(
    attractions: Mapping[int, float],
    menu: Iterable[int],
    beta: float,
) -> tuple[float, tuple[int, ...], tuple[float, ...]]:
    """Return normalized MNL welfare and school shares for one menu."""
    canonical = tuple(sorted(set(int(school) for school in menu)))
    denominator = 1.0 + sum(attractions[school] for school in canonical)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("MNL menu denominator must be positive and finite.")
    return (
        beta * math.log(denominator),
        canonical,
        tuple(attractions[school] / denominator for school in canonical),
    )


def prepare_shi_attractions(
    segment: AnalyticalWelfareSegment,
    beta: float,
) -> dict[int, float]:
    """Validate and exponentiate one segment's outside-normalized utilities."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite.")
    logs = {
        school: (segment.systematic_utilities[school] - segment.outside_utility) / beta
        for school in segment.eligible_schools
    }
    if any(not math.isfinite(value) for value in logs.values()):
        raise ValueError("MNL log attractions must be finite.")
    menu_logs = (0.0, *logs.values())
    if max(menu_logs) - min(menu_logs) > 700.0:
        raise ValueError(
            "Analytical MNL log-attraction range exceeds the supported "
            "double-precision range; use a high-precision evaluator."
        )
    attractions = {school: math.exp(value) for school, value in logs.items()}
    if any(not math.isfinite(value) or value <= 0 for value in attractions.values()):
        raise ValueError("MNL attractions must be finite and strictly positive.")
    return attractions


def shi_dual_potentials(
    segments: Iterable[AnalyticalWelfareSegment],
    school_prices: Mapping[int, float],
    *,
    beta: float,
) -> dict[int, float]:
    """Return full-menu dual potentials for a nonnegative school-price vector.

    The returned potential for each segment is the maximum priced menu value,
    including the empty menu.  Together with ``school_prices`` these values are
    dual feasible for every subset of active segments and schools, which makes
    them suitable for globally valid Benders cuts.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite.")
    prices = {int(school): float(price) for school, price in school_prices.items()}
    if any(not math.isfinite(price) or price < 0 for price in prices.values()):
        raise ValueError("Shi school prices must be finite and nonnegative.")
    schools = tuple(sorted(prices, key=lambda school: (prices[school], school)))
    school_index = {school: index for index, school in enumerate(schools)}
    price_array = np.asarray([prices[school] for school in schools], dtype=float)

    potentials = {}
    for segment in segments:
        missing = set(segment.eligible_schools) - set(prices)
        if missing:
            raise ValueError(
                f"Shi prices omit eligible schools {sorted(missing)} for segment "
                f"{segment.segment_id}."
            )
        attractions = prepare_shi_attractions(segment, beta)
        _, _, _, value = _best_shi_prefix(
            schools,
            frozenset(segment.eligible_schools),
            attractions,
            price_array,
            school_index,
            beta,
        )
        value = max(0.0, float(value))
        guard = 1e-12 * max(1.0, abs(value))
        potentials[int(segment.segment_id)] = math.nextafter(value + guard, math.inf)
    return potentials


def _canonical_menu_shares(
    menu: Iterable[int], attractions: Mapping[int, float]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    canonical = tuple(sorted(menu))
    denominator = 1.0 + sum(attractions[school] for school in canonical)
    return canonical, tuple(
        attractions[school] / denominator for school in canonical
    )


def _solve_restricted_shi_primal(
    segments: tuple[AnalyticalWelfareSegment, ...],
    schools: tuple[int, ...],
    school_index: Mapping[int, int],
    capacities: np.ndarray,
    generated: list[tuple[int, tuple[int, ...], float, tuple[float, ...]]],
    *,
    deadline: float | None,
):
    objective = np.asarray(
        [-segments[type_index].mass * welfare for type_index, _, welfare, _ in generated],
        dtype=float,
    )
    equality = np.zeros((len(segments), len(generated)), dtype=float)
    capacity_rows = np.zeros((len(schools), len(generated)), dtype=float)
    for column, (type_index, menu, _, shares) in enumerate(generated):
        mass = segments[type_index].mass
        equality[type_index, column] = 1.0
        for school, share in zip(menu, shares, strict=True):
            capacity_rows[school_index[school], column] = mass * share
    options = None
    if deadline is not None:
        if time.monotonic() >= deadline:
            raise TimeoutError("Shi primal LP reached its global deadline.")
        options = {"time_limit": max(1e-6, deadline - time.monotonic())}
    primal = linprog(
        objective,
        A_ub=capacity_rows if schools else None,
        b_ub=capacities if schools else None,
        A_eq=equality,
        b_eq=np.ones(len(segments), dtype=float),
        bounds=(0.0, None),
        method="highs",
        options=options,
    )
    if not primal.success:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Shi primal LP reached its global deadline.")
        raise RuntimeError(f"Shi primal LP failed: {primal.message}")
    return primal


def _extract_shi_primal_witness(
    segments: tuple[AnalyticalWelfareSegment, ...],
    schools: tuple[int, ...],
    generated: list[tuple[int, tuple[int, ...], float, tuple[float, ...]]],
    values: np.ndarray,
) -> tuple[
    dict[int, tuple[tuple[tuple[int, ...], float], ...]],
    dict[int, float],
]:
    probabilities: dict[int, list[tuple[tuple[int, ...], float]]] = {
        segment.segment_id: [] for segment in segments
    }
    quotas = {school: 0.0 for school in schools}
    for value, (type_index, menu, _, shares) in zip(values, generated, strict=True):
        probability = float(value)
        if probability <= 1e-12:
            continue
        segment = segments[type_index]
        probabilities[segment.segment_id].append((menu, probability))
        for school, share in zip(menu, shares, strict=True):
            quotas[school] += segment.mass * probability * share
    return (
        {
            segment_id: tuple(sorted(entries, key=lambda item: item[0]))
            for segment_id, entries in probabilities.items()
        },
        quotas,
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

    best_welfare = 0.0
    best_value = 0.0
    best_menu: tuple[int, ...] = ()
    initial_welfare, initial_value = evaluate()
    if initial_value > best_value:
        best_welfare = initial_welfare
        best_value = initial_value
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

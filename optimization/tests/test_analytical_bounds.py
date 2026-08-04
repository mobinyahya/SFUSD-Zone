"""Tests for analytical stable-welfare upper bounds."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.optimize import linprog

from optimization.analytical_bounds import (
    _best_shi_cardinality,
    _best_shi_prefix,
    solve_shi_menu_bound,
)
from optimization.analytical_welfare_oracle import solve_analytical_market
from optimization.problem import AnalyticalWelfareSegment


def _segment(segment_id, utilities):
    return AnalyticalWelfareSegment(
        segment_id=segment_id,
        node=0,
        mass=1.0,
        eligible_schools=tuple(utilities),
        priorities={school: 0.0 for school in utilities},
        systematic_utilities=utilities,
        outside_utility=0.0,
    )


def test_shi_prefix_pricing_matches_exhaustive_assortments():
    schools = (100, 200, 300, 400)
    prices = np.asarray([0.7, 0.1, 1.5, 0.4])
    index = {school: position for position, school in enumerate(schools)}
    attractions = {100: 0.2, 200: 3.0, 300: 1.2, 400: 5.0}
    order = tuple(sorted(schools, key=lambda school: prices[index[school]]))

    menu, _, _, value = _best_shi_prefix(
        order, frozenset(schools), attractions, prices, index, 1.0
    )
    exhaustive = []
    for size in range(len(schools) + 1):
        for candidate in itertools.combinations(schools, size):
            denominator = 1 + sum(attractions[school] for school in candidate)
            candidate_value = math.log(denominator) - sum(
                prices[index[school]] * attractions[school] / denominator
                for school in candidate
            )
            exhaustive.append((candidate_value, candidate))

    assert value == pytest.approx(max(exhaustive)[0])
    assert set(menu) == set(max(exhaustive)[1])


def test_complete_shi_bound_matches_extensive_primal_and_dominates_q20():
    segments = (
        _segment(1, {100: 2.0, 200: 0.0}),
        _segment(2, {100: 0.0, 200: 2.0}),
        _segment(3, {100: 1.0, 200: 1.0}),
    )
    capacities = {100: 1, 200: 1}
    result = solve_shi_menu_bound(segments, capacities, beta=1.0)

    menus = tuple(
        menu
        for size in range(3)
        for menu in itertools.combinations(capacities, size)
    )
    objective = []
    school_rows = {school: [] for school in capacities}
    for segment in segments:
        attractions = {
            school: math.exp(segment.systematic_utilities[school])
            for school in capacities
        }
        for menu in menus:
            denominator = 1 + sum(attractions[school] for school in menu)
            objective.append(-math.log(denominator))
            for school in capacities:
                school_rows[school].append(
                    attractions[school] / denominator if school in menu else 0.0
                )
    equalities = np.zeros((len(segments), len(objective)))
    for index in range(len(segments)):
        start = index * len(menus)
        equalities[index, start : start + len(menus)] = 1.0
    extensive = linprog(
        objective,
        A_ub=np.asarray(list(school_rows.values())),
        b_ub=np.asarray(list(capacities.values()), dtype=float),
        A_eq=equalities,
        b_eq=np.ones(len(segments)),
        bounds=(0.0, None),
        method="highs",
    )
    q20 = solve_analytical_market(
        segments, capacities, beta=1.0, cutoff_grid=20
    )

    assert extensive.success
    assert result.upper_bound == pytest.approx(-extensive.fun, abs=1e-8)
    assert result.upper_bound + 1e-8 >= q20.normalized_welfare
    assert result.max_pricing_violation <= 1e-8


@pytest.mark.parametrize("cardinality", range(5))
def test_cardinality_pricing_matches_exhaustive_assortments(cardinality):
    schools = (100, 200, 300, 400)
    prices = np.asarray([0.7, 0.1, 1.5, 0.4])
    index = {school: position for position, school in enumerate(schools)}
    attractions = {100: 0.2, 200: 3.0, 300: 1.2, 400: 5.0}

    menu, _, _, value = _best_shi_cardinality(
        frozenset(schools),
        attractions,
        prices,
        index,
        1.0,
        cardinality,
    )
    exhaustive = []
    for size in range(min(cardinality, len(schools)) + 1):
        for candidate in itertools.combinations(schools, size):
            denominator = 1 + sum(attractions[school] for school in candidate)
            candidate_value = math.log(denominator) - sum(
                prices[index[school]] * attractions[school] / denominator
                for school in candidate
            )
            exhaustive.append((candidate_value, candidate))

    assert value == pytest.approx(max(exhaustive)[0])
    assert set(menu) == set(max(exhaustive)[1])


def test_cardinality_shi_bound_matches_extensive_primal():
    segments = (
        _segment(1, {100: 2.0, 200: 0.0, 300: 1.0}),
        _segment(2, {100: 0.0, 200: 2.0, 300: 1.0}),
    )
    capacities = {100: 1, 200: 1, 300: 1}

    result = solve_shi_menu_bound(
        segments, capacities, beta=1.0, cardinality=1
    )
    unrestricted = solve_shi_menu_bound(segments, capacities, beta=1.0)

    assert result.upper_bound <= unrestricted.upper_bound + 1e-8
    assert result.max_pricing_violation <= 1e-8

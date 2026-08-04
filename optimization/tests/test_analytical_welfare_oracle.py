"""Tests for analytical expected-MNL stable-welfare recourse."""

from __future__ import annotations

import itertools
import math

import pytest

from optimization.analytical_welfare_oracle import (
    evaluate_zoned_analytical_welfare,
    integrate_analytical_market,
    solve_analytical_market,
)
from optimization.problem import AnalyticalWelfareMarket, AnalyticalWelfareSegment


def _segment(
    segment_id,
    *,
    node=0,
    mass=1.0,
    utilities=None,
    priorities=None,
    outside=0.0,
):
    utilities = utilities or {}
    priorities = priorities or {school: 0.0 for school in utilities}
    return AnalyticalWelfareSegment(
        segment_id=segment_id,
        node=node,
        mass=mass,
        eligible_schools=tuple(utilities),
        priorities=priorities,
        systematic_utilities=utilities,
        outside_utility=outside,
    )


def test_equal_school_and_outside_integrates_logit_choice():
    segment = _segment(1, utilities={100: 0.0})

    result = solve_analytical_market((segment,), {100: 1}, beta=1.0)

    assert result.cutoffs == {100: 0.0}
    assert result.assignment_measures[1][100] == pytest.approx(0.5)
    assert result.outside_measures[1] == pytest.approx(0.5)
    assert result.normalized_welfare == pytest.approx(math.log(2.0))
    assert result.stable
    assert result.least_cutoff_numerically_verified


def test_low_utility_school_remains_available_with_positive_share():
    segment = _segment(1, utilities={100: -10.0})

    result = solve_analytical_market((segment,), {100: 1}, beta=1.0)

    assert result.assignment_measures[1][100] > 0.0
    assert result.normalized_welfare == pytest.approx(math.log1p(math.exp(-10.0)))


def test_equal_school_utilities_receive_equal_menu_shares():
    segment = _segment(1, utilities={100: 2.0, 200: 2.0})

    result = integrate_analytical_market(
        (segment,), {100: 1, 200: 1}, {100: 0.0, 200: 0.0}, beta=1.0
    )

    assert result.assignment_measures[1][100] == pytest.approx(
        result.assignment_measures[1][200]
    )
    assert (
        result.outside_measures[1] + sum(result.assignment_measures[1].values())
        == pytest.approx(1.0)
    )


def test_continuum_cutoff_clears_expected_mnl_demand():
    segments = tuple(_segment(index, utilities={100: 0.0}) for index in range(2))

    result = solve_analytical_market(segments, {100: 0}, beta=1.0)

    assert result.cutoffs[100] == pytest.approx(1.0, abs=2e-9)
    assert result.demands[100] == pytest.approx(0.0, abs=2e-9)
    assert result.stable


def test_grid_cutoff_is_minimal_and_uses_analytical_mnl_cells():
    segments = tuple(_segment(index, utilities={100: 0.0}) for index in range(4))

    result = solve_analytical_market(
        segments, {100: 1}, beta=1.0, cutoff_grid=4
    )

    assert result.cutoff_indices == {100: 2}
    assert result.cutoffs == {100: 0.5}
    assert result.demands[100] == pytest.approx(1.0)
    assert result.normalized_welfare == pytest.approx(2 * math.log(2.0))
    assert result.grid_minimal
    assert result.complementarity_valid is None
    assert result.grid_underfill == {100: pytest.approx(0.0)}
    assert result.grid_lowered_demand[100] > 1.0


def test_grid_oracle_matches_brute_force_componentwise_least_cutoff():
    segments = (
        _segment(1, utilities={100: 1.0, 200: 0.0}),
        _segment(2, utilities={100: 0.0, 200: 1.0}),
        _segment(3, utilities={100: 0.5, 200: 0.5}),
    )
    capacities = {100: 1, 200: 1}
    result = solve_analytical_market(
        segments, capacities, beta=1.0, cutoff_grid=4
    )
    feasible = []
    for first, second in itertools.product(range(5), repeat=2):
        cutoffs = {100: first / 4, 200: second / 4}
        integrated = integrate_analytical_market(
            segments, capacities, cutoffs, beta=1.0
        )
        if all(
            integrated.demands[school] <= capacities[school] + 1e-10
            for school in capacities
        ):
            feasible.append(cutoffs)

    assert feasible
    assert all(
        result.cutoffs[school] <= cutoffs[school]
        for cutoffs in feasible
        for school in capacities
    )


def test_zoned_markets_are_isolated_and_additive():
    market = AnalyticalWelfareMarket(
        segments=(
            _segment(1, node=0, utilities={100: 1.0, 200: 1.0}),
            _segment(2, node=2, utilities={100: 1.0, 200: 1.0}),
        ),
        school_nodes={100: 1, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        beta=1.0,
        lottery_scale=20,
    )

    result = evaluate_zoned_analytical_welfare(
        market, {0: 0, 1: 0, 2: 1, 3: 1}, num_zones=2, cutoff_grid=20
    )

    expected = 2 * math.log1p(math.exp(1.0))
    assert result.normalized_welfare == pytest.approx(expected)
    assert result.normalized_welfare == pytest.approx(
        sum(zone.normalized_welfare for zone in result.zones.values())
    )
    assert result.assignment_measures[1][100] > 0
    assert 200 not in result.assignment_measures[1]
    assert result.assignment_measures[2][200] > 0
    assert result.stable


def test_simultaneous_threshold_events_enter_together():
    segment = _segment(
        1,
        utilities={100: 0.0, 200: 0.0},
        priorities={100: 0.0, 200: 0.0},
    )

    result = integrate_analytical_market(
        (segment,), {100: 1, 200: 1}, {100: 0.5, 200: 0.5}, beta=1.0
    )

    assert result.assignment_measures[1][100] == pytest.approx(1 / 6)
    assert result.assignment_measures[1][200] == pytest.approx(1 / 6)
    assert result.outside_measures[1] == pytest.approx(2 / 3)
    assert result.normalized_welfare == pytest.approx(0.5 * math.log(3.0))


def test_nonzero_outside_and_beta_use_normalized_inclusive_value():
    segment = _segment(1, utilities={100: 4.0}, outside=2.0)

    result = solve_analytical_market((segment,), {100: 1}, beta=2.0)

    assert result.assignment_measures[1][100] == pytest.approx(
        math.exp(1.0) / (1.0 + math.exp(1.0))
    )
    assert result.normalized_welfare == pytest.approx(2.0 * math.log1p(math.e))


def test_nested_grid_values_are_monotone_for_fixed_zoning():
    segments = tuple(
        _segment(index, utilities={100: 0.0}) for index in range(3)
    )

    coarse = solve_analytical_market(
        segments, {100: 1}, beta=1.0, cutoff_grid=2
    )
    fine = solve_analytical_market(
        segments, {100: 1}, beta=1.0, cutoff_grid=4
    )
    continuum = solve_analytical_market(segments, {100: 1}, beta=1.0)

    assert coarse.normalized_welfare <= fine.normalized_welfare
    assert fine.normalized_welfare <= continuum.normalized_welfare + 1e-9


def test_extreme_log_attraction_range_fails_instead_of_underflowing():
    segment = _segment(1, utilities={100: 1000.0, 200: -1000.0})

    with pytest.raises(ValueError, match="high-precision evaluator"):
        solve_analytical_market(
            (segment,), {100: 1, 200: 1}, beta=1.0
        )


def test_aggregate_overflow_fails_instead_of_reporting_infinite_welfare():
    segments = tuple(
        _segment(index, mass=8e307, utilities={100: 0.0}) for index in range(4)
    )

    with pytest.raises(FloatingPointError, match="Non-finite aggregate"):
        integrate_analytical_market(
            segments, {100: 1}, {100: 0.0}, beta=1.0
        )

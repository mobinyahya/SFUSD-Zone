"""Tests for the common-cutoff/common-STB conditional-loss bound."""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from optimization.analytical_priority_bound import (
    FLOATING_DIAGNOSTIC_SCOPE,
    _loss_level_supports,
    best_shi_cardinality_forced_inclusion,
    solve_common_cutoff_stb_bound,
)
from optimization.analytical_welfare_oracle import (
    evaluate_zoned_analytical_welfare,
)
from optimization.branch_price.patterns import ZonePattern, ZonePatternValidator
from optimization.problem import AnalyticalWelfareMarket, AnalyticalWelfareSegment
from optimization.tests.synthetic import make_grid_problem


def test_forced_inclusion_line_sweep_matches_every_tiny_menu():
    rng = random.Random(20260803)
    for school_count in range(1, 7):
        schools = tuple(range(100, 100 + school_count))
        school_index = {school: index for index, school in enumerate(schools)}
        for _ in range(12):
            attractions = {school: 10 ** rng.uniform(-1.5, 1.5) for school in schools}
            prices = np.asarray([rng.uniform(0.0, 6.0) for _ in schools], dtype=float)
            for cardinality in range(1, school_count + 1):
                for mandatory in schools:
                    menu, welfare, shares, actual = (
                        best_shi_cardinality_forced_inclusion(
                            frozenset(schools),
                            attractions,
                            prices,
                            school_index,
                            1.3,
                            cardinality,
                            mandatory,
                        )
                    )
                    expected = _exhaustive_forced_value(
                        schools,
                        attractions,
                        prices,
                        school_index,
                        beta=1.3,
                        cardinality=cardinality,
                        mandatory=mandatory,
                    )

                    assert actual == pytest.approx(expected, abs=2e-12)
                    assert mandatory in menu
                    assert len(menu) <= cardinality
                    denominator = 1.0 + sum(attractions[school] for school in menu)
                    assert welfare == pytest.approx(1.3 * math.log(denominator))
                    assert shares == pytest.approx(
                        tuple(attractions[school] / denominator for school in menu)
                    )


def test_forced_inclusion_line_sweep_handles_all_tied_tiny_coefficients():
    for school_count in range(1, 4):
        schools = tuple(range(school_count))
        school_index = {school: school for school in schools}
        for attraction_values in itertools.product((0.5, 1.0), repeat=school_count):
            attractions = dict(zip(schools, attraction_values, strict=True))
            for price_values in itertools.product((0.0, 1.0, 2.0), repeat=school_count):
                prices = np.asarray(price_values)
                for cardinality in range(1, school_count + 1):
                    for mandatory in schools:
                        actual = best_shi_cardinality_forced_inclusion(
                            frozenset(schools),
                            attractions,
                            prices,
                            school_index,
                            1.0,
                            cardinality,
                            mandatory,
                        )[-1]
                        expected = _exhaustive_forced_value(
                            schools,
                            attractions,
                            prices,
                            school_index,
                            beta=1.0,
                            cardinality=cardinality,
                            mandatory=mandatory,
                        )
                        assert actual == pytest.approx(expected, abs=2e-12)


def test_loss_supports_use_no_epsilon_enlargement():
    supports = _loss_level_supports({100: 1e-15, 200: 2e-15, 300: 0.0})

    assert supports == (
        (1e-15, (100, 200)),
        (2e-15, (200,)),
    )


def test_continuous_and_mip_bounds_dominate_every_tiny_exact_zoning():
    problem, market = _tiny_problem_and_market()
    prices = {100: 0.2, 200: 3.0}
    exact_values = _all_feasible_exact_values(problem, market)
    assert exact_values

    continuous = solve_common_cutoff_stb_bound(
        problem,
        market,
        prices,
        cardinality=1,
        inclusion_shortlist=1,
        relax_integrality=True,
        time_limit=10.0,
        workers=1,
    )
    geographic = solve_common_cutoff_stb_bound(
        problem,
        market,
        prices,
        cardinality=1,
        inclusion_shortlist=1,
        relax_integrality=True,
        enforce_geography=True,
        time_limit=10.0,
        workers=1,
    )
    integer = solve_common_cutoff_stb_bound(
        problem,
        market,
        prices,
        cardinality=1,
        inclusion_shortlist=1,
        relax_integrality=False,
        time_limit=10.0,
        workers=1,
    )

    exact_maximum = max(exact_values)
    assert continuous.status == "OPTIMAL_FLOATING_DIAGNOSTIC"
    assert geographic.status == "OPTIMAL_FLOATING_DIAGNOSTIC"
    assert integer.status == "OPTIMAL_FLOATING_DIAGNOSTIC"
    assert continuous.numerical_scope == FLOATING_DIAGNOSTIC_SCOPE
    assert continuous.continuous_lp_bound is not None
    assert continuous.integrality_strengthened_mip_obj_bound is None
    assert integer.continuous_lp_bound is None
    assert integer.integrality_strengthened_mip_obj_bound is not None
    assert continuous.continuous_lp_bound + 1e-8 >= exact_maximum
    assert geographic.continuous_lp_bound + 1e-8 >= exact_maximum
    assert integer.integrality_strengthened_mip_obj_bound + 1e-8 >= exact_maximum
    assert geographic.continuous_lp_bound <= continuous.continuous_lp_bound + 1e-8
    assert (
        continuous.continuous_lp_bound
        <= continuous.cardinality_shi_bound_at_prices + 1e-8
    )
    assert continuous.dimensions.omission_cap_constraints > 0
    assert continuous.dimensions.inclusion_cap_constraints > 0
    assert continuous.dimensions.base_cap_constraints == len(market.segments)
    assert continuous.dimensions.graph_only_school_nodes == 0
    assert continuous.dimensions.graph_only_schools == 0
    assert geographic.dimensions.labeled_nodes == problem.A
    assert geographic.dimensions.node_label_variables == problem.A * problem.Z
    assert geographic.dimensions.boundary_variables == problem.G.number_of_edges()


def test_multiple_graph_only_schools_share_their_node_label():
    problem, market = _tiny_problem_and_market()
    problem.G.nodes[1]["num_schools"] = 2
    problem.G.nodes[1]["school_ids"] = [300, 301]

    result = solve_common_cutoff_stb_bound(
        problem,
        market,
        {100: 0.2, 200: 3.0},
        cardinality=1,
        inclusion_shortlist=1,
        enforce_geography=True,
        time_limit=10.0,
        workers=1,
    )

    assert result.status == "OPTIMAL_FLOATING_DIAGNOSTIC"
    assert result.dimensions.graph_only_school_nodes == 1
    assert result.dimensions.graph_only_schools == 2
    assert result.dimensions.graph_only_school_label_variables == problem.Z


@pytest.mark.parametrize(
    "prices, message",
    [
        ({100: -0.1, 200: 0.0}, "non-negative"),
        ({100: 0.0}, "exactly"),
        ({100: math.inf, 200: 0.0}, "finite"),
    ],
)
def test_school_prices_are_validated(prices, message):
    problem, market = _tiny_problem_and_market()

    with pytest.raises(ValueError, match=message):
        solve_common_cutoff_stb_bound(
            problem,
            market,
            prices,
            cardinality=1,
            time_limit=10.0,
            workers=1,
        )


def _tiny_problem_and_market():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=1.0,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=1.0,
    )
    segments = (
        AnalyticalWelfareSegment(
            segment_id=1,
            node=1,
            mass=1.0,
            eligible_schools=(100, 200),
            priorities={100: 0.0, 200: 1.0},
            systematic_utilities={100: 2.0, 200: -1.0},
            outside_utility=0.0,
        ),
        AnalyticalWelfareSegment(
            segment_id=2,
            node=2,
            mass=1.0,
            eligible_schools=(100, 200),
            priorities={100: 1.0, 200: 0.0},
            systematic_utilities={100: 1.0, 200: 2.0},
            outside_utility=0.0,
        ),
    )
    market = AnalyticalWelfareMarket(
        segments=segments,
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 2, 200: 2},
        zone_restricted_schools=frozenset({100, 200}),
        beta=1.0,
        lottery_scale=2,
    )
    problem.analytical_welfare_market = market
    return problem, market


def _all_feasible_exact_values(problem, market):
    validator = ZonePatternValidator(problem)
    values = []
    for first, second in itertools.product(range(problem.Z), repeat=2):
        assignment = {0: 0, 1: first, 2: second, 3: 1}
        try:
            for zone in range(problem.Z):
                nodes = frozenset(
                    node for node, label in assignment.items() if label == zone
                )
                validator(
                    ZonePattern.from_graph(
                        label=zone,
                        nodes=nodes,
                        raw_welfare=0,
                        graph=problem.G,
                    )
                )
        except ValueError:
            continue
        result = evaluate_zoned_analytical_welfare(
            market,
            assignment,
            num_zones=problem.Z,
            cutoff_grid=market.lottery_scale,
        )
        assert result.stable
        values.append(result.normalized_welfare)
    return values


def _exhaustive_forced_value(
    schools,
    attractions,
    prices,
    school_index,
    *,
    beta,
    cardinality,
    mandatory,
):
    values = []
    for size in range(1, cardinality + 1):
        for menu in itertools.combinations(schools, size):
            if mandatory not in menu:
                continue
            denominator = 1.0 + sum(attractions[school] for school in menu)
            values.append(
                beta * math.log(denominator)
                - sum(
                    prices[school_index[school]] * attractions[school]
                    for school in menu
                )
                / denominator
            )
    return max(values)

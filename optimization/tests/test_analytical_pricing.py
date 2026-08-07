"""Data-free proof checks for nested analytical zone pricing."""

from __future__ import annotations

import itertools
import math

import pytest

from optimization.analytical_bounds import solve_shi_menu_bound
from optimization.branch_price.analytical_master import AnalyticalMasterDuals
from optimization.branch_price.analytical_patterns import AnalyticalPatternValuator
from optimization.branch_price.analytical_pricing import solve_analytical_pricing
from optimization.problem import (
    AnalyticalWelfareMarket,
    AnalyticalWelfareSegment,
)
from optimization.tests.synthetic import make_grid_problem


def _problem_market():
    problem = make_grid_problem(
        2,
        2,
        overage=-1,
        shortage=-1,
        boundary_prop=-1,
    )
    segments = tuple(
        AnalyticalWelfareSegment(
            segment_id=node,
            node=node,
            mass=0.5 + node,
            eligible_schools=(100, 200),
            priorities={100: 0.0, 200: 0.0},
            systematic_utilities={
                100: 2.0 if node < 2 else 0.0,
                200: 2.0 if node >= 2 else 0.0,
            },
            outside_utility=-0.25,
        )
        for node in problem.nodes
    )
    market = AnalyticalWelfareMarket(
        segments=segments,
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 3, 200: 5},
        zone_restricted_schools=frozenset({100, 200}),
        beta=1.0,
        lottery_scale=20,
    )
    problem.analytical_welfare_market = market
    return problem, market


def test_shi_witness_handles_mass_outside_empty_and_zero_capacity():
    outside_only = AnalyticalWelfareSegment(1, 0, 2.5, (), {}, {}, 4.0)
    eligible = AnalyticalWelfareSegment(
        2,
        0,
        3.0,
        (100,),
        {100: 0.0},
        {100: 8.0},
        1.0,
    )
    result = solve_shi_menu_bound(
        (outside_only, eligible),
        {100: 0},
        beta=2.0,
    )

    assert result.closed
    assert result.primal_objective == pytest.approx(0.0)
    assert result.menu_probabilities[1] == (((), 1.0),)
    assert result.menu_probabilities[2] == (((), 1.0),)
    assert result.quotas[100] == pytest.approx(0.0)


def test_positive_menu_tolerance_uses_repaired_upper_bound():
    segment = AnalyticalWelfareSegment(
        1,
        0,
        2.0,
        (100,),
        {100: 0.0},
        {100: 0.0},
        0.0,
    )
    result = solve_shi_menu_bound(
        (segment,),
        {100: 10},
        beta=1.0,
        tolerance=1.0,
    )

    assert result.primal_objective == pytest.approx(0.0)
    assert result.dual_objective == pytest.approx(0.0)
    assert result.repaired_upper_bound == pytest.approx(2 * math.log(2))
    assert result.repaired_upper_bound >= 2 * math.log(2)
    assert not result.closed


def test_analytical_pricing_matches_exhaustive_legal_zones():
    problem, _ = _problem_market()
    valuator = AnalyticalPatternValuator(problem)
    duals = AnalyticalMasterDuals(
        convexity={0: 0.4, 1: 0.0},
        coverage={1: 0.2, 2: -0.1},
        boundary=0.15,
    )
    exhaustive = []
    nodes = tuple(problem.nodes)
    for size in range(1, len(nodes) + 1):
        for selected in itertools.combinations(nodes, size):
            try:
                pattern = valuator.value(0, frozenset(selected))
            except ValueError:
                continue
            exhaustive.append((duals.reduced_cost(pattern), pattern))
    expected_value, expected = max(exhaustive, key=lambda item: item[0])

    result = solve_analytical_pricing(
        problem,
        0,
        duals,
        valuator=valuator,
        time_limit=5,
        node_limit=100,
        workers=1,
    )

    assert result.closed
    assert result.candidate is not None
    assert result.candidate_reduced_cost == pytest.approx(expected_value, abs=1e-7)
    assert result.candidate.nodes == expected.nodes
    assert result.reduced_cost_upper_bound + 1e-7 >= expected_value


def test_pricing_timeout_retains_model_free_upper_bound():
    problem, _ = _problem_market()
    valuator = AnalyticalPatternValuator(problem)
    duals = AnalyticalMasterDuals(
        convexity={0: 0.0, 1: 0.0},
        coverage={1: 0.0, 2: 0.0},
        boundary=0.0,
    )
    exhaustive = max(
        duals.reduced_cost(valuator.value(0, frozenset(selected)))
        for size in range(1, len(problem.nodes) + 1)
        for selected in itertools.combinations(problem.nodes, size)
        if _legal(valuator, 0, selected)
    )

    result = solve_analytical_pricing(
        problem,
        0,
        duals,
        valuator=valuator,
        time_limit=0,
        node_limit=0,
    )

    assert not result.closed
    assert result.reduced_cost_upper_bound + 1e-7 >= exhaustive
    assert result.candidate is None


def test_all_bundle_pricing_moves_a_noncentroid_school():
    problem, market = _problem_market()
    segments = tuple(
        AnalyticalWelfareSegment(
            segment.segment_id,
            segment.node,
            segment.mass,
            (100, 200, 300),
            {100: 0.0, 200: 0.0, 300: 0.0},
            {100: 0.0, 200: 0.0, 300: 8.0},
            segment.outside_utility,
        )
        for segment in market.segments
    )
    problem.analytical_welfare_market = AnalyticalWelfareMarket(
        segments,
        {100: 0, 200: 3, 300: 1},
        {100: 3, 200: 5, 300: 20},
        frozenset({100, 200, 300}),
        1.0,
        20,
    )
    initial_assignment = {0: 0, 1: 1, 2: 0, 3: 1}
    assert initial_assignment[problem.analytical_welfare_market.school_nodes[300]] == 1
    valuator = AnalyticalPatternValuator(problem)
    duals = AnalyticalMasterDuals(
        convexity={0: 0.0, 1: 0.0},
        coverage={1: 0.0, 2: 0.0},
        boundary=0.0,
    )

    result = solve_analytical_pricing(
        problem,
        0,
        duals,
        valuator=valuator,
        time_limit=5,
        node_limit=100,
    )

    assert result.closed
    assert result.candidate is not None
    assert 1 in result.candidate.nodes
    assert 300 in result.candidate.school_ids


def test_integral_positive_menu_residual_remains_in_pricing_bound():
    problem, _ = _problem_market()
    valuator = AnalyticalPatternValuator(problem)
    duals = AnalyticalMasterDuals(
        convexity={0: 0.0, 1: 0.0},
        coverage={1: 0.0, 2: 0.0},
        boundary=0.0,
    )
    exhaustive = max(
        duals.reduced_cost(valuator.value(0, frozenset(selected)))
        for size in range(1, len(problem.nodes) + 1)
        for selected in itertools.combinations(problem.nodes, size)
        if _legal(valuator, 0, selected)
    )

    result = solve_analytical_pricing(
        problem,
        0,
        duals,
        valuator=valuator,
        menu_tolerance=10.0,
        reduced_cost_tolerance=1e-7,
        time_limit=5,
        node_limit=100,
    )

    assert result.menu_residual_bound > 0
    assert not result.closed
    assert result.reduced_cost_upper_bound + 1e-7 >= exhaustive


def _legal(valuator, label, nodes):
    try:
        valuator.validator.validate_membership(
            label=label,
            nodes=frozenset(nodes),
            perimeter=sum(
                (left in nodes) != (right in nodes)
                for left, right in valuator.problem.G.edges
            ),
        )
    except ValueError:
        return False
    return True

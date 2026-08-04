"""Tests for the compact fixed analytical-state zoning master."""

from __future__ import annotations

import pytest

from optimization.analytical_fixed_state_master import (
    optimize_fixed_school_bundles,
    solve_fixed_analytical_states,
)
from optimization.problem import AnalyticalWelfareMarket, AnalyticalWelfareSegment
from optimization.tests.synthetic import make_grid_problem


def test_fixed_state_master_returns_feasible_q20_zoning():
    pytest.importorskip("gurobipy")
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
    market = AnalyticalWelfareMarket(
        segments=(
            AnalyticalWelfareSegment(
                1,
                1,
                1.0,
                (100, 200),
                {100: 0.0, 200: 0.0},
                {100: 2.0, 200: 0.0},
            ),
            AnalyticalWelfareSegment(
                2,
                2,
                1.0,
                (100, 200),
                {100: 0.0, 200: 0.0},
                {100: 0.0, 200: 2.0},
            ),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        beta=1.0,
        lottery_scale=20,
    )
    initial = {0: 0, 1: 0, 2: 1, 3: 1}

    result = solve_fixed_analytical_states(
        problem, market, initial, time_limit=10
    )

    assert result.status == "OPTIMAL_FLOATING_FIXED_STATE"
    assert result.q20_result.stable
    assert set(result.assignment) == set(problem.nodes)

    iterative = optimize_fixed_school_bundles(
        problem, market, initial, time_limit=10, max_iterations=5
    )
    assert iterative.q20_result.normalized_welfare + 1e-8 >= (
        result.q20_result.normalized_welfare
    )

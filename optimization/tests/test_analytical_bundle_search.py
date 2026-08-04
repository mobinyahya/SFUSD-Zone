"""Tests for analytical graph-school exchange screening."""

from __future__ import annotations

import pytest

from optimization.analytical_bundle_search import search_graph_school_swaps
from optimization.problem import AnalyticalWelfareMarket, AnalyticalWelfareSegment
from optimization.tests.synthetic import make_grid_problem


def test_bundle_search_preserves_or_improves_verified_q20():
    pytest.importorskip("gurobipy")
    problem = make_grid_problem(
        2,
        3,
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
                4,
                1.0,
                (100, 200),
                {100: 0.0, 200: 0.0},
                {100: 0.0, 200: 2.0},
            ),
        ),
        school_nodes={100: 0, 200: 5},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        beta=1.0,
        lottery_scale=20,
    )
    initial = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    # Both graph schools are centroids, so the screened neighborhood is empty.
    result = search_graph_school_swaps(
        problem, market, initial, exact_candidates=1, completion_candidates=1
    )

    assert result.assignment == initial
    assert result.considered_swaps == 0

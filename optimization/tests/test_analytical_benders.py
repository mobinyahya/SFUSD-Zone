"""Tests for direct analytical Shi logic-based Benders optimization."""

from __future__ import annotations

from itertools import product

import pytest

from optimization.analytical_benders import (
    make_shi_price_cut,
    solve_zoned_shi_benders,
)
from optimization.branch_price.analytical_patterns import AnalyticalPatternValuator
from optimization.column_generation_seeds import validate_complete_seed
from optimization.config import OptimizationConfig
from optimization.problem import AnalyticalWelfareMarket, AnalyticalWelfareSegment
from optimization.strategies import get_strategy
from optimization.tests.synthetic import make_grid_problem


def _benders_fixture():
    problem = make_grid_problem(
        2,
        2,
        overage=-1,
        shortage=-1,
        boundary_prop=-1,
    )
    problem.G.nodes[1]["school_ids"] = [300]
    problem.G.nodes[1]["num_schools"] = 1
    problem.G.graph["school_data"][300] = {}
    utilities = {
        0: {100: 2.5, 200: 0.0, 300: 0.5},
        1: {100: 0.5, 200: 0.0, 300: 3.0},
        2: {100: 1.5, 200: 2.0, 300: 0.2},
        3: {100: 0.0, 200: 2.5, 300: 0.5},
    }
    segments = tuple(
        AnalyticalWelfareSegment(
            segment_id=node,
            node=node,
            mass=1.0,
            eligible_schools=(100, 200, 300),
            priorities={100: 0.0, 200: 0.0, 300: 0.0},
            systematic_utilities=utilities[node],
            outside_utility=0.0,
        )
        for node in problem.nodes
    )
    market = AnalyticalWelfareMarket(
        segments=segments,
        school_nodes={100: 0, 200: 3, 300: 1},
        school_capacities={100: 2, 200: 2, 300: 1},
        zone_restricted_schools=frozenset({100, 200, 300}),
        beta=1.0,
        lottery_scale=20,
    )
    problem.analytical_welfare_market = market
    valuator = AnalyticalPatternValuator(problem)
    assignments = []
    pattern_maps = []
    objectives = []
    for zone_1, zone_2 in product(range(2), repeat=2):
        candidate = {0: 0, 1: zone_1, 2: zone_2, 3: 1}
        try:
            candidate = validate_complete_seed(
                problem,
                candidate,
                validator=valuator.validator,
            )
        except ValueError:
            continue
        patterns = tuple(
            valuator.value(
                label,
                frozenset(
                    node for node, assigned in candidate.items() if assigned == label
                ),
            )
            for label in range(problem.Z)
        )
        assignments.append(candidate)
        pattern_maps.append(patterns)
        objectives.append(sum(pattern.shi_welfare for pattern in patterns))
    return problem, valuator, assignments, pattern_maps, objectives


def test_price_cut_is_valid_for_every_tiny_zone():
    problem, _, _, pattern_maps, _ = _benders_fixture()
    source = pattern_maps[0][0]
    cut = make_shi_price_cut(problem, source.mechanism.school_prices)

    assert cut.rhs(source.nodes) == pytest.approx(
        source.mechanism.repaired_upper_bound,
        abs=1e-7,
    )

    for patterns in pattern_maps:
        for pattern in patterns:
            assert pattern.shi_welfare <= cut.rhs(pattern.nodes) + 1e-7


def test_benders_matches_exhaustive_tiny_zoning():
    problem, valuator, assignments, pattern_maps, objectives = _benders_fixture()
    seed_assignment = assignments[0]
    seed_patterns = pattern_maps[0]

    result = solve_zoned_shi_benders(
        problem,
        seed_patterns,
        seed_assignment,
        valuator=valuator,
        wall_time_limit=30,
        max_rounds=20,
        master_time_limit=5,
        workers=1,
    )

    assert result.closed
    assert result.status == "OPTIMAL"
    assert result.incumbent_objective == pytest.approx(max(objectives), abs=1e-7)
    assert result.upper_bound == pytest.approx(max(objectives), abs=1e-6)
    assert result.assignment in [
        assignment
        for assignment, objective in zip(assignments, objectives, strict=True)
        if objective == pytest.approx(max(objectives), abs=1e-7)
    ]
    assert result.point_cuts_added >= problem.Z


def test_zero_benders_rounds_preserve_seed_and_finite_bound():
    problem, valuator, assignments, pattern_maps, _ = _benders_fixture()

    result = solve_zoned_shi_benders(
        problem,
        pattern_maps[0],
        assignments[0],
        valuator=valuator,
        wall_time_limit=30,
        max_rounds=0,
    )

    assert not result.closed
    assert result.assignment == assignments[0]
    assert result.upper_bound >= result.incumbent_objective
    assert result.termination_reason == "round_limit"


def test_zoned_benders_config_and_registration():
    config = OptimizationConfig(
        strategy="zoned_benders",
        solver="cp_bool",
        levels=["Block_2"],
        years=[23],
        population_type="All",
        remove_city_wide=True,
    )

    strategy = config.make_strategy()

    assert strategy.name == "zoned_benders"
    assert get_strategy("zoned_benders").name == "zoned_benders"
    assert strategy.options["zoned_benders_wall_time_limit"] == 2700
    assert strategy.options["zoned_recom_seed_runs"] == 0
    assert strategy.options["zoned_benders_local_move_rounds"] == 0

    with pytest.raises(ValueError, match="positive and finite"):
        OptimizationConfig(
            strategy="zoned_benders",
            solver="cp_bool",
            levels=["Block_2"],
            years=[23],
            population_type="All",
            remove_city_wide=True,
            zoned_benders_master_time_limit=True,
        )

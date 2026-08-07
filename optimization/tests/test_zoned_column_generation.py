"""Outer analytical column-generation, seed, and persistence tests."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import product

import pytest

from optimization.branch_price.analytical_master import (
    RestrictedAnalyticalPatternMaster,
)
from optimization.branch_price.analytical_patterns import AnalyticalPatternValuator
from optimization.branch_price.analytical_root import solve_analytical_pattern_root
from optimization.column_generation_seeds import (
    load_seed_assignment,
    normalize_seed_labels,
    validate_complete_seed,
)
from optimization.config import OptimizationConfig
from optimization.problem import (
    AnalyticalWelfareMarket,
    AnalyticalWelfareSegment,
)
from optimization.solution import JsonArtifact, ZoneSolution
from optimization.solvers.recom import _ReComContext
from optimization.strategies import get_strategy
from optimization.tests.synthetic import make_grid_problem


def _root_fixture():
    problem = make_grid_problem(
        2,
        2,
        overage=-1,
        shortage=-1,
        boundary_prop=-1,
    )
    segments = tuple(
        AnalyticalWelfareSegment(
            node,
            node,
            1.0,
            (100, 200),
            {100: 0.0, 200: 0.0},
            {
                100: 2.0 if node < 2 else 0.0,
                200: 2.0 if node >= 2 else 0.0,
            },
            0.0,
        )
        for node in problem.nodes
    )
    problem.analytical_welfare_market = AnalyticalWelfareMarket(
        segments,
        {100: 0, 200: 3},
        {100: 2, 200: 2},
        frozenset({100, 200}),
        1.0,
        20,
    )
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    valuator = AnalyticalPatternValuator(problem)
    patterns = tuple(
        valuator.value(
            label,
            frozenset(node for node, zone in assignment.items() if zone == label),
        )
        for label in range(problem.Z)
    )
    return problem, assignment, valuator, patterns


def test_outer_root_closes_after_adding_all_bundle_columns():
    problem, assignment, valuator, patterns = _root_fixture()
    result = solve_analytical_pattern_root(
        problem,
        patterns,
        assignment,
        valuator=valuator,
        wall_time_limit=20,
        max_rounds=10,
        pricing_time_limit=5,
        pricing_node_limit=100,
        mip_time_limit=5,
        workers=1,
    )

    assert result.root_lp_closed
    assert result.rounds >= 2
    assert result.root_lp_upper_bound == pytest.approx(
        result.root_lp_objective, abs=1e-7
    )
    assert result.assignment == assignment
    assert result.incumbent_upper_bound_gap <= 1e-7


def test_restricted_mip_zero_time_returns_validated_seed():
    problem, assignment, valuator, patterns = _root_fixture()
    master = RestrictedAnalyticalPatternMaster(
        problem.G,
        problem.centroids,
        patterns,
        max_cut_edges=None,
        pattern_validator=valuator.validator,
    )
    result = master.solve_mip(
        time_limit=0,
        workers=1,
        random_seed=0,
        seed_assignment=assignment,
    )

    assert result.assignment == assignment
    assert result.objective == pytest.approx(sum(p.shi_welfare for p in patterns))


def test_zero_wall_time_preserves_seed_and_valid_upper_bound():
    problem, assignment, valuator, patterns = _root_fixture()

    result = solve_analytical_pattern_root(
        problem,
        patterns,
        assignment,
        valuator=valuator,
        wall_time_limit=0,
        max_rounds=10,
    )

    assert result.assignment == assignment
    assert not result.root_lp_closed
    assert result.root_lp_upper_bound + 1e-8 >= result.restricted_mip_objective
    assert result.seed_fallback_used
    legal_partition_values = []
    for zone_1, zone_2 in product(range(2), repeat=2):
        candidate = {
            0: 0,
            3: 1,
            1: zone_1,
            2: zone_2,
        }
        try:
            validate_complete_seed(problem, candidate)
        except ValueError:
            continue
        legal_partition_values.append(
            sum(
                valuator.value(
                    label,
                    frozenset(
                        node for node, zone in candidate.items() if zone == label
                    ),
                ).shi_welfare
                for label in range(problem.Z)
            )
        )
    assert result.root_lp_upper_bound + 1e-8 >= max(legal_partition_values)


def test_seed_labels_are_normalized_by_centroids():
    problem, assignment, _, _ = _root_fixture()
    swapped = {node: 11 if zone == 0 else 7 for node, zone in assignment.items()}

    normalized = normalize_seed_labels(problem, swapped)

    assert normalized == assignment
    assert validate_complete_seed(problem, normalized) == assignment


def test_cross_level_raw_seed_uses_companion_area_file(tmp_path):
    problem, assignment, _, _ = _root_fixture()
    raw_path = tmp_path / "zone_dict_BlockGroup_1.json"
    raw_path.write_text(json.dumps({"0": 0, "1": 1}))
    area_path = tmp_path / "zone_dict_area_BlockGroup_1.json"
    area_path.write_text(
        json.dumps(
            {
                str(problem.G.nodes[node]["area_id"]): zone
                for node, zone in assignment.items()
            }
        )
    )

    loaded = load_seed_assignment(raw_path, problem)

    assert loaded == assignment


def test_master_rejects_stale_analytical_pattern_fields():
    problem, _, valuator, patterns = _root_fixture()
    stale = replace(patterns[0], shi_welfare=patterns[0].shi_welfare + 1.0)

    with pytest.raises(ValueError, match="welfare disagrees"):
        RestrictedAnalyticalPatternMaster(
            problem.G,
            problem.centroids,
            (stale, patterns[1]),
            max_cut_edges=None,
            pattern_validator=valuator.validate_pattern,
        )


def test_recom_recourse_uses_cp_sum_of_rounded_balance_rows():
    problem, assignment, _, _ = _root_fixture()
    problem.frl_dev = 0.0
    for node in (0, 1):
        problem.G.nodes[node]["FRL"] = 0.496
    for node in (2, 3):
        problem.G.nodes[node]["FRL"] = 0.504
    context = _ReComContext(problem)

    state = context.build_state([assignment[node] for node in context.nodes])

    assert state.feasible


def test_solution_json_artifact_round_trip(tmp_path):
    problem, assignment, _, _ = _root_fixture()
    solution = ZoneSolution(
        problem,
        assignment,
        "FEASIBLE",
        objective=1.0,
        artifacts={
            "shi_mechanism": JsonArtifact(
                "artifacts/mechanism.json",
                summary={"finite_student_exact": False},
                payload={
                    "continuum_large_market_witness": True,
                    "finite_student_exact": False,
                },
            )
        },
    )

    solution.save(str(tmp_path))

    info = json.loads((tmp_path / "solution_BlockGroup_0.json").read_text())
    payload = json.loads((tmp_path / "artifacts/mechanism.json").read_text())
    assert info["artifacts"]["shi_mechanism"]["path"] == ("artifacts/mechanism.json")
    assert "continuum_large_market_witness" not in json.dumps(info)
    assert payload["continuum_large_market_witness"]
    assert not payload["finite_student_exact"]


def test_strategy_config_contract_and_registration():
    config = OptimizationConfig(
        strategy="zoned_column_generation",
        solver="cp_bool",
        levels=["Block_2"],
        years=[23],
        population_type="All",
        remove_city_wide=True,
    )

    strategy = config.make_strategy()

    assert strategy.name == "zoned_column_generation"
    assert get_strategy("zoned_column_generation").name == ("zoned_column_generation")
    assert strategy.options["zoned_cg_wall_time_limit"] == 2700

    with pytest.raises(ValueError, match="positive and finite"):
        OptimizationConfig(
            strategy="zoned_column_generation",
            solver="cp_bool",
            levels=["Block_2"],
            years=[23],
            population_type="All",
            remove_city_wide=True,
            zoned_cg_wall_time_limit=True,
        )

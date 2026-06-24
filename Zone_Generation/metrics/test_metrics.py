import math

import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import box

from Zone_Generation.choice import mnl
from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.optimization.data.contiguity import boundary_edges
from Zone_Generation.optimization.levels import LevelSpec
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.tests.synthetic import make_grid_problem
from Zone_Generation.metrics import MetricsCalculator
from Zone_Generation.metrics.spatial import compute_spatial_metrics


def _problem():
    problem = make_grid_problem(3, 3)
    problem.level = LevelSpec("Block", 0)
    problem.G.graph["school_data"] = {
        100: {
            "program_types": ["GE", "SB"],
            "math_score": 2500,
            "english_score": 2450,
            "ge_capacity": 1,
            "attendance_area": 100,
        },
        200: {
            "program_types": ["GE", "SA"],
            "math_score": 2600,
            "english_score": 2550,
            "ge_capacity": 1,
            "attendance_area": 200,
        },
    }
    return problem


def _assignment():
    return {
        0: 0,
        1: 0,
        3: 0,
        4: 0,
        2: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
    }


def _solution(
    objective=10.0,
    wall_time=1.5,
    time_to_convergence=None,
    metadata=None,
):
    return ZoneSolution(
        problem=_problem(),
        assignment=_assignment(),
        status="FEASIBLE",
        objective=objective,
        wall_time=wall_time,
        time_to_convergence=time_to_convergence,
        metadata=metadata or {"solver": "test"},
    )


def test_optimization_metrics_on_single_solution():
    solution = _solution(objective=12.0, time_to_convergence=0.9)
    result = MetricsCalculator(solution, config={"strategy": "single"}).compute()

    expected_cut_edges = boundary_edges(solution.problem.G, solution.assignment)
    assert result.metrics["num_zones"] == 2
    assert result.metrics["cut_edges"] == expected_cut_edges
    assert result.metrics["final_cut_edges"] == expected_cut_edges
    assert result.metrics["normalized_cut_edges"] == (
        expected_cut_edges / solution.problem.Z
    )
    assert result.metrics["fractional_cut_edges"] == (
        expected_cut_edges / solution.problem.G.number_of_edges()
    )
    assert 0 < result.metrics["avg_reock_score"] <= 1
    assert 0 < result.metrics["avg_polsby_popper_score"] <= 1
    assert result.metrics["final_objective"] == 12.0
    assert result.metrics["time_to_convergence"] == 0.9
    assert result.run["stages"][0]["time_to_convergence"] == 0.9
    assert result.metrics["contiguous"] == 1
    assert result.metrics["avg_total_programs_per_zone"] == 2.0
    assert result.metrics["solution_code"]
    assert set(result.zone_data) == {0, 1}
    assert result.zone_data[0]["ge_students"] == 4.0
    assert result.zone_data[0]["avg_math_score"] == 2500.0


def test_shape_metrics_use_block0_geometry_after_conversion():
    block0 = nx.Graph()
    block0.add_node(0, area_id=10, geometry=box(0, 0, 1, 1))
    block0.add_node(1, area_id=11, geometry=box(1, 0, 2, 1))
    block0.add_edge(0, 1)

    coarse = nx.Graph()
    coarse.add_node(0, block_ids=[10, 11], geometry=box(0, 0, 10, 1))

    expected = compute_spatial_metrics(
        ZoneSolution(
            problem=ZoneProblem(block0, LevelSpec("Block", 0), [0]),
            assignment={0: 0, 1: 0},
            status="FEASIBLE",
        ),
        config={"block0_graph": block0},
    )
    actual = compute_spatial_metrics(
        ZoneSolution(
            problem=ZoneProblem(coarse, LevelSpec("Block", 1), [0]),
            assignment={0: 0},
            status="FEASIBLE",
        ),
        config={"block0_graph": block0},
    )

    assert math.isclose(actual.avg_reock_score, expected.avg_reock_score)
    assert math.isclose(actual.fractional_cut_edges, expected.fractional_cut_edges)
    assert math.isclose(
        actual.avg_polsby_popper_score,
        expected.avg_polsby_popper_score,
    )


def test_recursive_stage_metrics_default_to_final_only():
    first = _solution(objective=20.0, wall_time=2.0, time_to_convergence=0.5)
    final = _solution(objective=10.0, wall_time=3.0, time_to_convergence=1.25)
    result = MetricsCalculator([first, final], config={"strategy": "recursive"}).compute()

    assert result.run["strategy"] == "recursive"
    assert result.run["selection"] == "last_solution_with_assignment"
    assert result.run["final_stage"] == "stage_01_Block_0"
    assert result.metrics["total_wall_time"] == 5.0
    assert result.metrics["time_to_convergence"] == 1.75
    assert result.metrics["objective_stage_00_Block_0"] == 20.0
    assert result.metrics["objective_stage_01_Block_0"] == 10.0
    assert result.metrics["time_to_convergence_stage_00_Block_0"] == 0.5
    assert result.metrics["time_to_convergence_stage_01_Block_0"] == 1.25
    assert "cut_edges_stage_01_Block_0" not in result.metrics
    assert result.run["stages"][1]["cut_edges"] is None
    assert result.metrics["cut_edges"] > 0
    assert result.run["final_cut_edges"] == result.metrics["cut_edges"]
    assert len(result.run["stages"]) == 2


def test_recursive_stage_metrics_can_be_enabled():
    first = _solution(objective=20.0, wall_time=2.0)
    final = _solution(objective=10.0, wall_time=3.0)
    result = MetricsCalculator(
        [first, final],
        config={"strategy": "recursive"},
        compute_stage_metrics=True,
    ).compute()

    assert result.metrics["cut_edges_stage_01_Block_0"] > 0
    assert result.metrics["normalized_cut_edges_stage_01_Block_0"] > 0
    assert result.metrics["fractional_cut_edges_stage_01_Block_0"] > 0
    assert result.run["stages"][1]["cut_edges"] > 0
    assert result.run["stages"][1]["fractional_cut_edges"] > 0


def test_iterative_metrics_select_best_choice_utility():
    low = _solution(
        objective=30.0,
        time_to_convergence=0.2,
        metadata={"choice_utility": 1.0},
    )
    best = _solution(
        objective=25.0,
        time_to_convergence=0.6,
        metadata={"choice_utility": 2.5},
    )
    later = _solution(
        objective=20.0,
        time_to_convergence=0.8,
        metadata={"choice_utility": 2.0},
    )

    result = MetricsCalculator(
        [low, best, later], config={"strategy": "iterative_choice"}
    ).compute()

    assert result.run["selection"] == "best_choice_utility"
    assert result.run["final_stage"] == "iteration_01_Block_0"
    assert result.metrics["final_objective"] == 25.0
    assert result.metrics["final_choice_utility"] == 2.5
    assert result.metrics["time_to_convergence"] == 0.2
    assert result.run["stages"][2]["choice_utility"] == 2.0
    assert result.run["stages"][2]["time_to_convergence"] == 0.8


def test_choice_metric_uses_configured_mnl_method(tmp_path, monkeypatch):
    utility_path = tmp_path / "utility.csv"
    student_path = tmp_path / "students.csv"
    pd.DataFrame(
        {
            "studentno": [1, 2],
            "100-GE-KG": [2.0, 1.0],
            "100-SA-KG": [3.0, 0.0],
            "200-GE-KG": [0.5, 4.0],
            "200-SA-KG": [0.2, 5.0],
        }
    ).to_csv(utility_path, index=False)
    pd.DataFrame(
        {
            "studentno": [1, 2],
            "census_blockgroup": [1001, 1002],
        }
    ).to_csv(student_path, index=False)

    monkeypatch.setattr(mnl, "DEFAULT_UTILITY_PATH", str(utility_path))
    monkeypatch.setattr(mnl, "DEFAULT_STUDENT_PATH", str(student_path))

    solution = ZoneSolution(
        problem=make_grid_problem(2, 2),
        assignment={0: 0, 1: 0, 2: 1, 3: 1},
        status="FEASIBLE",
    )
    column = MetricColumns.CHOICE_TOTAL_PREASSIGNMENT_UTILITY

    max_result = MetricsCalculator(
        solution,
        config={"choice_model": "mnl", "choice_model_method": "max"},
    ).compute()
    logsum_result = MetricsCalculator(
        solution,
        config={"choice_model": "mnl", "choice_model_method": "logsum"},
    ).compute()

    expected_logsum = (
        3.0
        + math.log1p(math.exp(-1.0))
        + 5.0
        + math.log1p(math.exp(-1.0))
    )
    assert max_result.metrics[column] == pytest.approx(8.0)
    assert logsum_result.metrics[column] == pytest.approx(expected_logsum)
    assert logsum_result.run["choice_preassignment_utility"]["method"] == "logsum"

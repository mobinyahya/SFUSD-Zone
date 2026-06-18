import math

import networkx as nx
from shapely.geometry import box

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


def _solution(objective=10.0, wall_time=1.5, metadata=None):
    return ZoneSolution(
        problem=_problem(),
        assignment=_assignment(),
        status="FEASIBLE",
        objective=objective,
        wall_time=wall_time,
        metadata=metadata or {"solver": "test"},
    )


def test_optimization_metrics_on_single_solution():
    solution = _solution(objective=12.0)
    result = MetricsCalculator(solution, config={"strategy": "single"}).compute()

    expected_cut_edges = boundary_edges(solution.problem.G, solution.assignment)
    assert result.metrics["num_zones"] == 2
    assert result.metrics["cut_edges"] == expected_cut_edges
    assert result.metrics["final_cut_edges"] == expected_cut_edges
    assert result.metrics["normalized_cut_edges"] == (
        expected_cut_edges / solution.problem.Z
    )
    assert 0 < result.metrics["avg_reock_score"] <= 1
    assert 0 < result.metrics["avg_polsby_popper_score"] <= 1
    assert result.metrics["final_objective"] == 12.0
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
    assert math.isclose(
        actual.avg_polsby_popper_score,
        expected.avg_polsby_popper_score,
    )


def test_recursive_stage_metrics_are_preserved():
    first = _solution(objective=20.0, wall_time=2.0)
    final = _solution(objective=10.0, wall_time=3.0)
    result = MetricsCalculator([first, final], config={"strategy": "recursive"}).compute()

    assert result.run["strategy"] == "recursive"
    assert result.run["selection"] == "last_solution_with_assignment"
    assert result.run["final_stage"] == "stage_01_Block_0"
    assert result.metrics["total_wall_time"] == 5.0
    assert result.metrics["objective_stage_00_Block_0"] == 20.0
    assert result.metrics["objective_stage_01_Block_0"] == 10.0
    assert result.metrics["cut_edges_stage_01_Block_0"] > 0
    assert result.metrics["normalized_cut_edges_stage_01_Block_0"] > 0
    assert len(result.run["stages"]) == 2


def test_iterative_metrics_select_best_choice_utility():
    low = _solution(objective=30.0, metadata={"choice_utility": 1.0})
    best = _solution(objective=25.0, metadata={"choice_utility": 2.5})
    later = _solution(objective=20.0, metadata={"choice_utility": 2.0})

    result = MetricsCalculator(
        [low, best, later], config={"strategy": "iterative_choice"}
    ).compute()

    assert result.run["selection"] == "best_choice_utility"
    assert result.run["final_stage"] == "iteration_01_Block_0"
    assert result.metrics["final_objective"] == 25.0
    assert result.metrics["final_choice_utility"] == 2.5
    assert result.run["stages"][2]["choice_utility"] == 2.0

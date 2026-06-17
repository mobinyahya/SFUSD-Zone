from Zone_Generation.pipeline.data.contiguity import boundary_edges
from Zone_Generation.pipeline.solution import ZoneSolution
from Zone_Generation.pipeline.tests.synthetic import make_grid_problem
from Zone_Generation.Running_Analysis.metrics import MetricsCalculator


def _problem():
    problem = make_grid_problem(3, 3)
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


def test_pipeline_metrics_on_single_solution():
    solution = _solution(objective=12.0)
    result = MetricsCalculator(solution, config={"strategy": "single"}).compute()

    expected_boundary = boundary_edges(solution.problem.G, solution.assignment)
    assert result.metrics["num_zones"] == 2
    assert result.metrics["boundary_cost"] == expected_boundary
    assert result.metrics["final_boundary_cost"] == expected_boundary
    assert result.metrics["final_objective"] == 12.0
    assert result.metrics["contiguous"] == 1
    assert result.metrics["avg_total_programs_per_zone"] == 2.0
    assert result.metrics["solution_code"]
    assert set(result.zone_data) == {0, 1}
    assert result.zone_data[0]["ge_students"] == 4.0
    assert result.zone_data[0]["avg_math_score"] == 2500.0


def test_recursive_stage_metrics_are_preserved():
    first = _solution(objective=20.0, wall_time=2.0)
    final = _solution(objective=10.0, wall_time=3.0)
    result = MetricsCalculator([first, final], config={"strategy": "recursive"}).compute()

    assert result.run["strategy"] == "recursive"
    assert result.run["selection"] == "last_solution_with_assignment"
    assert result.run["final_stage"] == "stage_01_BlockGroup_0"
    assert result.metrics["total_wall_time"] == 5.0
    assert result.metrics["objective_stage_00_BlockGroup_0"] == 20.0
    assert result.metrics["objective_stage_01_BlockGroup_0"] == 10.0
    assert len(result.run["stages"]) == 2


def test_iterative_metrics_select_best_choice_utility():
    low = _solution(objective=30.0, metadata={"choice_utility": 1.0})
    best = _solution(objective=25.0, metadata={"choice_utility": 2.5})
    later = _solution(objective=20.0, metadata={"choice_utility": 2.0})

    result = MetricsCalculator(
        [low, best, later], config={"strategy": "iterative_choice"}
    ).compute()

    assert result.run["selection"] == "best_choice_utility"
    assert result.run["final_stage"] == "iteration_01_BlockGroup_0"
    assert result.metrics["final_objective"] == 25.0
    assert result.metrics["final_choice_utility"] == 2.5
    assert result.run["stages"][2]["choice_utility"] == 2.0

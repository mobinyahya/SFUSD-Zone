import math
import pytest

from optimization.data.initial_solutions import initial_solution
from optimization.tests.synthetic import make_grid_problem


def test_voronoi_initial_solution_returns_valid_hint():
    problem = make_grid_problem(3, 3)

    result = initial_solution(problem, "voronoi")

    assert result is not None
    assert result.metadata["hints"] == "voronoi"
    _check_candidate_assignment(problem, result.assignment)


def test_feasible_initial_solution_satisfies_zoning_model():
    problem = make_grid_problem(3, 3, boundary_prop=0.5)

    result = initial_solution(
        problem,
        "feasible",
        solver_options={"feasible_hint_time_limit": 10, "seed": 3},
    )

    assert result is not None
    assert result.metadata["hints"] == "feasible"
    assert result.metadata["hint_solver"] == "cp_bool"
    _check_candidate_assignment(problem, result.assignment)
    assert all(
        len({node for node, zone in result.assignment.items() if zone == z}) > 0
        for z in range(problem.Z)
    )


def test_initial_solution_none_returns_no_hint():
    problem = make_grid_problem(3, 3)

    assert initial_solution(problem, "none") is None


def test_initial_solution_rejects_unknown_hints():
    problem = make_grid_problem(3, 3)

    with pytest.raises(ValueError, match="hints"):
        initial_solution(problem, "bad")


def test_initial_solution_reports_empty_candidate_zones():
    problem = make_grid_problem(3, 3, max_distance=0.5)

    with pytest.raises(ValueError) as exc:
        initial_solution(problem, "voronoi")

    message = str(exc.value)
    assert "Node 1 (area_id=1001) has no candidate zones for BlockGroup_0" in message
    assert "max_distance=0.5 excludes all 2 centroids" in message
    assert "Nearest centroid is zone 0 at node 0 (1.000 miles away)" in message
    assert "Increase max_distance" in message


@pytest.mark.parametrize("value", [0, -1, float("inf"), True, "bad"])
def test_feasible_initial_solution_rejects_invalid_time_limit(value):
    problem = make_grid_problem(3, 3)

    with pytest.raises(ValueError, match="feasible_hint_time_limit"):
        initial_solution(
            problem,
            "feasible",
            solver_options={"feasible_hint_time_limit": value},
        )


def _check_candidate_assignment(problem, assignment):
    assert set(assignment) == set(problem.nodes)
    for node, zone in assignment.items():
        assert zone in problem.candidate_zones(node)
    for zone, centroid in enumerate(problem.centroids):
        assert assignment[centroid] == zone


def test_grid_problem_auto_max_distance():
    problem = make_grid_problem(3, 3, max_distance="auto")
    # For a 3x3 grid with centroids at (0,0) and (2,2), the maximum distance
    # to the closest centroid is 2.0 (at (0,2) and (2,0)).
    assert math.isclose(problem.max_distance, 2.0)
    assert problem.candidate_zones(0) == {0}
    assert problem.candidate_zones(8) == {1}
    # Node 1 is (0,1): dist to 0 is 1.0 <= 2.0, dist to 8 is sqrt(5) > 2.0 -> {0}
    assert problem.candidate_zones(1) == {0}
    # Node 7 is (2,1): dist to 0 is sqrt(5) > 2.0, dist to 8 is 1.0 <= 2.0 -> {1}
    assert problem.candidate_zones(7) == {1}
    # Every node has at least one candidate zone
    for node in problem.G.nodes():
        assert len(problem.candidate_zones(node)) >= 1



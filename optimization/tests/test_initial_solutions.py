import pytest

from optimization.data.initial_solutions import initial_solution
from optimization.tests.synthetic import make_grid_problem


def test_voronoi_initial_solution_returns_valid_hint():
    problem = make_grid_problem(3, 3)

    result = initial_solution(problem, "voronoi")

    assert result is not None
    assert result.metadata["hints"] == "voronoi"
    _check_candidate_assignment(problem, result.assignment)


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


def _check_candidate_assignment(problem, assignment):
    assert set(assignment) == set(problem.nodes)
    for node, zone in assignment.items():
        assert zone in problem.candidate_zones(node)
    for zone, centroid in enumerate(problem.centroids):
        assert assignment[centroid] == zone

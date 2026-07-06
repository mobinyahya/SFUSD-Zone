import pytest

from Zone_Generation.optimization.data.initial_solutions import initial_solution
from Zone_Generation.optimization.tests.synthetic import make_grid_problem


def test_voronoi_initial_solution_returns_valid_hint():
    problem = make_grid_problem(3, 3)

    result = initial_solution(problem, "voronoi")

    assert result is not None
    assert result.metadata["hints"] == "voronoi"
    _check_candidate_assignment(problem, result.assignment)


def test_gerry_chain_initial_solution_returns_valid_hint():
    problem = make_grid_problem(3, 3)

    result = initial_solution(problem, "gerry_chain", cut_attempts=10)

    assert result is not None
    assert result.metadata["hints"] == "gerry_chain"
    _check_candidate_assignment(problem, result.assignment)


def test_initial_solution_none_returns_no_hint():
    problem = make_grid_problem(3, 3)

    assert initial_solution(problem, "none") is None


def test_initial_solution_rejects_unknown_hints():
    problem = make_grid_problem(3, 3)

    with pytest.raises(ValueError, match="hints"):
        initial_solution(problem, "bad")


def _check_candidate_assignment(problem, assignment):
    assert set(assignment) == set(problem.nodes)
    for node, zone in assignment.items():
        assert zone in problem.candidate_zones(node)
    for zone, centroid in enumerate(problem.centroids):
        assert assignment[centroid] == zone

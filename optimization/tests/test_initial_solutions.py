import math
import pytest

from optimization.config import OptimizationConfig
from optimization.data import initial_solutions
from optimization.data.feasibility import FeasibilityReport, FeasibilityViolation
from optimization.data.initial_solutions import (
    FEASIBLE_HINT_PAYLOAD,
    FeasibleHintError,
    _feasible_hint_namespace,
    feasibility_fingerprint,
    initial_solution,
)
from optimization.solvers import cpsat
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
    # to the closest centroid is 2.0 (at (0,2) and (2,0)). With the 1.2x factor,
    # max_distance becomes 2.4.
    assert math.isclose(problem.max_distance, 2.4)
    assert problem.candidate_zones(0) == {0}
    assert problem.candidate_zones(8) == {1}
    # Every node has at least one candidate zone
    for node in problem.G.nodes():
        assert len(problem.candidate_zones(node)) >= 1


def _cache_config(tmp_path):
    return OptimizationConfig(
        data={
            "scenario": "legacy",
            "overrides": {"roots": {"cache": str(tmp_path / "cache")}},
        }
    )


def _cached_grid_problem(tmp_path, **overrides):
    return make_grid_problem(
        3,
        3,
        boundary_prop=0.5,
        optimization_config=_cache_config(tmp_path),
        **overrides,
    )


def test_feasible_hint_cache_reuses_the_stored_assignment(tmp_path, monkeypatch):
    options = {"feasible_hint_time_limit": 10, "seed": 3}
    first = initial_solution(
        _cached_grid_problem(tmp_path), "feasible", solver_options=options
    )
    assert first.metadata["hint_cache"] == "miss"

    def refuse(*args, **kwargs):
        raise AssertionError("A cached feasible hint must not be re-solved.")

    monkeypatch.setattr(cpsat, "CpBoolSolver", refuse)
    second = initial_solution(
        _cached_grid_problem(tmp_path), "feasible", solver_options=options
    )

    assert second.metadata["hint_cache"] == "hit"
    assert second.metadata["hint_cache_key"] == first.metadata["hint_cache_key"]
    assert second.metadata["hint_solver_status"] == first.metadata["hint_solver_status"]
    assert second.assignment == first.assignment


def test_feasible_hint_cache_key_ignores_search_settings(tmp_path, monkeypatch):
    """Search effort does not change the feasible set, so it must not fork the key."""

    problem = _cached_grid_problem(tmp_path)
    first = initial_solution(
        problem,
        "feasible",
        solver_options={"feasible_hint_time_limit": 10, "seed": 3},
    )
    assert first.metadata["hint_cache"] == "miss"

    def refuse(*args, **kwargs):
        raise AssertionError("Search settings must not trigger a second solve.")

    monkeypatch.setattr(cpsat, "CpBoolSolver", refuse)
    for options in (
        {"feasible_hint_time_limit": 10, "seed": 4},
        {"feasible_hint_time_limit": 45, "seed": 3},
        {"feasible_hint_time_limit": 10, "seed": 3, "workers": 4},
        {"feasible_hint_time_limit": 10, "seed": 3, "linearization_level": 0},
        {"feasible_hint_time_limit": 10, "seed": 3, "symmetry_level": 2},
    ):
        reused = initial_solution(problem, "feasible", solver_options=options)
        assert reused.metadata["hint_cache"] == "hit"
        assert reused.metadata["hint_cache_key"] == first.metadata["hint_cache_key"]
        assert reused.assignment == first.assignment


def test_feasible_hint_cache_key_includes_centroid_neighbor_radius(tmp_path):
    """A positive radius fixes centroid neighborhoods, so it changes the model."""

    problem = _cached_grid_problem(tmp_path)
    first = initial_solution(
        problem,
        "feasible",
        solver_options={"feasible_hint_time_limit": 10, "seed": 3},
    )
    wider = initial_solution(
        problem,
        "feasible",
        solver_options={
            "feasible_hint_time_limit": 10,
            "seed": 3,
            "centroid_neighbor_radius": 1,
        },
    )

    assert wider.metadata["hint_cache"] == "miss"
    assert wider.metadata["hint_cache_key"] != first.metadata["hint_cache_key"]


def test_feasible_hint_cache_is_skipped_without_a_scenario():
    result = initial_solution(
        make_grid_problem(3, 3, boundary_prop=0.5),
        "feasible",
        solver_options={"feasible_hint_time_limit": 10, "seed": 3},
    )

    assert "hint_cache" not in result.metadata


def test_feasibility_fingerprint_tracks_constraints_and_data():
    problem = make_grid_problem(3, 3, boundary_prop=0.5)
    baseline = feasibility_fingerprint(problem)

    assert feasibility_fingerprint(make_grid_problem(3, 3, boundary_prop=0.5)) == (
        baseline
    )
    assert feasibility_fingerprint(make_grid_problem(3, 3, boundary_prop=0.4)) != (
        baseline
    )

    changed_data = make_grid_problem(3, 3, boundary_prop=0.5)
    changed_data.G.nodes[4]["ge_students"] = 2.0
    assert feasibility_fingerprint(changed_data) != baseline


def test_feasible_hint_cache_ignores_an_invalid_cached_assignment(tmp_path):
    options = {"feasible_hint_time_limit": 10, "seed": 3}
    problem = _cached_grid_problem(tmp_path)
    initial_solution(problem, "feasible", solver_options=options)

    namespace = _feasible_hint_namespace(problem, options)
    namespace.save_pickle(
        FEASIBLE_HINT_PAYLOAD,
        {"assignment": {0: 1}, "status": "FEASIBLE", "wall_time": 0.0},
    )
    result = initial_solution(problem, "feasible", solver_options=options)

    assert result.metadata["hint_cache"] == "rejected"
    assert "coverage" in result.metadata["hint_cache_rejected_reason"]
    _check_candidate_assignment(problem, result.assignment)


def test_feasible_hint_cache_re_solves_a_hint_the_validator_rejects(
    tmp_path, monkeypatch
):
    """A stored hint that fails the exact check is replaced, not handed on.

    This is the shape of the failure that took short bursts down: a warm start
    every structural check accepts, but which a consumer re-checking the exact
    constraints rejects. The cached copy must not survive it.
    """

    options = {"feasible_hint_time_limit": 10, "seed": 3}
    problem = _cached_grid_problem(tmp_path)
    stored = initial_solution(problem, "feasible", solver_options=options)
    assert stored.metadata["hint_cache"] == "miss"

    real_check = initial_solutions.check_zoning
    calls = []

    def reject_the_cached_one(prob, assignment, **kwargs):
        calls.append(assignment)
        if len(calls) == 1:
            return FeasibilityReport(
                (FeasibilityViolation("frl_upper", "synthetic", 0.5),)
            )
        return real_check(prob, assignment, **kwargs)

    monkeypatch.setattr(initial_solutions, "check_zoning", reject_the_cached_one)
    result = initial_solution(problem, "feasible", solver_options=options)

    assert result.metadata["hint_cache"] == "rejected"
    assert "frl_upper" in result.metadata["hint_cache_rejected_reason"]
    _check_candidate_assignment(problem, result.assignment)

    # The replacement was written back, so the next run is a clean hit.
    monkeypatch.setattr(initial_solutions, "check_zoning", real_check)
    assert (
        initial_solution(problem, "feasible", solver_options=options).metadata[
            "hint_cache"
        ]
        == "hit"
    )


def test_feasible_hint_raises_when_the_solver_disagrees_with_the_validator():
    """A hint the solver calls feasible but the exact check rejects is fatal."""

    problem = make_grid_problem(3, 3, boundary_prop=0.5)

    with pytest.raises(FeasibleHintError, match="fails the exact feasibility check"):
        _solve_with_validator(
            problem,
            FeasibilityReport((FeasibilityViolation("frl_upper", "synthetic", 0.5),)),
        )


def _solve_with_validator(problem, report):
    original = initial_solutions.check_zoning
    initial_solutions.check_zoning = lambda *args, **kwargs: report
    try:
        return initial_solution(
            problem,
            "feasible",
            solver_options={"feasible_hint_time_limit": 10, "seed": 3},
        )
    finally:
        initial_solutions.check_zoning = original


def test_hint_solver_options_reject_an_unclassified_option(monkeypatch):
    """A new hint-solver option must be classified before it can be used."""

    monkeypatch.setattr(initial_solutions, "_HINT_SEARCH_OPTIONS", ("seed",))

    with pytest.raises(ValueError, match="classified"):
        initial_solutions._hint_solver_options({"seed": 3})


def test_hint_cache_key_covers_every_feasibility_option(monkeypatch):
    """Classifying an option as feasibility-affecting must put it in the key."""

    monkeypatch.setattr(
        initial_solutions,
        "_HINT_FEASIBILITY_OPTIONS",
        ("centroid_neighbor_radius", "some_new_knob"),
    )

    with pytest.raises(ValueError, match="missing from the cache key"):
        initial_solutions._hint_model_identity({})

"""Data-free, end-to-end solver tests on a synthetic grid problem."""

import os

import pytest
from ortools.sat.python import cp_model

from Zone_Generation.choice.models import DistanceChoiceModel
from Zone_Generation.choice.objective import ChoiceObjective
from Zone_Generation.optimization.solvers import get_solver
from Zone_Generation.optimization.solvers.base import available_solvers
from Zone_Generation.optimization.tests.synthetic import make_grid_problem


def _check_valid(problem, solution):
    assert solution.feasible or solution.status == "STUB"
    # every node assigned to a candidate zone
    assert set(solution.assignment) == set(problem.nodes)
    for node, zone in solution.assignment.items():
        assert zone in problem.candidate_zones(node)
    # centroids anchor their zones
    for z, centroid in enumerate(problem.centroids):
        assert solution.assignment[centroid] == z
    assert solution.is_contiguous()


@pytest.mark.parametrize("name", ["cp_int", "cp_bool"])
def test_cpsat_solvers(name):
    problem = make_grid_problem(3, 3)
    solver = get_solver(name, solve_time_limit=10, workers=1)
    solution = solver.solve(problem)
    assert solution.status in ("OPTIMAL", "FEASIBLE")
    _check_valid(problem, solution)


def test_cp_int_does_not_add_exactly_one_constraint(monkeypatch):
    def fail_exactly_one(self, *args, **kwargs):
        raise AssertionError("cp_int should get assignment from integer domains")

    monkeypatch.setattr(cp_model.CpModel, "add_exactly_one", fail_exactly_one)
    problem = make_grid_problem(3, 3)
    solver = get_solver("cp_int", solve_time_limit=10, workers=1)
    solution = solver.solve(problem)
    assert solution.status in ("OPTIMAL", "FEASIBLE")
    _check_valid(problem, solution)


def test_cpsat_solver_saves_logs(tmp_path):
    problem = make_grid_problem(3, 3)
    solver = get_solver(
        "cp_int",
        solve_time_limit=10,
        workers=1,
        save_solver_logs=True,
        output_dir=str(tmp_path),
    )

    first = solver.solve(problem)
    second = solver.solve(problem)

    first_log = os.path.join(
        "solver_logs", "solver_00_BlockGroup_0_cp_int.log"
    )
    second_log = os.path.join(
        "solver_logs", "solver_01_BlockGroup_0_cp_int.log"
    )
    assert first.metadata["solver_log_path"] == first_log
    assert second.metadata["solver_log_path"] == second_log
    first_log_path = tmp_path / first_log
    second_log_path = tmp_path / second_log
    assert first_log_path.exists()
    assert second_log_path.exists()
    contents = first_log_path.read_text(encoding="utf-8")
    assert contents.strip()
    assert "CP-SAT" in contents or "CpSolverResponse" in contents


def test_local_search_stub():
    problem = make_grid_problem(3, 3)
    solution = get_solver("local_search").solve(problem)
    assert solution.status == "STUB"
    _check_valid(problem, solution)


@pytest.mark.parametrize("name", ["cp_int", "cp_bool"])
def test_cpsat_solvers_support_choice_objective(name):
    problem = make_grid_problem(3, 3)
    model = DistanceChoiceModel()
    evaluation = model.evaluate_with_cuts(
        problem,
        {node: min(problem.candidate_zones(node)) for node in problem.nodes},
    )
    lower, upper = model.utility_bounds(problem)
    problem.choice_objective = ChoiceObjective(
        cuts=evaluation.cuts,
        lower_bound=lower,
        upper_bound=upper,
        scale=100,
    )

    solution = get_solver(name, solve_time_limit=10, workers=1).solve(problem)

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    _check_valid(problem, solution)
    assert solution.metadata["objective_kind"] == "choice_utility"
    assert solution.objective == pytest.approx(
        model.evaluate(problem, solution.assignment), abs=0.05
    )


def test_explicit_candidates_cannot_unassign_centroids():
    problem = make_grid_problem(3, 3, candidates={0: {1}, 8: {0}})

    assert problem.candidate_zones(0) == {0}
    assert problem.candidate_zones(8) == {1}


def test_area_assignment_and_save(tmp_path):
    problem = make_grid_problem(3, 3)
    solution = get_solver("cp_int", solve_time_limit=10, workers=1).solve(problem)
    area = solution.area_assignment()
    # base graph: one area_id per node
    assert len(area) == len(problem.nodes)
    solution.save(str(tmp_path))
    assert (tmp_path / "zone_dict_BlockGroup_0.json").exists()
    assert (tmp_path / "solution_BlockGroup_0.json").exists()


@pytest.mark.skipif(
    "mip" not in available_solvers(), reason="gurobipy not installed"
)
def test_mip_solver():
    problem = make_grid_problem(3, 3)
    try:
        solution = get_solver("mip", solve_time_limit=10).solve(problem)
    except Exception as exc:  # no usable Gurobi license in this environment
        if "gurobi" in type(exc).__module__.lower():
            pytest.skip(f"Gurobi unavailable: {exc}")
        raise
    assert solution.status in ("OPTIMAL", "FEASIBLE")
    _check_valid(problem, solution)

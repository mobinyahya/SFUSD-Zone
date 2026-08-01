"""Data-free, end-to-end solver tests on a synthetic grid problem."""

import json
import os

import pytest
from ortools.sat.python import cp_model

from choice.models import DistanceChoiceModel
from choice.objective import ChoiceObjective
from optimization.config import OptimizationConfig
from optimization.data import contiguity
from optimization.problem import CutoffMarket, CutoffStudent
from optimization.solvers import get_solver
from optimization.solvers.balance import balance_constraints
from optimization.solvers.base import available_solvers
from optimization.tests.solver_contract import assert_valid_solution
from optimization.tests.synthetic import make_grid_problem


def test_config_passes_cpsat_parameters_to_solver():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="cp_int",
        linearization_level=0,
        cp_model_probing_level=1,
        symmetry_level=0,
        cp_sat_search_strategy="distance_to_centroid",
        secondary_objective=True,
    )

    solver = config.make_solver()

    assert solver.options["linearization_level"] == 0
    assert solver.options["cp_model_probing_level"] == 1
    assert solver.options["symmetry_level"] == 0
    assert solver.options["cp_sat_search_strategy"] == "distance_to_centroid"
    assert solver.options["secondary_objective"] is True


def test_cp_bool_solver_supports_secondary_objective():
    problem = make_grid_problem(3, 3)
    solver = get_solver(
        "cp_bool",
        solve_time_limit=10,
        workers=1,
        secondary_objective=True,
    )

    solution = solver.solve(problem)

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    assert_valid_solution(problem, solution)
    assert solution.objective == contiguity.boundary_edges(
        problem.G,
        solution.assignment,
    )


def test_cpsat_solver_applies_configured_cp_sat_parameters():
    solver = get_solver(
        "cp_int",
        solve_time_limit=10,
        relative_gap_limit=0.25,
        workers=1,
        seed=7,
        linearization_level=0,
        cp_model_probing_level=1,
        symmetry_level=0,
        cp_sat_search_strategy="distance_to_centroid",
    )
    cp_solver = cp_model.CpSolver()

    solver._configure_solver_parameters(cp_solver)

    assert cp_solver.parameters.max_time_in_seconds == 10
    assert cp_solver.parameters.relative_gap_limit == 0.25
    assert cp_solver.parameters.num_search_workers == 1
    assert cp_solver.parameters.random_seed == 7
    assert cp_solver.parameters.linearization_level == 0
    assert cp_solver.parameters.cp_model_probing_level == 1
    assert cp_solver.parameters.symmetry_level == 0
    assert cp_solver.parameters.search_branching == cp_model.PARTIAL_FIXED_SEARCH


def test_cpsat_solver_rejects_unknown_search_strategy():
    solver = get_solver("cp_int", cp_sat_search_strategy="bad_strategy")

    with pytest.raises(ValueError, match="cp_sat_search_strategy"):
        solver._configure_solver_parameters(cp_model.CpSolver())


@pytest.mark.parametrize("name", ["cp_int", "cp_bool"])
def test_cpsat_solvers_support_distance_to_centroid_search_strategy(name):
    problem = make_grid_problem(3, 3)
    solver = get_solver(
        name,
        solve_time_limit=10,
        workers=1,
        cp_sat_search_strategy="distance_to_centroid",
    )

    solution = solver.solve(problem)

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    assert_valid_solution(problem, solution)


@pytest.mark.parametrize("name", ["cp_int", "cp_bool"])
def test_cpsat_solvers_forbid_zone_without_closer_neighbor(name):
    problem = make_grid_problem(3, 3, candidates={4: {0}})
    problem.G.graph["closer_neighbors"][4][100] = frozenset()

    solution = get_solver(name, solve_time_limit=10, workers=1).solve(problem)

    assert solution.status == "INFEASIBLE"


def test_cp_bool_distance_to_centroid_search_orders_assignment_bools():
    problem = make_grid_problem(3, 3)
    solver = get_solver("cp_bool", cp_sat_search_strategy="distance_to_centroid")
    model = cp_model.CpModel()
    x, y = solver._build_assignment_vars(model, problem)

    solver._add_search_strategy(model, problem, x, y)

    strategy = model.Proto().search_strategy[0]
    expected_keys = sorted(
        x,
        key=lambda zone_node: (
            problem.distance(problem.centroids[zone_node[0]], zone_node[1]),
            zone_node[1],
            zone_node[0],
        ),
    )
    assert _decision_strategy_var_indices(model) == [
        x[key].Index() for key in expected_keys
    ]
    assert strategy.variable_selection_strategy == cp_model.CHOOSE_FIRST
    assert strategy.domain_reduction_strategy == cp_model.SELECT_MAX_VALUE


def test_cp_int_distance_to_centroid_search_orders_only_integer_vars():
    problem = make_grid_problem(3, 3)
    solver = get_solver("cp_int", cp_sat_search_strategy="distance_to_centroid")
    model = cp_model.CpModel()
    x, y = solver._build_assignment_vars(model, problem)

    solver._add_search_strategy(model, problem, x, y)

    strategy = model.Proto().search_strategy[0]
    expected_nodes = sorted(
        y,
        key=lambda node: (
            min(
                problem.distance(problem.centroids[z], node)
                for z in problem.candidate_zones(node)
            ),
            node,
        ),
    )
    decision_indices = _decision_strategy_var_indices(model)
    assert decision_indices == [y[node].Index() for node in expected_nodes]
    assert all(_var_name(model, idx).startswith("y_") for idx in decision_indices)
    assert strategy.variable_selection_strategy == cp_model.CHOOSE_FIRST
    assert strategy.domain_reduction_strategy == cp_model.SELECT_MIN_VALUE


def _decision_strategy_var_indices(model):
    return [expr.vars[0] for expr in model.Proto().search_strategy[0].exprs]


def _var_name(model, index):
    return model.Proto().variables[index].name


def test_cp_int_does_not_add_exactly_one_constraint(monkeypatch):
    def fail_exactly_one(self, *args, **kwargs):
        raise AssertionError("cp_int should get assignment from integer domains")

    monkeypatch.setattr(cp_model.CpModel, "add_exactly_one", fail_exactly_one)
    problem = make_grid_problem(3, 3)
    solver = get_solver("cp_int", solve_time_limit=10, workers=1)
    solution = solver.solve(problem)
    assert solution.status in ("OPTIMAL", "FEASIBLE")
    assert_valid_solution(problem, solution)


def test_negative_racial_dev_disables_racial_balance_constraints():
    problem = make_grid_problem(3, 3, racial_dev=-1)

    assert len(balance_constraints(problem)) == 2


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

    first_log = os.path.join("solver_logs", "solver_00_BlockGroup_0_cp_int.log")
    second_log = os.path.join("solver_logs", "solver_01_BlockGroup_0_cp_int.log")
    assert first.metadata["solver_log_path"] == first_log
    assert second.metadata["solver_log_path"] == second_log
    first_log_path = tmp_path / first_log
    second_log_path = tmp_path / second_log
    assert first_log_path.exists()
    assert second_log_path.exists()
    contents = first_log_path.read_text(encoding="utf-8")
    assert contents.strip()
    assert "CP-SAT" in contents or "CpSolverResponse" in contents
    assert any(line.startswith("Parameters:") for line in contents.splitlines())


def test_cpsat_solver_saves_progress(tmp_path):
    problem = make_grid_problem(3, 3)
    solver = get_solver(
        "cp_int",
        solve_time_limit=10,
        workers=1,
        save_solver_progress=True,
        output_dir=str(tmp_path),
    )

    solution = solver.solve(problem)
    assert solution.solver_progress

    solution.save(str(tmp_path))

    expected_dir = os.path.join("solver_progress", "solver_00_BlockGroup_0_cp_int")
    expected_log = os.path.join(expected_dir, "progress.jsonl")
    assert solution.metadata["solver_progress_path"] == expected_log
    assert solution.metadata["solver_progress_format"] == "jsonl"
    assert solution.metadata["solver_progress_count"] == len(solution.solver_progress)

    rows = [
        json.loads(line)
        for line in (tmp_path / expected_log).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == len(solution.solver_progress)
    assert rows[-1]["objective"] == pytest.approx(solution.objective)
    objectives = [row["objective"] for row in rows]
    assert all(next_obj < obj for obj, next_obj in zip(objectives, objectives[1:]))

    for idx, row in enumerate(rows):
        assert row["solution_index"] == idx
        assignment_path = tmp_path / expected_dir / row["assignment_path"]
        area_path = tmp_path / expected_dir / row["area_assignment_path"]
        assert assignment_path.exists()
        assert area_path.exists()
        assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
        assert {int(node) for node in assignment} == set(problem.nodes)


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
    assert_valid_solution(problem, solution, check_boundary_objective=False)
    assert solution.metadata["objective_kind"] == "choice_utility"
    assert solution.objective == pytest.approx(
        model.evaluate(problem, solution.assignment), abs=0.05
    )


def test_cp_bool_cutoffs_share_vertex_school_indicators_across_students():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        overage=0.0,
        shortage=0.0,
    )
    for node in problem.nodes:
        problem.G.nodes[node]["all_prog_students"] = 1.0
        problem.G.nodes[node]["all_prog_capacity"] = 0.0
    problem.cutoff_market = CutoffMarket(
        students=(
            CutoffStudent(
                studentno=1,
                node=1,
                preferences=(100, 200),
                priorities={100: 0, 200: 0},
            ),
            CutoffStudent(
                studentno=2,
                node=1,
                preferences=(100, 200),
                priorities={100: 0, 200: 0},
            ),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 2},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=10,
    )

    solver = get_solver("cp_bool", solve_time_limit=10, workers=1)
    model = cp_model.CpModel()
    x, _ = solver._build_assignment_vars(model, problem)
    solver._add_core_constraints(model, problem, x, {})
    solver._add_cutoff_objective(model, problem, x)

    names = [variable.name for variable in model.Proto().variables]
    assert names.count("same_zone_1_100") == 1
    assert names.count("same_zone_1_200") == 1
    assert names.count("threshold_100_0") == 1
    assert names.count("threshold_200_0") == 1
    assert sum(name.startswith("effective_threshold_") for name in names) == 4

    solution = solver.solve(problem)

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    assert solution.assignment[1] == 1
    assert solution.objective == 0.0
    assert solution.metadata["objective_kind"] == "school_cutoffs"
    assert solution.metadata["same_zone_indicator_count"] == 2
    assert solution.metadata["normalized_school_cutoffs"] == {100: 0.0, 200: 0.0}


def test_cp_bool_cutoffs_keep_citywide_school_accessible_outside_zone():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        fixed={1: 0},
    )
    problem.cutoff_market = CutoffMarket(
        students=(
            CutoffStudent(
                studentno=1,
                node=1,
                preferences=(200,),
                priorities={200: 0},
            ),
        ),
        school_nodes={200: 3},
        school_capacities={200: 0},
        zone_restricted_schools=frozenset(),
        lottery_scale=10,
    )

    solution = get_solver("cp_bool", solve_time_limit=10, workers=1).solve(problem)

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    assert solution.assignment[1] == 0
    assert solution.objective == 1.0
    assert solution.metadata["same_zone_indicator_count"] == 0
    assert solution.metadata["normalized_school_cutoffs"] == {200: 1.0}


def test_cp_int_rejects_cutoff_objective():
    problem = make_grid_problem(2, 2)
    problem.cutoff_market = CutoffMarket(
        students=(),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=10,
    )

    with pytest.raises(ValueError, match="only for cp_bool"):
        get_solver("cp_int", solve_time_limit=1, workers=1).solve(problem)


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


@pytest.mark.skipif("mip" not in available_solvers(), reason="gurobipy not installed")
def test_mip_solver_supports_secondary_objective():
    problem = make_grid_problem(3, 3)
    try:
        solution = get_solver(
            "mip",
            solve_time_limit=10,
            secondary_objective=True,
        ).solve(problem)
    except Exception as exc:  # no usable Gurobi license in this environment
        if "gurobi" in type(exc).__module__.lower():
            pytest.skip(f"Gurobi unavailable: {exc}")
        raise

    assert solution.status in ("OPTIMAL", "FEASIBLE")
    assert_valid_solution(problem, solution, check_boundary_objective=False)


@pytest.mark.skipif("mip" not in available_solvers(), reason="gurobipy not installed")
def test_mip_solver_forbids_zone_without_closer_neighbor():
    problem = make_grid_problem(3, 3, candidates={4: {0}})
    problem.G.graph["closer_neighbors"][4][100] = frozenset()
    try:
        solution = get_solver("mip", solve_time_limit=10).solve(problem)
    except Exception as exc:  # no usable Gurobi license in this environment
        if "gurobi" in type(exc).__module__.lower():
            pytest.skip(f"Gurobi unavailable: {exc}")
        raise

    assert solution.status == "INFEASIBLE"


@pytest.mark.skipif("mip" not in available_solvers(), reason="gurobipy not installed")
def test_mip_solver_saves_progress(tmp_path):
    problem = make_grid_problem(3, 3)
    try:
        solution = get_solver(
            "mip",
            solve_time_limit=10,
            save_solver_progress=True,
            output_dir=str(tmp_path),
        ).solve(problem)
    except Exception as exc:  # no usable Gurobi license in this environment
        if "gurobi" in type(exc).__module__.lower():
            pytest.skip(f"Gurobi unavailable: {exc}")
        raise

    assert solution.solver_progress
    solution.save(str(tmp_path))
    expected_log = os.path.join(
        "solver_progress", "solver_00_BlockGroup_0_mip", "progress.jsonl"
    )
    assert (tmp_path / expected_log).exists()

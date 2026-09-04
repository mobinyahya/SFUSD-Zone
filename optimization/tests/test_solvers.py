"""Data-free, end-to-end solver tests on a synthetic grid problem."""

import json
import os

import pytest
from ortools.sat.python import cp_model

from choice.objective import ChoiceCut, ChoiceObjective, ChoiceTerm
from optimization.config import OptimizationConfig
from optimization.data import contiguity
from optimization.data.edge_weights import BOUNDARY_WEIGHT_ATTR
from optimization.solvers import get_solver
from optimization.solvers.balance import balance_constraints
from optimization.solvers.base import available_solvers
from optimization.tests.solver_contract import (
    assert_valid_solution,
    solve_contract_problem,
)
from optimization.tests.synthetic import make_grid_problem, make_solver_contract_problem


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


@pytest.mark.parametrize(
    ("name", "secondary_objective"),
    [
        ("cp_int", False),
        ("cp_bool", False),
        ("cp_bool", True),
        ("mip", False),
        ("mip", True),
    ],
)
def test_solvers_use_weighted_boundary_objective(name, secondary_objective):
    problem = make_solver_contract_problem(weight_edges=True)
    problem.G.graph["weight_edges"] = True
    for edge, weight in zip(problem.G.edges(), [3, 11, 5]):
        problem.G.edges[edge][BOUNDARY_WEIGHT_ATTR] = weight

    solution = solve_contract_problem(
        name,
        problem,
        secondary_objective=secondary_objective,
    )

    assert_valid_solution(problem, solution)
    assert solution.objective == 11
    assert solution.metadata["objective_kind"] == "weighted_boundary_length"
    assert solution.metadata["objective_unit"] == "meter"


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


def test_cpsat_enumeration_forces_single_worker():
    solver = get_solver("cp_int", workers=8)
    cp_solver = cp_model.CpSolver()

    solver._configure_solver_parameters(cp_solver, enumerate_solutions=True)

    assert cp_solver.parameters.enumerate_all_solutions is True
    assert cp_solver.parameters.num_search_workers == 1


@pytest.mark.parametrize("name", ["cp_int", "cp_bool"])
def test_cpsat_enumerates_distinct_solutions_without_an_objective(name, monkeypatch):
    problem = make_grid_problem(3, 3)
    solver = get_solver(name, solve_time_limit=10, workers=8, seed=7)

    def fail_objective(*args):
        pytest.fail("Enumeration must not build an objective")

    monkeypatch.setattr(solver, "_add_model_objective", fail_objective)
    solutions = solver.enumerate_solutions(problem, 5)

    assert len(solutions) == 5
    assert (
        len({tuple(sorted(solution.assignment.items())) for solution in solutions}) == 5
    )
    for index, solution in enumerate(solutions):
        assert_valid_solution(problem, solution, check_boundary_objective=False)
        assert solution.objective is None
        assert solution.metadata["objective_kind"] == "none"
        assert solution.metadata["enumerated_solution_index"] == index
        assert solution.metadata["enumerated_solutions_found"] == 5


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


def test_negative_frl_dev_disables_frl_balance_constraint():
    problem = make_grid_problem(3, 3, frl_dev=-1)

    frl = next(
        constraint
        for constraint in balance_constraints(problem)
        if constraint.kind == "frl"
    )

    assert (frl.lower_ratio, frl.upper_ratio) == (None, None)


@pytest.mark.parametrize(
    ("overage", "shortage", "expected_bounds"),
    [
        (-1, -1, (None, None)),
        (-1, 0.2, (0.8, None)),
        (0.8, -1, (None, 1.8)),
    ],
)
def test_negative_capacity_tolerances_disable_corresponding_bounds(
    overage, shortage, expected_bounds
):
    problem = make_grid_problem(
        3,
        3,
        overage=overage,
        shortage=shortage,
    )

    constraints = balance_constraints(problem)
    capacity = next(
        (constraint for constraint in constraints if constraint.kind == "capacity"),
        None,
    )
    if expected_bounds == (None, None):
        assert capacity is None
    else:
        assert capacity is not None
        assert (capacity.lower_ratio, capacity.upper_ratio) == expected_bounds


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


@pytest.mark.parametrize(
    "name", [n for n in ["cp_int", "cp_bool", "mip"] if n in available_solvers()]
)
@pytest.mark.parametrize("aggregate", [False, True])
def test_solvers_agree_on_exact_choice_objective(name, aggregate):
    problem = make_grid_problem(3, 3)
    if aggregate:
        cuts = (
            ChoiceCut(
                node=None,
                constant=10.0,
                terms=(ChoiceTerm(coefficient=20.0, node=1, student_node=0),),
            ),
        )
    else:
        cuts = tuple(
            ChoiceCut(
                node=node,
                constant=5.0,
                terms=(
                    ChoiceTerm(coefficient=10.0, node=(node + 1) % len(problem.nodes)),
                ),
            )
            for node in problem.nodes
        )
    problem.choice_objective = ChoiceObjective(
        cuts=cuts,
        lower_bound=-100.0,
        upper_bound=100.0,
        scale=100,
        aggregate_cuts=aggregate,
    )

    solution = get_solver(name, solve_time_limit=30, workers=1).solve(problem)

    assert solution.status == "OPTIMAL"
    assert_valid_solution(problem, solution, check_boundary_objective=False)
    assert solution.metadata["objective_kind"] == "choice_utility"

    node_count = len(problem.nodes)
    co_zoned = sum(
        1
        for node in problem.nodes
        if solution.assignment[node] == solution.assignment[(node + 1) % node_count]
    )
    if aggregate:
        # One total-utility cut: 10 + 20 * a[0, 1], maximized by co-zoning 0 and 1.
        assert solution.assignment[0] == solution.assignment[1]
        assert solution.objective == pytest.approx(30.0)
    else:
        # Per-node cut n: u[n] <= 5 + 10 * a[n, n+1 mod 9]. Two zones must cut the
        # 9-node cycle an even number of times, so at most 7 pairs co-zone:
        # 7 * 15 + 2 * 5 == 115.
        assert co_zoned == 7
        assert solution.objective == pytest.approx(115.0)


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

"""Data-free, end-to-end solver tests on a synthetic grid problem."""

import json
import os

import pytest
from ortools.sat.python import cp_model

import Zone_Generation.optimization.solvers.recom as recom_module
import Zone_Generation.optimization.solvers.relaxed_recom as relaxed_recom_module
import Zone_Generation.optimization.solvers.short_bursts_recom as short_bursts_recom_module
from Zone_Generation.choice.models import DistanceChoiceModel
from Zone_Generation.choice.objective import ChoiceObjective
from Zone_Generation.Config.Constants import AREA_ETHNICITIES
from Zone_Generation.optimization.config import OptimizationConfig
from Zone_Generation.optimization.data import contiguity
from Zone_Generation.optimization.data.initial_solutions import (
    RECOM_SCHOOL_COUNT_COL,
    _initial_balance_metrics,
    recom_balance_epsilon,
    recom_balance_pop_col,
    recom_balance_target,
    recom_gerrychain_graph,
)
from Zone_Generation.optimization.solvers import get_solver
from Zone_Generation.optimization.solvers.balance import balance_constraints
from Zone_Generation.optimization.solvers.base import available_solvers
from Zone_Generation.optimization.solvers.recom import _Score, _race_penalty_key
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


def _check_recom_valid(problem, solution):
    assert solution.feasible
    assert set(solution.assignment) == set(problem.nodes)
    centroid_nodes = set(problem.centroids)
    for node, zone in solution.assignment.items():
        if node in centroid_nodes:
            assert 0 <= zone < problem.Z
        else:
            assert zone in problem.candidate_zones(node)
    assert solution.is_contiguous()


@pytest.mark.parametrize("name", ["cp_int", "cp_bool"])
def test_cpsat_solvers(name):
    problem = make_grid_problem(3, 3)
    solver = get_solver(name, solve_time_limit=10, workers=1)
    solution = solver.solve(problem)
    assert solution.status in ("OPTIMAL", "FEASIBLE")
    _check_valid(problem, solution)


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


def test_config_accepts_school_recom_balance_metric():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="short_bursts_recom",
        recom_balance_metric="schools",
    )

    solver = config.make_solver()

    assert solver.options["recom_balance_metric"] == "schools"


def test_config_normalizes_num_schools_recom_balance_metric_alias():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="short_bursts_recom",
        recom_balance_metric="num_schools",
    )

    assert config.recom_balance_metric == "schools"


def test_config_requires_time_limit_for_negative_recom_iterations():
    with pytest.raises(ValueError, match="solve_time_limits"):
        OptimizationConfig(
            levels=["BlockGroup_0"],
            solver="recom",
            recom_iterations=-1,
            solve_time_limits=[],
        )


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
    _check_valid(problem, solution)
    assert solution.objective == contiguity.boundary_edges(
        problem.G,
        solution.assignment,
    )


def test_recom_score_ordering_separates_feasibility_from_cut_edges():
    assert _Score(penalty=0.0, boundary=2) < _Score(penalty=0.0, boundary=3)
    assert _Score(penalty=1.0, boundary=100) < _Score(penalty=2.0, boundary=0)
    assert _Score(penalty=0.0, boundary=100) < _Score(penalty=1.0, boundary=0)

    first = _Score(penalty=1.0, boundary=0)
    second = _Score(penalty=1.0, boundary=100)
    assert first <= second
    assert second <= first


@pytest.mark.parametrize("name", ["recom", "relaxed_recom", "short_bursts_recom"])
def test_recom_score_does_not_penalize_swapped_centroid_labels(name):
    problem = make_grid_problem(2, 2, racial_dev=-1, frl_dev=1.0, overage=10.0)
    assignment = {0: 1, 1: 1, 2: 0, 3: 0}
    solver = get_solver(name)

    score = solver._score(problem, assignment)

    assert score.penalty == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["recom", "relaxed_recom", "short_bursts_recom"])
def test_recom_score_ignores_max_distance_candidates(name):
    problem = make_grid_problem(
        2,
        2,
        max_distance=0.1,
        racial_dev=-1,
        frl_dev=1.0,
        overage=10.0,
    )
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    assert not problem.candidate_zones(1)
    assert not problem.candidate_zones(2)

    score = get_solver(name)._score(problem, assignment)

    assert score.penalty == pytest.approx(0.0)
    assert score.components == {}


@pytest.mark.parametrize("name", ["recom", "relaxed_recom", "short_bursts_recom"])
def test_recom_score_keeps_explicit_candidate_penalty(name):
    problem = make_grid_problem(
        2,
        2,
        candidates={1: {1}},
        racial_dev=-1,
        frl_dev=1.0,
        overage=10.0,
    )
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    hard_penalty = float(problem.A + problem.Z + 1) * 1000.0

    score = get_solver(name)._score(problem, assignment)

    assert score.components == {"candidate": hard_penalty}
    assert score.penalty == pytest.approx(hard_penalty)


def test_recom_penalty_coefficients_include_each_constraint_family():
    problem = make_grid_problem(3, 3)
    solver = get_solver("recom")

    context = solver._penalty_context(problem)

    expected = {"shortage", "overage", "frl", "schools"} | {
        _race_penalty_key(ethnicity) for ethnicity in AREA_ETHNICITIES
    }
    assert expected <= set(context.coefficients)
    assert all(context.coefficients[key] > 0 for key in expected)


def test_recom_school_balance_metric_uses_school_counts():
    problem = make_grid_problem(3, 3)

    graph = recom_gerrychain_graph(problem, "schools")

    assert recom_balance_pop_col("schools") == RECOM_SCHOOL_COUNT_COL
    assert recom_balance_target(problem, "schools") == pytest.approx(1.0)
    assert graph.nodes[0][RECOM_SCHOOL_COUNT_COL] == pytest.approx(1.0)
    assert graph.nodes[4][RECOM_SCHOOL_COUNT_COL] == pytest.approx(0.0)


def test_recom_school_balance_metric_tightens_epsilon_to_one_school():
    problem = make_grid_problem(3, 3)
    for node in problem.nodes:
        problem.G.nodes[node]["num_schools"] = 1

    assert recom_balance_epsilon(problem, 0.25, "schools") == pytest.approx(2 / 9)
    assert recom_balance_epsilon(problem, 0.25, "students") == pytest.approx(0.25)


def test_short_bursts_recom_uses_school_metric_for_school_penalty():
    problem = make_grid_problem(3, 3)
    solver = get_solver("short_bursts_recom")

    metric = solver._proposal_balance_metric(
        problem,
        _Score(penalty=1.0, boundary=0, components={"schools": 1.0}),
        "students",
    )

    assert metric == "schools"
    assert (
        solver._proposal_balance_metric(
            problem,
            _Score(penalty=1.0, boundary=0, components={"schools": 1.0}),
            "nodes",
        )
        == "nodes"
    )
    assert (
        solver._proposal_balance_metric(
            problem,
            _Score(penalty=1.0, boundary=0, components={"frl": 1.0}),
            "students",
        )
        == "students"
    )


def test_recom_boundary_repair_can_fix_school_count_violation():
    problem = make_grid_problem(2, 3, racial_dev=-1, frl_dev=1.0, overage=10.0)
    for node in problem.nodes:
        problem.G.nodes[node]["num_schools"] = 1 if node in {0, 1, 3, 4} else 0
    assignment = {0: 0, 1: 0, 3: 0, 4: 0, 2: 1, 5: 1}
    solver = get_solver("recom")
    score = solver._score(problem, assignment)

    repaired, repaired_score, metadata = solver._repair_infeasible_solution(
        problem, assignment, score
    )

    assert score.components == {"schools": pytest.approx(2.0)}
    assert repaired_score.penalty == pytest.approx(0.0)
    assert metadata["repair_success"] is True
    assert contiguity.is_contiguous(problem.G, repaired, problem.centroids)


def test_gerry_chain_initialization_adds_school_multistart():
    problem = make_grid_problem(3, 3)

    assert _initial_balance_metrics(problem, "students") == ["students", "schools"]
    assert _initial_balance_metrics(problem, "schools") == ["schools", "students"]


def test_recom_balance_penalty_uses_target_difference_after_violation():
    problem = make_grid_problem(
        2,
        2,
        shortage=0.2,
        overage=10.0,
        frl_dev=0.1,
        racial_dev=-1,
    )
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    for node in (0, 1):
        problem.G.nodes[node]["ge_capacity"] = 0.75
        problem.G.nodes[node]["FRL"] = 0.35

    solver = get_solver("recom")
    context = solver._penalty_context(problem)
    components = solver._constraint_penalty_components(problem, assignment)

    assert components["shortage"] == pytest.approx(
        (2.0 - 1.5) * context.coefficients["shortage"]
    )
    assert components["frl"] == pytest.approx((1.0 - 0.7) * context.coefficients["frl"])
    assert "overage" not in components


def test_recom_capacity_overage_has_separate_penalty_coefficient():
    problem = make_grid_problem(
        2,
        2,
        shortage=10.0,
        overage=0.2,
        frl_dev=1.0,
        racial_dev=-1,
    )
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    for node in (0, 1):
        problem.G.nodes[node]["ge_capacity"] = 1.25

    solver = get_solver("recom")
    context = solver._penalty_context(problem)
    components = solver._constraint_penalty_components(problem, assignment)

    assert components["overage"] == pytest.approx(
        (2.5 - 2.0) * context.coefficients["overage"]
    )
    assert "shortage" not in components


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
    _check_valid(problem, solution)


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
    _check_valid(problem, solution)


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
    "name,options",
    [
        ("recom", {}),
        ("short_bursts_recom", {"short_bursts_length": 2}),
        ("relaxed_recom", {"relaxed_recom_min_boundary_edges": 0}),
    ],
)
def test_recom_solvers_save_progress_logs(tmp_path, name, options):
    problem = make_grid_problem(3, 3)
    solver = get_solver(
        name,
        solve_time_limit=10,
        recom_iterations=5,
        recom_cut_attempts=25,
        save_solver_logs=True,
        output_dir=str(tmp_path),
        seed=1,
        **options,
    )

    solution = solver.solve(problem)

    expected_log = os.path.join("solver_logs", f"solver_00_BlockGroup_0_{name}.log")
    assert solution.metadata["solver_log_path"] == expected_log
    assert solution.metadata["solver_log_format"] == "jsonl"

    rows = [
        json.loads(line)
        for line in (tmp_path / expected_log).read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["event"] == "initial"
    cut_rows = [row for row in rows if row["event"] == "cut"]
    assert len(cut_rows) == (
        solution.metadata["attempted_moves"] - solution.metadata["proposal_failures"]
    )
    for row in rows:
        assert isinstance(row["timestamp"], float)
        assert row["elapsed_seconds"] >= 0.0
        assert isinstance(row["iteration"], int)
        assert isinstance(row["cut_edges"], int)
        assert isinstance(row["feasible"], bool)
        assert isinstance(row["penalty_components"], dict)
        component_total = sum(
            value for value in row["penalty_components"].values() if value is not None
        )
        assert row["penalty"] == pytest.approx(component_total)
        if "best_penalty_components" in row:
            assert isinstance(row["best_penalty_components"], dict)
    for row in cut_rows:
        assert isinstance(row["accepted"], bool)


@pytest.mark.parametrize(
    "name,options",
    [
        ("recom", {}),
        ("short_bursts_recom", {"short_bursts_length": 2}),
        ("relaxed_recom", {"relaxed_recom_min_boundary_edges": 0}),
    ],
)
def test_recom_solvers_save_incumbent_progress(tmp_path, name, options):
    problem = make_grid_problem(3, 3)
    solver = get_solver(
        name,
        solve_time_limit=10,
        recom_iterations=5,
        recom_cut_attempts=25,
        save_solver_progress=True,
        output_dir=str(tmp_path),
        seed=1,
        **options,
    )

    solution = solver.solve(problem)
    assert solution.solver_progress
    assert "solver_log_path" not in solution.metadata

    solution.save(str(tmp_path))

    expected_dir = os.path.join("solver_progress", f"solver_00_BlockGroup_0_{name}")
    expected_log = os.path.join(expected_dir, "progress.jsonl")
    assert solution.metadata["solver_progress_path"] == expected_log
    assert solution.metadata["solver_progress_count"] == len(solution.solver_progress)

    rows = [
        json.loads(line)
        for line in (tmp_path / expected_log).read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["iteration"] == 0
    objectives = [row["objective"] for row in rows]
    assert all(next_obj < obj for obj, next_obj in zip(objectives, objectives[1:]))
    for row in rows:
        assert (tmp_path / expected_dir / row["assignment_path"]).exists()
        assert (tmp_path / expected_dir / row["area_assignment_path"]).exists()


def test_local_search_stub():
    problem = make_grid_problem(3, 3)
    solution = get_solver("local_search").solve(problem)
    assert solution.status == "STUB"
    _check_valid(problem, solution)


def test_recom_solver():
    problem = make_grid_problem(3, 3)
    solution = get_solver(
        "recom",
        solve_time_limit=1,
        recom_iterations=25,
        seed=1,
    ).solve(problem)

    assert solution.status == "FEASIBLE"
    _check_recom_valid(problem, solution)
    assert solution.metadata["solver"] == "recom"
    assert solution.metadata["hints"] == "gerry_chain"


def test_short_bursts_recom_solver():
    problem = make_grid_problem(3, 3)
    solution = get_solver(
        "short_bursts_recom",
        solve_time_limit=1,
        recom_iterations=25,
        short_bursts_length=5,
        seed=1,
    ).solve(problem)

    assert solution.status == "FEASIBLE"
    _check_recom_valid(problem, solution)
    assert solution.metadata["solver"] == "short_bursts_recom"
    assert solution.metadata["short_bursts_length"] == 5
    assert solution.metadata["completed_bursts"] <= 5


@pytest.mark.parametrize(
    "name,options",
    [
        ("recom", {}),
        ("short_bursts_recom", {"short_bursts_length": 2}),
        (
            "relaxed_recom",
            {"relaxed_recom_min_boundary_edges": 0},
        ),
    ],
)
def test_recom_solvers_require_time_limit_for_negative_iterations(
    name,
    options,
):
    problem = make_grid_problem(3, 3)

    with pytest.raises(ValueError, match="solve_time_limit"):
        get_solver(name, recom_iterations=-1, **options).solve(problem)


@pytest.mark.parametrize(
    "name,module,options",
    [
        ("recom", recom_module, {}),
        ("short_bursts_recom", short_bursts_recom_module, {"short_bursts_length": 2}),
        (
            "relaxed_recom",
            relaxed_recom_module,
            {"relaxed_recom_min_boundary_edges": 0},
        ),
    ],
)
def test_negative_recom_iterations_use_time_limit(
    monkeypatch,
    name,
    module,
    options,
):
    problem = make_grid_problem(3, 3)
    problem.hint = {
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

    class FakeClock:
        def __init__(self):
            self.current = 0.0

        def time(self):
            self.current += 0.2
            return self.current

    monkeypatch.setattr(module, "time", FakeClock())
    solution = get_solver(
        name,
        solve_time_limit=0.5,
        recom_iterations=-1,
        seed=1,
        **options,
    ).solve(problem)

    assert solution.status == "FEASIBLE"
    assert solution.metadata["iterations"] is None
    assert solution.metadata["attempted_moves"] > 0


def test_recom_uses_explicit_hint():
    problem = make_grid_problem(3, 3)
    hint = {
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
    problem.hint = hint

    solution = get_solver(
        "recom",
        solve_time_limit=1,
        recom_iterations=0,
    ).solve(problem)

    assert solution.status == "FEASIBLE"
    assert solution.assignment == hint
    assert solution.metadata["hints"] == "provided"


def test_short_bursts_recom_uses_explicit_hint():
    problem = make_grid_problem(3, 3)
    hint = {
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
    problem.hint = hint

    solution = get_solver(
        "short_bursts_recom",
        solve_time_limit=1,
        recom_iterations=0,
    ).solve(problem)

    assert solution.status == "FEASIBLE"
    assert solution.assignment == hint
    assert solution.metadata["hints"] == "provided"


@pytest.mark.parametrize("name", ["recom", "relaxed_recom", "short_bursts_recom"])
def test_recom_solvers_do_not_force_explicit_hint_centroid_labels(name):
    problem = make_grid_problem(2, 2, racial_dev=-1, frl_dev=1.0, overage=10.0)
    hint = {0: 1, 1: 1, 2: 0, 3: 0}
    problem.hint = hint

    solution = get_solver(
        name,
        solve_time_limit=1,
        recom_iterations=0,
        relaxed_recom_min_boundary_edges=0,
    ).solve(problem)

    assert solution.status == "FEASIBLE"
    assert solution.assignment == hint
    assert solution.metadata["hints"] == "provided"


def test_relaxed_recom_keeps_repaired_initial_assignment_even_if_infeasible():
    problem = make_grid_problem(3, 3, candidates={2: {1}})
    assignment = {
        0: 0,
        1: 0,
        2: 1,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 1,
    }
    solver = get_solver("relaxed_recom")

    prepared = solver._prepare_relaxed_assignment(problem, assignment)

    assert contiguity.is_contiguous(problem.G, prepared, problem.centroids)
    assert prepared[2] == 0
    assert prepared[2] not in problem.candidate_zones(2)
    assert not contiguity.is_contiguous(
        problem.G,
        solver._complete_assignment(problem, prepared),
        problem.centroids,
    )


def test_recom_rejects_choice_objective():
    problem = make_grid_problem(3, 3)
    problem.choice_objective = ChoiceObjective(
        cuts=(),
        lower_bound=-1.0,
        upper_bound=1.0,
        scale=100,
    )

    with pytest.raises(NotImplementedError, match="recom does not support"):
        get_solver("recom").solve(problem)


def test_short_bursts_recom_rejects_choice_objective():
    problem = make_grid_problem(3, 3)
    problem.choice_objective = ChoiceObjective(
        cuts=(),
        lower_bound=-1.0,
        upper_bound=1.0,
        scale=100,
    )

    with pytest.raises(NotImplementedError, match="short_bursts_recom"):
        get_solver("short_bursts_recom").solve(problem)


def test_short_bursts_recom_unknown_when_no_valid_solution():
    problem = make_grid_problem(3, 3, shortage=-0.1)

    solution = get_solver(
        "short_bursts_recom",
        solve_time_limit=1,
        recom_iterations=10,
        short_bursts_length=5,
        seed=1,
    ).solve(problem)

    assert solution.status == "UNKNOWN"
    assert solution.assignment == {}
    assert solution.objective is None
    assert solution.metadata["best_penalty"] > 0


def test_recom_rejects_unknown_hints():
    problem = make_grid_problem(3, 3)

    with pytest.raises(ValueError, match="hints"):
        get_solver("recom", hints="bad").solve(problem)


@pytest.mark.parametrize("name", ["recom", "relaxed_recom", "short_bursts_recom"])
def test_recom_solvers_return_error_for_hints_none(name):
    problem = make_grid_problem(3, 3)

    solution = get_solver(name, hints="none").solve(problem)

    assert solution.status == "ERROR"
    assert solution.assignment == {}
    assert solution.objective is None
    assert solution.metadata["hints"] == "none"
    assert "require hints" in solution.metadata["error_message"]


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


@pytest.mark.skipif("mip" not in available_solvers(), reason="gurobipy not installed")
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
    _check_valid(problem, solution)


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

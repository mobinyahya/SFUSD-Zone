"""Independent objective dispatch uses common costs and feasible domains."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from optimization.data.feasibility import check_zoning
from optimization.data.mid import MidMarket, MidProgram, MidStudent, MidType
from optimization.data.saa import SaaMarket, sample_school_preferences
from optimization.mid_oracle import finite_grid_oracle
from optimization.objectives import (
    METHODS,
    get_objective,
    heuristic_scorer,
    solve_objective,
    supported_objectives,
    validate_method_objective,
)
from optimization.tests.synthetic import make_grid_problem


@pytest.mark.parametrize("method", METHODS)
def test_school_distance_uses_the_same_cost_and_domains_across_methods(method):
    if method == "mip":
        pytest.importorskip("gurobipy")
    problem = make_grid_problem(
        3,
        3,
        fixed={1: 0},
        candidates={7: {1}},
        school_distance_area_weight=0.0,
        school_distance_student_weight=1.0,
        hint={n: 0 if n < 4 else 1 for n in range(9)},
    )
    costs = problem.school_distance_costs()
    solution = solve_objective(
        problem,
        method,
        "school_distance",
        options={
            "recom_iterations": 50,
            "solve_time_limit": 3,
            "workers": 1,
            "short_bursts_length": 5,
            "seed": 17,
            "hints": "none",
        },
    )

    assert solution.feasible
    assert check_zoning(solution.problem, solution.assignment).feasible
    assert solution.assignment[0] == solution.assignment[1] == 0
    assert solution.assignment[7] == solution.assignment[8] == 1
    assert solution.objective == pytest.approx(
        sum(costs[zone, node] for node, zone in solution.assignment.items()), abs=1e-7
    )
    assert solution.metadata["method"] == method
    assert solution.metadata["objective"] == "school_distance"
    assert problem.compactness_objective == "cut_edges"
    assert problem.candidates == {7: {1}}


def test_school_distance_scorer_uses_zone_centroids_with_multiple_schools():
    problem = make_grid_problem(
        2,
        3,
        school_distance_area_weight=0.0,
        school_distance_student_weight=1.0,
    )
    problem.G.nodes[1]["num_schools"] = 1
    assignment = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    costs = problem.school_distance_costs()

    assert heuristic_scorer(problem, "school_distance", {})((assignment,), None) == (
        pytest.approx(sum(costs[z, n] for n, z in assignment.items())),
    )


def _matching_market():
    return MidMarket(
        programs=(
            MidProgram("P0", 100, 2, False, 0),
            MidProgram("P1", 200, 2, False, 3),
        ),
        types=(
            MidType(0, 1, ("P0",), (0,), (1.0,), (100,)),
            MidType(1, 1, ("P1",), (0,), (10.0,), (1000,)),
            MidType(2, 1, ("P0",), (0,), (10.0,), (1000,)),
            MidType(3, 1, ("P1",), (0,), (1.0,), (100,)),
        ),
        student_count=4,
        outside_only_student_count=0,
        utility_student_count=4,
        utility_handling="omit_nonpositive",
    )


@pytest.mark.parametrize(
    "method",
    [
        "cp_bool",
        "cp_int",
        "recom",
        "relaxed_recom",
        "short_bursts",
        "adaptive_short_bursts",
    ],
)
def test_matching_welfare_uses_the_finite_mid_market_without_disabling_constraints(
    monkeypatch, method
):
    market = _matching_market()
    monkeypatch.setattr("optimization.data.mid.build_mid_market", lambda *_: market)
    monkeypatch.setattr(
        "optimization.solvers.burst_objectives.build_mid_market", lambda *_: market
    )
    problem = make_grid_problem(
        2,
        2,
        program_population="All",
        optimization_config=SimpleNamespace(),
        boundary_prop=0.5,
        overage=5.0,
        shortage=0.0,
        hint={0: 0, 1: 0, 2: 1, 3: 1},
    )
    solution = solve_objective(
        problem,
        method,
        "matching_welfare",
        options={
            "mid_lottery_scale": 4,
            "recom_iterations": 50,
            "solve_time_limit": 3,
            "workers": 1,
            "short_bursts_length": 5,
        },
    )

    assert solution.feasible
    assert check_zoning(solution.problem, solution.assignment).feasible
    assert solution.problem.overage == 5.0
    assert solution.problem.shortage == 0.0
    assert solution.problem.boundary_prop == 0.5
    assert solution.metadata["aggregate_capacity_overage_disabled"] is False
    assert solution.metadata["aggregate_capacity_shortage_disabled"] is False
    assert solution.objective == pytest.approx(
        finite_grid_oracle(market, solution.assignment, 4).welfare
    )
    assert solution.metadata["objective"] == "matching_welfare"


def test_integer_joint_mid_matches_boolean_model_with_domains_hard_assignments_and_bands():
    from optimization.solvers.mid import MidCpIntSolver, MidCpSatSolver

    market = _matching_market()
    problem = make_grid_problem(
        2,
        2,
        program_population="All",
        fixed={1: 0},
        candidates={2: {1}},
        overage=0.0,
        shortage=0.0,
        boundary_prop=0.5,
        hint={0: 0, 1: 0, 2: 1, 3: 1},
    )
    boolean = MidCpSatSolver(market, 4, solve_time_limit=3, workers=1).solve(problem)
    integer = MidCpIntSolver(market, 4, solve_time_limit=3, workers=1).solve(problem)

    assert integer.status == boolean.status == "OPTIMAL"
    assert integer.assignment == boolean.assignment == problem.hint
    assert integer.objective == boolean.objective
    assert integer.objective == pytest.approx(
        finite_grid_oracle(market, integer.assignment, 4).welfare
    )
    assert check_zoning(problem, integer.assignment).feasible
    assert integer.metadata["solver"] == "cp_int"
    assert integer.metadata["mid_solver_cutoff_agreement"] is True


def test_unsupported_objective_fails_before_problem_or_solver_loading():
    with pytest.raises(
        ValueError, match="does not support objective 'matching_welfare'"
    ):
        solve_objective(None, "mip", "matching_welfare")
    with pytest.raises(ValueError, match="Unknown method"):
        validate_method_objective("unrecognized", "cut_edges")
    with pytest.raises(ValueError, match="Unknown objective"):
        validate_method_objective("cp_bool", "unrecognized")


def test_welfare_alias_is_explicitly_matching_welfare():
    assert get_objective("welfare") is get_objective("matching_welfare")
    validate_method_objective("cp_bool", "welfare")
    assert "matching_welfare" in supported_objectives("cp_bool")
    assert "matching_welfare" not in supported_objectives("mip")


def test_existing_choice_model_cannot_silently_override_a_pipeline_objective():
    problem = replace(make_grid_problem(), choice_objective=SimpleNamespace())
    with pytest.raises(ValueError, match="choice_objective=None"):
        solve_objective(problem, "cp_bool", "cut_edges")


@pytest.mark.parametrize("method", METHODS[3:])
def test_heuristics_enforce_centroid_neighborhood_ownership(method):
    problem = make_grid_problem(3, 3, hint={n: 0 if n < 4 else 1 for n in range(9)})
    solution = solve_objective(
        problem,
        method,
        "cut_edges",
        options={
            "centroid_neighbor_radius": 1,
            "recom_iterations": 50,
            "solve_time_limit": 2,
        },
    )
    assert solution.feasible
    assert all(solution.assignment[n] == 0 for n in (0, 1, 3))
    assert all(solution.assignment[n] == 1 for n in (5, 7, 8))
    assert problem.candidate_zones(1) == {0, 1}


def test_conflicting_neighborhoods_and_hard_assignments_fail_explicitly():
    with pytest.raises(ValueError, match="centroid-neighborhood"):
        solve_objective(
            make_grid_problem(),
            "recom",
            "cut_edges",
            options={"centroid_neighbor_radius": 2},
        )
    with pytest.raises(ValueError, match="centroid-neighborhood"):
        solve_objective(
            make_grid_problem(fixed={1: 1}),
            "recom",
            "cut_edges",
            options={"centroid_neighbor_radius": 1},
        )
    with pytest.raises(ValueError, match="centroid-neighborhood"):
        solve_objective(make_grid_problem(fixed={0: 1}), "recom", "cut_edges")


@pytest.mark.parametrize("method", METHODS[3:])
def test_heuristics_use_choice_model_welfare_and_cache_revisited_zonings(
    monkeypatch, method
):
    calls = []

    def evaluate(problem, assignment):
        calls.append(tuple(sorted(assignment.items())))
        return float(assignment[1] == 1) * 10 + float(assignment[2] == 0)

    monkeypatch.setattr(
        "choice.models.build_mnl_choice_model",
        lambda *_args, **_kwargs: SimpleNamespace(evaluate=evaluate),
    )
    problem = make_grid_problem(
        2,
        2,
        hint={0: 0, 1: 0, 2: 1, 3: 1},
        optimization_config=SimpleNamespace(
            data_scenario=object(), choice_model_method="max"
        ),
    )
    solution = solve_objective(
        problem,
        method,
        "choice_welfare",
        options={
            "recom_iterations": 100,
            "short_bursts_length": 5,
            "solve_time_limit": 3,
        },
    )
    assert solution.feasible
    assert solution.objective == pytest.approx(
        float(solution.assignment[1] == 1) * 10 + float(solution.assignment[2] == 0)
    )
    assert solution.metadata["choice_utility"] == solution.objective
    assert solution.metadata["choice_model_method"] == "max"
    assert len(calls) == len(set(calls))


@pytest.mark.parametrize("method", METHODS[3:])
def test_heuristics_use_the_same_sampled_stable_assignment_welfare(monkeypatch, method):
    from optimization.saa_oracle import SaaOracle

    market = SaaMarket(
        programs=(MidProgram("A", 100, 1, False, 0), MidProgram("B", 200, 1, False, 3)),
        students=(
            MidStudent(1, ("A", "B"), (0, 0), (2.0, 1.0), (200, 100)),
            MidStudent(2, ("B", "A"), (0, 0), (2.0, 1.0), (200, 100)),
        ),
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )
    monkeypatch.setattr("optimization.data.saa.build_saa_market", lambda *_: market)
    problem = make_grid_problem(
        2,
        2,
        program_population="All",
        hint={0: 0, 1: 0, 2: 1, 3: 1},
        optimization_config=SimpleNamespace(
            saa_num_seeds=2, saa_tie_breaking_method="STB"
        ),
    )
    solution = solve_objective(
        problem,
        method,
        "saa_welfare",
        options={
            "recom_iterations": 50,
            "short_bursts_length": 5,
            "solve_time_limit": 3,
            "seed": 17,
            "workers": 1,
        },
    )
    samples = sample_school_preferences(market, 2, "STB", 17)
    oracles = [
        SaaOracle(market, sample, index, solution.problem, workers=1)
        for index, sample in enumerate(samples)
    ]
    try:
        expected = sum(
            oracle.solve(solution.assignment).welfare for oracle in oracles
        ) / len(oracles)
    finally:
        for oracle in oracles:
            oracle.model.dispose()

    assert solution.feasible
    assert solution.objective == pytest.approx(expected)
    assert solution.metadata["saa_num_seeds"] == 2
    assert solution.metadata["saa_sample_seeds"] == [sample.seed for sample in samples]

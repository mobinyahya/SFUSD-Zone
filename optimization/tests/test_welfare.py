"""Tests for stable finite-grid utilitarian-welfare optimization."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from optimization.data import contiguity
from optimization.problem import CutoffMarket, CutoffStudent
from optimization.solvers import get_solver
from optimization.solvers.budget_lbbd import (
    BudgetSetLbbdSolver,
    submodular_supergradient,
)
from optimization.solvers.welfare import (
    ApproximateWelfareSolver,
    BooleanBudgetWelfareSolver,
    WelfareSolver,
    add_boolean_budget_reification,
    add_finite_grid_recurrence,
)
from optimization.solvers.welfare_decomposition import (
    WelfareDecompositionSolver,
    _WelfareIncumbent,
)
from optimization.tests.synthetic import make_grid_problem
from optimization.welfare_oracle import (
    MAX_EXACT_CP_SAT_OBJECTIVE,
    raw_welfare_upper_bound,
    solve_zoned_welfare,
    validate_welfare_market,
)
from optimization.verify_welfare_scenario import _float_maps_close, _integer_keyed


def _student(studentno, node, preferences, utilities):
    return CutoffStudent(
        studentno,
        node,
        preferences,
        {school: 0 for school in preferences},
        utilities,
    )


@pytest.mark.parametrize("kind", ["addition", "removal"])
def test_submodular_supergradients_bound_every_budget(kind):
    coefficients = (9, 6, 6, 2)
    for reference_mask in range(1 << len(coefficients)):
        constant, slopes = submodular_supergradient(coefficients, reference_mask, kind)
        for target_mask in range(1 << len(coefficients)):
            value = max(
                (
                    coefficient
                    for index, coefficient in enumerate(coefficients)
                    if target_mask & (1 << index)
                ),
                default=0,
            )
            bound = constant + sum(
                slope
                for index, slope in enumerate(slopes)
                if target_mask & (1 << index)
            )
            assert value <= bound
            if target_mask == reference_mask:
                assert value == bound


@pytest.mark.parametrize(
    "target_available, expected_status",
    [(0, cp_model.OPTIMAL), (1, cp_model.INFEASIBLE)],
)
def test_lbbd_interval_capacity_cut_separates_persistent_overload(
    target_available, expected_status
):
    student = _student(1, 0, (100,), {100: 1.0})
    market = CutoffMarket(
        students=(student,),
        school_nodes={100: 1},
        school_capacities={100: 0},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    model = cp_model.CpModel()
    target = model.NewBoolVar("target")
    model.Add(target == target_available)
    master = SimpleNamespace(
        model=model,
        budgets={(0, 100, 0, 1): target},
    )
    interval = SimpleNamespace(
        student=student,
        low=1,
        high=2,
        higher=(),
    )
    solver = BudgetSetLbbdSolver(get_solver("cp_bool", workers=1), utility_scale=100)

    assert solver._add_interval_capacity_cut(master, market, 100, [interval]) == 1
    assert cp_model.CpSolver().Solve(model) == expected_status


def test_lbbd_complete_school_activation_enforces_exact_demand():
    students = (
        _student(1, 0, (100,), {100: 1.0}),
        _student(2, 0, (100,), {100: 1.0}),
    )
    market = CutoffMarket(
        students=students,
        school_nodes={100: 1},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=2,
    )
    model = cp_model.CpModel()
    budgets = {}
    for cell in range(1, 3):
        budget = model.NewBoolVar(f"budget_{cell}")
        model.Add(budget == 1)
        budgets[0, 100, 0, cell] = budget
    master = SimpleNamespace(model=model, budgets=budgets)
    solver = BudgetSetLbbdSolver(get_solver("cp_bool", workers=1), utility_scale=100)

    assert solver._activate_complete_school(master, market, 100) == 1
    assert cp_model.CpSolver().Solve(model) == cp_model.INFEASIBLE
    assert solver._complete_demand_boolean_count == 2


def test_welfare_oracle_integrates_lottery_assignment_mass():
    market = CutoffMarket(
        students=(
            _student(1, 0, (100,), {100: 2.0}),
            _student(2, 0, (100,), {100: 6.0}),
        ),
        school_nodes={100: 1},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )

    result = solve_zoned_welfare(market, {0: 0, 1: 0}, num_zones=1, utility_scale=100)

    assert result.cutoffs.school_cutoffs == {100: 2}
    assert result.assignments == {1: {100: 2}, 2: {100: 2}}
    assert result.outside_option_mass == {1: 2, 2: 2}
    assert result.welfare == pytest.approx(4.0)
    assert result.raw_scaled_welfare == 1600


def test_boolean_budget_model_reifies_qualification_utility_and_demand():
    market = CutoffMarket(
        students=(
            _student(1, 0, (100,), {100: 2.0}),
            _student(2, 0, (100,), {100: 6.0}),
        ),
        school_nodes={100: 1},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    model = cp_model.CpModel()
    formulation = add_boolean_budget_reification(
        model,
        market,
        {(0, 100): model.NewConstant(1)},
        utility_scale=100,
    )
    model.Maximize(formulation.raw_welfare)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.OPTIMAL
    assert solver.Value(formulation.cutoffs[100]) == 2
    assert [
        solver.Value(formulation.qualifications[100, 0, cell]) for cell in range(1, 5)
    ] == [0, 0, 1, 1]
    assert [
        solver.Value(expression)
        for expression in formulation.assignment_measures.values()
    ] == [2]
    assert int(round(solver.ObjectiveValue())) == 1600
    assert formulation.counts == {
        "budget_profile_count": 2,
        "qualification_boolean_count": 4,
        "budget_boolean_count": 4,
        "assignment_measure_count": 1,
        "cell_utility_variable_count": 8,
    }


def test_boolean_budget_model_matches_oracle_with_priority_tiers():
    market = CutoffMarket(
        students=(
            CutoffStudent(
                1,
                0,
                (100, 200),
                {100: 0, 200: 2},
                {100: 5.0, 200: 1.0},
            ),
            CutoffStudent(
                2,
                0,
                (200, 100),
                {200: 0, 100: 1},
                {200: 6.0, 100: 2.0},
            ),
            CutoffStudent(
                3,
                0,
                (100, 200),
                {100: 0, 200: 0},
                {100: 4.0, 200: 3.0},
            ),
        ),
        school_nodes={100: 1, 200: 2},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=3,
    )
    model = cp_model.CpModel()
    same_zone = {
        pair: model.NewConstant(1)
        for pair in {
            (student.node, school)
            for student in market.students
            for school in student.preferences
        }
    }
    formulation = add_boolean_budget_reification(
        model,
        market,
        same_zone,
        utility_scale=100,
    )
    model.Maximize(formulation.raw_welfare)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    oracle = solve_zoned_welfare(
        market,
        {0: 0, 1: 0, 2: 0},
        num_zones=1,
        utility_scale=100,
    )

    assert status == cp_model.OPTIMAL
    assert int(round(solver.ObjectiveValue())) == oracle.raw_scaled_welfare


def test_approximate_welfare_recurrence_enforces_priority_cutoff_off_by_one():
    market = CutoffMarket(
        students=(
            CutoffStudent(1, 0, (100,), {100: 1}, {100: 1.0}),
            CutoffStudent(2, 0, (100,), {100: 0}, {100: 10.0}),
            CutoffStudent(3, 0, (100,), {100: 0}, {100: 9.0}),
        ),
        school_nodes={100: 1},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    model = cp_model.CpModel()
    cutoffs, raw_welfare = add_finite_grid_recurrence(
        model,
        market,
        {(0, 100): model.NewConstant(1)},
        utility_scale=1,
    )
    model.Maximize(raw_welfare)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.OPTIMAL
    # A priority-zero applicant's lottery scores are 0, 1, 2, 3. A cutoff of
    # four rejects all four cells while admitting all priority-one cells.
    assert solver.Value(cutoffs[100]) == 4
    assert int(round(solver.ObjectiveValue())) == 4
    variable_names = [variable.name for variable in model.Proto().variables]
    assert variable_names.count("welfare_threshold_100_0") == 1
    assert variable_names.count("welfare_threshold_100_1") == 1


def test_approximate_welfare_uses_cumulative_preference_rejections():
    market = CutoffMarket(
        students=(
            CutoffStudent(
                1,
                0,
                (100, 200),
                {100: 0, 200: 0},
                {100: 5.0, 200: 3.0},
            ),
        ),
        school_nodes={100: 1, 200: 2},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    model = cp_model.CpModel()
    cutoffs, raw_welfare = add_finite_grid_recurrence(
        model,
        market,
        {
            (0, 100): model.NewConstant(1),
            (0, 200): model.NewConstant(1),
        },
        utility_scale=1,
    )
    model.Add(cutoffs[100] == 2)
    model.Add(cutoffs[200] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.OPTIMAL
    # r0=4, r1=min(4,2)=2, r2=min(2,1)=1, so d100=2 and d200=1.
    assert solver.Value(raw_welfare) == 2 * 5 + 1 * 3


def test_approximate_welfare_jointly_optimizes_feasible_zoning_and_access():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        hint={0: 0, 1: 0, 2: 1, 3: 1},
    )
    problem.cutoff_market = CutoffMarket(
        students=(
            CutoffStudent(
                1,
                1,
                (200, 100),
                {200: 0, 100: 0},
                {200: 10.0, 100: 1.0},
            ),
            CutoffStudent(
                2,
                1,
                (200, 100),
                {200: 0, 100: 0},
                {200: 8.0, 100: 2.0},
            ),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 2, 200: 2},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    zoning_solver = get_solver(
        "cp_bool",
        solve_time_limit=10,
        workers=1,
        seed=42,
    )

    solution = ApproximateWelfareSolver(
        zoning_solver,
        utility_scale=100,
    ).solve(problem)

    assert solution.status == "OPTIMAL"
    assert solution.assignment[1] == solution.assignment[3] == 1
    assert solution.is_contiguous()
    assert solution.objective == pytest.approx(18.0)
    assert solution.metadata["objective_kind"] == "approximate_welfare"
    assert solution.metadata["same_zone_indicator_count"] == 2
    assert solution.metadata["rejection_threshold_count"] == 2
    assert solution.metadata["demand_expression_count"] == 4


def test_serialized_school_cutoff_keys_are_normalized():
    stored = {"100": 1.25, "200": 2.5}

    assert _integer_keyed(stored) == {100: 1.25, 200: 2.5}
    assert _float_maps_close({100: 1.25, 200: 2.5}, _integer_keyed(stored))


@pytest.mark.parametrize(
    "student, outside, message",
    [
        (
            _student(1, 0, (100, 200), {100: 1.0, 200: 2.0}),
            0.0,
            "nonincreasing",
        ),
        (_student(1, 0, (100,), {100: -1.0}), 0.0, "outside option"),
        (_student(1, 0, (100,), {100: 2.0}), 1.0, "utility zero"),
    ],
)
def test_welfare_market_rejects_invalid_cardinal_preferences(student, outside, message):
    market = CutoffMarket(
        students=(student,),
        school_nodes={
            school: index for index, school in enumerate(student.preferences)
        },
        school_capacities={school: 1 for school in student.preferences},
        zone_restricted_schools=frozenset(student.preferences),
        lottery_scale=4,
        outside_option_utility=outside,
    )

    with pytest.raises(ValueError, match=message):
        validate_welfare_market(market)


def test_welfare_market_rejects_inexact_cp_sat_objective_range():
    utility = (MAX_EXACT_CP_SAT_OBJECTIVE + 1) / 4
    market = CutoffMarket(
        students=(_student(1, 0, (100,), {100: utility}),),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )

    with pytest.raises(ValueError, match="exact CP-SAT reporting range"):
        validate_welfare_market(market, utility_scale=1)


def test_gurobi_heuristic_bound_cannot_cap_integer_master():
    pytest.importorskip("gurobipy")
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    utility = float(MAX_EXACT_CP_SAT_OBJECTIVE - 2048)
    problem.cutoff_market = CutoffMarket(
        students=(_student(1, 1, (100,), {100: utility}),),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=1,
    )
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    result = solve_zoned_welfare(
        problem.cutoff_market,
        assignment,
        num_zones=2,
        utility_scale=1,
    )
    solver = WelfareDecompositionSolver(
        get_solver("cp_bool", solve_time_limit=1, workers=1), utility_scale=1
    )

    _, bound, _ = solver._assignment_relaxation_mip(
        problem, _WelfareIncumbent(assignment, result), 1.0
    )

    assert bound == raw_welfare_upper_bound(problem.cutoff_market, 1)


def test_external_direct_cap_cannot_certify_global_optimum():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = CutoffMarket(
        students=(_student(1, 1, (100,), {100: 5.0}),),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    problem.hint = {0: 0, 1: 0, 2: 1, 3: 1}
    incumbent = solve_zoned_welfare(
        problem.cutoff_market,
        problem.hint,
        num_zones=problem.Z,
        utility_scale=10,
    )
    zoning_solver = get_solver(
        "cp_bool",
        solve_time_limit=10,
        workers=1,
        seed=42,
        welfare_raw_upper_bound=incumbent.raw_scaled_welfare,
    )

    solution = WelfareSolver(zoning_solver, utility_scale=10).solve(problem)

    assert solution.metadata["configured_raw_upper_bound"] == (
        incumbent.raw_scaled_welfare
    )
    assert not solution.metadata["global_optimum_certified"]
    assert solution.status == "FEASIBLE"


def test_gurobi_transport_bound_remains_diagnostic():
    pytest.importorskip("gurobipy")
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = CutoffMarket(
        students=(_student(1, 1, (100,), {100: 5.0}),),
        school_nodes={100: 0},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    result = solve_zoned_welfare(
        problem.cutoff_market,
        assignment,
        num_zones=problem.Z,
        utility_scale=10,
    )
    solver = WelfareDecompositionSolver(
        get_solver("cp_bool", solve_time_limit=1, workers=1),
        utility_scale=10,
    )

    _, returned_bound, _ = solver._assignment_transport_mip(
        problem,
        _WelfareIncumbent(assignment, result),
        1.0,
    )

    assert returned_bound == solver._global_capacity_upper_bound(problem.cutoff_market)
    assert solver._transport_mip_details["proof_grade_raw_upper_bound"] == (
        returned_bound
    )


def test_welfare_decomposition_allows_zero_recom_time_when_disabled():
    solver = WelfareDecompositionSolver(
        get_solver("cp_bool", workers=1),
        utility_scale=100,
        recom_seed_runs=0,
        recom_time_limit=0,
    )

    assert solver.recom_time_limit == 0.0


def test_welfare_decomposition_rejects_zero_recom_time_when_enabled():
    with pytest.raises(ValueError, match="recom_time_limit"):
        WelfareDecompositionSolver(
            get_solver("cp_bool", workers=1),
            utility_scale=100,
            recom_seed_runs=1,
            recom_time_limit=0,
        )


def test_direct_demand_decomposition_keeps_lns_with_hints():
    decomposition = WelfareDecompositionSolver(
        get_solver("cp_bool", workers=5, hints="voronoi"),
        utility_scale=100,
        theta_enabled=False,
    )
    solver = decomposition._new_solver(1.0, 0)

    assert solver.parameters.num_search_workers == 5
    assert solver.parameters.use_lns
    assert solver.parameters.use_rins_lns
    assert solver.parameters.use_lb_relax_lns


@pytest.mark.parametrize("theta_enabled", [True, False])
def test_welfare_decomposition_matches_exhaustive_tiny_zonings(theta_enabled):
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = CutoffMarket(
        students=(
            _student(1, 0, (100, 200), {100: 2.0, 200: 0.5}),
            _student(2, 1, (100, 200), {100: 6.0, 200: 1.0}),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    closer = contiguity.closer_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    brute_force = []
    for zone_1, zone_2 in itertools.product(range(2), repeat=2):
        assignment = {0: 0, 1: zone_1, 2: zone_2, 3: 1}
        if any(
            node != problem.centroids[zone]
            and not any(assignment[neighbor] == zone for neighbor in closer[node, zone])
            for node, zone in assignment.items()
        ):
            continue
        brute_force.append(
            solve_zoned_welfare(
                problem.cutoff_market,
                assignment,
                num_zones=2,
                utility_scale=100,
            ).raw_scaled_welfare
        )

    zoning_solver = get_solver("cp_bool", solve_time_limit=10, workers=1, seed=42)
    solution = WelfareDecompositionSolver(
        zoning_solver,
        utility_scale=100,
        assignment_relaxation_enabled=False,
        theta_enabled=theta_enabled,
    ).solve(problem)

    assert solution.status == "OPTIMAL"
    assert solution.metadata["raw_scaled_welfare"] == max(brute_force)
    assert solution.metadata["global_optimum_certified"]
    assert solution.metadata["grid_minimal"]
    assert solution.metadata["stable"]
    assert solution.metadata["welfare_decomposition_theta_enabled"] is theta_enabled
    assert solution.metadata["welfare_decomposition_round_time_limit"] == 180.0
    assert solution.metadata["welfare_assignment_relaxation_enabled"] is False
    assert solution.metadata["welfare_submodular_access_start_enabled"] is False
    assert (
        solution.metadata["welfare_adjacent_zone_subset_improvement_enabled"] is False
    )
    assert solution.metadata["assignment_relaxation_status"] == "DISABLED"
    assert solution.metadata["assignment_relaxation_raw_upper_bound"] is None
    assert solution.metadata["decomposition_pressure_starts_enabled"] is False
    assert solution.metadata["decomposition_local_moves_enabled"] is False
    assert solution.metadata["zoned_recom_seed_runs"] == 0
    assert solution.metadata["welfare_recom_time_limit"] == 600.0
    assert solution.metadata["welfare_branch_price_enabled"] is False
    assert solution.metadata["welfare_branch_price_time_limit"] == 45.0
    assert solution.metadata["decomposition_generate_assigned_pairs"] is True
    assert solution.metadata["revealed_preference_cut_count"] == 0
    assert solution.metadata["conditional_demand_pair_count"] > 0
    assert solution.metadata["conditional_demand_profile_count"] > 0
    assert solution.metadata["conditional_demand_capacity_constraint_count"] > 0
    if theta_enabled:
        assert solution.metadata["welfare_cut_count"] > 0
        assert solution.metadata["welfare_prefix_variable_count"] > 0
        assert solution.metadata["direct_demand_objective_variable_count"] == 0
    else:
        assert solution.metadata["welfare_cut_count"] == 0
        assert solution.metadata["welfare_prefix_depth"] == 0
        assert solution.metadata["welfare_prefix_variable_count"] == 0
        assert solution.metadata["direct_demand_objective_variable_count"] > 0
        assert solution.metadata["direct_demand_complete_pair_hints_enabled"] is True
    heuristic_kinds = {row["kind"] for row in solution.metadata["heuristic_candidates"]}
    assert "recom_welfare_search" not in heuristic_kinds
    assert "branch_price_root" not in heuristic_kinds
    assert "submodular_access_start" not in heuristic_kinds
    assert "exact_subset_welfare" not in heuristic_kinds
    assert "pressure" not in heuristic_kinds
    assert "refined_lexicographic_pressure" not in heuristic_kinds
    evaluated_rounds = [
        row
        for row in solution.metadata["decomposition_rounds"]
        if row["master_objective"] is not None
    ]
    assert evaluated_rounds
    assert "zoning_no_good_count" not in solution.metadata
    for row in evaluated_rounds:
        assert "zoning_no_goods_added" not in row
        assert row["master_oracle_welfare_gap"] == (
            row["master_objective"] - row["oracle_candidate_welfare"]
        )


def test_direct_demand_candidate_activates_zero_demand_utility_gap():
    student = _student(1, 0, (100,), {100: 5.0})
    market = CutoffMarket(
        students=(student,),
        school_nodes={100: 1},
        school_capacities={100: 1},
        zone_restricted_schools=frozenset({100}),
        lottery_scale=4,
    )
    problem = make_grid_problem(1, 2, population_type="All")
    problem.cutoff_market = market
    decomposition = WelfareDecompositionSolver(
        get_solver("cp_bool", workers=1),
        utility_scale=100,
        theta_enabled=False,
    )
    assignment = {0: 0, 1: 0}
    cutoffs = {100: 4}

    optimistic, exact, gap_keys = decomposition._evaluate_direct_demand_candidate(
        problem,
        market,
        assignment,
        cutoffs,
        {},
    )

    expected_key = decomposition._conditional_profile_key(student, 100, ())
    assert optimistic == 2000
    assert exact == 0
    assert gap_keys == {expected_key}
    assert decomposition._evaluate_direct_demand_candidate(
        problem,
        market,
        assignment,
        cutoffs,
        {expected_key: (student, 100, ())},
    ) == (0, 0, set())


def test_direct_demand_master_bounds_exact_welfare_for_every_active_subset():
    students = (
        _student(1, 0, (100, 200), {100: 6.0, 200: 1.0}),
        _student(2, 0, (100, 200), {100: 4.0, 200: 3.0}),
    )
    market = CutoffMarket(
        students=students,
        school_nodes={100: 0, 200: 0},
        school_capacities={100: 2, 200: 2},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=2,
    )
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = market
    assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    incumbent = _WelfareIncumbent(
        assignment,
        solve_zoned_welfare(
            market,
            assignment,
            num_zones=2,
            utility_scale=100,
        ),
    )
    template = WelfareDecompositionSolver(
        get_solver("cp_bool", workers=1, hints="none"),
        utility_scale=100,
        theta_enabled=False,
    )
    profile_counts, representatives = template._conditional_profile_data(market)
    profile_keys = sorted(representatives)

    for active_mask in range(1 << len(profile_keys)):
        active_profiles = {
            key: representatives[key]
            for index, key in enumerate(profile_keys)
            if active_mask & (1 << index)
        }
        for cutoff_100, cutoff_200 in itertools.product(range(3), repeat=2):
            decomposition = WelfareDecompositionSolver(
                get_solver("cp_bool", workers=1, hints="none"),
                utility_scale=100,
                theta_enabled=False,
            )
            model, x, cutoffs, _objective = decomposition._build_direct_demand_master(
                problem,
                market,
                incumbent,
                active_profiles,
                profile_counts,
                representatives,
                max_cutoff=2,
                raw_upper_bound=raw_welfare_upper_bound(market, 100),
            )
            assert not model.Proto().solution_hint.vars
            if active_profiles:
                assert decomposition._conditional_capacity_constraint_count > 0
            for (zone, node), variable in x.items():
                model.Add(variable == int(assignment[node] == zone))
            model.Add(cutoffs[100] == cutoff_100)
            model.Add(cutoffs[200] == cutoff_200)

            cp_solver = cp_model.CpSolver()
            assert cp_solver.Solve(model) == cp_model.OPTIMAL
            optimistic, exact, _gap_keys = (
                decomposition._evaluate_direct_demand_candidate(
                    problem,
                    market,
                    assignment,
                    {100: cutoff_100, 200: cutoff_200},
                    active_profiles,
                )
            )
            assert int(round(cp_solver.ObjectiveValue())) == optimistic
            assert optimistic >= exact
            if len(active_profiles) == len(profile_keys):
                assert optimistic == exact
            for key, demand in decomposition._conditional_demand_vars.items():
                student, school, higher = representatives[key]
                expected = decomposition._fixed_profile_demand(
                    problem,
                    market,
                    assignment,
                    {100: cutoff_100, 200: cutoff_200},
                    student,
                    school,
                    higher,
                )
                assert cp_solver.Value(demand) == expected


def test_boolean_budget_solver_matches_exhaustive_tiny_zonings():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = CutoffMarket(
        students=(
            _student(1, 0, (100, 200), {100: 2.0, 200: 0.5}),
            _student(2, 1, (100, 200), {100: 6.0, 200: 1.0}),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    closer = contiguity.closer_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    brute_force = []
    for zone_1, zone_2 in itertools.product(range(2), repeat=2):
        assignment = {0: 0, 1: zone_1, 2: zone_2, 3: 1}
        if any(
            node != problem.centroids[zone]
            and not any(assignment[neighbor] == zone for neighbor in closer[node, zone])
            for node, zone in assignment.items()
        ):
            continue
        brute_force.append(
            solve_zoned_welfare(
                problem.cutoff_market,
                assignment,
                num_zones=2,
                utility_scale=100,
            ).raw_scaled_welfare
        )

    problem.hint = {0: 0, 1: 0, 2: 1, 3: 1}
    zoning_solver = get_solver("cp_bool", solve_time_limit=10, workers=1, seed=42)
    solution = BooleanBudgetWelfareSolver(zoning_solver, utility_scale=100).solve(
        problem
    )

    assert solution.status == "OPTIMAL"
    assert solution.metadata["raw_scaled_welfare"] == max(brute_force)
    assert solution.metadata["global_optimum_certified"]
    assert solution.metadata["finite_grid_formulation"] == (
        "priority_contingent_budget_sets"
    )
    assert solution.metadata["qualification_boolean_count"] > 0
    assert solution.metadata["budget_boolean_count"] > 0
    assert solution.metadata["assignment_measure_count"] > 0


def test_budget_lbbd_matches_exhaustive_tiny_zonings():
    problem = make_grid_problem(
        2,
        2,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    problem.cutoff_market = CutoffMarket(
        students=(
            _student(1, 0, (100, 200), {100: 2.0, 200: 0.5}),
            _student(2, 1, (100, 200), {100: 6.0, 200: 1.0}),
        ),
        school_nodes={100: 0, 200: 3},
        school_capacities={100: 1, 200: 1},
        zone_restricted_schools=frozenset({100, 200}),
        lottery_scale=4,
    )
    closer = contiguity.closer_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    brute_force = []
    for zone_1, zone_2 in itertools.product(range(2), repeat=2):
        assignment = {0: 0, 1: zone_1, 2: zone_2, 3: 1}
        if any(
            node != problem.centroids[zone]
            and not any(assignment[neighbor] == zone for neighbor in closer[node, zone])
            for node, zone in assignment.items()
        ):
            continue
        brute_force.append(
            solve_zoned_welfare(
                problem.cutoff_market,
                assignment,
                num_zones=2,
                utility_scale=100,
            ).raw_scaled_welfare
        )

    problem.hint = {0: 0, 1: 0, 2: 1, 3: 1}
    zoning_solver = get_solver("cp_bool", solve_time_limit=30, workers=1, seed=42)
    solution = BudgetSetLbbdSolver(zoning_solver, utility_scale=100).solve(problem)

    assert solution.status == "OPTIMAL"
    assert solution.metadata["raw_scaled_welfare"] == max(brute_force)
    assert solution.metadata["global_optimum_certified"]
    assert solution.metadata["submodular_utility_cut_count"] > 0
    assert solution.metadata["decomposition_rounds"]

"""Focused tests for MID market compression, cutoffs, and joint solving."""

import math

import pytest

from optimization.config import OptimizationConfig
from optimization.data.mid import (
    MidMarket,
    MidProgram,
    MidType,
    build_mid_market,
    compress_mid_students,
    make_mid_student,
    preprocess_mid_market,
)
from optimization.mid_oracle import (
    continuum_oracle,
    evaluate_cutoffs,
    finite_grid_oracle,
    separate_mid_types,
)
from optimization.solvers.mid import MidCpSatSolver
from optimization.tests.synthetic import make_grid_problem


def test_mid_utility_handling_and_type_compression():
    omitted = make_mid_student(
        4,
        ["negative", "best", "zero"],
        [-2.0, 0.004, 0.0],
        [0, 2, 1],
        "omit_nonpositive",
    )
    exponentiated = make_mid_student(
        4,
        ["low", "high"],
        [-2.0, 1.0],
        [0, 1],
        "exponentiate",
    )
    underflow = make_mid_student(
        4,
        ["high", "very-low"],
        [0.0, -1000.0],
        [0, 0],
        "exponentiate",
    )

    assert omitted.programs == ("best",)
    assert omitted.scaled_utilities == (1,)
    assert exponentiated.programs == ("high", "low")
    assert exponentiated.utilities == pytest.approx((1.0, math.exp(-3)))
    assert underflow.programs == ("high", "very-low")
    assert underflow.utilities[1] > 0

    compressed = compress_mid_students([omitted, omitted])
    assert len(compressed) == 1
    assert compressed[0].count == 2
    assert compressed[0].utility_sums == pytest.approx((0.008,))
    assert compressed[0].scaled_utility_sums == (2,)


def test_mid_oracles_return_least_partial_cutoffs():
    market = MidMarket(
        programs=(
            MidProgram("A", 1, 1, True, None),
            MidProgram("B", 2, 1, True, None),
        ),
        types=(
            MidType(
                node=0,
                count=2,
                programs=("A", "B"),
                priorities=(0, 0),
                utility_sums=(4.0, 2.0),
                scaled_utility_sums=(400, 200),
            ),
        ),
        student_count=2,
        outside_only_student_count=0,
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )

    finite = finite_grid_oracle(market, {0: 0}, 20)
    continuous = continuum_oracle(market, {0: 0})

    assert finite.cutoffs == {"A": 10.0, "B": 0.0}
    assert finite.demands == {"A": 1.0, "B": 1.0}
    assert finite.assignment_masses == ((10.0, 10.0),)
    assert finite.welfare == 3.0
    assert finite.fixed_point_value == 6000
    assert finite.type_fixed_point_values == (6000,)
    assert finite.stable and finite.minimal
    assert continuous.cutoffs["A"] == pytest.approx(0.5, abs=1e-7)
    assert continuous.welfare == pytest.approx(3.0)
    assert continuous.stable and continuous.minimal


def test_finite_oracle_compares_capacity_in_exact_integer_mass():
    market = MidMarket(
        programs=(MidProgram("A", 1, 1, True, None),),
        types=tuple(
            MidType(node, 1, ("A",), (0,), (1.0,), (100,)) for node in range(20)
        ),
        student_count=20,
        outside_only_student_count=0,
        utility_student_count=20,
        utility_handling="omit_nonpositive",
    )

    result = finite_grid_oracle(market, {node: 0 for node in range(20)}, 20)

    assert result.cutoffs["A"] == 19
    assert result.demand_masses["A"] == 20
    assert result.demands["A"] == 1
    assert result.welfare == 1


def test_finite_oracle_preserves_cutoffs_above_float_integer_precision():
    market = MidMarket(
        programs=(MidProgram("A", 1, 0, True, None),),
        types=(MidType(0, 1, ("A",), (0,), (1.0,), (100,)),),
        student_count=1,
        outside_only_student_count=0,
        utility_student_count=1,
        utility_handling="omit_nonpositive",
    )
    scale = 2**53 + 1

    result = finite_grid_oracle(market, {0: 0}, scale)

    assert result.cutoffs["A"] == scale
    assert isinstance(result.cutoffs["A"], int)
    assert result.stable and result.minimal


def test_mid_oracle_couples_citywide_capacity_with_restricted_access():
    market = MidMarket(
        programs=(
            MidProgram("restricted", 1, 1, False, 0),
            MidProgram("citywide", 2, 1, True, None),
        ),
        types=(
            MidType(
                0,
                1,
                ("restricted", "citywide"),
                (0, 0),
                (2.0, 1.0),
                (200, 100),
            ),
            MidType(
                1,
                1,
                ("restricted", "citywide"),
                (0, 0),
                (2.0, 1.0),
                (200, 100),
            ),
        ),
        student_count=2,
        outside_only_student_count=0,
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )

    result = finite_grid_oracle(market, {0: 0, 1: 1}, 20)

    assert result.demands == {"restricted": 1, "citywide": 1}
    assert result.outside_mass == 0
    assert result.welfare == 3


def test_mid_solver_optimizes_restricted_program_welfare():
    problem = make_grid_problem(
        2,
        2,
        program_population="All",
        overage=-1,
        shortage=-1,
    )
    problem.hint = {0: 0, 1: 0, 2: 1, 3: 1}
    market = MidMarket(
        programs=(
            MidProgram("A", 100, 1, False, 0),
            MidProgram("B", 200, 1, False, 3),
        ),
        types=(
            MidType(1, 1, ("A", "B"), (0, 0), (2.0, 1.0), (200, 100)),
            MidType(2, 1, ("B", "A"), (0, 0), (2.0, 1.0), (200, 100)),
        ),
        student_count=2,
        outside_only_student_count=0,
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )

    solution = MidCpSatSolver(market, 20, solve_time_limit=10, workers=1).solve(problem)

    assert solution.status == "OPTIMAL"
    assert solution.assignment == {0: 0, 1: 0, 2: 1, 3: 1}
    assert solution.objective == 4.0
    assert solution.metadata["mid_access_pair_count"] == 4
    assert solution.metadata["mid_finite_grid_stable"] is True
    assert solution.metadata["aggregate_capacity_overage_disabled"] is True
    assert solution.metadata["mid_objective_upper_bound"] == 8000
    assert (
        solution.metadata["mid_model_hint_count"]
        == solution.metadata["mid_model_variable_count"]
    )


def test_mid_preprocessing_removes_unusable_alternatives():
    problem = make_grid_problem(
        2,
        2,
        candidates={1: {1}, 2: {0}},
        program_population="All",
        overage=-1,
        shortage=-1,
    )
    market = MidMarket(
        programs=(
            MidProgram("zero", 100, 0, True, None),
            MidProgram("restricted", 100, 2, False, 0),
            MidProgram("citywide", 200, 2, True, None),
        ),
        types=(
            MidType(
                1,
                1,
                ("zero", "restricted", "citywide"),
                (0, 0, 0),
                (3.0, 2.0, 1.0),
                (300, 200, 100),
            ),
            MidType(
                2,
                1,
                ("restricted", "citywide"),
                (0, 0),
                (2.0, 1.0),
                (200, 100),
            ),
        ),
        student_count=2,
        outside_only_student_count=0,
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )

    prepared = preprocess_mid_market(market, problem)

    assert [program.program_id for program in prepared.programs] == [
        "restricted",
        "citywide",
    ]
    assert prepared.types[0].programs == ("citywide",)
    assert prepared.types[0].utility_sums == (1.0,)
    assert prepared.types[1].programs == ("restricted", "citywide")
    assert prepared.preference_count == 3


def test_mid_threshold_domain_can_exceed_lottery_scale():
    problem = make_grid_problem(
        2,
        2,
        program_population="All",
        overage=-1,
        shortage=-1,
    )
    market = MidMarket(
        programs=(MidProgram("A", 100, 1, True, None),),
        types=(
            MidType(1, 1, ("A",), (2,), (1.0,), (100,)),
            MidType(2, 1, ("A",), (2,), (1.0,), (100,)),
        ),
        student_count=2,
        outside_only_student_count=0,
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )

    solution = MidCpSatSolver(market, 20, solve_time_limit=10, workers=1).solve(problem)

    assert solution.status == "OPTIMAL"
    assert solution.metadata["mid_solver_cutoffs"]["A"] == 50
    assert solution.metadata["mid_finite_grid_outside_mass"] == 1.0
    assert solution.metadata["mid_solver_cutoff_agreement"] is True


def test_mid_raw_solver_objective_is_reported_as_an_exact_integer():
    problem = make_grid_problem(
        2,
        2,
        program_population="All",
        overage=-1,
        shortage=-1,
    )
    market = MidMarket(
        programs=(MidProgram("A", 100, 1, True, None),),
        types=(MidType(1, 1, ("A",), (0,), (1.01,), (101,)),),
        student_count=1,
        outside_only_student_count=0,
        utility_student_count=1,
        utility_handling="omit_nonpositive",
    )
    scale = 100_000_000_000_001

    solution = MidCpSatSolver(market, scale, solve_time_limit=10, workers=1).solve(
        problem
    )

    assert solution.status == "OPTIMAL"
    assert solution.metadata["mid_raw_solver_objective"] == 101 * scale


def test_mid_separation_activates_overload_then_utility_gaps():
    market = MidMarket(
        programs=(MidProgram("A", 1, 1, True, None),),
        types=(
            MidType(0, 1, ("A",), (0,), (2.0,), (200,)),
            MidType(1, 1, ("A",), (0,), (1.0,), (100,)),
        ),
        student_count=2,
        outside_only_student_count=0,
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )
    zoning = {0: 0, 1: 0}

    overloaded = evaluate_cutoffs(market, zoning, {"A": 0}, 20)
    overload_separation = separate_mid_types(market, overloaded, set(), 20)
    feasible = evaluate_cutoffs(market, zoning, {"A": 20}, 20)
    utility_separation = separate_mid_types(market, feasible, set(), 20)

    assert overload_separation.overloaded_programs == ("A",)
    assert overload_separation.overload_type_indices == (0, 1)
    assert overload_separation.utility_gap_type_indices == ()
    assert utility_separation.overloaded_programs == ()
    assert utility_separation.overload_type_indices == ()
    assert utility_separation.utility_gap_type_indices == (0, 1)


def test_mid_full_type_activation_matches_monolithic_model():
    problem = make_grid_problem(
        2,
        2,
        program_population="All",
        overage=-1,
        shortage=-1,
    )
    problem.hint = {0: 0, 1: 0, 2: 1, 3: 1}
    market = MidMarket(
        programs=(
            MidProgram("A", 100, 1, False, 0),
            MidProgram("B", 200, 1, False, 3),
        ),
        types=(
            MidType(1, 1, ("A", "B"), (0, 0), (2.0, 1.0), (200, 100)),
            MidType(2, 1, ("B", "A"), (0, 0), (2.0, 1.0), (200, 100)),
        ),
        student_count=2,
        outside_only_student_count=0,
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )

    monolithic = MidCpSatSolver(market, 20, solve_time_limit=10, workers=1).solve(
        problem
    )
    generated = MidCpSatSolver(
        market,
        20,
        activated_type_indices={0, 1},
        solve_time_limit=10,
        workers=1,
    ).solve(problem)

    assert generated.status == "OPTIMAL"
    assert (
        generated.metadata["mid_raw_solver_objective"]
        == monolithic.metadata["mid_raw_solver_objective"]
    )
    assert (
        generated.metadata["mid_solver_cutoffs"]
        == monolithic.metadata["mid_solver_cutoffs"]
    )
    assert generated.metadata["mid_remaining_variable_count"] == 4
    assert generated.metadata["formulation"] == "mid_generated_utility_decomposition"


@pytest.mark.real_data
def test_mid_market_builds_for_summer_26_zoning():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="cp_bool",
        strategy="mid",
        data={"scenario": "summer-26-zoning", "overrides": {}},
    )
    problem = config.make_dataset().problem_for("BlockGroup_0")

    market = build_mid_market(problem, config)

    assert market.student_count == 3953
    assert market.utility_student_count == 3881
    assert market.outside_only_student_count >= 72
    assert len(market.programs) == 130
    assert sum(student_type.count for student_type in market.types) == 3953

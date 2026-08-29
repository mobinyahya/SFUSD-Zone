"""Focused tests for sampled stable-matching welfare optimization."""

from types import SimpleNamespace

import pytest

from optimization.config import OptimizationConfig
from optimization.data.initial_solutions import InitialSolution
from optimization.data.mid import MidProgram, MidStudent
from optimization.data.saa import (
    SaaMarket,
    SaaSample,
    sample_school_preferences,
)
from optimization.saa_oracle import (
    SaaCut,
    SaaOracle,
    SaaOracleResult,
    access_values,
)
from optimization.solvers import get_solver
from optimization.solution import ZoneSolution
from optimization.solvers.cpsat import CP_SAT_SCALE
from optimization.solvers.saa import scaled_saa_cut
from optimization.strategies import get_strategy
from optimization.strategies import saa as saa_module
from optimization.tests.synthetic import FakeDataset, make_grid_problem


def test_saa_samples_strict_stb_and_mtb_priorities_deterministically():
    market = _market()

    stb = sample_school_preferences(market, 2, "STB", 17)
    repeated = sample_school_preferences(market, 2, "STB", 17)
    mtb = sample_school_preferences(market, 2, "MTB", 17)

    assert stb == repeated
    assert stb[0].seed != stb[1].seed
    assert stb[0].school_orders[0] == stb[0].school_orders[1]
    assert all(set(order) == {0, 1} for sample in mtb for order in sample.school_orders)


def test_saa_recourse_dual_cut_is_tight_and_globally_valid():
    problem = make_grid_problem(
        2,
        2,
        program_population="All",
        overage=-1,
        shortage=-1,
    )
    market = _market()
    sample = SaaSample(seed=1, school_orders=((0, 1), (1, 0)))
    recourse = SaaOracle(market, sample, 0, problem)
    preferred_zoning = {0: 0, 1: 0, 2: 1, 3: 1}
    crossed_zoning = {0: 0, 1: 1, 2: 0, 3: 1}

    preferred = recourse.solve(preferred_zoning)
    crossed = recourse.solve(crossed_zoning)

    assert preferred.welfare == pytest.approx(4.0)
    assert crossed.welfare == pytest.approx(2.0)
    assert preferred.cut.value(
        access_values(market, problem, preferred_zoning)
    ) == pytest.approx(preferred.welfare)
    assert (
        preferred.cut.value(access_values(market, problem, crossed_zoning))
        >= crossed.welfare - 1e-8
    )


def test_saa_recourse_enforces_sampled_school_priorities():
    problem = make_grid_problem(2, 2, program_population="All")
    market = SaaMarket(
        programs=(
            MidProgram("A", 100, 1, True, None),
            MidProgram("B", 200, 1, True, None),
        ),
        students=(
            MidStudent(1, ("A", "B"), (0, 0), (10.0, 1.0), (1000, 100)),
            MidStudent(2, ("A", "B"), (0, 0), (2.0, 1.0), (200, 100)),
        ),
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )
    zoning = {0: 0, 1: 0, 2: 1, 3: 1}

    first_student_wins = SaaOracle(
        market, SaaSample(1, ((0, 1), (0, 1))), 0, problem
    ).solve(zoning)
    second_student_wins = SaaOracle(
        market, SaaSample(2, ((1, 0), (1, 0))), 1, problem
    ).solve(zoning)

    assert first_student_wins.welfare == pytest.approx(11.0)
    assert second_student_wins.welfare == pytest.approx(3.0)


@pytest.mark.parametrize("backend", ["cp_bool", "mip"])
def test_saa_strategy_optimizes_expected_welfare(monkeypatch, backend):
    problem = make_grid_problem(2, 2, program_population="All")
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population="All")
    monkeypatch.setattr(saa_module, "build_saa_market", lambda *args: _market())
    monkeypatch.setattr(saa_module, "initial_solution", lambda *args, **kwargs: None)
    solver = get_solver(backend, solve_time_limit=10, workers=1)
    strategy = get_strategy(
        "saa",
        levels=["BlockGroup_0"],
        solve_time_limits=[10],
        gap_limits=[0],
        hints="none",
        max_iterations=5,
        tolerance=1e-8,
        saa_num_seeds=2,
        saa_tie_breaking_method="MTB",
        seed=11,
    )

    solutions = strategy.run(dataset, solver)
    final = solutions[-1]

    assert final.status == "OPTIMAL"
    assert final.objective == pytest.approx(4.0)
    assert final.assignment == {0: 0, 1: 0, 2: 1, 3: 1}
    assert final.metadata["saa_selected_incumbent"] is True
    assert final.metadata["saa_num_seeds"] == 2
    assert final.metadata["saa_master_backend"] == backend
    assert final.metadata["saa_iteration_count"] <= 5
    assert final.metadata["aggregate_capacity_overage_disabled"] is True


def test_saa_positive_tolerance_does_not_claim_exact_optimality(monkeypatch):
    problem = make_grid_problem(2, 2, program_population="All")
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population="All")
    monkeypatch.setattr(saa_module, "build_saa_market", lambda *args: _market())
    monkeypatch.setattr(saa_module, "initial_solution", lambda *args, **kwargs: None)
    monkeypatch.setattr(saa_module, "SaaMipSolver", _BoundedMaster)
    monkeypatch.setattr(saa_module, "SaaOracle", _LowWelfareOracle)
    strategy = get_strategy(
        "saa",
        levels=["BlockGroup_0"],
        max_iterations=5,
        tolerance=3,
        hints="none",
        saa_num_seeds=1,
    )

    final = strategy.run(dataset, SimpleNamespace(name="mip", options={}))[-1]

    assert final.status == "FEASIBLE"
    assert final.metadata["saa_absolute_gap"] == pytest.approx(2.0)
    assert final.metadata["saa_certified_optimal"] is False
    assert final.metadata["saa_termination_reason"] == "bound_gap"


def test_saa_rounding_gap_does_not_claim_exact_optimality(monkeypatch):
    problem = make_grid_problem(2, 2, program_population="All")
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population="All")
    monkeypatch.setattr(saa_module, "build_saa_market", lambda *args: _market())
    monkeypatch.setattr(saa_module, "initial_solution", lambda *args, **kwargs: None)
    monkeypatch.setattr(saa_module, "SaaMipSolver", _NearBoundMaster)
    monkeypatch.setattr(saa_module, "SaaOracle", _LowWelfareOracle)
    strategy = get_strategy(
        "saa",
        levels=["BlockGroup_0"],
        max_iterations=2,
        tolerance=1e-6,
        hints="none",
        saa_num_seeds=1,
    )

    final = strategy.run(dataset, SimpleNamespace(name="mip", options={}))[-1]

    assert final.status == "FEASIBLE"
    assert final.metadata["saa_absolute_gap"] == pytest.approx(5e-8)
    assert final.metadata["saa_certified_optimal"] is False


def test_saa_preserves_feasible_hint_when_master_has_no_solution(monkeypatch):
    problem = make_grid_problem(2, 2, program_population="All")
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population="All")
    hint = {0: 0, 1: 0, 2: 1, 3: 1}
    monkeypatch.setattr(saa_module, "build_saa_market", lambda *args: _market())
    monkeypatch.setattr(
        saa_module,
        "initial_solution",
        lambda *args, **kwargs: InitialSolution(
            assignment=hint,
            metadata={"hints": "feasible", "hint_solver": "cp_bool"},
        ),
    )
    monkeypatch.setattr(saa_module, "SaaMipSolver", _UnknownMaster)
    monkeypatch.setattr(saa_module, "SaaOracle", _LowWelfareOracle)
    strategy = get_strategy(
        "saa",
        levels=["BlockGroup_0"],
        max_iterations=2,
        hints="feasible",
        saa_num_seeds=1,
    )

    final = strategy.run(dataset, SimpleNamespace(name="mip", options={}))[-1]

    assert final.status == "FEASIBLE"
    assert final.assignment == hint
    assert final.objective == pytest.approx(2.0)
    assert final.metadata["saa_incumbent_iteration"] is None
    assert final.metadata["saa_iteration_count"] == 1
    assert final.metadata["saa_master_backend"] == "mip"


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_config_rejects_invalid_saa_num_seeds(value):
    with pytest.raises(ValueError, match="saa_num_seeds"):
        OptimizationConfig(levels=["BlockGroup_0"], saa_num_seeds=value)


@pytest.mark.parametrize("value", ["random", 1, None])
def test_config_rejects_invalid_saa_tie_breaking_method(value):
    with pytest.raises(ValueError, match="saa_tie_breaking_method"):
        OptimizationConfig(levels=["BlockGroup_0"], saa_tie_breaking_method=value)


def test_cp_sat_cut_rounding_is_valid_and_anchor_tight():
    first = (1, 2)
    second = (3, 4)
    cut = SaaCut(
        sample_index=0,
        constant=0.1234567,
        coefficients=((first, 0.3333333), (second, -0.2222222)),
        anchor_access=((first, 1), (second, 0)),
    )

    constant, coefficients = scaled_saa_cut(cut, CP_SAT_SCALE)
    scaled_coefficients = dict(coefficients)
    for first_value in (0, 1):
        for second_value in (0, 1):
            access = {first: first_value, second: second_value}
            scaled_value = (
                constant
                + scaled_coefficients[first] * first_value
                + scaled_coefficients[second] * second_value
            ) / CP_SAT_SCALE
            assert scaled_value >= cut.value(access) - 1e-12

    anchor = dict(cut.anchor_access)
    scaled_anchor = (
        constant + sum(coefficient * anchor[pair] for pair, coefficient in coefficients)
    ) / CP_SAT_SCALE
    assert scaled_anchor - cut.value(anchor) <= 1 / CP_SAT_SCALE + 1e-12


def _market() -> SaaMarket:
    return SaaMarket(
        programs=(
            MidProgram("A", 100, 1, False, 0),
            MidProgram("B", 200, 1, False, 3),
        ),
        students=(
            MidStudent(1, ("A", "B"), (0, 0), (2.0, 1.0), (200, 100)),
            MidStudent(2, ("B", "A"), (0, 0), (2.0, 1.0), (200, 100)),
        ),
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )


class _LowWelfareOracle:
    def __init__(self, market, sample, sample_index, problem, workers=None):
        self.sample_index = sample_index

    def solve(self, zoning):
        return SaaOracleResult(
            welfare=2.0,
            cut=SaaCut(self.sample_index, 2.0, ()),
        )


class _BoundedMaster:
    def __init__(self, market, samples, cuts, **options):
        pass

    def solve(self, problem):
        return ZoneSolution(
            problem=problem,
            assignment={0: 0, 1: 1, 2: 0, 3: 1},
            status="OPTIMAL",
            objective=4.0,
            wall_time=0.0,
            metadata={"saa_master_best_bound": 4.0, "saa_num_seeds": 1},
        )


class _UnknownMaster:
    def __init__(self, market, samples, cuts, **options):
        pass

    def solve(self, problem):
        return ZoneSolution(
            problem=problem,
            assignment={},
            status="UNKNOWN",
            wall_time=0.0,
            metadata={"saa_master_best_bound": None, "saa_num_seeds": 1},
        )


class _NearBoundMaster(_BoundedMaster):
    def solve(self, problem):
        solution = super().solve(problem)
        solution.objective = 2.00000005
        solution.metadata["saa_master_best_bound"] = 2.00000005
        return solution

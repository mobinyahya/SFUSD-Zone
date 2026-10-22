"""Welfare adapters preserve pipeline inputs and report evaluated incumbents."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from choice.objective import ChoiceCut
from optimization.config import OptimizationConfig
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import make_grid_problem
from optimization import welfare_objectives as adapter


ASSIGNMENT = {0: 0, 1: 0, 2: 1, 3: 1}


def _problem(**config_options):
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="cp_bool",
        frl_dev=0.15,
        racial_dev=0.25,
        overage=0.45,
        shortage=0.1,
        boundary_prop=0.6,
        max_iterations=4,
        data={
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"program_population": "All"}}},
        },
        **config_options,
    )
    return make_grid_problem(
        2,
        2,
        program_population="All",
        frl_dev=config.frl_dev,
        racial_dev=config.racial_dev,
        overage=config.overage,
        shortage=config.shortage,
        boundary_prop=config.boundary_prop,
        fixed={1: 0},
        candidates={node: {zone} for node, zone in ASSIGNMENT.items()},
        hint=dict(ASSIGNMENT),
        optimization_config=config,
    )


def _assert_domains(problem, source):
    assert problem.G is source.G
    assert problem.fixed == source.fixed
    assert problem.candidates == source.candidates
    for name in ("frl_dev", "racial_dev", "overage", "shortage", "boundary_prop"):
        assert getattr(problem, name) == getattr(source, name)


def test_choice_adapter_keeps_domains_and_best_evaluated_incumbent(monkeypatch):
    from optimization.strategies import iterative_choice

    problem = _problem()
    solved = []
    evaluations = iter((20.0, 12.0, 18.0))

    class Model:
        def utility_bounds(self, source):
            _assert_domains(source, problem)
            return 0.0, 1000.0

        def evaluate_with_cuts(self, source, assignment):
            return SimpleNamespace(
                utility=next(evaluations),
                cuts=(ChoiceCut(node=0, constant=100.0),),
            )

    class Solver:
        options = {"solve_time_limit": 30.0}

        def with_options(self, **options):
            return self

        def solve(self, source):
            _assert_domains(source, problem)
            assert source.hint == problem.hint
            solved.append(source)
            if len(solved) == 4:
                return ZoneSolution(source, {}, "UNKNOWN")
            return ZoneSolution(
                source,
                dict(ASSIGNMENT),
                "OPTIMAL",
                objective=1000.0 - len(solved),
                wall_time=0.5,
            )

    monkeypatch.setattr(
        iterative_choice, "build_mnl_choice_model", lambda *a, **k: Model()
    )
    monkeypatch.setattr("optimization.solvers.get_solver", lambda *a, **k: Solver())
    times = iter((10.0, 17.0))
    monkeypatch.setattr(
        adapter, "time", SimpleNamespace(perf_counter=lambda: next(times))
    )

    result = adapter.solve_welfare_objective(
        problem, "cp_bool", "choice_welfare", {"solve_time_limit": 30.0}
    )

    assert len(solved) == 4
    assert result.objective == 20.0
    assert result.status == "FEASIBLE"
    assert result.wall_time == 7.0
    assert result.assignment == ASSIGNMENT
    assert result.metadata["choice_termination_reason"] == "master_unknown"
    assert result.metadata["choice_iteration_count"] == 4
    assert result.metadata["aggregate_capacity_overage_disabled"] is False
    assert result.metadata["aggregate_capacity_shortage_disabled"] is False
    assert [
        s["evaluated_objective"] for s in result.metadata["objective_iterations"]
    ] == [
        20.0,
        12.0,
        18.0,
        None,
    ]
    assert problem.choice_objective is None
    assert problem.hint == ASSIGNMENT


@pytest.mark.parametrize("certified", [False, True])
def test_saa_adapter_prefers_final_incumbent_and_its_certificate(
    monkeypatch, certified
):
    problem = _problem()

    class Strategy:
        capacity_balance = False
        options = {}

        def run(self, dataset, solver):
            assert self.capacity_balance is True
            source = dataset.problem_for(problem.level, hint=None)
            _assert_domains(source, problem)
            assert source.hint == problem.hint
            master = ZoneSolution(
                source,
                dict(ASSIGNMENT),
                "OPTIMAL",
                objective=100.0,
                metadata={"saa_welfare": 20.0},
            )
            incumbent = replace(
                master,
                status="OPTIMAL" if certified else "FEASIBLE",
                objective=20.0,
                metadata={
                    "saa_welfare": 20.0,
                    "saa_selected_incumbent": True,
                    "saa_certified_optimal": certified,
                    "saa_global_upper_bound": 20.0 if certified else 100.0,
                    "saa_absolute_gap": 0.0 if certified else 80.0,
                    "saa_iteration_count": 1,
                    "saa_termination_reason": "bound_gap"
                    if certified
                    else "time_limit",
                    "mid_discrete_welfare": 21.0,
                },
            )
            return [master, incumbent]

    monkeypatch.setattr(OptimizationConfig, "make_strategy", lambda self: Strategy())
    monkeypatch.setattr("optimization.solvers.get_solver", lambda *a, **k: object())

    result = adapter.solve_welfare_objective(
        problem, "cp_bool", "saa_welfare", {"solve_time_limit": 30.0}
    )

    assert result.objective == 20.0
    assert result.status == ("OPTIMAL" if certified else "FEASIBLE")
    assert result.metadata["saa_certified_optimal"] is certified
    assert result.metadata["saa_selected_incumbent"] is True
    assert result.metadata["mid_discrete_welfare"] == 21.0
    assert result.metadata["saa_absolute_gap"] == (0.0 if certified else 80.0)
    _assert_domains(result.problem, problem)


def test_saa_adapter_keeps_explicit_hint_before_market_construction(monkeypatch):
    from optimization.strategies import saa

    problem = _problem()

    class MarketReached(Exception):
        pass

    def unexpected_initializer(*args, **kwargs):
        pytest.fail(
            "An explicit pipeline hint must bypass default hint initialization."
        )

    def inspect_market(source, config):
        _assert_domains(source, problem)
        assert source.hint == ASSIGNMENT
        assert config.data_scenario is problem.optimization_config.data_scenario
        raise MarketReached

    monkeypatch.setattr(saa, "initial_solution", unexpected_initializer)
    monkeypatch.setattr(saa, "build_saa_market", inspect_market)
    with pytest.raises(MarketReached):
        adapter.solve_welfare_objective(
            problem, "cp_bool", "saa_welfare", {"solve_time_limit": 30.0}
        )


@pytest.mark.parametrize("objective", ["choice_welfare", "saa_welfare"])
def test_welfare_adapter_with_no_master_returns_unknown(monkeypatch, objective):
    problem = _problem()

    class Strategy:
        options = {}

        def run(self, dataset, solver):
            return []

    monkeypatch.setattr(OptimizationConfig, "make_strategy", lambda self: Strategy())
    monkeypatch.setattr("optimization.solvers.get_solver", lambda *a, **k: object())
    result = adapter.solve_welfare_objective(
        problem, "cp_bool", objective, {"solve_time_limit": 0.0}
    )
    assert result.status == "UNKNOWN"
    assert result.assignment == {}
    assert result.objective is None
    assert result.metadata["objective_iterations"] == []

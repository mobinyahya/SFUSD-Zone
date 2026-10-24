"""Tests for the choice-aware short bursts strategy."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from optimization.data.mid import MidMarket, MidProgram, MidType
from optimization.mid_oracle import finite_grid_oracle
from optimization.solvers import get_solver
from optimization.solvers.recom_policy import PRESETS, ReComSettings
from optimization.strategies import get_strategy
from optimization.strategies import short_bursts_choice as sbc_module
from optimization.tests.synthetic import make_grid_problem, make_solver_contract_problem


LEVEL = "BlockGroup_0"


def _make_test_market() -> MidMarket:
    return MidMarket(
        programs=(
            MidProgram("P0", 100, 2, False, 0),
            MidProgram("P1", 200, 2, False, 3),
        ),
        types=(
            MidType(0, 1, ("P0", "P1"), (0, 0), (2.0, 1.0), (200, 100)),
            MidType(1, 1, ("P0", "P1"), (0, 0), (2.0, 1.0), (200, 100)),
            MidType(2, 1, ("P1", "P0"), (0, 0), (2.0, 1.0), (200, 100)),
            MidType(3, 1, ("P1", "P0"), (0, 0), (2.0, 1.0), (200, 100)),
        ),
        student_count=4,
        outside_only_student_count=0,
        utility_student_count=4,
        utility_handling="omit_nonpositive",
    )


def _dataset(problem, program_population: str = "All"):
    from optimization.tests.synthetic import FakeDataset

    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population=program_population)
    return dataset


def _use_market(monkeypatch, market: MidMarket) -> None:
    """Serve a synthetic market instead of building one from real data."""

    monkeypatch.setattr(sbc_module, "build_mid_market", lambda problem, config: market)


def _strategy(**options):
    options.setdefault("levels", [LEVEL])
    return get_strategy("short_bursts_choice", **options)


def _short_bursts(preset: str = "uniform_bursts", **options):
    options = {**PRESETS[preset].as_options(), **options}
    options.setdefault("solve_time_limit", 60.0)
    options.setdefault("workers", 1)
    return get_solver("recom", **options)


def test_strategy_is_registered() -> None:
    from optimization.strategies.base import available_strategies

    assert "short_bursts_choice" in available_strategies()


def test_requires_the_recom_solver(monkeypatch) -> None:
    problem = make_solver_contract_problem(hint={0: 0, 1: 0, 2: 1, 3: 1})
    _use_market(monkeypatch, _make_test_market())

    with pytest.raises(ValueError, match="solver='recom'"):
        _strategy().run(_dataset(problem), get_solver("cp_bool"))


def test_requires_all_program_population(monkeypatch) -> None:
    problem = make_solver_contract_problem(hint={0: 0, 1: 0, 2: 1, 3: 1})
    _use_market(monkeypatch, _make_test_market())

    with pytest.raises(ValueError, match="program_population='All'"):
        _strategy().run(_dataset(problem, "GE"), _short_bursts())


def test_requires_a_hint(monkeypatch) -> None:
    _use_market(monkeypatch, _make_test_market())
    problem = make_solver_contract_problem(hint=None)

    solutions = _strategy().run(_dataset(problem), _short_bursts(hints="none"))

    assert len(solutions) == 1
    assert solutions[0].status == "ERROR"
    assert "hint" in solutions[0].metadata["error_message"].lower()


@pytest.mark.parametrize(
    ("strategy_options", "solver_options", "match"),
    [
        ({}, {"recom_pair_selection": "invalid"}, "recom_pair_selection"),
        ({}, {"recom_burst_length": -1}, "recom_burst_length"),
        ({"mid_lottery_scale": -1}, {}, "mid_lottery_scale"),
        ({"mid_lottery_scale": 1.5}, {}, "mid_lottery_scale"),
        ({"mid_lottery_scale": True}, {}, "mid_lottery_scale"),
        ({"solve_time_limits": [float("inf")]}, {}, "finite and non-negative"),
    ],
)
def test_validates_options(
    monkeypatch, strategy_options, solver_options, match
) -> None:
    problem = make_solver_contract_problem(hint={0: 0, 1: 0, 2: 1, 3: 1})
    _use_market(monkeypatch, _make_test_market())

    with pytest.raises(ValueError, match=match):
        _strategy(**strategy_options).run(
            _dataset(problem), _short_bursts(**solver_options)
        )


@pytest.mark.parametrize("preset", ["uniform_bursts", "entropy", "hedge"])
def test_solve_basic(monkeypatch, preset: str) -> None:
    problem = make_solver_contract_problem(hint={0: 0, 1: 0, 2: 1, 3: 1})
    _use_market(monkeypatch, _make_test_market())

    solutions = _strategy().run(
        _dataset(problem),
        _short_bursts(preset, recom_iterations=10, recom_burst_length=5, seed=42),
    )

    assert len(solutions) == 1
    sol = solutions[0]
    assert sol.status == "FEASIBLE"
    assert sol.feasible
    assert sol.objective is not None
    assert ReComSettings.from_options(sol.metadata) == replace(
        PRESETS[preset], burst_length=5
    )
    assert sol.metadata["objective_kind"] == "mid_program_welfare"
    assert sol.metadata["initial_feasible"] is True
    assert sol.metadata["strategy"] == "short_bursts_choice"
    assert sol.metadata["formulation"] == "short_bursts_discrete_mid_oracle"
    assert sol.metadata["mid_oracle_type"] == "finite"
    assert sol.metadata["mid_lottery_scale"] == 20
    assert sol.metadata["mid_preprocessing_seconds"] >= 0.0
    assert sol.metadata["workers"] == 1


def test_never_selects_infeasible_partition(monkeypatch) -> None:
    problem = make_solver_contract_problem(hint={0: 0, 1: 0, 2: 1, 3: 1}, frl_dev=0.01)
    problem.G.nodes[0]["FRL"] = 0.1
    problem.G.nodes[1]["FRL"] = 0.9
    problem.G.nodes[2]["FRL"] = 0.5
    problem.G.nodes[3]["FRL"] = 0.5

    # Huge utility for the school reachable only in the infeasible partition.
    market = MidMarket(
        programs=(
            MidProgram("P0", 100, 2, False, 0),
            MidProgram("P1", 200, 4, False, 3),
        ),
        types=(
            MidType(0, 1, ("P0",), (0,), (1.0,), (100,)),
            MidType(1, 1, ("P1",), (0,), (1000.0,), (100000,)),
            MidType(2, 1, ("P1",), (0,), (1.0,), (100,)),
            MidType(3, 1, ("P1",), (0,), (1.0,), (100,)),
        ),
        student_count=4,
        outside_only_student_count=0,
        utility_student_count=4,
        utility_handling="omit_nonpositive",
    )
    _use_market(monkeypatch, market)

    solutions = _strategy().run(
        _dataset(problem),
        _short_bursts(recom_iterations=25, recom_burst_length=5, seed=42),
    )

    sol = solutions[0]
    assert sol.status == "FEASIBLE"
    assert sol.feasible
    # The initial partition is retained because every move is infeasible.
    assert sol.assignment == {0: 0, 1: 0, 2: 1, 3: 1}


def test_selects_higher_welfare(monkeypatch) -> None:
    problem = make_grid_problem(
        2, 2, frl_dev=1.0, overage=5.0, shortage=0.0, hint={0: 0, 1: 0, 2: 1, 3: 1}
    )
    # Preferences favor the alternate partition {0: 1, 1: 0, 2: 1, 3: 0}.
    market = MidMarket(
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
    _use_market(monkeypatch, market)

    solutions = _strategy().run(
        _dataset(problem),
        _short_bursts(recom_iterations=20, recom_burst_length=5, seed=1),
    )

    sol = solutions[0]
    assert sol.status == "FEASIBLE"
    assert sol.metadata["restarts"] >= 1
    assert sol.metadata["final_welfare"] > sol.metadata["initial_welfare"]
    # The reported objective is the discrete oracle's welfare for the zoning kept.
    expected = finite_grid_oracle(
        sbc_module.preprocess_mid_market(market, problem),
        sol.assignment,
        20,
        check_minimality=False,
    )
    assert sol.objective == pytest.approx(expected.welfare)


def test_parallel_workers_match_serial_scores(monkeypatch) -> None:
    market = _make_test_market()
    _use_market(monkeypatch, market)
    kwargs = dict(recom_iterations=20, recom_burst_length=5, seed=1)

    def solve(workers: int):
        problem = make_grid_problem(
            2, 2, frl_dev=1.0, overage=5.0, shortage=0.0, hint={0: 0, 1: 0, 2: 1, 3: 1}
        )
        return _strategy().run(
            _dataset(problem), _short_bursts(workers=workers, **kwargs)
        )[0]

    parallel = solve(4)
    serial = solve(1)

    assert parallel.status == "FEASIBLE"
    assert parallel.feasible
    assert parallel.metadata["workers"] == 4
    assert parallel.objective is not None
    # Worker count must not change the search or its scores.
    assert parallel.assignment == serial.assignment
    assert parallel.objective == pytest.approx(serial.objective)

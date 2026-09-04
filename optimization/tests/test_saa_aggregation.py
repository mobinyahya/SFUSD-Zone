"""Tests for SAA cut aggregation and dynamic time limit schedule."""

from types import SimpleNamespace
import pytest

from benchmark.config import optimization_config_from_dict
from optimization.data.mid import MidProgram, MidStudent
from optimization.data.saa import SaaMarket
from optimization.saa_oracle import SaaCut, aggregate_saa_cuts
from optimization.solution import ZoneSolution
from optimization.solvers import get_solver
from optimization.strategies import get_strategy
from optimization.strategies import saa as saa_module
from optimization.strategies.budget import master_time_limit
from optimization.tests.synthetic import FakeDataset, make_grid_problem


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


def test_aggregate_saa_cuts_averages_coefficients_and_preserves_anchor():
    p1 = (1, 0)
    p2 = (2, 3)
    cut1 = SaaCut(
        sample_index=0,
        constant=10.0,
        coefficients=((p1, 2.0), (p2, -4.0)),
        anchor_access=((p1, 1), (p2, 0)),
    )
    cut2 = SaaCut(
        sample_index=1,
        constant=20.0,
        coefficients=((p1, 4.0), (p2, 2.0)),
        anchor_access=((p1, 1), (p2, 0)),
    )

    agg = aggregate_saa_cuts((cut1, cut2))

    assert agg.sample_index is None
    assert agg.constant == pytest.approx(15.0)
    coeffs = dict(agg.coefficients)
    assert coeffs[p1] == pytest.approx(3.0)
    assert coeffs[p2] == pytest.approx(-1.0)
    assert agg.anchor_access == ((p1, 1), (p2, 0))

    anchor_dict = {p1: 1, p2: 0}
    val1 = cut1.value(anchor_dict)
    val2 = cut2.value(anchor_dict)
    assert agg.value(anchor_dict) == pytest.approx((val1 + val2) / 2.0)


def test_aggregate_saa_cuts_edge_cases():
    with pytest.raises(ValueError, match="Cannot aggregate empty"):
        aggregate_saa_cuts(())

    single = SaaCut(0, 5.0, (((1, 2), 3.0),), ((1, 2), 1))
    agg = aggregate_saa_cuts((single,))
    assert agg.sample_index is None
    assert agg.constant == 5.0
    assert agg.coefficients == (((1, 2), 3.0),)


def test_config_rejects_unknown_saa_aggregate_cut_keys():
    """SAA cut aggregation has no config toggle; stray keys must not be ignored."""

    dict_cfg = {
        "levels": ["BlockGroup_0"],
        "strategy": "saa",
        "solver": "cp_bool",
        "data": {
            "scenario": "legacy",
            "overrides": {"filters": {"optimization": {"program_population": "All"}}},
        },
        "saa_aggregated_cuts": True,
        "saa_aggregate_cuts": True,
    }

    with pytest.raises(ValueError, match="Unknown optimization config keys"):
        optimization_config_from_dict(dict_cfg)


def test_master_time_limit_schedule_distribution():
    # 3 iterations, remaining = 60s
    # iter 0: weight 1, remaining weights 1+2+3 = 6 -> 60 * 1/6 = 10s
    assert master_time_limit(60.0, 0, 3) == pytest.approx(10.0)

    # iter 1: remaining = 50s, weight 2, remaining weights 2+3 = 5 -> 50 * 2/5 = 20s
    assert master_time_limit(50.0, 1, 3) == pytest.approx(20.0)

    # iter 2: remaining = 30s, weight 3, remaining weights 3 = 3 -> 30 * 3/3 = 30s
    assert master_time_limit(30.0, 2, 3) == pytest.approx(30.0)


@pytest.mark.parametrize("backend", ["cp_bool", "mip"])
def test_saa_strategy_with_aggregate_cuts(monkeypatch, backend):
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
    assert final.metadata["saa_aggregate_cuts"] is True
    assert (
        final.metadata["saa_budget_policy"] == "linearly_increasing_with_carry_forward"
    )
    assert final.metadata["saa_total_budget_seconds"] == 10.0
    assert final.metadata["saa_budget_accounting"] == "wall_clock"
    assert final.metadata["saa_num_seeds"] == 2
    # The reference MID welfare is scored once, on the selected incumbent only.
    assert final.metadata["mid_welfare"] == pytest.approx(4.0)
    assert final.metadata["mid_discrete_welfare"] == pytest.approx(4.0)
    assert final.metadata["mid_incumbent_welfare"] == pytest.approx(4.0)
    assert final.metadata["mid_oracle_type"] == "finite"
    assert final.metadata["mid_lottery_scale"] == 20
    assert "mid_discrete_cutoffs" in final.metadata
    assert final.metadata["saa_reference_oracle_seconds"] >= 0.0

    # Verify that in every iteration stage, exactly 1 cut is added and that the
    # per-candidate search does not pay for reference-oracle metadata.
    for stage in solutions[:-1]:
        assert stage.metadata["saa_cuts_added"] == 1
        assert stage.metadata["saa_aggregate_cuts"] is True
        assert "saa_master_time_limit" in stage.metadata
        assert isinstance(stage.metadata["saa_welfare"], float)
        assert stage.metadata["saa_welfare"] > 0.0
        assert "mid_discrete_cutoffs" not in stage.metadata


def _zero_budget_strategy(**overrides):
    options = {
        "levels": ["BlockGroup_0"],
        "solve_time_limits": [0],
        "gap_limits": [0],
        "hints": "none",
        "max_iterations": 5,
        "saa_num_seeds": 2,
        "seed": 11,
    }
    options.update(overrides)
    return get_strategy("saa", **options)


def _saa_dataset():
    problem = make_grid_problem(2, 2, program_population="All")
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population="All")
    return dataset


def test_saa_exhausted_budget_without_incumbent_returns_unknown_stage(monkeypatch):
    """An expired budget must still satisfy the strategy's non-empty contract."""

    dataset = _saa_dataset()
    monkeypatch.setattr(saa_module, "build_saa_market", lambda *args: _market())
    monkeypatch.setattr(saa_module, "initial_solution", lambda *args, **kwargs: None)

    solutions = _zero_budget_strategy().run(
        dataset, get_solver("cp_bool", solve_time_limit=10, workers=1)
    )

    assert len(solutions) == 1
    final = solutions[0]
    assert final.status == "UNKNOWN"
    assert final.assignment == {}
    assert final.metadata["saa_termination_reason"] == "time_limit"
    assert final.metadata["saa_iteration_count"] == 0
    assert final.metadata["saa_certified_optimal"] is False
    assert final.metadata["mid_welfare"] is None


def test_saa_exhausted_budget_with_feasible_hint_returns_hint_incumbent(monkeypatch):
    """A hint incumbent must be returned even when no master solve ever runs."""

    dataset = _saa_dataset()
    hint_assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    hint = ZoneSolution(
        problem=dataset.problem_for("BlockGroup_0"),
        assignment=dict(hint_assignment),
        status="FEASIBLE",
        wall_time=0.0,
        metadata={"hints": "feasible"},
    )
    monkeypatch.setattr(saa_module, "build_saa_market", lambda *args: _market())
    monkeypatch.setattr(saa_module, "initial_solution", lambda *args, **kwargs: hint)

    solutions = _zero_budget_strategy(hints="feasible").run(
        dataset, get_solver("cp_bool", solve_time_limit=10, workers=1)
    )

    assert len(solutions) == 1
    final = solutions[0]
    assert final.assignment == hint_assignment
    assert final.status == "FEASIBLE"
    assert final.objective is not None
    assert final.metadata["saa_selected_incumbent"] is True
    assert final.metadata["saa_termination_reason"] == "time_limit"
    assert final.metadata["mid_discrete_welfare"] is not None

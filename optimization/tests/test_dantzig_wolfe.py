"""Data-free checks of whole-zone welfare, master duals, and search integration."""

from dataclasses import replace
from itertools import combinations
import time
from types import SimpleNamespace

import pytest

from optimization.config import OptimizationConfig
from optimization.data.mid import MidMarket, MidProgram, MidType
from optimization.mid_oracle import finite_grid_oracle
from optimization.solvers import get_solver
from optimization.strategies import get_strategy
from optimization.strategies import dantzig_wolfe as dw
from optimization.tests.synthetic import FakeDataset, make_grid_problem
from optimization.zone_pricing import price_zone
from optimization.zone_columns import (
    MasterResult,
    ZoneColumn,
    ZonePool,
    restrict_market,
    solve_master,
)


def market():
    return MidMarket(
        programs=(
            MidProgram("A", 100, 1, False, 0),
            MidProgram("B", 200, 1, False, 3),
            MidProgram("C", 300, 1, True, None),
        ),
        types=tuple(
            MidType(
                n, 1, ("C", "A", "B"), (0, n % 2, 0), (10.0, 4.0, 2.0), (1000, 400, 200)
            )
            for n in range(4)
        ),
        student_count=4,
        outside_only_student_count=0,
        utility_student_count=4,
        utility_handling="omit_nonpositive",
    )


def test_citywide_removed_and_zone_welfare_is_additive():
    p = make_grid_problem(2, 2)
    m = market()
    restricted = restrict_market(m)
    assert [p.program_id for p in restricted.programs] == ["A", "B"]
    assert restricted.student_count == 4
    assert all(t.programs == ("A", "B") for t in restricted.types)
    assert restricted.types[1].priorities == (1, 0)
    assert len(m.programs) == 3  # no mutation of the source market
    pool = ZonePool(p, m, lottery_scale=5)
    for assignment in (
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 0, 1: 1, 2: 0, 3: 1},
        {0: 0, 1: 1, 2: 1, 3: 1},
    ):
        columns = pool.add_partition(assignment)
        full = finite_grid_oracle(restricted, assignment, 5)
        assert sum(c.score for c in columns) == pytest.approx(full.welfare)


def test_students_with_only_citywide_preferences_are_retained():
    m = market()
    m = replace(m, types=(MidType(0, 4, ("C",), (0,), (40.0,), (4000,)),))
    restricted = restrict_market(m)
    assert restricted.student_count == restricted.outside_only_student_count == 4
    assert restricted.types[0].programs == ()
    pool = ZonePool(make_grid_problem(2, 2), m)
    assert pool.column(0, {0}).score == 0


def test_master_matches_exhaustive_complete_pool_and_duals():
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, market(), lottery_scale=5)
    for size in range(1, p.A):
        for members in combinations(p.nodes, size):
            for z in range(p.Z):
                if pool.feasible(z, frozenset(members)):
                    pool.add(pool.column(z, members))
    columns = tuple(pool.columns.values())
    exhaustive = max(
        a.score + b.score
        for a in columns
        for b in columns
        if a.zone == 0
        and b.zone == 1
        and not a.nodes & b.nodes
        and a.nodes | b.nodes == pool.nodes
    )
    lp = solve_master(p, columns, 5)
    integer = solve_master(p, columns, 5, integer=True)
    assert integer.objective == pytest.approx(exhaustive)
    assert lp.objective >= integer.objective - 1e-6
    assert max(lp.reduced_cost(c) for c in columns) <= 1e-6
    assert sum(lp.node_duals.values()) + sum(lp.zone_duals.values()) == pytest.approx(
        lp.objective
    )
    assert len(integer.selected) == p.Z
    assert set.union(*(set(c.nodes) for c in integer.selected)) == pool.nodes
    assert sum(len(c.nodes) for c in integer.selected) == p.A


def test_master_recombines_columns_from_different_seed_partitions():
    p = make_grid_problem(1, 6)
    p.centroids = [0, 2, 5]
    p.centroid_school_ids = [100, 150, 200]
    # Scores deliberately isolate master recombination from the welfare oracle.
    columns = [
        ZoneColumn(z, frozenset(nodes), score, 0)
        for z, nodes, score in (
            (0, (0, 1), 10),
            (1, (2, 4), 0),
            (2, (3, 5), 0),
            (0, (0, 4), 0),
            (1, (2, 3), 20),
            (2, (1, 5), 0),
            (0, (0, 2), 0),
            (1, (1, 3), 0),
            (2, (4, 5), 10),
        )
    ]
    result = solve_master(p, columns, 5, integer=True)
    assert result.objective == 40
    assert [c.nodes for c in result.selected] == [
        frozenset({0, 1}),
        frozenset({2, 3}),
        frozenset({4, 5}),
    ]


def test_boundary_is_counted_once_and_cap_enforced():
    p = make_grid_problem(2, 2)
    pool = ZonePool(p)
    columns = pool.add_partition({0: 0, 1: 0, 2: 1, 3: 1})
    assert sum(c.score for c in columns) == -2
    p.boundary_prop = 0.25
    assert solve_master(p, columns, 5).status == "INFEASIBLE"
    p.boundary_prop = 0.5
    assert solve_master(p, columns, 5, integer=True).objective == -2


def test_boundary_dual_enters_reduced_cost_with_correct_sign():
    dual = MasterResult(
        "OPTIMAL", node_duals={0: 1.0, 1: 2.0}, zone_duals={0: 3.0}, boundary_dual=4.0
    )
    assert dual.reduced_cost(ZoneColumn(0, frozenset({0, 1}), 20.0, 6)) == 2.0


def test_pool_rejects_disconnected_unbalanced_or_forbidden_columns():
    p = make_grid_problem(2, 2, fixed={0: 0}, candidates={1: {1}})
    pool = ZonePool(p)
    assert not pool.feasible(0, frozenset({0, 3}))
    assert not pool.feasible(0, frozenset({0, 1}))
    assert not pool.feasible(1, frozenset({0, 2}))
    assert not pool.feasible(0, frozenset({2}))
    assert pool.feasible(0, frozenset({0, 2}))
    p.G.nodes[0]["FRL"] = 1.0
    p.frl_dev = 0.0
    assert not ZonePool(p).feasible(0, frozenset({0}))


def test_pricing_builds_new_zones_using_shadow_prices():
    p = make_grid_problem(1, 4)
    pool = ZonePool(p)
    pool.add_partition({0: 0, 1: 0, 2: 1, 3: 1})
    # Expensive cover price on node 1 encourages zone 0 to drop it. No whole
    # partition is needed to generate this new individual zone.
    master = MasterResult(
        "OPTIMAL",
        node_duals={0: -2.0, 1: 10.0, 2: 0.0, 3: 0.0},
        zone_duals={0: 0.0, 1: 0.0},
    )
    result = price_zone(pool, 0, master, {}, deadline=time.monotonic() + 5)
    assert result.status == "OPTIMAL"
    assert result.nodes == frozenset({0})
    assert result.reduced_cost > 0


def test_recom_visitor_collects_non_improving_partitions():
    p = make_grid_problem(2, 2, hint={0: 0, 1: 0, 2: 1, 3: 1})
    visited = []
    solver = get_solver("recom", recom_iterations=100, solve_time_limit=5, seed=42)
    result = solver.sample_feasible(p, lambda a: visited.append(a))
    assert len({tuple(sorted(a.items())) for a in visited}) > 1
    assert len(visited) == result.metadata["accepted_moves"] + 1
    assert all(len(a) == p.A for a in visited)


def test_recom_visitor_can_stop_at_initial_partition():
    p = make_grid_problem(2, 2, hint={0: 0, 1: 0, 2: 1, 3: 1})
    result = get_solver("recom", recom_iterations=100).sample_feasible(
        p, lambda _: False
    )
    assert result.feasible
    assert result.metadata["stop_reason"] == "visitor_stop"
    assert result.metadata["attempted_moves"] == 0
    assert result.metadata["initial_feasible"] is True


@pytest.mark.parametrize("objective", ["mid", "boundary"])
def test_strategy_end_to_end(monkeypatch, objective):
    p = make_grid_problem(2, 2, hint={0: 0, 1: 0, 2: 1, 3: 1})
    dataset = FakeDataset(p)
    dataset.config = SimpleNamespace(include_citywide=False, program_population="All")
    monkeypatch.setattr(dw, "build_mid_market", lambda *_: market())
    strategy = get_strategy(
        "dantzig_wolfe",
        levels=["BlockGroup_0"],
        solve_time_limits=[5],
        dw_objective=objective,
        dw_recom_samples=10,
        dw_recom_chains=2,
        max_iterations=2,
    )
    solver = get_solver("recom", recom_iterations=30, workers=1)
    solution = strategy.run(dataset, solver)[0]
    assert solution.feasible
    assert set(solution.assignment) == set(p.nodes)
    assert len(set(solution.assignment.values())) == p.Z
    assert solution.status == "OPTIMAL"
    assert solution.metadata["dw_pricing_certified"] is True
    assert solution.metadata["dw_absolute_gap"] <= 1e-6
    if objective == "mid":
        assert solution.objective == pytest.approx(
            finite_grid_oracle(
                restrict_market(market()), solution.assignment, 20
            ).welfare
        )
        assert solution.metadata["dw_citywide_programs_removed"] == 1


def test_zero_budget_returns_unknown_without_sampling(monkeypatch):
    p = make_grid_problem(2, 2)
    dataset = FakeDataset(p)
    dataset.config = SimpleNamespace(include_citywide=False, program_population="All")
    strategy = get_strategy(
        "dantzig_wolfe",
        levels=["BlockGroup_0"],
        solve_time_limits=[0],
        dw_objective="boundary",
    )
    solution = strategy.run(dataset, get_solver("recom"))[0]
    assert solution.status == "UNKNOWN"
    assert solution.metadata["stop_reason"] == "time_limit"


def test_config_and_example():
    from pathlib import Path

    config = OptimizationConfig.from_yaml(
        Path(__file__).parents[1] / "dantzig_wolfe.example.yaml"
    )
    assert config.make_strategy().name == "dantzig_wolfe"
    assert config.include_citywide is False
    assert config.make_strategy().options["dw_recom_samples"] == 1000


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"solver": "cp_bool"}, "solver='recom'"),
        ({"budget_accounting": "solver_time"}, "wall_clock"),
        ({"dw_recom_samples": -1}, "integer"),
        ({"dw_recom_time_limit": True}, "dw_recom_time_limit"),
        ({"tolerance": float("nan")}, "tolerance"),
        ({"dw_objective": "saa"}, "dw_objective"),
    ],
)
def test_invalid_config(overrides, match):
    options = dict(strategy="dantzig_wolfe", solver="recom", dw_objective="boundary")
    with pytest.raises(ValueError, match=match):
        OptimizationConfig(**{**options, **overrides})


def test_config_rejects_citywide():
    with pytest.raises(ValueError, match="include_citywide=false"):
        OptimizationConfig(
            strategy="dantzig_wolfe",
            solver="recom",
            dw_objective="boundary",
            data={
                "scenario": "legacy",
                "overrides": {"filters": {"optimization": {"include_citywide": True}}},
            },
        )

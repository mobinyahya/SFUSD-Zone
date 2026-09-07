"""Compare global pricing and complete search with independent enumeration."""

from dataclasses import replace
from itertools import combinations
import math
import random
import time
from types import SimpleNamespace

import pytest

from optimization.branch_price import branch_and_price
from optimization.tests.synthetic import make_grid_problem
from optimization.tests.test_dantzig_wolfe import market
from optimization.zone_columns import MasterResult, ZoneColumn, ZonePool, solve_master
from optimization.zone_pricing import PricingResult, compatible, price_zone


def all_columns(pool):
    for size in range(1, pool.problem.A + 1):
        for members in combinations(pool.problem.nodes, size):
            for zone in range(pool.problem.Z):
                if pool.feasible(zone, frozenset(members)):
                    yield pool.column(zone, members)


@pytest.mark.parametrize("scale", [1, 3, 5])
@pytest.mark.parametrize("phase_one", [False, True])
def test_global_price_matches_every_feasible_zone(scale, phase_one):
    p = make_grid_problem(2, 2, fixed={0: 0}, candidates={3: {1}})
    # Fractional utilities are deliberately different from scaled utilities.
    m = market()
    m = replace(
        m,
        types=tuple(replace(t, utility_sums=(10.0, 4.00017, 2.00003)) for t in m.types),
    )
    pool = ZonePool(p, m, scale)
    rng = random.Random(scale)
    lp = MasterResult(
        "OPTIMAL",
        node_duals={n: rng.uniform(-5, 5) for n in p.nodes},
        zone_duals={0: 1.37, 1: -0.98},
        boundary_dual=0.123,
        phase_one=phase_one,
    )
    decisions = {(1, 0): 0}
    columns = tuple(all_columns(pool))
    for zone in range(p.Z):
        eligible = [c for c in columns if c.zone == zone and compatible(c, decisions)]
        expected = max(lp.reduced_cost(c) for c in eligible)
        actual = price_zone(
            pool,
            zone,
            lp,
            decisions,
            deadline=time.monotonic() + 10,
            phase_one=phase_one,
        )
        assert actual.status == "OPTIMAL"
        assert actual.bound == pytest.approx(expected, abs=1e-7)
        assert actual.reduced_cost == pytest.approx(expected, abs=1e-7)
        column = pool.column(zone, actual.nodes)
        assert compatible(column, decisions)
        if not phase_one:
            assert actual.score == pytest.approx(column.score, abs=1e-7)


def test_omitted_colocated_students_do_not_consume_seats_or_get_welfare():
    p = make_grid_problem(2, 2)
    m = market()
    m = replace(
        m,
        types=(
            replace(
                m.types[0],
                count=4,
                utility_sums=(40.0, 16.0, 8.0),
                scaled_utility_sums=(4000, 1600, 800),
            ),
        ),
    )
    pool = ZonePool(p, m, 5)
    result = price_zone(
        pool, 0, None, {}, fixed_nodes={3}, deadline=time.monotonic() + 5
    )
    assert result.status == "OPTIMAL"
    assert result.score == 0


@pytest.mark.parametrize("cap", [-1, 0.5])
def test_empty_pool_reaches_exhaustive_global_mid_optimum(cap):
    p = make_grid_problem(2, 2, boundary_prop=cap)
    pool = ZonePool(p, market(), 5)
    full = tuple(all_columns(pool))
    enumerated = solve_master(p, full, 5, integer=True)
    assert not pool.columns  # Enumeration above did not seed the search.
    result = branch_and_price(pool, deadline=time.monotonic() + 15)
    assert result.status == "OPTIMAL"
    assert sum(c.score for c in result.selected) == pytest.approx(enumerated.objective)
    assert result.upper_bound == pytest.approx(enumerated.objective)
    assert any(h["phase_one"] and h["columns_added"] for h in result.history)


def test_infeasible_partition_is_proved_by_phase_one_pricing():
    p = make_grid_problem(2, 2, boundary_prop=0.0)
    pool = ZonePool(p, market(), 3)
    result = branch_and_price(pool, deadline=time.monotonic() + 10)
    assert result.status == "INFEASIBLE"
    assert not result.selected


def test_pricing_obeys_branch_fixings_even_for_colocated_school():
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, market(), 3)
    lp = MasterResult(
        "OPTIMAL", node_duals=dict.fromkeys(p.nodes, 0.0), zone_duals={0: 0.0, 1: 0.0}
    )
    result = price_zone(
        pool, 0, lp, {(0, 0): 0, (3, 0): 1}, deadline=time.monotonic() + 5
    )
    assert result.status == "OPTIMAL"
    assert 0 not in result.nodes and 3 in result.nodes
    assert result.score == pytest.approx(pool.column(0, result.nodes).score)


def test_interrupted_pricing_never_closes_a_node_or_certifies_optimality():
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, market(), 5)
    incumbent = pool.add_partition({0: 0, 1: 0, 2: 1, 3: 1})

    def interrupted(*args, **kwargs):
        return PricingResult("FEASIBLE", math.inf)

    result = branch_and_price(pool, pricer=interrupted, incumbent=incumbent)
    assert result.status == "FEASIBLE"
    assert result.reason == "pricing_incomplete"
    assert result.upper_bound > sum(c.score for c in result.selected)


def test_phase_one_deficits_cannot_hide_overcoverage():
    p = make_grid_problem(2, 2)
    columns = (
        ZoneColumn(0, frozenset(p.nodes), 0.0, 0),
        ZoneColumn(1, frozenset(p.nodes), 0.0, 0),
    )
    lp = solve_master(p, columns, 5, phase_one=True)
    assert lp.status == "OPTIMAL"
    assert lp.artificial_mass == pytest.approx(1.0)
    assert sum(lp.values) == pytest.approx(1.0)


def test_branching_closes_a_strict_integer_gap_and_reprices_children():
    # The fractional matching on two disjoint triangles has value 3; every
    # integer perfect matching needs a zero-value cross edge, so its value is 2.
    # A complete, enumerative pricer independently exercises the search engine.
    p = make_grid_problem(1, 6)
    p.centroids = [0, 2, 5]
    p.centroid_school_ids = [100, 150, 200]

    class PairPool:
        problem = p
        nodes = frozenset(p.nodes)
        market = SimpleNamespace(types=(SimpleNamespace(utility_sums=(10.0,)),))
        evaluator = None

        def __init__(self):
            self.columns = {}

        def column(self, zone, nodes):
            score = float(all(n < 3 for n in nodes) or all(n >= 3 for n in nodes))
            return ZoneColumn(zone, frozenset(nodes), score, 0)

        def add(self, column):
            key = (column.zone, column.nodes)
            if key in self.columns:
                return False
            self.columns[key] = column
            return True

    pool = PairPool()
    full = tuple(
        pool.column(z, pair) for z in range(3) for pair in combinations(p.nodes, 2)
    )
    assert solve_master(p, full, 5).objective == pytest.approx(3.0)
    assert solve_master(p, full, 5, integer=True).objective == pytest.approx(2.0)
    seen_branches = []

    def enumerate_price(pool, zone, lp, decisions, **kwargs):
        seen_branches.append(dict(decisions))
        candidates = [c for c in full if c.zone == zone and compatible(c, decisions)]
        if not candidates:
            return PricingResult("INFEASIBLE", -math.inf)
        best = max(candidates, key=lp.reduced_cost)
        reduced = lp.reduced_cost(best)
        return PricingResult("OPTIMAL", reduced, best.nodes, best.score, reduced)

    result = branch_and_price(
        pool, pricer=enumerate_price, deadline=time.monotonic() + 15
    )
    assert result.status == "OPTIMAL"
    assert result.nodes > 1
    assert sum(c.score for c in result.selected) == pytest.approx(2.0)
    assert result.upper_bound == pytest.approx(2.0)
    assert any(seen_branches)


def test_zero_time_with_no_incumbent_is_unknown_not_infeasible():
    pool = ZonePool(make_grid_problem(2, 2), market(), 3)
    result = branch_and_price(pool, deadline=time.monotonic() - 1)
    assert result.status == "UNKNOWN"
    assert result.upper_bound is not None


def test_exact_search_improves_beyond_the_seed_pool_optimum():
    pool = ZonePool(make_grid_problem(2, 2), market(), 5)
    incumbent = pool.add_partition({0: 0, 1: 0, 2: 1, 3: 0})
    seed_score = sum(c.score for c in incumbent)
    restricted = solve_master(pool.problem, pool.columns.values(), 5, integer=True)
    assert restricted.objective == pytest.approx(seed_score)
    result = branch_and_price(pool, incumbent=incumbent, deadline=time.monotonic() + 10)
    assert result.status == "OPTIMAL"
    assert sum(c.score for c in result.selected) > seed_score + 1e-6


def test_unlimited_strategy_can_certify_without_recom(monkeypatch):
    from optimization.strategies import get_strategy
    from optimization.strategies import dantzig_wolfe
    from optimization.solvers import get_solver
    from optimization.tests.synthetic import FakeDataset

    dataset = FakeDataset(make_grid_problem(2, 2))
    dataset.config = SimpleNamespace(include_citywide=False, program_population="All")
    monkeypatch.setattr(dantzig_wolfe, "build_mid_market", lambda *_: market())
    strategy = get_strategy(
        "dantzig_wolfe",
        levels=["BlockGroup_0"],
        solve_time_limits=[math.inf],
        dw_recom_samples=0,
        max_iterations=1,
        mid_lottery_scale=3,
    )
    solution = strategy.run(dataset, get_solver("recom", hints="none"))[0]
    assert solution.status == "OPTIMAL"
    assert solution.metadata["dw_seed_partitions"] == 0
    assert solution.metadata["dw_lp_iterations"] > 1  # no iteration truncation
    assert solution.metadata["total_time_limit"] is None


def test_pricing_uses_the_same_balance_feasibility_slack_as_the_pool():
    p = make_grid_problem(2, 2, frl_dev=0.0)
    p.G.nodes[0]["FRL"] = 0.5000005
    pool = ZonePool(p, market(), 3)
    assert pool.feasible(0, frozenset({0}))
    decisions = {(n, 0): int(n == 0) for n in p.nodes}
    result = price_zone(pool, 0, None, decisions, deadline=time.monotonic() + 5)
    assert result.status == "OPTIMAL"
    assert result.nodes == frozenset({0})

"""Compare global pricing and complete search with independent enumeration."""

from dataclasses import replace
from itertools import combinations
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import pytest

from optimization.branch_price import branch_and_price
from optimization.tests.synthetic import make_grid_problem
from optimization.tests.test_dantzig_wolfe import market
from optimization.zone_columns import MasterResult, ZoneColumn, ZonePool, solve_master
from optimization.zone_pricing import (
    PricingResult,
    ZonePricer,
    compatible,
    price_zone,
)


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


def test_every_lp_and_mip_goes_through_gurobi():
    """Pinned deliberately: these two modules were the last pywraplp holdouts.

    SCIP could not find a single nonzero-welfare solution to the real pricing
    MIP in 60s -- its incumbent was worse than the seed column it was handed --
    while Gurobi found clearly improving columns in the same budget. The import
    is the only thing a test can check cheaply, so it checks that.
    """
    import optimization.branch_price as bp
    import optimization.zone_columns as zc
    import optimization.zone_pricing as zp

    for module in (zc, zp, bp):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "pywraplp" not in source, f"{module.__name__} still uses OR-Tools"
    assert zp.gp.__name__ == "gurobipy"
    assert zc.gp.__name__ == "gurobipy"


def test_the_search_builds_one_pricing_model_per_label_not_per_call():
    """Rebuilding is what made pricing hopeless on real data.

    One model there is 30,779 variables and 71,602 rows, and the shipped loop
    rebuilt it per label per column-generation round per branch node. Only the
    branch fixings and the master duals differ between calls, and both are
    attribute updates on a live model.
    """
    p = make_grid_problem(1, 6)
    p.centroids = [0, 2, 5]
    p.centroid_school_ids = [100, 150, 200]
    pool = ZonePool(p, market(), 3)
    result = branch_and_price(pool, deadline=time.monotonic() + 20)

    # Many calls, one model per label.
    assert result.pricing_calls > p.Z
    assert result.pricing_models == p.Z


def test_a_reused_pricing_model_still_respects_each_call_s_branch_fixings():
    """Branch fixings are variable bounds, so they must be released each call.

    A stale bound would silently answer the previous node's question, which is
    exactly the failure mode a persistent model invites.
    """
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, market(), 3)
    lp = MasterResult(
        "OPTIMAL", node_duals=dict.fromkeys(p.nodes, 0.0), zone_duals={0: 0.0, 1: 0.0}
    )
    with ZonePricer(pool) as pricer:
        first = pricer(
            pool, 0, lp, {(0, 0): 0, (3, 0): 1}, deadline=time.monotonic() + 5
        )
        assert first.status == "OPTIMAL"
        assert 0 not in first.nodes and 3 in first.nodes
        # The opposite fixing, on the same model.
        second = pricer(
            pool, 0, lp, {(0, 0): 1, (3, 0): 0}, deadline=time.monotonic() + 5
        )
        assert second.status == "OPTIMAL"
        assert 0 in second.nodes and 3 not in second.nodes
        # And no fixings at all releases both bounds again.
        third = pricer(pool, 0, lp, {}, deadline=time.monotonic() + 5)
        assert third.status == "OPTIMAL"
        assert third.reduced_cost >= max(first.reduced_cost, second.reduced_cost)
        assert pricer.models_built == 1


def test_a_timed_out_pricing_solve_still_yields_its_improving_column():
    """Optimality is needed to *bound*, never to add a column.

    On real data every label timed out at FEASIBLE, so the shipped
    ``status == "OPTIMAL"`` gate threw away every column pricing found and the
    reported bound collapsed to the trivial first-choice constant. A positive
    reduced cost is an improving column whatever the solver's status.
    """
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, market(), 5)
    # A deliberately suboptimal seed, so improving columns certainly exist.
    incumbent = pool.add_partition({0: 0, 1: 0, 2: 1, 3: 0})
    seeded = len(pool.columns)
    full = tuple(all_columns(pool))

    def never_optimal(pool_, zone, lp, decisions, **kwargs):
        """Exact pricing that refuses to ever claim optimality."""
        candidates = [c for c in full if c.zone == zone and compatible(c, decisions)]
        if not candidates:
            return PricingResult("INFEASIBLE", -math.inf)
        best = max(candidates, key=lp.reduced_cost)
        return PricingResult(
            "FEASIBLE", math.inf, best.nodes, best.score, lp.reduced_cost(best)
        )

    result = branch_and_price(
        pool, pricer=never_optimal, incumbent=incumbent, deadline=time.monotonic() + 10
    )
    # Columns landed even though pricing never proved anything...
    assert len(pool.columns) > seeded
    assert any(h["columns_added"] for h in result.history)
    # ...and the node was still never closed, so no false optimality claim.
    assert result.status == "FEASIBLE"
    assert result.reason == "pricing_incomplete"
    assert result.upper_bound > sum(c.score for c in result.selected)


def test_a_zone_the_pool_rejects_is_skipped_rather_than_raising():
    """A MIP works to a tolerance; the pool re-checks feasibility exactly."""
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, market(), 3)
    incumbent = pool.add_partition({0: 0, 1: 0, 2: 1, 3: 1})
    disconnected = frozenset({0, 3})
    assert not pool.feasible(0, disconnected)

    def returns_infeasible_zone(pool_, zone, lp, decisions, **kwargs):
        return PricingResult("FEASIBLE", math.inf, disconnected, 99.0, 99.0)

    result = branch_and_price(
        pool,
        pricer=returns_infeasible_zone,
        incumbent=incumbent,
        deadline=time.monotonic() + 10,
    )
    assert result.reason == "pricing_incomplete"
    assert all(nodes != disconnected for _, nodes in pool.columns)


def test_one_label_cannot_eat_the_whole_pricing_round():
    """A certified bound needs a finite bound from every label.

    The shipped loop handed each pricing call the whole remaining budget, so on
    real data label 0 consumed all of it and labels 1-5 returned unpriced
    before even building a model. ``certified_bound`` is then infinite by
    construction and the reported bound can never leave the trivial constant.
    """
    p = make_grid_problem(1, 6)
    p.centroids = [0, 2, 5]
    p.centroid_school_ids = [100, 150, 200]
    pool = ZonePool(p, market(), 3)
    budgets = []

    def record_budget(pool_, zone, lp, decisions, *, deadline, **kwargs):
        budgets.append(deadline - time.monotonic())
        return PricingResult("FEASIBLE", math.inf)

    started = time.monotonic()
    branch_and_price(pool, pricer=record_budget, deadline=started + 30)
    assert len(budgets) == p.Z
    # A third for the first of three labels, not everything: that is the fix.
    assert budgets[0] == pytest.approx(30.0 / p.Z, abs=0.5), budgets
    # Time a label does not spend rolls forward rather than being discarded, so
    # the shares grow; by the last label there is nobody left to starve. This
    # pricer returns instantly, so nothing is consumed and the growth is pure.
    assert budgets == sorted(budgets)
    assert budgets[-1] == pytest.approx(30.0, abs=0.5), budgets


def test_an_unlimited_search_still_gives_every_label_unlimited_pricing():
    """``solve_time_limits: [.inf]`` must not be sliced into finite shares."""
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, market(), 3)
    budgets = []

    def record_budget(pool_, zone, lp, decisions, *, deadline, **kwargs):
        budgets.append(deadline)
        return PricingResult("OPTIMAL", 0.0)

    branch_and_price(pool, pricer=record_budget, deadline=math.inf)
    assert budgets and all(budget == math.inf for budget in budgets)

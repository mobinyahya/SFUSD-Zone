"""The search engine: pricing against enumeration, and the tree around it."""

from itertools import combinations
import math
from pathlib import Path
import random
import time

import pytest

from optimization.branch_price import branch_and_price
from optimization.tests.synthetic import make_grid_problem
from optimization.tests.test_dantzig_wolfe import (
    individual_market,
    market,
    matching_objective,
)
from optimization.zone_columns import DualPoint, ZoneColumn, ZonePool, solve_master
from optimization.zone_pricing import (
    PricingResult,
    ZonePricer,
    _Harvest,
    compatible,
    price_zone,
)
from optimization.zone_welfare import (
    BoundaryZoneObjective,
    MidZoneObjective,
    StableMatchingZoneObjective,
)


def objectives(scale=5):
    return {
        "boundary": BoundaryZoneObjective(),
        "mid": MidZoneObjective(market(), scale),
        "stable_matching": matching_objective(),
    }


def all_columns(pool, zone=None):
    for label in range(pool.problem.Z) if zone is None else (zone,):
        members = sorted(pool.family.candidates[label])
        for size in range(1, len(members) + 1):
            for chosen in combinations(members, size):
                column = pool.admit(label, frozenset(chosen))
                if column is not None:
                    yield column


def zero_duals(problem, phase_one=False):
    return DualPoint(
        dict.fromkeys(problem.nodes, 0.0),
        dict.fromkeys(range(problem.Z), 0.0),
        0.0,
        phase_one,
    )


# ---------------------------------------------------------------------- #
# Pricing against enumeration
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["boundary", "mid", "stable_matching"])
def test_pricing_reproduces_the_welfare_oracle_at_every_fixed_membership(kind):
    """With membership pinned, the pricing block must equal the exact value.

    This is the tightest statement available about the welfare blocks: the MID
    recurrence and the matching rows are solved by CP-SAT against a zone value
    computed independently -- the least-cutoff oracle and a deferred-acceptance
    run respectively -- at every single admissible zone of the instance.
    """

    p = make_grid_problem(2, 2)
    pool = ZonePool(p, objectives()[kind])
    duals = zero_duals(p)
    with ZonePricer(pool, scale=1000, columns_per_call=1, parallel=False) as pricer:
        checked = 0
        for column in all_columns(pool):
            decisions = {
                (node, column.zone): int(node in column.nodes)
                for node in pool.family.candidates[column.zone]
            }
            result = pricer(
                pool, column.zone, duals, decisions, deadline=time.monotonic() + 30
            )
            assert result.status == "OPTIMAL"
            assert result.bound == pytest.approx(column.score, abs=1e-3)
            checked += 1
    assert checked > 4


@pytest.mark.parametrize("kind", ["boundary", "mid", "stable_matching"])
@pytest.mark.parametrize("phase_one", [False, True])
def test_global_pricing_matches_the_best_of_every_admissible_zone(kind, phase_one):
    # A live boundary cap, so the boundary dual has cut variables to price.
    p = make_grid_problem(2, 2, fixed={0: 0}, candidates={3: {1}}, boundary_prop=0.75)
    pool = ZonePool(p, objectives()[kind])
    columns = tuple(all_columns(pool))
    rng = random.Random(len(kind))
    decisions = {(1, 0): 0}
    with ZonePricer(pool, scale=1000, columns_per_call=4, parallel=False) as pricer:
        for _ in range(3):
            duals = DualPoint(
                {n: rng.uniform(-5, 5) for n in p.nodes},
                {0: 1.37, 1: -0.98},
                0.123,
                phase_one,
            )
            for zone in range(p.Z):
                eligible = [
                    c for c in columns if c.zone == zone and compatible(c, decisions)
                ]
                expected = max(duals.reduced_cost(c) for c in eligible)
                result = pricer(
                    pool,
                    zone,
                    duals,
                    decisions,
                    deadline=time.monotonic() + 30,
                    phase_one=phase_one,
                )
                assert result.status == "OPTIMAL"
                # Valid, and tight up to the reported rounding allowance.
                assert result.bound >= expected - 1e-9
                assert result.bound <= expected + result.allowance + 1e-9
                assert result.reduced_cost == pytest.approx(
                    expected, abs=result.allowance + 1e-9
                )
                for nodes in result.candidates:
                    column = pool.column(zone, nodes)
                    assert compatible(column, decisions)


def test_every_harvested_zone_is_admissible_and_improving():
    """Degenerate pivots are answered with more columns per round, not fewer."""

    p = make_grid_problem(1, 5)
    pool = ZonePool(p, BoundaryZoneObjective())
    duals = DualPoint({n: -1.0 for n in p.nodes}, {0: 0.0, 1: 0.0})
    with ZonePricer(pool, columns_per_call=8, parallel=False) as pricer:
        result = pricer(pool, 0, duals, {}, deadline=time.monotonic() + 20)
    assert result.status == "OPTIMAL"
    assert result.candidates
    assert all(pool.feasible(0, nodes) for nodes in result.candidates)
    assert all(duals.reduced_cost(pool.column(0, n)) > 0 for n in result.candidates)
    assert result.nodes == result.candidates[0]


def test_the_harvest_caps_deduplicates_and_drops_non_improving_solutions():
    """The callback's bookkeeping, without depending on the search's path."""

    harvest = _Harvest([], limit=2)
    harvest.found = [
        (30.0, frozenset({0})),
        (20.0, frozenset({1})),
        (10.0, frozenset({2})),
        (30.0, frozenset({0})),
        (-5.0, frozenset({3})),
    ]
    # Best first, deduplicated, and non-improving solutions dropped.
    assert harvest.improving(1.0, 0.0) == (
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
    )
    assert harvest.improving(1.0, 25.0) == (frozenset({0}),)
    assert harvest.improving(1.0, 100.0) == ()


def test_pricing_respects_branch_fixings_on_a_reused_model():
    """Fixings are CP-SAT assumptions, so they have to be released each call.

    A stale assumption would silently answer the previous node's question,
    which is exactly the failure mode a persistent model invites.
    """

    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 3))
    duals = zero_duals(p)
    with ZonePricer(pool, parallel=False) as pricer:
        first = pricer(
            pool, 0, duals, {(1, 0): 0, (2, 0): 1}, deadline=time.monotonic() + 20
        )
        assert first.status == "OPTIMAL"
        assert 1 not in first.nodes and 2 in first.nodes
        second = pricer(
            pool, 0, duals, {(1, 0): 1, (2, 0): 0}, deadline=time.monotonic() + 20
        )
        assert second.status == "OPTIMAL"
        assert 1 in second.nodes and 2 not in second.nodes
        third = pricer(pool, 0, duals, {}, deadline=time.monotonic() + 20)
        assert third.status == "OPTIMAL"
        assert third.bound >= max(first.bound, second.bound) - 1e-9
        assert pricer.models_built == 1


def test_a_branch_pinning_a_non_candidate_into_a_label_is_infeasible():
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, BoundaryZoneObjective())
    assert 3 not in pool.family.candidates[0]
    with ZonePricer(pool, parallel=False) as pricer:
        result = pricer(
            pool, 0, zero_duals(p), {(3, 0): 1}, deadline=time.monotonic() + 10
        )
    assert result.status == "INFEASIBLE"
    assert result.bound == -math.inf


def test_a_label_with_an_empty_family_prices_as_infeasible():
    p = make_grid_problem(2, 2, frl_dev=0.0)
    p.G.nodes[0]["FRL"] = 1.0
    pool = ZonePool(p, BoundaryZoneObjective())
    # The anchor itself violates the FRL band, so no set containing it is legal.
    assert not pool.family.feasible(0, frozenset({0}))
    result = price_zone(
        pool, 0, zero_duals(p), {}, deadline=time.monotonic() + 10, parallel=False
    )
    assert result.status == "INFEASIBLE"


# ---------------------------------------------------------------------- #
# The tree
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["boundary", "mid", "stable_matching"])
@pytest.mark.parametrize("cap", [-1, 0.5])
def test_empty_pool_reaches_the_exhaustive_global_optimum(kind, cap):
    p = make_grid_problem(2, 2, boundary_prop=cap)
    pool = ZonePool(p, objectives()[kind])
    enumerated = solve_master(p, tuple(all_columns(pool)), 5, integer=True)
    assert not pool.columns  # Enumeration above did not seed the search.
    result = branch_and_price(
        pool, deadline=time.monotonic() + 60, pricing_parallel=False
    )
    assert result.status == "OPTIMAL"
    assert sum(c.score for c in result.selected) == pytest.approx(enumerated.objective)
    assert result.upper_bound >= enumerated.objective - 1e-9
    assert result.upper_bound <= enumerated.objective + result.bound_slack + 1e-9
    assert any(h["phase_one"] and h["columns_added"] for h in result.history)


def test_infeasible_partition_is_proved_by_phase_one_pricing():
    p = make_grid_problem(2, 2, boundary_prop=0.0)
    pool = ZonePool(p, MidZoneObjective(market(), 3))
    result = branch_and_price(
        pool, deadline=time.monotonic() + 30, pricing_parallel=False
    )
    assert result.status == "INFEASIBLE"
    assert not result.selected


def test_exact_search_improves_beyond_the_seed_pool_optimum():
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 5))
    incumbent = pool.add_partition({0: 0, 1: 1, 2: 1, 3: 1})
    seed_score = sum(c.score for c in incumbent)
    restricted = solve_master(p, tuple(pool.columns.values()), 5, integer=True)
    assert restricted.objective == pytest.approx(seed_score)
    result = branch_and_price(
        pool,
        incumbent=incumbent,
        deadline=time.monotonic() + 60,
        pricing_parallel=False,
    )
    assert result.status == "OPTIMAL"
    assert sum(c.score for c in result.selected) > seed_score + 1e-6


def test_dual_smoothing_leaves_the_optimum_and_the_bound_alone():
    """Smoothing changes where pricing aims, not what the search may conclude."""

    p = make_grid_problem(2, 2)
    reference = None
    for smoothing in (1.0, 0.5, 0.2):
        pool = ZonePool(p, MidZoneObjective(market(), 5))
        enumerated = solve_master(p, tuple(all_columns(pool)), 5, integer=True)
        result = branch_and_price(
            pool,
            deadline=time.monotonic() + 60,
            dual_smoothing=smoothing,
            pricing_parallel=False,
        )
        assert result.status == "OPTIMAL"
        assert sum(c.score for c in result.selected) == pytest.approx(
            enumerated.objective
        )
        assert result.upper_bound >= enumerated.objective - 1e-9
        reference = reference or enumerated.objective


@pytest.mark.parametrize("method", ["barrier", "dual"])
def test_both_dual_points_reach_the_same_certified_optimum(method):
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 5))
    enumerated = solve_master(p, tuple(all_columns(pool)), 5, integer=True)
    result = branch_and_price(
        pool,
        deadline=time.monotonic() + 60,
        master_method=method,
        pricing_parallel=False,
    )
    assert result.status == "OPTIMAL"
    assert sum(c.score for c in result.selected) == pytest.approx(enumerated.objective)


def test_parallel_and_sequential_pricing_agree():
    p = make_grid_problem(2, 2)
    values = []
    for parallel in (False, True):
        pool = ZonePool(p, MidZoneObjective(market(), 5))
        result = branch_and_price(
            pool,
            deadline=time.monotonic() + 60,
            workers=2,
            pricing_parallel=parallel,
        )
        assert result.status == "OPTIMAL"
        values.append(sum(c.score for c in result.selected))
    assert values[0] == pytest.approx(values[1])


def test_branching_closes_a_strict_integer_gap_and_reprices_children():
    # The fractional matching on two disjoint triangles has value 3; every
    # integer perfect matching needs a zero-value cross edge, so its value is 2.
    # A complete, enumerative pricer independently exercises the search engine.
    p = make_grid_problem(1, 6, schools={0: 100, 2: 150, 5: 200})

    class FakeObjective(BoundaryZoneObjective):
        def upper_bound(self):
            # Three labels, one unit each: valid for this fixture's scores.
            return 3.0

    class PairPool:
        problem = p
        nodes = frozenset(p.nodes)
        objective = FakeObjective()

        def __init__(self):
            self.columns = {}

        def column(self, zone, nodes):
            score = float(all(n < 3 for n in nodes) or all(n >= 3 for n in nodes))
            return ZoneColumn(zone, frozenset(nodes), score, 0)

        def admit(self, zone, nodes):
            return self.column(zone, nodes)

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

    def enumerate_price(pool, zone, duals, decisions, **kwargs):
        seen_branches.append(dict(decisions))
        candidates = [c for c in full if c.zone == zone and compatible(c, decisions)]
        if not candidates:
            return PricingResult("INFEASIBLE", -math.inf)
        best = max(candidates, key=duals.reduced_cost)
        reduced = duals.reduced_cost(best)
        return PricingResult("OPTIMAL", reduced, best.nodes, reduced, (best.nodes,))

    result = branch_and_price(
        pool, pricer=enumerate_price, deadline=time.monotonic() + 15
    )
    assert result.status == "OPTIMAL"
    assert result.nodes > 1
    assert sum(c.score for c in result.selected) == pytest.approx(2.0)
    assert result.upper_bound == pytest.approx(2.0)
    assert any(seen_branches)


def test_interrupted_pricing_never_closes_a_node_or_certifies_optimality():
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 5))
    incumbent = pool.add_partition({0: 0, 1: 0, 2: 1, 3: 1})

    def interrupted(*args, **kwargs):
        return PricingResult("FEASIBLE", math.inf)

    result = branch_and_price(pool, pricer=interrupted, incumbent=incumbent)
    assert result.status == "FEASIBLE"
    assert result.reason == "pricing_incomplete"
    assert result.upper_bound > sum(c.score for c in result.selected)


def test_a_timed_out_pricing_solve_still_yields_its_improving_column():
    """Optimality is needed to *bound*, never to add a column.

    On real data every label timed out, so a ``status == "OPTIMAL"`` gate threw
    away every column pricing found and the reported bound collapsed to the
    trivial first-choice constant. A positive reduced cost is an improving
    column whatever the solver's status.
    """

    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 5))
    incumbent = pool.add_partition({0: 0, 1: 1, 2: 1, 3: 1})
    seeded = len(pool.columns)
    full = tuple(all_columns(pool))

    def never_optimal(pool_, zone, duals, decisions, **kwargs):
        """Exact pricing that refuses to ever claim optimality."""

        candidates = [c for c in full if c.zone == zone and compatible(c, decisions)]
        if not candidates:
            return PricingResult("INFEASIBLE", -math.inf)
        best = max(candidates, key=duals.reduced_cost)
        return PricingResult(
            "FEASIBLE",
            math.inf,
            best.nodes,
            duals.reduced_cost(best),
            (best.nodes,),
        )

    result = branch_and_price(
        pool, pricer=never_optimal, incumbent=incumbent, deadline=time.monotonic() + 10
    )
    assert len(pool.columns) > seeded
    assert any(h["columns_added"] for h in result.history)
    assert result.status == "FEASIBLE"
    assert result.reason == "pricing_incomplete"
    assert result.upper_bound > sum(c.score for c in result.selected)


def test_a_zone_the_family_rejects_is_skipped_rather_than_raising():
    """A pricing solve works to a tolerance; the family re-checks exactly."""

    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 3))
    incumbent = pool.add_partition({0: 0, 1: 0, 2: 1, 3: 1})
    inadmissible = frozenset({1, 2})
    assert not pool.feasible(0, inadmissible)

    def returns_bad_zone(pool_, zone, duals, decisions, **kwargs):
        return PricingResult("FEASIBLE", math.inf, inadmissible, 99.0, (inadmissible,))

    result = branch_and_price(
        pool,
        pricer=returns_bad_zone,
        incumbent=incumbent,
        deadline=time.monotonic() + 10,
    )
    assert result.reason == "pricing_incomplete"
    assert all(nodes != inadmissible for _, nodes in pool.columns)


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


def test_zero_time_with_no_incumbent_is_unknown_not_infeasible():
    pool = ZonePool(make_grid_problem(2, 2), MidZoneObjective(market(), 3))
    result = branch_and_price(pool, deadline=time.monotonic() - 1)
    assert result.status == "UNKNOWN"
    assert result.upper_bound is not None


def test_the_search_builds_one_pricing_model_per_label_and_phase():
    p = make_grid_problem(1, 6, schools={0: 100, 2: 150, 5: 200})
    pool = ZonePool(p, MidZoneObjective(market(schools=(0, 5)), 3))
    result = branch_and_price(
        pool, deadline=time.monotonic() + 60, pricing_parallel=False
    )
    assert result.pricing_calls > p.Z
    # One Phase-I model and one Phase-II model per label, and no more.
    assert result.pricing_models <= 2 * p.Z
    assert result.pricing_models >= p.Z


def test_one_label_cannot_eat_the_whole_pricing_round():
    """A certified bound needs a finite bound from every label.

    An earlier loop handed each pricing call the whole remaining budget, so on
    real data label 0 consumed all of it and labels 1-5 returned unpriced
    before even building a model.
    """

    p = make_grid_problem(1, 6, schools={0: 100, 2: 150, 5: 200})
    pool = ZonePool(p, MidZoneObjective(market(schools=(0, 5)), 3))
    budgets = []

    def record_budget(pool_, zone, duals, decisions, *, deadline, **kwargs):
        budgets.append(deadline - time.monotonic())
        return PricingResult("FEASIBLE", math.inf)

    started = time.monotonic()
    branch_and_price(pool, pricer=record_budget, deadline=started + 30)
    assert len(budgets) == p.Z
    # A third for the first of three labels, not everything: that is the fix.
    assert budgets[0] == pytest.approx(30.0 / p.Z, abs=0.5), budgets
    # Time a label does not spend rolls forward rather than being discarded, so
    # the shares grow; by the last label there is nobody left to starve.
    assert budgets == sorted(budgets)
    assert budgets[-1] == pytest.approx(30.0, abs=0.5), budgets


def test_an_unlimited_search_still_gives_every_label_unlimited_pricing():
    """``solve_time_limits: [.inf]`` must not be sliced into finite shares."""

    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 3))
    budgets = []

    def record_budget(pool_, zone, duals, decisions, *, deadline, **kwargs):
        budgets.append(deadline)
        return PricingResult("OPTIMAL", 0.0)

    branch_and_price(pool, pricer=record_budget, deadline=math.inf)
    assert budgets and all(budget == math.inf for budget in budgets)


def test_the_master_is_gurobi_and_the_subproblem_is_cp_sat():
    """Deliberate split, and both halves are load-bearing.

    Gurobi for the LP because its duals *are* the pricing objective and the
    OR-Tools LP backend is not used anywhere in this repo. CP-SAT for the
    subproblem because the MID ``min`` recurrence is a native constraint there
    and a big-``M`` disjunction in a MIP -- about 24,000 auxiliary integers on
    the real instance, and the weakest part of that relaxation.
    """

    import optimization.branch_price as bp
    import optimization.zone_columns as zc
    import optimization.zone_pricing as zp

    for module in (zc, zp, bp):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "pywraplp" not in source, f"{module.__name__} uses an OR-Tools LP"
    assert zc.gp.__name__ == "gurobipy"
    assert zp.cp_model.__name__.startswith("ortools")
    assert not hasattr(zp, "gp")


def test_individual_and_compressed_markets_describe_the_same_cohort():
    """The two welfare definitions must be scoring the same students."""

    compressed = MidZoneObjective(market(), 5)
    sampled = StableMatchingZoneObjective(individual_market(), seed=1)
    assert sum(t.count for t in compressed.market.types) == len(sampled.market.students)
    assert {p.school_node for p in compressed.market.programs} == {
        p.school_node for p in sampled.market.programs
    }


class _EndlessPricer:
    """Proves a zero residual every round and always offers another column.

    This is the situation a district-sized instance is actually in: pricing
    keeps finding columns of strictly positive reduced cost long after the
    node's certified bound has stopped moving. A zero residual makes
    ``certified`` the restricted LP's own dual objective, so with the pool's
    best tiling as the incumbent the node is closable on the first round.
    """

    def __init__(self, pool):
        self.pool = pool
        self.supply = {
            zone: [
                column.nodes
                for column in all_columns(pool, zone)
                if (zone, column.nodes) not in pool.columns
            ]
            for zone in range(pool.problem.Z)
        }
        self.calls = 0

    def __call__(self, pool, zone, duals, decisions, *, deadline, phase_one=False):
        self.calls += 1
        remaining = self.supply[zone]
        nodes = remaining.pop() if remaining else frozenset()
        return PricingResult(
            "OPTIMAL", 0.0, nodes, 1.0, (nodes,) if nodes else (), 0.0, 0.0
        )


def test_a_node_closes_as_soon_as_its_bound_meets_the_incumbent():
    """The pruning test must not wait for column generation to run dry.

    Pricing on a real instance never runs dry, so evaluating the
    incumbent-versus-bound test only after ``if added: continue`` means a node
    whose incumbent already attains its certified bound keeps generating
    columns until the wall clock stops it. Measured before this was fixed: the
    incumbent reached the certified bound in round 3 of 12 and the search
    still consumed its whole budget and reported ``time_limit``.
    """

    problem = make_grid_problem(1, 5)
    pool = ZonePool(problem, MidZoneObjective(market(), 5))
    best = max(
        (
            columns
            for columns in (
                tuple(
                    pool.admit(zone, frozenset(nodes))
                    for zone, nodes in enumerate(split)
                )
                for split in ((range(cut), range(cut, 5)) for cut in range(1, 5))
            )
            if all(column is not None for column in columns)
        ),
        key=lambda columns: sum(column.score for column in columns),
    )
    for column in best:
        pool.add(column)
    pricer = _EndlessPricer(pool)
    result = branch_and_price(
        pool,
        deadline=math.inf,
        incumbent=best,
        pricer=pricer,
        pricing_parallel=False,
    )
    assert result.status == "OPTIMAL"
    assert result.reason == "tree_exhausted"
    # One Phase-I solve that finds no artificial mass, one Phase-II solve, one
    # pricing round per label, then closed -- not a walk through the pricer's
    # whole supply.
    assert result.lp_iterations == 2
    assert len(result.history) == 1
    assert pricer.calls == problem.Z
    assert sum(column.score for column in result.selected) == pytest.approx(
        sum(column.score for column in best)
    )


# ---------------------------------------------------------------------- #
# The budgeted-elastic master, inside the search
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("overlap_prop", [0.1, 0.5])
def test_an_overlap_budget_leaves_the_optimum_and_the_bound_valid(overlap_prop):
    """Elasticity changes where the LP may go, never what may be concluded.

    The relaxation is looser at every positive ``K``, so the bound is allowed
    to be weaker -- but it must still *be* a bound, and the incumbent it
    returns must still be a tiling, because only the exact integer master ever
    produces one.
    """

    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 5))
    enumerated = solve_master(p, tuple(all_columns(pool)), 5, integer=True)
    result = branch_and_price(
        pool,
        deadline=time.monotonic() + 60,
        overlap_prop=overlap_prop,
        pricing_parallel=False,
    )
    assert result.selected
    assert sum(c.score for c in result.selected) == pytest.approx(enumerated.objective)
    assert result.upper_bound >= enumerated.objective - 1e-9
    # Still a partition: the elastic rows are Phase-II LP only.
    assert {c.zone for c in result.selected} == set(range(p.Z))
    assert frozenset().union(*(c.nodes for c in result.selected)) == pool.nodes
    assert sum(len(c.nodes) for c in result.selected) == p.A


def test_the_budget_diagnostics_are_reported_only_when_it_is_enabled():
    """``K`` is tuned by sweeping it, which needs these five numbers per round.

    They are withheld at ``K = 0`` because the history goes into every saved
    run and five always-zero keys per round is not free there.
    """

    p = make_grid_problem(2, 2)
    keys = {
        "overlap_budget",
        "overlap_dual",
        "elastic_mass",
        "elastic_nodes",
        "duals_pinned",
    }
    for overlap_prop, expected in ((0.0, False), (0.25, True)):
        pool = ZonePool(p, MidZoneObjective(market(), 5))
        result = branch_and_price(
            pool,
            deadline=time.monotonic() + 60,
            overlap_prop=overlap_prop,
            pricing_parallel=False,
        )
        phase_two = [h for h in result.history if not h["phase_one"]]
        assert phase_two
        assert all((keys <= set(h)) is expected for h in phase_two)
        # Phase I is never elastic whatever K is.
        assert all(
            h.get("overlap_budget", 0.0) == 0.0
            for h in result.history
            if h["phase_one"]
        )


def test_the_search_shrinks_the_budget_rather_than_calling_a_non_tiling_integral():
    """The one dead end elasticity creates, and its exit.

    Integral marginals with mismatch still spent leaves nothing fractional to
    branch on and an LP that is not a partition. Halving ``K`` is the exit, and
    ``K = 0`` is the exact master, so it terminates. The pool here holds one
    tiling and one colliding column worth far more, so the first Phase-II LP
    puts all its weight on the collision and takes that path.
    """

    p = make_grid_problem(1, 6)
    # Scores stay inside BoundaryZoneObjective's own a-priori bound of zero, so
    # the root node is not closed against the incumbent before it is priced.
    columns = (
        ZoneColumn(0, frozenset({0, 1, 2}), -1.0, 0),
        ZoneColumn(1, frozenset({3, 4, 5}), -1.0, 0),
        ZoneColumn(0, frozenset({0, 1, 2, 3}), -0.001, 0),
    )

    def priced_out(pool_, zone, duals, decisions, **kwargs):
        """Never offers a column, so the LP is the only thing moving."""

        return PricingResult("OPTIMAL", 0.0, frozenset(), 0.0, ())

    pool = ZonePool(p, BoundaryZoneObjective())
    for column in columns:
        pool.add(column)
    result = branch_and_price(
        pool,
        pricer=priced_out,
        incumbent=columns[:2],
        deadline=time.monotonic() + 30,
        overlap_prop=1.0,
    )
    phase_two = [h for h in result.history if not h["phase_one"]]
    budgets = [h["overlap_budget"] for h in phase_two]
    assert len(budgets) > 1
    # Monotone down, and it really did move: K never rises, so the bound the
    # node finally certifies is the tighter one.
    assert budgets == sorted(budgets, reverse=True)
    assert budgets[-1] < budgets[0]
    assert max(h["elastic_mass"] for h in phase_two) > 1e-6
    # The collision was never reported as a solution.
    assert sum(c.score for c in result.selected) == pytest.approx(-2.0)
    assert frozenset().union(*(c.nodes for c in result.selected)) == pool.nodes

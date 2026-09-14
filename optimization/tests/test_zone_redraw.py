"""Two-zone redraws: the neighbourhood, the theory that lets it be additive.

The single-zone pricer is tested in ``test_branch_price.py`` against
exhaustive enumeration of every admissible zone. This module tests the other
column source, and the three claims that make it safe to run alongside
pricing rather than instead of it:

* it solves its neighbourhood *exactly* -- the best re-partition of two zones'
  joint territory, checked against enumeration of all partitions;
* every solution it passes through is a complete admissible tiling, so it is
  the only source here that grows the number of ways the master can cover
  ``V``; and
* it needs no dual point, because the cover prices on the joint territory
  cancel between any two of its solutions.
"""

from itertools import combinations, product
import math
import random
from types import SimpleNamespace

import pytest

from optimization.branch_price import branch_and_price
from optimization.data.mid import MidMarket, MidProgram, MidStudent, MidType
from optimization.data.saa import SaaMarket
from optimization.solvers import get_solver
from optimization.strategies import get_strategy
from optimization.strategies import dantzig_wolfe as dw
from optimization.tests.synthetic import FakeDataset, make_grid_problem
from optimization.tests.test_dantzig_wolfe import individual_market, market
from optimization.zone_columns import DualPoint, ZonePool, solve_master
from optimization.zone_family import boundary_limit, perimeter
from optimization.zone_redraw import ZoneRedraw, redraw_pair
from optimization.zone_welfare import (
    BoundaryZoneObjective,
    MidZoneObjective,
    StableMatchingZoneObjective,
)


_UTILITIES = (6.0, 4.0, 2.0, 1.0)


def _ranked(problem, schools, node):
    """The school nodes, nearest first, as ranks into ``schools``."""

    distances = problem.G.graph["distance_dict"]
    return sorted(
        range(len(schools)), key=lambda i: (distances[schools[i]][node], i)
    )


def _programs(schools, capacity):
    return tuple(
        MidProgram(f"P{i}", 100 + 10 * i, capacity, False, node)
        for i, node in enumerate(schools)
    )


def mid_market(problem, schools, *, capacity=3, count=1):
    """Every node ranks one program per school node, nearest first."""

    programs = _programs(schools, capacity)
    types = tuple(
        MidType(
            node,
            count,
            tuple(programs[i].program_id for i in _ranked(problem, schools, node)),
            tuple(0 for _ in schools),
            _UTILITIES[: len(schools)],
            tuple(int(1000 * value) for value in _UTILITIES[: len(schools)]),
        )
        for node in sorted(problem.nodes)
    )
    total = count * problem.A
    return MidMarket(programs, types, total, 0, total, "omit_nonpositive")


def saa_market(problem, schools, *, capacity=3, count=1):
    """The same preferences, un-compressed, with every priority tied."""

    programs = _programs(schools, capacity)
    students = tuple(
        MidStudent(
            node,
            tuple(programs[i].program_id for i in _ranked(problem, schools, node)),
            tuple(0 for _ in schools),
            _UTILITIES[: len(schools)],
            tuple(int(1000 * value) for value in _UTILITIES[: len(schools)]),
        )
        for node in sorted(problem.nodes)
        for _ in range(count)
    )
    return SaaMarket(programs, students, len(students), "omit_nonpositive")


def build_pool(kind, problem, schools, *, lottery=5, capacity=2):
    if kind == "boundary":
        objective = BoundaryZoneObjective()
    elif kind == "mid":
        objective = MidZoneObjective(
            mid_market(problem, schools, capacity=capacity), lottery
        )
    elif kind == "stable_matching":
        objective = StableMatchingZoneObjective(
            saa_market(problem, schools, capacity=capacity),
            tie_breaking_method="STB",
            seed=7,
        )
    else:  # pragma: no cover - guard
        raise ValueError(kind)
    return ZonePool(problem, objective)


def admissible_partitions(pool):
    """Every admissible tiling of the instance, by brute force."""

    problem = pool.problem
    nodes = sorted(problem.nodes)
    for labels in product(range(problem.Z), repeat=len(nodes)):
        assignment = dict(zip(nodes, labels))
        columns = []
        for zone in range(problem.Z):
            column = pool.admit(
                zone, frozenset(n for n, z in assignment.items() if z == zone)
            )
            if column is None:
                break
            columns.append(column)
        if len(columns) == problem.Z:
            yield tuple(columns)


def pooled_tilings(pool):
    """Every tiling the *pool* can assemble, one column per label."""

    by_label = [
        [c for c in pool.columns.values() if c.zone == zone]
        for zone in range(pool.problem.Z)
    ]
    found = []
    for combination in product(*by_label):
        covered = frozenset().union(*(c.nodes for c in combination))
        if covered == pool.nodes and sum(
            len(c.nodes) for c in combination
        ) == pool.problem.A:
            found.append(combination)
    return found


#: A lopsided but admissible tiling of the seven-vertex path: the two end
#: zones hold their anchor alone and the middle zone holds everything else.
#: Program capacity is below a zone's student count there, so the seed is far
#: from optimal and the redraw has something to find -- with a *balanced*
#: seed every applicant already takes their first choice, the seed attains the
#: a-priori welfare bound, and a branch node closes before any work happens.
LOPSIDED = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}


def three_zone_pool(kind, *, capacity=2, assignment=None, **overrides):
    """A seven-vertex path anchored at both ends and the middle."""

    problem = make_grid_problem(1, 7, schools={0: 100, 3: 110, 6: 120}, **overrides)
    pool = build_pool(kind, problem, (0, 3, 6), capacity=capacity)
    seed = pool.add_partition(dict(assignment or LOPSIDED))
    return problem, pool, seed


def territory_splits(pool, seed, pair):
    """Every admissible re-partition of a pair's joint territory."""

    by_label = {column.zone: column for column in seed}
    territory = by_label[pair[0]].nodes | by_label[pair[1]].nodes
    members = sorted(territory)
    found = []
    for size in range(1, len(members)):
        for chosen in combinations(members, size):
            first = pool.admit(pair[0], frozenset(chosen))
            second = pool.admit(pair[1], territory - frozenset(chosen))
            if first is not None and second is not None:
                found.append((first, second))
    return found


# ---------------------------------------------------------------------- #
# Exactness of the neighbourhood
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["boundary", "mid", "stable_matching"])
def test_redraw_reaches_the_best_partition_of_the_joint_territory(kind):
    """With two labels, the pair's territory is ``V``, so the redraw is exact.

    That makes this the strongest available statement about the model: the
    optimum it returns is compared against the exact welfare of every
    admissible partition of the instance, enumerated.
    """

    problem = make_grid_problem(1, 5)
    pool = build_pool(kind, problem, (0, 4))
    seed = pool.add_partition({0: 0, 1: 0, 2: 0, 3: 1, 4: 1})
    result = redraw_pair(
        pool, seed, (0, 1), deadline=math.inf, workers=1, splits_per_call=32
    )
    best = max(
        sum(column.score for column in columns)
        for columns in admissible_partitions(pool)
    )
    assert result.status == "OPTIMAL"
    assert result.score == pytest.approx(best)
    assert result.incumbent_score == pytest.approx(sum(c.score for c in seed))
    # The scaled objective rounds utilities upward only, so its proven optimum
    # never sits below the exact one, and never above it by more than the
    # allowance.
    assert result.bound >= best - 1e-9
    assert result.bound <= best + result.allowance + 1e-9


@pytest.mark.parametrize("kind", ["boundary", "mid", "stable_matching"])
def test_redraw_of_a_pair_is_exact_over_a_strict_sub_territory(kind):
    """With three labels the territory is a strict subset of ``V``.

    The optimum is checked against enumeration of every admissible
    re-partition of that subset, and every solution is checked to complete the
    untouched third zone into a tiling of the whole district.
    """

    problem, pool, seed = three_zone_pool(kind)
    frozen = next(column for column in seed if column.zone == 2)
    territory = frozenset({0, 1, 2, 3, 4, 5})
    result = redraw_pair(
        pool, seed, (0, 1), deadline=math.inf, workers=1, splits_per_call=32
    )
    enumerated = territory_splits(pool, seed, (0, 1))
    assert enumerated
    best = max(sum(c.score for c in columns) for columns in enumerated)
    assert result.status == "OPTIMAL"
    assert result.score == pytest.approx(best)
    assert {column.zone for column in result.columns} == {0, 1}
    assert result.territory == len(territory)
    for columns in result.splits:
        assert frozenset().union(*(c.nodes for c in columns)) == territory
        # ... and therefore complete zone 2's untouched column into a tiling.
        whole = (*columns, frozen)
        assert frozenset().union(*(c.nodes for c in whole)) == pool.nodes
        assert sum(len(c.nodes) for c in whole) == problem.A


@pytest.mark.parametrize("kind", ["boundary", "mid", "stable_matching"])
def test_every_harvested_split_is_admissible_for_both_labels(kind):
    problem, pool, seed = three_zone_pool(kind)
    redraw = ZoneRedraw(pool, workers=1, splits_per_call=32)
    result = redraw(seed, (1, 2), deadline=math.inf)
    assert result.splits
    for columns in result.splits:
        for column in columns:
            assert pool.family.feasible(column.zone, column.nodes)
            assert column.score == pytest.approx(
                pool.objective.score(column.nodes, perimeter(problem, column.nodes))
            )
    # Best first, by the oracle's exact score rather than the scaled one.
    scores = [sum(c.score for c in columns) for columns in result.splits]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------- #
# Why no duals are needed
# ---------------------------------------------------------------------- #
def test_the_cover_prices_cancel_over_the_joint_territory():
    """Welfare and reduced cost rank the pair's splits identically.

    Every solution covers ``U`` exactly once, so ``sum_{v in U} pi_v`` is the
    same number for all of them and cannot change which is best. This is why
    the redraw runs with no master, no LP and no dual point, and why
    maximizing pair welfare also maximizes the pair's summed reduced cost.
    """

    problem, pool, seed = three_zone_pool("mid")
    pair = (0, 1)
    enumerated = territory_splits(pool, seed, pair)
    assert len(enumerated) > 1, "need at least two splits to compare"
    by_label = {column.zone: column for column in seed}
    held = tuple(by_label[label] for label in pair)
    rng = random.Random(11)
    for _ in range(8):
        duals = DualPoint(
            {node: rng.uniform(-3.0, 3.0) for node in problem.nodes},
            {zone: rng.uniform(-3.0, 3.0) for zone in range(problem.Z)},
        )
        base_welfare = sum(column.score for column in held)
        base_reduced = sum(duals.reduced_cost(column) for column in held)
        for columns in enumerated:
            welfare_gap = sum(c.score for c in columns) - base_welfare
            reduced_gap = (
                sum(duals.reduced_cost(c) for c in columns) - base_reduced
            )
            assert reduced_gap == pytest.approx(welfare_gap)


def test_an_improving_redraw_gives_the_pool_a_tiling_it_did_not_have():
    """The measured obstruction, directly: one tiling in, several out."""

    problem, pool, seed = three_zone_pool("mid")
    assert len(pooled_tilings(pool)) == 1
    redraw = ZoneRedraw(pool, workers=1, splits_per_call=32)
    final, metadata = redraw.sweep(seed, deadline=math.inf)
    assert metadata["dw_redraw_columns_added"] > 0
    tilings = pooled_tilings(pool)
    assert len(tilings) > 1
    # ... and the restricted master can now reach a better partition than the
    # seed without any pricing having happened.
    columns = tuple(pool.columns.values())
    mip = solve_master(problem, columns, 30.0, integer=True)
    assert mip.status == "OPTIMAL"
    assert mip.objective > sum(column.score for column in seed)
    assert sum(column.score for column in final) == pytest.approx(mip.objective)


# ---------------------------------------------------------------------- #
# The neighbourhood's structure
# ---------------------------------------------------------------------- #
def test_pairs_are_the_graph_adjacent_labels_most_shared_boundary_first():
    _, pool, seed = three_zone_pool("boundary")
    redraw = ZoneRedraw(pool, workers=1)
    assert redraw.pairs(seed) == ((0, 1), (1, 2))


def test_a_non_adjacent_pair_cannot_move_a_single_block():
    """The claim that justifies not enumerating non-adjacent pairs."""

    _, pool, seed = three_zone_pool("mid")
    by_label = {column.zone: column for column in seed}
    assert (0, 2) not in ZoneRedraw(pool, workers=1).pairs(seed)
    result = redraw_pair(pool, seed, (0, 2), deadline=math.inf, workers=1)
    assert result.status == "OPTIMAL"
    assert result.gain == 0.0
    assert len(result.splits) == 1
    assert {column.zone: column.nodes for column in result.splits[0]} == {
        0: by_label[0].nodes,
        2: by_label[2].nodes,
    }


def test_a_pair_is_exhausted_by_its_own_redraw_and_a_sweep_terminates():
    """A pair's territory is invariant under its own redraw.

    So an optimally solved pair stays optimal until another pair disturbs one
    of its two zones, the memo needs no invalidation logic, and a sweep that
    finds no improvement re-solves nothing at all.
    """

    _, pool, seed = three_zone_pool("mid")
    redraw = ZoneRedraw(pool, workers=1, splits_per_call=32)
    final, first = redraw.sweep(seed, deadline=math.inf)
    assert first["dw_redraw_pairs_solved"] > 0
    _, second = redraw.sweep(final, deadline=math.inf)
    assert second["dw_redraw_pairs_solved"] == 0
    assert second["dw_redraw_improvements"] == 0
    assert second["dw_redraw_columns_added"] == 0
    # A swept partition is two-zone optimal: no pair improves it.
    for pair in ZoneRedraw(pool, workers=1).pairs(final):
        assert redraw_pair(
            pool, final, pair, deadline=math.inf, workers=1
        ).gain == pytest.approx(0.0)


def test_sweeping_never_lowers_the_incumbent_and_reports_its_gain():
    _, pool, seed = three_zone_pool("stable_matching")
    redraw = ZoneRedraw(pool, workers=1)
    final, metadata = redraw.sweep(seed, deadline=math.inf)
    before = sum(column.score for column in seed)
    after = sum(column.score for column in final)
    assert after >= before
    assert metadata["dw_redraw_welfare_gain"] == pytest.approx(after - before)
    assert redraw.welfare_gain == pytest.approx(after - before)
    assert {column.zone for column in final} == set(range(pool.problem.Z))


def test_splits_per_call_bounds_the_harvest():
    _, pool, seed = three_zone_pool("mid")
    wide = redraw_pair(
        pool, seed, (0, 1), deadline=math.inf, workers=1, splits_per_call=32
    )
    narrow = redraw_pair(
        pool, seed, (0, 1), deadline=math.inf, workers=1, splits_per_call=1
    )
    assert len(narrow.splits) <= 1
    assert len(wide.splits) >= len(narrow.splits)
    assert narrow.score == pytest.approx(wide.score)


def test_a_spent_deadline_returns_without_building_a_model():
    _, pool, seed = three_zone_pool("mid")
    result = redraw_pair(pool, seed, (0, 1), deadline=-math.inf, workers=1)
    assert result.status == "TIME_LIMIT"
    assert result.splits == ()
    assert result.improved is False
    final, metadata = ZoneRedraw(pool, workers=1).sweep(seed, deadline=-math.inf)
    assert final == seed
    assert metadata["dw_redraw_pairs_solved"] == 0


def test_a_malformed_incumbent_is_rejected():
    _, pool, seed = three_zone_pool("boundary")
    redraw = ZoneRedraw(pool, workers=1)
    with pytest.raises(ValueError):
        redraw(seed[:-1], (0, 1), deadline=math.inf)
    with pytest.raises(ValueError):
        redraw((seed[0], seed[0], seed[1]), (0, 1), deadline=math.inf)
    with pytest.raises(ValueError):
        redraw(seed, (1, 1), deadline=math.inf)


# ---------------------------------------------------------------------- #
# Branch fixings and the boundary cap
# ---------------------------------------------------------------------- #
def test_redraw_honours_the_branch_fixings_it_is_given():
    _, pool, seed = three_zone_pool("mid")
    free = redraw_pair(
        pool, seed, (0, 1), deadline=math.inf, workers=1, splits_per_call=32
    )
    assert free.status == "OPTIMAL"

    for value in (0, 1):
        pinned = redraw_pair(
            pool,
            seed,
            (0, 1),
            deadline=math.inf,
            decisions={(2, 0): value},
            workers=1,
            splits_per_call=32,
        )
        assert pinned.status == "OPTIMAL"
        assert pinned.splits
        for columns in pinned.splits:
            held = {column.zone: column.nodes for column in columns}
            assert (2 in held[0]) is bool(value)
        # A fixing can only remove candidates, so it can only lose welfare.
        assert pinned.score <= free.score + 1e-9

    # Pinning a block the pair cannot reach leaves nothing to search.
    unreachable = redraw_pair(
        pool, seed, (0, 1), deadline=math.inf, decisions={(6, 0): 1}, workers=1
    )
    assert unreachable.status == "INFEASIBLE"


def test_a_sweep_skips_an_incumbent_the_branch_has_already_excluded():
    _, pool, seed = three_zone_pool("mid")
    redraw = ZoneRedraw(pool, workers=1)
    final, metadata = redraw.sweep(
        seed, deadline=math.inf, decisions={(1, 0): 1}
    )
    assert metadata["dw_redraw_status"] == "incumbent_incompatible"
    assert final == seed
    assert metadata["dw_redraw_pairs_solved"] == 0


def test_redraw_enforces_the_district_boundary_cap_exactly():
    """The other zones' perimeters are fixed, so the cap is a row, not a price.

    Every three-zone partition of a path cuts exactly two edges, so on the
    six-edge instance a budget of two is exactly attainable and a budget of
    one is attainable by nothing. Both are integer arithmetic on the pair's own
    cut indicators against the untouched zones' perimeters -- no dual, and no
    slack.
    """

    # floor(0.4 * 6) = 2: exactly the boundary every partition here spends.
    _, pool, seed = three_zone_pool("mid", boundary_prop=0.4)
    limit = boundary_limit(pool.problem)
    assert limit == 2
    assert sum(column.perimeter for column in seed) / 2 == limit
    result = redraw_pair(
        pool, seed, (0, 1), deadline=math.inf, workers=1, splits_per_call=32
    )
    assert result.status == "OPTIMAL"
    assert result.splits
    by_label = {column.zone: column for column in seed}
    for columns in result.splits:
        replaced = {**by_label, **{c.zone: c for c in columns}}
        assert sum(c.perimeter for c in replaced.values()) / 2 <= limit

    # floor(0.2 * 6) = 1, which no partition can meet: the pair's own two
    # zones already spend that much before the third is counted.
    _, starved_pool, starved_seed = three_zone_pool("mid", boundary_prop=0.2)
    assert boundary_limit(starved_pool.problem) == 1
    starved = ZoneRedraw(starved_pool, workers=1)(
        starved_seed, (0, 1), deadline=math.inf
    )
    assert starved.status == "INFEASIBLE"


# ---------------------------------------------------------------------- #
# Additivity: the search keeps its bound
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["boundary", "mid", "stable_matching"])
def test_branch_and_price_certifies_the_same_optimum_with_the_redraw(kind):
    """The redraw supplies columns; the bound still comes only from pricing."""

    problem = make_grid_problem(1, 5)
    best = max(
        sum(column.score for column in columns)
        for columns in admissible_partitions(build_pool(kind, problem, (0, 4)))
    )
    outcomes = {}
    for enabled in (False, True):
        pool = build_pool(kind, problem, (0, 4))
        pool.add_partition({0: 0, 1: 0, 2: 0, 3: 1, 4: 1})
        redraw = ZoneRedraw(pool, workers=1) if enabled else None
        result = branch_and_price(
            pool,
            deadline=math.inf,
            pricing_parallel=False,
            pricing_scale=1000,
            redraw=redraw,
        )
        assert result.status == "OPTIMAL"
        assert sum(column.score for column in result.selected) == pytest.approx(best)
        # A valid upper bound either way: the redraw never writes node_bound.
        assert result.upper_bound >= best - 1e-9
        outcomes[enabled] = result
    assert outcomes[True].columns_added >= outcomes[False].columns_added


def test_branch_and_price_records_what_the_redraw_contributed():
    problem, pool, seed = three_zone_pool("mid")
    redraw = ZoneRedraw(pool, workers=1, splits_per_call=32)
    result = branch_and_price(
        pool,
        deadline=math.inf,
        incumbent=seed,
        pricing_parallel=False,
        redraw=redraw,
        redraw_time_limit=30.0,
    )
    assert result.status == "OPTIMAL"
    records = [entry for entry in result.history if "redraw" in entry]
    assert records, "the redraw should run once a node's LP is priced out"
    assert redraw.pairs_solved > 0
    assert len(pooled_tilings(pool)) > 1


def test_the_redraw_never_lowers_the_search_bound_below_the_optimum():
    """A pricing-free sanity check on additivity.

    With an enumerative pricer that always reports an exact bound, adding the
    redraw must leave the certified bound valid and the status OPTIMAL. If the
    redraw were ever consulted for a bound this would fail, because its
    optimum ranges over a *subset* of each label's family.
    """

    problem, pool, seed = three_zone_pool("boundary")
    redraw = ZoneRedraw(pool, workers=1)
    result = branch_and_price(
        pool,
        deadline=math.inf,
        incumbent=seed,
        pricing_parallel=False,
        redraw=redraw,
    )
    best = max(
        sum(column.score for column in columns)
        for columns in admissible_partitions(pool)
    )
    assert result.status == "OPTIMAL"
    assert sum(c.score for c in result.selected) == pytest.approx(best)
    assert result.upper_bound >= best - 1e-9


# ---------------------------------------------------------------------- #
# Strategy plumbing
# ---------------------------------------------------------------------- #
def _dataset(problem=None):
    problem = problem or make_grid_problem(2, 2, hint={0: 0, 1: 0, 2: 1, 3: 1})
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(
        include_citywide_choice_opt=False, program_population="All"
    )
    dataset.problem = problem
    return dataset


@pytest.mark.parametrize("enabled", [True, False])
def test_strategy_sweeps_before_the_search_when_the_redraw_is_on(
    monkeypatch, enabled
):
    dataset = _dataset()
    monkeypatch.setattr(dw, "build_mid_market", lambda *_: market())
    monkeypatch.setattr(dw, "build_saa_market", lambda *_: individual_market())
    strategy = get_strategy(
        "dantzig_wolfe",
        levels=["BlockGroup_0"],
        solve_time_limits=[60],
        dw_objective="mid",
        dw_recom_samples=0,
        dw_pricing_parallel=False,
        dw_redraw=enabled,
        dw_redraw_time_limit=20,
        hints="voronoi",
        mid_lottery_scale=5,
    )
    solution = strategy.run(dataset, get_solver("cp_bool", workers=1))[0]
    assert solution.status == "OPTIMAL"
    assert solution.metadata["dw_redraw"] is enabled
    if enabled:
        assert solution.metadata["dw_redraw_method"] == "two_zone_cpsat_redraw"
        seeding = solution.metadata["dw_redraw_seeding"]
        assert seeding["dw_redraw_status"] == "swept"
        assert seeding["dw_redraw_pairs_solved"] >= 1
        assert solution.metadata["dw_redraw_total_pairs_solved"] >= 1
    else:
        assert "dw_redraw_total_pairs_solved" not in solution.metadata
        assert solution.metadata["dw_redraw_seeding"] is None

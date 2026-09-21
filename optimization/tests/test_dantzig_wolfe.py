"""Zone families, zone values, the master, and the strategy that drives them.

The search engine itself is tested in ``test_branch_price.py``; this module is
about the three things the engine consumes -- which sets are legal zones, what
a zone is worth, and what the restricted master does with them.
"""

from dataclasses import replace
from itertools import combinations, product
import math
from types import SimpleNamespace

import pytest

from optimization.config import OptimizationConfig
from optimization.data.mid import MidMarket, MidProgram, MidStudent, MidType
from optimization.data.saa import SaaMarket
from optimization.mid_oracle import finite_grid_oracle
from optimization.solvers import get_solver
from optimization.strategies import get_strategy
from optimization.strategies import dantzig_wolfe as dw
from optimization.tests.synthetic import FakeDataset, make_grid_problem
from optimization.zone_columns import (
    DualPoint,
    ZoneColumn,
    ZonePool,
    smooth,
    overlap_limit,
    overlap_weight,
    solve_master,
)
from optimization.zone_family import boundary_limit, build_zone_family
from optimization.zone_welfare import (
    BoundaryZoneObjective,
    MidZoneObjective,
    StableMatchingZoneObjective,
    deferred_acceptance,
    restrict_market,
)


def market(schools=(0, 3), count=4):
    """A MID market with one citywide program, on a grid's two school nodes."""

    return MidMarket(
        programs=(
            MidProgram("A", 100, 1, False, schools[0]),
            MidProgram("B", 200, 1, False, schools[1]),
            MidProgram("C", 300, 1, True, None),
        ),
        types=tuple(
            MidType(
                n, 1, ("C", "A", "B"), (0, n % 2, 0), (10.0, 4.0, 2.0), (1000, 400, 200)
            )
            for n in range(count)
        ),
        student_count=count,
        outside_only_student_count=0,
        utility_student_count=count,
        utility_handling="omit_nonpositive",
    )


def individual_market(schools=(0, 3), count=4, capacities=(1, 2)):
    """The same shape as :func:`market`, un-compressed, for the matching model."""

    return SaaMarket(
        programs=(
            MidProgram("A", 100, capacities[0], False, schools[0]),
            MidProgram("B", 200, capacities[1], False, schools[1]),
        ),
        students=tuple(
            MidStudent(
                node=n,
                programs=("A", "B"),
                priorities=(n % 2, 0),
                utilities=(4.0, 2.0),
                scaled_utilities=(400, 200),
            )
            for n in range(count)
        ),
        utility_student_count=count,
        utility_handling="omit_nonpositive",
    )


def matching_objective(**kwargs):
    return StableMatchingZoneObjective(
        individual_market(**kwargs), tie_breaking_method="STB", seed=7
    )


# ---------------------------------------------------------------------- #
# Zone families
# ---------------------------------------------------------------------- #
def test_family_anchors_each_label_and_excludes_the_other_centroids():
    p = make_grid_problem(2, 2)
    family = build_zone_family(p)
    assert family.forced[0] == frozenset({0})
    assert family.forced[1] == frozenset({3})
    # Node 3 is zone 1's anchor, so it is never a candidate for zone 0.
    assert 3 not in family.candidates[0]
    assert 0 not in family.candidates[1]
    assert not family.feasible(0, frozenset({1, 2}))  # no anchor
    assert family.feasible(0, frozenset({0, 1, 2}))


def test_family_requires_closer_neighbour_support_not_merely_connectedness():
    """On a path anchored at both ends, the families are prefixes and suffixes.

    Every member needs a same-zone neighbour strictly closer to its anchor, so
    zone 0 can only take ``{0, ..., j}`` and zone 1 only ``{i, ..., 4}``.
    ``{0, 2}`` is rejected although each of its members is a candidate, and
    ``{1, 2, 3}`` is rejected although it is connected: it has no anchor.
    """

    p = make_grid_problem(1, 5)
    family = build_zone_family(p)
    for size in range(1, 5):
        assert family.feasible(0, frozenset(range(size)))
        assert family.feasible(1, frozenset(range(5 - size, 5)))
    assert not family.feasible(0, frozenset({0, 2}))
    assert not family.feasible(0, frozenset({0, 1, 3}))
    assert not family.feasible(1, frozenset({1, 2, 3}))
    assert not family.feasible(1, frozenset({2, 4}))


def test_family_honours_explicit_candidates_fixings_and_balance_rows():
    p = make_grid_problem(2, 2, fixed={0: 0}, candidates={1: {1}})
    family = build_zone_family(p)
    assert 1 not in family.candidates[0]
    assert 1 in family.candidates[1]
    assert family.feasible(0, frozenset({0, 2}))
    assert family.feasible(1, frozenset({1, 3}))
    p = make_grid_problem(2, 2, frl_dev=0.0)
    p.G.nodes[0]["FRL"] = 1.0
    assert not build_zone_family(p).feasible(0, frozenset({0}))


def test_family_forces_the_centroid_neighbourhood_at_a_positive_radius():
    p = make_grid_problem(1, 5)
    family = build_zone_family(p, centroid_neighbor_radius=1)
    assert family.forced[0] == frozenset({0, 1})
    assert not family.feasible(0, frozenset({0}))
    assert family.feasible(0, frozenset({0, 1}))
    # Zone 1's ball is fixed out of zone 0 entirely.
    assert 3 not in family.candidates[0]


def test_family_matches_cp_bool_on_every_partition_of_a_small_grid():
    """The families are the base model's label rows, so their product is it.

    A tuple of sets, one per label, is admissible for ``cp_bool`` exactly when
    each set lies in its own family and the sets partition ``V``: every row of
    the base model touches one label, apart from the assignment row (which
    becomes the master's cover row) and the boundary cap (the master's boundary
    row). This is Proposition 8's hypothesis, checked by enumeration.
    """

    p = make_grid_problem(1, 4)
    family = build_zone_family(p)
    solver = get_solver("cp_bool", solve_time_limit=30, workers=1, hints="none")
    enumerated = {
        tuple(sorted(solution.assignment.items()))
        for solution in solver.enumerate_solutions(p, 64)
        if solution.assignment
    }
    admissible = {
        tuple(enumerate(labels))
        for labels in product(range(p.Z), repeat=p.A)
        if all(
            family.feasible(
                z, frozenset(n for n, label in enumerate(labels) if label == z)
            )
            for z in range(p.Z)
        )
    }
    assert admissible == enumerated


# ---------------------------------------------------------------------- #
# Zone values
# ---------------------------------------------------------------------- #
def test_citywide_removed_and_mid_zone_welfare_is_additive():
    p = make_grid_problem(2, 2)
    m = market()
    restricted = restrict_market(m)
    assert [program.program_id for program in restricted.programs] == ["A", "B"]
    assert restricted.student_count == 4
    assert all(t.programs == ("A", "B") for t in restricted.types)
    assert restricted.types[1].priorities == (1, 0)
    assert len(m.programs) == 3  # no mutation of the source market
    pool = ZonePool(p, MidZoneObjective(m, 5))
    for assignment in (
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 0, 1: 1, 2: 0, 3: 1},
        {0: 0, 1: 1, 2: 1, 3: 1},
    ):
        columns = pool.add_partition(assignment)
        full = finite_grid_oracle(restricted, assignment, 5)
        assert sum(c.score for c in columns) == pytest.approx(full.welfare)


def test_students_with_only_citywide_preferences_are_retained():
    m = replace(market(), types=(MidType(0, 4, ("C",), (0,), (40.0,), (4000,)),))
    restricted = restrict_market(m)
    assert restricted.student_count == restricted.outside_only_student_count == 4
    assert restricted.types[0].programs == ()
    pool = ZonePool(make_grid_problem(2, 2), MidZoneObjective(m, 20))
    assert pool.column(0, {0}).score == 0


def test_sampled_matching_zone_welfare_is_additive():
    p = make_grid_problem(2, 2)
    objective = matching_objective()
    pool = ZonePool(p, objective)
    for assignment in (
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 0, 1: 1, 2: 0, 3: 1},
        {0: 0, 1: 1, 2: 1, 3: 1},
    ):
        columns = pool.add_partition(assignment)
        whole = sum(
            deferred_acceptance(
                objective.prepared,
                frozenset(n for n, a in assignment.items() if a == z),
            )[1]
            for z in range(p.Z)
        )
        assert sum(c.score for c in columns) == pytest.approx(whole)


def test_deferred_acceptance_is_the_best_stable_matching_by_brute_force():
    """Corollary 1, checked on a market small enough to enumerate matchings.

    Maximizing utility over the stable matchings of a market returns the
    applicant-proposing deferred-acceptance outcome, which is why a column's
    value can be computed by running the mechanism instead of solving the
    access polytope.
    """

    prepared = matching_objective(count=4, capacities=(1, 2)).prepared
    seat, welfare = deferred_acceptance(prepared, frozenset(range(4)))
    stable = [choice for choice in _matchings(prepared) if _stable(prepared, choice)]
    assert stable
    assert welfare == pytest.approx(max(_value(prepared, c) for c in stable))
    assert _stable(prepared, tuple(seat.get(i) for i in range(4)))


def _value(prepared, choice):
    return math.fsum(
        dict(prepared.preferences[i])[program_id]
        for i, program_id in enumerate(choice)
        if program_id is not None
    )


def _matchings(prepared):
    for choice in product((None, "A", "B"), repeat=len(prepared.preferences)):
        if all(
            choice.count(program_id) <= prepared.capacity[program_id]
            for program_id in ("A", "B")
        ):
            yield choice


def _stable(prepared, choice):
    """No applicant-program pair strictly improves on ``choice``."""

    for i, assigned in enumerate(choice):
        row = dict(prepared.preferences[i])
        held = row.get(assigned, 0.0) if assigned is not None else 0.0
        for program_id, value in row.items():
            if value <= held:
                continue
            seated = [j for j, other in enumerate(choice) if other == program_id]
            if len(seated) < prepared.capacity[program_id]:
                return False
            if any(
                prepared.position[program_id][j] > prepared.position[program_id][i]
                for j in seated
            ):
                return False
    return True


def test_zone_objectives_report_a_valid_a_priori_upper_bound():
    assert BoundaryZoneObjective().upper_bound() == 0.0
    # Four types, best utility 4.0 each once the citywide program is removed.
    assert MidZoneObjective(market(), 5).upper_bound() == pytest.approx(16.0)
    assert matching_objective().upper_bound() == pytest.approx(16.0)


# ---------------------------------------------------------------------- #
# The master
# ---------------------------------------------------------------------- #
def test_master_matches_exhaustive_complete_pool_and_duals():
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 5))
    for size in range(1, p.A):
        for members in combinations(p.nodes, size):
            for z in range(p.Z):
                column = pool.admit(z, frozenset(members))
                if column is not None:
                    pool.add(column)
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


@pytest.mark.parametrize("method", ["barrier", "dual", "primal", "auto"])
def test_every_master_method_agrees_on_the_objective(method):
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, MidZoneObjective(market(), 5))
    pool.add_partition({0: 0, 1: 0, 2: 1, 3: 1})
    pool.add_partition({0: 0, 1: 1, 2: 0, 3: 1})
    columns = tuple(pool.columns.values())
    reference = solve_master(p, columns, 5, method="dual")
    lp = solve_master(p, columns, 5, method=method)
    assert lp.status == "OPTIMAL"
    assert lp.objective == pytest.approx(reference.objective, abs=1e-6)
    # And its duals still price the pool correctly, whichever point they are.
    assert max(lp.reduced_cost(c) for c in columns) <= 1e-5


def test_master_rejects_an_unknown_method():
    p = make_grid_problem(2, 2)
    with pytest.raises(ValueError, match="dw_master_method"):
        solve_master(p, (), 5, method="crossover")


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


def test_boundary_is_counted_once_and_the_cap_is_enforced():
    p = make_grid_problem(2, 2)
    pool = ZonePool(p, BoundaryZoneObjective())
    columns = pool.add_partition({0: 0, 1: 0, 2: 1, 3: 1})
    assert sum(c.score for c in columns) == -2
    p.boundary_prop = 0.25
    assert boundary_limit(p) == 1
    assert solve_master(p, columns, 5).status == "INFEASIBLE"
    p.boundary_prop = 0.5
    assert solve_master(p, columns, 5, integer=True).objective == -2


def test_boundary_dual_enters_reduced_cost_with_correct_sign():
    dual = DualPoint({0: 1.0, 1: 2.0}, {0: 3.0}, 4.0)
    assert dual.reduced_cost(ZoneColumn(0, frozenset({0, 1}), 20.0, 6)) == 2.0


def test_boundary_row_dual_is_non_negative_on_a_binding_cap():
    p = make_grid_problem(2, 2, boundary_prop=0.5)
    pool = ZonePool(p, BoundaryZoneObjective())
    pool.add_partition({0: 0, 1: 0, 2: 1, 3: 1})
    pool.add_partition({0: 0, 1: 1, 2: 0, 3: 1})
    lp = solve_master(p, tuple(pool.columns.values()), 5)
    assert lp.status == "OPTIMAL"
    assert lp.boundary_dual >= -1e-9


# ---------------------------------------------------------------------- #
# Dual stabilization
# ---------------------------------------------------------------------- #
def test_smoothing_is_a_convex_combination_and_identity_at_alpha_one():
    previous = DualPoint({0: 0.0, 1: 10.0}, {0: 4.0}, 2.0)
    current = DualPoint({0: 4.0, 1: 0.0}, {0: 0.0}, 0.0)
    assert smooth(previous, current, 1.0).node_duals == current.node_duals
    assert smooth(None, current, 0.5).node_duals == current.node_duals
    blended = smooth(previous, current, 0.25)
    assert blended.node_duals == {0: 1.0, 1: 7.5}
    assert blended.zone_duals == {0: 3.0}
    assert blended.boundary_dual == 1.5


def test_smoothing_clamps_the_boundary_multiplier_and_validates_alpha():
    current = DualPoint({}, {}, -3.0)
    assert smooth(None, current, 1.0).boundary_dual == 0.0
    for alpha in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="dw_dual_smoothing"):
            smooth(None, current, alpha)


def test_smoothed_duals_still_price_a_column_the_same_way():
    """Smoothing changes where pricing aims, not what a reduced cost means."""

    column = ZoneColumn(0, frozenset({0, 1}), 20.0, 6)
    raw = DualPoint({0: 1.0, 1: 2.0}, {0: 3.0}, 4.0)
    blended = smooth(raw, DualPoint({0: 3.0, 1: 2.0}, {0: 3.0}, 4.0), 0.5)
    assert blended.reduced_cost(column) == pytest.approx(raw.reduced_cost(column) - 1.0)


# ---------------------------------------------------------------------- #
# The strategy
# ---------------------------------------------------------------------- #
def _dataset(problem=None):
    problem = problem or make_grid_problem(2, 2, hint={0: 0, 1: 0, 2: 1, 3: 1})
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(
        include_citywide_choice_opt=False, program_population="All"
    )
    dataset.problem = problem
    return dataset


@pytest.mark.parametrize("objective", ["mid", "stable_matching", "boundary"])
def test_strategy_end_to_end(monkeypatch, objective):
    dataset = _dataset()
    monkeypatch.setattr(dw, "build_mid_market", lambda *_: market())
    monkeypatch.setattr(dw, "build_saa_market", lambda *_: individual_market())
    strategy = get_strategy(
        "dantzig_wolfe",
        levels=["BlockGroup_0"],
        solve_time_limits=[60],
        dw_objective=objective,
        dw_recom_samples=10,
        dw_recom_chains=2,
        dw_pricing_parallel=False,
        hints="voronoi",
        mid_lottery_scale=5,
        seed=7,
    )
    solver = get_solver("cp_bool", workers=1, hints="voronoi")
    solution = strategy.run(dataset, solver)[0]
    assert solution.feasible
    assert set(solution.assignment) == set(dataset.problem.nodes)
    assert len(set(solution.assignment.values())) == dataset.problem.Z
    assert solution.status == "OPTIMAL"
    assert solution.metadata["dw_pricing_certified"] is True
    assert solution.metadata["dw_absolute_gap"] >= 0.0
    assert solution.metadata["dw_contiguity"] == "anchored_closer_neighbor"
    if objective == "mid":
        assert solution.objective == pytest.approx(
            finite_grid_oracle(
                restrict_market(market()), solution.assignment, 5
            ).welfare
        )
        assert solution.metadata["objective_kind"] == "mid_program_welfare"
    if objective == "stable_matching":
        assert solution.metadata["dw_tie_breaking_method"] == "MTB"
        assert solution.metadata["dw_sample_seed"] > 0


def test_strategy_seeds_from_the_cp_bool_hint_before_pricing(monkeypatch):
    dataset = _dataset()
    monkeypatch.setattr(dw, "build_mid_market", lambda *_: market())
    strategy = get_strategy(
        "dantzig_wolfe",
        levels=["BlockGroup_0"],
        solve_time_limits=[60],
        dw_objective="mid",
        dw_recom_samples=0,
        dw_pricing_parallel=False,
        hints="voronoi",
        mid_lottery_scale=5,
    )
    solution = strategy.run(dataset, get_solver("cp_bool", workers=1))[0]
    assert solution.metadata["dw_hint_admissible"] is True
    assert solution.metadata["dw_hint_seed_columns"] == dataset.problem.Z
    assert solution.metadata["dw_recom_samples_visited"] == 0


def test_recom_seeding_only_admits_zones_the_family_accepts(monkeypatch):
    dataset = _dataset(make_grid_problem(1, 5, hint={0: 0, 1: 0, 2: 0, 3: 1, 4: 1}))
    monkeypatch.setattr(dw, "build_mid_market", lambda *_: market(schools=(0, 4)))
    strategy = get_strategy(
        "dantzig_wolfe",
        levels=["BlockGroup_0"],
        solve_time_limits=[45],
        dw_objective="mid",
        dw_recom_samples=40,
        dw_recom_chains=2,
        dw_recom_time_limit=5,
        dw_pricing_parallel=False,
        hints="voronoi",
        mid_lottery_scale=5,
    )
    solution = strategy.run(
        dataset, get_solver("cp_bool", workers=1, recom_iterations=200)
    )[0]
    metadata = solution.metadata
    assert metadata["dw_recom_samples_visited"] >= metadata["dw_recom_admissible"]
    # Whatever the sampler produced, nothing inadmissible entered the pool.
    family = build_zone_family(dataset.problem)
    for entry in metadata["dw_selected_columns"]:
        assert family.feasible(entry["zone"], frozenset(entry["nodes"]))


def test_zero_budget_returns_unknown_without_sampling():
    dataset = _dataset()
    strategy = get_strategy(
        "dantzig_wolfe",
        levels=["BlockGroup_0"],
        solve_time_limits=[0],
        dw_objective="boundary",
        hints="none",
    )
    solution = strategy.run(dataset, get_solver("cp_bool"))[0]
    assert solution.status == "UNKNOWN"
    assert solution.metadata["stop_reason"] == "time_limit"


# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #
def test_config_and_example():
    from pathlib import Path

    config = OptimizationConfig.from_yaml(
        Path(__file__).parents[1] / "dantzig_wolfe.example.yaml"
    )
    strategy = config.make_strategy()
    assert strategy.name == "dantzig_wolfe"
    assert config.solver == "cp_bool"
    assert config.include_citywide_choice_opt is False
    assert strategy.options["dw_objective"] in ("mid", "stable_matching")
    assert strategy.options["dw_master_method"] == "barrier"
    assert strategy.options["dw_overlap_prop"] == 0.0


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"solver": "recom"}, "solver='cp_bool'"),
        ({"budget_accounting": "solver_time"}, "wall_clock"),
        ({"dw_recom_samples": -1}, "integer"),
        ({"dw_recom_time_limit": True}, "dw_recom_time_limit"),
        ({"tolerance": float("nan")}, "tolerance"),
        ({"dw_objective": "saa"}, "dw_objective"),
        ({"dw_master_method": "crossover"}, "dw_master_method"),
        ({"dw_dual_smoothing": 0.0}, "dw_dual_smoothing"),
        ({"dw_dual_smoothing": 1.5}, "dw_dual_smoothing"),
        ({"dw_overlap_prop": -0.1}, "dw_overlap_prop"),
        ({"dw_overlap_prop": 1.5}, "dw_overlap_prop"),
        ({"dw_overlap_prop": True}, "dw_overlap_prop"),
        ({"dw_overlap_prop": float("inf")}, "dw_overlap_prop"),
        ({"dw_pricing_scale": 0}, "dw_pricing_scale"),
        ({"dw_pricing_columns_per_call": 0}, "dw_pricing_columns_per_call"),
        ({"dw_pricing_parallel": "yes"}, "dw_pricing_parallel"),
    ],
)
def test_invalid_config(overrides, match):
    options = dict(strategy="dantzig_wolfe", solver="cp_bool", dw_objective="boundary")
    with pytest.raises(ValueError, match=match):
        OptimizationConfig(**{**options, **overrides})


def test_config_rejects_citywide():
    with pytest.raises(ValueError, match="include_citywide_choice_opt=false"):
        OptimizationConfig(
            strategy="dantzig_wolfe",
            solver="cp_bool",
            dw_objective="boundary",
            data={
                "scenario": "legacy",
                "overrides": {
                    "filters": {"optimization": {"include_citywide_choice_opt": True}}
                },
            },
        )


@pytest.mark.parametrize("objective", ["mid", "stable_matching"])
def test_config_requires_all_programs_for_welfare_objectives(objective):
    with pytest.raises(ValueError, match="program_population='All'"):
        OptimizationConfig(
            strategy="dantzig_wolfe",
            solver="cp_bool",
            dw_objective=objective,
            data={
                "scenario": "legacy",
                "overrides": {
                    "filters": {
                        "optimization": {
                            "include_citywide_zoning": False,
                            "include_citywide_choice_opt": False,
                            "program_population": "GE",
                        }
                    }
                },
            },
        )


def test_matching_objective_rejects_an_unknown_tie_breaking_method():
    with pytest.raises(ValueError, match="MTB, STB"):
        StableMatchingZoneObjective(individual_market(), tie_breaking_method="NTB")


def test_unknown_objective_is_rejected_by_the_builder():
    with pytest.raises(ValueError, match="dw_objective"):
        dw.build_objective(
            make_grid_problem(2, 2), SimpleNamespace(), {"dw_objective": "saa"}
        )


def test_pool_rejects_a_market_that_does_not_live_on_the_graph():
    with pytest.raises(ValueError, match="belong to the graph"):
        ZonePool(make_grid_problem(1, 2), MidZoneObjective(market(), 5))
    with pytest.raises(ValueError, match="belong to the graph"):
        ZonePool(make_grid_problem(1, 2), matching_objective())


def test_pool_refuses_to_score_an_inadmissible_zone():
    pool = ZonePool(make_grid_problem(2, 2), BoundaryZoneObjective())
    assert pool.admit(0, frozenset({1, 2})) is None
    with pytest.raises(ValueError, match="inadmissible"):
        pool.column(0, frozenset({1, 2}))


def test_a_partition_with_one_bad_zone_still_contributes_its_good_ones():
    """The rejection filter is per zone, which is what makes ReCom worth running.

    A recombination step rewrites two zones; the rest of the partition is
    whatever a previously admissible one had, so discarding the whole sample
    over one bad zone would throw away most of what the sampler produces.
    """

    p = make_grid_problem(3, 3)
    pool = ZonePool(p, BoundaryZoneObjective())
    # Zone 0 = {0, 4} has no closer support at 4; zone 1 takes the rest and is
    # perfectly admissible.
    assignment = {node: 1 for node in p.nodes}
    assignment[0] = assignment[4] = 0
    assert pool.admit(0, frozenset({0, 4})) is None
    assert pool.admit(1, frozenset(p.nodes) - {0, 4}) is not None
    pool.columns.clear()
    assert pool.admit_partition(assignment) is None
    assert {key[0] for key in pool.columns} == {1}
    assert math.isfinite(sum(c.score for c in pool.columns.values()))


# ---------------------------------------------------------------------- #
# The budgeted-elastic master
# ---------------------------------------------------------------------- #
def _collision_instance():
    """One tiling, plus a high-value label-0 zone that collides with label 1.

    Raw columns, so the LP's behaviour is isolated from the welfare oracle.
    Node 2 is the contested one: it belongs to the tiling's label-1 zone and
    to the newcomer.
    """

    p = make_grid_problem(1, 6)
    p.centroids = [0, 2, 5]
    p.centroid_school_ids = [100, 150, 200]
    tiling = [
        ZoneColumn(0, frozenset({0, 1}), 1.0, 0),
        ZoneColumn(1, frozenset({2, 3}), 1.0, 0),
        ZoneColumn(2, frozenset({4, 5}), 1.0, 0),
    ]
    collides = ZoneColumn(0, frozenset({0, 1, 2}), 100.0, 0)
    return p, (*tiling, collides)


def test_a_colliding_column_has_a_zero_step_length_until_the_budget_exists():
    """The measured obstruction, reproduced, and then removed.

    The exact master cannot give the newcomer any weight: node 2's cover row
    is already satisfied by the only label-1 column available, so raising the
    newcomer would over-cover it. The pool grows by a column worth 100 against
    a tiling worth 3 and the LP does not move -- a zero step length, which is
    exactly what 400 to 555 columns per run did on the real instances. With a
    budget the surplus variable absorbs the collision and the same column
    enters.
    """

    p, columns = _collision_instance()
    seeded = solve_master(p, columns[:-1], 5, method="dual")
    assert seeded.objective == pytest.approx(3.0)

    exact = solve_master(p, columns, 5, method="dual")
    assert exact.status == "OPTIMAL"
    assert exact.objective == pytest.approx(seeded.objective)
    assert exact.values[-1] == pytest.approx(0.0)
    assert exact.elastic_mass == 0.0

    lp = solve_master(p, columns, 5, method="dual", overlap_budget=0.5)
    assert lp.status == "OPTIMAL"
    # lambda on the newcomer rises to the budget: one unit of surplus at node
    # 2 per unit of weight, so 0.5 buys 0.5, worth 99 each over the tiling.
    assert lp.values[-1] == pytest.approx(0.5)
    assert lp.objective == pytest.approx(3.0 + 99.0 * 0.5)
    assert lp.elastic_mass == pytest.approx(0.5)
    assert lp.elastic_nodes == 1


def test_zero_budget_is_the_exact_master():
    p, columns = _collision_instance()
    exact = solve_master(p, columns, 5, method="dual")
    for budget in (0.0, -0.0):
        lp = solve_master(p, columns, 5, method="dual", overlap_budget=budget)
        assert lp.objective == pytest.approx(exact.objective)
        assert lp.elastic_mass == 0.0
        assert lp.overlap_dual == 0.0
        assert lp.duals_pinned == 0


@pytest.mark.parametrize("budget", [0.1, 0.5, 1.0, 2.0])
def test_the_budget_binds_and_the_relaxation_only_ever_loosens(budget):
    p, columns = _collision_instance()
    exact = solve_master(p, columns, 5, method="dual")
    lp = solve_master(p, columns, 5, method="dual", overlap_budget=budget)
    assert lp.status == "OPTIMAL"
    assert lp.elastic_mass <= budget + 1e-9
    # A relaxation: every tiling stays feasible at its own objective, so the
    # value can only rise. That is what keeps Proposition 9's bound valid.
    assert lp.objective >= exact.objective - 1e-9


@pytest.mark.parametrize("method", ["barrier", "dual"])
def test_the_budget_dual_completes_the_dual_objective(method):
    """Strong duality, including the budget row, at whichever dual point.

    Omitting ``mu_K * K`` would report a dual objective *below* the
    relaxation's own optimum, and a bound below the relaxation it was computed
    from is wrong rather than merely weak.
    """

    p, columns = _collision_instance()
    budget = 0.5
    lp = solve_master(p, columns, 5, method=method, overlap_budget=budget)
    assert lp.status == "OPTIMAL"
    assert lp.duals().dual_objective(p, budget) == pytest.approx(lp.objective, abs=1e-6)
    # ... and without the term it is strictly short, so the test has teeth.
    assert lp.duals().dual_objective(p) < lp.objective - 1e-6


@pytest.mark.parametrize("method", ["barrier", "dual"])
def test_the_budget_dual_is_the_M_the_penalty_form_would_have_guessed(method):
    """``|alpha_v| <= w_v mu_K`` is the elastic pair's dual-feasibility row.

    It is the whole reason for budgeting instead of penalizing: that bound is
    the exact-penalty threshold a hand-chosen ``M`` has to clear, and here the
    LP reports it rather than being told it.
    """

    p, columns = _collision_instance()
    lp = solve_master(p, columns, 5, method=method, overlap_budget=0.5)
    assert lp.status == "OPTIMAL"
    assert lp.overlap_dual >= -1e-9
    assert lp.overlap_dual > 1e-6  # the budget binds here
    for node, dual in lp.node_duals.items():
        assert abs(dual) <= overlap_weight(p, node) * lp.overlap_dual + 1e-6
    # The contested node is where the budget is spent, so its price is the one
    # held at the bound.
    assert lp.duals_pinned >= 1


def test_the_exact_and_phase_one_masters_refuse_a_budget():
    p, columns = _collision_instance()
    with pytest.raises(ValueError, match="must be exact"):
        solve_master(p, columns, 5, integer=True, overlap_budget=0.5)
    with pytest.raises(ValueError, match="Phase I"):
        solve_master(p, columns, 5, phase_one=True, overlap_budget=0.5)
    with pytest.raises(ValueError, match="dw_overlap_prop"):
        solve_master(p, columns, 5, overlap_budget=-1.0)


def test_overlap_weight_floors_a_student_free_node_at_one_unit():
    """Otherwise a free node is double-claimable for nothing.

    Two labels could each run their own support chain through it without
    either paying, which is the one way a budget in student mass could be
    spent on geometry rather than on students.
    """

    p = make_grid_problem(1, 3)
    for node in p.nodes:
        p.G.nodes[node][p.student_attribute] = 0.0
    assert [overlap_weight(p, n) for n in p.nodes] == [1.0, 1.0, 1.0]
    assert overlap_limit(p, 0.5) == pytest.approx(1.5)
    p.G.nodes[1][p.student_attribute] = 40.0
    assert overlap_limit(p, 1.0) == pytest.approx(42.0)
    assert overlap_limit(p, 0.0) == 0.0


def test_smoothing_carries_the_budget_dual_and_clamps_it():
    previous = DualPoint({0: 0.0}, {0: 0.0}, 0.0, False, 4.0)
    current = DualPoint({0: 0.0}, {0: 0.0}, 0.0, False, 0.0)
    assert smooth(previous, current, 0.25).overlap_dual == pytest.approx(3.0)
    assert smooth(previous, current, 1.0).overlap_dual == 0.0
    assert smooth(None, DualPoint({}, {}, 0.0, False, -1e-12), 0.5).overlap_dual == 0.0

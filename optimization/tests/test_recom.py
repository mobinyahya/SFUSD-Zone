"""Synthetic tests for the ReCom solver and its policies."""

from __future__ import annotations

import math
import random
import time
from dataclasses import replace

import networkx as nx
import pytest

from optimization.config import OptimizationConfig
from optimization.data.contiguity import boundary_cost, boundary_edges
from optimization.data.edge_weights import BOUNDARY_WEIGHT_ATTR
from optimization.solvers import get_solver
from optimization.solvers.balance import balance_constraints
from optimization.solvers.base import available_solvers
from optimization.solvers.recom import _ReComContext, _ReComKernel, _State, _ZoneStats
from optimization.solvers.recom_policy import (
    PRESETS,
    DualWeights,
    Exp3Pairs,
    HedgeWeights,
    Policy,
    ProportionalPairs,
    ReComSettings,
    Temperature,
    Weights,
    ZonePairs,
    entropy_beta,
)
from optimization.tests.synthetic import make_grid_problem, make_solver_contract_problem


def _solver(preset: str = "entropy", **options):
    return get_solver("recom", **{**PRESETS[preset].as_options(), **options})


# --------------------------------------------------------------------------- #
# registration and configuration
# --------------------------------------------------------------------------- #
def test_recom_is_the_only_registered_recom_method() -> None:
    solvers = set(available_solvers())
    assert "recom" in solvers
    assert not solvers & {
        "relaxed_recom",
        "short_bursts",
        "adaptive_short_bursts",
        "adapative_short_bursts",
    }


def test_config_passes_recom_settings_to_solver() -> None:
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="recom",
        recom_iterations=321,
        recom_burst_length=17,
        recom_weights="hedge",
        recom_pair_selection="zone",
        recom_entropy=None,
    )

    options = config.make_solver().options

    assert options["recom_iterations"] == 321
    assert ReComSettings.from_options(options) == ReComSettings(
        burst_length=17, weights="hedge", pair_selection="zone", entropy=None
    )
    assert ReComSettings.from_options(OptimizationConfig().make_solver().options) == (
        ReComSettings()
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"recom_burst_length": -1}, "recom_burst_length"),
        ({"recom_weights": "adam"}, "recom_weights"),
        ({"recom_pair_selection": "lagrangian_softmax"}, "recom_pair_selection"),
        ({"recom_pair_score": "compatible"}, "recom_pair_score"),
        ({"recom_cut_selection": "relaxed"}, "recom_cut_selection"),
        ({"recom_temperature": 0}, "recom_temperature"),
        ({"recom_entropy": 1.5}, "recom_entropy"),
        ({"recom_exploration": -0.1}, "recom_exploration"),
        ({"recom_objective_weight": -1}, "recom_objective_weight"),
        ({"recom_iterations": -1, "solve_time_limits": []}, "solve_time_limits"),
    ],
)
def test_config_rejects_invalid_recom_settings(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        OptimizationConfig(levels=["BlockGroup_0"], **overrides)


def test_settings_round_trip_through_options() -> None:
    for settings in PRESETS.values():
        assert ReComSettings.from_options(settings.as_options()) == settings


# --------------------------------------------------------------------------- #
# the solver contract, for every preset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset", PRESETS)
def test_recom_requires_a_hint(preset: str) -> None:
    problem = make_solver_contract_problem(hint=None)

    solution = _solver(preset, hints="none", recom_iterations=1).solve(problem)

    assert solution.status == "ERROR"
    assert solution.assignment == {}
    assert "hint" in solution.metadata["error_message"].lower()


@pytest.mark.parametrize("preset", PRESETS)
def test_recom_can_generate_a_voronoi_hint(preset: str) -> None:
    problem = make_solver_contract_problem(hint=None, max_distance=0.0)

    solution = _solver(preset, hints="voronoi", recom_iterations=0).solve(problem)

    _assert_valid_recom_solution(problem, solution)
    assert solution.metadata["hint_source"] == "generated"


@pytest.mark.parametrize("preset", PRESETS)
def test_recom_allows_negative_capacity_tolerances(preset: str) -> None:
    problem = make_solver_contract_problem(overage=-1, shortage=-1)
    for node in problem.nodes:
        problem.G.nodes[node]["ge_capacity"] = 0.0

    _assert_valid_recom_solution(
        problem, _solver(preset, recom_iterations=0).solve(problem)
    )


@pytest.mark.parametrize("preset", PRESETS)
def test_recom_ignores_centroid_anchors_and_max_distance(preset: str) -> None:
    swapped = {0: 1, 1: 1, 2: 0, 3: 0}
    problem = make_solver_contract_problem(hint=swapped, max_distance=0.0)

    solution = _solver(preset, recom_iterations=0).solve(problem)

    _assert_valid_recom_solution(problem, solution)
    assert solution.assignment == swapped


@pytest.mark.parametrize("preset", PRESETS)
def test_recom_rejects_a_structurally_invalid_hint(preset: str) -> None:
    problem = make_solver_contract_problem(hint={0: 0, 1: 1, 2: 0, 3: 1})

    solution = _solver(preset, recom_iterations=1).solve(problem)

    assert solution.status == "ERROR"
    assert "contiguous" in solution.metadata["error_message"]


@pytest.mark.parametrize("preset", PRESETS)
def test_recom_honors_explicit_candidates(preset: str) -> None:
    problem = make_solver_contract_problem(
        candidates={1: {0}, 2: {1}}, max_distance=0.0
    )

    solution = _solver(preset, recom_iterations=20, recom_burst_length=4, seed=7).solve(
        problem
    )

    _assert_valid_recom_solution(problem, solution)
    assert solution.assignment[1] == 0
    assert solution.assignment[2] == 1


@pytest.mark.parametrize("preset", PRESETS)
def test_recom_rejects_choice_objectives(preset: str) -> None:
    problem = make_solver_contract_problem(choice_objective=object())

    with pytest.raises(NotImplementedError, match="recom"):
        _solver(preset).solve(problem)


@pytest.mark.parametrize("preset", PRESETS)
def test_recom_uses_weighted_boundary_cost(preset: str) -> None:
    problem = make_solver_contract_problem(weight_edges=True)
    problem.G.graph["weight_edges"] = True
    for edge, weight in zip(problem.G.edges(), [3, 11, 5]):
        problem.G.edges[edge][BOUNDARY_WEIGHT_ATTR] = weight

    solution = _solver(preset, recom_iterations=5, recom_burst_length=2, seed=3).solve(
        problem
    )

    _assert_valid_recom_solution(problem, solution)
    assert solution.objective == boundary_cost(
        problem.G, solution.assignment, weight_edges=True
    )
    assert solution.objective != boundary_edges(problem.G, solution.assignment)
    assert solution.metadata["objective_kind"] == "weighted_boundary_length"
    assert solution.metadata["objective_unit"] == "meter"


@pytest.mark.parametrize("pair_score", ["pressure", "pooled"])
@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_repairs_an_infeasible_start(preset: str, pair_score: str) -> None:
    # One school per zone, but the start puts both in zone 0.
    problem = make_grid_problem(4, 4, school_count_tolerance=0.0)
    start = [int(pos == 14) for pos in range(16)]
    context = _ReComContext(problem)
    assert not context.build_state(start).feasible
    problem = replace(problem, hint=context.assignment_dict(start))

    solution = _solver(
        preset, recom_iterations=400, recom_pair_score=pair_score, seed=5
    ).solve(problem)

    # Even the rejection chain: it accepts any move that lands in the feasible set.
    _assert_valid_recom_solution(problem, solution)
    assert solution.metadata["time_to_first_feasible"] is not None


# --------------------------------------------------------------------------- #
# the walk
# --------------------------------------------------------------------------- #
def test_the_chain_rejects_infeasible_proposals() -> None:
    problem = _problem_without_feasible_contiguous_partition()

    solution = _solver("chain", recom_iterations=10, seed=3).solve(problem)

    assert solution.status == "UNKNOWN"
    assert solution.metadata["accepted_moves"] == 0
    assert solution.metadata["rejected_moves"] == 10


@pytest.mark.parametrize("burst_length", [0, 3])
def test_unrejected_walks_accept_infeasible_proposals(burst_length: int) -> None:
    problem = _problem_without_feasible_contiguous_partition()

    solution = _solver(
        "uniform_bursts", recom_iterations=7, recom_burst_length=burst_length, seed=3
    ).solve(problem)

    assert solution.status == "UNKNOWN"
    assert solution.metadata["attempted_moves"] == 7
    assert solution.metadata["accepted_moves"] == 7
    assert solution.metadata["rejected_moves"] == 0


def test_bursts_share_the_move_budget() -> None:
    solution = _solver(
        "uniform_bursts", recom_iterations=7, recom_burst_length=3, seed=5
    ).solve(make_solver_contract_problem())

    assert solution.metadata["attempted_moves"] == 7
    assert solution.metadata["completed_bursts"] == 3


def test_negative_iterations_use_only_the_time_limit() -> None:
    problem = make_solver_contract_problem()

    solution = _solver("chain", recom_iterations=-1, solve_time_limit=0.0).solve(
        problem
    )

    _assert_valid_recom_solution(problem, solution)
    assert solution.metadata["attempted_moves"] == 0
    assert solution.metadata["stop_reason"] == "time_limit"


def test_sample_feasible_walks_the_chain_whatever_the_options() -> None:
    problem = make_grid_problem(3, 3, frl_dev=1.0, overage=-1, shortage=-1)
    problem = replace(problem, hint={n: int(n % 3 > 0) for n in problem.nodes})
    visited: list[dict[int, int]] = []

    def visit(assignment):
        visited.append(assignment)
        return len(visited) < 5

    solution = _solver("dual", recom_iterations=100, seed=2).sample_feasible(
        problem, visit
    )

    assert len(visited) == 5
    assert solution.metadata["stop_reason"] == "visitor_stop"
    assert solution.metadata["recom_reject_infeasible"] is True
    assert solution.metadata["recom_burst_length"] == 0


def test_scored_walks_repair_an_infeasible_start() -> None:
    """Burst selection prefers feasible samples, so a scored walk needs no
    feasible start; only feasible samples are ever scored."""
    # Three schools per zone: only {0, 1} | {2, 3} is feasible.
    problem = make_solver_contract_problem(
        hint={0: 0, 1: 1, 2: 1, 3: 1}, school_count_tolerance=0.0
    )
    scored: list[dict[int, int]] = []

    def scorer(assignments, base):
        scored.extend(assignments)
        return tuple(1.0 for _ in assignments)

    solution = _solver(recom_iterations=50, seed=1).solve_with_scorer(
        problem, scorer, objective_kind="test_score"
    )

    assert solution.metadata["initial_feasible"] is False
    assert solution.status == "FEASIBLE"
    # The one feasible split, under either labelling (centroids are not pinned).
    assert scored and all(a[0] == a[1] != a[2] == a[3] for a in scored)


class _SlowScorer:
    """Records when each scoring batch runs so overrun can be measured."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.call_times: list[float] = []

    def __call__(self, assignments, base_assignment):
        self.call_times.append(time.monotonic())
        time.sleep(self.delay)
        return tuple(float(len(assignment)) for assignment in assignments)


def test_scored_bursts_overrun_the_deadline_by_at_most_one_batch() -> None:
    """Scoring is synchronous, so one batch may straddle the deadline - only one."""

    problem = make_grid_problem(2, 2, frl_dev=1.0, hint={0: 0, 1: 0, 2: 1, 3: 1})
    time_limit = 0.2
    scorer = _SlowScorer(delay=0.15)
    solver = _solver(
        "uniform_bursts",
        solve_time_limit=time_limit,
        recom_iterations=-1,
        recom_burst_length=2,
        seed=7,
    )

    started = time.monotonic()
    solution = solver.solve_with_scorer(problem, scorer, objective_kind="test_score")
    deadline = started + time_limit
    elapsed = time.monotonic() - started

    assert solution.status == "FEASIBLE"
    assert solution.metadata["stop_reason"] == "time_limit"
    assert len(scorer.call_times) >= 2, scorer.call_times
    assert len([t for t in scorer.call_times if t >= deadline]) <= 1, scorer.call_times
    assert elapsed <= time_limit + scorer.delay + 0.5, elapsed


# --------------------------------------------------------------------------- #
# kernel and state
# --------------------------------------------------------------------------- #
def test_uniform_cuts_prefer_pair_feasible_cuts() -> None:
    problem = make_solver_contract_problem()
    for node, schools in enumerate([0, 2, 2, 0]):
        problem.G.nodes[node]["num_schools"] = schools
    context = _ReComContext(problem)
    state = context.build_state(context.validate_hint(problem.hint or {}))

    for seed in range(10):
        rng = random.Random(seed)
        policy = Policy(context, rng, PRESETS["chain"], objective_local=True)
        cut = _ReComKernel(context, rng, None).propose(state, policy).cut
        assert cut.stats_a.schools == 2
        assert cut.stats_b.schools == 2
        assert cut.pair_feasible


def test_wilson_sampler_returns_a_spanning_tree() -> None:
    context = _ReComContext(make_solver_contract_problem())
    kernel = _ReComKernel(context, random.Random(4), None)
    nodes = set(range(len(context.nodes)))
    adjacency = {node: context.adjacency[node] for node in nodes}

    tree = kernel._random_spanning_tree(sorted(nodes), adjacency)

    graph = nx.Graph((u, v) for u, neighbors in tree.items() for v in neighbors)
    assert set(graph) == nodes
    assert nx.is_tree(graph)


@pytest.mark.parametrize("preset", ["uniform_bursts", "entropy", "hedge", "dual"])
def test_incremental_state_matches_a_full_rebuild(preset: str) -> None:
    hint = {node: 0 if node % 5 < 2 else 1 for node in range(25)}
    problem = make_grid_problem(5, 5, hint=hint, boundary_prop=0.3)
    context = _ReComContext(problem)
    state = context.build_state(context.validate_hint(hint))
    rng = random.Random(11)
    kernel = _ReComKernel(context, rng, None)
    policy = Policy(context, rng, PRESETS[preset], objective_local=True)

    for _ in range(50):
        kernel.apply(state, kernel.propose(state, policy))
        rebuilt = context.build_state(list(state.assignment))

        assert state.zone_nodes == rebuilt.zone_nodes
        for actual, expected in zip(state.zone_stats, rebuilt.zone_stats):
            assert actual.node_count == expected.node_count
            assert actual.students == pytest.approx(expected.students)
            assert actual.values == pytest.approx(expected.values)
            assert actual.schools == pytest.approx(expected.schools)
        for actual, expected in zip(state.zone_residuals, rebuilt.zone_residuals):
            assert actual == pytest.approx(expected)
        assert state.boundary_pairs == rebuilt.boundary_pairs
        assert state.boundary_costs == rebuilt.boundary_costs
        assert state.boundary_cost == rebuilt.boundary_cost
        assert state.cut_edges == rebuilt.cut_edges
        assert state.violations == pytest.approx(rebuilt.violations)


def test_residuals_are_percent_of_the_zone_and_clip_to_violations() -> None:
    problem = make_grid_problem(2, 2, shortage=0.0, overage=0.0)
    problem.G.nodes[0]["ge_students"] = 100.0
    context = _ReComContext(problem)

    state = context.build_state([0, 0, 1, 1])

    # Zone 0 holds 101 students and 2 seats; the shortage row needs 101 seats.
    assert state.zone_residuals[0][0] == pytest.approx(99.0 / 101.0 * 100.0)
    assert state.zone_violations[1][0] == 0.0
    assert state.zone_residuals[1][0] <= 0.0
    for residuals, violations in zip(state.zone_residuals, state.zone_violations):
        assert violations == tuple(max(0.0, g) for g in residuals)


def test_school_count_tolerance_zero_penalizes_uneven_school_counts() -> None:
    problem = make_solver_contract_problem(school_count_tolerance=0.0)
    context = _ReComContext(problem)
    assert context.school_bounds == (3.0, 3.0)

    even = context.build_state(context.validate_hint({0: 0, 1: 0, 2: 1, 3: 1}))
    lopsided = context.build_state(context.validate_hint({0: 0, 1: 1, 2: 1, 3: 1}))

    assert even.feasible
    assert not lopsided.feasible


# --------------------------------------------------------------------------- #
# policy pieces
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [0.2, 0.3, 0.5, 0.9])
def test_entropy_temperature_hits_its_target(target: float) -> None:
    scores = [-x * x for x in range(12)]

    probabilities = Temperature(entropy=target).probabilities(scores)

    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    assert entropy / math.log(len(scores)) == pytest.approx(target, abs=0.01)
    assert entropy_beta([1.0, 1.0, 1.0], target) == 0.0


def test_entropy_temperature_shares_ties_that_already_reach_the_target() -> None:
    probabilities = Temperature(entropy=0.3).probabilities([0.0, 0.0, -5.0, -9.0])

    assert probabilities == pytest.approx([0.5, 0.5, 0.0, 0.0])


def test_fixed_temperature_is_a_plain_softmax() -> None:
    probabilities = Temperature(tau=2.0).probabilities([0.0, -2.0])

    assert probabilities[0] / probabilities[1] == pytest.approx(math.e)


def _zone_state(zone_violations: list[tuple[float, ...]], pairs) -> _State:
    """Single-node zones with hand-set violations, for pair selection only."""
    zones = len(zone_violations)
    return _State(
        assignment=list(range(zones)),
        zone_nodes=[{z} for z in range(zones)],
        zone_stats=[_ZoneStats(1, 1.0, (1.0, 1.0), 1.0)] * zones,
        zone_residuals=list(zone_violations),
        zone_violations=list(zone_violations),
        violations=tuple(sum(v) for v in zip(*zone_violations)),
        boundary_pairs={pair: 1 for pair in pairs},
        boundary_costs={pair: 1 for pair in pairs},
        boundary_cost=len(pairs),
    )


class _OneRowContext:
    zone_count = 4
    zone_row_count = 1
    cut_row = None


_CYCLE = [(0, 1), (1, 2), (2, 3), (0, 3)]


def test_zone_pairs_target_the_violated_zone_with_a_random_neighbor() -> None:
    state = _zone_state([(0.0,), (20.0,), (0.0,), (0.0,)], _CYCLE)
    weights = DualWeights(_OneRowContext(), PRESETS["dual"])
    settings = replace(PRESETS["dual"], exploration=0.0)
    pairs = ZonePairs(weights, settings, random.Random(3), 0.0)

    drawn = [pairs.choose(state, _CYCLE) for _ in range(100)]

    assert all(1 in pair for pair in drawn)
    assert set(drawn) == {(0, 1), (1, 2)}


def test_zone_pairs_keep_an_exploration_floor() -> None:
    state = _zone_state([(0.0,), (20.0,), (0.0,), (0.0,)], _CYCLE)
    weights = DualWeights(_OneRowContext(), PRESETS["dual"])
    pairs = ZonePairs(weights, PRESETS["dual"], random.Random(5), 0.0)

    assert (2, 3) in {pairs.choose(state, _CYCLE) for _ in range(200)}


class _PooledContext(_OneRowContext):
    def row_students(self, stats: _ZoneStats) -> tuple[float, ...]:
        return (stats.students,)


def _signed_state(residuals: list[float], students: list[float] | None = None):
    """One-row zones with signed residuals: positive is over, negative slack."""
    state = _zone_state([(max(0.0, g),) for g in residuals], _CYCLE)
    state.zone_residuals = [(g,) for g in residuals]
    if students is not None:
        state.zone_stats = [_ZoneStats(1, s, (1.0, s), 1.0) for s in students]
    return state


def test_pooled_scores_pairs_by_what_the_other_zone_can_absorb() -> None:
    # Zones 0 and 1 are both over; 3 has two points of slack, 2 one.
    state = _signed_state([3.0, 1.0, -1.0, -2.0])
    settings = replace(PRESETS["exact_penalty"], exploration=0.0)
    weights = Weights(_PooledContext(), settings)

    pressure = ProportionalPairs(weights, settings, random.Random(1), 0.0)
    pooled = ProportionalPairs(
        weights, replace(settings, pair_score="pooled"), random.Random(1), 0.0
    )

    # _CYCLE is (0, 1), (1, 2), (2, 3), (0, 3).
    assert pressure._pressures(state, _CYCLE) == [4.0, 1.0, 0.0, 3.0]
    assert pooled._pressures(state, _CYCLE) == [0.0, 1.0, 0.0, 2.0]
    assert {pooled.choose(state, _CYCLE) for _ in range(100)} == {(1, 2), (0, 3)}


def test_pooled_converts_slack_through_student_counts() -> None:
    # 4% of 100 students over is 4 students; 2% slack in 50 absorbs 1 of them.
    state = _signed_state([4.0, -2.0, 0.0, 0.0], students=[100.0, 50.0, 1.0, 1.0])
    settings = replace(PRESETS["exact_penalty"], pair_score="pooled")
    pairs = ProportionalPairs(
        Weights(_PooledContext(), settings), settings, random.Random(1), 0.0
    )

    assert pairs._pooled(state, 0, 1) == pytest.approx(1.0)


def test_pooled_hedge_pressure_is_the_absorbable_fraction() -> None:
    state = _signed_state([3.0, 1.0, -1.0, -2.0])
    settings = replace(PRESETS["hedge"], pair_score="pooled")
    pairs = ZonePairs(
        HedgeWeights(_PooledContext(), settings), settings, random.Random(1), 0.0
    )

    assert pairs._pooled(state, 0, 3) == pytest.approx(2.0 / 3.0)
    assert pairs._pooled(state, 1, 2) == pytest.approx(1.0)


def test_pooled_zone_pairs_pick_the_neighbour_with_slack() -> None:
    # Zone 0 is the only violated zone; neighbour 1 is over too, 3 has slack.
    state = _signed_state([3.0, 0.5, 0.0, -2.0])
    state.zone_violations[1] = (0.0,)
    settings = replace(PRESETS["dual"], exploration=0.0, pair_score="pooled")
    pairs = ZonePairs(
        DualWeights(_PooledContext(), settings), settings, random.Random(2), 0.0
    )

    assert {pairs.choose(state, _CYCLE) for _ in range(100)} == {(0, 3)}


def test_hedge_raises_rows_that_stay_violated_and_keeps_mean_one() -> None:
    context = _OneRowContext()
    weights = HedgeWeights(context, PRESETS["hedge"])
    state = _zone_state([(0.0,), (5.0,), (0.0,), (0.0,)], _CYCLE)

    for _ in range(10):
        weights.observe(state)
        weights.update(state)

    assert weights.w[1] > 1.0 > weights.w[0]
    assert sum(weights.w) / len(weights.w) == pytest.approx(1.0)
    assert min(weights.w) > 0.0  # fixed share: no row is written off
    assert weights.pressure(1, (5.0,)) == weights.w[1]
    assert weights.pressure(0, (0.0,)) == 0.0


def test_dual_multipliers_rise_on_violation_and_fall_on_slack() -> None:
    context = _OneRowContext()
    weights = DualWeights(context, replace(PRESETS["dual"], learning_rate=0.5))
    violated = _zone_state([(30.0,), (0.0,), (0.0,), (0.0,)], _CYCLE)

    weights.update(violated)
    weights.update(violated)
    assert weights.lam[0] == pytest.approx(1.0)  # the step is clipped at 1 point
    assert weights.weight(0) == pytest.approx(2.0)

    slack = _zone_state([(-30.0,), (0.0,), (0.0,), (0.0,)], _CYCLE)
    for _ in range(5):
        weights.update(slack)
    assert weights.lam[0] == 0.0  # projected: never negative


def test_exp3_learns_to_avoid_a_pair_whose_moves_do_damage() -> None:
    state = _zone_state([(1.0,), (1.0,), (1.0,), (1.0,)], _CYCLE)
    weights = DualWeights(_OneRowContext(), PRESETS["bandit"])
    settings = replace(PRESETS["bandit"], bandit_window=200.0)
    pairs = Exp3Pairs(weights, settings, random.Random(1), 1.0)

    for _ in range(400):
        pair = pairs.choose(state, _CYCLE)
        pairs.feedback(0.0, 5.0 if pair == (0, 1) else -5.0)

    late = [pairs.choose(state, _CYCLE) for _ in range(200)]
    assert late.count((0, 1)) < 200 / len(_CYCLE) / 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _problem_without_feasible_contiguous_partition():
    problem = make_solver_contract_problem()
    for node, value in enumerate([1.0, 1.0, 0.0, 0.0]):
        problem.G.nodes[node]["FRL"] = value
    problem.G.graph["F"] = 0.5
    problem.frl_dev = 0.1
    return problem


def _assert_valid_recom_solution(problem, solution) -> None:
    assert solution.status == "FEASIBLE"
    assert set(solution.assignment) == set(problem.nodes)
    assert set(solution.assignment.values()) == set(range(problem.Z))
    for node, zone in solution.assignment.items():
        if problem.candidates is not None and node in problem.candidates:
            assert zone in problem.candidates[node]
        elif problem.fixed is not None and node in problem.fixed:
            assert zone == problem.fixed[node]

    for zone in range(problem.Z):
        nodes = [
            node for node, assigned in solution.assignment.items() if assigned == zone
        ]
        assert nx.is_connected(problem.G.subgraph(nodes))
        students = sum(problem.students(node) for node in nodes)
        for constraint in balance_constraints(problem):
            value = sum(constraint.value(node) for node in nodes)
            if constraint.lower_ratio is not None:
                assert value >= constraint.lower_ratio * students - 1e-6
            if constraint.upper_ratio is not None:
                assert value <= constraint.upper_ratio * students + 1e-6

    total_schools = sum(problem.num_schools(node) for node in problem.nodes)
    if total_schools:
        average = total_schools / problem.Z
        for zone in range(problem.Z):
            schools = sum(
                problem.num_schools(node)
                for node, assigned in solution.assignment.items()
                if assigned == zone
            )
            assert max(0.0, average - 1.0) - 1e-6 <= schools <= average + 1.0 + 1e-6

    assert solution.objective == boundary_cost(
        problem.G, solution.assignment, weight_edges=problem.weight_edges
    )
    assert solution.is_contiguous()

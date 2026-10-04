"""Adaptive short bursts: selectable objectives and the cut-edge constraint row."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from optimization.config import OptimizationConfig
from optimization.data.contiguity import boundary_edges
from optimization.data.mid import MidMarket, MidProgram, MidType
from optimization.mid_oracle import finite_grid_oracle
from optimization.solvers import burst_objectives
from optimization.solvers import get_solver
from optimization.solvers.recom import (
    _Lagrangian,
    _NoProposal,
    _ReComContext,
    _ReComKernel,
)
from optimization.tests.synthetic import make_grid_problem

# 2x2 grid: nodes 0 1 / 2 3, schools at 0 and 3. The two splits that keep the
# schools apart are A = {0,1}|{2,3} and B = {0,2}|{1,3}; the school-count band
# of 1 +/- 1 also admits zones with no school.
SPLIT_A = {0: 0, 1: 0, 2: 1, 3: 1}
SPLIT_B = {0: 0, 2: 0, 1: 1, 3: 1}


# --------------------------------------------------------------------------- #
# cut-edge constraint row
# --------------------------------------------------------------------------- #
def test_boundary_prop_adds_a_cut_edge_row_to_the_violation_vector() -> None:
    problem = make_grid_problem(3, 3, boundary_prop=0.25)
    context = _ReComContext(problem)
    assignment = [0, 1, 0, 1, 0, 1, 0, 1, 0]  # a checkerboard cuts every edge
    state = context.build_state(assignment)

    limit = math.floor(0.25 * problem.G.number_of_edges())
    assert context.cut_row == context.violation_count - 1
    assert context.violation_labels()[context.cut_row] == "boundary"
    assert state.cut_edges == problem.G.number_of_edges()
    assert state.violations[context.cut_row] == pytest.approx(state.cut_edges - limit)
    assert not state.feasible


def test_no_cut_edge_row_without_boundary_prop() -> None:
    context = _ReComContext(make_grid_problem(3, 3))

    assert context.cut_row is None
    assert "boundary" not in context.violation_labels()
    assert context.violation_count == context.zone_row_count


def test_incremental_cut_edges_match_a_rebuild() -> None:
    problem = make_grid_problem(4, 4, boundary_prop=0.2, weight_edges=False)
    for normalize in (False, True):
        context = _ReComContext(problem, normalize_fractional=normalize)
        state = context.build_state([0 if pos % 4 < 2 else 1 for pos in range(16)])
        kernel = _ReComKernel(context, random.Random(3), None)
        for _ in range(40):
            try:
                kernel.apply(state, kernel.propose(state, "uniform"))
            except _NoProposal:
                continue
            rebuilt = context.build_state(list(state.assignment))
            assert state.cut_edges == rebuilt.cut_edges
            assert state.violations == pytest.approx(rebuilt.violations)


def test_cut_cap_enters_the_lagrangian_without_a_cut_edge_objective() -> None:
    """A constraint is penalized whatever the objective is."""

    problem = make_grid_problem(3, 3, boundary_prop=0.1)
    context = _ReComContext(problem)
    state = context.build_state([0, 0, 0, 1, 1, 1, 1, 1, 1])
    cut_row = context.cut_row
    assert state.violations[cut_row] > 0

    weights = tuple(
        3.0 if row == cut_row else 0.0 for row in range(len(state.violations))
    )
    no_objective = _Lagrangian(weights, objective_weight=0.0)

    penalty = 3.0 * state.violations[cut_row] ** 2
    assert no_objective.candidate(state.boundary_cost, state.violations) == (
        pytest.approx(penalty)
    )
    # Its per-zone shares add back up to the whole penalty.
    assert sum(no_objective.zone(context, state, zone) for zone in (0, 1)) == (
        pytest.approx(penalty)
    )
    # As both objective and constraint, both terms are present.
    both = _Lagrangian(weights, objective_weight=1.0)
    assert both.candidate(state.boundary_cost, state.violations) == pytest.approx(
        state.boundary_cost + penalty
    )


def test_adaptive_short_bursts_enforce_boundary_prop() -> None:
    problem = make_grid_problem(4, 4, boundary_prop=0.2)
    limit = math.floor(0.2 * problem.G.number_of_edges())

    solution = get_solver(
        "adaptive_short_bursts",
        recom_iterations=300,
        short_bursts_length=10,
        seed=11,
        adaptive_short_bursts_objective="cut_edges",
    ).solve(problem)

    assert solution.status == "FEASIBLE"
    assert boundary_edges(problem.G, solution.assignment) <= limit
    assert solution.metadata["best_cut_edges"] <= limit
    assert "boundary" in solution.metadata["violation_labels"]


# --------------------------------------------------------------------------- #
# choice objectives
# --------------------------------------------------------------------------- #
def _choice_problem(**overrides):
    params = dict(
        frl_dev=1.0,
        overage=-1.0,
        shortage=-1.0,
        hint=dict(SPLIT_A),
        program_population="All",
        optimization_config=SimpleNamespace(),
    )
    params.update(overrides)
    return make_grid_problem(2, 2, **params)


def _capacity_market() -> MidMarket:
    """Two at node 1 and three at node 2 all rank only school 100's program.

    Split A seats node 1 and designates node 2 into school 200's three free
    seats: 0 unassigned, 3 designated.  Split B seats two of node 2, strands
    the third in a zone with no free seat, and designates node 1: about 1
    unassigned and 2 designated (the 20-point lottery grid seats 1.95 of node
    2, not 2).  Nobody ranks school 200, so its seats only exist for
    designation, which preprocessing must not lose.
    """

    return MidMarket(
        programs=(
            MidProgram("100-GE-KG", 100, 2, False, 0),
            MidProgram("200-GE-KG", 200, 3, False, 3),
        ),
        types=(
            MidType(1, 2, ("100-GE-KG",), (0,), (2.0,), (200,)),
            MidType(2, 3, ("100-GE-KG",), (0,), (3.0,), (300,)),
        ),
        student_count=5,
        outside_only_student_count=0,
        utility_student_count=5,
        utility_handling="omit_nonpositive",
    )


def _welfare_market() -> MidMarket:
    """Preferences favor split B."""

    return MidMarket(
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


def _use_market(monkeypatch, market: MidMarket) -> None:
    monkeypatch.setattr(
        burst_objectives, "build_mid_market", lambda problem, config: market
    )


def _same_partition(assignment, split) -> bool:
    return all(
        (assignment[u] == assignment[v]) == (split[u] == split[v])
        for u in split
        for v in split
    )


@pytest.mark.parametrize(
    ("split", "expected"),
    [(SPLIT_A, (0.0, 3.0)), (SPLIT_B, (1.0, 2.05))],
)
def test_designation_split_counts_unassigned_and_designated(split, expected) -> None:
    problem = _choice_problem()
    raw = _capacity_market()
    market = burst_objectives.preprocess_mid_market(raw, problem)
    result = finite_grid_oracle(market, split, 20, check_minimality=False)

    split_counts = burst_objectives.designation_split(
        market, burst_objectives.designation_seats(raw), split, result, 20
    )

    assert split_counts == pytest.approx(expected)


def _adaptive(**options):
    options.setdefault("recom_iterations", 40)
    options.setdefault("short_bursts_length", 5)
    options.setdefault("seed", 1)
    options.setdefault("workers", 1)
    return get_solver("adaptive_short_bursts", **options)


def _feasible_splits(problem) -> list[dict[int, int]]:
    """Every feasible contiguous two-zone partition, by brute force."""

    context = _ReComContext(problem, normalize_fractional=True)
    splits = []
    for mask in range(1, 2 ** len(context.nodes) - 1):
        assignment = {node: (mask >> pos) & 1 for pos, node in enumerate(context.nodes)}
        try:
            positions = context.validate_hint(assignment)
        except ValueError:
            continue
        if context.build_state(positions).feasible:
            splits.append(assignment)
    return splits


def _scores(market, problem, metric, **weights) -> list[tuple[float, dict]]:
    raw = market
    evaluator = burst_objectives.MidBatchEvaluator(
        burst_objectives.preprocess_mid_market(raw, problem),
        20,
        1,
        burst_objectives.designation_seats(raw),
    )
    scorer = burst_objectives.MidBurstScorer(evaluator, metric, **weights)
    splits = _feasible_splits(problem)
    return list(zip(scorer(tuple(splits), None), splits, strict=True))


@pytest.mark.parametrize(
    ("unassigned_weight", "designated_weight"),
    [(10.0, 1.0), (0.0, 1.0), (1.0, 0.0)],
)
def test_capacity_match_reaches_the_weighted_optimum(
    monkeypatch, unassigned_weight, designated_weight
) -> None:
    market = _capacity_market()
    _use_market(monkeypatch, market)
    problem = _choice_problem()
    best = min(
        _scores(
            market,
            problem,
            "capacity_match",
            unassigned_weight=unassigned_weight,
            designated_weight=designated_weight,
        ),
        key=lambda item: item[0],
    )[0]

    solution = _adaptive(
        adaptive_short_bursts_objective="capacity_match",
        adaptive_short_bursts_unassigned_weight=unassigned_weight,
        adaptive_short_bursts_designated_weight=designated_weight,
    ).solve(problem)

    assert solution.status == "FEASIBLE"
    assert solution.objective == pytest.approx(best)
    meta = solution.metadata
    assert meta["objective_kind"] == "capacity_match"
    assert meta["adaptive_short_bursts_objective"] == "capacity_match"
    assert solution.objective == pytest.approx(
        unassigned_weight * meta["best_unassigned_students"]
        + designated_weight * meta["best_designated_students"]
    )


def test_capacity_match_weights_change_the_answer(monkeypatch) -> None:
    _use_market(monkeypatch, _capacity_market())

    def solve(unassigned_weight, designated_weight):
        return _adaptive(
            adaptive_short_bursts_objective="capacity_match",
            adaptive_short_bursts_unassigned_weight=unassigned_weight,
            adaptive_short_bursts_designated_weight=designated_weight,
        ).solve(_choice_problem())

    # Unassigned students dominate: everyone is placed, three by designation.
    placed = solve(10.0, 1.0)
    assert _same_partition(placed.assignment, SPLIT_A)
    assert placed.metadata["best_unassigned_students"] == pytest.approx(0.0)
    # Designations alone are penalized: the walk strands students instead.
    stranded = solve(0.0, 1.0)
    assert stranded.metadata["best_designated_students"] < 3.0
    assert stranded.metadata["best_unassigned_students"] > 0.0


def test_welfare_objective_maximizes_mid_welfare(monkeypatch) -> None:
    market = _welfare_market()
    _use_market(monkeypatch, market)
    problem = _choice_problem()
    best = max(_scores(market, problem, "welfare"), key=lambda item: item[0])[0]

    solution = _adaptive(adaptive_short_bursts_objective="welfare").solve(problem)

    expected = finite_grid_oracle(
        burst_objectives.preprocess_mid_market(market, problem),
        solution.assignment,
        20,
        check_minimality=False,
    )
    assert solution.status == "FEASIBLE"
    assert solution.objective == pytest.approx(expected.welfare)
    assert solution.objective == pytest.approx(best)
    assert solution.metadata["objective_kind"] == "welfare"


def test_welfare_objective_still_enforces_boundary_prop(monkeypatch) -> None:
    """The cap reaches the Lagrangian and feasibility with no cut-edge objective."""

    _use_market(monkeypatch, _welfare_market())
    # Both splits cut two of the four edges; a cap of one rules both out.
    problem = _choice_problem(boundary_prop=0.25)

    solution = _adaptive(adaptive_short_bursts_objective="welfare").solve(problem)

    assert solution.status == "UNKNOWN"
    assert solution.metadata["initial_feasible"] is False
    assert "boundary" in solution.metadata["violation_labels"]


def test_choice_objectives_require_the_all_program_population(monkeypatch) -> None:
    _use_market(monkeypatch, _welfare_market())

    with pytest.raises(ValueError, match="program_population='All'"):
        _adaptive(adaptive_short_bursts_objective="welfare").solve(
            _choice_problem(program_population="GE")
        )


def test_unknown_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="adaptive_short_bursts_objective"):
        _adaptive(adaptive_short_bursts_objective="nonsense").solve(_choice_problem())


def test_config_passes_adaptive_objective_options() -> None:
    solver = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="adaptive_short_bursts",
        adaptive_short_bursts_objective="capacity_match",
        adaptive_short_bursts_unassigned_weight=4,
        adaptive_short_bursts_designated_weight=0.5,
    ).make_solver()

    assert solver.options["adaptive_short_bursts_objective"] == "capacity_match"
    assert solver.options["adaptive_short_bursts_unassigned_weight"] == 4.0
    assert solver.options["adaptive_short_bursts_designated_weight"] == 0.5
    assert (
        OptimizationConfig(levels=["BlockGroup_0"])
        .make_solver()
        .options["adaptive_short_bursts_objective"]
        == "cut_edges"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"adaptive_short_bursts_objective": "bad"}, "adaptive_short_bursts_objective"),
        ({"adaptive_short_bursts_unassigned_weight": -1}, "unassigned_weight"),
        (
            {"adaptive_short_bursts_designated_weight": float("nan")},
            "designated_weight",
        ),
        (
            {
                "adaptive_short_bursts_objective": "capacity_match",
                "adaptive_short_bursts_unassigned_weight": 0,
                "adaptive_short_bursts_designated_weight": 0,
            },
            "positive unassigned or designated weight",
        ),
    ],
)
def test_config_rejects_invalid_adaptive_objective_options(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        OptimizationConfig(levels=["BlockGroup_0"], **overrides)


def test_choice_objectives_refuse_a_market_without_utilities(monkeypatch) -> None:
    """An estimate that covers nobody makes every zoning score the same."""

    market = MidMarket(
        programs=(MidProgram("100-GE-KG", 100, 2, False, 0),),
        types=(MidType(1, 2, (), (), (), ()),),
        student_count=2,
        outside_only_student_count=2,
        utility_student_count=0,
        utility_handling="omit_nonpositive",
    )
    _use_market(monkeypatch, market)

    with pytest.raises(ValueError, match="no student with a choice-utility row"):
        _adaptive(adaptive_short_bursts_objective="capacity_match").solve(
            _choice_problem()
        )

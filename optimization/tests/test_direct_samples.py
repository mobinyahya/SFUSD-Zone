"""Tests for the Boolean per-seed stable-matching model.

The whole claim of this formulation is *exactness without separation*: with
Boolean assignment variables the aggregated stability row (A4) alone describes
the stable matchings of the market a zoning leaves, so at any fixed zoning the
integral optimum is the welfare of the student-optimal stable matching for that
zoning and that lottery draw. Nothing else about it matters if that is false,
so the gate below computes deferred acceptance from scratch in this file -- a
plain student-proposing loop over the sampled priority orders, short enough to
read -- rather than calling `optimization.direct_samples`, which is the code
under test.

`SaaOracleResult.welfare` is deliberately *not* used as ground truth. It is the
optimum of the continuous stable-admissions LP over the same row family, which
is a relaxation rather than a description of the matching polytope, and on the
real instance it over-reports realised DA by 195-252 welfare -- the very gap
Boolean variables are here to remove.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from optimization.config import OptimizationConfig
from optimization.data.mid import MidProgram, MidStudent
from optimization.data.saa import (
    SaaMarket,
    SaaSample,
    preprocess_saa_market,
    sample_school_preferences,
)
from optimization.direct_samples import (
    access_mask,
    deferred_acceptance,
    matching_welfare,
    replay,
    scaled_matching_welfare,
)
from optimization.saa_oracle import access_state
from optimization.solvers import get_solver
from optimization.solvers.direct_samples import DirectSamplesCpSatSolver
from optimization.strategies import get_strategy
from optimization.strategies import direct_samples as direct_samples_module
from optimization.tests.synthetic import FakeDataset, make_grid_problem


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
def _problem():
    """3x3 grid, two zones anchored at opposite corners, penalties disabled."""

    return make_grid_problem(3, 3, program_population="All", overage=-1, shortage=-1)


def _market() -> SaaMarket:
    """A market tight enough that access changes who gets seated.

    Total capacity is 4 seats against 7 students, both zoned programs are
    oversubscribed, and the citywide program is the only alternative -- so the
    matching actually binds and moving a node between zones moves welfare.
    """

    programs = (
        MidProgram("A", 100, 2, False, 0),
        MidProgram("B", 200, 1, False, 8),
        MidProgram("C", 300, 1, True, None),
    )
    students = (
        # (node, programs, priorities, utilities, scaled utilities)
        MidStudent(1, ("A", "B", "C"), (1, 0, 0), (9.0, 5.0, 1.0), (900, 500, 100)),
        MidStudent(2, ("B", "A"), (0, 0), (8.0, 4.0), (800, 400)),
        MidStudent(3, ("A", "C"), (0, 1), (7.0, 2.0), (700, 200)),
        MidStudent(4, ("B", "A", "C"), (1, 1, 0), (6.0, 3.5, 1.5), (600, 350, 150)),
        MidStudent(5, ("A", "B"), (2, 0), (5.5, 5.0), (550, 500)),
        MidStudent(6, ("C", "B"), (0, 2), (4.0, 3.0), (400, 300)),
        MidStudent(7, ("A",), (0,), (2.5,), (250,)),
    )
    return SaaMarket(
        programs=programs,
        students=students,
        utility_student_count=len(students),
        utility_handling="omit_nonpositive",
    )


def _instance(problem, *, num_seeds=1, base_seed=5):
    """A preprocessed market and the priority orders drawn on it, in that order.

    Preprocessing renumbers programs and the draws index programs by position,
    so building the two the other way round would attach each program the order
    of a different one.
    """

    market = preprocess_saa_market(_market(), problem)
    return market, sample_school_preferences(market, num_seeds, "STB", base_seed)


def _zonings(problem, count=8):
    """A spread of two-zone assignments respecting the centroid anchors."""

    free = [node for node in problem.nodes if node not in (0, 8)]
    every = [
        {0: 0, 8: 1, **dict(zip(free, combination))}
        for combination in itertools.product((0, 1), repeat=len(free))
    ]
    # A coprime stride rather than a contiguous block: enumeration order varies
    # the last node fastest, so the first few assignments differ in one vertex
    # and would make the gates eight repetitions of the same test.
    stride = 37
    return [every[(index * stride) % len(every)] for index in range(count)]


# ---------------------------------------------------------------------- #
# Independent ground truth: student-proposing deferred acceptance
# ---------------------------------------------------------------------- #
def _deferred_acceptance(problem, market, sample, zoning) -> tuple[float, int]:
    """Welfare of the student-optimal stable matching, computed from scratch.

    Student-proposing DA returns the stable matching every student weakly
    prefers to every other stable matching, so it maximises total student
    utility over the stable set -- exactly what the CP-SAT block is supposed to
    report. Written as an explicit propose/reject loop with no heaps and no
    shared code, and returns the welfare in both the market's own utilities and
    the scaled integers the solver accumulates.
    """

    programs = market.program_by_id
    rank_of = {
        (student_index, program.program_id): position
        for index, program in enumerate(market.programs)
        for position, student_index in enumerate(sample.school_orders[index])
    }
    capacity = {
        program.program_id: int(program.capacity) for program in market.programs
    }

    # Each student's preference list, filtered to what this zoning reaches.
    lists = []
    for student in market.students:
        options = []
        for rank, program_id in enumerate(student.programs):
            pair, fixed = access_state(problem, student.node, programs[program_id])
            reachable = (
                bool(fixed) if pair is None else zoning[pair[0]] == zoning[pair[1]]
            )
            if reachable:
                options.append(
                    (
                        program_id,
                        student.utilities[rank],
                        student.scaled_utilities[rank],
                    )
                )
        lists.append(options)

    held: dict[str, list[int]] = {program_id: [] for program_id in capacity}
    matched: dict[int, tuple[float, int]] = {}
    next_offer = [0] * len(lists)
    free = [index for index, options in enumerate(lists) if options]
    while free:
        student_index = free.pop()
        while next_offer[student_index] < len(lists[student_index]):
            offer = lists[student_index][next_offer[student_index]]
            program_id, utility, scaled = offer
            next_offer[student_index] += 1
            if capacity[program_id] <= 0:
                continue
            roster = held[program_id]
            if len(roster) < capacity[program_id]:
                roster.append(student_index)
                matched[student_index] = (utility, scaled)
                break
            worst = max(roster, key=lambda other: rank_of[(other, program_id)])
            if rank_of[(student_index, program_id)] < rank_of[(worst, program_id)]:
                roster.remove(worst)
                roster.append(student_index)
                matched[student_index] = (utility, scaled)
                del matched[worst]
                free.append(worst)
                break
    return (
        sum(utility for utility, _ in matched.values()),
        sum(scaled for _, scaled in matched.values()),
    )


# ---------------------------------------------------------------------- #
# Model harness: the matching block over a pinned zoning
# ---------------------------------------------------------------------- #
def _fixed_model(problem, market, samples, zoning):
    """The matching block alone, with ``x`` pinned to ``zoning``.

    Builds no zoning rows, so the contiguity, FRL and school-count constraints
    cannot make an enumerated zoning infeasible and silently turn the
    comparison into a comparison of nothing.
    """

    solver = DirectSamplesCpSatSolver(market, samples)
    model = cp_model.CpModel()
    x = {}
    for node in problem.nodes:
        for zone in problem.candidate_zones(node):
            variable = model.NewBoolVar(f"x_{zone}_{node}")
            model.Add(variable == int(zoning[node] == zone))
            x[(zone, node)] = variable
    solver._add_model_objective(model, problem, x, {})
    return solver, model, x


def _solve_fixed(problem, market, samples, zoning) -> int:
    """Raw (scaled, summed over seeds) optimum of the pinned matching block."""

    _, model, _ = _fixed_model(problem, market, samples, zoning)
    cp_solver = cp_model.CpSolver()
    cp_solver.parameters.num_search_workers = 1
    status = cp_solver.Solve(model)
    assert status == cp_model.OPTIMAL
    return int(round(cp_solver.ObjectiveValue()))


# ---------------------------------------------------------------------- #
# GATE: exactness
# ---------------------------------------------------------------------- #
def test_integral_optimum_is_deferred_acceptance_welfare():
    problem = _problem()
    market, samples = _instance(problem)

    for zoning in _zonings(problem):
        _, scaled_truth = _deferred_acceptance(problem, market, samples[0], zoning)
        assert _solve_fixed(problem, market, samples, zoning) == scaled_truth, zoning


def test_exactness_holds_across_lottery_draws():
    """Not an artefact of one draw: re-sample and re-check."""

    problem = _problem()
    for base_seed in (1, 2, 3):
        market, samples = _instance(problem, base_seed=base_seed)
        zoning = _zonings(problem, count=3)[1]
        _, scaled_truth = _deferred_acceptance(problem, market, samples[0], zoning)
        assert _solve_fixed(problem, market, samples, zoning) == scaled_truth


def test_ground_truth_is_sensitive_to_the_zoning():
    """Guard the gate itself: a constant truth would make it vacuous."""

    problem = _problem()
    market, samples = _instance(problem)
    welfares = {
        _deferred_acceptance(problem, market, samples[0], zoning)[0]
        for zoning in _zonings(problem)
    }
    assert len(welfares) > 1


def test_no_comb_row_can_improve_the_integral_optimum():
    """The point of Boolean variables: nothing is left to separate.

    A comb inequality is valid for the stable matchings, so if the integral
    optimum already *is* a stable matching there is no violated comb to find.
    Checked by confirming the optimal solution has no blocking pair -- the
    property combs exist to enforce and the aggregated row is claimed to give
    for free once ``D`` is Boolean.
    """

    problem = _problem()
    market, samples = _instance(problem)
    sample = samples[0]
    programs = market.program_by_id

    for zoning in _zonings(problem, count=4):
        solver, model, _ = _fixed_model(problem, market, samples, zoning)
        cp_solver = cp_model.CpSolver()
        cp_solver.parameters.num_search_workers = 1
        assert cp_solver.Solve(model) == cp_model.OPTIMAL

        seated = {}
        for (_, student_index, rank), seat in solver._variables.seats.items():
            if cp_solver.Value(seat):
                seated[student_index] = rank
        roster: dict[str, list[int]] = {
            program.program_id: [] for program in market.programs
        }
        for student_index, rank in seated.items():
            roster[market.students[student_index].programs[rank]].append(student_index)

        mask = access_mask(market, problem, zoning)
        position = {
            program.program_id: {
                student: place
                for place, student in enumerate(sample.school_orders[index])
            }
            for index, program in enumerate(market.programs)
        }
        for student_index, student in enumerate(market.students):
            held = seated.get(student_index, len(student.programs))
            for rank in range(held):
                if not mask[student_index][rank]:
                    continue
                program_id = student.programs[rank]
                quota = int(programs[program_id].capacity)
                admitted = roster[program_id]
                better = sum(
                    1
                    for other in admitted
                    if position[program_id][other] < position[program_id][student_index]
                )
                assert better >= quota, (zoning, student_index, program_id)


# ---------------------------------------------------------------------- #
# The sample average
# ---------------------------------------------------------------------- #
def test_multi_sample_objective_is_the_sum_of_the_per_sample_optima():
    problem = _problem()
    market, samples = _instance(problem, num_seeds=3, base_seed=11)
    zoning = _zonings(problem, count=4)[2]

    combined = _solve_fixed(problem, market, samples, zoning)
    separate = [
        _solve_fixed(problem, market, (sample,), zoning) for sample in samples
    ]

    assert combined == sum(separate)
    # And each block is still exact on its own draw.
    for sample, value in zip(samples, separate):
        assert value == _deferred_acceptance(problem, market, sample, zoning)[1]


def test_every_seat_carries_a_stability_row():
    """One (A4) row per seat variable, in every seed. Nothing is skipped.

    A pair whose row went missing would be free to block, which the exactness
    gate would only catch if that pair happened to matter at one of the eight
    enumerated zonings.
    """

    problem = _problem()
    market, samples = _instance(problem, num_seeds=2, base_seed=11)
    solver, _, _ = _fixed_model(problem, market, samples, _zonings(problem)[0])
    variables = solver._variables

    assert len(variables.seats) == 2 * market.preference_count
    assert variables.stability_row_count == len(variables.seats)
    assert len(variables.shafts) == len(variables.seats)


def test_citywide_programs_need_no_access_variable():
    """Access is unconditional there, so it enters as a constant, not a column.

    That is also why a citywide seat carries no (A3) row: there is no variable
    to bound it by. The indicators built must be exactly the zoned
    student-node/school-node pairs a zoning can still decide.
    """

    problem = _problem()
    market, samples = _instance(problem)
    solver, _, _ = _fixed_model(problem, market, samples, _zonings(problem)[0])

    programs = market.program_by_id
    citywide = {program.program_id for program in market.programs if program.citywide}
    assert citywide
    assert any(
        program_id in citywide
        for student in market.students
        for program_id in student.programs
    )
    decidable = {
        tuple(sorted((student.node, programs[program_id].school_node)))
        for student in market.students
        for program_id in student.programs
        if program_id not in citywide
        and student.node != programs[program_id].school_node
    }
    assert set(solver._variables.access) == decidable


def test_samples_are_distinct_draws():
    problem = _problem()
    _, samples = _instance(problem, num_seeds=3, base_seed=11)
    assert len({sample.seed for sample in samples}) == 3


def test_solver_rejects_misaligned_priority_orders():
    problem = _problem()
    market, samples = _instance(problem)
    truncated = SaaSample(
        seed=samples[0].seed,
        school_orders=tuple(
            order[:-1] if index == 0 else order
            for index, order in enumerate(samples[0].school_orders)
        ),
    )
    with pytest.raises(ValueError, match="is not Gamma"):
        DirectSamplesCpSatSolver(market, (truncated,))


# ---------------------------------------------------------------------- #
# The replay: hints, verification and the module's own DA
# ---------------------------------------------------------------------- #
def test_replay_matches_the_independent_deferred_acceptance():
    """The production replay is what verifies the solver, so verify it too."""

    problem = _problem()
    market, samples = _instance(problem, num_seeds=3, base_seed=11)
    for zoning in _zonings(problem):
        matchings = replay(market, samples, problem, zoning)
        for sample, matching in zip(samples, matchings):
            truth, scaled_truth = _deferred_acceptance(problem, market, sample, zoning)
            assert matching_welfare(market, matching) == pytest.approx(truth)
            assert scaled_matching_welfare(market, matching) == scaled_truth


def test_unassigned_students_carry_no_rank():
    problem = _problem()
    market, samples = _instance(problem)
    # Everything in zone 0 but the far corner: program B is unreachable for
    # every student, so the market cannot seat all seven.
    zoning = {node: 0 for node in problem.nodes}
    zoning[8] = 1
    matching = deferred_acceptance(
        market, samples[0], access_mask(market, problem, zoning)
    )
    assert any(rank == -1 for rank in matching)
    assert all(
        rank == -1 or 0 <= rank < len(market.students[index].programs)
        for index, rank in enumerate(matching)
    )


def test_hinted_values_are_a_feasible_optimal_matching():
    """Fixing every hinted variable must leave a feasible model at DA welfare.

    A wrong hint is invisible in a normal run -- CP-SAT repairs it and reports
    the same answer a little slower -- so pin the hint down and solve it.
    """

    problem = _problem()
    market, samples = _instance(problem, num_seeds=2, base_seed=11)
    zoning = _zonings(problem, count=4)[2]
    problem.hint = dict(zoning)

    solver = DirectSamplesCpSatSolver(market, samples)
    model = cp_model.CpModel()
    x = {
        (zone, node): model.NewBoolVar(f"x_{zone}_{node}")
        for node in problem.nodes
        for zone in problem.candidate_zones(node)
    }
    for node in problem.nodes:
        model.AddExactlyOne(
            x[(zone, node)] for zone in problem.candidate_zones(node)
        )
    solver._add_model_objective(model, problem, x, {})
    solver._add_hints(model, problem, x, {})

    assert model.Proto().solution_hint.vars

    cp_solver = cp_model.CpSolver()
    cp_solver.parameters.num_search_workers = 1
    # Every hinted variable becomes a fixed variable, so an inconsistent hint
    # is an infeasible model rather than a silently repaired one.
    cp_solver.parameters.fix_variables_to_their_hinted_value = True
    assert cp_solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    expected = sum(
        _deferred_acceptance(problem, market, sample, zoning)[1] for sample in samples
    )
    assert int(round(cp_solver.ObjectiveValue())) == expected


# ---------------------------------------------------------------------- #
# Solver and strategy wiring
# ---------------------------------------------------------------------- #
def test_strategy_requires_the_cp_bool_solver():
    dataset = FakeDataset(_problem())
    dataset.config = SimpleNamespace(program_population="All")
    strategy = get_strategy("direct_samples", levels=["BlockGroup_0"])

    with pytest.raises(ValueError, match="requires solver='cp_bool'"):
        strategy.run(dataset, get_solver("cp_int"))


def test_strategy_requires_all_program_population():
    dataset = FakeDataset(_problem())
    dataset.config = SimpleNamespace(program_population="GE")
    strategy = get_strategy("direct_samples", levels=["BlockGroup_0"])

    with pytest.raises(ValueError, match="program_population='All'"):
        strategy.run(dataset, get_solver("cp_bool"))


def test_strategy_returns_a_zoning_whose_da_welfare_is_the_objective(monkeypatch):
    """End to end: the joint optimum equals DA at the zoning it chose.

    The model is exact at every fixed zoning, so whatever zoning branch and
    bound settles on, the reported objective has to be that zoning's realised
    matching welfare. Any disagreement means the zoning block and the matching
    block are reading different access structures.
    """

    problem = _problem()
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population="All")
    market, samples = _instance(problem, num_seeds=2, base_seed=7)
    monkeypatch.setattr(
        direct_samples_module, "build_saa_market", lambda *args, **kwargs: market
    )
    monkeypatch.setattr(
        direct_samples_module,
        "sample_school_preferences",
        lambda *args, **kwargs: samples,
    )
    monkeypatch.setattr(
        direct_samples_module, "initial_solution", lambda *args, **kwargs: None
    )
    strategy = get_strategy(
        "direct_samples",
        levels=["BlockGroup_0"],
        solve_time_limits=[60],
        gap_limits=[0],
        hints="none",
        saa_num_seeds=2,
    )

    solutions = strategy.run(dataset, get_solver("cp_bool", workers=1))
    final = solutions[-1]
    truth = sum(
        _deferred_acceptance(problem, market, sample, final.assignment)[0]
        for sample in samples
    ) / len(samples)

    assert final.status == "OPTIMAL"
    assert final.objective == pytest.approx(truth, abs=1e-9)
    assert final.metadata["objective_kind"] == "direct_samples_sample_average_welfare"
    assert final.metadata["direct_samples_welfare"] == pytest.approx(truth, abs=1e-9)
    assert final.metadata["direct_samples_solver_welfare"] == pytest.approx(
        truth, abs=1e-9
    )
    assert final.metadata["direct_samples_replay_agreement"] is True
    assert final.metadata["direct_samples_num_seeds"] == 2
    assert final.metadata["direct_samples_sample_seeds"] == [
        sample.seed for sample in samples
    ]
    assert final.metadata["direct_samples_gamma_size"] == sum(
        len(student.programs) for student in market.students
    )
    assert final.metadata["direct_samples_seat_vars"] == 2 * sum(
        len(student.programs) for student in market.students
    )
    assert final.metadata["direct_samples_access_indicator_count"] > 0
    assert final.metadata["direct_samples_best_objective_bound"] == pytest.approx(
        truth, abs=1e-6
    )
    assert sum(final.metadata["direct_samples_sample_welfares"]) / 2 == pytest.approx(
        truth, abs=1e-9
    )
    assert final.metadata["direct_samples_preprocessing_seconds"] >= 0.0
    # Budget welfare, the cross-strategy yardstick, is reported alongside.
    assert final.metadata["mid_welfare"] == final.metadata["mid_discrete_welfare"]
    assert final.metadata["mid_lottery_scale"] == 20


def test_strategy_warm_starts_the_matching_blocks(monkeypatch):
    """The production hint path: a real warm start, not a monkeypatched one.

    ``hints: none`` leaves the matching blocks unhinted, which is the one
    configuration that never exercises the replay inside `_add_hints`.
    """

    problem = _problem()
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population="All")
    market, samples = _instance(problem, num_seeds=2, base_seed=7)
    monkeypatch.setattr(
        direct_samples_module, "build_saa_market", lambda *args, **kwargs: market
    )
    monkeypatch.setattr(
        direct_samples_module,
        "sample_school_preferences",
        lambda *args, **kwargs: samples,
    )
    strategy = get_strategy(
        "direct_samples",
        levels=["BlockGroup_0"],
        solve_time_limits=[60],
        gap_limits=[0],
        hints="voronoi",
        saa_num_seeds=2,
    )

    final = strategy.run(dataset, get_solver("cp_bool", workers=1))[-1]
    truth = sum(
        _deferred_acceptance(problem, market, sample, final.assignment)[0]
        for sample in samples
    ) / len(samples)

    assert final.metadata["hints"] == "voronoi"
    # One hint per zoning variable, plus the matching block's own.
    assert final.metadata["direct_samples_model_hint_count"] > len(problem.nodes) * 2
    assert final.status == "OPTIMAL"
    assert final.objective == pytest.approx(truth, abs=1e-9)
    assert final.metadata["direct_samples_replay_agreement"] is True


# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #
def test_config_accepts_the_strategy_and_passes_the_sample_options():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="cp_bool",
        strategy="direct_samples",
        saa_num_seeds=4,
        saa_tie_breaking_method="stb",
    )
    options = config.make_strategy().options

    assert options["saa_num_seeds"] == 4
    assert options["saa_tie_breaking_method"] == "STB"

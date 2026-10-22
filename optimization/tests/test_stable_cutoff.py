"""Tests for the exact per-student cutoff model of zoned deferred acceptance.

The whole claim of this formulation is *exactness*: at any fixed zoning the
integral optimum of (F1)-(F6) is the welfare of the student-optimal stable
matching for that zoning and that lottery draw. Nothing else about it matters if
that is false, so the gate below computes deferred acceptance from scratch in
this file -- a plain student-proposing loop over the sampled priority orders,
short enough to read -- rather than calling any production code.

`SaaOracleResult.welfare` is deliberately *not* used as ground truth. It is the
optimum of the continuous stable-admissions LP (see the class docstring in
:mod:`optimization.saa_oracle`), which is a relaxation of the many-to-one stable
matching polytope rather than a description of it, and on the real instance it
over-reports realised DA by 195-252 welfare -- an order of magnitude more than
anything these tests are trying to resolve.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import gurobipy as gp
import pytest
from gurobipy import GRB

from optimization.config import OptimizationConfig
from optimization.data.mid import MidProgram, MidStudent
from optimization.data.saa import SaaMarket, SaaSample
from optimization.saa_oracle import access_state
from optimization.solvers import get_solver
from optimization.solvers.stable_cutoff import StableCutoffMipSolver
from optimization.stable_cutoff import (
    prepare_stable_cutoff_instance,
    undecided_access_pairs,
    validate_sample_orders,
)
from optimization.strategies import get_strategy
from optimization.strategies import stable_cutoff as stable_cutoff_module
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
    cutoffs actually bind and moving a node between zones moves welfare.
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
    return prepare_stable_cutoff_instance(
        _market(),
        problem,
        num_seeds=num_seeds,
        tie_breaking_method="STB",
        base_seed=base_seed,
    )


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
def _deferred_acceptance(problem, market, sample, zoning) -> float:
    """Welfare of the student-optimal stable matching, computed from scratch.

    Student-proposing DA returns the stable matching every student weakly
    prefers to every other stable matching, so it maximises total student
    utility over the stable set -- exactly what the MIP is supposed to report.
    Written as an explicit propose/reject loop with no heaps and no shared code.
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
                options.append((program_id, student.utilities[rank]))
        lists.append(options)

    held: dict[str, list[int]] = {program_id: [] for program_id in capacity}
    matched: dict[int, float] = {}
    next_offer = [0] * len(lists)
    free = [index for index, options in enumerate(lists) if options]
    while free:
        student_index = free.pop()
        while next_offer[student_index] < len(lists[student_index]):
            program_id, utility = lists[student_index][next_offer[student_index]]
            next_offer[student_index] += 1
            if capacity[program_id] <= 0:
                continue
            roster = held[program_id]
            if len(roster) < capacity[program_id]:
                roster.append(student_index)
                matched[student_index] = utility
                break
            worst = max(roster, key=lambda other: rank_of[(other, program_id)])
            if rank_of[(student_index, program_id)] < rank_of[(worst, program_id)]:
                roster.remove(worst)
                roster.append(student_index)
                matched[student_index] = utility
                del matched[worst]
                free.append(worst)
                break
    return sum(matched.values())


# ---------------------------------------------------------------------- #
# Model harness: the matching block over a pinned zoning
# ---------------------------------------------------------------------- #
def _solve_fixed(
    problem,
    instance,
    zoning,
    *,
    non_wastefulness=True,
    aggregate_stability=True,
    relax=False,
) -> float:
    """Optimum of the matching block with ``x`` pinned to ``zoning``.

    Builds only the matching block, so the zoning block's contiguity, FRL and
    school-count rows cannot make an enumerated zoning infeasible and silently
    turn the comparison into a comparison of nothing.
    """

    solver = StableCutoffMipSolver(
        instance.market,
        instance.samples,
        non_wastefulness=non_wastefulness,
        aggregate_stability=aggregate_stability,
    )
    solver.relax_matching = relax
    with gp.Env(params={"OutputFlag": 0}) as env:
        with gp.Model("stable_cutoff_test", env=env) as model:
            x = {}
            for node in problem.nodes:
                for zone in problem.candidate_zones(node):
                    value = 1.0 if zoning[node] == zone else 0.0
                    x[(zone, node)] = model.addVar(
                        lb=value,
                        ub=value,
                        vtype=GRB.CONTINUOUS if relax else GRB.BINARY,
                        name=f"x_{zone}_{node}",
                    )
            model.update()
            solver._add_model_objective(model, problem, x)
            model.optimize()
            assert model.Status == GRB.OPTIMAL
            return float(model.ObjVal)


# ---------------------------------------------------------------------- #
# GATE: exactness
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "non_wastefulness,aggregate_stability",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_integral_optimum_is_deferred_acceptance_welfare(
    non_wastefulness, aggregate_stability
):
    problem = _problem()
    instance = _instance(problem)
    sample = instance.samples[0]

    for zoning in _zonings(problem):
        truth = _deferred_acceptance(problem, instance.market, sample, zoning)
        optimum = _solve_fixed(
            problem,
            instance,
            zoning,
            non_wastefulness=non_wastefulness,
            aggregate_stability=aggregate_stability,
        )
        assert optimum == pytest.approx(truth, abs=1e-6), zoning


def test_exactness_holds_across_lottery_draws():
    """Not an artefact of one draw: re-sample and re-check."""

    problem = _problem()
    for base_seed in (1, 2, 3):
        instance = _instance(problem, base_seed=base_seed)
        zoning = _zonings(problem, count=3)[1]
        truth = _deferred_acceptance(
            problem, instance.market, instance.samples[0], zoning
        )
        assert _solve_fixed(problem, instance, zoning) == pytest.approx(truth, abs=1e-6)


def test_ground_truth_is_sensitive_to_the_zoning():
    """Guard the gate itself: a constant truth would make it vacuous."""

    problem = _problem()
    instance = _instance(problem)
    welfares = {
        _deferred_acceptance(problem, instance.market, instance.samples[0], zoning)
        for zoning in _zonings(problem)
    }
    assert len(welfares) > 1


# ---------------------------------------------------------------------- #
# GATE: validity of the relaxation
# ---------------------------------------------------------------------- #
def test_relaxation_never_falls_below_deferred_acceptance():
    problem = _problem()
    instance = _instance(problem)
    sample = instance.samples[0]

    for zoning in _zonings(problem):
        truth = _deferred_acceptance(problem, instance.market, sample, zoning)
        relaxed = _solve_fixed(problem, instance, zoning, relax=True)
        assert relaxed >= truth - 1e-6, zoning


def test_non_wastefulness_changes_no_relaxation_value():
    """(S1) is a presolve aid, not a cut: it moves no LP value anywhere."""

    problem = _problem()
    instance = _instance(problem)
    for zoning in _zonings(problem):
        with_s1 = _solve_fixed(
            problem, instance, zoning, non_wastefulness=True, relax=True
        )
        without_s1 = _solve_fixed(
            problem, instance, zoning, non_wastefulness=False, relax=True
        )
        assert with_s1 == pytest.approx(without_s1, abs=1e-6), zoning


def test_aggregate_stability_can_only_tighten_the_relaxation():
    problem = _problem()
    instance = _instance(problem)
    for zoning in _zonings(problem):
        with_s2 = _solve_fixed(
            problem, instance, zoning, aggregate_stability=True, relax=True
        )
        without_s2 = _solve_fixed(
            problem, instance, zoning, aggregate_stability=False, relax=True
        )
        assert with_s2 <= without_s2 + 1e-6, zoning


# ---------------------------------------------------------------------- #
# The sample average
# ---------------------------------------------------------------------- #
def test_multi_sample_objective_is_the_mean_of_the_per_sample_optima():
    problem = _problem()
    joint = _instance(problem, num_seeds=3, base_seed=11)
    zoning = _zonings(problem, count=4)[2]

    combined = _solve_fixed(problem, joint, zoning)
    separate = [
        _solve_fixed(
            problem,
            SimpleNamespace(market=joint.market, samples=(sample,)),
            zoning,
        )
        for sample in joint.samples
    ]

    assert combined == pytest.approx(sum(separate) / len(separate), abs=1e-6)
    # And each block is still exact on its own draw.
    for sample, value in zip(joint.samples, separate):
        assert value == pytest.approx(
            _deferred_acceptance(problem, joint.market, sample, zoning), abs=1e-6
        )


def test_samples_are_distinct_draws():
    problem = _problem()
    instance = _instance(problem, num_seeds=3, base_seed=11)
    assert len(set(instance.sample_seeds)) == 3
    assert instance.num_seeds == 3
    assert instance.gamma_size == sum(
        len(student.programs) for student in instance.market.students
    )


# ---------------------------------------------------------------------- #
# Market preparation
# ---------------------------------------------------------------------- #
def test_misaligned_priority_orders_are_rejected():
    problem = _problem()
    instance = _instance(problem)
    truncated = SaaSample(
        seed=instance.samples[0].seed,
        school_orders=tuple(
            order[:-1] if index == 0 else order
            for index, order in enumerate(instance.samples[0].school_orders)
        ),
    )

    with pytest.raises(ValueError, match="is not Gamma"):
        validate_sample_orders(instance.market, (truncated,))
    with pytest.raises(ValueError, match="align with market programs"):
        validate_sample_orders(instance.market, (SaaSample(seed=1, school_orders=()),))
    with pytest.raises(ValueError, match="at least one priority sample"):
        validate_sample_orders(instance.market, ())


def test_undecided_access_pairs_exclude_settled_pairs():
    problem = _problem()
    instance = _instance(problem)
    pairs = undecided_access_pairs(problem, instance.market)

    # The citywide program has no school node, and node 0 hosts program A, so
    # neither contributes a pair; the school nodes are 0 and 8.
    assert pairs
    assert all(low < high for low, high in pairs)
    assert all({low, high} & {0, 8} for low, high in pairs)


# ---------------------------------------------------------------------- #
# Solver and strategy wiring
# ---------------------------------------------------------------------- #
def test_solver_rejects_non_boolean_switches():
    problem = _problem()
    instance = _instance(problem)
    with pytest.raises(ValueError, match="non_wastefulness must be a Boolean"):
        StableCutoffMipSolver(instance.market, instance.samples, non_wastefulness="yes")
    with pytest.raises(ValueError, match="aggregate_stability must be a Boolean"):
        StableCutoffMipSolver(instance.market, instance.samples, aggregate_stability=1)


def test_strategy_requires_the_mip_solver():
    problem = _problem()
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population="All")
    strategy = get_strategy("stable_cutoff", levels=["BlockGroup_0"])

    with pytest.raises(ValueError, match="requires solver='mip'"):
        strategy.run(dataset, get_solver("cp_bool"))


def test_strategy_requires_all_program_population():
    problem = _problem()
    dataset = FakeDataset(problem)
    dataset.config = SimpleNamespace(program_population="GE")
    strategy = get_strategy("stable_cutoff", levels=["BlockGroup_0"])

    with pytest.raises(ValueError, match="program_population='All'"):
        strategy.run(dataset, get_solver("mip"))


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
    instance = _instance(problem, num_seeds=2, base_seed=7)
    monkeypatch.setattr(
        stable_cutoff_module,
        "build_stable_cutoff_instance",
        lambda *args, **kwargs: instance,
    )
    monkeypatch.setattr(
        stable_cutoff_module, "initial_solution", lambda *args, **kwargs: None
    )
    strategy = get_strategy(
        "stable_cutoff",
        levels=["BlockGroup_0"],
        solve_time_limits=[60],
        gap_limits=[0],
        hints="none",
    )

    solutions = strategy.run(dataset, get_solver("mip", solve_time_limit=60, workers=1))
    final = solutions[-1]
    truth = sum(
        _deferred_acceptance(problem, instance.market, sample, final.assignment)
        for sample in instance.samples
    ) / len(instance.samples)

    assert final.status == "OPTIMAL"
    assert final.objective == pytest.approx(truth, abs=1e-6)
    assert final.metadata["objective_kind"] == "stable_cutoff_sample_average_welfare"
    assert final.metadata["stable_cutoff_num_seeds"] == 2
    assert final.metadata["stable_cutoff_sample_seeds"] == list(instance.sample_seeds)
    assert final.metadata["stable_cutoff_gamma_size"] == instance.gamma_size
    assert final.metadata["stable_cutoff_y_vars"] == 2 * instance.gamma_size
    assert final.metadata["stable_cutoff_z_vars"] == 2 * instance.gamma_size
    assert final.metadata["stable_cutoff_non_wastefulness"] is True
    assert final.metadata["stable_cutoff_aggregate_stability"] is True
    assert final.metadata["stable_cutoff_access_pair_count"] > 0
    assert final.metadata["stable_cutoff_model_variable_count"] > 0
    assert final.metadata["stable_cutoff_model_constraint_count"] > 0
    assert final.metadata["stable_cutoff_relative_gap"] == pytest.approx(0.0, abs=1e-6)
    assert final.metadata["stable_cutoff_best_objective_bound"] == pytest.approx(
        truth, abs=1e-4
    )
    assert sum(final.metadata["stable_cutoff_sample_welfares"]) / 2 == pytest.approx(
        truth, abs=1e-6
    )
    assert final.metadata["stable_cutoff_preprocessing_seconds"] >= 0.0


# ---------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------- #
def test_config_passes_stable_cutoff_options_through():
    config = OptimizationConfig(
        levels=["BlockGroup_0"],
        solver="mip",
        strategy="stable_cutoff",
        stable_cutoff_num_seeds=4,
        stable_cutoff_non_wastefulness=False,
        stable_cutoff_aggregate_stability=False,
        stable_cutoff_tie_breaking_method="mtb",
    )
    options = config.make_strategy().options

    assert options["stable_cutoff_num_seeds"] == 4
    assert options["stable_cutoff_non_wastefulness"] is False
    assert options["stable_cutoff_aggregate_stability"] is False
    assert options["stable_cutoff_tie_breaking_method"] == "MTB"


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"stable_cutoff_num_seeds": 0}, "positive integer"),
        ({"stable_cutoff_num_seeds": 2.5}, "positive integer"),
        ({"stable_cutoff_num_seeds": True}, "positive integer"),
        ({"stable_cutoff_non_wastefulness": "yes"}, "must be a Boolean"),
        ({"stable_cutoff_aggregate_stability": 1}, "must be a Boolean"),
        ({"stable_cutoff_tie_breaking_method": "coin"}, "MTB, STB"),
        ({"stable_cutoff_tie_breaking_method": 3}, "MTB, STB"),
    ],
)
def test_config_rejects_bad_stable_cutoff_options(overrides, message):
    with pytest.raises(ValueError, match=message):
        OptimizationConfig(
            levels=["BlockGroup_0"], strategy="stable_cutoff", **overrides
        )


def test_dual_bound_is_reported_without_an_incumbent():
    """A run that times out before its first zoning must still say what it proved.

    The certified bound is this formulation's reason to exist, and on the real
    instance without a feasible hint the normal timeout shape is a live dual
    bound and no incumbent at all. Reporting the bound only when
    ``SolCount > 0`` discards precisely the number the run was for.
    """

    problem = _problem()
    instance = _instance(problem)
    solver = StableCutoffMipSolver(instance.market, instance.samples)
    with gp.Env(params={"OutputFlag": 0}) as env:
        with gp.Model("stable_cutoff_bound", env=env) as model:
            x = {}
            for node in problem.nodes:
                for zone in problem.candidate_zones(node):
                    x[(zone, node)] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_{zone}_{node}"
                    )
            model.update()
            solver._add_model_objective(model, problem, x)

            # A live dual bound, no incumbent: the shape of a timed-out run.
            no_incumbent = SimpleNamespace(
                SolCount=0,
                IsMIP=1,
                ObjBound=1234.5,
                NumVars=model.NumVars,
                NumConstrs=model.NumConstrs,
                NumNZs=model.NumNZs,
            )
            metadata = solver._additional_solution_metadata(no_incumbent, "UNKNOWN")

    assert metadata["stable_cutoff_best_objective_bound"] == pytest.approx(1234.5)
    # No incumbent means no gap and no welfare to report.
    assert "stable_cutoff_relative_gap" not in metadata
    assert "stable_cutoff_welfare" not in metadata


def test_infinite_dual_bound_is_omitted_rather_than_reported():
    """Before the root relaxation closes Gurobi's ObjBound is infinite."""

    problem = _problem()
    instance = _instance(problem)
    solver = StableCutoffMipSolver(instance.market, instance.samples)
    with gp.Env(params={"OutputFlag": 0}) as env:
        with gp.Model("stable_cutoff_inf", env=env) as model:
            x = {}
            for node in problem.nodes:
                for zone in problem.candidate_zones(node):
                    x[(zone, node)] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_{zone}_{node}"
                    )
            model.update()
            solver._add_model_objective(model, problem, x)
            unstarted = SimpleNamespace(
                SolCount=0,
                IsMIP=1,
                ObjBound=float("inf"),
                NumVars=model.NumVars,
                NumConstrs=model.NumConstrs,
                NumNZs=model.NumNZs,
            )
            metadata = solver._additional_solution_metadata(unstarted, "UNKNOWN")

    assert "stable_cutoff_best_objective_bound" not in metadata

"""Reference capacity counts preserve MID usage without changing demographics."""

import json
from dataclasses import replace

import networkx as nx
import pytest

from metrics.base import MetricsContext
from metrics.diversity import compute
from optimization.data.capacity_reference import (
    apply_capacity_reference,
    build_capacity_reference,
)
from optimization.data.feasibility import check_zoning
from optimization.data.initial_solutions import feasibility_fingerprint
from optimization.data.mid import MidMarket, MidProgram, MidType
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers import get_solver
from optimization.solvers.balance import balance_constraints
from optimization.tests.synthetic import make_grid_problem

ZONING = {0: 0, 1: 0, 2: 1, 3: 1}


def _market():
    return MidMarket(
        programs=(
            MidProgram("100-GE-KG", 100, 10, False, 0, 10),
            MidProgram("200-GE-KG", 200, 10, False, 3, 10),
            MidProgram("900-GE-KG", 900, 1, True, None, 0),
        ),
        types=tuple(
            MidType(n, 1, ("900-GE-KG", "100-GE-KG"), (0, 0), (2.0, 1.0), (2, 1))
            if n < 2
            else MidType(n, 1, (), (), (), ())
            for n in range(4)
        ),
        student_count=4,
        outside_only_student_count=2,
        utility_student_count=2,
        utility_handling="omit_nonpositive",
    )


def _problem():
    problem = make_grid_problem(2, 2, program_population="All", overage=0, shortage=0)
    # Physical capacity is far above expected usage.
    for n in problem.nodes:
        problem.G.nodes[n]["all_prog_capacity"] = 100 if n in (0, 3) else 0
    return problem


def test_fractional_citywide_departures_and_designated_seats():
    problem = _problem()
    reference = build_capacity_reference(problem, ZONING, _market(), 20)
    assert reference["students_by_area"] == {1000: 0.5, 1001: 0.5, 1002: 1, 1003: 1}
    assert reference["seats_by_school"] == {100: 1, 200: 2}
    assert reference["non_aa_students"] == 1
    assert reference["designated_students"] == 2
    assert reference["unassigned_students"] == 0


@pytest.mark.parametrize("school_capacity", [0, 1, 2])
def test_only_assigned_students_count_and_source_zones_balance(school_capacity):
    market = _market()
    market = replace(
        market,
        programs=tuple(
            replace(
                p, capacity=school_capacity, attendance_area_capacity=school_capacity
            )
            if p.school_id == 200
            else p
            for p in market.programs
        ),
    )
    problem = _problem()
    reference = build_capacity_reference(problem, ZONING, market, 20)
    assert reference["unassigned_students"] == 2 - school_capacity
    assert reference["students_by_area"][1002] == school_capacity / 2
    assert reference["students_by_area"][1003] == school_capacity / 2
    assert reference["seats_by_school"].get(200, 0) == school_capacity
    apply_capacity_reference(problem, reference)
    for zone in range(problem.Z):
        nodes = [n for n in problem.nodes if ZONING[n] == zone]
        assert sum(problem.capacity_students(n) for n in nodes) == pytest.approx(
            sum(problem.capacity_seats(n) for n in nodes)
        )
    # Even zero tolerance must accept the source, including unassigned cases.
    assert problem.overage == problem.shortage == 0
    assert check_zoning(problem, ZONING).feasible


def test_mixed_program_seats_are_attributed_proportionally():
    market = _market()
    market = replace(
        market,
        programs=tuple(
            replace(p, attendance_area_capacity=5) if p.school_id == 100 else p
            for p in market.programs
        ),
    )
    reference = build_capacity_reference(_problem(), ZONING, market, 20)
    assert reference["students_by_area"][1000] == 0.25
    assert reference["students_by_area"][1001] == 0.25
    assert reference["seats_by_school"][100] == 0.5
    assert reference["non_aa_students"] == 1.5


@pytest.mark.parametrize("method", ["cp_bool", "cp_int"])
def test_capacity_only_overlay_reaches_solvers_validator_and_cache(method):
    problem = _problem()
    original = replace(problem)
    fingerprint = feasibility_fingerprint(problem)
    assert not check_zoning(problem, ZONING).feasible
    before = compute(MetricsContext(ZoneSolution(problem, ZONING, "FEASIBLE")))
    market = _market()
    market = replace(
        market,
        programs=tuple(
            replace(p, capacity=1, attendance_area_capacity=1)
            if p.school_id == 200
            else p
            for p in market.programs
        ),
    )
    reference = build_capacity_reference(problem, ZONING, market, 20)
    assert reference["unassigned_students"] == 1
    apply_capacity_reference(problem, reference)
    assert check_zoning(problem, ZONING).feasible
    assert not check_zoning(original, ZONING).feasible
    assert feasibility_fingerprint(problem) != fingerprint
    assert problem.capacity(0) == 100
    assert problem.students(0) == 1
    assert (
        next(c for c in balance_constraints(problem) if c.kind == "frl").students(
            problem, 0
        )
        == 1
    )
    after = compute(MetricsContext(ZoneSolution(problem, ZONING, "FEASIBLE")))
    assert before == after
    solution = get_solver(method, workers=1, solve_time_limit=5).solve(problem)
    assert solution.feasible
    assert check_zoning(problem, solution.assignment).feasible
    # Seat counts independently affect the feasible-set fingerprint.
    seats_fingerprint = feasibility_fingerprint(problem)
    problem.capacity_seat_counts[0] += 1
    assert feasibility_fingerprint(problem) != seats_fingerprint


def test_portable_counts_survive_json_node_renumbering_and_aggregation():
    problem = _problem()
    reference = json.loads(
        json.dumps(build_capacity_reference(problem, ZONING, _market(), 20))
    )
    renumbered = replace(
        problem,
        G=nx.relabel_nodes(problem.G, {n: 10 + n for n in problem.G}),
        centroids=[10, 13],
    )
    apply_capacity_reference(renumbered, reference)
    assert renumbered.capacity_students(10) == 0.5
    assert renumbered.capacity_seats(13) == 2
    graph = nx.Graph()
    graph.add_node(7, block_ids=[1000, 1001], school_ids=[100])
    graph.add_node(8, block_ids=[1002, 1003], school_ids=[200])
    aggregated = replace(
        problem, G=graph, level=LevelSpec("BlockGroup", 1), centroids=[7, 8]
    )
    apply_capacity_reference(aggregated, reference)
    assert aggregated.capacity_students(7) == 1
    assert aggregated.capacity_students(8) == 2
    assert aggregated.capacity_seats(7) == 1
    assert aggregated.capacity_seats(8) == 2


def test_mismatched_market_fails_instead_of_dropping_students():
    problem = _problem()
    problem.G.nodes[0]["all_prog_students"] = 2
    with pytest.raises(ValueError, match="populations must agree"):
        build_capacity_reference(problem, ZONING, _market(), 20)


def test_old_reference_semantics_are_rejected():
    problem = _problem()
    reference = build_capacity_reference(problem, ZONING, _market(), 20)
    reference["schema_version"] = 1
    with pytest.raises(ValueError, match="schema/geography"):
        apply_capacity_reference(problem, reference)

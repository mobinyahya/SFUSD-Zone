"""School-level FRL dissimilarity read off a MID match."""

from __future__ import annotations

import pytest

from optimization.data.mid import (
    MidMarket,
    MidProgram,
    MidStudent,
    MidType,
    compress_mid_students,
)
from optimization.objectives import WEIGHTED_SUM_TERMS, checked_objective_weights
from optimization.solvers import burst_objectives
from optimization.solvers.burst_objectives import designation_seats, mid_outcome

# Nodes 0 1 / 2 3; school 100 sits at node 0, school 200 at node 3.
SPLIT_A = {0: 0, 1: 0, 2: 1, 3: 1}
LOTTERY_SCALE = 20


def _market(*types: MidType, citywide_capacity: int = 0) -> MidMarket:
    programs = [
        MidProgram("100-GE-KG", 100, 4, False, 0),
        MidProgram("200-GE-KG", 200, 4, False, 3),
    ]
    if citywide_capacity:
        programs.append(MidProgram("300-GE-KG", 300, citywide_capacity, True, None))
    return MidMarket(
        programs=tuple(programs),
        types=types,
        student_count=sum(t.count for t in types),
        outside_only_student_count=0,
        utility_student_count=sum(t.count for t in types),
        utility_handling="omit_nonpositive",
    )


def _dissimilarity(market: MidMarket, zoning=SPLIT_A) -> float:
    outcome = mid_outcome(market, designation_seats(market), zoning, LOTTERY_SCALE)
    return outcome.frl_dissimilarity


def test_fully_sorted_schools_score_100() -> None:
    market = _market(
        MidType(0, 4, ("100-GE-KG",), (0,), (2.0,), (200,), frl_sum=4.0),
        MidType(3, 4, ("200-GE-KG",), (0,), (2.0,), (200,), frl_sum=0.0),
    )
    assert _dissimilarity(market) == pytest.approx(100.0)


def test_one_shared_school_scores_0() -> None:
    market = _market(
        MidType(0, 4, ("300-GE-KG",), (0,), (2.0,), (200,), frl_sum=4.0),
        MidType(3, 4, ("300-GE-KG",), (0,), (2.0,), (200,), frl_sum=0.0),
        citywide_capacity=8,
    )
    assert _dissimilarity(market) == pytest.approx(0.0)


def test_designated_students_join_their_zones_free_seats() -> None:
    # Node 1 lies in zone 0 but ranks only school 200, which zone 0 cannot
    # reach, so both its students -- one FRL in expectation -- are designated
    # to the two GE seats school 100 still has free.
    market = _market(
        MidType(0, 2, ("100-GE-KG",), (0,), (2.0,), (200,), frl_sum=2.0),
        MidType(3, 2, ("200-GE-KG",), (0,), (2.0,), (200,), frl_sum=0.0),
        MidType(1, 2, ("200-GE-KG",), (0,), (2.0,), (200,), frl_sum=1.0),
    )
    # School 100: 4 students, 3 FRL; school 200: 2 students, 0 FRL.
    expected = 50.0 * (abs(3 / 3 - 1 / 3) + abs(0 / 3 - 2 / 3))
    assert _dissimilarity(market) == pytest.approx(expected)


def test_no_frl_variation_scores_0() -> None:
    market = _market(
        MidType(0, 4, ("100-GE-KG",), (0,), (2.0,), (200,), frl_sum=0.0),
        MidType(3, 4, ("200-GE-KG",), (0,), (2.0,), (200,), frl_sum=0.0),
    )
    assert _dissimilarity(market) == 0.0


def test_compression_sums_frl_within_a_type() -> None:
    students = [
        MidStudent(0, ("A",), (0,), (1.0,), (100,), frl=0.25),
        MidStudent(0, ("A",), (0,), (1.0,), (100,), frl=0.5),
        MidStudent(1, ("A",), (0,), (1.0,), (100,), frl=1.0),
    ]
    types = {t.node: t for t in compress_mid_students(students)}
    assert types[0].count == 2
    assert types[0].frl_sum == pytest.approx(0.75)
    assert types[1].frl_sum == pytest.approx(1.0)


def test_frl_dissimilarity_is_a_weighted_sum_term() -> None:
    assert "frl_dissimilarity" in WEIGHTED_SUM_TERMS
    assert "frl_dissimilarity" in burst_objectives.CHOICE_METRICS
    assert checked_objective_weights({"capacity_match": 1, "frl_dissimilarity": 2}) == {
        "capacity_match": 1.0,
        "frl_dissimilarity": 2.0,
    }

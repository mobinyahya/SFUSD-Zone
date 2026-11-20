"""Utility-model MID markets: per-lottery lists, the eligible-list cap and the
duplicates the simulator drops, through the lottery DA and preprocessing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from optimization.data.mid import (
    APPENDED,
    DESIGNATED,
    LISTED,
    UNLIMITED_BUDGET,
    LotteryLists,
    LotterySpec,
    MidMarket,
    MidProgram,
    MidType,
    preprocess_mid_market,
)
from optimization.lottery_da import lottery_outcome, lottery_placements

#: Zoned A at node 1, citywide B, zoned C at node 0 (the student's own).
PROGRAMS = (
    MidProgram("10-GE-KG", 10, 1, False, 1, 1),
    MidProgram("20-GE-KG", 20, 1, True, None, 0),
    MidProgram("30-GE-KG", 30, 1, False, 0, 1),
)
A, B, C = (p.program_id for p in PROGRAMS)


def _market(rows_by_lottery, *, budget, programs=PROGRAMS, add_aa=False):
    """``rows_by_lottery[l][i]``: student ``i``'s entries
    ``(program, tier, open tier, kind, copy)`` in lottery ``l``."""
    program_ids = tuple(sorted(p.program_id for p in programs))
    column = {pid: j for j, pid in enumerate(program_ids)}
    students = len(rows_by_lottery[0])
    flat = [entry for rows in rows_by_lottery for row in rows for entry in row]
    segments = np.zeros(len(rows_by_lottery) * students + 1, dtype=np.int64)
    np.cumsum(
        [len(row) for rows in rows_by_lottery for row in rows], out=segments[1:]
    )
    lists = LotteryLists(
        student_ids=tuple(str(i) for i in range(students)),
        segments=segments,
        program=np.array([column[e[0]] for e in flat], dtype=np.int32),
        tier=np.array([e[1] for e in flat], dtype=np.int64),
        open_tier=np.array([e[2] for e in flat], dtype=np.int64),
        kind=np.array([e[3] for e in flat], dtype=np.int8),
        copy=np.array([e[4] for e in flat], dtype=np.int32),
        utility=np.arange(len(flat), dtype=np.float64),
        draw=np.full(len(flat), 0.5),
        budget=np.full(students, budget, dtype=np.int64),
    )
    types = tuple(
        MidType(
            node=0,
            count=1,
            programs=(),
            priorities=(),
            utility_sums=(),
            scaled_utility_sums=(),
            open_priorities=(),
            student_id=str(i),
            reserve_class=0,
            kinds=(),
        )
        for i in range(students)
    )
    return MidMarket(
        programs=programs,
        types=types,
        student_count=students,
        outside_only_student_count=0,
        utility_student_count=students,
        utility_handling="omit_nonpositive",
        policy="test",
        lottery=LotterySpec(
            fractions={},
            num_classes=1,
            strict=False,
            seed=0,
            lotteries=len(rows_by_lottery),
            add_aa_schools=add_aa,
            student_ids=tuple(sorted(lists.student_ids)),
            program_ids=program_ids,
            lists=lists,
            random_seed=2023,
        ),
    )


def _placed(market, zoning):
    program, kind, _ = lottery_placements(market, zoning)
    ids = [p.program_id for p in market.programs]
    return [
        [(ids[p], int(k)) if p >= 0 else None for p, k in zip(row, kinds)]
        for row, kinds in zip(program, kind)
    ]


# One student: drawn list A, B, C; C also designated. A is out of zone and
# closed, so it does not count towards the cap; B (citywide) does.
ROW = [
    (A, 5, -1, LISTED, -1),
    (B, 5, 5, LISTED, -1),
    (C, 5, -1, LISTED, -1),
    (C, 1, -1, DESIGNATED, 2),
]
ZONING = {0: 0, 1: 1}


def test_the_cap_counts_only_programs_the_zoning_makes_eligible():
    # Cap 1: B is the one eligible listed program, so C is off the list and
    # its designation stands.
    full = _market([[ROW]], budget=1)
    assert _placed(full, ZONING) == [[(B, LISTED)]]
    # With A in the student's zone it is eligible, takes the one place, and
    # B is off the list.
    assert _placed(full, {0: 0, 1: 0}) == [[(A, LISTED)]]


def test_a_full_listed_program_still_takes_its_place_on_the_list():
    no_seat = (PROGRAMS[0], MidProgram(B, 20, 0, True, None, 0), PROGRAMS[2])
    market = _market([[ROW]], budget=1, programs=no_seat)
    # B has no seat but is on the capped list; C is reached as a designation.
    assert _placed(market, ZONING) == [[(C, DESIGNATED)]]
    # Preprocessing keeps the seatless listed B: dropping it would hand its
    # place to C.
    problem = SimpleNamespace(candidate_zones=lambda node: {0, 1})
    preprocessed = preprocess_mid_market(market, problem)
    assert _placed(preprocessed, ZONING) == [[(C, DESIGNATED)]]
    assert B in {p.program_id for p in preprocessed.programs}


def test_a_designation_whose_program_is_on_the_list_is_the_dropped_duplicate():
    # Uncapped, C is listed, so its designation is never tried.
    no_seat_c = (PROGRAMS[0], PROGRAMS[1], MidProgram(C, 30, 0, False, 0, 0))
    row = [(C, 5, -1, LISTED, -1), (C, 1, -1, DESIGNATED, 0)]
    market = _market([[row]], budget=UNLIMITED_BUDGET, programs=no_seat_c)
    assert _placed(market, ZONING) == [[None]]


def test_each_lottery_matches_its_own_lists():
    first = [(B, 5, 5, LISTED, -1)]
    second = [(C, 5, -1, LISTED, -1)]
    market = _market([[first], [second]], budget=UNLIMITED_BUDGET)
    assert _placed(market, ZONING) == [[(B, LISTED)], [(C, LISTED)]]
    outcome = lottery_outcome(market, ZONING)
    # Welfare averages the drawn utility of each lottery's placement.
    assert outcome.welfare == pytest.approx((0.0 + 1.0) / 2)
    with pytest.raises(ValueError, match="draws its own lotteries"):
        lottery_placements(market, ZONING, draws=np.zeros((1, 2)))


def test_preprocessing_drops_unreachable_entries_and_remaps_copies():
    # A zoned program no zoning puts with the student goes; the designation's
    # copy still points at its listed program after the shift.
    row = [
        (A, 5, -1, LISTED, -1),
        (C, 5, -1, LISTED, -1),
        (C, 1, -1, DESIGNATED, 1),
        (A, 1, -1, APPENDED, 0),
    ]
    market = _market([[row]], budget=UNLIMITED_BUDGET)
    problem = SimpleNamespace(candidate_zones=lambda node: {node})
    preprocessed = preprocess_mid_market(market, problem)
    lists = preprocessed.lottery.lists
    kept = [preprocessed.lottery.program_ids[j] for j in lists.program]
    assert kept == [C, C]
    assert lists.copy.tolist() == [-1, 0]
    assert _placed(preprocessed, ZONING) == _placed(market, ZONING) == [[(C, LISTED)]]

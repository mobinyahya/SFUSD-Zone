"""Policy-aware MID markets: out-of-zone access tiers and the reserve DA."""

from __future__ import annotations

import numpy as np
import pytest

from assignment.student_assignment.da.da_with_guardrails import DAwithGuards
from optimization.data.mid import (
    APPENDED,
    DESIGNATED,
    LISTED,
    LotterySpec,
    MidMarket,
    MidProgram,
    MidType,
    require_label_access,
)
from optimization.lottery_da import _deferred_acceptance, lottery_outcome
from optimization.mid_oracle import finite_grid_oracle


def _market(open_priorities, *, priorities=(0,)) -> MidMarket:
    """One zoned program at node 1, wanted by one student living at node 0."""
    return MidMarket(
        programs=(MidProgram("10-GE-KG", 10, 1, False, 1, 1),),
        types=(
            MidType(
                node=0,
                count=1,
                programs=("10-GE-KG",),
                priorities=priorities,
                utility_sums=(1.0,),
                scaled_utility_sums=(100,),
                open_priorities=open_priorities,
            ),
        ),
        student_count=1,
        outside_only_student_count=0,
        utility_student_count=1,
        utility_handling="omit_nonpositive",
        policy="test",
    )


@pytest.mark.parametrize(
    ("open_priorities", "zoning", "matched"),
    [
        (None, {0: 0, 1: 0}, True),  # in zone
        (None, {0: 0, 1: 1}, False),  # out of zone, no policy
        ((-1,), {0: 0, 1: 1}, False),  # out of zone, no exception
        ((0,), {0: 0, 1: 1}, True),  # out of zone, sibling access
    ],
)
def test_open_tiers_give_out_of_zone_access(open_priorities, zoning, matched):
    result = finite_grid_oracle(_market(open_priorities), zoning, 4)
    assert result.demands["10-GE-KG"] == (1.0 if matched else 0.0)


def test_in_zone_tier_outranks_the_open_tier():
    # Two students, one seat: the in-zone one (tier 1) beats the sibling
    # (open tier 0) though both are eligible.
    market = MidMarket(
        programs=(MidProgram("10-GE-KG", 10, 1, False, 1, 1),),
        types=(
            MidType(0, 1, ("10-GE-KG",), (1,), (1.0,), (100,), open_priorities=(0,)),
            MidType(1, 1, ("10-GE-KG",), (1,), (1.0,), (100,), open_priorities=(0,)),
        ),
        student_count=2,
        outside_only_student_count=0,
        utility_student_count=2,
        utility_handling="omit_nonpositive",
        policy="test",
    )
    result = finite_grid_oracle(market, {0: 0, 1: 1}, 4)
    assert result.assignment_masses == ((0,), (4,))


def test_label_access_models_reject_policy_markets():
    require_label_access(_market(None), "a model")
    with pytest.raises(ValueError, match="does not model matching policy"):
        require_label_access(_market((0,)), "a model")


def test_reserve_da_matches_the_assignment_guardrail_da():
    for trial in range(200):
        rng = np.random.default_rng(trial)
        students = int(rng.integers(5, 80))
        programs = int(rng.integers(1, 10))
        classes = int(rng.integers(2, 4))
        capacity = rng.integers(0, 10, size=programs)
        strict = bool(rng.random() < 0.3)
        fractions = rng.dirichlet(np.ones(classes), size=programs) * rng.uniform(
            0.3, 1.0, size=(programs, 1)
        )
        fractions[rng.random(programs) < 0.3] = 0.0
        student_class = rng.integers(0, classes, size=students)
        tiers = rng.integers(0, 4, size=(students, programs)).astype(float)
        draws = rng.random((students, programs))
        access = rng.random((students, programs)) < 0.85
        priorities = np.where(access, tiers + draws, -1.0)
        prefs = [
            list(rng.permutation(programs)[: int(rng.integers(1, programs + 1))] + 1)
            for _ in range(students)
        ]
        reference = DAwithGuards(
            capacity, priorities, prefs, student_class, int(strict)
        )
        reference.setguards(fractions, classes)
        expected, _ = reference.run()

        start = np.zeros(students + 1, dtype=np.int64)
        start[1:] = np.cumsum([len(listed) for listed in prefs])
        flat = [(s, p - 1) for s, listed in enumerate(prefs) for p in listed]
        ranks = _deferred_acceptance(
            start,
            np.array([p for _, p in flat], dtype=np.int64),
            np.array([tiers[s, p] if access[s, p] else -1.0 for s, p in flat]),
            np.full(len(flat), -1.0),
            np.array([draws[s, p] for s, p in flat]),
            np.zeros(students, dtype=np.int64),
            student_class.astype(np.int64),
            capacity.astype(np.int64),
            np.ones(programs, dtype=np.bool_),
            np.full(programs, -1, dtype=np.int64),
            np.array(
                [
                    [int(capacity[p] * fractions[p, c]) for c in range(classes)]
                    for p in range(programs)
                ],
                dtype=np.int64,
            ),
            (fractions > 0).any(axis=1),
            strict,
            np.zeros(1, dtype=np.int64),
            np.zeros(len(flat), dtype=np.int64),
            False,
        )
        matched = [prefs[s][r] if r >= 0 else 0 for s, r in enumerate(ranks)]
        assert matched == expected.tolist(), trial


def _lottery_market(types, programs, *, fractions=None, add_aa=False, lotteries=1):
    return MidMarket(
        programs=programs,
        types=types,
        student_count=len(types),
        outside_only_student_count=sum(not t.programs for t in types),
        utility_student_count=len(types),
        utility_handling="omit_nonpositive",
        policy="test",
        lottery=LotterySpec(
            fractions=fractions or {},
            num_classes=2,
            strict=False,
            seed=0,
            lotteries=lotteries,
            add_aa_schools=add_aa,
            student_ids=tuple(t.student_id for t in types),
            program_ids=tuple(sorted(p.program_id for p in programs)),
        ),
    )


def _student(i, programs, kinds, *, node=0, open_priorities=None, tiers=None):
    return MidType(
        node=node,
        count=1,
        programs=programs,
        priorities=tiers or (0,) * len(programs),
        utility_sums=tuple(1.0 / (k + 1) for k in range(len(programs))),
        scaled_utility_sums=tuple(100 // (k + 1) for k in range(len(programs))),
        open_priorities=open_priorities or (-1,) * len(programs),
        student_id=str(i),
        reserve_class=0,
        kinds=kinds,
    )


def test_unused_reserved_seats_stay_open_and_designation_happens_in_the_match():
    # Two GE seats reserved one per class and three class-0 students: one
    # takes the class-0 seat, the empty class-1 seat stays open to a second,
    # and the third is designated to the citywide program.
    programs = (
        MidProgram("10-GE-KG", 10, 2, False, 0, 2),
        MidProgram("20-GE-KG", 20, 1, True, None, 0),
    )
    types = tuple(
        _student(i, ("10-GE-KG", "20-GE-KG"), (LISTED, DESIGNATED), tiers=(1, 0))
        for i in range(3)
    )
    market = _lottery_market(
        types, programs, fractions={"10-GE-KG": (0.5, 0.5)}, lotteries=3
    )
    outcome = lottery_outcome(market, {0: 0})
    assert (outcome.unassigned, outcome.designated) == (0.0, 1.0)
    with pytest.raises(ValueError, match="lottery_da"):
        finite_grid_oracle(market, {0: 0}, 20)


def test_appended_programs_only_in_zone_and_no_designated_duplicate():
    # A zoned GE program at node 1, appended for and designated to a student
    # at node 0 whose sibling opens it.
    programs = (MidProgram("10-GE-KG", 10, 1, False, 1, 1),)
    student = _student(
        0,
        ("10-GE-KG", "10-GE-KG"),
        (APPENDED, DESIGNATED),
        tiers=(5, 0),
        open_priorities=(-1, 0),
    )
    market = _lottery_market((student,), programs, add_aa=True)
    in_zone = lottery_outcome(market, {0: 0, 1: 0})
    assert (in_zone.unassigned, in_zone.designated) == (0.0, 0.0)  # appended
    out_of_zone = lottery_outcome(market, {0: 0, 1: 1})
    assert (out_of_zone.unassigned, out_of_zone.designated) == (0.0, 1.0)  # sibling
    closed = _lottery_market(
        (
            _student(
                0,
                ("10-GE-KG", "10-GE-KG"),
                (APPENDED, DESIGNATED),
                tiers=(5, 0),
            ),
        ),
        programs,
        add_aa=True,
    )
    assert lottery_outcome(closed, {0: 0, 1: 1}).unassigned == 1.0


def test_matching_policy_must_name_a_policy_and_a_recom_step():
    from optimization.config import OptimizationConfig

    with pytest.raises(ValueError, match="names no policy config"):
        OptimizationConfig(matching_policy="no_such_policy")
    with pytest.raises(ValueError, match="ReCom MID scorers only"):
        OptimizationConfig(
            matching_policy="restrictive_aa_sib_ctip",
            data={"scenario": "attendance-areas-2627", "overrides": {}},
            strategy="pipeline",
            steps=[
                {
                    "name": "welfare",
                    "level": "BlockGroup_0",
                    "method": "cp_bool",
                    "objective": "matching_welfare",
                }
            ],
        )


def test_an_unset_matching_policy_keeps_task_ids():
    from benchmark.config import optimization_config_hash
    from optimization.config import OptimizationConfig

    manifest = {"sources": "fixed"}
    plain = optimization_config_hash(OptimizationConfig(), source_manifest=manifest)
    seeded = optimization_config_hash(
        OptimizationConfig(mid_lottery_seed=3, mid_lotteries=2),
        source_manifest=manifest,
    )
    assert plain == seeded
    policy = optimization_config_hash(
        OptimizationConfig(matching_policy="restrictive_aa_sib_ctip"),
        source_manifest=manifest,
    )
    assert policy != plain

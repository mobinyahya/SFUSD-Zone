"""Policy-aware MID markets: out-of-zone access tiers and the reserve DA."""

from __future__ import annotations

import numpy as np
import pytest

from assignment.student_assignment.da.da_with_guardrails import DAwithGuards
from optimization.data.mid import (
    MidMarket,
    MidProgram,
    MidType,
    ReserveSpec,
    require_label_access,
)
from optimization.mid_oracle import finite_grid_oracle
from optimization.reserve_da import _deferred_acceptance, reserve_da_oracle


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
        )
        matched = [prefs[s][r] if r >= 0 else 0 for s, r in enumerate(ranks)]
        assert matched == expected.tolist(), trial


def test_reserve_market_result_reads_like_a_mid_match():
    # Two seats reserved one per class, three class-0 students: one takes the
    # class-0 seat, the class-1 seat stays open to the next class-0 student,
    # and the third is unmatched.
    types = tuple(
        MidType(
            0,
            1,
            ("10-GE-KG",),
            (0,),
            (1.0,),
            (100,),
            student_id=str(i),
            reserve_class=0,
        )
        for i in range(3)
    )
    market = MidMarket(
        programs=(MidProgram("10-GE-KG", 10, 2, False, 0, 2),),
        types=types,
        student_count=3,
        outside_only_student_count=0,
        utility_student_count=3,
        utility_handling="omit_nonpositive",
        policy="test",
        reserves=ReserveSpec({"10-GE-KG": (0.5, 0.5)}, 2, False, 0),
    )
    result = reserve_da_oracle(market, {0: 0}, 20)
    assert result.demands == {"10-GE-KG": 2.0}
    assert sorted(row[0] for row in result.assignment_masses) == [0, 20, 20]
    assert result.outside_mass == 1.0
    # finite_grid_oracle hands reserve markets to the DA.
    assert finite_grid_oracle(market, {0: 0}, 20).demands == result.demands

    strict = MidMarket(
        **{
            **market.__dict__,
            "reserves": ReserveSpec({"10-GE-KG": (0.5, 0.5)}, 2, True, 0),
        }
    )
    assert reserve_da_oracle(strict, {0: 0}, 20).demands == {"10-GE-KG": 1.0}


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
        OptimizationConfig(mid_reserve_seed=3), source_manifest=manifest
    )
    assert plain == seeded
    policy = optimization_config_hash(
        OptimizationConfig(matching_policy="restrictive_aa_sib_ctip"),
        source_manifest=manifest,
    )
    assert policy != plain

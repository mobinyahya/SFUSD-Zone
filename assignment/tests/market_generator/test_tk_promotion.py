"""TK-to-K promotion: the claim, the priority, and the rank convention.

From 2024-25 SFUSD promotes TK students into kindergarten at the same pathway
and school. ``assignment/docs/TK_PROMOTION_PLAN.md`` is the design; the two
things worth protecting here are that the weight is actually read (the
kindergarten loop used to swallow unknown keys) and that the zone restriction
still beats the boost, which is the whole behavioural difference between
policies #3 and #4.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from assignment.student_assignment.choice_ranks import promotion_first_choice_ranks
from assignment.student_assignment.data_interfaces.students import Students
from assignment.student_assignment.market_generator.policy import Policy
from assignment.student_assignment.market_generator.priority_generator import (
    PriorityGenerator,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


# 101 is the oversubscribed school both students rank; 102 is the promote's
# feeder; 103 is the promote's attendance-area school, their fallback.
PROGRAM_INDICES = {"101-GE-KG": 1, "102-GE-KG": 2, "103-GE-KG": 3}
POPULAR, FEEDER, ATTENDANCE_AREA = 1, 2, 3


def _students(promote, *, sibling=None, ctip=(0, 0), mr_applicant=(0, 1)):
    """Two kindergarten students: a TK promote and a stranger."""
    students = Students.__new__(Students)
    students.n = 2
    students.num_programs = 3
    students.rounds = 1
    students.student_data = pd.DataFrame(
        {
            "promote": promote,
            "mr_applicant": list(mr_applicant),
            "ctip1": list(ctip),
            "first_participating_round_ordinal": [0, 0],
            # The promote's attendance area is 103, not their feeder: under a
            # zone policy that is the difference between keeping the feeder
            # and losing it.
            "idschoolattendance": [103, 101],
        },
        index=pd.Index([10, 11], name="studentno"),
    )
    students.student_data["currentlpsibling"] = [[], []]
    students.student_data["selected_cohortstring"] = [[], []]
    students.student_data["selected_ranked_idschool"] = [[], []]
    students.student_data["selected_programs"] = [[], []]
    students._sibling = (
        np.zeros((2, 3), dtype=int) if sibling is None else np.asarray(sibling)
    )
    students.idx2studentno = {0: 10, 1: 11}
    students.studentno2idx = {10: 0, 11: 1}
    return students


def _config(weights, **overrides):
    config = {
        "grade": "KG",
        "year": 26,
        "subconfig-name": "tk-promotion",
        "priority-weights": weights,
        "distance-boost": None,
        "distance-priority": {"thresholds": [0.5, 1, 2]},
        "restrict-zone": False,
        "non_designation_boost": 128,
        "assignment-algorithm": "DA",
    }
    config.update(overrides)
    return config


def _market(
    config,
    students,
    *,
    capacity=(0, 1, 1),
    zone_eligibility=None,
    program_indices=None,
    citywide_language_programs=(),
):
    zone_priority = np.array([[0, 0, 1], [0, 1, 0]])
    if zone_eligibility is None:
        zone_eligibility = np.ones((2, 3), dtype=int)
    return SimpleNamespace(
        n=2,
        num_programs=3,
        config=config,
        students=students,
        schools=SimpleNamespace(citywide_schools=[102]),
        programs=SimpleNamespace(
            indices=PROGRAM_INDICES if program_indices is None else program_indices,
            capacity=np.asarray(capacity),
            citywide_language_program_indices=(
                lambda _schools: list(citywide_language_programs)
            ),
        ),
        zones=SimpleNamespace(
            zone_priority_matrix=zone_priority,
            zone_eligibility_matrix=np.asarray(zone_eligibility),
        ),
        # The promote ranks the popular school, the feeder, then their
        # attendance area; the stranger ranks only the feeder.
        preference_generator=SimpleNamespace(pref_length=np.array([3, 1])),
    )


PREFERENCES = np.array([[POPULAR, FEEDER, ATTENDANCE_AREA], [FEEDER, 0, 0]])


def _assignment_for(market):
    """Run DA over the market's own priorities, zone restriction included."""
    generator = MarketGenerator.__new__(MarketGenerator)
    generator.config = market.config
    generator.students = market.students
    generator.programs = market.programs
    generator.preference_generator = market.preference_generator
    priorities = PriorityGenerator(market).get_priorities_without_lottery(
        Policy("Con1", 1, 0, "MTB"), PREFERENCES
    )
    match, _, _, _ = generator._generate_assignment(PREFERENCES, priorities)
    return match


# --- the claim ---------------------------------------------------------


def test_promotion_matrix_marks_the_feeder_program_and_nothing_else():
    students = _students([["102-GE-KG"], []])

    promotion = students.promotion(PROGRAM_INDICES)

    np.testing.assert_array_equal(promotion, [[0, 1, 0], [0, 0, 0]])


def test_promotion_matrix_ignores_a_program_outside_this_market():
    # A Mission Bay feeder in a run that excludes Mission Bay.
    students = _students([["909-GE-KG"], []])

    np.testing.assert_array_equal(students.promotion(PROGRAM_INDICES), np.zeros((2, 3)))


def test_promotion_matrix_is_empty_for_years_without_the_column():
    students = _students([["102-GE-KG"], []])
    students.student_data = students.student_data.drop(columns=["promote"])

    np.testing.assert_array_equal(students.promotion(PROGRAM_INDICES), np.zeros((2, 3)))


# --- the weight --------------------------------------------------------


def test_promote_weight_is_read_by_the_kindergarten_loop():
    """The weight is not silently swallowed -- see plan sections 2.2 and 4.1."""
    students = _students([["102-GE-KG"], []])
    baseline = PriorityGenerator(
        _market(_config({"ctip": 8}), students)
    )._set_policy_priorities(1, "Con1")
    boosted = PriorityGenerator(
        _market(_config({"ctip": 8, "promote": 1024}), students)
    )._set_policy_priorities(1, "Con1")

    difference = boosted - baseline
    np.testing.assert_array_equal(difference, [[0, 1024, 0], [0, 0, 0]])


def test_promote_outranks_sibling_ctip_and_zone_combined():
    students = _students(
        [["102-GE-KG"], []],
        sibling=[[0, 0, 0], [0, 1, 0]],
        ctip=(0, 1),
    )
    weights = {"ctip": 8, "sibling": 16, "zone": 256, "promote": 1024}

    priorities = PriorityGenerator(
        _market(_config(weights), students)
    )._set_policy_priorities(1, "Con1")

    assert priorities[0, FEEDER - 1] > priorities[1, FEEDER - 1]


def test_promote_survives_the_citywide_language_program_mask():
    """The language mask replaces priorities; the claim has to outlive it."""
    language_indices = {"101-GE-KG": 1, "102-CN-KG": 2, "103-GE-KG": 3}
    weights = {
        "ctip": 8,
        "sibling": 16,
        "zone": 256,
        "promote": 1024,
        "language-programs": {"lp-sibling": 16, "lp": 8, "sibling": 4, "ctip": 2},
    }
    market = _market(
        _config(weights),
        _students([["102-CN-KG"], []]),
        program_indices=language_indices,
        citywide_language_programs=[FEEDER],
    )

    priorities = PriorityGenerator(market)._set_policy_priorities(1, "Con1")

    assert priorities[0, FEEDER - 1] == 1024


def test_older_years_are_untouched_by_a_configured_promote_weight():
    students = _students([["102-GE-KG"], []])
    students.student_data = students.student_data.drop(columns=["promote"])
    weights = {"ctip": 8, "sibling": 16, "zone": 256}

    without_key = PriorityGenerator(
        _market(_config(weights), students)
    )._set_policy_priorities(1, "Con1")
    with_key = PriorityGenerator(
        _market(_config({**weights, "promote": 1024}), students)
    )._set_policy_priorities(1, "Con1")

    np.testing.assert_array_equal(with_key, without_key)


def test_unknown_kindergarten_priority_category_is_rejected():
    generator = PriorityGenerator(
        _market(_config({"promotes": 1024}), _students([[], []]))
    )

    with pytest.raises(ValueError, match="Unknown priority category 'promotes'"):
        generator._set_policy_priorities(1, "Con1")


# --- the behaviour -----------------------------------------------------


def test_promote_keeps_the_feeder_seat_when_their_first_choice_fills():
    students = _students(
        [["102-GE-KG"], []],
        sibling=[[0, 0, 0], [0, 1, 0]],
        ctip=(0, 1),
    )
    weights = {"ctip": 8, "sibling": 16, "zone": 256, "promote": 1024}
    market = _market(_config(weights), students, capacity=(0, 1, 1))

    match = _assignment_for(market)

    # The promote takes the feeder's only seat; the stranger, who outranks
    # them on every other priority, is the one left out.
    np.testing.assert_array_equal(match, [FEEDER, 0])


def test_promote_releases_the_feeder_seat_when_they_win_elsewhere():
    students = _students(
        [["102-GE-KG"], []],
        sibling=[[0, 0, 0], [0, 1, 0]],
        ctip=(0, 1),
    )
    weights = {"ctip": 8, "sibling": 16, "zone": 256, "promote": 1024}
    market = _market(_config(weights), students, capacity=(1, 1, 1))

    match = _assignment_for(market)

    # Capacity is gross: the seat the promote gives up re-enters the market.
    np.testing.assert_array_equal(match, [POPULAR, FEEDER])


def test_restrict_zone_blocks_an_out_of_zone_feeder():
    """Plan section 2.4: under a zone policy the block wins, boost or not."""
    weights = {"ctip": 8, "sibling": 16, "zone": 256, "promote": 1024}
    unrestricted = _market(
        _config(weights),
        _students(
            [["102-GE-KG"], []],
            sibling=[[0, 0, 0], [0, 1, 0]],
            ctip=(0, 1),
        ),
        capacity=(0, 1, 1),
    )
    restricted = _market(
        _config(weights, **{"restrict-zone": True}),
        _students(
            [["102-GE-KG"], []],
            sibling=[[0, 0, 0], [0, 1, 0]],
            ctip=(0, 1),
        ),
        capacity=(0, 1, 1),
        # The promote's zone is their attendance area, which is not the
        # feeder's: two thirds of real promotes are in this position.
        zone_eligibility=[[0, 0, 1], [0, 1, 0]],
    )

    np.testing.assert_array_equal(_assignment_for(unrestricted), [FEEDER, 0])
    np.testing.assert_array_equal(
        _assignment_for(restricted), [ATTENDANCE_AREA, FEEDER]
    )


# --- the rank convention -----------------------------------------------


def _rank_frame():
    return pd.DataFrame(
        {
            "promote": [
                ["102-GE-KG"],
                ["102-GE-KG"],
                ["102-GE-KG"],
                [],
            ],
            "mr_applicant": [0, 1, 0, 0],
        }
    )


def test_promotion_first_choice_applies_only_to_non_applicants():
    matches = np.array([FEEDER, FEEDER, POPULAR, FEEDER])
    listed = np.array([6.0, 6.0, 2.0, 3.0])

    ranks = promotion_first_choice_ranks(
        _rank_frame(), PROGRAM_INDICES, matches, listed
    )

    np.testing.assert_array_equal(
        ranks,
        [
            1.0,  # promote, filed no request: the feeder is their whole list
            6.0,  # promote who also applied: the feeder really was 6th
            2.0,  # won something they ranked above the feeder
            3.0,  # not a promote
        ],
    )


def test_promotion_first_choice_leaves_years_without_the_columns_alone():
    frame = _rank_frame().drop(columns=["promote", "mr_applicant"])
    listed = np.array([6.0, 6.0, 2.0, 3.0])

    ranks = promotion_first_choice_ranks(
        frame, PROGRAM_INDICES, np.array([FEEDER] * 4), listed
    )

    np.testing.assert_array_equal(ranks, listed)


def test_promotion_first_choice_skips_unassigned_students():
    matches = np.array([0, FEEDER, POPULAR, FEEDER])
    listed = np.array([np.nan, 6.0, 2.0, 3.0])

    ranks = promotion_first_choice_ranks(
        _rank_frame(), PROGRAM_INDICES, matches, listed
    )

    assert np.isnan(ranks[0])

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from assignment.student_assignment.market_generator.preference_generator import (
    PreferenceGenerator,
)
from assignment.student_assignment.market_generator.priority_generator import (
    PriorityGenerator,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


def _preference_market(config, initial_preferences):
    program_indices = {
        "101-GE-KG": 1,
        "200-GE-KG": 2,
        "102-GE-KG": 3,
        "101-CB-KG": 4,
    }
    students = SimpleNamespace(
        first_round=np.zeros(3, dtype=int),
        attendance_area=pd.Series({10: 101, 11: 102, 12: 999}),
        idx2studentno={0: 10, 1: 11, 2: 12},
        student_preferences=Mock(return_value=initial_preferences),
    )
    programs = SimpleNamespace(
        indices=program_indices,
        index_list=lambda codes: [program_indices[code] for code in codes],
        school_to_indices={101: [1, 4], 200: [2], 102: [3]},
        citywide_program_indices=lambda schools: [2] if 200 in schools else [],
    )
    return SimpleNamespace(
        n=3,
        num_programs=4,
        config=config,
        students=students,
        programs=programs,
        schools=SimpleNamespace(citywide_schools=[200]),
    )


def test_add_aa_schools_appends_missing_ge_program_to_real_preferences():
    initial_preferences = np.array(
        [
            [2, 4, 0, 0],
            [3, 2, 0, 0],
            [4, 0, 0, 0],
        ]
    )
    market = _preference_market(
        {"add_aa_schools": True, "grade": "KG"}, initial_preferences
    )
    generator = PreferenceGenerator(market)

    preferences = generator.initialize_real_preferences(designate=False)

    np.testing.assert_array_equal(
        preferences,
        np.array(
            [
                [2, 4, 1, 0],
                [3, 2, 0, 0],
                [4, 0, 0, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(generator.pref_length, [3, 2, 1])


def test_add_aa_schools_applies_to_utility_model_preferences():
    initial_preferences = np.array(
        [
            [2, 0, 0, 0],
            [2, 3, 0, 0],
            [4, 0, 0, 0],
        ]
    )
    market = _preference_market(
        {"add_aa_schools": True, "designate": False, "grade": "KG"},
        initial_preferences,
    )
    generator = PreferenceGenerator(market)
    generator._get_eligibility = Mock(return_value=np.ones((3, 4)))
    generator._truncate_utility_model_preferences = Mock(
        return_value=initial_preferences
    )

    preferences = generator.get_utility_model_preferences_after_truncation()

    np.testing.assert_array_equal(
        preferences,
        np.array(
            [
                [2, 1, 0, 0],
                [2, 3, 0, 0],
                [4, 0, 0, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(generator.pref_length, [2, 2, 1])


def test_drop_below_aa_truncates_real_preferences_after_aa_is_added():
    initial_preferences = np.array(
        [
            [2, 1, 4, 3],
            [1, 2, 0, 0],
            [4, 2, 3, 0],
        ]
    )
    market = _preference_market(
        {
            "add_aa_schools": True,
            "drop_below_aa": True,
            "grade": "KG",
        },
        initial_preferences,
    )
    generator = PreferenceGenerator(market)

    preferences = generator.initialize_real_preferences(designate=False)

    np.testing.assert_array_equal(
        preferences,
        np.array(
            [
                [2, 1, 0, 0],
                [1, 2, 3, 0],
                [4, 2, 3, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(generator.pref_length, [2, 3, 3])


def test_drop_below_aa_truncates_utility_model_preferences():
    initial_preferences = np.array(
        [
            [2, 1, 4, 0],
            [1, 3, 2, 0],
            [4, 2, 3, 0],
        ]
    )
    market = _preference_market(
        {"drop_below_aa": True, "designate": False, "grade": "KG"},
        initial_preferences,
    )
    generator = PreferenceGenerator(market)
    generator._get_eligibility = Mock(return_value=np.ones((3, 4)))
    generator._truncate_utility_model_preferences = Mock(
        return_value=initial_preferences
    )

    preferences = generator.get_utility_model_preferences_after_truncation()

    np.testing.assert_array_equal(
        preferences,
        np.array(
            [
                [2, 1, 0, 0],
                [1, 3, 0, 0],
                [4, 2, 3, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(generator.pref_length, [2, 2, 3])


def test_drop_below_aa_defaults_to_false():
    initial_preferences = np.array(
        [
            [2, 1, 4, 0],
            [1, 3, 2, 0],
            [4, 2, 3, 0],
        ]
    )
    market = _preference_market({"grade": "KG"}, initial_preferences)
    generator = PreferenceGenerator(market)

    preferences = generator.initialize_real_preferences(designate=False)

    np.testing.assert_array_equal(preferences, initial_preferences)
    np.testing.assert_array_equal(generator.pref_length, [3, 3, 3])


def test_drop_below_aa_keeps_full_ranked_length_before_designation():
    initial_preferences = np.array(
        [
            [2, 4, 3, 0],
            [1, 2, 0, 0],
            [4, 2, 3, 0],
        ]
    )
    market = _preference_market(
        {
            "add_aa_schools": True,
            "drop_below_aa": True,
            "grade": "KG",
        },
        initial_preferences,
    )
    generator = PreferenceGenerator(market)
    generator._get_eligibility = Mock(return_value=np.ones((3, 4)))
    generator._generate_designation_program_ordering = Mock()
    generator._designation_ordering = {10: [], 11: [], 12: []}

    preferences = generator.initialize_real_preferences(designate=True)

    np.testing.assert_array_equal(
        preferences,
        np.array(
            [
                [2, 4, 3, 1],
                [1, 2, 3, 0],
                [4, 2, 3, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(generator.pref_length, [4, 3, 3])


def test_remove_non_aa_or_citywide_filters_real_preferences_by_school():
    initial_preferences = np.array(
        [
            [3, 4, 2, 0],
            [1, 2, 3, 0],
            [4, 2, 3, 0],
        ]
    )
    market = _preference_market(
        {"remove_non_aa_or_citywide": True}, initial_preferences
    )
    generator = PreferenceGenerator(market)

    preferences = generator.initialize_real_preferences(designate=False)

    np.testing.assert_array_equal(
        preferences,
        np.array(
            [
                [4, 2, 0, 0],
                [2, 3, 0, 0],
                [2, 0, 0, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(generator.pref_length, [2, 2, 1])


def test_remove_non_aa_or_citywide_uses_top_allowed_utility_preferences():
    original_preferences = np.array(
        [
            [3, 2, 1, 4],
            [1, 4, 2, 3],
            [4, 3, 2, 1],
        ]
    )
    market = _preference_market(
        {
            "remove_non_aa_or_citywide": True,
            "designate": False,
            "grade": "KG",
            "utility-model": {"list-length": "2"},
        },
        original_preferences,
    )
    market.students.student_data = pd.DataFrame({"num_ranked": [2, 2, 2]})
    market.umodel = SimpleNamespace(original_preferences=original_preferences)
    generator = PreferenceGenerator(market)
    generator._get_eligibility = Mock(return_value=np.ones((3, 4)))

    preferences = generator.get_utility_model_preferences_after_truncation()

    np.testing.assert_array_equal(
        preferences,
        np.array(
            [
                [2, 1, 0, 0],
                [2, 3, 0, 0],
                [2, 0, 0, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(generator.pref_length, [2, 2, 1])


def test_remove_non_aa_or_citywide_filters_designation_options():
    initial_preferences = np.array(
        [
            [1, 0, 0, 0],
            [3, 0, 0, 0],
            [2, 0, 0, 0],
        ]
    )
    designated_preferences = np.array(
        [
            [1, 3, 2, 0],
            [3, 1, 2, 0],
            [2, 4, 3, 0],
        ]
    )
    market = _preference_market(
        {"remove_non_aa_or_citywide": True}, initial_preferences
    )
    generator = PreferenceGenerator(market)
    generator._get_eligibility = Mock(return_value=np.ones((3, 4)))
    generator._add_designation_programs_to_preferences = Mock(
        return_value=designated_preferences
    )

    preferences = generator.initialize_real_preferences(designate=True)

    np.testing.assert_array_equal(
        preferences,
        np.array(
            [
                [1, 2, 0, 0],
                [3, 2, 0, 0],
                [2, 0, 0, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(generator.pref_length, [1, 1, 1])


def test_aa_boost_applies_only_to_attendance_area_ge_program():
    students = SimpleNamespace(
        attendance_area=pd.Series({10: 101, 11: 102, 12: 999}),
        idx2studentno={0: 10, 1: 11, 2: 12},
        sibling=lambda programs: np.zeros((3, 4)),
    )
    config = {
        "aa_boost": 1000,
        "distance-boost": None,
        "distance-priority": {"thresholds": [0.5, 1, 2]},
        "grade": "KG",
        "priority-weights": {},
        "subconfig-name": "aa-policy",
        "year": 22,
    }
    market = SimpleNamespace(
        n=3,
        num_programs=4,
        config=config,
        students=students,
        programs=SimpleNamespace(
            indices={
                "101-GE-KG": 1,
                "101-CB-KG": 2,
                "102-GE-KG": 3,
                "300-GE-KG": 4,
            }
        ),
    )
    generator = PriorityGenerator(market)

    priorities = generator._set_policy_priorities(0, "Con1")

    np.testing.assert_array_equal(
        priorities,
        np.array(
            [
                [1000, 0, 0, 0],
                [0, 0, 1000, 0],
                [0, 0, 0, 0],
            ]
        ),
    )

    market.config["aa_boost"] = 500
    updated_priorities = generator._set_policy_priorities(0, "Con1")
    assert updated_priorities[0, 0] == 500


def test_overscribe_aa_defaults_to_false():
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {"grade": "KG"}

    match = np.array([0])
    rank = np.array([1])
    updated_match, updated_rank = market._overscribe_attendance_area(
        np.array([[1]]), match, rank
    )

    np.testing.assert_array_equal(updated_match, match)
    np.testing.assert_array_equal(updated_rank, rank)


def test_overscribe_aa_only_assigns_unassigned_students_to_aa_ge():
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {"grade": "KG", "overscribe_aa": True}
    market.students = SimpleNamespace(
        attendance_area=pd.Series({10: 101, 11: 102, 12: 999}),
        idx2studentno={0: 10, 1: 11, 2: 12},
    )
    market.programs = SimpleNamespace(
        indices={"101-GE-KG": 1, "200-GE-KG": 2, "102-GE-KG": 3}
    )
    market.preference_generator = SimpleNamespace(
        pref_length=np.array([2, 2, 1])
    )

    match, rank = market._overscribe_attendance_area(
        np.array([[2, 1, 0], [2, 3, 0], [2, 0, 0]]),
        np.array([0, 2, 0]),
        np.array([2, 1, 2]),
    )

    np.testing.assert_array_equal(match, [1, 2, 0])
    np.testing.assert_array_equal(rank, [2, 1, 2])


def test_overscribe_aa_allows_enrollment_above_capacity():
    market = MarketGenerator.__new__(MarketGenerator)
    market.config = {
        "assignment-algorithm": "DA",
        "grade": "KG",
        "overscribe_aa": True,
    }
    market.students = SimpleNamespace(
        attendance_area=pd.Series({10: 101, 11: 101}),
        idx2studentno={0: 10, 1: 11},
        studentno2idx={10: 0, 11: 1},
    )
    market.programs = SimpleNamespace(
        capacity=np.array([1, 0]),
        indices={"101-GE-KG": 1, "200-GE-KG": 2},
    )
    market.preference_generator = SimpleNamespace(pref_length=np.array([1, 1]))
    prefs = np.array([[1], [1]])

    match, rank, _ = market._generate_assignment(
        prefs, np.array([[10, 0], [20, 0]])
    )

    np.testing.assert_array_equal(match, [1, 1])
    assert np.count_nonzero(match == 1) > market.programs.capacity[0]

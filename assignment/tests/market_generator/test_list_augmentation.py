import numpy as np
import pandas as pd

from assignment.student_assignment.market_generator.list_augmentation import (
    augment_preferences,
    identify_oversubscribed_programs,
    identify_targeted_students,
)


def test_targeting_normalizes_known_ethnicity_aliases():
    students = pd.DataFrame(
        {
            "ctip1": [1, 1, 1, 1, 1, 1],
            "resolved_ethnicity": [
                "Hispanic/Latinx",
                "Black/African American",
                "Other Pacific Islander",
                "Multi-Racial",
                "American Indian",
                "White",
            ],
        }
    )

    targeted = identify_targeted_students(
        students,
        np.ones(len(students), dtype=int),
        {"targeting-method": "ctip_x_ethnicity"},
    )

    np.testing.assert_array_equal(targeted, [True, True, True, True, True, False])


def test_oversubscription_counts_only_submitted_preferences():
    preferences = np.array(
        [
            [1, 2],
            [1, 2],
            [0, 2],
        ]
    )

    oversubscribed = identify_oversubscribed_programs(
        preferences,
        capacity=np.array([1, 1]),
        school_to_indices={},
        config={
            "oversubscribed-method": "apps_per_seat",
            "oversubscribed-ratio-threshold": 1.5,
        },
        pref_lengths=np.array([1, 1, 0]),
    )

    np.testing.assert_array_equal(oversubscribed, [1])


def test_augmentation_intersects_eligibility_and_preserves_designation_tail():
    preferences = np.array([[1, 2, 3, 3, 4]])
    student_data = pd.DataFrame(
        {"ctip1": [1], "resolved_ethnicity": ["Hispanic/Latinx"]}
    )

    augmented, lengths, _ = augment_preferences(
        preferences,
        pref_lengths=np.array([1]),
        targeted_mask=np.array([True]),
        oversubscribed_programs=np.array([2, 3]),
        student_data=student_data,
        distance_matrix=np.array([[9.0, 1.0, 2.0, 3.0, 4.0]]),
        config={"max-augmented-programs": 1},
        eligibility_matrix=np.array([[True, False, True, True, True]]),
    )

    np.testing.assert_array_equal(augmented, [[3, 1, 2, 4, 0]])
    np.testing.assert_array_equal(lengths, [2])

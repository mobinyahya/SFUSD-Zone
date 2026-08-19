import numpy as np
import pandas as pd
import pytest

from assignment.student_assignment.choice_ranks import (
    cumulative_choice_rates,
    listed_preference_rank_matrix,
    normalize_assignment_ranks,
    ranks_for_matches,
    ranks_from_preference_order,
)


def test_listed_ranks_preserve_gaps_and_exact_program_identity():
    students = pd.DataFrame(
        {
            "selected_ranked_idschool": [[101, 202], [101]],
            "selected_programs": [["GE", "GE"], ["SN"]],
            "selected_listed_ranks": [[1, 4], [2]],
            "grade": ["KG", "KG"],
        }
    )
    program_indices = {
        "101-GE-KG": 1,
        "202-GE-KG": 2,
        "101-SN-KG": 3,
        "101-GE-01": 4,
    }

    ranks = listed_preference_rank_matrix(students, program_indices)

    np.testing.assert_allclose(
        ranks,
        [[1, 4, np.nan, np.nan], [np.nan, np.nan, 2, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        ranks_for_matches(ranks, np.array([2, 0])),
        [4, np.nan],
        equal_nan=True,
    )


def test_utility_ranks_use_unfiltered_preference_order():
    preferences = np.array([[2, 3, 1], [3, 1, 2]])

    ranks = ranks_from_preference_order(preferences, np.array([1, 0]))

    np.testing.assert_allclose(ranks, [3, np.nan], equal_nan=True)


def test_empty_listed_rank_metadata_uses_source_positions():
    students = pd.DataFrame(
        {
            "selected_ranked_idschool": [[101, 202]],
            "selected_programs": [["GE", "GE"]],
            "selected_listed_ranks": [[]],
            "grade": ["KG"],
        }
    )

    ranks = listed_preference_rank_matrix(
        students,
        {"101-GE-KG": 1, "202-GE-KG": 2},
    )

    np.testing.assert_allclose(ranks, [[1, 2]])


def test_choice_rates_require_assignment_and_exclude_designations():
    students = pd.DataFrame(
        {
            "programno": [0, 1, 2, 3],
            "designation": [0, 0, 1, 0],
            "rank": [1, 2, 1, np.nan],
        }
    )

    outcomes = cumulative_choice_rates(students, "rank", [1, 2, 3])

    assert outcomes[1].numerator == 0
    assert outcomes[2].numerator == 1
    assert outcomes[3].numerator == 1
    assert outcomes[2].denominator == 4
    assert outcomes[2].value == 0.25


def test_legacy_listed_ranks_are_reconstructed_but_utility_ranks_fail():
    legacy = pd.DataFrame(
        {
            "programno": [1, 0],
            "rank": [1, 1],
            "In-Zone Rank": [1, 1],
        }
    )

    normalized = normalize_assignment_ranks(legacy, listed_ranks=[4, np.nan])

    assert normalized["rank"].tolist()[0] == 4
    assert pd.isna(normalized.loc[1, "rank"])
    assert pd.isna(normalized.loc[1, "mechanism_rank"])

    legacy["assigned_utility"] = [2.5, np.nan]
    with pytest.raises(ValueError, match="explicit utility_rank"):
        normalize_assignment_ranks(legacy, listed_ranks=[4, np.nan])

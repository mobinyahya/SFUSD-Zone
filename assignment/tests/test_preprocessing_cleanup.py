import ast
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from assignment.create_simulator_input import ConvertEstimates
from assignment.filter_student_choices import (
    filter_student_choices,
    load_estimates,
    merge_students_with_estimates,
)


def test_estimate_school_ids_are_integers(tmp_path):
    estimates_path = tmp_path / "estimates.csv"
    pd.DataFrame(
        {
            "studentno": ["22-7"],
            "101-GE-KG": [0.8],
            "202-SN-KG": [0.2],
        }
    ).to_csv(estimates_path, index=False)

    estimates = load_estimates(str(estimates_path), number_of_programs=2)
    school_ids = ast.literal_eval(estimates.loc[0, "r1_ranked_idschool"])

    assert school_ids == [101, 202]
    assert all(isinstance(school_id, int) for school_id in school_ids)


def test_filter_keeps_populated_lists_aligned_and_empty_ancillary_lists_empty():
    students = pd.DataFrame(
        {
            "latitude": [37.0],
            "longitude": [-122.0],
            "r1_ranked_idschool": ["[101, 202, 303]"],
            "r1_listed_ranks": ["[1, 2, 3]"],
            "r1_programs": ["['GE', 'GE', 'SN']"],
            "r1_randomnumber": ["[0.1, 0.2, 0.3]"],
            "r1_cohortstring": ["[]"],
        },
        index=[42],
    )
    schools = pd.DataFrame(
        {
            "school_id": [101, 202, 303],
            "lat": [37.0, 37.0, 37.0],
            "lon": [-122.0, -122.2, -122.3],
        }
    )

    result = filter_student_choices(
        students, schools, SimpleNamespace(distance=1.0, number=None)
    )

    assert ast.literal_eval(result.loc[42, "r1_ranked_idschool"]) == [101, 303]
    assert ast.literal_eval(result.loc[42, "r1_listed_ranks"]) == [1, 3]
    assert ast.literal_eval(result.loc[42, "r1_programs"]) == ["GE", "SN"]
    assert ast.literal_eval(result.loc[42, "r1_randomnumber"]) == [0.1, 0.3]
    assert ast.literal_eval(result.loc[42, "r1_cohortstring"]) == []


def test_filter_rejects_misaligned_populated_choice_lists():
    students = pd.DataFrame(
        {
            "latitude": [37.0],
            "longitude": [-122.0],
            "r1_ranked_idschool": ["[101, 202]"],
            "r1_programs": ["['GE', 'GE']"],
            "r1_randomnumber": ["[0.1]"],
        }
    )
    schools = pd.DataFrame(
        {
            "school_id": [101, 202],
            "lat": [37.0, 37.0],
            "lon": [-122.0, -122.1],
        }
    )

    with pytest.raises(ValueError, match=r"r1_randomnumber.*1 values.*expected 2"):
        filter_student_choices(
            students, schools, SimpleNamespace(distance=None, number=None)
        )


def test_merge_clears_choice_metadata_that_does_not_match_estimates():
    students = pd.DataFrame(
        {
            "studentno": [7],
            "grade": ["KG"],
            "r1_ranked_idschool": ["[999]"],
            "r1_programs": ["['GE']"],
            "r1_listed_ranks": ["[1]"],
            "r1_randomnumber": ["[0.5]"],
            "r1_cohortstring": ["['CL;x']"],
        }
    )
    estimates = pd.DataFrame(
        {
            "studentno": [7],
            "r1_ranked_idschool": ["[101, 202]"],
            "r1_programs": ["['GE', 'GE']"],
            "grade": ["KG"],
        }
    )

    merged = merge_students_with_estimates(students, estimates)

    for column in ["r1_listed_ranks", "r1_randomnumber", "r1_cohortstring"]:
        assert merged.loc[0, column] == "[]"


def test_distance_override_replaces_only_named_coefficient():
    converter = ConvertEstimates.__new__(ConvertEstimates)
    converter._distance_weight = -2.5
    weights_df = pd.DataFrame(
        {"coefficient": [10, 1, 3]},
        index=["intercept", "distance", "score"],
    )

    weights, features = converter._load_weights(weights_df)

    assert features == ["intercept", "distance", "score"]
    np.testing.assert_array_equal(weights, [10.0, -2.5, 3.0])
    assert weights_df.loc["distance", "coefficient"] == 1


def test_distance_override_fails_without_named_feature():
    converter = ConvertEstimates.__new__(ConvertEstimates)
    converter._distance_weight = -2.5
    weights_df = pd.DataFrame(
        {"coefficient": [10.0, 1.0]}, index=["intercept", "miles"]
    )

    with pytest.raises(ValueError, match="exactly one feature named 'distance'"):
        converter._load_weights(weights_df)


def test_build_estimates_uses_configured_student_order_and_normalized_grade(tmp_path):
    student_path = tmp_path / "configured_students.csv"
    pd.DataFrame({"studentno": [22, 11], "grade": [6, "06"]}).to_csv(
        student_path, index=False
    )

    converter = ConvertEstimates.__new__(ConvertEstimates)
    converter._student_data_file = str(student_path)
    converter._model_path = str(tmp_path)
    converter._distance_weight = None
    converter._grade = "06"
    converter._year = 22
    converter._students = SimpleNamespace(n=2, idx2studentno={0: 11, 1: 22})
    converter._programs = SimpleNamespace(num_programs=1)
    initial = pd.DataFrame(
        {"101-GE-06": [1.0, 2.0]},
        index=pd.Index(["2223-11", "2223-22"], name="studentno"),
    )
    converter._build_initial_estimate = lambda: initial.copy()
    converter._reorder_columns = lambda frame: frame
    converter._get_program_type_eligibility_matrix = lambda: np.zeros((2, 1))

    converter.build_estimates()

    saved = pd.read_csv(tmp_path / "estimates.csv")
    assert saved["studentno"].tolist() == ["2223-22", "2223-11"]

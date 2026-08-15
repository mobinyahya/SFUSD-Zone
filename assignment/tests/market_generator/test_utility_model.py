from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from assignment.student_assignment.market_generator.utility_model import UtilityModel


@pytest.fixture
def interfaces():
    programs = SimpleNamespace(
        num_programs=2,
        program_df=pd.DataFrame({"program_id": ["p1", "p2"], "programno": [1, 2]}),
    )
    students = SimpleNamespace(
        n=2,
        student_data=pd.DataFrame(index=pd.Index([10, 20], name="studentno")),
        idx2studentno={0: 10, 1: 20},
    )
    return programs, students


def test_csv_alignment_and_saved_csv_round_trip(tmp_path, interfaces):
    programs, students = interfaces
    source = tmp_path / "source.csv"
    saved = tmp_path / "saved.csv"
    pd.DataFrame(
        [[4.0, 3.0], [-np.inf, 1.0]],
        index=pd.Index(["2223-20", "2223-10"], name="studentno"),
        columns=["2223-p2", "2223-p1"],
    ).to_csv(source)
    expected = np.array([[1.0, -np.inf], [3.0, 4.0]])

    model = UtilityModel(source, programs, students)
    model.draw_utility_model_randomness(gumbel_scale=0)
    np.testing.assert_array_equal(model.original_utilities, expected)
    model.save_utility_matrix(saved)

    reloaded = UtilityModel(saved, programs, students)
    reloaded.draw_utility_model_randomness(gumbel_scale=0)
    np.testing.assert_array_equal(reloaded.original_utilities, expected)


def test_npy_negative_infinity_and_reduced_matrix_round_trip(tmp_path, interfaces):
    programs, students = interfaces
    programs.only_keep_cols = np.array([0, 2])
    students.only_keep_rows = np.array([0, 2])
    source = tmp_path / "source.npy"
    saved = tmp_path / "saved.npy"
    full = np.array([[1.0, 99.0, -np.inf], [98.0, 97.0, 96.0], [3.0, 95.0, 4.0]])
    np.save(source, full)
    expected = np.array([[1.0, -np.inf], [3.0, 4.0]])

    model = UtilityModel(source, programs, students)
    model.draw_utility_model_randomness(
        rows_to_keep=students.only_keep_rows,
        cols_to_keep=programs.only_keep_cols,
        gumbel_scale=0,
    )
    np.testing.assert_array_equal(model.original_utilities, expected)
    model.save_utility_matrix(saved)

    reloaded = UtilityModel(saved, programs, students)
    reloaded.draw_utility_model_randomness(
        rows_to_keep=students.only_keep_rows,
        cols_to_keep=programs.only_keep_cols,
        gumbel_scale=0,
    )
    np.testing.assert_array_equal(reloaded.original_utilities, expected)


@pytest.mark.parametrize("missing", ["student", "program"])
def test_csv_missing_required_identity_is_fatal(tmp_path, interfaces, missing):
    programs, students = interfaces
    utility_file = tmp_path / "utilities.csv"
    utilities = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index([10, 20], name="studentno"),
        columns=["p1", "p2"],
    )
    if missing == "student":
        utilities = utilities.drop(index=20)
    else:
        utilities = utilities.drop(columns="p2")
    utilities.to_csv(utility_file)

    with pytest.raises(ValueError, match=f"missing required {missing}"):
        UtilityModel(utility_file, programs, students).draw_utility_model_randomness(
            gumbel_scale=0
        )


def test_csv_duplicate_rows_and_columns_are_fatal(tmp_path, interfaces):
    programs, students = interfaces
    duplicate_rows = tmp_path / "duplicate_rows.csv"
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index([10, "2223-10"], name="studentno"),
        columns=["p1", "p2"],
    ).to_csv(duplicate_rows)
    with pytest.raises(ValueError, match="duplicate student rows"):
        UtilityModel(duplicate_rows, programs, students).draw_utility_model_randomness(
            gumbel_scale=0
        )

    duplicate_columns = tmp_path / "duplicate_columns.csv"
    pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.Index([10, 20], name="studentno"),
        columns=["p1", "p1"],
    ).to_csv(duplicate_columns)
    with pytest.raises(ValueError, match="duplicate columns"):
        UtilityModel(
            duplicate_columns, programs, students
        ).draw_utility_model_randomness(gumbel_scale=0)


@pytest.mark.parametrize("invalid", [np.nan, "not-a-number"])
def test_csv_invalid_utility_is_fatal(tmp_path, interfaces, invalid):
    programs, students = interfaces
    utility_file = tmp_path / "utilities.csv"
    pd.DataFrame(
        [[invalid, 2.0], [3.0, 4.0]],
        index=pd.Index([10, 20], name="studentno"),
        columns=["p1", "p2"],
    ).to_csv(utility_file)

    with pytest.raises(ValueError, match="non-numeric or NaN"):
        UtilityModel(utility_file, programs, students).draw_utility_model_randomness(
            gumbel_scale=0
        )


@pytest.mark.parametrize(
    "utilities, message",
    [
        (np.ones((2, 2, 1)), "two-dimensional"),
        (np.ones((1, 2)), "does not match required shape"),
        (np.array([[1.0, np.inf], [3.0, 4.0]]), "positive-infinite"),
    ],
)
def test_npy_shape_and_values_are_validated(tmp_path, interfaces, utilities, message):
    programs, students = interfaces
    utility_file = tmp_path / "utilities.npy"
    np.save(utility_file, utilities)

    with pytest.raises(ValueError, match=message):
        UtilityModel(utility_file, programs, students).draw_utility_model_randomness(
            gumbel_scale=0
        )

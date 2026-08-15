import numpy as np
import pandas as pd
import pytest

from assignment.scripts.preprocessing.extract_real_assignment import (
    extract_real_assignment,
)


def test_extract_real_assignment_retains_unassigned_and_missing_rank():
    students = pd.DataFrame(
        {
            "studentno": [1, 2],
            "r1_idschool": [101, np.nan],
            "r1_programcode": ["GE", np.nan],
            "grade": ["KG", "KG"],
        }
    )
    programs = pd.DataFrame({"program_id": ["101-GE-KG"], "programno": [7]})

    assignments = extract_real_assignment(students, programs)

    assert assignments["studentno"].tolist() == [1, 2]
    assert assignments["programno"].tolist() == [1, 0]
    assert assignments["programcodes"].tolist() == ["101-GE-KG", ""]
    assert assignments["rank"].isna().all()
    assert assignments["In-Zone Rank"].isna().all()
    assert assignments["designation"].tolist() == [0, 0]


def test_extract_real_assignment_masks_unassigned_rank():
    students = pd.DataFrame(
        {
            "studentno": [1, 2],
            "r1_idschool": [101, np.nan],
            "r1_programcode": ["GE", np.nan],
            "grade": ["KG", "KG"],
            "r1_rank": [2, 1],
            "r1_isdesignation": [1, np.nan],
        }
    )
    programs = pd.DataFrame({"program_id": ["101-GE-KG"], "programno": [1]})

    assignments = extract_real_assignment(students, programs)

    assert assignments.loc[0, "rank"] == 2
    assert pd.isna(assignments.loc[1, "rank"])
    assert assignments["designation"].tolist() == [1, 0]


def test_extract_real_assignment_requires_program_table():
    students = pd.DataFrame(
        {
            "studentno": [1],
            "r1_idschool": [101],
            "r1_programcode": ["GE"],
            "grade": ["KG"],
        }
    )

    with pytest.raises(ValueError, match="df_programs is required"):
        extract_real_assignment(students)


def test_extract_real_assignment_requires_exact_program_mapping():
    students = pd.DataFrame(
        {
            "studentno": [1],
            "r1_idschool": [101],
            "r1_programcode": ["GE"],
            "grade": ["KG"],
        }
    )
    programs = pd.DataFrame({"program_id": ["202-GE-KG"], "programno": [1]})

    with pytest.raises(ValueError, match="no exact program mapping"):
        extract_real_assignment(students, programs)


def test_extract_real_assignment_normalizes_numeric_grade():
    students = pd.DataFrame(
        {
            "studentno": [1],
            "r1_idschool": [101],
            "r1_programcode": ["GE"],
            "grade": [6],
        }
    )
    programs = pd.DataFrame({"program_id": ["101-GE-06"], "programno": [1]})

    assignments = extract_real_assignment(students, programs)

    assert assignments.loc[0, "programcodes"] == "101-GE-06"


def test_extract_real_assignment_normalizes_sparse_program_numbers():
    students = pd.DataFrame(
        {
            "studentno": [1],
            "r1_idschool": [404],
            "r1_programcode": ["GE"],
            "grade": ["KG"],
        }
    )
    programs = pd.DataFrame(
        {
            "program_id": ["404-GE-KG", "101-GE-KG"],
            "programno": [4, 1],
        }
    )

    assignments = extract_real_assignment(students, programs)

    assert assignments.loc[0, "programno"] == 2


def test_extract_real_assignment_rejects_duplicate_students():
    students = pd.DataFrame(
        {
            "studentno": [1, 1],
            "r1_idschool": [101, 101],
            "r1_programcode": ["GE", "GE"],
            "grade": ["KG", "KG"],
        }
    )
    programs = pd.DataFrame({"program_id": ["101-GE-KG"], "programno": [1]})

    with pytest.raises(ValueError, match="duplicate studentno"):
        extract_real_assignment(students, programs)


def test_extract_real_assignment_rejects_duplicate_program_numbers():
    students = pd.DataFrame(
        {
            "studentno": [1],
            "r1_idschool": [101],
            "r1_programcode": ["GE"],
            "grade": ["KG"],
        }
    )
    programs = pd.DataFrame(
        {
            "program_id": ["101-GE-KG", "202-GE-KG"],
            "programno": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="duplicate programno"):
        extract_real_assignment(students, programs)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("r1_rank", -1, "Invalid r1_rank"),
        ("r1_rank", 1.5, "Invalid r1_rank"),
        ("r1_isdesignation", "invalid", "Invalid r1_isdesignation"),
        ("r1_isdesignation", 0.5, "Invalid r1_isdesignation"),
    ],
)
def test_extract_real_assignment_rejects_invalid_metrics(column, value, message):
    students = pd.DataFrame(
        {
            "studentno": [1],
            "r1_idschool": [101],
            "r1_programcode": ["GE"],
            "grade": ["KG"],
            column: [value],
        }
    )
    programs = pd.DataFrame({"program_id": ["101-GE-KG"], "programno": [1]})

    with pytest.raises(ValueError, match=message):
        extract_real_assignment(students, programs)

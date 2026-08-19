import numpy as np
import pandas as pd
import pytest

from assignment.scripts.preprocessing.extract_real_assignment import (
    extract_real_assignment,
)
from assignment.student_assignment.choice_ranks import ASSIGNMENT_SCHEMA_VERSION


def _student_frame(data):
    frame = pd.DataFrame(data)
    ranked_schools = []
    ranked_programs = []
    listed_ranks = []
    for _, row in frame.iterrows():
        school = pd.to_numeric(
            pd.Series([row["r1_idschool"]]), errors="coerce"
        ).iloc[0]
        program = row["r1_programcode"]
        if pd.notna(school) and school > 0 and pd.notna(program):
            ranked_schools.append([int(school)])
            ranked_programs.append([str(program)])
            rank = pd.to_numeric(
                pd.Series([row.get("r1_rank")]), errors="coerce"
            ).iloc[0]
            listed_ranks.append(
                [int(rank)]
                if pd.notna(rank) and rank > 0 and float(rank).is_integer()
                else [1]
            )
        else:
            ranked_schools.append([])
            ranked_programs.append([])
            listed_ranks.append([])
    frame["r1_ranked_idschool"] = ranked_schools
    frame["r1_programs"] = ranked_programs
    frame["r1_listed_ranks"] = listed_ranks
    return frame


def test_extract_real_assignment_reconstructs_submitted_rank():
    students = _student_frame(
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
    assert assignments["assignment_schema_version"].eq(
        ASSIGNMENT_SCHEMA_VERSION
    ).all()
    assert assignments["rank_basis"].eq("listed").all()
    assert assignments.loc[0, "rank"] == 1
    assert pd.isna(assignments.loc[1, "rank"])
    assert assignments["In-Zone Rank"].isna().all()
    assert assignments["designation"].tolist() == [0, 0]


def test_extract_real_assignment_masks_unassigned_rank():
    students = _student_frame(
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
    students = _student_frame(
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
    students = _student_frame(
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
    students = _student_frame(
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
    students = _student_frame(
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
    students = _student_frame(
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
    students = _student_frame(
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


def test_extract_real_assignment_rejects_unassigned_designation():
    students = _student_frame(
        {
            "studentno": [1],
            "r1_idschool": [np.nan],
            "r1_programcode": [np.nan],
            "grade": ["KG"],
            "r1_isdesignation": [1],
        }
    )
    programs = pd.DataFrame({"program_id": ["101-GE-KG"], "programno": [1]})

    with pytest.raises(ValueError, match="Unassigned students cannot be designated"):
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
    students = _student_frame(
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

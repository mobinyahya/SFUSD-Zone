import pandas as pd

from assignment.scripts.preprocessing.prepare_kg_r1_inputs import (
    prepare_program_inputs,
    prepare_student_inputs,
)


def test_prepare_student_inputs_restricts_grade_round_and_specials():
    students = pd.DataFrame(
        {
            "studentno": [1, 2, 3, 4],
            "grade": ["KG", "KG", "KG", "01"],
            "r1_ranked_idschool": ["[10]", "[]", "[20]", "[30]"],
            "r1_programs": ["['GE']", "[]", "['AF']", "['GE']"],
        }
    )

    full, without_special = prepare_student_inputs(students)

    assert full["studentno"].tolist() == [1, 3]
    assert without_special["studentno"].tolist() == [1]
    assert full.loc[0, "r1_ranked_idschool"] == "[10]"


def test_prepare_program_inputs_restricts_grade_and_specials():
    programs = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "program_id": ["10-GE-KG", "20-AF-KG", "30-GE-01"],
            "program_type": ["GE", "AF", "GE"],
            "programno": [1, 2, 3],
        }
    )

    full, without_special = prepare_program_inputs(programs)

    assert full["program_id"].tolist() == ["10-GE-KG", "20-AF-KG"]
    assert without_special["program_id"].tolist() == ["10-GE-KG"]
    assert without_special["programno"].tolist() == [1]
    assert "Unnamed: 0" not in full.columns

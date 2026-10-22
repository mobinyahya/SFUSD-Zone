from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from assignment.student_assignment.da.da_with_guardrails import DAwithGuards
from assignment.student_assignment.da.guardrail_setup import GuardrailSetup


def test_lower_values_are_disadvantaged_reserve_class_zero():
    setup = GuardrailSetup.__new__(GuardrailSetup)
    setup.students = SimpleNamespace(
        student_data=pd.DataFrame({"score": [1.0, 4.0, 6.0, 9.0]})
    )

    setup._create_categories(
        {
            "column": "score",
            "thresholds": [5.0],
            "lower_disadvantaged": True,
        }
    )

    np.testing.assert_array_equal(setup.classOfStudent, [0, 0, 1, 1])


def test_zone_fractions_use_zone_labels_and_include_empty_categories():
    student_data = pd.DataFrame(
        {"diversity_category": [0, 1, 0]}, index=[100, 101, 102]
    )
    setup = GuardrailSetup.__new__(GuardrailSetup)
    setup.students = SimpleNamespace(student_data=student_data)
    setup.student2zone = {100: 10, 101: 42, 102: 10}
    setup.classOfStudent = np.array([0, 1, 0])
    setup.num_classes = 3

    fractions = setup._calculate_zone_fractions()

    assert list(fractions.columns) == [0, 1, 2]
    np.testing.assert_array_equal(fractions.loc[10], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(fractions.loc[42], [0.0, 1.0, 0.0])

    setup.programs = SimpleNamespace(
        program_type=pd.Series(["GE"]),
        indices={},
    )
    setup.market = SimpleNamespace(config={"citywide-separate-reserves": False})
    setup.program2zone = {0: 42}
    setup.num_programs = 1
    setup.n = 3
    setup._set_up_program_reserves({})

    np.testing.assert_array_equal(setup.program_reserve_frac[0], [0.0, 1.0, 0.0])


def test_zone_fractions_derive_class_count_for_lightweight_callers():
    setup = GuardrailSetup.__new__(GuardrailSetup)
    setup.students = SimpleNamespace(
        student_data=pd.DataFrame(
            {"diversity_category": [0, 1, 1, 0]},
            index=[101, 102, 103, 104],
        )
    )
    setup.student2zone = {101: 0, 102: 0, 103: 1, 104: 1}

    fractions = setup._calculate_zone_fractions()

    assert list(fractions.columns) == [0, 1]
    np.testing.assert_array_equal(fractions.loc[0], [0.5, 0.5])
    np.testing.assert_array_equal(fractions.loc[1], [0.5, 0.5])


def test_citywide_reserve_ratio_length_is_validated():
    setup = GuardrailSetup.__new__(GuardrailSetup)
    setup.classOfStudent = np.array([0, 1])
    setup.num_classes = 2
    setup.num_programs = 1
    setup.n = 2
    setup.programs = SimpleNamespace(
        program_type=pd.Series(["GE"]),
        indices={"618-GE-KG": 1},
    )
    setup.market = SimpleNamespace(
        config={
            "citywide-separate-reserves": True,
            "citywide-reserve-ratios": [1.0],
        }
    )

    with pytest.raises(ValueError, match="length does not match"):
        setup._set_up_program_reserves({})


def test_citywide_reserves_use_loaded_non_kg_program_ids():
    setup = GuardrailSetup.__new__(GuardrailSetup)
    setup.classOfStudent = np.array([0, 0])
    setup.num_classes = 1
    setup.num_programs = 2
    setup.n = 2
    setup.programs = SimpleNamespace(
        program_type=pd.Series(["GE", "GE"]),
        indices={"618-GE-06": 1, "999-GE-06": 2},
    )
    setup.market = SimpleNamespace(
        config={
            "grade": "06",
            "citywide-separate-reserves": False,
        }
    )

    setup._set_up_program_reserves({"reserve_fraction": [1.0], "citywide_only": True})

    np.testing.assert_array_equal(setup.program_reserve_frac[:, 0], [1.0, 0.0])


def test_fractional_strict_reserve_remainders_may_stay_empty():
    da = DAwithGuards(
        SchoolCaps=np.array([1]),
        StudentPrts=np.array([[2.0], [1.0]]),
        StudPrefs=np.array([[1], [1]]),
        classOfStudent=np.array([0, 1]),
        strictGuards=1,
    )
    da.setguards(np.array([[0.57, 0.43]]), numOfClasses=2)

    match, _ = da.run()

    np.testing.assert_array_equal(match, [0, 0])

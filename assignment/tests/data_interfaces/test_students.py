"""
Tests for data_interfaces/students.py with randomly generated
students, programs, and schools files.

Usage: python -m pytest tests/data_interfaces/test_students.py -k <test_name> -s
Run all the tests in this file: python -m pytest tests/data_interfaces/test_students.py

Last modified: November 27th, 2023
"""

import numpy as np
import pandas as pd
import pytest
from loaders import (
    load_program_records,
    load_scenario,
    load_school_records,
    load_student_records,
    parse_ranked_programs,
    parse_ranked_schools,
)

from assignment.student_assignment.data_interfaces import Programs
from assignment.student_assignment.data_interfaces.students import Students

from ..utils_for_tests import (
    LP_TYPE,
    NUM_SCHOOLS,
    NUM_STUDENT,
    YEAR,
    configure_synthetic_assignment_data,
    delete_temp_files,
    generate_random_program_school_files,
    generate_random_student_file,
)


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    config = {"grade": "06", "year": YEAR, "use-new-capacities": False}
    configure_synthetic_assignment_data(
        config, "06", tmp_path_factory.mktemp("student-distance-cache")
    )
    return config


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    request.addfinalizer(delete_temp_files)


def get_students(config):
    scenario = load_scenario(config["data"], environ={})
    runtime_config = {
        **config,
        "grade": scenario.filter("assignment", "grades")[0],
        "year": int(scenario.filter("assignment", "year")[:2]),
        "special_programs": scenario.filter("assignment", "special_programs"),
    }
    students = Students(
        load_student_records(
            scenario, "assignment.students", filter_group="assignment"
        ),
        get_programs(runtime_config),
        load_school_records(
            scenario,
            "assignment.school_coordinates",
            filter_group="assignment",
        ),
        None,
        runtime_config,
        data_scenario=scenario,
    )

    return students


def get_programs(config):
    scenario = load_scenario(config["data"], environ={})
    grade = scenario.filter("assignment", "grades")[0]
    runtime_config = {**config, "grade": grade, "year": YEAR}
    programs = Programs(
        load_program_records(
            scenario, "assignment.programs", filter_group="assignment"
        ),
        None,
        runtime_config,
    )
    return programs


def test_bayview_to_all_and_brown_ms(config):
    """
    Test getting bayview_to_all_ms and bayview_to_brown_ms priority settings.
    """
    _, school_ids = generate_random_program_school_files()
    priority_1s = np.random.randint(0, NUM_STUDENT, 100)
    _, priority_idx, _, _ = generate_random_student_file(
        school_ids, priority_1s=priority_1s
    )
    students = get_students(config)
    actual = students.bayview_to_all_ms

    assert (np.unique(actual[priority_idx]) == [1]).all()
    other_cols = np.delete(actual, priority_idx)
    assert (np.unique(other_cols) == [0]).all()

    actual = students.bayview_to_brown_ms
    assert (np.unique(actual[priority_idx]) == [1]).all()
    other_cols = np.delete(actual, priority_idx)
    assert (np.unique(other_cols) == [0]).all()


def test_zip_94124(config):
    """
    Test getting a list containing 1 at the indices with corresponding student
    with a zipcode of 94124.
    """
    _, school_ids = generate_random_program_school_files()
    special_zipcode_inds, _, _, _ = generate_random_student_file(school_ids)
    students = get_students(config)
    actual = students.zip_94124
    assert (np.unique(actual[special_zipcode_inds]) == [1]).all()
    other_cols = np.delete(actual, special_zipcode_inds)
    assert (np.unique(other_cols) == [0]).all()


def test_language_pathway_priorities(config):
    """
    Test getting language pathway priorities matrix based on students'
    previous pathway.
    """
    _, school_ids, lp_schools, _ = generate_random_program_school_files(return_lp=True)
    prev_lp = generate_random_student_file(school_ids, lp_schools)[2]
    students = get_students(config)
    # The column masked by the priority function should be depend on the
    # input program_type2indexes paran. We use 1-indexed programno.
    program_type2indexes = {LP_TYPE: np.arange(students.num_programs) + 1}
    actual = students.language_pathway_priority(program_type2indexes)

    lp_inds = np.argwhere(prev_lp == 1).flatten()
    assert (np.unique(actual[lp_inds]) == [1]).all()
    other_cols = np.delete(actual, lp_inds, axis=0)
    assert (np.unique(other_cols) == [0]).all()

    lp_cols = np.random.choice(np.arange(students.num_programs), 20)
    program_type2indexes = {
        LP_TYPE: lp_cols + 1,
        "MN": np.arange(students.num_programs),
    }
    actual = students.language_pathway_priority(program_type2indexes)

    assert (np.unique(actual[lp_inds][:, lp_cols]) == [1]).all()
    other_cols = np.delete(actual, lp_inds, axis=0)
    other_cols = np.delete(actual, lp_cols, axis=1)
    assert (np.unique(other_cols) == [0]).all()


def test_language_pathway_sibling(config):
    """
    Test getting the sibling's language pathway priorities matrix based on
    currentlpsibling column.
    """
    (
        _,
        school_ids,
        lp_schools,
        _,
        prog_id2idx,
        lp_school2idx,
    ) = generate_random_program_school_files(return_lp=True, return_idx=True)
    lp_sibling_schools = generate_random_student_file(school_ids, lp_schools)[3]
    students = get_students(config)
    actual = students.language_pathway_sibling(prog_id2idx)
    for i, sib in enumerate(lp_sibling_schools):
        if len(sib):
            col_inds = np.array([lp_school2idx[x] for x in sib])
            assert set(np.argwhere(actual[i] == 1).flatten()) == set(col_inds)
        else:
            assert (np.unique(actual[i]) == [0]).all()


def test_msf(config):
    """
    Test getting MSF indicator matrix based on msf column.
    """
    msf = [i % NUM_SCHOOLS for i in range(NUM_STUDENT + 1)]
    _, school_ids = generate_random_program_school_files()
    generate_random_student_file(school_ids, msf=msf)

    students = get_students(config)
    # The column masked by the priority function should be depend on the
    # input program_type2indexes paran. We use 1-indexed programno.
    school2idx = {x: [x + 1] for x in msf}
    actual = students.msf(school2idx)
    expected_args = [f"{i},{i % NUM_SCHOOLS}" for i in range(NUM_STUDENT)]
    actual_args = np.argwhere(actual == 1)
    assert actual_args.shape == (NUM_STUDENT, 2)
    actual_args_str = [f"{x},{y}" for [x, y] in actual_args]
    assert set(actual_args_str) == set(expected_args)


def test_sibling(config):
    """
    Test getting priorities based on students' siblings' programs.
    """
    _, school_ids = generate_random_program_school_files(
        lp_school_inds=np.arange(30), spe_school_inds=[]
    )
    siblings = [[school_ids[i % NUM_SCHOOLS]] for i in range(NUM_STUDENT + 1)]
    generate_random_student_file(school_ids, siblings=siblings)
    students = get_students(config)
    programs = get_programs(config)

    # Expected output for a student of index i should be 1 for program
    # i % NUM_SCHOOLS. If i % NUM_SCHOOLS < 30, the corresponding school also
    # has an language program so we have column  i % NUM_SCHOOLS + NUM_SCHOOLS
    # be 1.
    actual = students.sibling(programs)
    actual_args = np.argwhere(actual == 1)
    expected_args = []
    for i in range(NUM_STUDENT):
        expected_args.append([i, i % NUM_SCHOOLS])
        if i % NUM_SCHOOLS < 30:
            expected_args.append([i, i % NUM_SCHOOLS + NUM_SCHOOLS])
    expected_args_strs = [f"{x[0]},{x[1]}" for x in expected_args]
    actual_args_strs = [f"{x[0]},{x[1]}" for x in actual_args]
    assert set(actual_args_strs) == set(expected_args_strs)


def _minimal_student_inputs(tmp_path, student_rows):
    program_file = tmp_path / "programs.csv"
    school_file = tmp_path / "schools.csv"
    pd.DataFrame(
        {
            "programno": [1],
            "program_id": ["101-GE-06"],
            "school_id": [101],
            "program_type": ["GE"],
            "capacity": [10],
            "r1_assigned": [0],
            "r2_capacity": [10],
        }
    ).to_csv(program_file, index=False)
    pd.DataFrame({"school_id": [101], "lat": [37.75], "lon": [-122.45]}).to_csv(
        school_file, index=False
    )
    student_frame = pd.DataFrame(student_rows)
    student_frame["grade"] = "06"
    student_frame["r1_ranked_idschool"] = student_frame["r1_ranked_idschool"].map(
        parse_ranked_schools
    )
    student_frame["r1_programs"] = student_frame["r1_programs"].map(
        parse_ranked_programs
    )
    student_frame["first_participating_round"] = 1
    student_frame["first_participating_round_ordinal"] = 0
    student_frame["selected_ranked_idschool"] = student_frame["r1_ranked_idschool"].map(
        list
    )
    student_frame["selected_programs"] = student_frame["r1_programs"].map(list)
    student_frame.attrs["source_rows"] = list(range(len(student_frame)))
    student_frame.attrs["source_row_count"] = len(student_frame)
    config = {"grade": "06", "year": 21}
    programs = Programs(program_file, None, config)
    return student_frame, school_file, programs, config


def _student_row(studentno=1, schools="[101]", programs="['GE']"):
    return {
        "studentno": studentno,
        "grade": 6,
        "r1_ranked_idschool": schools,
        "r1_programs": programs,
        "latitude": 37.75,
        "longitude": -122.45,
        "HOCidx1": 0.5,
    }


def test_normalized_preferences_are_used_without_reparsing(tmp_path):
    inputs = _minimal_student_inputs(tmp_path, [_student_row(schools="[101, ]")])

    students = Students(inputs[0], inputs[2], inputs[1], None, inputs[3])

    assert students.student_data.loc[1, "selected_ranked_idschool"] == [101]
    np.testing.assert_array_equal(
        students.student_preferences(1, inputs[2].index_list), [[1]]
    )


def test_old_distance_cache_files_are_ignored(tmp_path):
    inputs = _minimal_student_inputs(tmp_path, [_student_row()])
    cache_file = tmp_path / "student_program_distances_06_2122.csv"
    pd.DataFrame({"studentno": [999], "101-GE-06": [1.0]}).to_csv(
        cache_file, index=False
    )

    students = Students(inputs[0], inputs[2], inputs[1], None, inputs[3])

    assert students.distance_data.index.tolist() == [1]
    assert students.distance_data.columns.tolist() == ["101-GE-06"]


def test_unknown_ranked_program_is_fatal(tmp_path):
    inputs = _minimal_student_inputs(tmp_path, [_student_row(schools="[999]")])

    with pytest.raises(ValueError, match="unknown program IDs.*999-GE-06"):
        Students(inputs[0], inputs[2], inputs[1], None, inputs[3])


def test_ranked_school_program_lengths_must_match(tmp_path):
    inputs = _minimal_student_inputs(tmp_path, [_student_row(schools="[101, 102]")])

    with pytest.raises(ValueError, match="2 ranked schools but 1 ranked programs"):
        Students(inputs[0], inputs[2], inputs[1], None, inputs[3])


def test_duplicate_student_identities_are_fatal(tmp_path):
    inputs = _minimal_student_inputs(tmp_path, [_student_row(), _student_row()])

    with pytest.raises(ValueError, match="duplicate studentno"):
        Students(inputs[0], inputs[2], inputs[1], None, inputs[3])

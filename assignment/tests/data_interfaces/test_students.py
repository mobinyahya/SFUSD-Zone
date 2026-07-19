"""
Tests for data_interfaces/students.py with randomly generated
students, programs, and schools files.

Usage: python -m pytest tests/data_interfaces/test_students.py -k <test_name> -s
Run all the tests in this file: python -m pytest tests/data_interfaces/test_students.py

Last modified: November 27th, 2023
"""

import numpy as np
import pytest

from assignment.student_assignment.configerator import Configerator
from assignment.student_assignment.data_interfaces import Programs
from assignment.student_assignment.data_interfaces.students import Students

from ..utils_for_tests import *


@pytest.fixture(scope="module")
def config():
    configerator = Configerator()
    configerator.config["subconfigs"] = ["choice_model_real_match_grade6"]
    configerator.load_subconfig_by_name("choice_model_real_match_grade6")
    # Set after load_subconfig_by_name: loading a subconfig rebuilds the
    # config from the original file and would discard these overrides.
    config = configerator.config
    config["grade"] = "06"
    config["use-new-capacities"] = False
    config["year"] = YEAR

    # Use the temp file path for generated data.
    config["paths"]["sfusd"] = TEMP_CLEANED_PAR_FOLDER
    config["paths"]["student-save"] = TEMP_STUDENT_SAVE_FOLDER
    return config


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    request.addfinalizer(delete_temp_files)


def get_students(config):
    students = Students(
        TEMP_STUDENT_FILE,
        get_programs(config),
        TEMP_SCHOOL_FILE,
        None,
        config,
    )

    return students


def get_programs(config):
    program_data_file = TEMP_PROGRAMS_FILE.format(config["grade"])
    programs = Programs(program_data_file, None, config)
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
    _, school_ids, lp_schools, _ = generate_random_program_school_files(
        return_lp=True
    )
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

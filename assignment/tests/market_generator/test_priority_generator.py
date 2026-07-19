"""
Tests for market_generator/priority_generator.py with randomly generated
students, programs, and schools files.

Usage: python -m pytest tests/market_generator/test_priority_generator.py -k <test_name> -s
Run all the tests in this file: python -m pytest tests/market_generator/test_priority_generator.py

Last modified: November 26th, 2023
"""

import numpy as np
import pytest

from assignment.student_assignment.configerator import Configerator
from assignment.student_assignment.market_generator.priority_generator import (
    PriorityGenerator,
)
from assignment.student_assignment.market_generator.school_choice_market import (
    SchoolChoiceMarket,
)

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


def get_priority_generator(config):
    market = SchoolChoiceMarket(config=config)
    return PriorityGenerator(market)


def get_ctip(priority_generator):
    return priority_generator.market.students.ctip


def get_msf(priority_generator):
    return priority_generator.market.students.msf(
        priority_generator.market.programs.school_to_indices
    )


def get_sibling(priority_generator):
    return priority_generator.market.students.sibling(
        priority_generator.market.programs
    )


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    request.addfinalizer(delete_temp_files)


def test_get_brown_ms_priorities(config):
    """
    Test getting 6th grade brown middle school priorities.
    """
    # brown middle school has the program code: "858-GE-06".
    brown_ms_index, school_ids = generate_random_program_school_files(858)
    special_zipcode_inds, bayview_to_brown_inds, _, _ = (
        generate_random_student_file(school_ids)
    )

    priority_generator = get_priority_generator(config)
    sibling = get_sibling(priority_generator)
    ctip = get_ctip(priority_generator)

    weights = {"sibling": 16, "bayview-to-brown": 8, "zip-94124": 4, "ctip": 2}
    actual, program_mask = priority_generator._get_brown_ms_priorities(
        weights, sibling, ctip
    )

    expect_brown_ms_col = np.zeros(priority_generator.market.students.n)
    expect_brown_ms_col[special_zipcode_inds] += weights["zip-94124"]
    expect_brown_ms_col[bayview_to_brown_inds] += weights["bayview-to-brown"]
    expect_brown_ms_col += ctip[:, 0] * weights["ctip"]
    expect_brown_ms_col += sibling[:, brown_ms_index] * weights["sibling"]
    assert (expect_brown_ms_col == actual[:, brown_ms_index]).all()

    other_cols = np.delete(actual, brown_ms_index, axis=1)
    assert (np.unique(other_cols) == [0]).all()

    expected_program_mask = np.zeros(priority_generator.market.num_programs)
    expected_program_mask[brown_ms_index] = 1
    assert np.equal(program_mask, expected_program_mask).all()


def test_get_sixth_grade_language_program_priorities(config):
    """
    Test getting 6th grade language program priorities.
    """
    _, school_ids, lp_schools, lp_mask = generate_random_program_school_files(
        return_lp=True
    )
    _, _, prev_lp, lp_sibling_schools = generate_random_student_file(
        school_ids, lp_schools
    )
    priority_generator = get_priority_generator(config)
    sibling = get_sibling(priority_generator)
    ctip = get_ctip(priority_generator)
    msf = get_msf(priority_generator)

    weights = {"lp-sibling": 32, "lp": 16, "sibling": 8, "msf": 4, "ctip": 2}
    (
        actual,
        program_mask,
    ) = priority_generator._get_sixth_grade_language_program_priorities(
        weights, sibling, msf, ctip
    )
    assert (lp_mask == program_mask).all()

    lp_program_inds = np.argwhere(np.array(lp_mask) == 1).flatten()
    expected = (
        weights["sibling"] * sibling
        + weights["msf"] * msf
        + weights["ctip"] * ctip
    )
    expected = expected[:, lp_program_inds]
    expected += np.repeat(
        [prev_lp * weights["lp"]], len(lp_program_inds), axis=0
    ).T

    school_to_prog = {lp_schools[i]: i for i in range(len(lp_schools))}
    for i, sibling_lps in enumerate(lp_sibling_schools):
        for sibling_lp in sibling_lps:
            expected[i, school_to_prog[sibling_lp]] += weights["lp-sibling"]

    assert (actual[:, lp_program_inds] == expected).all()

    other_cols = np.delete(actual, lp_program_inds, axis=1)
    assert (np.unique(other_cols) == [0]).all()


def test_bayview_student_priorities(config):
    """
    Test getting 6th grade priorities for bayview students.
    """
    _, school_ids = generate_random_program_school_files()
    _, bayview_to_all_ms_inds, _, _ = generate_random_student_file(school_ids)
    priority_generator = get_priority_generator(config)

    weights = {
        "bayview-to-all": 4,
    }
    actual, program_mask = priority_generator._get_bayview_student_priorities(
        weights
    )

    assert (
        np.unique(actual[bayview_to_all_ms_inds, :])
        == [weights["bayview-to-all"]]
    ).all()
    assert np.equal(
        program_mask, np.zeros(priority_generator.market.num_programs)
    ).all()

    other_rows = np.delete(actual, bayview_to_all_ms_inds, axis=0)
    assert (np.unique(other_rows) == [0]).all()


def test_get_remaining_ms_priorities(config):
    """
    Test getting 6th grade priority for programs not handled by language
    programs or Brown middle school programs.
    """
    _, school_ids = generate_random_program_school_files()
    _ = generate_random_student_file(school_ids)
    priority_generator = get_priority_generator(config)

    sibling = get_sibling(priority_generator)
    ctip = get_ctip(priority_generator)
    msf = get_msf(priority_generator)

    weights = {"sibling": 16, "msf": 8, "ctip": 2}

    # All programs are handled by other priority classes.
    program_mask = np.ones(priority_generator.market.num_programs)
    actual = priority_generator._get_remaining_ms_priorities(
        weights, sibling, msf, ctip, program_mask
    )
    assert (np.unique(actual) == [0]).all()

    # Random choice of masked programs.
    program_mask = np.random.choice(
        [0, 1], priority_generator.market.num_programs
    )
    all_priority = (
        weights["sibling"] * sibling
        + weights["msf"] * msf
        + weights["ctip"] * ctip
    )
    actual = priority_generator._get_remaining_ms_priorities(
        weights, sibling, msf, ctip, program_mask
    )
    for i in np.argwhere(program_mask == 0).flatten():
        assert (actual[:, i] == all_priority[:, i]).all()
    other_cols = np.delete(
        actual, np.argwhere(program_mask == 0).flatten(), axis=1
    )
    assert (np.unique(other_cols) == [0]).all()


def test_sixth_grade_priorities(config):
    """
    Test getting all 6th grade priorities matrix based on Brown middle school,
    bayview students, language program, and remaining priorities.
    """
    (
        brown_ms_index,
        school_ids,
        lp_schools,
        lp_mask,
    ) = generate_random_program_school_files(858, return_lp=True)
    (
        special_zipcode_inds,
        priority_1s_inds,
        prev_lp,
        lp_sibling_schools,
    ) = generate_random_student_file(school_ids)
    priority_generator = get_priority_generator(config)
    sibling = get_sibling(priority_generator)
    ctip = get_ctip(priority_generator)
    msf = get_msf(priority_generator)

    actual = priority_generator._sixth_grade_priorities()

    expected = np.zeros(
        (priority_generator.market.n, priority_generator.market.num_programs)
    )
    program_mask = lp_mask.copy()
    program_mask[brown_ms_index] = 1

    # Brown middle school priorities.
    # weights = {"sibling": 16, "bayview-to-brown": 8, "zip-94124": 4, "ctip": 2}
    weights = config["priority-weights"]["brown-ms"]
    expect_brown_ms_col = np.zeros(priority_generator.market.students.n)
    expect_brown_ms_col[special_zipcode_inds] += weights["zip-94124"]
    expect_brown_ms_col[priority_1s_inds] += weights["bayview-to-brown"]
    expect_brown_ms_col += ctip[:, 0] * weights["ctip"]
    expect_brown_ms_col += sibling[:, brown_ms_index] * weights["sibling"]
    expected[:, brown_ms_index] = expect_brown_ms_col

    # Bayview students.
    weights = config["priority-weights"]["bayview-students"]
    expected[priority_1s_inds, :] += weights["bayview-to-all"]

    # Language programs.
    weights = config["priority-weights"]["language-programs"]
    lp_program_inds = np.argwhere(lp_mask == 1).flatten()
    lp_expected = (
        weights["sibling"] * sibling
        + weights["msf"] * msf
        + weights["ctip"] * ctip
    )[:, lp_program_inds]
    lp_expected += np.repeat(
        [prev_lp * weights["lp"]], len(lp_program_inds), axis=0
    ).T
    school_to_prog = {lp_schools[i]: i for i in range(len(lp_schools))}
    for i, sibling_lps in enumerate(lp_sibling_schools):
        for sibling_lp in sibling_lps:
            lp_expected[i, school_to_prog[sibling_lp]] += weights["lp-sibling"]
    expected[:, lp_program_inds] += lp_expected

    # Remaining columns.
    weights = config["priority-weights"]["remaining"]
    all_remaining = (
        weights["sibling"] * sibling
        + weights["msf"] * msf
        + weights["ctip"] * ctip
    )
    for i in np.argwhere(program_mask == 0).flatten():
        expected[:, i] += all_remaining[:, i]

    assert (expected == actual).all()


@pytest.fixture(scope="module")
def config_ninth():
    configerator = Configerator()
    configerator.config["subconfigs"] = ["choice_model_real_match_grade9"]
    configerator.load_subconfig_by_name("choice_model_real_match_grade9")
    # Set after load_subconfig_by_name: loading a subconfig rebuilds the
    # config from the original file and would discard these overrides.
    config = configerator.config
    config["grade"] = "09"
    config["use-new-capacities"] = False
    config["year"] = YEAR
    config["paths"]["sfusd"] = TEMP_CLEANED_PAR_FOLDER
    config["paths"]["student-save"] = TEMP_STUDENT_SAVE_FOLDER
    return config


def test_brown_ms_to_hs_priorities(config_ninth):
    """
    Test getting the 9th grade priority for students from brown middle school.
    """
    _, school_ids = generate_random_program_school_files(
        grade="09",
    )
    _, brown_ms_to_hs_priorities, _, _ = generate_random_student_file(
        school_ids,
        grade="09",
    )

    priority_generator = get_priority_generator(config_ninth)
    brown_to_hs = priority_generator._get_brown_ms_to_hs_priorities()

    assert (np.unique(brown_to_hs[brown_ms_to_hs_priorities, :]) == [1]).all()

    other_rows = np.delete(brown_to_hs, brown_ms_to_hs_priorities, axis=0)
    assert (np.unique(other_rows) == [0]).all()


def test_ninth_grade_priorities(config_ninth):
    """
    Test getting all 9th grade priorities matrix with brown-ms-to-hs,
    language program, and selective high schools eligibility priorities.
    """
    (
        special_program_no,
        school_ids,
        lp_schools,
        lp_mask,
    ) = generate_random_program_school_files(
        815,
        grade="09",
        return_lp=True,
    )
    _, priority_1s_inds, prev_lp, lp_sib_sch = generate_random_student_file(
        school_ids,
        grade="09",
    )
    priority_generator = get_priority_generator(config_ninth)
    sibling = get_sibling(priority_generator)
    ctip = get_ctip(priority_generator)

    priorities = priority_generator._ninth_grade_priorities()

    weights = priority_generator.market.config["priority-weights"]
    expected = np.zeros(
        (priority_generator.market.n, priority_generator.market.num_programs)
    )

    expected[priority_1s_inds, :] += weights["brown-ms-to-hs"]
    expected += sibling * weights["sibling"] + ctip * weights["ctip"]

    # Language programs.
    lp_program_inds = np.argwhere(lp_mask == 1).flatten()
    lp_expected = np.zeros(expected[:, lp_program_inds].shape)
    lp_expected += np.repeat(
        [prev_lp * weights["lp"]], len(lp_program_inds), axis=0
    ).T
    school_to_prog = {lp_schools[i]: i for i in range(len(lp_schools))}
    for i, sibling_lps in enumerate(lp_sib_sch):
        for sibling_lp in sibling_lps:
            lp_expected[i, school_to_prog[sibling_lp]] += weights["lp-sibling"]
    expected[:, lp_program_inds] += lp_expected

    # Selective high schools eligibility for 1-indexed program.
    expected[:, special_program_no + 1] -= 500
    expected[priority_1s_inds, special_program_no + 1] += 500

    assert (expected == priorities).all()

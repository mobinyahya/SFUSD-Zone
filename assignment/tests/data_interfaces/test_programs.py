"""
Tests for data_interfaces/programs.py with randomly generated programs file.

Usage: python -m pytest tests/data_interfaces/test_programs.py -k <test_name> -s
Run all the tests in this file: python -m pytest tests/data_interfaces/test_programs.py

Last modified: November 27th, 2023
"""

import numpy as np
import pandas as pd
import pytest

from assignment.student_assignment.data_interfaces.programs import Programs

from ..utils_for_tests import (
    LP_TYPE,
    NUM_SCHOOLS,
    TEMP_PROGRAMS_FILE,
    YEAR,
    check_equal_dicts,
    delete_temp_files,
    generate_random_program_school_files,
)


@pytest.fixture(scope="module")
def config():
    return {"grade": "06", "year": YEAR, "use-new-capacities": False}


def get_programs(config):
    program_data_file = TEMP_PROGRAMS_FILE.format(config["grade"])
    programs = Programs(program_data_file, None, config)
    return programs


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    request.addfinalizer(delete_temp_files)


def generate_random_programs(grade="06", num_schools=NUM_SCHOOLS):
    """
    Generate the program file randomly, and output the related program indices
    that are useful to test the programs class.

    Params:
    - grade (str): grade of the program and school file.
    Returns:
    - school2programno: Dictionary matching school id to list of respective
        programno (indices).
    - prog_id2idx: Dictionary matching program id to programno (indices).
    - all_codes: Dictionary matching programno (indices) to program id.
    - lp_indices: Indices of language programs.
    - type2indices: Dictionary matching program type to programno (indices).
    - school_ids: List of the school ids of programs.
    """
    (
        _,
        school_ids,
        lp_schools,
        _,
        prog_id2idx,
        _,
    ) = generate_random_program_school_files(
        grade=grade,
        num_schools=num_schools,
        return_lp=True,
        return_idx=True,
        spe_school_inds=[],
    )
    # num_schools of General education programs will be added to school id lists
    # followed by language programs.
    school2programno = {id: [i + 1] for i, id in enumerate(school_ids[:num_schools])}
    for i, id in enumerate(lp_schools):
        school2programno[id].append(i + 1 + num_schools)
    prog_idx2id = {value: key for key, value in prog_id2idx.items()}
    lp_indices = [i + num_schools + 1 for i in range(len(lp_schools))]
    type2indices = {"GE": list(range(1, num_schools + 1)), LP_TYPE: lp_indices}
    return (
        school2programno,
        prog_id2idx,
        prog_idx2id,
        lp_indices,
        type2indices,
        school_ids,
    )


def test_program_setup(config):
    """
    Test _set_up_programno to set up program indices matching program id to
    1-indexed indices and program codes matching program indices to ids.
    """
    out = generate_random_programs()
    indices, codes = out[1], out[2]
    programs = get_programs(config)
    assert indices == programs.indices
    assert codes == programs.codes


def test_language_program_indices(config):
    """
    Test getting list of indices of all language pathways programs.
    """
    lp_indices = generate_random_programs()[3]
    programs = get_programs(config)
    actual = programs.language_program_indices()
    assert set(actual) == set(lp_indices)


def test_school_to_indices(config):
    """
    Test getting dictionary of school ids to list of all indices of programs in
    that school.
    """
    school2programno = generate_random_programs()[0]
    programs = get_programs(config)
    actual = programs.school_to_indices
    check_equal_dicts(actual, school2programno)


def test_program_type_to_indices(config):
    """
    Test getting dictionary matching program types to list of all indices of
    programs of that type.
    """
    type2indices = generate_random_programs()[4]
    programs = get_programs(config)
    actual = programs.program_type_to_indices
    check_equal_dicts(actual, type2indices)


def test_citywide_language_program_indices(config):
    """
    Test getting list of indices of all citywide language pathways programs from
    the programs object with randomly selected school ids as citywide schools.
    """
    school2programno, _, _, lp_indices, _, school_ids = generate_random_programs()
    programs = get_programs(config)
    lp_indices = set(lp_indices)
    citywide_school_ids = np.random.choice(school_ids, 30, replace=False)
    prog_indx = [prog for x in citywide_school_ids for prog in school2programno[x]]
    lp_prog_indx = [x for x in prog_indx if x in lp_indices]

    actual = programs.citywide_language_program_indices(citywide_school_ids)
    assert set(actual) == set(lp_prog_indx)


def test_program_rows_are_sorted_by_programno(tmp_path):
    program_file = tmp_path / "programs.csv"
    pd.DataFrame(
        {
            "programno": [3, 1, 2],
            "program_id": ["303-GE-KG", "101-GE-KG", "202-GE-KG"],
            "school_id": [303, 101, 202],
            "program_type": ["CB", "GE", "SN"],
            "capacity": [30, 10, 20],
        }
    ).to_csv(program_file, index=False)

    programs = Programs(program_file, None, {"grade": "KG"})

    assert programs.program_df["program_id"].tolist() == [
        "101-GE-KG",
        "202-GE-KG",
        "303-GE-KG",
    ]
    assert programs.school.tolist() == [101, 202, 303]
    assert programs.program_type.tolist() == ["GE", "SN", "CB"]
    assert programs.capacity.tolist() == [10, 20, 30]
    assert programs.indices == {
        "101-GE-KG": 1,
        "202-GE-KG": 2,
        "303-GE-KG": 3,
    }


@pytest.mark.parametrize("duplicate_column", ["program_id", "programno"])
def test_duplicate_program_identities_are_fatal(tmp_path, duplicate_column):
    program_file = tmp_path / "programs.csv"
    data = {
        "programno": [1, 2],
        "program_id": ["101-GE-KG", "202-GE-KG"],
    }
    data[duplicate_column][1] = data[duplicate_column][0]
    pd.DataFrame(data).to_csv(program_file, index=False)

    with pytest.raises(ValueError, match=f"duplicate {duplicate_column}"):
        Programs(program_file, None, {"grade": "KG"})


def test_sparse_program_numbers_are_normalized_and_track_source_columns(tmp_path):
    program_file = tmp_path / "programs.csv"
    pd.DataFrame(
        {
            "programno": [4, 1],
            "program_id": ["404-GE-KG", "101-GE-KG"],
            "program_type": ["GE", "GE"],
        }
    ).to_csv(program_file, index=False)

    programs = Programs(program_file, None, {"grade": "KG"})

    assert programs.program_df["program_id"].tolist() == [
        "101-GE-KG",
        "404-GE-KG",
    ]
    assert programs.program_df["programno"].tolist() == [1, 2]
    np.testing.assert_array_equal(programs.only_keep_cols, [0, 3])

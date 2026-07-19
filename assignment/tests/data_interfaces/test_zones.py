"""
Pytests for data_interfaces/zones.py with randomly generated zone files.

Example Usage: python -m pytest tests/data_interfaces/test_zones.py -k <test_name> -s
Run all the tests in this file: python -m pytest tests/data_interfaces/test_zones.py

Last modified: November 16th, 2023
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from student_assignment.data_interfaces import Programs, Zones
from student_assignment.definitions import CW2AA

from ..utils_for_tests import *

# The temp files to store zone data into and delete after each test.
TEMP_FILE_NAMES = ["test_zones_temp_file.csv", "test_zones_temp_file2.csv"]
TEMP_FILE_NAME = TEMP_FILE_NAMES[0]  # additional field for easier access.

GRADE = "KG"
PROGRAM_PREFIX = "-GE-" + GRADE


@pytest.fixture(scope="module")
def aa_config():
    config = {}
    config["zone-building-blocks"] = "attendance_area"
    config["grade"] = GRADE
    return config


@pytest.fixture(scope="module")
def bg_config():
    config = {}
    config["zone-building-blocks"] = "block_group"
    config["grade"] = GRADE
    return config


@pytest.fixture(scope="module")
def home_based_config():
    config = {}
    config["zone-building-blocks"] = "home_based"
    config["grade"] = GRADE
    return config


@pytest.fixture(scope="module")
def random_area2zone_zone_list():
    """
    Randomly generate area (3 digit integers) to zone dictionary and zone lists
    with 13 zones, each contains randomly 4 to 6 areas similar to that in the
    zone file that we are currently using. Example with one zone: area2zone =
    {100: 0, 101: 0, 102: 0, 103: 0}, zone_list = [[100, 101, 102, 103]].

    Returns:
    - dictionary: area id to zone dictionary as {area id: zone index}.
    - list of list of int: a list of zone, each element is a list of area ids,
        each zone contains randomly 4 to 6 areas.
    """
    area2zone = {}
    zone_list = []
    used_int = set()
    for i in range(13):
        cur_areas = set()
        for _ in range(np.random.randint(4, 7)):
            cur_rand = np.random.randint(100, 1000)
            while cur_rand in used_int:
                cur_rand = np.random.randint(100, 1000)
            used_int.add(cur_rand)
            area2zone[cur_rand] = i
            cur_areas.add(cur_rand)
        zone_list.append(cur_areas)
    return area2zone, zone_list


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    request.addfinalizer(delete_temp_files)


def delete_temp_files():
    """
    Delete the temp files that we use to load data for classes.
    """
    for temp_file in TEMP_FILE_NAMES:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def generate_area2program(zone_list: [], use_random=False, count=1):
    """
    Generate a dictionary {area_id: [program for areas in the same zone]} of
    from list of zones of areas for attendance_area based zones. E.g. for zones
    [[100, 101]], The output would be {101: ["101-GE-KG", "102-GE-KG"], 102:
    ["101-GE-KG", "102-GE-KG"]}. Or, if use_random, generate a random program
    name for areas and have areas in the same zone have the same program list.

    Params:
    - zone_list: list of zones.
    - use_random: True to randomly generate program names.
    - count: number of program lists to generate, if count is 1, return the
        program list itself, otherwise return a list of program lists.
    Returns:
    - dictionary: {area id: [programs for the area]}.
    """
    used_int = set([x for zone in zone_list for x in zone])
    area2program_list = []
    for _ in range(count):
        area2program = {}
        for zone in zone_list:
            if use_random:
                programs = []
                for _ in range(np.random.randint(1, 3)):
                    cur_rand = np.random.randint(100, 1000)
                    while cur_rand in used_int:
                        cur_rand = np.random.randint(100, 1000)
                    used_int.add(cur_rand)
                    programs.append(str(cur_rand) + PROGRAM_PREFIX)
            else:
                programs = [str(area) + PROGRAM_PREFIX for area in zone]
            for area in zone:
                area2program[area] = programs
        area2program_list.append(area2program)
    if count == 1:
        return area2program_list[0]
    return area2program_list


def generate_programs_with_area2programs(
    area2programs_list: list, return_indices=False
):
    """
    Generate program class instant based on the program_ids in the given list
    of area to programs dictionaries.

    Params:
    - area2programs_list ([dict]): list of area to programs dictionaries.
    - return_indices: whether to reutrn the dictionary matching program id and
        programno.
    Returns:
    - Programs: the program class with data needed for testing the Zone class.
    - dictionary: program id to programno dictionary if return_indices is true.
    """
    # Retrieve all the program ids.
    program_ids = set()
    for area2programs in area2programs_list:
        values = [x for values in area2programs.values() for x in values]
        for value in values:
            program_ids.add(value)

    program_ids = list(program_ids)
    dists_df = pd.DataFrame(
        data={
            "programno": np.arange(len(program_ids)),
            "program_id": program_ids,
        }
    )
    dists_df.to_csv(TEMP_FILE_NAME, index=False)
    programs_obj = Programs(TEMP_FILE_NAME, "", {"grade": GRADE})

    if return_indices:
        return programs_obj, {
            program_ids[i]: i for i in range(len(program_ids))
        }
    return programs_obj


def merge_area2programs_dict(area2programs_list: []):
    """
    Merge a list of area2programs to a single dictionary.

    Params:
    - area2programs_list ([{}]): list of area2programs dictionaries to merge.
    Returns:
    - dictionary: the merged area to programs dictionary in the format
        {area id: [programs for the area]}.
    """
    area2programs_merged = {}
    for area2programs in area2programs_list:
        for key, value in area2programs.items():
            if key in area2programs_merged:
                area2programs_merged[key] += value
            else:
                area2programs_merged[key] = value

    # Remove duplicate values in each pair.
    area2programs_merged_unique_values = {}
    for key, value in area2programs_merged.items():
        area2programs_merged_unique_values[key] = list(set(value))

    return area2programs_merged_unique_values


def test_create_zone(random_area2zone_zone_list, aa_config):
    """
    Test creating attendance_area based zone from a input zone file with a
    random zone list.
    """
    expected_area2zone, expected_zone_list = random_area2zone_zone_list

    f_rows = [",".join([str(elm) for elm in row]) for row in expected_zone_list]
    with open(TEMP_FILE_NAME, "w") as f:
        f.write("\n".join(f_rows))

    z = Zones(aa_config, pd.DataFrame(data={}), None, None)

    dict_zone, zone_list = z._create_zone(TEMP_FILE_NAME)
    assert dict_zone == expected_area2zone
    assert zone_list == expected_zone_list


def test_create_zone_dictionary(aa_config):
    """
    Test creating attendance_area based zone dictionary from a input zone list.
    """
    z = Zones(aa_config, pd.DataFrame(data={}), None, None)
    actual = z._create_zone_dictionary([[1, 2, 3], [4, 5]])
    expected = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    assert actual == expected


def test_create_zone_home_based(home_based_config, random_area2zone_zone_list):
    """
    Test creating home_based zone from a input zone file with a random zone
    list. The return dictionary should matching the json input file contents.
    """
    _, zone_list = random_area2zone_zone_list
    area2program = generate_area2program(zone_list)
    with open(TEMP_FILE_NAME, "w") as f:
        json.dump(area2program, f)

    z = Zones(home_based_config, pd.DataFrame(data={}), None, None)
    student2programs, zone_list = z._create_zone(TEMP_FILE_NAME)
    assert len(zone_list) == 0

    # Transform id int to str to match the loaded content from json.
    area2program = {str(key): value for (key, value) in area2program.items()}
    check_equal_dicts(student2programs, area2program)


def test_get_area_id2program_id_dict(random_area2zone_zone_list, aa_config):
    """
    Test get_area_id2ge_program_id_dict function to create dictionary mapping
    zone_ids to list of area_id's and dictionary of area ids to eligible
    programs with random zones and areas list with attendance_area based zones.
    """
    area2zone, zone_list = random_area2zone_zone_list
    aa_schools = pd.DataFrame(
        data={"attendance_area": [x for row in zone_list for x in row]}
    )
    z = Zones(aa_config, aa_schools, None, None)
    z.area2zone = area2zone
    z.get_area_id2ge_program_id_dict()

    check_equal_dicts(z.zone2area_list, zone_list)

    ge_aa_dict_expected = generate_area2program(zone_list)
    check_equal_dicts(z.area_id2ge_program_id, ge_aa_dict_expected)


def test_set_zone_aa_dict(random_area2zone_zone_list, aa_config):
    """
    Test set_area_id2prog_list_dict function to create a dictionary mapping an
    area to the list of accessible programs with attendance_area based zones.
    The programs should include both eligible programs (programs from areas
    within the same zone) and the additional accessible programs in function
    params.
    """
    area2zone, zone_list = random_area2zone_zone_list
    aa_schools = pd.DataFrame(
        data={"attendance_area": [x for row in zone_list for x in row]}
    )

    # Generate random programs.
    area_random_programs = generate_area2program(zone_list, use_random=True)
    area2programs = generate_area2program(zone_list, use_random=False)
    programs = generate_programs_with_area2programs(
        [area2programs, area_random_programs]
    )

    with open(TEMP_FILE_NAME, "w") as f:
        f.write(str(area_random_programs))

    z = Zones(aa_config, aa_schools, programs, None)
    z.area2zone = area2zone

    lp_zone_list = [TEMP_FILE_NAME]
    z.set_area_id2prog_list_dict(lp_zone_path_list=lp_zone_list)

    assert z.lp_area_id2prog_list == area_random_programs

    expected_area2programs = merge_area2programs_dict(
        [area2programs, area_random_programs]
    )
    check_equal_dicts(z.area_id2prog_list, expected_area2programs)


def test_set_zone_aa_dict_2zones(random_area2zone_zone_list, aa_config):
    """
    Test set_area_id2prog_list_dict function with 2 input files to create a
    dictionary mapping an  area to the list of accessible programs with
    attendance_area based zones. The programs should include both eligible
    programs (programs from areas within the same zone) and the additional
    accessible programs from function params.
    """
    area2zone, zone_list = random_area2zone_zone_list
    aa_schools = pd.DataFrame(
        data={"attendance_area": [x for row in zone_list for x in row]}
    )

    area_rand_programs_list = generate_area2program(
        zone_list, use_random=True, count=2
    )
    area2programs = generate_area2program(zone_list, use_random=False)
    programs = generate_programs_with_area2programs(
        area_rand_programs_list + [area2programs]
    )

    for i, temp_file in enumerate(TEMP_FILE_NAMES):
        with open(temp_file, "w") as f:
            f.write(str(area_rand_programs_list[i]))

    z = Zones(aa_config, aa_schools, programs, None)
    z.area2zone = area2zone
    z.set_area_id2prog_list_dict(lp_zone_path_list=TEMP_FILE_NAMES)

    expected_lp_area2programs = merge_area2programs_dict(
        area_rand_programs_list
    )
    check_equal_dicts(z.lp_area_id2prog_list, expected_lp_area2programs)

    expected_area2programs = merge_area2programs_dict(
        area_rand_programs_list + [area2programs]
    )
    check_equal_dicts(z.area_id2prog_list, expected_area2programs)


def test_get_area2school_id_block_groups(bg_config):
    """
    Test _get_area2school_id function with block_group based zones to generate
    dictionaries mapping block group id to a list of school ids and school ids
    to the ids of the block group they are in.
    """
    # Faked data with 10 attendance area per block group.
    num_schools = 100
    schools_per_bg = 10
    block_groups = [int(6e10 + i // schools_per_bg) for i in range(num_schools)]
    school_ids = np.arange(100, 100 + num_schools)
    bg_schools = pd.DataFrame(
        data={"BlockGroup": block_groups}, index=school_ids
    )

    z = Zones(bg_config, bg_schools, None, None)

    expected_area2school_id = {
        6e10 + i: [100 + i * schools_per_bg + j for j in range(schools_per_bg)]
        for i in range(num_schools // schools_per_bg)
    }
    expected_school_id2area = {
        school_ids[i]: block_groups[i] for i in range(num_schools)
    }

    actual_area2school_id, actual_school_id2area = z._get_area2school_id()
    assert expected_area2school_id == actual_area2school_id
    assert actual_school_id2area == expected_school_id2area


def test_get_area2school_id_attendance_areas(aa_config):
    """
    Test _get_area2school_id function with attendance_area based zones to
    generate dictionaries mapping area id to a list of schools (represented by
    the area id itself in attendance_area based zones) and schools to area ids.
    """
    rand_aa = np.arange(100, 201)
    aa_schools = pd.DataFrame(data={"attendance_area": rand_aa})

    z = Zones(aa_config, aa_schools, None, None)
    expected_area2school_id = {x: [x] for x in rand_aa}
    expected_school_id2area = {x: x for x in rand_aa}
    # We only add city wide schools to school_id2area dict.
    expected_school_id2area.update(CW2AA)

    actual_area2school_id, actual_school_id2area = z._get_area2school_id()
    assert expected_area2school_id == actual_area2school_id
    assert expected_school_id2area == actual_school_id2area


def test_programs_for_area_id_home_based(
    home_based_config, random_area2zone_zone_list
):
    """
    Test the programs_for_area_id function to get a num_programs length
    indicator vector for an area indicating program eligibility. The vector
    should contains 1 for indices of the eligible programs and 0 otherwise.
    """
    _, zone_list = random_area2zone_zone_list
    area2program = generate_area2program(zone_list)
    programs, indices = generate_programs_with_area2programs(
        [area2program], return_indices=True
    )
    z = Zones(home_based_config, pd.DataFrame(data={}), programs, None)

    with open(TEMP_FILE_NAME, "w") as f:
        json.dump(area2program, f)
    z.set_zone(TEMP_FILE_NAME)
    z.set_area_id2prog_list_dict()

    for zone in zone_list:
        # programs_for_area_id returns an indicator vector aligned directly
        # with the 0-indexed programno (== position in `indices`).
        zone_indices = [indices[f"{area}-GE-KG"] for area in zone]
        expected_programs_vec = np.array([0 for _ in range(len(indices))])
        expected_programs_vec[zone_indices] = 1

        for area in zone:
            actual = z.programs_for_area_id(0, 0, 0, area)
            assert np.equal(expected_programs_vec, actual).all()

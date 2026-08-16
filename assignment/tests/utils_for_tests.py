"""
Util functions for the PyTests that are used across test files.

Last modified: November 27th, 2023
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

YEAR = 21
LP_TYPE = "CB"
SPE_ED_TYPE = "AF"
NUM_STUDENT = 1000
NUM_SCHOOLS = 100

YEAR_PREFIX = f"{YEAR}{YEAR + 1}"
TEST_FILES_DIR = Path(__file__).parent / "test_files"
TEMP_CLEANED_PAR_FOLDER = f"{TEST_FILES_DIR}/"
TEMP_STUDENT_SAVE_FOLDER = f"{TEST_FILES_DIR}/"
CLEANED_FOLDER_PATH = f"{TEMP_CLEANED_PAR_FOLDER}/Data/Cleaned/"
SF_ZIPCODES = list(range(94102, 94135)) + [94158]

TEMP_STUDENT_FILE = f"{CLEANED_FOLDER_PATH}student_{YEAR_PREFIX}.csv"
TEMP_PROGRAMS_FILE = CLEANED_FOLDER_PATH + "programs_{}" + f"_{YEAR_PREFIX}.csv"
TEMP_SCHOOL_FILE = (
    CLEANED_FOLDER_PATH + "schools_rehauled_{}" + f"_{YEAR_PREFIX}.csv"
)


def configure_synthetic_assignment_data(config, grade, cache_root):
    """Point a strict assignment config at the generated test tables."""
    config.pop("grade", None)
    config.pop("year", None)
    config.pop("remove-special-lps", None)
    paths = config.setdefault("paths", {})
    for key in [
        "sfusd",
        "student-save",
        "student-data",
        "program-data",
        "school-data",
        "estimate-path",
        "zone-files",
        "citywide-or-lp-zones",
    ]:
        paths.pop(key, None)
    config["data"] = {
        "scenario": "legacy",
        "overrides": {
            "roots": {"cache": str(cache_root)},
            "sources": {
                "assignment.students": {
                    "path": TEMP_STUDENT_FILE,
                    "classification": "restricted",
                },
                "assignment.programs": {
                    "path": TEMP_PROGRAMS_FILE.format(grade),
                    "classification": "internal",
                },
                "assignment.schools": {
                    "path": TEMP_SCHOOL_FILE.format(grade),
                    "classification": "internal",
                },
                "assignment.school_coordinates": {
                    "path": TEMP_SCHOOL_FILE.format(grade),
                    "classification": "internal",
                },
            },
            "filters": {
                "assignment": {
                    "year": YEAR_PREFIX,
                    "grades": [grade],
                    "student_population": "applicant",
                    "rounds": "all",
                    "special_programs": "include",
                    "capacity_profile": "default",
                    "include_mission_bay": False,
                }
            },
        },
    }
    return config


def check_equal_dicts(dict1, dict2):
    """
    Check whether two dictionaries are the same. Dictionary 2 can also be a list
    if the key in the first dictionary matches its indices. Raise assert error
    if the two dictionaries are not equal.

    Params:
    - dict1 (dict): first dictionary to compare.
    - dict2 (dict or list): second dictionary to compare.
    """
    assert len(dict1) == len(dict2)
    for key, value in dict1.items():
        assert sorted(value) == sorted(list(dict2[key]))


def delete_temp_files(out_path=TEMP_STUDENT_SAVE_FOLDER):
    """
    Delete temp files used. Do not delete the temp folder in case other files
    are stored in the same folder.
    Params:
    - out_path (str): Out folder storing the student-to-program distance files.
    """
    grades = ["06", "09"]
    files = [TEMP_STUDENT_FILE]
    for grade in grades:
        file = f"{out_path}student_program_distances_{grade}_{YEAR_PREFIX}.csv"
        files.append(file)
        files.append(TEMP_SCHOOL_FILE.format(grade))
        files.append(TEMP_PROGRAMS_FILE.format(grade))
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def generate_random_schools(num_schools=100, special_school_id=None):
    """
    Randomly generate the list of unique school ids.

    Params:
    - num_schools (int): Number of school ids to generate.
    - special_school_id (int): the specified school id to include if provided.
    Returns:
    - school_ids ([Int]): List of generated school ids.
    """
    school_ids = set()
    excluded_school_ids = {909, 999}
    if special_school_id:
        school_ids.add(special_school_id)
        num_schools -= 1

    for _ in range(num_schools):
        cur_rand = np.random.randint(100, 1000)
        while cur_rand in school_ids or cur_rand in excluded_school_ids:
            cur_rand = np.random.randint(100, 1000)
        school_ids.add(cur_rand)
    return np.array(list(school_ids))


def generate_random_student_file(
    school_ids,
    lp_schools=[],
    msf=None,
    siblings=None,
    special_zipcode=94124,
    priority_1s=np.arange(10),
    grade="06",
    student_file=TEMP_STUDENT_FILE,
    num_students=NUM_STUDENT,
):
    """
    Randomly generate students with required info for priority generator.
    Add extra student with "KG" grade to ensure that the grade column is loaded
    as string when using the file for 06 and 09 students. The student data will
    be stored in provided student file.

    Params:
    - school_ids ([Int]): list of school ids.
    - lp_schools ([Int]): list of school ids with language programs.
    - msf ([Int]): MSF data for students, randomly generate MSF is not provided.
    - sibling ([[Int]]): siblings column for students, randomly generate if
        not provided.
    - special_zipcode (int): special zipcode to include, such as 94124
        for brown middle schools.
    - priority_1s ([1 or 0]): list of 1 or 0 to set priority, such as for
        bayview_to_brown priority. Default to list of 0 to 9.
    - grade (str): grade of the students.
    - student_file (str): student file to format with grade.
    - num_students (Int): number of students to generate.
    Returns:
    - special_zipcode_inds ([Int]): List of indices for students with the
        specical zipcode.
    - priority_1s ([Int]): list of 1 or 0 to set priority, such as for
        bayview_to_brown priority. The same as input if provided, or the same
        to the default param values.
    - prev_lp ([1 or 0]): whether the student was in the previous LP.
    - lp_sibling_schools [[Int]]: list of list of LP school ids for siblings.
    """
    zipcodes = np.random.choice(SF_ZIPCODES, num_students)
    # Ensure that the special zipcode is included.
    if special_zipcode not in zipcodes:
        for i in np.random.randint(0, num_students, 5):
            zipcodes[i] = special_zipcode
    special_zipcode_inds = np.argwhere(zipcodes == special_zipcode).flatten()
    ctips = np.random.choice([0, 1], num_students + 1)
    siblings = (
        siblings
        if siblings
        else [
            np.random.choice(
                school_ids, np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
            )
            for _ in range(num_students + 1)
        ]
    )
    # Find the schools of siblings that have language programs.
    lp_schools = set(lp_schools) if len(lp_schools) else set()
    lp_sibling_schools = [
        list(set(x).intersection(lp_schools)) for x in siblings
    ]
    currentlpsiblings = [
        [f"{y}-{LP_TYPE}-{grade}" for y in x] for x in lp_sibling_schools
    ]
    # Randomly select whether the strudent was previously in any LP
    prev_lp = np.random.choice([0, 1], num_students + 1)

    priority_1_0 = np.zeros(num_students + 1)
    priority_1_0[priority_1s] = 1
    # Modern Students schema (post-2024): round-1 ranked lists and a location
    # are now mandatory. Each student ranks a few random schools as GE
    # programs. Keep serialized-list CSV cells for the shared loader parser.
    n_rows = num_students + 1
    ranked_lists = [
        list(
            np.random.choice(school_ids, np.random.randint(1, 5), replace=False)
        )
        for _ in range(n_rows)
    ]
    r1_ranked_idschool = [
        "[{}]".format(",".join(str(int(s)) for s in lst))
        for lst in ranked_lists
    ]
    r1_programs = [str(["GE" for _ in lst]) for lst in ranked_lists]
    student_df = pd.DataFrame(
        data={
            "studentno": np.arange(num_students + 1),
            "ctip1": ctips,
            "sibling": [
                "[{}]".format(",".join([str(y) for y in x])) for x in siblings
            ],
            "currentlpsibling": currentlpsiblings,
            "previous_pathway": [LP_TYPE if x == 1 else None for x in prev_lp],
            "msf": msf
            if msf
            else np.random.choice(school_ids, num_students + 1),
            "bayview_to_brown_ms": priority_1_0,
            "bayview_to_all_ms": priority_1_0,
            "brown_ms_to_hs": priority_1_0,
            "sota_ranked": priority_1_0,
            "grade": [grade for _ in range(num_students)] + ["KG"],
            "zipcode": np.append(zipcodes, [special_zipcode - 1]),
            "HOCidx1": [0.1 for _ in range(num_students)] + [None],
            "r1_ranked_idschool": r1_ranked_idschool,
            "r1_programs": r1_programs,
            "latitude": np.random.uniform(37.71, 37.81, n_rows),
            "longitude": np.random.uniform(-122.51, -122.38, n_rows),
        }
    )
    student_df.to_csv(student_file, index=False)

    # Remove the last index for KG.
    return (
        special_zipcode_inds,
        priority_1s,
        prev_lp[:-1],
        lp_sibling_schools[:-1],
    )


def generate_random_program_school_files(
    special_school_id=858,
    grade="06",
    lp_school_inds=None,
    spe_school_inds=None,
    return_lp=False,
    return_idx=False,
    program_file=TEMP_PROGRAMS_FILE,
    school_file=TEMP_SCHOOL_FILE,
    num_schools=NUM_SCHOOLS,
):
    """
    Generate a list of school ids for schools and related program files with the
    special program id at random index in program list.

    Params:
    - special_school_id (Int): school id for program to take care of, such as
        for brown middle school ("858-GE-06").
    - grade (str): grade of the program and school file.
    - lp_school_inds ([Int]): Indices for school with language program, randomly
        selected if not provided.
    - spe_school_inds ([Int]): Indices for school with special education,
        randomly selected if not provided.
    - return_lp (bool): whether to return info related to language programs.
    - return_idx (bool): whether to return info related to indices.
    - program_file (str): path to store the program file.
    - school_file (str): path to store the school file.
    Returns:
    - special_program_no: index for special program, e.g. brown middle school.
    - school_ids: the school ids for generated schools.
    - lp_schools: return if return_lp is true, represents the LP school ids.
    - lp_mask: return if return_lp is true, contains 1 for index in programs
        if the program is for LP, 0 otherwise.
    - all_indices: Return if return_idx is true. Indices matching program id to
        program indices/programno.
    - lp_school2idx: Return if return_idx is true. Indices matching lp school
        ids to program indices/programno.
    """
    school_ids = generate_random_schools(num_schools, special_school_id)

    # Assume that general education programs are available to all schools.
    program_ids = [f"{x}-GE-{grade}" for x in school_ids]
    program_types = ["GE" for _ in range(num_schools)]
    # Add addition program types to program list. "CB" is for language programs,
    # "AF" is for special education. Assume 1/3 of the school has language
    # programs and 1/3 of the schools has special education if not specified.
    lp_schools = (
        school_ids[lp_school_inds]
        if lp_school_inds is not None
        else np.random.choice(school_ids, int(num_schools / 3), replace=False)
    )
    program_types += [LP_TYPE for _ in range(len(lp_schools))]
    program_ids += [f"{x}-{LP_TYPE}-{grade}" for x in lp_schools]
    spe_ed_schools = (
        school_ids[spe_school_inds]
        if spe_school_inds is not None
        else np.random.choice(school_ids, int(num_schools / 3), replace=False)
    )
    program_types += [SPE_ED_TYPE for _ in range(len(spe_ed_schools))]
    program_ids += [f"{x}-{SPE_ED_TYPE}-{grade}" for x in spe_ed_schools]

    special_program_id = f"{special_school_id}-GE-{grade}"
    special_program_no = np.argwhere(
        np.array(program_ids) == special_program_id
    ).flatten()[0]
    program_df = pd.DataFrame(
        data={
            "programno": np.arange(1, len(program_ids) + 1),
            "program_id": program_ids,
            "school_id": list(school_ids)
            + list(lp_schools)
            + list(spe_ed_schools),
            "program_type": program_types,
            # Add r1_assigned to avoid KeyError when initializing programs object.
            "r1_assigned": [10 for _ in range(len(program_ids))],
        }
    )
    program_df.to_csv(program_file.format(grade), index=False)
    school_df = pd.DataFrame(
        data={
            "school_id": school_ids,
            "category": ["attendance_area" for _ in range(num_schools)],
            "lat": np.random.uniform(37.71, 37.81, num_schools),
            "lon": np.random.uniform(-122.51, -122.38, num_schools),
        }
    )
    school_df.to_csv(school_file.format(grade), index=False)

    out = (
        special_program_no,
        school_ids,
    )
    if return_lp:
        lp_mask = (
            [0] * num_schools
            + [1] * len(lp_schools)
            + [0] * len(spe_ed_schools)
        )
        out += (
            lp_schools,
            np.array(lp_mask),
        )
    if return_idx:
        all_indices = {id: i + 1 for i, id in enumerate(program_ids)}
        lp_school2idx = {id: i + NUM_SCHOOLS for i, id in enumerate(lp_schools)}
        out += (all_indices, lp_school2idx)

    return out

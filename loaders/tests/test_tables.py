from __future__ import annotations

import pandas as pd
import pytest

from loaders import ResolvedSource, SPECIAL_PROGRAMS
from loaders.tables import (
    load_program_records,
    load_school_records,
    load_student_records,
    normalize_grade,
    normalize_student_records,
    parse_ranked_programs,
    parse_ranked_schools,
    read_csv_source,
)


def _student_scenario(
    tmp_path,
    scenario_factory,
    frame,
    *,
    rounds="all",
    special_programs="include",
    include_mission_bay=False,
):
    path = tmp_path / "students.csv"
    frame.to_csv(path, index=False)
    registry_selection = (
        {"year": "2324", "capacity_profile": "status_quo"}
        if include_mission_bay
        else {}
    )
    return scenario_factory(
        {"assignment.students": {"path": str(path)}},
        {
            "assignment": {
                **registry_selection,
                "grades": ["KG"],
                "rounds": rounds,
                "special_programs": special_programs,
                "include_mission_bay": include_mission_bay,
            }
        },
    )


def _round_columns(round_number, schools, programs, listed=None):
    listed = list(range(1, len(schools) + 1)) if listed is None else listed
    return {
        f"r{round_number}_ranked_idschool": str(schools),
        f"r{round_number}_programs": str(programs),
        f"r{round_number}_listed_ranks": str(listed),
        f"r{round_number}_randomnumber": str(
            [round_number + index / 10 for index in range(len(schools))]
        ),
        f"r{round_number}_cohortstring": str(
            [f"c{round_number}-{index}" for index in range(len(schools))]
        ),
        f"r{round_number}_designation_randomnumber": round_number / 10,
    }


def test_safe_list_parsing_grade_normalization_and_authoritative_special_set():
    assert normalize_grade("k") == "KG"
    assert normalize_grade(6) == "06"
    assert normalize_grade("06") == "06"
    assert parse_ranked_schools("[101, '102', 103.0]") == [101, 102, 103]
    assert parse_ranked_schools("[101, , 102, ]") == [101, 102]
    assert parse_ranked_programs("['GE', 'SB']") == ["GE", "SB"]
    assert SPECIAL_PROGRAMS == {
        "AF",
        "DA",
        "DT",
        "ED",
        "MM",
        "MS",
        "SA",
        "TC",
        "AO",
    }
    with pytest.raises(ValueError, match="safely parse"):
        parse_ranked_schools("__import__('os').system('false')")
    with pytest.raises(ValueError, match="Invalid school ID"):
        parse_ranked_schools("[101, school, ]")


def test_read_csv_source_reads_one_resolved_source(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("value\n1\n", encoding="utf-8")

    loaded = read_csv_source(ResolvedSource(path=path))

    assert loaded.to_dict("records") == [{"value": 1}]


def test_round_selection_is_chronological_and_derives_first_participation(
    tmp_path, scenario_factory
):
    rows = []
    for studentno, choices in (
        (1, {1: ([], []), 2: ([20], ["GE"]), 3: ([30], ["GE"]), 4: ([40], ["SB"])}),
        (2, {1: ([50], ["GE"]), 2: ([], []), 3: ([], []), 4: ([], [])}),
        (3, {1: ([], []), 2: ([], []), 3: ([60], ["GE"]), 4: ([], [])}),
    ):
        row = {"studentno": studentno, "grade": "KG"}
        for round_number, (schools, programs) in choices.items():
            row.update(_round_columns(round_number, schools, programs))
        rows.append(row)
    students = pd.DataFrame(rows)
    students.attrs["source_rows"] = [7, 8, 9]
    students.attrs["source_row_count"] = 20
    scenario = _student_scenario(tmp_path, scenario_factory, students, rounds=[4, 1, 2])

    loaded = normalize_student_records(students, scenario, "assignment")

    assert scenario.filter("assignment", "rounds") == (1, 2, 4)
    assert loaded["studentno"].tolist() == [1, 2]
    assert loaded["first_participating_round"].tolist() == [2, 1]
    assert loaded["first_participating_round_ordinal"].tolist() == [1, 0]
    assert loaded["selected_ranked_idschool"].tolist() == [[20], [50]]
    assert loaded["selected_programs"].tolist() == [["GE"], ["GE"]]
    assert loaded["selected_listed_ranks"].tolist() == [[1], [1]]
    assert loaded["selected_randomnumber"].tolist() == [[2.0], [1.0]]
    assert loaded["selected_cohortstring"].tolist() == [["c2-0"], ["c1-0"]]
    assert loaded["selected_designation_randomnumber"].tolist() == [0.2, 0.1]
    assert "r3_ranked_idschool" not in loaded
    assert "r3_designation_randomnumber" not in loaded
    assert loaded["studentno"].is_unique
    assert loaded.attrs == {"source_rows": [7, 8], "source_row_count": 20}


def test_requested_unavailable_round_fails(tmp_path, scenario_factory):
    students = pd.DataFrame(
        {
            "studentno": [1],
            "grade": ["KG"],
            "r1_ranked_idschool": ["[20]"],
            "r1_programs": ["['GE']"],
        }
    )
    scenario = _student_scenario(tmp_path, scenario_factory, students, rounds=[1, 4])

    with pytest.raises(ValueError, match=r"rounds are absent: \[4\]"):
        normalize_student_records(students, scenario, "assignment")


@pytest.mark.parametrize("bad_identity", [None, ""])
def test_student_identities_must_be_non_null(tmp_path, scenario_factory, bad_identity):
    students = pd.DataFrame(
        {
            "studentno": [bad_identity],
            "grade": ["KG"],
            "r1_ranked_idschool": ["[20]"],
            "r1_programs": ["['GE']"],
        }
    )
    scenario = _student_scenario(tmp_path, scenario_factory, students)
    with pytest.raises(ValueError, match="missing studentno"):
        normalize_student_records(students, scenario, "assignment")


def test_student_identities_must_be_unique(tmp_path, scenario_factory):
    students = pd.DataFrame(
        {
            "studentno": [1, 1],
            "grade": ["KG", "KG"],
            "r1_ranked_idschool": ["[20]", "[30]"],
            "r1_programs": ["['GE']", "['GE']"],
        }
    )
    scenario = _student_scenario(tmp_path, scenario_factory, students)
    with pytest.raises(ValueError, match="duplicate studentno.*1"):
        normalize_student_records(students, scenario, "assignment")


@pytest.mark.parametrize(
    ("mode", "expected_students", "expected_first_rounds"),
    [
        ("include", [1, 2, 3, 4, 5], [1, 1, 2, 1, 1]),
        ("exclude_only_special", [1, 2, 3, 4], [1, 2, 2, 1]),
        ("exclude_any_special", [3], [2]),
    ],
)
def test_special_program_modes_across_selected_rounds(
    tmp_path,
    scenario_factory,
    mode,
    expected_students,
    expected_first_rounds,
):
    choices = {
        1: {1: ([10, 11], ["GE", "SA"]), 2: ([], [])},
        2: {1: ([20], ["AF"]), 2: ([21], ["GE"])},
        3: {1: ([], []), 2: ([30], ["GE"])},
        4: {1: ([40], ["GE"]), 2: ([41], ["TC"])},
        5: {1: ([50], ["MM"]), 2: ([], [])},
    }
    rows = []
    for studentno, rounds in choices.items():
        row = {"studentno": studentno, "grade": "KG"}
        for round_number, (schools, programs) in rounds.items():
            row.update(_round_columns(round_number, schools, programs))
        rows.append(row)
    students = pd.DataFrame(rows)
    scenario = _student_scenario(
        tmp_path,
        scenario_factory,
        students,
        rounds=[1, 2],
        special_programs=mode,
    )

    loaded = normalize_student_records(students, scenario, "assignment")

    assert loaded["studentno"].tolist() == expected_students
    assert loaded["first_participating_round"].tolist() == expected_first_rounds
    if mode == "exclude_only_special":
        mixed = loaded.loc[loaded["studentno"] == 1].iloc[0]
        assert mixed["r1_ranked_idschool"] == [10]
        assert mixed["r1_programs"] == ["GE"]
        assert mixed["r1_listed_ranks"] == [1]
        assert mixed["r1_randomnumber"] == [1.0]
        assert mixed["r1_cohortstring"] == ["c1-0"]


@pytest.mark.parametrize(
    ("include_mission_bay", "expected_schools", "expected_ranks"),
    [
        (True, [999, 999, 20], [1, 2, 3]),
        (False, [20], [3]),
    ],
)
def test_mission_bay_filter_keeps_all_choice_metadata_aligned(
    tmp_path,
    scenario_factory,
    include_mission_bay,
    expected_schools,
    expected_ranks,
):
    students = pd.DataFrame(
        {
            "studentno": [1],
            "grade": ["KG"],
            "r1_ranked_idschool": ["[909, 999, 20]"],
            "r1_programs": ["['GE', 'SB', 'GE']"],
            "r1_listed_ranks": ["[1, 2, 3]"],
            "r1_randomnumber": ["[.1, .2, .3]"],
            "r1_cohortstring": ["['a', 'b', 'c']"],
            "sibling": ["[909, 999, 20]"],
            "currentlpsibling": ["['909-SE-KG', '999-GE-KG', '20-CN-KG']"],
        }
    )
    scenario = _student_scenario(
        tmp_path,
        scenario_factory,
        students,
        rounds=[1],
        include_mission_bay=include_mission_bay,
    )

    loaded = normalize_student_records(students, scenario, "assignment")

    assert loaded.loc[0, "r1_ranked_idschool"] == expected_schools
    assert loaded.loc[0, "r1_listed_ranks"] == expected_ranks
    if include_mission_bay:
        assert loaded.loc[0, "sibling"] == [999, 999, 20]
        assert loaded.loc[0, "currentlpsibling"] == [
            "999-SE-KG",
            "999-GE-KG",
            "20-CN-KG",
        ]
    else:
        assert loaded.loc[0, "r1_programs"] == ["GE"]
        assert loaded.loc[0, "r1_randomnumber"] == [0.3]
        assert loaded.loc[0, "r1_cohortstring"] == ["c"]
        assert loaded.loc[0, "sibling"] == [20]
        assert loaded.loc[0, "currentlpsibling"] == ["20-CN-KG"]


def test_round_lists_must_be_paired_and_aligned(tmp_path, scenario_factory):
    unpaired = pd.DataFrame(
        {
            "studentno": [1],
            "grade": ["KG"],
            "r1_ranked_idschool": ["[10]"],
        }
    )
    scenario = _student_scenario(tmp_path, scenario_factory, unpaired)
    with pytest.raises(ValueError, match="occur in pairs"):
        load_student_records(scenario, "assignment.students")

    unequal = unpaired.assign(r1_programs="['GE']", r1_listed_ranks="[1, 2]")
    scenario = _student_scenario(tmp_path, scenario_factory, unequal)
    with pytest.raises(ValueError, match="1 ranked schools but 2 values"):
        load_student_records(scenario, "assignment.students")


def test_program_grade_special_filtering_and_capacity_overlay(
    tmp_path, scenario_factory
):
    programs = tmp_path / "programs.csv"
    capacities = tmp_path / "capacities.csv"
    pd.DataFrame(
        {
            "program_id": ["999-GE-KG", "834-SA-KG", "431-GE-06"],
            "school_id": [999, 834, 431],
            "program_type": ["GE", "SA", "GE"],
            "capacity": [0, 4, 100],
        }
    ).to_csv(programs, index=False)
    pd.DataFrame(
        {
            "program_id": ["909-GE-KG", "834-SA-KG", "431-GE-06"],
            "school_id": [909, 834, 431],
            "program_type": ["GE", "SA", "GE"],
            "capacity": [66, 5, 101],
        }
    ).to_csv(capacities, index=False)
    scenario = scenario_factory(
        {
            "assignment.programs": {"path": str(capacities)},
            "assignment.programs.catalog": {"path": str(programs)},
        },
        {
            "assignment": {
                "year": "2324",
                "grades": ["KG"],
                "special_programs": "exclude_only_special",
                "capacity_profile": "status_quo",
                "include_mission_bay": True,
            }
        },
    )

    loaded = load_program_records(scenario)

    assert loaded["program_id"].tolist() == ["999-GE-KG"]
    assert loaded["school_id"].tolist() == [999]
    assert loaded["capacity"].tolist() == [66]


def test_explicit_capacity_scenario_overrides_matching_selected_programs(
    tmp_path, scenario_factory
):
    programs = tmp_path / "programs.csv"
    capacities = tmp_path / "capacities.csv"
    pd.DataFrame(
        {
            "program_id": ["10-GE-KG", "20-CB-KG"],
            "school_id": [10, 20],
            "program_type": ["GE", "CB"],
            "capacity": [3, 4],
        }
    ).to_csv(programs, index=False)
    pd.DataFrame(
        {
            "Grade": ["K"],
            "SchNum": [10],
            "PathwayCode": ["GE"],
            "Scenario_A_Capacity": [8],
        }
    ).to_csv(capacities, index=False)
    scenario = scenario_factory(
        {
            "assignment.programs": {"path": str(programs)},
            "assignment.capacity": {"path": str(capacities)},
        },
        {"assignment": {"capacity_scenario": "A"}},
    )

    loaded = load_program_records(scenario)

    assert loaded["capacity"].tolist() == [8, 4]


def test_program_loading_rejects_unavailable_grade(tmp_path, scenario_factory):
    programs = tmp_path / "programs.csv"
    pd.DataFrame(
        {
            "program_id": ["10-GE-KG"],
            "school_id": [10],
            "program_type": ["GE"],
            "capacity": [10],
        }
    ).to_csv(programs, index=False)
    scenario = scenario_factory(
        {"assignment.programs": {"path": str(programs)}},
        {"assignment": {"grades": ["06"]}},
    )

    with pytest.raises(ValueError, match="requested grades.*06"):
        load_program_records(scenario)


def test_school_loading_applies_mission_bay_policy_and_central_alias(
    tmp_path, scenario_factory
):
    schools = tmp_path / "schools.csv"
    pd.DataFrame(
        {
            "school_id": [909, 999, 20],
            "school_name": ["Mission Bay", "Mission Bay", "Other"],
        }
    ).to_csv(schools, index=False)
    included = scenario_factory(
        {"assignment.schools": {"path": str(schools)}},
        {
            "assignment": {
                "year": "2324",
                "capacity_profile": "status_quo",
                "include_mission_bay": True,
            }
        },
        scenario_id="included",
    )
    excluded = scenario_factory(
        {"assignment.schools": {"path": str(schools)}},
        {"assignment": {"include_mission_bay": False}},
        scenario_id="excluded",
    )

    assert load_school_records(included)["school_id"].tolist() == [999, 20]
    assert load_school_records(excluded)["school_id"].tolist() == [20]

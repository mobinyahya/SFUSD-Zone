from __future__ import annotations

import pandas as pd
import pytest

from loaders import ResolvedSource, SPECIAL_PROGRAMS
from loaders.tables import (
    ATTENDANCE_AREA_FILL_ROLE,
    apply_capacity_scenario,
    fill_attendance_areas,
    load_program_records,
    load_school_records,
    load_student_records,
    normalize_grade,
    normalize_student_records,
    parse_ranked_programs,
    parse_ranked_schools,
    read_csv_source,
)


def test_mission_bay_webster_capacity_transfer_is_selected_and_conserves_seats(
    scenario_factory,
):
    scenario = scenario_factory(
        {},
        {
            "assignment": {
                "year": "2627",
                "include_mission_bay": True,
                "capacity_scenario": "mission_bay_ge_to_webster_2627",
            }
        },
    )
    programs = pd.DataFrame(
        {
            "school_id": [1731, 497, 1731],
            "program_type": ["GE", "GE", "MM"],
            "capacity": [69, 44, 5],
        }
    )
    result = apply_capacity_scenario(programs, scenario, "assignment")
    assert result["capacity"].tolist() == [0, 113, 5]
    assert result["capacity"].sum() == programs["capacity"].sum()
    assert programs["capacity"].tolist() == [69, 44, 5]

    with pytest.raises(ValueError, match="expects 69 seats"):
        apply_capacity_scenario(
            programs.assign(capacity=[68, 44, 5]), scenario, "assignment"
        )


@pytest.mark.parametrize(
    ("year", "scenario_name", "extra_seats"),
    [
        ("2324", "webster_plus_66_ge_2324", 66),
        ("2425", "webster_plus_69_ge_2425", 69),
        ("2526", "webster_plus_69_ge_2526", 69),
    ],
)
def test_webster_capacity_additions_use_no_mission_bay_programs(
    scenario_factory, year, scenario_name, extra_seats
):
    scenario = scenario_factory(
        {},
        {
            "assignment": {
                "year": year,
                "include_mission_bay": False,
                "capacity_scenario": scenario_name,
            }
        },
    )
    programs = pd.DataFrame(
        {
            "school_id": [497, 497, 500],
            "program_type": ["GE", "MM", "GE"],
            "capacity": [27, 5, 38],
        }
    )
    result = apply_capacity_scenario(programs, scenario, "assignment")
    assert result["capacity"].tolist() == [27 + extra_seats, 5, 38]
    assert result["capacity"].sum() == programs["capacity"].sum() + extra_seats
    assert programs["capacity"].tolist() == [27, 5, 38]

    with pytest.raises(ValueError, match="no Mission Bay programs"):
        apply_capacity_scenario(
            pd.concat(
                [
                    programs,
                    pd.DataFrame(
                        {"school_id": [999], "program_type": ["GE"], "capacity": [0]}
                    ),
                ],
                ignore_index=True,
            ),
            scenario,
            "assignment",
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
                "grade": "KG",
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


def test_assignment_uses_exact_frl_counts_with_student_file_fallback(
    tmp_path, scenario_factory
):
    student_path = tmp_path / "students.csv"
    estimate_path = tmp_path / "frl-counts.csv"
    pd.DataFrame(
        {
            "studentno": [1, 2, 3],
            "grade": ["KG", "KG", "KG"],
            "census_block": [1001, 1002, 1003],
            "census_blockgroup": [100, 100, 100],
            "census_tract": [10, 10, 10],
            "freelunch_prob": [0.1, 0.2, 0.3],
            "reducedlunch_prob": [0.1, 0.2, 0.3],
            "FRL Score": [0.9, 0.9, 0.9],
            "r1_ranked_idschool": ["[10]", "[10]", "[10]"],
            "r1_programs": ["['GE']", "['GE']", "['GE']"],
        }
    ).to_csv(student_path, index=False)
    pd.DataFrame(
        {
            "BlockID": [1001, 1002],
            "Not FRL": [1, 0],
            "FRLunch": [3, 0],
            "Students": [4, 0],
            "FRL Rate": [0.8, None],
        }
    ).to_csv(estimate_path, index=False)
    scenario = scenario_factory(
        {
            "assignment.students": {
                "path": str(student_path),
                "geography_vintage": "2020",
            },
            "assignment.frl_estimate": {
                "path": str(estimate_path),
                "geography_vintage": "2020",
            },
        },
        {
            "assignment": {
                "grade": "KG",
                "rounds": [1],
                "geography_vintage": "2020",
                "frl_estimate": "updated_2526",
            }
        },
    )

    loaded = load_student_records(scenario, "assignment.students")

    assert loaded["freelunch_prob"].tolist() == pytest.approx([0.75, 0.4, 0.6])
    assert loaded["reducedlunch_prob"].tolist() == [0.0, 0.0, 0.0]
    assert loaded["FRL Score"].tolist() == pytest.approx([0.75, 0.9, 0.9])


def test_frl_estimate_rejects_inconsistent_counts(tmp_path, scenario_factory):
    students = pd.DataFrame(
        {
            "studentno": [1],
            "grade": ["KG"],
            "census_block": [1001],
            "census_blockgroup": [100],
            "census_tract": [10],
            "FRL Score": [0.2],
            "r1_ranked_idschool": ["[10]"],
            "r1_programs": ["['GE']"],
        }
    )
    student_path = tmp_path / "enrolled_2122.csv"
    estimate_path = tmp_path / "bad-frl-counts.csv"
    students.to_csv(student_path, index=False)
    pd.DataFrame(
        {
            "BlockID": [1001],
            "Not FRL": [1],
            "FRLunch": [2],
            "Students": [4],
        }
    ).to_csv(estimate_path, index=False)
    scenario = scenario_factory(
        {
            "optimization.students": {
                "path": str(student_path),
                "geography_vintage": "2020",
            },
            "optimization.frl_estimate": {
                "path": str(estimate_path),
                "geography_vintage": "2020",
            },
        },
        {
            "optimization": {
                "years": ["2122"],
                "grades": ["KG"],
                "rounds": [1],
                "geography_vintage": "2020",
                "frl_estimate": "updated_2526",
            }
        },
    )

    with pytest.raises(ValueError, match=r"must equal Not FRL \+ FRLunch"):
        load_student_records(scenario, "optimization.students")


def test_students_who_rank_nothing_are_dropped_with_a_warning(
    tmp_path, scenario_factory
):
    # A student who ranks nothing cannot enter a choice market, so the table
    # drops them. From 2024-25 that is worth saying out loud: the transfer
    # years' enrolled table holds students who filed no request at all, and
    # although the converter gives each of them a list, a silent drop here
    # would shrink the population a metric is computed over without a trace.
    rows = [
        {"studentno": 1, "grade": "KG", **_round_columns(1, [20], ["GE"])},
        {"studentno": 2, "grade": "KG", **_round_columns(1, [], [])},
    ]
    students = pd.DataFrame(rows)
    scenario = _student_scenario(tmp_path, scenario_factory, students, rounds=[1])

    with pytest.warns(UserWarning, match="rank no school"):
        loaded = normalize_student_records(students, scenario, "assignment")

    assert loaded["studentno"].tolist() == [1]


def test_the_enrolled_population_drops_students_with_no_enrolled_school(
    tmp_path, scenario_factory
):
    # A blank enrolled_idschool, and SFUSD's placeholder 899, both mean
    # "enrolled nowhere". Mission Bay (999) is a real school and must survive
    # even in a run that excludes it, so the filter runs before the blanking.
    rows = [
        {
            "studentno": number,
            "grade": "KG",
            "enrolled_idschool": school,
            **_round_columns(1, [20], ["GE"]),
        }
        for number, school in [(1, 20.0), (2, None), (3, 899.0), (4, 999.0)]
    ]
    students = pd.DataFrame(rows)
    path = tmp_path / "students.csv"
    students.to_csv(path, index=False)

    def load(group, population):
        scenario = scenario_factory(
            {f"{group}.students": {"path": str(path)}},
            {
                group: {
                    "grade" if group == "assignment" else "grades": (
                        "KG" if group == "assignment" else ["KG"]
                    ),
                    "student_population": population,
                }
            },
        )
        return normalize_student_records(students.copy(), scenario, group)

    with pytest.warns(UserWarning, match="2 of 4 students in the enrolled"):
        enrolled = load("assignment", "enrolled")
    assert enrolled["studentno"].tolist() == [1, 4]
    assert enrolled.attrs["source_rows"] == [0, 3]
    # The applicant population, and optimization's enrolled population, keep
    # every row.
    assert load("assignment", "applicant")["studentno"].tolist() == [1, 2, 3, 4]
    assert load("optimization", "enrolled")["studentno"].tolist() == [1, 2, 3, 4]


def test_the_promotion_claim_is_filtered_and_aliased_like_any_program_list(
    tmp_path, scenario_factory
):
    # promote is a program-ID list, the same shape as currentlpsibling, so a
    # run that excludes Mission Bay must not be left holding a claim on a
    # school it does not have -- nor may the scalar feeder_school keep it.
    rows = [
        {
            "studentno": 1,
            "grade": "KG",
            "promote": "['909-GE-KG']",
            "feeder_school": 909,
            **_round_columns(1, [20], ["GE"]),
        },
        {
            "studentno": 2,
            "grade": "KG",
            "promote": "['20-GE-KG']",
            "feeder_school": 20,
            **_round_columns(1, [20], ["GE"]),
        },
    ]
    students = pd.DataFrame(rows)

    scenario = _student_scenario(
        tmp_path, scenario_factory, students, rounds=[1], include_mission_bay=True
    )
    loaded = normalize_student_records(students, scenario, "assignment").set_index(
        "studentno"
    )
    assert loaded.loc[1, "promote"] == ["1731-GE-KG"]
    assert loaded.loc[1, "feeder_school"] == 1731

    scenario = _student_scenario(
        tmp_path, scenario_factory, students, rounds=[1], include_mission_bay=False
    )
    loaded = normalize_student_records(students, scenario, "assignment").set_index(
        "studentno"
    )
    assert loaded.loc[1, "promote"] == []
    assert pd.isna(loaded.loc[1, "feeder_school"])
    # The student who is not at Mission Bay keeps their claim either way.
    assert loaded.loc[2, "promote"] == ["20-GE-KG"]


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
        (True, [1731, 1731, 20], [1, 2, 3]),
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
        assert loaded.loc[0, "sibling"] == [1731, 1731, 20]
        assert loaded.loc[0, "currentlpsibling"] == [
            "1731-SE-KG",
            "1731-GE-KG",
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
                "grade": "KG",
                "special_programs": "exclude_only_special",
                "capacity_profile": "status_quo",
                "include_mission_bay": True,
            }
        },
    )

    loaded = load_program_records(scenario)

    assert loaded["program_id"].tolist() == ["1731-GE-KG"]
    assert loaded["school_id"].tolist() == [1731]
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


def test_pooled_capacity_overlay_splits_seats_over_lottery_pools(
    tmp_path, scenario_factory
):
    programs = tmp_path / "programs.csv"
    capacities = tmp_path / "capacities.csv"
    pd.DataFrame(
        {
            "program_id": ["10-SE-KG", "10-SN-KG", "20-JE-KG", "20-JN-KG", "30-GE-KG"],
            "school_id": [10, 10, 20, 20, 30],
            "program_type": ["SE", "SN", "JE", "JN", "GE"],
            "capacity": [18, 26, 22, 0, 5],
        }
    ).to_csv(programs, index=False)
    pd.DataFrame(
        {
            "Grade": ["K", "K", "K"],
            "SchNum": [10, 20, 40],
            "PathwayCode": ["SE+SN", "JE+JN", "SE+SN"],
            "Scenario_A_Capacity": [88, 45, 33],
        }
    ).to_csv(capacities, index=False)
    scenario = scenario_factory(
        {
            "assignment.programs": {"path": str(programs)},
            "assignment.capacity": {"path": str(capacities)},
        },
        {"assignment": {"capacity_scenario": "A"}},
    )

    with pytest.warns(UserWarning, match=r"schools \[40\].*not modelled"):
        loaded = load_program_records(scenario)

    # 88 seats keep the 18:26 split; a pool with no seats now stays empty;
    # school 40 has no such program and is ignored.
    assert loaded["capacity"].tolist() == [36, 52, 45, 0, 5]

    pd.DataFrame(
        {
            "program_id": ["20-JE-KG", "20-JN-KG"],
            "school_id": [20, 20],
            "program_type": ["JE", "JN"],
            "capacity": [0, 0],
        }
    ).to_csv(programs, index=False)
    # No pool holds seats, so there is no proportion to follow, and none is
    # assumed.
    with (
        pytest.warns(UserWarning),
        pytest.raises(ValueError, match="School 20 JE\\+JN: Cannot split 45 seats"),
    ):
        load_program_records(scenario)


def test_capacity_scenario_categories_decide_attendance_and_citywide_schools(
    tmp_path, scenario_factory
):
    schools = tmp_path / "schools.csv"
    capacities = tmp_path / "capacities.csv"
    pd.DataFrame(
        {
            "school_id": [10, 20, 30, 999],
            "category": ["Attendance", "Citywide", "Citywide", "Attendance"],
        }
    ).to_csv(schools, index=False)
    # School 10 keeps attendance-area seats in one program only, which is
    # enough; school 20 gains some; 30 is unlisted; Mission Bay is listed under
    # its district code against the table's legacy placeholder.
    pd.DataFrame(
        {
            "Grade": ["K", "K", "K", "K", "06"],
            "SchNum": [10, 10, 20, 1731, 30],
            "PathwayCode": ["GE", "SB", "GE", "GE", "GE"],
            "Scenario_A_Capacity": [22, 44, 66, 66, 10],
            "Scenario_A_Category": [
                "Attendance",
                "Attendance",
                "Attendance",
                "Citywide",
                "Attendance",
            ],
        }
    ).to_csv(capacities, index=False)
    scenario = scenario_factory(
        {
            "assignment.schools": {"path": str(schools)},
            "assignment.capacity": {"path": str(capacities)},
        },
        {
            "assignment": {
                "year": "2324",
                "capacity_profile": "status_quo",
                "capacity_scenario": "A",
                "include_mission_bay": True,
            }
        },
    )

    loaded = load_school_records(scenario).set_index("school_id")["category"]

    assert loaded.to_dict() == {
        10: "Attendance",
        20: "Attendance",
        30: "Citywide",
        1731: "Citywide",
    }

    plain = scenario_factory(
        {"assignment.schools": {"path": str(schools)}},
        {"assignment": {"capacity_scenario": "programs"}},
        scenario_id="plain",
    )
    assert load_school_records(plain)["category"].tolist() == [
        "Attendance",
        "Citywide",
        "Citywide",
    ]


def test_capacity_scenario_categories_must_agree_per_school(tmp_path, scenario_factory):
    schools = tmp_path / "schools.csv"
    capacities = tmp_path / "capacities.csv"
    pd.DataFrame({"school_id": [10], "category": ["Attendance"]}).to_csv(
        schools, index=False
    )
    pd.DataFrame(
        {
            "SchNum": [10, 10],
            "PathwayCode": ["GE", "SB"],
            "Scenario_A_Capacity": [22, 44],
            "Scenario_A_Category": ["Attendance", "Citywide"],
        }
    ).to_csv(capacities, index=False)
    scenario = scenario_factory(
        {
            "assignment.schools": {"path": str(schools)},
            "assignment.capacity": {"path": str(capacities)},
        },
        {"assignment": {"capacity_scenario": "A"}},
    )
    with pytest.raises(ValueError, match=r"schools \[10\] more than one category"):
        load_school_records(scenario)


@pytest.mark.parametrize(
    ("grade", "year", "rows", "expected"),
    [
        # K-8 seats at grade 6 are the Main Round assignments; others unchanged.
        ("06", "1819", [(618, "GE", 90, 7), (500, "GE", 90, 7)], [7, 90]),
        # SOTA and Lowell admit every eligible applicant ...
        ("09", "1819", [(815, "GE", 50, 0), (697, "GE", 50, 0)], [10000, 10000]),
        # ... except Lowell's lottery years.
        ("09", "2223", [(815, "GE", 50, 0), (697, "GE", 50, 0)], [10000, 50]),
    ],
)
def test_grade_capacity_rules_precede_capacity_overlays(
    tmp_path, scenario_factory, grade, year, rows, expected
):
    programs = tmp_path / "programs.csv"
    pd.DataFrame(
        {
            "program_id": [f"{school}-{kind}-{grade}" for school, kind, *_ in rows],
            "school_id": [row[0] for row in rows],
            "program_type": [row[1] for row in rows],
            "capacity": [row[2] for row in rows],
            "r1_assigned": [row[3] for row in rows],
        }
    ).to_csv(programs, index=False)
    schools = tmp_path / "schools.csv"
    pd.DataFrame({"school_id": [row[0] for row in rows]}).to_csv(schools, index=False)
    sources = {
        "assignment.programs": {"path": str(programs)},
        "assignment.schools": {"path": str(schools)},
    }
    scenario = scenario_factory(sources, {"assignment": {"year": year, "grade": grade}})
    loaded = load_program_records(scenario)
    assert loaded["capacity"].tolist() == expected
    if grade == "06":
        assert loaded["r2_capacity"].tolist()[0] == 7

    overlay = tmp_path / "capacity.csv"
    pd.DataFrame(
        {
            "SchNum": [rows[0][0]],
            "PathwayCode": [rows[0][1]],
            "Scenario_A_Capacity": [3],
        }
    ).to_csv(overlay, index=False)
    overlaid = load_program_records(
        scenario_factory(
            {**sources, "assignment.capacity": {"path": str(overlay)}},
            {"assignment": {"year": year, "grade": grade, "capacity_scenario": "A"}},
        )
    )
    assert overlaid["capacity"].tolist() == [3, *expected[1:]]


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
        {"assignment": {"grade": "06"}},
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

    assert load_school_records(included)["school_id"].tolist() == [1731, 20]
    assert load_school_records(excluded)["school_id"].tolist() == [20]


@pytest.mark.parametrize(
    ("source_ids", "kept_name"),
    [
        ([909, 999, 20], "placeholder 999"),
        ([909, 1731, 999, 20], "district 1731"),
    ],
)
def test_school_loading_prefers_canonical_then_recent_mission_bay_rows(
    tmp_path, scenario_factory, source_ids, kept_name
):
    names = {909: "placeholder 909", 999: "placeholder 999", 1731: "district 1731"}
    schools = tmp_path / "schools.csv"
    pd.DataFrame(
        {
            "school_id": source_ids,
            "school_name": [names.get(school, "Other") for school in source_ids],
        }
    ).to_csv(schools, index=False)
    scenario = scenario_factory(
        {"assignment.schools": {"path": str(schools)}},
        {
            "assignment": {
                "year": "2324",
                "capacity_profile": "status_quo",
                "include_mission_bay": True,
            }
        },
    )

    loaded = load_school_records(scenario).set_index("school_id")
    assert loaded.index.tolist() == [1731, 20]
    assert loaded.loc[1731, "school_name"] == kept_name


def _with_planned_school(scenario, *, include=False, **site):
    """``scenario`` with a test Mandarin Immersion entry, optionally included.

    The flag is set after loading because ``load_scenario`` rejects including
    the real entry, whose site and pool split are still unknown.
    """
    import dataclasses

    definition = {
        "school_id": 400,
        "school_name": "Mandarin Immersion TK-8",
        "category": "Citywide",
        "lat": None,
        "lon": None,
        "programs": {"KG": {"ME": 27, "MN": 39}},
        **site,
    }
    filters = {
        group: {**values, "include_mandarin_immersion": include}
        for group, values in scenario.filters.items()
    }
    return dataclasses.replace(
        scenario,
        filters=filters,
        planned_schools={"mandarin_immersion": definition},
    )


def test_planned_school_is_absent_by_default(tmp_path, scenario_factory):
    schools = tmp_path / "schools.csv"
    pd.DataFrame({"school_id": [10], "category": ["Attendance"]}).to_csv(
        schools, index=False
    )
    sources = {"assignment.schools": {"path": str(schools)}}
    plain = _with_planned_school(scenario_factory(sources, {"assignment": {}}))
    assert load_school_records(plain)["school_id"].tolist() == [10]
    assert "planned_schools" not in plain.source_manifest([])


def test_included_planned_school_joins_school_and_program_tables(
    tmp_path, scenario_factory, monkeypatch
):
    import loaders.geography

    schools = tmp_path / "schools.csv"
    programs = tmp_path / "programs.csv"
    capacities = tmp_path / "capacities.csv"
    pd.DataFrame(
        {"school_id": [10], "category": ["Attendance"], "lat": [37.7], "lon": [-122.4]}
    ).to_csv(schools, index=False)
    pd.DataFrame(
        {
            "program_id": ["10-GE-KG"],
            "school_id": [10],
            "program_type": ["GE"],
            "capacity": [44],
            "programno": [1],
        }
    ).to_csv(programs, index=False)
    pd.DataFrame(
        {
            "Grade": ["K"],
            "SchNum": [400],
            "PathwayCode": ["ME+MN"],
            "Scenario_A_Capacity": [88],
        }
    ).to_csv(capacities, index=False)

    def fake_match(frame, *args, row_mask=None, **kwargs):
        located = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        located[row_mask] = 60750607001053
        return pd.DataFrame(
            {
                "Block": located,
                "BlockGroup": located // 1000,
                "Tract": located // 10000,
            }
        )

    monkeypatch.setattr(loaders.geography, "match_points_to_census", fake_match)
    sources = {
        "assignment.schools": {"path": str(schools)},
        "assignment.programs": {"path": str(programs)},
        "assignment.capacity": {"path": str(capacities)},
    }
    scenario = _with_planned_school(
        scenario_factory(sources, {"assignment": {"capacity_scenario": "A"}}),
        include=True,
        lat=37.77,
        lon=-122.39,
    )

    loaded_schools = load_school_records(scenario).set_index("school_id")
    assert loaded_schools.loc[400, "category"] == "Citywide"
    assert loaded_schools.loc[400, "Block"] == 60750607001053

    loaded = load_program_records(scenario).set_index("program_id")
    # The registry's 27:39 split carries the scenario's 88 seats.
    assert loaded.loc[["400-ME-KG", "400-MN-KG"], "capacity"].tolist() == [36, 52]
    assert loaded["programno"].tolist() == [1, 2, 3]
    assert (
        scenario.source_manifest([])["planned_schools"]["assignment"][0]["school_id"]
        == 400
    )


def test_programs_carry_attendance_area_seats_from_the_scenario_or_the_school(
    tmp_path, scenario_factory
):
    schools = tmp_path / "schools.csv"
    programs = tmp_path / "programs.csv"
    capacities = tmp_path / "capacities.csv"
    pd.DataFrame(
        {
            "school_id": [10, 20, 30],
            "category": ["Attendance", "Citywide", "Attendance"],
        }
    ).to_csv(schools, index=False)
    pd.DataFrame(
        {
            "program_id": ["10-GE-KG", "10-SE-KG", "10-SN-KG", "20-GE-KG", "30-GE-KG"],
            "school_id": [10, 10, 10, 20, 30],
            "program_type": ["GE", "SE", "SN", "GE", "GE"],
            "capacity": [40, 10, 10, 50, 22],
        }
    ).to_csv(programs, index=False)
    # School 10's GE seats are split: 44 attendance-area, 22 citywide.
    pd.DataFrame(
        {
            "Grade": ["K", "K", "K"],
            "SchNum": [10, 10, 20],
            "PathwayCode": ["GE", "SE+SN", "GE"],
            "Scenario_A_Capacity": [66, 44, 66],
            "Scenario_A_AttendanceAreaCapacity": [44, 0, 0],
            "Scenario_A_Category": ["Attendance", "Attendance", "Citywide"],
        }
    ).to_csv(capacities, index=False)
    sources = {
        "assignment.schools": {"path": str(schools)},
        "assignment.programs": {"path": str(programs)},
        "assignment.capacity": {"path": str(capacities)},
    }

    overlaid = load_program_records(
        scenario_factory(sources, {"assignment": {"capacity_scenario": "A"}})
    ).set_index("program_id")
    # School 30 is not in the scenario, so its GE seats fall back to the rule.
    assert overlaid["capacity"].tolist() == [66, 22, 22, 66, 22]
    assert overlaid["attendance_area_capacity"].tolist() == [44, 0, 0, 0, 22]

    plain = load_program_records(
        scenario_factory(sources, {"assignment": {}}, scenario_id="plain")
    )
    assert plain["attendance_area_capacity"].tolist() == [40, 0, 0, 0, 22]

def _fill_scenario(tmp_path, scenario_factory, *, bind=True):
    """Two unlabelled polygons: A holds school 20, B holds schools 30 and 31."""
    import geopandas as gpd
    from shapely.geometry import box

    areas = tmp_path / "areas.shp"
    gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326"
    ).to_file(areas)
    schools = tmp_path / "schools.csv"
    pd.DataFrame(
        {
            "school_id": [20, 30, 31, 40],
            "category": ["Attendance", "Attendance", "Attendance", "Citywide"],
            "lat": [0.5, 0.5, 0.6, 0.4],
            "lon": [0.5, 1.5, 1.6, 0.4],
        }
    ).to_csv(schools, index=False)
    sources = {"assignment.schools": {"path": str(schools)}}
    if bind:
        sources[ATTENDANCE_AREA_FILL_ROLE] = {"path": str(areas)}
    return scenario_factory(sources, {"assignment": {}}, scenario_id=f"fill{bind}")


def test_blank_attendance_areas_are_filled_from_the_map_only(
    tmp_path, scenario_factory
):
    students = pd.DataFrame(
        {
            "studentno": [1, 2, 3, 4, 5],
            # 1 lives in polygon B but the district says 20: the record wins.
            "idschoolattendance": [20, None, 30, None, None],
            "latitude": [0.5, 0.5, 0.5, 5.0, 0.5],
            "longitude": [1.5, 1.5, 1.5, 5.0, 0.5],
        }
    )
    scenario = _fill_scenario(tmp_path, scenario_factory)

    with pytest.warns(UserWarning, match="Filled 2 blank .*; 1 remain blank"):
        filled = fill_attendance_areas(students, scenario)

    # B holds 30 and 31; only 30 is anyone's district area, so B is 30.
    assert filled["idschoolattendance"].tolist()[:3] == [20, 30, 30]
    assert pd.isna(filled["idschoolattendance"].iloc[3])
    assert filled["idschoolattendance"].iloc[4] == 20
    assert filled["attendance_area_source"].fillna("none").tolist() == [
        "district",
        "map",
        "district",
        "none",
        "map",
    ]
    # Caches keyed on the students alone must see the map and the schools
    # that name its polygons.
    assert {ATTENDANCE_AREA_FILL_ROLE, "assignment.schools"} <= set(
        scenario.source_manifest(["assignment.students"])["sources"]
    )

    unbound = _fill_scenario(tmp_path, scenario_factory, bind=False)
    pd.testing.assert_frame_equal(fill_attendance_areas(students, unbound), students)
    assert (
        "assignment.schools"
        not in unbound.source_manifest(["assignment.students"])["sources"]
    )


def test_attendance_area_fill_rejects_an_unresolvable_polygon(
    tmp_path, scenario_factory
):
    # Nobody's district area is 30 or 31, so polygon B cannot be named.
    students = pd.DataFrame(
        {
            "studentno": [1, 2],
            "idschoolattendance": [20, None],
            "latitude": [0.5, 0.5],
            "longitude": [0.5, 1.5],
        }
    )
    with pytest.raises(ValueError, match=r"holds schools \[30, 31\]"):
        fill_attendance_areas(students, _fill_scenario(tmp_path, scenario_factory))

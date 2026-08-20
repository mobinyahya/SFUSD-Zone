from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from assignment.student_assignment.choice_ranks import ASSIGNMENT_SCHEMA_VERSION
from assignment.student_assignment.evaluation.match_evaluator import MatchEvaluator


@dataclass
class StudentsStub:
    student_data: pd.DataFrame
    round_participation: np.ndarray


def _listed_assignments(
    studentno,
    programno,
    programcodes,
    submitted_rank,
    designation,
    mechanism_rank=None,
):
    submitted_rank = pd.Series(submitted_rank, dtype=float)
    if mechanism_rank is None:
        mechanism_rank = submitted_rank
    mechanism_rank = pd.Series(mechanism_rank, dtype=float)
    return pd.DataFrame(
        {
            "assignment_schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "studentno": studentno,
            "programno": programno,
            "programcodes": programcodes,
            "rank_basis": "listed",
            "submitted_rank": submitted_rank,
            "utility_rank": np.nan,
            "rank": submitted_rank,
            "mechanism_rank": mechanism_rank,
            "designation": designation,
            "In-Zone Rank": mechanism_rank,
        }
    )


def test_basic_report_preserves_benchmark_contract():
    student_data = pd.DataFrame(
        {
            "studentno": [1, 2, 3],
            "freelunch_prob": [0.9, 0.2, 0.1],
            "reducedlunch_prob": [0.0, 0.0, 0.0],
            "resolved_ethnicity": [
                "Black or African American",
                "White",
                "Asian",
            ],
            "SES_category": [3, 1, 1],
            "grade": ["KG", "KG", "KG"],
            "census_blockgroup": [10, 10, 20],
        }
    ).set_index("studentno")
    assignments = _listed_assignments(
        [1, 2, 3],
        [1, 2, 0],
        ["101-GE-KG", "202-GE-KG", pd.NA],
        [1, 3, None],
        [0, 1, 0],
        [1, 2, None],
    ).set_index("studentno")
    distances = pd.DataFrame(
        {"101-GE-KG": [0.25, 2.0, 1.0], "202-GE-KG": [3.0, 4.0, 2.0]},
        index=pd.Index([1, 2, 3], name="studentno"),
    )
    students = StudentsStub(student_data, np.ones((3, 1), dtype=int))

    metrics = MatchEvaluator(students, assignments, distances).eval_assignment_basic()

    assert metrics["Distance Av"] == 2.125
    assert metrics["Unassigned"] == 1 / 3
    assert metrics["Top 1 choice"] == 0.5
    assert metrics["Top 1 choice numerator"] == 1
    assert metrics["Top 1 choice denominator"] == 2
    assert np.isscalar(metrics["Dist >= 3, Rank >= 5"])
    assert metrics["Dissimilarity SES3"] == 0.25
    assert "BG Cohesion (3)" in metrics
    assert metrics["# Racial majority schools"] == 2
    assert len(metrics) == 70


def test_full_report_covers_metric_families_without_mutating_inputs(
    tmp_path, monkeypatch
):
    students = pd.DataFrame(
        {
            "studentno": range(1, 7),
            "r1_ranked_idschool": [
                "[101]",
                "[101]",
                "[202]",
                "[202]",
                "[101]",
                "[202]",
            ],
            "r1_programs": ["['GE']"] * 6,
            "r1_listed_ranks": ["[1]", "[2]", "[1]", "[4]", "[5]", "[3]"],
            "census_block": [11, 12, 13, 14, 15, 16],
            "latitude": [37.70, 37.71, 37.72, 37.73, 37.74, 37.75],
            "longitude": [-122.40, -122.41, -122.42, -122.43, -122.44, -122.45],
            "freelunch_prob": [0.9, 0.8, 0.2, 0.1, 0.6, 0.3],
            "reducedlunch_prob": [None, 0.0, 0.0, 0.0, 0.0, 0.0],
            "resolved_ethnicity": [
                "Black or African American",
                "Hispanic/Latino",
                "Asian",
                "White",
                "Two or More Races",
                "Decline to State",
            ],
            "median_hh_income": [50_000, 70_000, 120_000, 140_000, 80_000, 110_000],
            "ctip1": [1, 0, 1, 0, 1, 0],
            "zipcode": [94110, "94110", 94111, 94111, 94113, 94112],
            "idschoolattendance": [101, 101, 202, 202, 101, 202],
        }
    )
    assignments = _listed_assignments(
        range(1, 7),
        [1, 1, 2, 2, 0, 2],
        [
            "101-GE-KG",
            "101-GE-KG",
            "202-GE-KG",
            "202-GE-KG",
            "",
            "202-GE-KG",
        ],
        [1, 2, 1, 4, None, 3],
        [0, 1, 0, 0, 0, 0],
    )
    original_assignments = assignments.copy(deep=True)
    distance_cache = pd.DataFrame(
        {
            "101-GE-KG": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "202-GE-KG": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        },
        index=pd.Index(range(1, 7), name="studentno"),
    )
    programs = pd.DataFrame(
        {
            "program_id": ["101-GE-KG", "202-GE-KG", "202-SA-KG"],
            "school_id": [101, 202, 202],
            "program_type": ["GE", "GE", "SA"],
            "capacity": [10, 10, 2],
            "programno": [1, 2, 3],
        }
    )
    schools = pd.DataFrame(
        {
            "school_id": [101, 202],
            "school_name": ["Alpha", "Beta"],
            "category": ["Attendance", "Citywide"],
            "lat": [37.70, 37.75],
            "lon": [-122.40, -122.45],
        }
    )
    program_path = tmp_path / "programs.csv"
    school_path = tmp_path / "schools.csv"
    programs.to_csv(program_path, index=False)
    schools.to_csv(school_path, index=False)

    evaluator = MatchEvaluator(
        students,
        assignments,
        first_round=True,
        no_special_program=True,
        program_file=program_path,
        schools_latlon_path=school_path,
        distance_cache=distance_cache,
    )
    assert evaluator.student_data["assignment_dist"].iloc[:4].tolist() == [
        1.0,
        2.0,
        9.0,
        10.0,
    ]
    assert pd.isna(evaluator.student_data["assignment_dist"].iloc[4])
    assert evaluator.student_data["assignment_dist"].iloc[5] == 12.0
    evaluator.student_data["assignment_dist"] = evaluator.student_data["studentno"].map(
        {1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: np.nan, 6: 2.0}
    )
    assert evaluator.programs["program_id"].tolist() == ["101-GE-KG", "202-GE-KG"]
    metrics = evaluator.eval_assignment_full()

    expected = {
        "Distance Av (All Assigned)",
        "#Schools above 10% district FRL",
        "#GE programs above +15% district FRL (Non-Designated)",
        "Dissimilarity (High FRL)",
        "Black exposure to FRL prob",
        "Prop Top 3 choice Non-Designated (Black)",
        "Nb assigned students (All Assigned) to GE program",
        "#Students in schools above +15% district FRL (ET (2024))",
        "# Racial majority schools",
        "count_students_Pacific Islander",
        "utilization_GE",
    }
    assert expected <= set(metrics.index)
    assert len(metrics) >= 900
    assert (
        metrics[
            "#GE programs that have exactly 0 African American or Pacific Islander students"
        ]
        == 1
    )
    assert (
        metrics[
            "#GE programs that have 1-4 African American or Pacific Islander students"
        ]
        == 1
    )
    assert metrics["#Students in schools above +15% district FRL (ET (2024))"] == 0
    assert metrics["#GE programs above +15% district FRL (Non-Designated)"] == 1
    assert metrics["Prop Top 1 choice (All Assigned)"] == 2 / 5
    assert metrics["Prop Top 1 choice (All Assigned) numerator"] == 2
    assert metrics["Prop Top 1 choice (All Assigned) denominator"] == 5
    assert metrics["Prop Top 3 choice (All Assigned)"] == 3 / 5
    assert metrics["Prop Top 1 choice (All Students)"] == 2 / 6
    assert metrics["Prop Top 1 choice (All Students) numerator"] == 2
    assert metrics["Prop Top 1 choice (All Students) denominator"] == 6
    assert metrics["Top 3 in-zone choice (All Assigned)"] == 3 / 5
    assert metrics["Variance of rank (All Assigned)"] == pytest.approx(1.7)
    assert metrics["Prop Distance > 3 and designated (All Assigned)"] == 1 / 5
    assert (
        metrics["Prop Distance > 3 and Top 3 choice, non-designated (All Assigned)"]
        == 2 / 5
    )
    assert metrics["Prop Distance > 3 and non-designated (All Assigned)"] == 3 / 5
    pd.testing.assert_frame_equal(assignments, original_assignments)

    prepare_calls = []
    prepare_aggregates = evaluator._prepare_full_report_aggregates

    def track_prepare(student_data, **kwargs):
        prepare_calls.append((len(student_data), kwargs))
        return prepare_aggregates(student_data, **kwargs)

    monkeypatch.setattr(evaluator, "_prepare_full_report_aggregates", track_prepare)
    reports = evaluator.eval_aggregate_metric_reports("config-a")
    assert set(reports) == {"program", "zip_code", "attendance_area", "citywide"}
    expected_contexts = (
        1
        + pd.to_numeric(students["zipcode"]).nunique()
        + students["idschoolattendance"].nunique()
    )
    assert len(prepare_calls) == expected_contexts
    assert sum(
        call_kwargs.get("include_program_report_stats", False)
        for _, call_kwargs in prepare_calls
    ) == 1

    prepare_calls.clear()
    citywide_only = evaluator.eval_aggregate_metric_reports(
        "config-a", include_local_metrics=False
    )
    assert set(citywide_only) == {"citywide"}
    assert len(prepare_calls) == 1
    assert not prepare_calls[0][1].get("include_program_report_stats", False)

    program_metrics = reports["program"]
    assert program_metrics["config_name"].eq("config-a").all()
    alpha = program_metrics.set_index("program_id").loc["101-GE-KG"]
    assert alpha["school_name"] == "Alpha"
    assert alpha["school_category"] == "Attendance"
    assert alpha["mean_travel_dist_assigned"] == 4
    assert alpha["mean_travel_dist_designated"] == 4
    assert alpha["percent_designated"] == 0.5
    assert alpha["program_utilization"] == 0.2

    zip_metrics = reports["zip_code"]
    assert set(zip_metrics.columns) == {"config_name", "zip_code", *metrics.index}
    zip_94110 = zip_metrics.set_index("zip_code").loc[94110]
    assert zip_94110["Tot Nb Students (Round 1)"] == 2
    assert zip_94110["Tot Nb Assigned (Round 1)"] == 2
    zip_94113 = zip_metrics.set_index("zip_code").loc[94113]
    assert zip_94113["Tot Nb Assigned (Round 1)"] == 0
    assert zip_94113["Unassigned"] == 1

    attendance_metrics = reports["attendance_area"]
    assert set(attendance_metrics.columns) == {
        "config_name",
        "attendance_area",
        *metrics.index,
    }
    attendance_101 = attendance_metrics.set_index("attendance_area").loc[101]
    assert attendance_101["Tot Nb Students (Round 1)"] == 3
    assert attendance_101["#Unassigned"] == 1
    citywide = reports["citywide"]
    assert citywide.loc[0, "config_name"] == "config-a"
    assert set(citywide.columns) == {"config_name", *metrics.index}

    legacy_assignments = assignments[
        [
            "studentno",
            "programno",
            "programcodes",
            "rank",
            "In-Zone Rank",
        ]
    ].copy()
    legacy_assignments["rank"] = 1
    legacy_evaluator = MatchEvaluator(
        students,
        legacy_assignments,
        first_round=True,
        no_special_program=True,
        program_file=program_path,
        schools_latlon_path=school_path,
    )
    assert not legacy_evaluator.student_data["designation"].any()
    assert legacy_evaluator.student_data["submitted_rank"].tolist()[:4] == [
        1,
        2,
        1,
        4,
    ]
    with pytest.raises(ValueError, match="cached student-program distance matrix"):
        legacy_evaluator.eval_aggregate_metric_reports("legacy")


def test_program_report_uses_exact_program_assignments_and_schema():
    students = pd.DataFrame(
        {
            "studentno": range(1, 9),
            "census_block": range(11, 19),
            "latitude": [37.70, 37.71, 37.72, 37.73, 37.74, 37.75, 37.76, 37.77],
            "longitude": [
                -122.40,
                -122.41,
                -122.42,
                -122.43,
                -122.44,
                -122.45,
                -122.46,
                -122.47,
            ],
            "freelunch_prob": [0.9, 0.8, 0.7, 0.0, 0.0, 0.1, 0.4, 0.6],
            "reducedlunch_prob": [0.0] * 8,
            "resolved_ethnicity": [
                "Asian",
                "Black or African American",
                "Hispanic/Latino",
                "White",
                "Other",
                "Pacific Islander",
                "Decline to State",
                "Two or More Races",
            ],
            "median_hh_income": [
                50_000,
                70_000,
                80_000,
                140_000,
                130_000,
                110_000,
                90_000,
                100_000,
            ],
            "ctip1": [1, 0, 1, 0, 1, 0, 1, 0],
            "zipcode": [94110, 94110, 94110, 94110, 94111, 94111, 94113, 94110],
            "idschoolattendance": [101, 101, 101, 101, 101, 202, 202, 101],
        }
    )
    assignments = _listed_assignments(
        range(1, 9),
        [1, 1, 1, 2, 2, 3, 4, 0],
        [
            "101-X-KG",
            "101-X-KG",
            "101-X-KG",
            "101-Y-KG",
            "101-Y-KG",
            "202-Z-KG",
            "202-U-KG",
            "",
        ],
        [1, 2, 3, 1, 4, 2, 2, None],
        [0, 1, 0, 1, 0, 0, 1, 0],
    )
    assignments["overage_seat"] = [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    programs = pd.DataFrame(
        {
            "program_id": [
                "101-X-KG",
                "101-Y-KG",
                "202-Z-KG",
                "202-U-KG",
                "202-V-KG",
            ],
            "programno": [1, 2, 3, 4, 5],
            "school_id": [101, 101, 202, 202, 202],
            "program_type": ["GE", "GE", "GE", "SA", "GE"],
            "capacity": [2, 4, 4, 0, 3],
        }
    )
    schools = pd.DataFrame(
        {
            "school_id": [101, 202],
            "school_name": ["Alpha", "Beta"],
            "category": ["Attendance", "Citywide"],
            "lat": [37.70, 37.75],
            "lon": [-122.40, -122.45],
        }
    )
    distances = pd.DataFrame(
        {
            "101-X-KG": [1.0, 9.0, 5.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "101-Y-KG": [1.0, 2.0, 3.0, 7.0, 9.0, 6.0, 7.0, 8.0],
            "202-Z-KG": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 7.0, 8.0],
            "202-U-KG": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 8.0],
            "202-V-KG": [1.0] * 8,
        },
        index=pd.Index(range(1, 9), name="studentno"),
    )
    evaluator = MatchEvaluator(
        students,
        assignments,
        program_data=programs,
        schools_data=schools,
        distance_cache=distances,
        overscribe_aa=True,
    )

    assert evaluator.student_data["assigned school"].tolist() == [
        101,
        101,
        101,
        101,
        101,
        202,
        202,
        0,
    ]
    assert evaluator.student_data["assigned school"].dtype.kind in "iu"
    assert evaluator.student_data.loc[:5, "programtype"].eq("GE").all()

    reports = evaluator.eval_aggregate_metric_reports("config-a")
    assert set(reports) == {"program", "zip_code", "attendance_area", "citywide"}
    metrics = reports["citywide"].iloc[0]
    for rank, numerator in [(1, 1), (2, 2), (3, 3)]:
        name = f"Prop Top {rank} choice (All Students)"
        assert metrics[name] == numerator / 8
        assert metrics[f"{name} numerator"] == numerator
        assert metrics[f"{name} denominator"] == 8

    ge_labels = {
        "#GE programs above +10% district FRL",
        "Proportion of GE programs above +10% district FRL",
        "#GE programs above +15% district FRL",
        "Proportion of GE programs above +15% district FRL",
        "#GE programs above +15% district FRL (Non-Designated)",
        "#GE programs below -10% district FRL",
        "Proportion of GE programs below -10% district FRL",
        "#GE programs below -15% district FRL",
        "Proportion of GE programs below -15% district FRL",
        "AALPI in GE programs with +10% FRL",
        "AALPI in GE programs with +15% FRL",
        "AALPI in GE programs with -10% FRL",
        "AALPI in GE programs with -15% FRL",
    }
    assert ge_labels <= set(metrics.index)
    assert metrics["#Schools above 10% district FRL"] == 0
    assert metrics["#GE programs above +10% district FRL"] == 1
    assert metrics["Proportion of GE programs above +10% district FRL"] == 1 / 3
    assert metrics["#GE programs above +15% district FRL"] == 1
    assert metrics["Proportion of GE programs above +15% district FRL"] == 1 / 3
    assert metrics["#GE programs above +15% district FRL (Non-Designated)"] == 1
    assert metrics["#GE programs below -10% district FRL"] == 2
    assert metrics["Proportion of GE programs below -10% district FRL"] == 2 / 3
    assert metrics["#GE programs below -15% district FRL"] == 2
    assert metrics["Proportion of GE programs below -15% district FRL"] == 2 / 3
    assert metrics["AALPI in GE programs with +10% FRL"] == 2 / 3
    assert metrics["AALPI in GE programs with +15% FRL"] == 2 / 3
    assert metrics["AALPI in GE programs with -10% FRL"] == 1 / 3
    assert metrics["AALPI in GE programs with -15% FRL"] == 1 / 3

    old_labels = {
        "#GE above +10% district FRL",
        "GE above +10% district FRL",
        "#GE above +15% district FRL",
        "GE above +15% district FRL",
        "#GE Schools above +15% district FRL (Non-Designated)",
        "#GE below -10% district FRL",
        "GE below -10% district FRL",
        "#GE below -15% district FRL",
        "GE below -15% district FRL",
        "AALPI in GE with +10% FLR",
        "AALPI in GE with +15% FLR",
        "AALPI in GE with -10% FLR",
        "AALPI in GE with -15% FLR",
    }
    assert old_labels.isdisjoint(metrics.index)
    assert not any(
        name.startswith("Proportion of students in top ")
        and name.endswith("(All Students)")
        for name in metrics.index
    )

    demographic_columns = [
        column
        for slug in [
            "asian",
            "black",
            "decline_to_state",
            "hispanic",
            "other",
            "pacific_islander",
            "two_or_more_races",
            "white",
        ]
        for column in [
            f"non_designated_{slug}_students",
            f"designated_{slug}_students",
        ]
    ]
    program_metrics = reports["program"]
    assert program_metrics.columns.tolist() == [
        "config_name",
        "program_id",
        "school_id",
        "school_name",
        "school_category",
        "program_type",
        "capacity",
        "assigned",
        "designated",
        "mean_travel_dist_assigned",
        "mean_travel_dist_designated",
        "percent_designated",
        "frl_assigned",
        "frl_designated",
        "frl_non_designated",
        "program_utilization",
        "overage",
        "underage",
        "prop_top_1",
        "prop_top_2",
        "prop_top_3",
        *demographic_columns,
    ]
    assert program_metrics["program_id"].tolist() == programs["program_id"].tolist()
    by_program = program_metrics.set_index("program_id")

    program_x = by_program.loc["101-X-KG"]
    assert program_x["school_id"] == 101
    assert program_x["school_name"] == "Alpha"
    assert program_x["school_category"] == "Attendance"
    assert program_x["program_type"] == "GE"
    assert program_x["capacity"] == 2
    assert program_x["assigned"] == 3
    assert program_x["designated"] == 1
    assert program_x["mean_travel_dist_assigned"] == 5
    assert program_x["mean_travel_dist_designated"] == 9
    assert program_x["percent_designated"] == 1 / 3
    assert program_x["frl_assigned"] == pytest.approx(0.8)
    assert program_x["frl_designated"] == pytest.approx(0.8)
    assert program_x["frl_non_designated"] == pytest.approx(0.8)
    assert program_x["program_utilization"] == 1.5
    assert program_x["overage"] == 0.5
    assert program_x["underage"] == 0
    assert program_x["prop_top_1"] == 1 / 3
    assert program_x["prop_top_2"] == 1 / 3
    assert program_x["prop_top_3"] == 2 / 3
    assert program_x["non_designated_asian_students"] == 1
    assert program_x["designated_black_students"] == 1
    assert program_x["non_designated_hispanic_students"] == 1

    program_y = by_program.loc["101-Y-KG"]
    assert program_y["assigned"] == 2
    assert program_y["designated"] == 1
    assert program_y["mean_travel_dist_assigned"] == 8
    assert program_y["mean_travel_dist_designated"] == 7
    assert program_y["percent_designated"] == 0.5
    assert program_y[
        ["frl_assigned", "frl_designated", "frl_non_designated"]
    ].eq(0).all()
    assert program_y["program_utilization"] == 0.5
    assert program_y["overage"] == 0
    assert program_y["underage"] == 0.5
    assert program_y[["prop_top_1", "prop_top_2", "prop_top_3"]].eq(0).all()
    assert program_y["non_designated_other_students"] == 1
    assert program_y["designated_white_students"] == 1

    program_z = by_program.loc["202-Z-KG"]
    assert program_z["assigned"] == 1
    assert program_z["designated"] == 0
    assert program_z["frl_assigned"] == pytest.approx(0.1)
    assert pd.isna(program_z["frl_designated"])
    assert program_z["frl_non_designated"] == pytest.approx(0.1)
    assert program_z["program_utilization"] == 0.25
    assert program_z["overage"] == 0
    assert program_z["underage"] == 0.75
    assert program_z["prop_top_1"] == 0
    assert program_z["prop_top_2"] == 1
    assert program_z["prop_top_3"] == 1
    assert program_z["non_designated_pacific_islander_students"] == 1

    zero_capacity_program = by_program.loc["202-U-KG"]
    assert zero_capacity_program["assigned"] == 1
    assert zero_capacity_program["designated"] == 1
    assert zero_capacity_program["mean_travel_dist_assigned"] == 11
    assert zero_capacity_program["mean_travel_dist_designated"] == 11
    assert zero_capacity_program["percent_designated"] == 1
    assert zero_capacity_program["frl_assigned"] == pytest.approx(0.4)
    assert zero_capacity_program["frl_designated"] == pytest.approx(0.4)
    assert pd.isna(zero_capacity_program["frl_non_designated"])
    assert zero_capacity_program[
        ["program_utilization", "overage", "underage"]
    ].isna().all()
    assert zero_capacity_program[["prop_top_1", "prop_top_2", "prop_top_3"]].eq(
        0
    ).all()
    assert zero_capacity_program["designated_decline_to_state_students"] == 1

    unassigned_program = by_program.loc["202-V-KG"]
    assert unassigned_program["assigned"] == 0
    assert unassigned_program["designated"] == 0
    assert unassigned_program[
        ["frl_assigned", "frl_designated", "frl_non_designated"]
    ].isna().all()
    assert unassigned_program[demographic_columns].eq(0).all()
    assert (
        unassigned_program[
            [
                "mean_travel_dist_assigned",
                "mean_travel_dist_designated",
                "percent_designated",
                "prop_top_1",
                "prop_top_2",
                "prop_top_3",
            ]
        ]
        .isna()
        .all()
    )
    assert unassigned_program["program_utilization"] == 0
    assert unassigned_program["overage"] == 0
    assert unassigned_program["underage"] == 1
    assert {
        "school_frl_enrolled",
        "mean_travel_dist_enrolled",
        "mean_student_choice_assigned",
        "percent_assigned",
        "school_utilization",
    }.isdisjoint(program_metrics.columns)

    assert metrics["overage"] == pytest.approx(1 / 8)
    zip_metrics = reports["zip_code"].set_index("zip_code")
    assert zip_metrics.loc[94110, "overage"] == pytest.approx(1 / 5)
    assert zip_metrics.loc[94111, "overage"] == 0
    attendance_metrics = reports["attendance_area"].set_index("attendance_area")
    assert attendance_metrics.loc[101, "overage"] == pytest.approx(1 / 6)
    assert attendance_metrics.loc[202, "overage"] == 0

    evaluator.overscribe_aa = False
    evaluator.update_assignments(assignments)
    assert evaluator.eval_assignment_full()["overage"] == 0


def test_ge_district_frl_metrics_use_all_assigned_students():
    students = pd.DataFrame(
        {
            "studentno": [1, 2, 3],
            "census_block": [11, 12, 13],
            "latitude": [37.70, 37.71, 37.72],
            "longitude": [-122.40, -122.41, -122.42],
            "freelunch_prob": [0.7, 0.3, 1.0],
            "reducedlunch_prob": [0.0, 0.0, 0.0],
            "resolved_ethnicity": [
                "Black or African American",
                "White",
                "Asian",
            ],
            "median_hh_income": [50_000, 100_000, 70_000],
            "ctip1": [1, 0, 1],
        }
    )
    assignments = _listed_assignments(
        [1, 2, 3],
        [1, 2, 3],
        ["101-X-KG", "101-Y-KG", "202-SA-KG"],
        [1, 1, 1],
        [0, 0, 0],
    )
    programs = pd.DataFrame(
        {
            "program_id": ["101-X-KG", "101-Y-KG", "202-SA-KG"],
            "programno": [1, 2, 3],
            "school_id": [101, 101, 202],
            "program_type": ["GE", "GE", "SA"],
            "capacity": [10, 10, 10],
        }
    )
    schools = pd.DataFrame(
        {
            "school_id": [101, 202],
            "lat": [37.70, 37.72],
            "lon": [-122.40, -122.42],
        }
    )

    metrics = MatchEvaluator(
        students,
        assignments,
        program_data=programs,
        schools_data=schools,
    ).eval_assignment_full()

    assert metrics["#GE programs above +10% district FRL"] == 0
    assert metrics["#GE programs above +15% district FRL"] == 0
    assert metrics["#GE programs above +15% district FRL (Non-Designated)"] == 0
    assert metrics["AALPI in GE programs with +10% FRL"] == 0


def test_folder_discovery_only_returns_assignment_csvs(tmp_path):
    from assignment.scripts.analysis.analyze_trends import _collect_csv_files

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assignment = run_dir / "assignment.csv"
    pd.DataFrame(
        {
            "studentno": [1],
            "programno": [1],
            "programcodes": ["101-GE-KG"],
            "rank": [1],
            "designation": [0],
            "In-Zone Rank": [1],
        }
    ).to_csv(assignment, index=False)
    pd.DataFrame({"studentno": [1], "101-GE-KG": [1.0]}).to_csv(
        run_dir / "utility_matrix.csv", index=False
    )

    paths, is_single = _collect_csv_files({"folder": str(run_dir)})

    assert paths == [str(assignment)]
    assert not is_single


def _minimal_full_inputs(tmp_path):
    students = pd.DataFrame(
        {
            "studentno": [1, 2],
            "r1_ranked_idschool": ["[101]", "[202]"],
            "r1_programs": ["['GE']", "['GE']"],
            "census_block": [11, 12],
            "latitude": [37.7, 37.8],
            "longitude": [-122.4, -122.5],
            "freelunch_prob": [0.5, 0.5],
            "reducedlunch_prob": [0.0, 0.0],
            "resolved_ethnicity": ["Asian", "White"],
            "median_hh_income": [80_000, 120_000],
            "ctip1": [1, 0],
        }
    )
    assignments = _listed_assignments(
        [1, 2],
        [1, 2],
        ["101-GE-KG", "202-GE-KG"],
        [1, 1],
        [0, 0],
    )
    programs = pd.DataFrame(
        {
            "program_id": ["101-GE-KG", "202-GE-KG"],
            "programno": [1, 2],
            "school_id": [101, 202],
            "program_type": ["GE", "GE"],
            "capacity": [10, 10],
        }
    )
    schools = pd.DataFrame(
        {
            "school_id": [101, 202],
            "school_name": ["Alpha", "Beta"],
            "category": ["Attendance", "Citywide"],
            "lat": [37.7, 37.8],
            "lon": [-122.4, -122.5],
        }
    )
    program_path = tmp_path / "programs.csv"
    school_path = tmp_path / "schools.csv"
    programs.to_csv(program_path, index=False)
    schools.to_csv(school_path, index=False)
    return students, assignments, program_path, school_path


def _make_full_evaluator(students, assignments, program_path, school_path):
    return MatchEvaluator(
        students,
        assignments,
        first_round=True,
        program_file=program_path,
        schools_latlon_path=school_path,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate", "duplicate school_id"),
        ("blank_name", "missing or blank school_name"),
        ("missing_category", "missing or blank category"),
        ("unassigned_program_school", "absent from school metadata"),
    ],
)
def test_program_report_requires_complete_school_metadata(tmp_path, case, message):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    programs = pd.read_csv(program_path)
    schools = pd.read_csv(school_path)
    if case == "duplicate":
        schools = pd.concat([schools, schools.iloc[[0]]], ignore_index=True)
    elif case == "blank_name":
        schools.loc[0, "school_name"] = " "
    elif case == "missing_category":
        schools.loc[0, "category"] = np.nan
    else:
        programs.loc[len(programs)] = ["303-GE-KG", 3, 303, "GE", 10]

    evaluator = MatchEvaluator(
        students,
        assignments,
        first_round=True,
        program_data=programs,
        schools_data=schools,
    )

    with pytest.raises(ValueError, match=message):
        evaluator.eval_assignment_metrics_by_program()


def test_full_evaluator_can_replace_assignments_without_reloading_sources(tmp_path):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    evaluator = _make_full_evaluator(
        students, assignments, program_path, school_path
    )
    updated_assignments = assignments.copy()
    updated_assignments.loc[
        0,
        [
            "programno",
            "programcodes",
            "submitted_rank",
            "rank",
            "mechanism_rank",
            "In-Zone Rank",
        ],
    ] = [2, "202-GE-KG", np.nan, np.nan, 2, 2]

    evaluator.update_assignments(updated_assignments)
    fresh_evaluator = _make_full_evaluator(
        students,
        updated_assignments,
        program_path,
        school_path,
    )

    pd.testing.assert_frame_equal(
        evaluator.student_data,
        fresh_evaluator.student_data,
    )


@pytest.mark.parametrize("coverage_error", ["missing", "duplicate", "extra"])
def test_full_evaluator_requires_exact_assignment_coverage(tmp_path, coverage_error):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    if coverage_error == "missing":
        assignments = assignments.iloc[[0]].copy()
    elif coverage_error == "duplicate":
        assignments = pd.concat([assignments, assignments.iloc[[0]]], ignore_index=True)
    else:
        extra = assignments.iloc[[0]].copy()
        extra["studentno"] = 3
        assignments = pd.concat([assignments, extra], ignore_index=True)

    with pytest.raises(ValueError, match=coverage_error):
        _make_full_evaluator(students, assignments, program_path, school_path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("programcodes", "999-GE-KG", "unknown program IDs"),
        ("programno", 2, "does not match programcodes"),
        ("programcodes", "", "require programcodes"),
    ],
)
def test_full_evaluator_validates_positive_program_assignments(
    tmp_path, column, value, message
):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    assignments.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        _make_full_evaluator(students, assignments, program_path, school_path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("rank", -1),
        ("rank", np.inf),
        ("In-Zone Rank", 1.5),
        ("designation", -1),
        ("designation", 0.5),
        ("designation", 2),
    ],
)
def test_full_evaluator_rejects_invalid_metric_values(tmp_path, column, value):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    assignments[column] = assignments[column].astype(object)
    assignments.loc[0, column] = value

    with pytest.raises(ValueError, match=f"invalid {column}"):
        _make_full_evaluator(students, assignments, program_path, school_path)


def test_full_evaluator_rejects_designated_unassigned_student(tmp_path):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    assignments.loc[0, ["programno", "programcodes", "designation"]] = [0, "", 1]
    assignments.loc[
        0,
        [
            "submitted_rank",
            "rank",
            "mechanism_rank",
            "In-Zone Rank",
        ],
    ] = np.nan

    with pytest.raises(ValueError, match="Unassigned students cannot be designated"):
        _make_full_evaluator(students, assignments, program_path, school_path)


def test_full_evaluator_uses_simulator_program_number_namespace(tmp_path):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    programs = pd.read_csv(program_path)
    programs["programno"] = [1, 4]
    programs.to_csv(program_path, index=False)

    evaluator = _make_full_evaluator(students, assignments, program_path, school_path)

    assert evaluator._program_number_by_id == {
        "101-GE-KG": 1,
        "202-GE-KG": 2,
    }


def test_full_evaluator_requires_assigned_school_location(tmp_path):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    schools = pd.read_csv(school_path)
    schools.loc[schools["school_id"] == 101, "lat"] = np.nan
    schools.to_csv(school_path, index=False)

    with pytest.raises(ValueError, match="without known locations"):
        _make_full_evaluator(students, assignments, program_path, school_path)


def test_first_round_uses_nonempty_parsed_lists(tmp_path):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    students.loc[1, "r1_ranked_idschool"] = "[]"
    assignments = assignments.iloc[[0]].copy()

    evaluator = _make_full_evaluator(students, assignments, program_path, school_path)

    assert evaluator.student_data["studentno"].tolist() == [1]
    assert evaluator.student_data["r1_ranked_idschool"].tolist() == [[101]]


def test_first_round_rejects_no_participants_and_invalid_lists(tmp_path):
    students, assignments, program_path, school_path = _minimal_full_inputs(tmp_path)
    students["r1_ranked_idschool"] = "[]"
    with pytest.raises(ValueError, match="at least one student"):
        _make_full_evaluator(students, assignments, program_path, school_path)

    students.loc[0, "r1_ranked_idschool"] = "not a list"
    with pytest.raises(ValueError, match="invalid serialized list"):
        _make_full_evaluator(students, assignments, program_path, school_path)


def test_metric_aggregation_preserves_nan_and_zeroes_one_observation_std():
    from assignment.scripts.analysis.analyze_trends import _aggregate_metrics

    one = _aggregate_metrics(
        [pd.Series({"finite": 2.0, "missing": np.nan})], is_single_file=False
    )
    assert one["std"]["finite"] == 0
    assert np.isnan(one["mean"]["missing"])
    assert np.isnan(one["std"]["missing"])

    multiple = _aggregate_metrics(
        [
            pd.Series({"finite": 2.0, "missing": np.nan}),
            pd.Series({"finite": 4.0, "missing": 3.0}),
        ],
        is_single_file=False,
    )
    assert multiple["mean"]["finite"] == 3
    assert np.isnan(multiple["mean"]["missing"])
    assert np.isnan(multiple["std"]["missing"])


def test_trend_worker_propagates_input_failures(tmp_path):
    from assignment.scripts.analysis.analyze_trends import _evaluate_csv_worker

    missing = str(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        _evaluate_csv_worker((23, missing, missing, missing, missing, None))


def test_folder_discovery_rejects_partial_assignment_csv(tmp_path):
    from assignment.scripts.analysis.analyze_trends import _collect_csv_files

    pd.DataFrame({"studentno": [1], "programno": [1], "rank": [1]}).to_csv(
        tmp_path / "partial.csv", index=False
    )

    with pytest.raises(ValueError, match="appears to be an assignment"):
        _collect_csv_files({"folder": str(tmp_path)})


def test_trend_analysis_rejects_no_runs(tmp_path, monkeypatch):
    from assignment.scripts.analysis import analyze_trends

    config_path = tmp_path / "analysis.yaml"
    config_path.write_text(f"output_dir: {tmp_path / 'output'}\nruns: []\n")
    monkeypatch.setattr("sys.argv", ["analyze_trends.py", "--config", str(config_path)])

    with pytest.raises(ValueError, match="No runs"):
        analyze_trends.main()


def test_legacy_assignment_aggregation_preserves_iteration_nan(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from assignment.student_assignment.evaluation import evaluate_assignments

    class EvaluatorStub:
        def __init__(self, students, assignments, distances):
            self.value = assignments["value"].iloc[0]

        def eval_assignment_basic(self):
            return pd.Series({"metric": self.value})

    monkeypatch.setattr(evaluate_assignments, "MatchEvaluator", EvaluatorStub)
    pd.DataFrame({"studentno": [1], "value": [1.0]}).to_csv(
        tmp_path / "policy_iteration0.csv", index=False
    )
    pd.DataFrame({"studentno": [1], "value": [np.nan]}).to_csv(
        tmp_path / "policy_iteration1.csv", index=False
    )
    market = SimpleNamespace(students=SimpleNamespace(distance_data=pd.DataFrame()))
    table_path = tmp_path / "summary.csv"

    evaluator = evaluate_assignments.EvaluateAssignments(market, iterations=2)
    evaluator.evaluate_results(tmp_path, ["policy"], table_path)

    summary = pd.read_csv(table_path)
    policy = summary[summary["Assignment"] == "policy"].iloc[0]
    assert pd.isna(policy["metric"])
    assert policy["Iterations"] == 2


def test_legacy_assignment_analysis_fails_on_missing_iteration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from assignment.student_assignment.evaluation import evaluate_assignments

    class EvaluatorStub:
        def __init__(self, students, assignments, distances):
            pass

        def eval_assignment_basic(self):
            return pd.Series({"metric": 1.0})

    monkeypatch.setattr(evaluate_assignments, "MatchEvaluator", EvaluatorStub)
    pd.DataFrame({"studentno": [1], "value": [1.0]}).to_csv(
        tmp_path / "policy_iteration0.csv", index=False
    )
    market = SimpleNamespace(students=SimpleNamespace(distance_data=pd.DataFrame()))
    table_path = tmp_path / "summary.csv"

    evaluator = evaluate_assignments.EvaluateAssignments(market, iterations=2)
    with pytest.raises(FileNotFoundError):
        evaluator.evaluate_results(tmp_path, ["policy"], table_path)
    assert not table_path.exists()

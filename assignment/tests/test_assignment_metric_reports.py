from dataclasses import dataclass

import numpy as np
import pandas as pd

from assignment.student_assignment.evaluation.match_evaluator import MatchEvaluator


@dataclass
class StudentsStub:
    student_data: pd.DataFrame
    round_participation: np.ndarray


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
    assignments = pd.DataFrame(
        {
            "studentno": [1, 2, 3],
            "programno": [1, 2, 0],
            "programcodes": ["101-GE-KG", "202-GE-KG", pd.NA],
            "rank": [1, 3, 4],
            "designation": [0, 1, 0],
            "In-Zone Rank": [1, 2, 4],
        }
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
    assert np.isscalar(metrics["Dist >= 3, Rank >= 5"])
    assert metrics["Dissimilarity SES3"] == 0.25
    assert "BG Cohesion (3)" in metrics
    assert metrics["# Racial majority schools"] == 2
    assert len(metrics) == 48


def test_full_report_covers_metric_families_without_mutating_inputs(tmp_path):
    students = pd.DataFrame(
        {
            "studentno": range(1, 7),
            "r1_ranked_idschool": [101, 101, 202, 202, 101, 202],
            "r1_programs": ["['GE']"] * 6,
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
        }
    )
    assignments = pd.DataFrame(
        {
            "studentno": range(1, 7),
            "programno": [1, 1, 2, 2, 0, 2],
            "programcodes": [
                "101-GE-KG",
                "101-GE-KG",
                "202-GE-KG",
                "202-GE-KG",
                "",
                "202-GE-KG",
            ],
            "rank": [1, 2, 1, 4, 5, 3],
            "designation": [0, 1, 0, 0, 0, 0],
            "In-Zone Rank": [1, 2, 1, 4, 5, 3],
        }
    )
    original_assignments = assignments.copy(deep=True)
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
    )
    evaluator.student_data["assignment_dist"] = evaluator.student_data["studentno"].map(
        {1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: np.nan, 6: 2.0}
    )
    assert evaluator.programs["program_id"].tolist() == ["101-GE-KG", "202-GE-KG"]
    metrics = evaluator.eval_assignment_full()

    expected = {
        "Distance Av (All Assigned)",
        "#Schools above 10% district FRL",
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
    assert metrics["Prop Distance > 3 and designated (All Assigned)"] == 1 / 5
    assert (
        metrics["Prop Distance > 3 and Top 3 choice, non-designated (All Assigned)"]
        == 2 / 5
    )
    assert metrics["Prop Distance > 3 and non-designated (All Assigned)"] == 3 / 5
    pd.testing.assert_frame_equal(assignments, original_assignments)

    legacy_evaluator = MatchEvaluator(
        students,
        assignments.drop(columns=["designation", "In-Zone Rank"]),
        first_round=True,
        no_special_program=True,
        program_file=program_path,
        schools_latlon_path=school_path,
    )
    assert not legacy_evaluator.student_data["designation"].any()
    pd.testing.assert_series_equal(
        legacy_evaluator.student_data["In-Zone Rank"],
        legacy_evaluator.student_data["rank"],
        check_names=False,
    )


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

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

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
            "r1_ranked_idschool": [
                "[101]",
                "[101]",
                "[202]",
                "[202]",
                "[101]",
                "[202]",
            ],
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
    assert metrics["Prop Top 1 choice (All Assigned)"] == 2 / 5
    assert metrics["Prop Top 3 choice (All Assigned)"] == 3 / 5
    assert metrics["Top 3 in-zone choice (All Assigned)"] == 3 / 5
    assert metrics["Variance of rank (All Assigned)"] == pytest.approx(1.7)
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
        }
    )
    assignments = pd.DataFrame(
        {
            "studentno": [1, 2],
            "programno": [1, 2],
            "programcodes": ["101-GE-KG", "202-GE-KG"],
            "rank": [1, 1],
            "designation": [0, 0],
            "In-Zone Rank": [1, 1],
        }
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
    pd.DataFrame({"value": [1.0]}).to_csv(tmp_path / "policy_iteration0.csv")
    pd.DataFrame({"value": [np.nan]}).to_csv(tmp_path / "policy_iteration1.csv")
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
    pd.DataFrame({"value": [1.0]}).to_csv(tmp_path / "policy_iteration0.csv")
    market = SimpleNamespace(students=SimpleNamespace(distance_data=pd.DataFrame()))
    table_path = tmp_path / "summary.csv"

    evaluator = evaluate_assignments.EvaluateAssignments(market, iterations=2)
    with pytest.raises(FileNotFoundError):
        evaluator.evaluate_results(tmp_path, ["policy"], table_path)
    assert not table_path.exists()

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import analyze_zone_policy_outcomes as outcomes  # noqa: E402
from analysis.evaluate_zone_subconfig_matches import SUBCONFIGS  # noqa: E402


def assignment(
    programno: list[int],
    rank: list[int],
    designation: list[int],
) -> outcomes.AssignmentData:
    return outcomes.AssignmentData(
        programno=np.asarray(programno, dtype=np.int16),
        rank=np.asarray(rank, dtype=np.int16),
        designation=np.asarray(designation, dtype=np.int8),
        assigned_utility=None,
    )


def test_defaults_use_fresh_roots_and_current_policy_order():
    assert outcomes.DEFAULT_CHOICE_ROOT.name == (
        "zone_subconfigs_rerun_20260811T043406Z_choice_model_25"
    )
    assert outcomes.DEFAULT_REAL_ROOT.name == (
        "zone_subconfigs_rerun_20260811T043406Z_real_preferences_no_special_25"
    )
    assert tuple(outcomes.SUBCONFIGS) == tuple(SUBCONFIGS)
    assert len(outcomes.SUBCONFIGS) == 19


def test_generated_config_validation_warns_for_unfiltered_fake_top(tmp_path, caplog):
    base_config = {
        "data": {
            "scenario": "legacy",
            "overrides": {
                "sources": {
                    "assignment.students": {
                        "path": str(tmp_path / "students.csv"),
                        "classification": "restricted",
                    },
                    "assignment.programs": {
                        "path": str(tmp_path / "programs.csv"),
                        "classification": "internal",
                    },
                    "assignment.schools": {
                        "path": str(tmp_path / "schools.csv"),
                        "classification": "internal",
                    },
                    "assignment.estimate": {
                        "path": str(tmp_path / "estimates.csv"),
                        "classification": "restricted",
                    },
                },
                "filters": {
                    "assignment": {
                        "year": "2324",
                        "grades": ["KG"],
                        "student_population": "applicant",
                        "rounds": [1],
                        "special_programs": "exclude_any_special",
                        "capacity_profile": "status_quo",
                        "include_mission_bay": True,
                    }
                },
            },
        },
        "iterations": {"start": 0, "end": 25},
        "random-seed": 2023,
        "r1-only": True,
        "utility-model": {"enable": True, "list-length": 10},
        "ties-options": ["MTB"],
    }
    (tmp_path / "simulation_config.yaml").write_text(
        "subconfigs:\n" + "".join(f"- {label}\n" for label in SUBCONFIGS),
        encoding="utf-8",
    )
    for label in SUBCONFIGS:
        policy_dir = tmp_path / label
        policy_dir.mkdir()
        config = {**base_config, "subconfig-name": label}
        (policy_dir / "policy_config.generated.yaml").write_text(
            json.dumps(config), encoding="utf-8"
        )

    path, _ = outcomes.validate_generated_configs(tmp_path, "choice_model")

    assert path == tmp_path / outcomes.BASELINE_POLICY / "policy_config.generated.yaml"
    assert caplog.messages == [
        f"Fake-top results may be less meaningful without _4 filtering and AA "
        f"oversubscription: {tmp_path / label / 'policy_config.generated.yaml'}"
        for label in outcomes.FAKE_TOP_POLICIES
    ]


def test_inverse_preference_ranks_uses_full_exact_program_order():
    preferences = np.array([[3, 1, 2], [2, 3, 1]])

    result = outcomes.inverse_preference_ranks(preferences, 3)

    np.testing.assert_array_equal(result, [[2, 3, 1], [3, 1, 2]])
    with pytest.raises(ValueError, match="permutations"):
        outcomes.inverse_preference_ranks(np.array([[1, 1, 2]]), 3)


def test_real_raw_ranks_use_selected_exact_school_program():
    students = pd.DataFrame(
        {
            "selected_ranked_idschool": [[20, 10], [20, 10]],
            "selected_programs": [["GE", "GE"], ["SE", "GE"]],
        },
        index=pd.Index([101, 102], name="studentno"),
    )
    program_indices = {
        "10-GE-KG": 1,
        "20-GE-KG": 2,
        "20-SE-KG": 3,
    }

    result = outcomes.build_real_raw_preference_ranks(
        students,
        program_indices,
        "KG",
    )

    np.testing.assert_array_equal(result[0], [2, 1, 0])
    np.testing.assert_array_equal(result[1], [2, 0, 1])


def test_real_raw_order_retains_removed_program_position_and_top_school():
    students = pd.DataFrame(
        {
            "selected_ranked_idschool": [[30, 10]],
            "selected_programs": [["SA", "GE"]],
        },
        index=pd.Index([101], name="studentno"),
    )

    result = outcomes.build_real_raw_preference_data(
        students,
        {"10-GE-KG": 1},
        "KG",
        valid_school_ids={10, 30},
    )

    np.testing.assert_array_equal(result.ranks, [[2]])
    np.testing.assert_array_equal(result.applicant_mask, [True])
    np.testing.assert_array_equal(result.top_school_ids, [30])
    assert result.unloaded_program_entries == 1


def test_read_validated_assignment_checks_order_schema_and_program_mapping(tmp_path):
    path = tmp_path / "assignment_iteration0.csv"
    pd.DataFrame(
        {
            "studentno": [11, 12],
            "programno": [2, 0],
            "programcodes": ["20-GE-KG", None],
            "rank": [1, 3],
            "designation": [0, 0],
            "In-Zone Rank": [1, 3],
        }
    ).to_csv(path, index=False)

    result = outcomes.read_validated_assignment(
        path,
        "real_preferences",
        np.array([11, 12]),
        np.array(["10-GE-KG", "20-GE-KG"], dtype=object),
    )

    np.testing.assert_array_equal(result.programno, [2, 0])
    with pytest.raises(ValueError, match="student set or order"):
        outcomes.read_validated_assignment(
            path,
            "real_preferences",
            np.array([12, 11]),
            np.array(["10-GE-KG", "20-GE-KG"], dtype=object),
        )

    frame = pd.read_csv(path)
    frame.loc[0, "programcodes"] = "10-GE-KG"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="number/code mapping"):
        outcomes.read_validated_assignment(
            path,
            "real_preferences",
            np.array([11, 12]),
            np.array(["10-GE-KG", "20-GE-KG"], dtype=object),
        )


def test_choice_assignment_allows_regeneratable_negative_infinite_utility(tmp_path):
    path = tmp_path / "assignment_iteration0.csv"
    path.write_text(
        "studentno,programno,programcodes,rank,designation,assigned_utility,In-Zone Rank\n"
        "11,1,10-GE-KG,1,0,-inf,1\n"
        "12,0,,2,0,,2\n",
        encoding="utf-8",
    )

    result = outcomes.read_validated_assignment(
        path,
        "choice_model",
        np.array([11, 12]),
        np.array(["10-GE-KG"], dtype=object),
    )

    assert result.assigned_utility is not None
    assert np.isneginf(result.assigned_utility[0])
    assert np.isnan(result.assigned_utility[1])


def test_fake_top_choice_corrects_saved_rank_and_attributes_disallowed_raw_top():
    current = assignment([2, 2, 0], [1, 1, 4], [0, 0, 0])
    raw_ranks = np.array(
        [
            [1, 2],
            [2, 1],
            [1, 2],
        ]
    )

    result = outcomes.fake_top_choice_metrics(
        current,
        raw_ranks,
        np.array([True, True, True]),
        np.array([True, False, True]),
    )

    assert result["assigned_count"] == 2
    assert result["reported_top1_count"] == 2
    assert result["fake_top1_count"] == 1
    assert result["fake_share_reported_top1"] == pytest.approx(0.5)
    assert result["corrected_raw_top1_count"] == 1
    assert result["corrected_raw_top1_rate_applicants"] == pytest.approx(1 / 3)
    assert result["fake_raw_top_disallowed_count"] == 1
    assert result["fake_raw_top_disallowed_share_fake"] == 1


def test_frl_tiers_use_strict_point_fifteen_boundaries():
    district = 0.5

    assert outcomes.frl_tier(0.65, district, missing="missing") == "medium"
    assert outcomes.frl_tier(0.650001, district, missing="missing") == "high"
    assert outcomes.frl_tier(0.35, district, missing="missing") == "medium"
    assert outcomes.frl_tier(0.349999, district, missing="missing") == "low"
    assert outcomes.frl_tier(float("nan"), district, missing="none") == "none"


def test_enrollment_statistics_report_empty_over_and_mean_status():
    values = np.array([8, 10, 12, *([10] * 22)], dtype=float)

    result = outcomes.enrollment_value_statistics(values, capacity=10)

    assert result["mean_enrollment"] == 10
    assert result["mean_utilization"] == 1
    assert result["mean_seat_gap"] == 0
    assert result["mean_empty_seats"] == pytest.approx(2 / 25)
    assert result["mean_over_seats"] == pytest.approx(2 / 25)
    assert result["share_iterations_under_90pct"] == pytest.approx(1 / 25)
    assert result["share_iterations_under_100pct"] == pytest.approx(1 / 25)
    assert result["share_iterations_over_100pct"] == pytest.approx(1 / 25)
    assert result["mean_status"] == "at_capacity"


def test_school_ses_statistics_classify_each_iteration_and_pair_baseline_delta():
    values = np.array([0.8, 0.5, 0.2, np.nan, *([0.5] * 21)])
    baseline = np.array([0.7, 0.4, 0.3, 0.9, *([0.4] * 21)])

    result = outcomes.school_ses_statistics(values, baseline, district_mean=0.5)

    assert result["share_iterations_high"] == pytest.approx(1 / 25)
    assert result["share_iterations_low"] == pytest.approx(1 / 25)
    assert result["share_iterations_no_enrollment"] == pytest.approx(1 / 25)
    assert result["share_iterations_medium"] == pytest.approx(22 / 25)
    assert result["percent_iterations_high"] == pytest.approx(4)
    assert result["percent_iterations_medium"] == pytest.approx(88)
    assert result["paired_delta_iterations"] == 24
    assert result["mean_policy_minus_baseline_frl"] == pytest.approx(2.2 / 24)
    assert result["transition"].startswith("medium_to_")


def test_haversine_and_travel_metrics_use_strict_three_miles_and_raw_missing():
    assert outcomes.haversine_miles(0, 0, 0, 1) == pytest.approx(3958.8 * np.pi / 180)
    current = assignment([1, 1, 1, 0], [4, 4, 1, 2], [1, 0, 0, 0])

    result = outcomes.travel_iteration_metrics(
        current,
        np.array([4, 0, 1, np.nan]),
        np.array([4.0, 4.0, 3.0, np.nan]),
    )

    assert result["long_all_count"] == 2
    assert result["long_designated_count"] == 1
    assert result["long_non_designated_count"] == 1
    assert result["designated_share_long_travelers"] == pytest.approx(0.5)
    assert result["long_saved_rank_ge4_count"] == 2
    assert result["long_raw_rank_ge4_count"] == 1
    assert result["long_raw_rank_missing_count"] == 1
    assert result["long_raw_rank_missing_non_designated_count"] == 1


def test_outcome_change_counts_handle_higher_and_lower_directions():
    policy = np.array([3.0, 1.0, 2.0, np.nan])
    baseline = np.array([2.0, 2.0, 2.0, 1.0])
    eligible = np.ones(4, dtype=bool)

    higher = outcomes.outcome_change_counts(
        policy, baseline, eligible, "higher_is_better"
    )
    lower = outcomes.outcome_change_counts(
        policy, baseline, eligible, "lower_is_better"
    )

    assert higher["eligible_count"] == 3
    assert (higher["win_count"], higher["tie_count"], higher["loss_count"]) == (
        1,
        1,
        1,
    )
    assert (lower["win_count"], lower["tie_count"], lower["loss_count"]) == (
        1,
        1,
        1,
    )
    assert higher["mean_delta"] == 0


def test_macro_summary_is_iteration_macro_average_not_pooled_rate():
    frame = pd.DataFrame(
        {
            "mode": ["choice_model", "choice_model"],
            "mode_order": [0, 0],
            "policy": ["status_quo", "status_quo"],
            "policy_order": [11, 11],
            "iteration": [0, 1],
            "rate": [0.25, 0.75],
            "count": [1, 9],
        }
    )

    result = outcomes.macro_summary(
        frame, ["mode", "mode_order", "policy", "policy_order"]
    ).iloc[0]

    assert result["iterations"] == 2
    assert result["mean_rate"] == pytest.approx(0.5)
    assert result["sd_rate"] == pytest.approx(np.sqrt(0.125))
    assert result["mean_count"] == 5


def test_write_outputs_refuses_existing_directory(tmp_path):
    output = tmp_path / "analysis"
    frames = {"one.csv": pd.DataFrame({"value": [1]})}
    metadata = {"definition": "test"}

    outcomes.write_outputs(output, frames, metadata)

    assert pd.read_csv(output / "one.csv")["value"].tolist() == [1]
    assert json.loads((output / "methodology.json").read_text()) == metadata
    with pytest.raises(FileExistsError):
        outcomes.write_outputs(output, frames, metadata)

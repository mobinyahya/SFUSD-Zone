import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import evaluate_zone_subconfig_matches as zone_subconfigs  # noqa: E402


def test_build_simulation_config_overrides_selected_zone_plans(tmp_path):
    base = {
        "data": {
            "scenario": "legacy",
            "overrides": {
                "sources": {
                    "assignment.zones": {
                        "Con1": {
                            "path": "attendance-areas.csv",
                            "classification": "public",
                        }
                    }
                }
            },
        },
        "output_dir": "unused",
        "paths": {"assignment-folder": "old"},
        "subconfigs": ["old"],
    }
    small_zones = tmp_path / "small.csv"
    medium_zones = tmp_path / "medium.csv"

    result = zone_subconfigs.build_simulation_config(
        base, tmp_path / "matches", small_zones, medium_zones
    )

    assert result["subconfigs"] == list(zone_subconfigs.SUBCONFIGS)
    assert result["iterations"] == {"start": 0, "end": 25}
    zones = result["data"]["overrides"]["sources"]["assignment.zones"]
    assert zones["18zone_2"]["path"] == str(small_zones)
    assert zones["6zone-1"]["path"] == str(medium_zones)
    assert zones["Con1"]["path"] == "attendance-areas.csv"
    assert "18zone_2" not in base["data"]["overrides"]["sources"]["assignment.zones"]


def test_write_metrics_csv_preserves_requested_subconfig_order(tmp_path):
    metrics = {
        label: pd.Series({"metric one": index})
        for index, label in enumerate(zone_subconfigs.SUBCONFIGS)
    }

    output = zone_subconfigs.write_metrics_csv(metrics, tmp_path)

    frame = pd.read_csv(output, index_col="metric")
    assert list(frame.columns) == list(zone_subconfigs.SUBCONFIGS)
    assert frame.loc["metric one"].tolist() == list(
        range(len(zone_subconfigs.SUBCONFIGS))
    )


def test_write_metrics_csv_accepts_scope_suffix(tmp_path):
    output = zone_subconfigs.write_metrics_csv(
        {"policy": pd.Series({"metric one": 1})},
        tmp_path,
        suffix="_all_rounds",
    )

    assert output.name.startswith(
        "zone_subconfigs_25_eval_assignment_full_all_rounds_"
    )


def test_build_simulation_config_applies_real_preference_settings(tmp_path):
    students = tmp_path / "student_2324.csv"
    result = zone_subconfigs.build_simulation_config(
        {"data": {"scenario": "legacy", "overrides": {}}},
        tmp_path / "matches",
        tmp_path / "small.csv",
        tmp_path / "medium.csv",
        real_student_data=students,
    )

    assert result["data"]["overrides"]["sources"]["assignment.students"][
        "path"
    ] == str(students)
    assert result["utility-model"] == {
        "designate-lp-for-all": False,
        "enable": False,
        "list-length": "0.8*round(real_length)",
    }
    assert result["random-seed"] == 2023
    assert result["r1-only"] is True
    assert (
        result["data"]["overrides"]["filters"]["assignment"][
            "special_programs"
        ]
        == "exclude_any_special"
    )
    assert result["rounds-merged-options"] == [0]


def test_build_simulation_config_can_retain_special_programs(tmp_path):
    result = zone_subconfigs.build_simulation_config(
        {"data": {"scenario": "legacy", "overrides": {}}},
        tmp_path / "matches",
        tmp_path / "small.csv",
        tmp_path / "medium.csv",
        real_student_data=tmp_path / "student_2324.csv",
        include_special_programs=True,
    )

    assert result["utility-model"]["enable"] is False
    assert (
        result["data"]["overrides"]["filters"]["assignment"][
            "special_programs"
        ]
        == "include"
    )


def test_build_simulation_config_selects_policies_and_program_data(tmp_path):
    programs = tmp_path / "programs.csv"
    result = zone_subconfigs.build_simulation_config(
        {"data": {"scenario": "legacy", "overrides": {}}},
        tmp_path / "matches",
        tmp_path / "small.csv",
        tmp_path / "medium.csv",
        subconfigs=("policy_a", "policy_b"),
        program_data=programs,
    )

    assert result["subconfigs"] == ["policy_a", "policy_b"]
    assert result["data"]["overrides"]["sources"]["assignment.programs"][
        "path"
    ] == str(programs)


def test_build_simulation_config_records_all_round_evaluation(tmp_path):
    result = zone_subconfigs.build_simulation_config(
        {"data": {"scenario": "legacy", "overrides": {}}},
        tmp_path / "matches",
        tmp_path / "small.csv",
        tmp_path / "medium.csv",
        evaluation_population="all_rounds",
    )

    assert result["evaluation-population"] == "all_rounds"

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import evaluate_zone_subconfig_matches as zone_subconfigs  # noqa: E402


def test_build_simulation_config_overrides_selected_zone_plans(tmp_path):
    base = {
        "output_dir": "unused",
        "paths": {
            "assignment-folder": "old",
            "zone-files": {
                "18zone_2": "old-small.csv",
                "6zone-1": "old-medium.csv",
                "Con1": "attendance-areas.csv",
            },
        },
        "subconfigs": ["old"],
    }
    small_zones = tmp_path / "small.csv"
    medium_zones = tmp_path / "medium.csv"

    result = zone_subconfigs.build_simulation_config(
        base, tmp_path / "matches", small_zones, medium_zones
    )

    assert result["subconfigs"] == list(zone_subconfigs.SUBCONFIGS)
    assert result["iterations"] == {"start": 0, "end": 25}
    assert result["paths"]["zone-files"] == {
        "18zone_2": str(small_zones),
        "6zone-1": str(medium_zones),
        "Con1": "attendance-areas.csv",
    }
    assert result["paths"]["student-save"] == str(tmp_path / "matches/precomputed")
    assert base["paths"]["zone-files"]["18zone_2"] == "old-small.csv"


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


def test_build_simulation_config_applies_real_preference_settings(tmp_path):
    students = tmp_path / "student_2324.csv"
    result = zone_subconfigs.build_simulation_config(
        {"paths": {"zone-files": {}}},
        tmp_path / "matches",
        tmp_path / "small.csv",
        tmp_path / "medium.csv",
        real_student_data=students,
    )

    assert result["paths"]["student-data"] == str(students)
    assert result["paths"]["student-save"] == str(tmp_path / "matches/precomputed")
    assert result["utility-model"] == {
        "designate-lp-for-all": False,
        "enable": False,
        "list-length": "0.8*round(real_length)",
    }
    assert result["random-seed"] == 2023
    assert result["r1-only"] is True
    assert result["remove-special-lps"] is True
    assert result["rounds-merged-options"] == [0]


def test_build_simulation_config_can_retain_special_programs(tmp_path):
    result = zone_subconfigs.build_simulation_config(
        {"paths": {"zone-files": {}}},
        tmp_path / "matches",
        tmp_path / "small.csv",
        tmp_path / "medium.csv",
        real_student_data=tmp_path / "student_2324.csv",
        include_special_programs=True,
    )

    assert result["utility-model"]["enable"] is False
    assert result["remove-special-lps"] is False

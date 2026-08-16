import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.attendance_area_student_demographics import (  # noqa: E402
    add_frl_datasets,
    build_summary,
    configured_students,
    fallback_block_report,
)


def block_geometry() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"census_block_2020": ["100", "200", "300"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )


def test_add_frl_datasets_uses_fallback_without_changing_old_values():
    students = pd.DataFrame(
        {
            "studentno": [1, 2, 3],
            "idschoolattendance": [10, 20, 30],
            "latitude": [0.5, 0.5, 0.5],
            "longitude": [0.5, 1.5, 2.5],
            "freelunch_prob": [0.1, 0.2, 0.3],
            "reducedlunch_prob": [0.05, 0.1, 0.15],
        }
    )
    lookup = pd.Series({"100": 0.8, "200": float("nan")})

    result = add_frl_datasets(students, lookup, block_geometry())

    assert result["new_frl"].tolist() == pytest.approx([0.8, 0.3, 0.45])
    assert result["old_frl"].tolist() == pytest.approx([0.15, 0.3, 0.45])
    assert result["frl_fallback_reason"].isna().tolist() == [True, False, False]
    assert result["frl_fallback_reason"].iloc[1:].tolist() == [
        "blank updated FRL rate",
        "absent from updated lookup",
    ]

    report = fallback_block_report(result)
    assert report["census_block_2020"].tolist() == ["300", "200"]
    assert report["student_count"].tolist() == [1, 1]


def test_build_summary_counts_race_and_expected_frl_for_every_attendance_area():
    students = pd.DataFrame(
        {
            "studentno": [1, 2, 3],
            "idschoolattendance": [100, 100, 200],
            "resolved_ethnicity": ["Chinese", "Hispanic/Latino", None],
            "new_frl": [0.8, 0.2, 0.5],
            "old_frl": [0.1, 0.2, 0.3],
        }
    )
    schools = pd.DataFrame(
        {
            "school_id": [100, 200, 300],
            "school_name": ["One", "Two", "Three"],
        }
    )

    result = build_summary(students, schools).set_index("attendance_area")

    assert result.loc[100, "total_students"] == 2
    assert result.loc[100, "new_frl"] == pytest.approx(1.0)
    assert result.loc[100, "old_frl"] == pytest.approx(0.3)
    assert result.loc[100, "race_Asian"] == 1
    assert result.loc[100, "race_Hispanic"] == 1
    assert result.loc[200, "race_Other"] == 1
    assert result.loc[300, "total_students"] == 0
    assert result.loc[300, "new_frl"] == 0
    assert result.loc[300, "old_frl"] == 0

    race_columns = [column for column in result if column.startswith("race_")]
    assert result[race_columns].sum(axis=1).equals(result["total_students"])


def test_configured_students_uses_normalized_assignment_selection(tmp_path):
    student_path = tmp_path / "students.csv"
    pd.DataFrame(
        {
            "studentno": [1, 2, 3],
            "grade": ["KG", "KG", "01"],
            "idschoolattendance": [100, 100, 100],
            "resolved_ethnicity": ["Chinese", "White", "White"],
            "latitude": [0.5, 0.5, 0.5],
            "longitude": [0.5, 0.5, 0.5],
            "freelunch_prob": [0.1, 0.2, 0.3],
            "reducedlunch_prob": [0.0, 0.0, 0.0],
            "r1_ranked_idschool": ["[]", "[100]", "[100]"],
            "r1_programs": ["[]", "['SA']", "['GE']"],
            "r2_ranked_idschool": ["[100]", "[100]", "[100]"],
            "r2_programs": ["['GE']", "['GE']", "['GE']"],
        }
    ).to_csv(student_path, index=False)
    config = {
        "data": {
            "scenario": "mission-bay-2324",
            "overrides": {
                "sources": {"assignment.students": {"path": str(student_path)}},
                "filters": {
                    "assignment": {
                        "year": "2324",
                        "grades": ["KG"],
                        "student_population": "applicant",
                        "rounds": [1, 2],
                        "special_programs": "exclude_any_special",
                        "capacity_profile": "status_quo",
                        "include_mission_bay": True,
                    }
                },
            },
        }
    }

    students = configured_students(
        config, pd.Series({"100": 0.8}), block_geometry()
    )

    assert students["studentno"].tolist() == [1]
    assert students["first_participating_round"].tolist() == [2]
    assert students["selected_programs"].tolist() == [["GE"]]

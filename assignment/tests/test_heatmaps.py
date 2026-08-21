import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from assignment.student_assignment.evaluation.heatmaps import (
    average_heatmap_data,
    build_attendance_area_data,
    render_attendance_area_heatmap,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


def _areas():
    return gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )


def _schools():
    return pd.DataFrame(
        {
            "school_id": [497, 999, 909, 100],
            "school_name": [
                "Webster ES",
                "Mission Bay ES",
                "Mission Bay ES",
                "Other ES (PK-5)",
            ],
            "category": ["Attendance"] * 4,
            "lat": [0.25, 0.75, 0.75, 0.5],
            "lon": [0.25, 0.75, 0.75, 1.5],
        }
    )


def _utilization():
    return pd.DataFrame(
        {
            "config_name": ["policy"] * 3,
            "school_id": [497, 999, 100],
            "capacity": [20.0, 60.0, 20.0],
            "assigned": [30.0, 70.0, 10.0],
        }
    )


def test_webster_and_mission_bay_share_one_attendance_area():
    result = build_attendance_area_data(_areas(), _schools(), _utilization())

    combined = result.loc[
        result["attendance_area_name"].eq("Webster / Mission Bay")
    ].iloc[0]
    assert combined["capacity"] == 80
    assert combined["assigned"] == 100
    assert combined["utilization"] == 1.25
    assert combined["seat_difference"] == -20
    assert (
        result.loc[result["attendance_area_name"].eq("Other"), "seat_difference"].iat[0]
        == 10
    )


def test_heatmap_data_are_averaged_by_policy_and_school():
    first = _utilization()
    second = first.copy()
    second["assigned"] += [2, 4, 6]

    averaged = average_heatmap_data([first, second]).set_index("school_id")

    assert averaged.loc[497, "assigned"] == 31
    assert averaged.loc[999, "assigned"] == 72
    assert averaged.loc[100, "assigned"] == 13


def test_parallel_heatmap_payloads_cover_each_policy():
    first = _utilization()
    second = first.copy()
    second["assigned"] += 2
    payloads = [
        {"heatmaps": [frame], "expected_config_names": ["policy"]}
        for frame in (first, second)
    ]

    averaged = MarketGenerator.combine_heatmap_batch_payloads(payloads).set_index(
        "school_id"
    )

    assert averaged.loc[497, "assigned"] == 31
    assert averaged.loc[999, "assigned"] == 71


def test_heatmap_renderer_writes_png(tmp_path):
    result = build_attendance_area_data(_areas(), _schools(), _utilization())

    output = render_attendance_area_heatmap(result, "policy", tmp_path / "heatmap.png")

    assert output.is_file()
    assert output.stat().st_size > 0


def test_every_attendance_area_requires_a_school():
    schools = _schools().loc[lambda frame: frame["school_id"] != 100]

    with pytest.raises(ValueError, match="Every attendance-area polygon"):
        build_attendance_area_data(_areas(), schools, _utilization())

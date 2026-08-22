import json

import geopandas as gpd
import pandas as pd
import pytest
from loaders import DataScenario, ResolvedSource
from shapely.geometry import box

from assignment.student_assignment.evaluation.heatmaps import (
    AttendanceAreaArtifactStore,
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


def _scenario(tmp_path):
    area_source = tmp_path / "areas.geojson"
    school_source = tmp_path / "schools.csv"
    area_source.write_text("areas-v1", encoding="utf-8")
    school_source.write_text("schools-v1", encoding="utf-8")
    scenario = DataScenario(
        id="heatmap-test",
        schema_version=2,
        roots={"cache": tmp_path / "cache"},
        filters={"assignment": {"include_mission_bay": True}},
        _source_values={
            "assignment.attendance_areas": ResolvedSource(path=area_source),
            "assignment.schools": ResolvedSource(path=school_source),
        },
    )
    return scenario, area_source, school_source


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


def test_attendance_area_geometry_is_source_aware_and_cached(tmp_path):
    scenario, _, _ = _scenario(tmp_path)
    calls = {"areas": 0, "schools": 0}

    def load_areas():
        calls["areas"] += 1
        return _areas()

    def load_schools():
        calls["schools"] += 1
        return _schools()

    store = AttendanceAreaArtifactStore(
        scenario,
        area_loader=load_areas,
        school_loader=load_schools,
    )
    first, first_path = store.geometry()
    second, second_path = store.geometry()

    assert calls == {"areas": 1, "schools": 1}
    assert first_path == second_path
    assert first_path.is_relative_to(
        tmp_path / "cache/attendance_area_heatmap_geometry/v1"
    )
    manifest = json.loads(
        (first_path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["classification"] == "internal-derived"
    assert set(manifest["sources"]["sources"]) == {
        "assignment.attendance_areas",
        "assignment.schools",
    }
    assert first["school_ids"].tolist() == second["school_ids"].tolist()


def test_attendance_area_cache_changes_when_school_source_changes(tmp_path):
    scenario, _, school_source = _scenario(tmp_path)
    calls = 0

    def load_areas():
        nonlocal calls
        calls += 1
        return _areas()

    store = AttendanceAreaArtifactStore(
        scenario,
        area_loader=load_areas,
        school_loader=_schools,
    )
    _, first_path = store.geometry()
    school_source.write_text("schools-version-two", encoding="utf-8")
    _, second_path = store.geometry()

    assert calls == 2
    assert first_path != second_path

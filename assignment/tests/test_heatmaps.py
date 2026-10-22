import json

import geopandas as gpd
import pandas as pd
import pytest
from loaders import DataScenario, ResolvedSource
from shapely.geometry import box

from assignment.student_assignment.evaluation.heatmaps import (
    AttendanceAreaArtifactStore,
    attach_heatmap_metrics,
    average_heatmap_data,
    build_attendance_area_geometry,
    build_heatmap_geometry,
    render_assignment_heatmap,
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


def _metrics():
    return pd.DataFrame(
        {
            "config_name": ["policy"] * 3,
            "policy": ["real_match"] * 3,
            "building_block": ["attendance_area"] * 3,
            "zone_file": [""] * 3,
            "area_id": [497, 999, 100],
            "capacity": [20.0, 60.0, 20.0],
            "assigned": [15.0, 60.0, 20.0],
            "unassigned": [0.0, 0.0, 3.0],
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
    geometry = build_attendance_area_geometry(_areas(), _schools()).rename(
        columns={
            "attendance_area_name": "geographic_area_name",
            "school_ids": "area_ids",
        }
    )
    result = attach_heatmap_metrics(geometry, _metrics())

    combined = result.loc[
        result["geographic_area_name"].eq("Webster / Mission Bay")
    ].iloc[0]
    assert combined["capacity"] == 80
    assert combined["assigned"] == 75
    assert combined["unfilled_spots"] == 5
    assert combined["seat_balance"] == 5
    other = result.loc[result["geographic_area_name"].eq("Other")].iloc[0]
    assert other["unassigned"] == 3
    assert other["seat_balance"] == -3


def test_heatmap_data_are_averaged_by_policy_and_area():
    first = _metrics()
    second = first.copy()
    second["assigned"] += [2, 4, 0]
    second["unassigned"] += [0, 0, 2]

    averaged = average_heatmap_data([first, second]).set_index("area_id")

    assert averaged.loc[497, "assigned"] == 16
    assert averaged.loc[999, "assigned"] == 62
    assert averaged.loc[100, "unassigned"] == 4


def test_parallel_heatmap_payloads_cover_each_policy():
    first = _metrics()
    second = first.copy()
    second["assigned"] += 2
    payloads = [
        {"heatmaps": [frame], "expected_config_names": ["policy"]}
        for frame in (first, second)
    ]

    averaged = MarketGenerator.combine_heatmap_batch_payloads(payloads).set_index(
        "area_id"
    )

    assert averaged.loc[497, "assigned"] == 16
    assert averaged.loc[999, "assigned"] == 61


def test_attendance_area_policy_file_dissolves_to_zones(tmp_path):
    scenario, _, _ = _scenario(tmp_path)
    zone_file = tmp_path / "zones.csv"
    zone_file.write_text("497,999,909,100\n", encoding="utf-8")
    attendance_geometry = build_attendance_area_geometry(_areas(), _schools())

    result = build_heatmap_geometry(
        scenario,
        "attendance_area",
        zone_file,
        attendance_area_geometry=attendance_geometry,
    )

    assert result["geographic_area_name"].tolist() == ["Zone 1"]
    assert set(result["area_ids"].iat[0]) == {497, 999, 909, 100}


def test_census_zone_geometry_ignores_water_only_areas(tmp_path, monkeypatch):
    scenario, _, _ = _scenario(tmp_path)
    zone_file = tmp_path / "zones.csv"
    zone_file.write_text("1,60759804011,99\n2\n", encoding="utf-8")
    census_geometry = gpd.GeoDataFrame(
        {"Block": [10, 20, 60759804011000]},
        geometry=[
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(-100, -100, -50, -50),
        ],
        crs="EPSG:4326",
    )
    crosswalk = pd.DataFrame(
        {
            "Block": [10, 20, 60759804011000],
            "BlockGroup": [1, 2, 60759804011],
        }
    )
    monkeypatch.setattr(
        "assignment.student_assignment.evaluation.heatmaps.load_census_geometry",
        lambda *_args: census_geometry,
    )
    monkeypatch.setattr(
        "assignment.student_assignment.evaluation.heatmaps.load_geography_crosswalk",
        lambda *_args: crosswalk,
    )

    result = build_heatmap_geometry(scenario, "block_group", zone_file)

    assert result["geographic_area_name"].tolist() == ["Zone 1", "Zone 2"]
    assert result["area_ids"].tolist() == [(1,), (2,)]


def test_heatmap_renderer_writes_png(tmp_path):
    geometry = build_attendance_area_geometry(_areas(), _schools()).rename(
        columns={
            "attendance_area_name": "geographic_area_name",
            "school_ids": "area_ids",
        }
    )
    result = attach_heatmap_metrics(geometry, _metrics())

    output = render_assignment_heatmap(result, "policy", tmp_path / "heatmap.png")

    assert output.is_file()
    assert output.stat().st_size > 0


def test_every_attendance_area_requires_a_school():
    schools = _schools().loc[lambda frame: frame["school_id"] != 100]

    with pytest.raises(ValueError, match="Every attendance-area polygon"):
        build_attendance_area_geometry(_areas(), schools)


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

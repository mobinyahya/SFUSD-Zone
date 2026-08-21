from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from loaders.geography import (
    load_census_geometry,
    match_points_to_census,
    normalize_census_geography,
)
from loaders.tables import filter_outside_district_students


def _geography_scenario(
    tmp_path,
    scenario_factory,
    *,
    group="assignment",
    outside_district_students="ignore",
):
    blocks_path = tmp_path / "blocks.shp"
    blockgroups_path = tmp_path / "blockgroups.shp"
    tracts_path = tmp_path / "tracts.shp"
    crosswalk_path = tmp_path / "crosswalk.csv"
    gpd.GeoDataFrame(
        {
            "GEOID20": [
                "060750101001000",
                "060750101001001",
                "060759901001000",
            ],
            "ALAND20": [100, 100, 0],
            "AWATER20": [0, 0, 100],
            "geometry": [
                box(-122.50, 37.70, -122.49, 37.71),
                box(-122.49, 37.70, -122.48, 37.71),
                box(-122.48, 37.70, -122.47, 37.71),
            ],
        },
        crs="EPSG:4326",
    ).to_file(blocks_path)
    parent_geometry = [
        box(-122.50, 37.70, -122.48, 37.71),
        box(-122.48, 37.70, -122.47, 37.71),
    ]
    gpd.GeoDataFrame(
        {
            "GEOID20": ["060750101001", "060759901001"],
            "ALAND20": [200, 0],
            "AWATER20": [0, 100],
            "geometry": parent_geometry,
        },
        crs="EPSG:4326",
    ).to_file(blockgroups_path)
    gpd.GeoDataFrame(
        {
            "GEOID20": ["06075010100", "06075990100"],
            "ALAND20": [200, 0],
            "AWATER20": [0, 100],
            "geometry": parent_geometry,
        },
        crs="EPSG:4326",
    ).to_file(tracts_path)
    pd.DataFrame(
        {
            "Block": [60750101001000, 60750101001001, 60759901001000],
            "BlockGroup": [60750101001, 60750101001, 60759901001],
            "Tract": [6075010100, 6075010100, 6075990100],
        }
    ).to_csv(crosswalk_path, index=False)
    prefix = f"{group}.geography"
    sources = {
        f"{prefix}.blocks": {"path": str(blocks_path)},
        f"{prefix}.blockgroups": {"path": str(blockgroups_path)},
        f"{prefix}.tracts": {"path": str(tracts_path)},
        f"{prefix}.crosswalk": {"path": str(crosswalk_path)},
    }
    if group == "optimization":
        sources["optimization.census"] = sources.pop("optimization.geography.blocks")
        sources["optimization.crosswalk"] = sources.pop(
            "optimization.geography.crosswalk"
        )
    return scenario_factory(
        sources=sources,
        filters={
            group: {
                "geography_vintage": "2020",
                "outside_district_students": outside_district_students,
            }
        },
    )


@pytest.mark.parametrize("group", ["assignment", "optimization"])
def test_loads_2020_geography_without_water_only_areas(
    tmp_path, scenario_factory, group
):
    scenario = _geography_scenario(tmp_path, scenario_factory, group=group)

    blocks = load_census_geometry(scenario, group, "Block")
    blockgroups = load_census_geometry(scenario, group, "BlockGroup")
    tracts = load_census_geometry(scenario, group, "Tract")

    assert blocks["Block"].tolist() == [60750101001000, 60750101001001]
    assert blockgroups["BlockGroup"].tolist() == [60750101001]
    assert tracts["Tract"].tolist() == [6075010100]


def test_points_outside_geometry_remain_unmatched(tmp_path, scenario_factory):
    scenario = _geography_scenario(tmp_path, scenario_factory)
    points = pd.DataFrame(
        {
            "studentno": [1, 2, 3],
            "latitude": [37.705, 37.705, 37.705],
            "longitude": [-122.495, -122.501, -122.60],
        }
    )

    matched = match_points_to_census(
        points,
        scenario,
        "assignment",
        latitude_column="latitude",
        longitude_column="longitude",
    )

    assert matched.loc[0, "Block"] == 60750101001000
    assert matched.loc[0, "BlockGroup"] == 60750101001
    assert matched.loc[0, "Tract"] == 6075010100
    assert matched.loc[[1, 2]].isna().all().all()


def test_same_vintage_ids_are_kept_and_other_vintage_is_remapped(
    tmp_path, scenario_factory
):
    scenario = _geography_scenario(tmp_path, scenario_factory)
    current = pd.DataFrame(
        {
            "studentno": [1],
            "census_block": [60750101001001],
            "census_blockgroup": [60750101001],
            "census_tract": [6075010100],
        }
    )
    legacy = current.assign(latitude=37.705, longitude=-122.495)

    kept = normalize_census_geography(
        current,
        scenario,
        "assignment",
        source_vintage="2020",
        style="student",
    )
    remapped = normalize_census_geography(
        legacy,
        scenario,
        "assignment",
        source_vintage="2010",
        style="student",
    )

    assert kept.loc[0, "census_block"] == 60750101001001
    assert remapped.loc[0, "census_block"] == 60750101001000
    assert remapped.attrs["geography_vintage"] == "2020"


@pytest.mark.parametrize(
    ("policy", "expected_students", "expected_source_rows"),
    [
        ("ignore", [1], [5]),
        ("include", [1, 2], [5, 6]),
    ],
)
def test_outside_district_student_policy(
    tmp_path,
    scenario_factory,
    policy,
    expected_students,
    expected_source_rows,
):
    scenario = _geography_scenario(
        tmp_path,
        scenario_factory,
        outside_district_students=policy,
    )
    students = pd.DataFrame(
        {
            "studentno": [1, 2],
            "latitude": [37.705, 37.705],
            "longitude": [-122.495, -122.501],
        }
    )
    students.attrs.update(source_rows=[5, 6], source_row_count=10)
    normalized = normalize_census_geography(
        students,
        scenario,
        "assignment",
        source_vintage="2010",
        style="student",
    )

    filtered = filter_outside_district_students(
        normalized, scenario, "assignment"
    )

    assert filtered["studentno"].tolist() == expected_students
    assert filtered.attrs["source_rows"] == expected_source_rows
    assert filtered.attrs["source_row_count"] == 10

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
    tmp_path, scenario_factory, *, outside_district_students="ignore"
):
    blocks_path = tmp_path / "blocks.shp"
    blockgroups_path = tmp_path / "blockgroups.shp"
    tracts_path = tmp_path / "tracts.shp"
    crosswalk_path = tmp_path / "crosswalk.csv"
    gpd.GeoDataFrame(
        {
            "GEOID20": ["060750101001000", "060750101001001"],
            "geometry": [
                box(-122.50, 37.70, -122.49, 37.71),
                box(-122.49, 37.70, -122.48, 37.71),
            ],
        },
        crs="EPSG:4326",
    ).to_file(blocks_path)
    parent_geometry = [box(-122.50, 37.70, -122.48, 37.71)]
    gpd.GeoDataFrame(
        {"GEOID20": ["060750101001"], "geometry": parent_geometry},
        crs="EPSG:4326",
    ).to_file(blockgroups_path)
    gpd.GeoDataFrame(
        {"GEOID20": ["06075010100"], "geometry": parent_geometry},
        crs="EPSG:4326",
    ).to_file(tracts_path)
    pd.DataFrame(
        {
            "Block": [60750101001000, 60750101001001],
            "BlockGroup": [60750101001, 60750101001],
            "Tract": [6075010100, 6075010100],
        }
    ).to_csv(crosswalk_path, index=False)
    return scenario_factory(
        sources={
            "assignment.geography.blocks": {"path": str(blocks_path)},
            "assignment.geography.blockgroups": {"path": str(blockgroups_path)},
            "assignment.geography.tracts": {"path": str(tracts_path)},
            "assignment.geography.crosswalk": {"path": str(crosswalk_path)},
        },
        filters={
            "assignment": {
                "geography_vintage": "2020",
                "outside_district_students": outside_district_students,
            }
        },
    )


def test_loads_blocks_and_derives_parent_geometry(tmp_path, scenario_factory):
    scenario = _geography_scenario(tmp_path, scenario_factory)

    blocks = load_census_geometry(scenario, "assignment", "Block")
    blockgroups = load_census_geometry(scenario, "assignment", "BlockGroup")
    tracts = load_census_geometry(scenario, "assignment", "Tract")

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

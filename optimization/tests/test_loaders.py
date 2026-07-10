import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from optimization.data import loaders
from optimization.data.loaders import IngestConfig


def test_projected_centroids_latlon_avoids_geographic_crs_warning():
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(-122.5, 37.7, -122.4, 37.8)]},
        crs="EPSG:4326",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Geometry is in a geographic CRS.*")
        centroids = loaders._projected_centroids_latlon(gdf)

    assert centroids.crs == "EPSG:4326"
    assert 37.7 < centroids.iloc[0].y < 37.8
    assert -122.5 < centroids.iloc[0].x < -122.4


def test_student_cache_path_includes_student_filtering_inputs():
    baseline = loaders._student_cache_path(IngestConfig(unit="Block"))
    different_years = loaders._student_cache_path(
        IngestConfig(unit="Block", years=[14])
    )
    different_population = loaders._student_cache_path(
        IngestConfig(unit="Block", population_type="All")
    )
    different_optout = loaders._student_cache_path(
        IngestConfig(unit="Block", drop_optout=False)
    )

    assert baseline != different_years
    assert baseline != different_population
    assert baseline != different_optout


def test_load_census_shapefile_enriches_geographic_ids(tmp_path, monkeypatch):
    source = gpd.GeoDataFrame(
        {
            "geoid10": [1001, 1002],
            "geometry": [Point(-122.4, 37.7), Point(-122.5, 37.8)],
        },
        crs="EPSG:4326",
    )
    optimization_dir = tmp_path / "Optimization"
    optimization_dir.mkdir()
    pd.DataFrame(
        {"Block": [1001, 1002], "BlockGroup": [100, 100], "Tract": [10, 10]}
    ).to_csv(optimization_dir / "block_blockgroup_tract.csv", index=False)
    monkeypatch.setattr(loaders.gpd, "read_file", lambda _: source.copy())
    monkeypatch.setattr(loaders, "get_dropbox_path", lambda _: str(tmp_path))

    census = loaders.load_census_shapefile("BlockGroup")

    assert list(census["Block"]) == [1001, 1002]
    assert list(census["BlockGroup"]) == [100, 100]


def test_load_distance_dict_reads_cached_matrix(tmp_path, monkeypatch):
    optimization_dir = tmp_path / "Optimization"
    optimization_dir.mkdir()
    matrix = pd.DataFrame(
        [[0.0, 1.25], [1.25, 0.0]],
        index=pd.Index([100, 200], name="BlockGroup"),
        columns=[100, 200],
    )
    matrix.to_csv(optimization_dir / "distances_bg2bg.csv")
    monkeypatch.setattr(loaders, "DROPBOX_PATH", str(tmp_path))

    distances = loaders.load_distance_dict(
        IngestConfig(unit="BlockGroup"), {100: 4, 200: 9}
    )

    assert distances == {4: {4: 0.0, 9: 1.25}, 9: {4: 1.25, 9: 0.0}}


def test_load_distance_dict_reads_rectangular_school_matrix(tmp_path, monkeypatch):
    optimization_dir = tmp_path / "Optimization"
    optimization_dir.mkdir()
    matrix = pd.DataFrame(
        [[0.0, 1.25, 2.5]],
        index=pd.Index([100], name="Block"),
        columns=[100, 200, 300],
    )
    matrix.to_csv(optimization_dir / "distances_b2b_schools.csv")
    monkeypatch.setattr(loaders, "DROPBOX_PATH", str(tmp_path))

    distances = loaders.load_distance_dict(
        IngestConfig(unit="Block"), {100: 4, 200: 9, 300: 12}
    )

    assert distances == {
        4: {4: 0.0, 9: 1.25, 12: 2.5},
        9: {4: 1.25},
        12: {4: 2.5},
    }


def test_load_distance_dict_builds_and_caches_matrix(tmp_path, monkeypatch):
    locations = pd.DataFrame(
        {"Lat": [0.0, 0.0], "Lon": [0.0, 90.0]},
        index=pd.Index([100, 200], name="BlockGroup"),
    )
    monkeypatch.setattr(loaders, "DROPBOX_PATH", str(tmp_path))
    monkeypatch.setattr(loaders, "load_area_latlon", lambda _: locations)

    distances = loaders.load_distance_dict(
        IngestConfig(unit="BlockGroup"), {100: 4, 200: 9}
    )

    assert distances[4][4] == 0.0
    assert distances[4][9] == pytest.approx(distances[9][4])
    assert distances[4][9] > 0
    assert (tmp_path / "Optimization" / "distances_bg2bg.csv").exists()

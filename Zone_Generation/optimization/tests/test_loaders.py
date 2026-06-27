import warnings

import geopandas as gpd
from shapely.geometry import box

from Zone_Generation.optimization.data.loaders import (
    IngestConfig,
    _projected_centroids_latlon,
    _student_cache_path,
)


def test_projected_centroids_latlon_avoids_geographic_crs_warning():
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(-122.5, 37.7, -122.4, 37.8)]},
        crs="EPSG:4326",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Geometry is in a geographic CRS.*")
        centroids = _projected_centroids_latlon(gdf)

    assert centroids.crs == "EPSG:4326"
    assert 37.7 < centroids.iloc[0].y < 37.8
    assert -122.5 < centroids.iloc[0].x < -122.4


def test_student_cache_path_includes_student_filtering_inputs():
    baseline = _student_cache_path(IngestConfig(unit="Block"))
    different_years = _student_cache_path(IngestConfig(unit="Block", years=[14]))
    different_population = _student_cache_path(
        IngestConfig(unit="Block", population_type="All")
    )
    different_optout = _student_cache_path(
        IngestConfig(unit="Block", drop_optout=False)
    )

    assert baseline != different_years
    assert baseline != different_population
    assert baseline != different_optout

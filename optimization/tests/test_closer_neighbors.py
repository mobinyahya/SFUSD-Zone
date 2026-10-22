import pickle

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import box

from optimization.data.closer_neighbors import (
    CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION,
    CloserNeighborArtifactStore,
)
from optimization.levels import LevelSpec


def _base_geometry():
    return gpd.GeoDataFrame(
        {
            "Block": [10, 11, 12],
            "geometry": [
                box(-122.400, 37.740, -122.390, 37.760),
                box(-122.420, 37.770, -121.000, 37.780),
                box(-122.505, 37.745, -122.495, 37.755),
            ],
        },
        crs="EPSG:4326",
    )


def _schools():
    return pd.DataFrame({"school_id": [100], "lat": [37.750], "lon": [-122.500]})


def _graph():
    G = nx.path_graph(3)
    for node, area_id in enumerate((10, 11, 12)):
        G.nodes[node]["area_id"] = area_id
    G.graph["distance_dict"] = {2: {0: 1.0, 1: 2.0, 2: 0.0}}
    return G


def _scenario(tmp_path, scenario_factory):
    census = tmp_path / "areas.shp"
    crosswalk = tmp_path / "crosswalk.csv"
    schools = tmp_path / "schools.csv"
    census.write_bytes(b"geometry-source")
    crosswalk.write_text("Block,BlockGroup\n10,1\n", encoding="utf-8")
    schools.write_text("school_id,lat,lon\n100,37.75,-122.5\n", encoding="utf-8")
    return scenario_factory(
        sources={
            "optimization.census": {"path": str(census)},
            "optimization.crosswalk": {"path": str(crosswalk)},
            "optimization.schools": {"path": str(schools)},
        },
        cache_root=tmp_path / "cache",
    )


def test_geometry_uses_closest_polygon_points_not_neighbor_centroids(
    tmp_path, scenario_factory
):
    G = _graph()
    store = CloserNeighborArtifactStore(
        _scenario(tmp_path, scenario_factory),
        geometry_loader=lambda unit: _base_geometry(),
        school_loader=_schools,
    )

    data = store.for_graph("Block_0", G)

    # The long neighbor's centroid is much farther east, but its nearest polygon
    # point is closer to the school than node 0's nearest point.
    assert G.graph["distance_dict"][2][1] > G.graph["distance_dict"][2][0]
    assert data.distances_miles[1][100] < data.distances_miles[0][100]
    assert data.closer_neighbors[0][100] == frozenset({1})


def test_aggregated_level_dissolves_all_member_geometries(tmp_path, scenario_factory):
    G = nx.path_graph(2)
    G.nodes[0]["block_ids"] = [10, 12]
    G.nodes[1]["block_ids"] = [11]
    store = CloserNeighborArtifactStore(
        _scenario(tmp_path, scenario_factory),
        geometry_loader=lambda unit: _base_geometry(),
        school_loader=_schools,
    )

    data = store.for_graph("Block_1", G)

    assert data.distances_miles[0][100] == 0.0
    assert data.closer_neighbors[1][100] == frozenset({0})


def test_cache_is_shared_by_level_and_reused_without_source_loaders(
    tmp_path, scenario_factory
):
    G = _graph()
    scenario = _scenario(tmp_path, scenario_factory)
    first = CloserNeighborArtifactStore(
        scenario,
        geometry_loader=lambda unit: _base_geometry(),
        school_loader=_schools,
    )
    expected = first.for_graph("Block_0", G)
    path = first.cache_path("Block_0")

    def fail(*args, **kwargs):
        raise AssertionError("valid cache should not reload source data")

    cached = CloserNeighborArtifactStore(
        scenario,
        geometry_loader=fail,
        school_loader=fail,
    ).for_graph("Block_0", G)

    assert cached == expected
    assert path == (
        tmp_path
        / "cache"
        / "closer_neighbors"
        / "v3"
        / "closer_neighbors_Block_0.pickle"
    )
    with path.open("rb") as file:
        payload = pickle.load(file)
    assert payload["schema_version"] == CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION
    assert payload["level"] == "Block_0"
    assert len(payload["variants"]) == 1


def test_level_file_keeps_distinct_graph_membership_variants(
    tmp_path, scenario_factory
):
    store = CloserNeighborArtifactStore(
        _scenario(tmp_path, scenario_factory),
        geometry_loader=lambda unit: _base_geometry(),
        school_loader=_schools,
    )
    first = nx.path_graph(2)
    first.nodes[0]["block_ids"] = [10, 12]
    first.nodes[1]["block_ids"] = [11]
    second = nx.path_graph(2)
    second.nodes[0]["block_ids"] = [10]
    second.nodes[1]["block_ids"] = [11, 12]

    store.for_graph("Block_1", first)
    store.for_graph("Block_1", second)

    with store.cache_path("Block_1").open("rb") as file:
        payload = pickle.load(file)
    assert len(payload["variants"]) == 2


def test_cache_schema_version_invalidates_stale_in_file_data(
    tmp_path, scenario_factory
):
    G = _graph()
    scenario = _scenario(tmp_path, scenario_factory)
    path = (
        tmp_path
        / "cache"
        / "closer_neighbors"
        / "v3"
        / "closer_neighbors_Block_0.pickle"
    )
    path.parent.mkdir(parents=True)
    with path.open("wb") as file:
        pickle.dump(
            {
                "schema_version": CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION - 1,
                "level": "Block_0",
                "unit": "Block",
                "variants": {"stale": {}},
            },
            file,
        )
    store = CloserNeighborArtifactStore(
        scenario,
        geometry_loader=lambda unit: _base_geometry(),
        school_loader=_schools,
    )

    data = store.for_graph("Block_0", G)

    assert data.closer_neighbors[0][100] == frozenset({1})
    with path.open("rb") as file:
        payload = pickle.load(file)
    assert payload["schema_version"] == CLOSER_NEIGHBOR_CACHE_SCHEMA_VERSION
    assert "stale" not in payload["variants"]


def test_source_bytes_are_part_of_closer_neighbor_variant_identity(
    tmp_path, scenario_factory
):
    scenario = _scenario(tmp_path, scenario_factory)
    first = CloserNeighborArtifactStore(
        scenario,
        geometry_loader=lambda unit: _base_geometry(),
        school_loader=_schools,
    )
    first.for_graph("Block_0", _graph())

    scenario.source("optimization.schools").path.write_text(
        "school_id,lat,lon\n100,37.7500,-122.5000\n",
        encoding="utf-8",
    )
    second = CloserNeighborArtifactStore(
        scenario,
        geometry_loader=lambda unit: _base_geometry(),
        school_loader=_schools,
    )
    second.for_graph("Block_0", _graph())

    with second.cache_path("Block_0").open("rb") as file:
        payload = pickle.load(file)
    assert len(payload["variants"]) == 2
    assert (
        len({variant["source_fingerprint"] for variant in payload["variants"].values()})
        == 2
    )


def test_crosswalk_bytes_are_part_of_closer_neighbor_variant_identity(
    tmp_path, scenario_factory
):
    scenario = _scenario(tmp_path, scenario_factory)
    first = CloserNeighborArtifactStore(
        scenario,
        geometry_loader=lambda unit: _base_geometry(),
        school_loader=_schools,
    )
    first.for_graph("Block_0", _graph())

    scenario.source("optimization.crosswalk").path.write_text(
        "Block,BlockGroup\n10,2\n", encoding="utf-8"
    )
    second = CloserNeighborArtifactStore(
        scenario,
        geometry_loader=lambda unit: _base_geometry(),
        school_loader=_schools,
    )
    second.for_graph("Block_0", _graph())

    with second.cache_path("Block_0").open("rb") as file:
        payload = pickle.load(file)
    assert len(payload["variants"]) == 2
    assert (
        len({variant["source_fingerprint"] for variant in payload["variants"].values()})
        == 2
    )


@pytest.mark.parametrize(
    "level_name",
    [
        "Block_0",
        "Block_1",
        "Block_2",
        "Block_3",
        "Block_4",
        "BlockGroup_0",
        "BlockGroup_1",
        "BlockGroup_2",
    ],
)
def test_artifact_supports_every_predefined_level(
    tmp_path, level_name, scenario_factory
):
    level = LevelSpec.parse(level_name)
    geometry = _base_geometry().rename(columns={"Block": level.unit})
    if level.is_base:
        G = _graph()
    else:
        G = nx.path_graph(2)
        G.nodes[0]["block_ids"] = [10, 12]
        G.nodes[1]["block_ids"] = [11]
    store = CloserNeighborArtifactStore(
        _scenario(tmp_path, scenario_factory),
        geometry_loader=lambda unit: geometry,
        school_loader=_schools,
    )

    data = store.for_graph(level, G)

    assert set(data.closer_neighbors) == set(G.nodes())
    assert store.cache_path(level).exists()

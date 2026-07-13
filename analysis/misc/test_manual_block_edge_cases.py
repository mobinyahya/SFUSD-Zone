import json

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pytest
from shapely.geometry import box

from analysis.misc.manual_block_edge_cases import (
    BASE_RADIUS_MILES,
    build_manifest,
    compile_edge_additions,
    compile_edge_additions_file,
    compile_file,
    compile_selections,
    enumerate_cases,
    render_case_plot,
    _set_local_bounds,
)


def _graph():
    G = nx.path_graph(3)
    for node in G:
        G.nodes[node].update(
            area_id=1000 + node,
            lat=37.0,
            lon=-122.0 + node * 0.001,
        )
    G.graph["distance_dict"] = {
        0: {0: 0.0, 1: 2.0, 2: 1.0},
        1: {0: 2.0, 1: 0.0, 2: 1.0},
        2: {0: 1.0, 1: 1.0, 2: 0.0},
    }
    G.graph["closer_neighbors"] = {
        0: {100: frozenset(), 200: frozenset()},
        1: {100: frozenset({0}), 200: frozenset({0})},
        2: {100: frozenset(), 200: frozenset()},
    }
    G.graph["school_geometry_distances_miles"] = {
        0: {100: 0.0, 200: 0.0},
        1: {100: 2.0, 200: 2.0},
        2: {100: 1.0, 200: 1.0},
    }
    return G


def test_enumerate_cases_treats_each_node_school_pair_separately():
    G = _graph()

    cases = enumerate_cases(G, {100: 0, 200: 0})

    assert [(case["case_number"], case["focal_node"], case["school_id"]) for case in cases] == [
        (1, 2, 100),
        (2, 2, 200),
    ]


def test_manifest_defaults_to_existing_graph_neighbors_only():
    manifest = build_manifest(_graph(), {100: 0})
    case = manifest["cases"][0]

    assert manifest["include_nearby_non_neighbors"] is False
    assert case["plot_radius_miles"] is None
    assert case["centroid_label"] is None
    assert case["closer_candidate_labels"] == []
    assert {info["node"] for info in case["labels"].values()} == {1, 2}


def test_optional_radius_labels_closer_missing_nodes_and_compiles_stable_ids():
    manifest = build_manifest(
        _graph(),
        {100: 0},
        include_nearby_non_neighbors=True,
    )
    case = manifest["cases"][0]
    target_label = next(
        label
        for label, info in case["labels"].items()
        if info["node"] == 0
    )

    assert case["case_number"] == 1
    assert case["focal_label"] == 1
    assert case["plot_radius_miles"] == BASE_RADIUS_MILES
    assert int(target_label) in case["closer_candidate_labels"]

    edges, provenance = compile_selections(manifest, {1: [int(target_label)]})

    assert edges == [[1000, 1002]]
    assert provenance == {"1000:1002": [1]}


def test_optional_radius_allows_case_without_globally_closer_node():
    G = _graph()
    G.graph["school_geometry_distances_miles"][0][100] = 0.1
    G.graph["school_geometry_distances_miles"][2][100] = 0.0

    manifest = build_manifest(
        G,
        {100: 0},
        include_nearby_non_neighbors=True,
    )
    case = manifest["cases"][0]

    assert case["plot_radius_miles"] == BASE_RADIUS_MILES
    assert case["nearest_closer_endpoint_miles"] is None
    assert case["closer_candidate_labels"] == []


def test_compile_selections_rejects_existing_neighbor():
    manifest = build_manifest(
        _graph(),
        {100: 0},
        include_nearby_non_neighbors=True,
    )
    neighbor_label = manifest["cases"][0]["existing_neighbor_labels"][0]

    with pytest.raises(ValueError, match="strictly closer non-neighbor"):
        compile_selections(manifest, {1: [neighbor_label]})


def test_compile_selections_rejects_old_closer_neighbor_definition():
    manifest = build_manifest(_graph(), {100: 0})
    manifest["schema_version"] -= 1

    with pytest.raises(ValueError, match="obsolete closer-neighbor definition"):
        compile_selections(manifest, {})


def test_compile_file_writes_production_override(tmp_path):
    manifest = build_manifest(
        _graph(),
        {100: 0},
        include_nearby_non_neighbors=True,
    )
    case = manifest["cases"][0]
    target_label = case["closer_candidate_labels"][0]
    manifest_path = tmp_path / "manifest.json"
    selections_path = tmp_path / "selections.yaml"
    output_path = tmp_path / "edges.yaml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    selections_path.write_text(f"1: [{target_label}]\n", encoding="utf-8")

    count = compile_file(manifest_path, selections_path, output_path)

    assert count == 1
    assert "1000" in output_path.read_text(encoding="utf-8")
    assert "1002" in output_path.read_text(encoding="utf-8")


def test_compile_edge_additions_normalizes_and_deduplicates():
    edges = compile_edge_additions(
        {
            1002: [1000, 1001],
            1000: [1002],
        }
    )

    assert edges == [[1000, 1002], [1001, 1002]]


def test_compile_edge_additions_rejects_self_edges():
    with pytest.raises(ValueError, match="self-edge"):
        compile_edge_additions({1000: [1000]})


def test_compile_edge_additions_file_writes_separate_override(tmp_path):
    additions_path = tmp_path / "additions.yaml"
    output_path = tmp_path / "compiled.yaml"
    additions_path.write_text("1002: [1000]\n", encoding="utf-8")

    count = compile_edge_additions_file(additions_path, output_path)

    assert count == 1
    assert output_path.read_text(encoding="utf-8") == (
        "edges:\n- - 1000\n  - 1002\n"
    )


@pytest.mark.parametrize("include_nearby_non_neighbors", [False, True])
def test_render_case_plot_writes_numbered_png(
    tmp_path, include_nearby_non_neighbors
):
    G = _graph()
    manifest = build_manifest(
        G,
        {100: 0},
        include_nearby_non_neighbors=include_nearby_non_neighbors,
    )
    geometry = gpd.GeoDataFrame(
        {
            "node": [0, 1, 2],
            "geometry": [
                box(-122.001, 36.999, -122.0005, 37.001),
                box(-122.0005, 36.999, -121.9995, 37.001),
                box(-121.9995, 36.999, -121.9985, 37.001),
            ],
        },
        crs="EPSG:4326",
    )
    output = tmp_path / f"case_{include_nearby_non_neighbors}.png"

    render_case_plot(manifest["cases"][0], G, geometry, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_local_bounds_include_complete_neighbor_geometry():
    G = _graph()
    geometry = gpd.GeoDataFrame(
        {
            "node": [1, 2],
            "geometry": [
                box(-123.0, 36.5, -121.5, 37.5),
                box(-122.0, 36.9, -121.9, 37.1),
            ],
        },
        crs="EPSG:4326",
    )
    fig, ax = plt.subplots()

    _set_local_bounds(ax, geometry, G, [1, 2])

    assert ax.get_xlim()[0] < -123.0
    assert ax.get_xlim()[1] > -121.5
    assert ax.get_ylim()[0] < 36.5
    assert ax.get_ylim()[1] > 37.5
    plt.close(fig)

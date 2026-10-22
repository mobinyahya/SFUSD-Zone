import geopandas as gpd
import networkx as nx
from shapely.geometry import box

from optimization.data.edge_weights import (
    BOUNDARY_WEIGHT_ATTR,
    MANUAL_EDGE_ATTR,
    SHARED_BOUNDARY_ATTR,
    assign_boundary_weights,
)


def test_boundary_weights_measure_shared_edges_and_weight_manual_bridges():
    graph = nx.Graph()
    graph.add_nodes_from((node, {"area_id": 1000 + node}) for node in range(4))
    graph.add_edges_from([(0, 1), (1, 2), (0, 3)])
    geometry = gpd.GeoDataFrame(
        {
            "Block": [1000, 1001, 1002, 1003],
            "geometry": [
                box(0, 0, 10, 10),
                box(10, 0, 20, 10),
                box(20, 10, 30, 20),
                box(100, 100, 110, 110),
            ],
        },
        crs="EPSG:32610",
    )

    assign_boundary_weights(
        graph,
        geometry,
        "Block",
        source_edges={(0, 1), (1, 2)},
        manual_area_edges=[(1000, 1003)],
    )

    assert graph.edges[0, 1][SHARED_BOUNDARY_ATTR] == 10
    assert graph.edges[0, 1][BOUNDARY_WEIGHT_ATTR] == 10
    assert graph.edges[0, 1][MANUAL_EDGE_ATTR] is False
    assert graph.edges[1, 2][SHARED_BOUNDARY_ATTR] == 0
    assert graph.edges[1, 2][BOUNDARY_WEIGHT_ATTR] == 1
    assert graph.edges[0, 3][SHARED_BOUNDARY_ATTR] == 0
    assert graph.edges[0, 3][BOUNDARY_WEIGHT_ATTR] == 10
    assert graph.edges[0, 3][MANUAL_EDGE_ATTR] is True
    assert graph.graph["manual_edge_weight_m"] == 10

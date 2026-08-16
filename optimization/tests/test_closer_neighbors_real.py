"""Source-backed closer-neighbor coverage for both Census Block vintages."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pytest

from Config.Constants import AUX_BG
from optimization.config import OptimizationConfig
from optimization.data.closer_neighbors import (
    SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY,
)
from optimization.data.dataset import Dataset
from optimization.data.loaders import load_area_table, load_census_shapefile

SHARED_DATA_ROOT = Path("/share/data/school_choice")

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not SHARED_DATA_ROOT.is_dir(),
        reason="shared SFUSD source data is unavailable",
    ),
]


@lru_cache
def _dataset(vintage: str) -> Dataset:
    return Dataset(
        OptimizationConfig(
            levels=["Block_0"],
            data={
                "scenario": "legacy",
                "overrides": {
                    "filters": {"optimization": {"geography_vintage": vintage}}
                },
            },
        )
    )


def _empty_supports(vintage: str) -> tuple[nx.Graph, dict[int, list[dict]]]:
    dataset = _dataset(vintage)
    graph = dataset.graph_for("Block_0")
    closer_neighbors = dataset.closer_neighbors_for("Block_0")
    distances = graph.graph[SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY]
    first_node = next(iter(graph.nodes()))
    school_ids = tuple(sorted(distances[first_node]))
    centroid_nodes = {
        school_id: dataset.centroids_for("Block_0", [school_id])[0]
        for school_id in school_ids
    }

    assert set(closer_neighbors) == set(graph.nodes())
    failures: dict[int, list[dict]] = defaultdict(list)
    for node in sorted(graph.nodes()):
        block_id = int(graph.nodes[node]["area_id"])
        for school_id in school_ids:
            if node == centroid_nodes[school_id]:
                continue
            if not closer_neighbors[node][school_id]:
                neighbors = sorted(graph.neighbors(node))
                nearest_neighbor = min(
                    neighbors,
                    key=lambda neighbor: (
                        distances[neighbor][school_id],
                        int(graph.nodes[neighbor]["area_id"]),
                    ),
                    default=None,
                )
                failures[block_id].append(
                    {
                        "school_id": school_id,
                        "centroid_block": int(
                            graph.nodes[centroid_nodes[school_id]]["area_id"]
                        ),
                        "distance": distances[node][school_id],
                        "degree": len(neighbors),
                        "nearest_neighbor_block": (
                            int(graph.nodes[nearest_neighbor]["area_id"])
                            if nearest_neighbor is not None
                            else None
                        ),
                        "nearest_neighbor_distance": (
                            distances[nearest_neighbor][school_id]
                            if nearest_neighbor is not None
                            else None
                        ),
                    }
                )
    return graph, dict(failures)


def test_2020_auxiliary_blocks_are_filtered_from_optimization_data_and_graph():
    dataset = _dataset("2020")
    census = load_census_shapefile("Block", dataset.data)
    census_ids = set(census["Block"].astype("int64"))
    present_auxiliary_ids = census_ids & set(AUX_BG)

    area = load_area_table(dataset.ingest)
    area_ids = set(area["Block"].astype("int64"))
    graph = dataset.graph_for("Block_0")
    graph_ids = {int(attributes["area_id"]) for _, attributes in graph.nodes(data=True)}

    assert present_auxiliary_ids == set(AUX_BG[:3])
    assert census_ids - area_ids == present_auxiliary_ids
    assert census_ids - graph_ids == present_auxiliary_ids


@pytest.mark.parametrize("vintage", ["2010", "2020"])
def test_every_non_centroid_block_school_pair_has_a_closer_neighbor(vintage):
    graph, failures = _empty_supports(vintage)
    if not failures:
        return

    pair_count = sum(len(cases) for cases in failures.values())
    school_ids = sorted(
        {case["school_id"] for cases in failures.values() for case in cases}
    )
    detail_lines = []
    for block, cases in sorted(failures.items()):
        schools = sorted(case["school_id"] for case in cases)
        if cases[0]["degree"] == 0:
            detail_lines.append(f"  Block {block:015d}, degree=0: schools {schools}")
            continue
        if len(cases) > 10:
            case = cases[0]
            neighbor_distance = case["nearest_neighbor_distance"]
            detail_lines.append(
                f"  Block {block:015d}, degree={case['degree']}: schools "
                f"{schools}; example school={case['school_id']}, "
                f"centroid_block={case['centroid_block']:015d}, "
                f"distance={case['distance']:.6f} miles, "
                f"nearest_adjacent_block={case['nearest_neighbor_block']:015d}, "
                f"nearest_adjacent_distance={neighbor_distance:.6f} miles, "
                f"delta={neighbor_distance - case['distance']:.6f} miles"
            )
            continue
        for case in cases:
            neighbor_distance = case["nearest_neighbor_distance"]
            detail_lines.append(
                f"  Block {block:015d}, degree={case['degree']}, "
                f"school={case['school_id']}, "
                f"centroid_block={case['centroid_block']:015d}, "
                f"distance={case['distance']:.6f} miles, "
                f"nearest_adjacent_block={case['nearest_neighbor_block']:015d}, "
                f"nearest_adjacent_distance={neighbor_distance:.6f} miles, "
                f"delta={neighbor_distance - case['distance']:.6f} miles"
            )
    details = "\n".join(detail_lines)
    pytest.fail(
        f"Census {vintage} has {pair_count} non-centroid Block-school pairs with "
        f"no strictly closer adjacent Block across {len(failures)} of "
        f"{graph.number_of_nodes()} Blocks and {len(school_ids)} schools:\n{details}",
        pytrace=False,
    )

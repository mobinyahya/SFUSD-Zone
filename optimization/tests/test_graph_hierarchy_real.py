"""In-memory hierarchy verification against real SFUSD base graphs."""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path

import networkx as nx
import pytest

from Config.Constants import AREA_ETHNICITIES
from optimization.data import graph_builder
from optimization.levels import LEVEL_NODE_TARGETS


pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        os.environ.get("SFUSD_RUN_REAL_GRAPH_HIERARCHY") != "1",
        reason="set SFUSD_RUN_REAL_GRAPH_HIERARCHY=1 for the cached real-data test",
    ),
]


def _real_base_graph(unit: str) -> nx.Graph:
    roots = [
        Path("/share/data/school_choice/Zones/Optimization/Graphs"),
        Path.home()
        / "sfusd-local-data/zones/SFUSD/Optimization/Zones/Graphs/optimization",
    ]
    required = set(AREA_ETHNICITIES) | {
        "area_id",
        "ge_students",
        "all_prog_students",
        "school_ids",
    }
    for root in roots:
        for path in sorted(root.glob(f"{unit}_*/{unit}_0.pickle")):
            with path.open("rb") as file:
                graph = pickle.load(file)
            _, attrs = next(iter(graph.nodes(data=True)))
            if required <= set(attrs):
                return graph
    pytest.skip(f"current real-data {unit}_0 graph is unavailable")


def _base_ids(attrs: dict) -> list[int]:
    if "area_id" in attrs:
        return [int(attrs["area_id"])]
    return [int(area_id) for area_id in attrs["block_ids"]]


def _assert_valid_child(parent: nx.Graph, child: nx.Graph, target: int) -> None:
    partition = child.graph["partition"]
    assert set(partition) == set(parent)
    assert len(child) <= target
    assert len(child) < len(parent)
    assert child.graph["target_node_count"] == target
    assert child.graph["actual_node_count"] == len(child)
    assert child.graph["partition_backend"] == "kahip"
    assert child.graph["partition_mode"] == "strong"
    assert math.isclose(
        sum(attrs["ge_students"] for _, attrs in parent.nodes(data=True)),
        sum(attrs["ge_students"] for _, attrs in child.nodes(data=True)),
    )

    parent_base_ids = sorted(
        area_id for _, attrs in parent.nodes(data=True) for area_id in _base_ids(attrs)
    )
    child_base_ids = sorted(
        area_id for _, attrs in child.nodes(data=True) for area_id in _base_ids(attrs)
    )
    assert child_base_ids == parent_base_ids

    school_parent_nodes = {
        node for node, attrs in parent.nodes(data=True) if attrs["school_ids"]
    }
    school_child_nodes = {
        node for node, attrs in child.nodes(data=True) if attrs["school_ids"]
    }
    assert len(school_child_nodes) == len(school_parent_nodes)
    assert child.graph["school_singleton_count"] == len(school_parent_nodes)
    for node in school_parent_nodes:
        child_node = partition[node]
        assert len(_base_ids(child.nodes[child_node])) == len(
            _base_ids(parent.nodes[node])
        )
        assert child.nodes[child_node]["school_ids"] == parent.nodes[node]["school_ids"]

    non_school = parent.subgraph(set(parent) - school_parent_nodes).copy()
    components = [
        sorted(component) for component in nx.connected_components(non_school)
    ]
    components.sort(key=lambda component: (-len(component), component[0]))
    requested_counts = graph_builder._component_partition_counts(
        non_school,
        components,
        target - len(school_parent_nodes),
        child.graph["partition_population_attribute"],
    )
    imbalance = child.graph["partition_imbalance"]
    for component, requested_count in zip(components, requested_counts):
        weights = graph_builder._integer_population_weights(
            parent,
            component,
            child.graph["partition_population_attribute"],
        )
        totals: dict[int, int] = {}
        for node, weight in zip(component, weights):
            child_node = partition[node]
            totals[child_node] = totals.get(child_node, 0) + weight
        upper_bound = (1 + imbalance) * math.ceil(sum(weights) / requested_count)
        assert max(totals.values()) <= upper_bound

    parents_by_child: dict[int, list[int]] = {}
    for parent_node, child_node in partition.items():
        parents_by_child.setdefault(child_node, []).append(parent_node)
    assert set(parents_by_child) == set(child)
    assert all(
        nx.is_connected(parent.subgraph(nodes)) for nodes in parents_by_child.values()
    )

    expected_edges = {
        tuple(sorted((partition[u], partition[v])))
        for u, v in parent.edges()
        if partition[u] != partition[v]
    }
    assert {tuple(sorted(edge)) for edge in child.edges()} == expected_edges


@pytest.mark.parametrize("unit", ["Block", "BlockGroup"])
def test_real_graph_hierarchy_is_nested_and_school_preserving(unit):
    parent = _real_base_graph(unit)
    for _, target in sorted(LEVEL_NODE_TARGETS[unit].items()):
        child = graph_builder.aggregate_level(parent, target, "GE")
        _assert_valid_child(parent, child, target)
        parent = child

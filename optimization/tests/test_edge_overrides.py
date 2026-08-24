import networkx as nx
import pytest

from loaders import edge_overrides
from optimization.config import OptimizationConfig
from optimization.data import graph_builder
from optimization.data.dataset import Dataset
from optimization.solution import graph_fingerprint
from optimization.tests.synthetic import make_grid_graph


def test_load_block_edge_overrides_normalizes_and_deduplicates(tmp_path):
    path = tmp_path / "edges.yaml"
    path.write_text(
        "edges:\n  - [1002, 1000]\n  - [1000, 1002]\n  - [1001, 1002]\n",
        encoding="utf-8",
    )

    assert edge_overrides.load_block_edge_overrides(path) == [
        (1000, 1002),
        (1001, 1002),
    ]


def test_load_block_edge_overrides_rejects_self_edges(tmp_path):
    path = tmp_path / "edges.yaml"
    path.write_text("edges:\n  - [1000, 1000]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="self-edge"):
        edge_overrides.load_block_edge_overrides(path)


def test_default_block_edge_overrides_merge_reviewed_and_explicit_files(
    tmp_path, monkeypatch
):
    reviewed = tmp_path / "reviewed.yaml"
    additions = tmp_path / "additions.yaml"
    reviewed.write_text("edges:\n  - [1000, 1002]\n", encoding="utf-8")
    additions.write_text(
        "edges:\n  - [1002, 1000]\n  - [1001, 1002]\n", encoding="utf-8"
    )
    monkeypatch.setattr(edge_overrides, "DEFAULT_BLOCK_EDGE_OVERRIDES", reviewed)
    monkeypatch.setattr(edge_overrides, "DEFAULT_BLOCK_EDGE_ADDITIONS", additions)

    assert edge_overrides.load_block_edge_overrides() == [
        (1000, 1002),
        (1001, 1002),
    ]


def test_apply_block_edge_overrides_resolves_stable_area_ids():
    G = nx.Graph()
    G.add_node(20, area_id=1000)
    G.add_node(10, area_id=1001)

    applied = edge_overrides.apply_block_edge_overrides(G, [(1000, 1001)])

    assert applied == [(1000, 1001)]
    assert G.has_edge(20, 10)
    assert G.graph["manual_block_edges"] == [(1000, 1001)]


def test_manual_edges_propagate_through_aggregation():
    base = make_grid_graph(1, 3)
    edge_overrides.apply_block_edge_overrides(base, [(1000, 1002)])

    coarse = graph_builder.aggregate(base, {0: 0, 1: 1, 2: 2})

    assert coarse.has_edge(0, 2)
    assert coarse.graph["manual_block_edges"] == [(1000, 1002)]


def test_explicit_edge_additions_change_block_cache_namespace(
    tmp_path, data_config_factory
):
    reviewed = tmp_path / "reviewed.yaml"
    additions = tmp_path / "additions.yaml"
    reviewed.write_text("edges: []\n", encoding="utf-8")
    additions.write_text("edges: []\n", encoding="utf-8")
    data = data_config_factory(
        sources={
            "optimization.manual_edges": {
                "reviewed": {"path": str(reviewed)},
                "additions": {"path": str(additions)},
            }
        }
    )
    baseline = Dataset(OptimizationConfig(levels=["Block_0"], data=data))

    additions.write_text("edges:\n  - [1000, 1001]\n", encoding="utf-8")
    changed = Dataset(OptimizationConfig(levels=["Block_0"], data=data))

    assert changed._graph_cache_namespace() != baseline._graph_cache_namespace()


def test_graph_fingerprint_changes_when_topology_changes():
    first = nx.Graph()
    first.add_nodes_from([(0, {"area_id": 1000}), (1, {"area_id": 1001})])
    second = first.copy()
    second.add_edge(0, 1)

    assert graph_fingerprint(first) != graph_fingerprint(second)


def test_graph_fingerprint_tracks_weighted_boundary_cost():
    first = nx.Graph()
    first.add_nodes_from([(0, {"area_id": 1000}), (1, {"area_id": 1001})])
    first.add_edge(0, 1, boundary_weight=10)
    first.graph["weight_edges"] = True
    second = first.copy()
    second.edges[0, 1]["boundary_weight"] = 11

    assert graph_fingerprint(first) != graph_fingerprint(second)

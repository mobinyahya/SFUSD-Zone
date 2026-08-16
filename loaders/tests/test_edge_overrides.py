from __future__ import annotations

import networkx as nx
import pytest

from loaders import edge_overrides
from loaders.edge_overrides import (
    apply_block_edge_overrides,
    block_edge_override_fingerprint,
    load_block_edge_overrides,
)

EXPECTED_EDGES = [
    (60750101001001, 60750179021015),
    (60750102001001, 60750102001003),
    (60750105001001, 60750179021001),
    (60750254031020, 60750254031021),
    (60750301022004, 60750301022005),
    (60750601001167, 60750601001168),
    (60750607001003, 60750607001017),
    (60750607001003, 60750607001021),
    (60750607001007, 60750607001021),
    (60750607001007, 60750607001033),
    (60750607001030, 60750607001049),
    (60750607001030, 60750607001053),
    (60750607001031, 60750607001053),
    (60750607001036, 60750607001048),
    (60759806001001, 60759806001031),
    (60759806001003, 60759806001031),
    (60759809001002, 60759809001040),
]


def test_default_manual_edges_are_all_17_canonical_edges():
    assert load_block_edge_overrides() == EXPECTED_EDGES
    assert len(load_block_edge_overrides()) == 17
    assert len(block_edge_override_fingerprint()) == 12


def test_missing_explicit_manual_edge_file_is_fatal(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required manual Block edge file"):
        load_block_edge_overrides(tmp_path / "missing.yaml")


def test_missing_default_manual_edge_file_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        edge_overrides,
        "DEFAULT_BLOCK_EDGE_OVERRIDES",
        tmp_path / "missing-default.yaml",
    )

    with pytest.raises(FileNotFoundError, match="missing-default.yaml"):
        load_block_edge_overrides()


def test_present_empty_manual_edge_file_is_valid(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("edges: []\n", encoding="utf-8")

    assert load_block_edge_overrides(path) == []


def test_apply_manual_edges_preserves_graph_metadata():
    graph = nx.Graph()
    area_ids = sorted({area for edge in EXPECTED_EDGES for area in edge})
    graph.add_nodes_from(
        (index, {"area_id": area_id}) for index, area_id in enumerate(area_ids)
    )

    applied = apply_block_edge_overrides(graph)

    assert applied == EXPECTED_EDGES
    assert graph.number_of_edges() == 17
    assert graph.graph["manual_block_edges"] == EXPECTED_EDGES
    assert (
        graph.graph["manual_block_edge_fingerprint"]
        == block_edge_override_fingerprint()
    )

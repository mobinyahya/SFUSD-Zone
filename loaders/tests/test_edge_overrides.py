from __future__ import annotations

import networkx as nx
import pytest

from loaders.config import PACKAGE_CONFIG_ROOT
from loaders.edge_overrides import (
    apply_block_edge_overrides,
    load_block_edge_overrides,
)

PACKAGED_EDGE_FILES = (
    PACKAGE_CONFIG_ROOT / "manual_block_edges.yaml",
    PACKAGE_CONFIG_ROOT / "manual_block_edge_additions.yaml",
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


def _packaged_edges() -> list[tuple[int, int]]:
    return sorted(
        {
            edge
            for path in PACKAGED_EDGE_FILES
            for edge in load_block_edge_overrides(path)
        }
    )


def test_packaged_2010_manual_edges_are_all_17_canonical_edges():
    assert _packaged_edges() == EXPECTED_EDGES


def test_missing_explicit_manual_edge_file_is_fatal(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required manual Block edge file"):
        load_block_edge_overrides(tmp_path / "missing.yaml")


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

    applied = apply_block_edge_overrides(graph, _packaged_edges())

    assert applied == EXPECTED_EDGES
    assert graph.number_of_edges() == 17
    assert graph.graph["manual_block_edges"] == EXPECTED_EDGES
    assert len(graph.graph["manual_block_edge_fingerprint"]) == 12

import networkx as nx
import pandas as pd

from Zone_Generation.optimization.data import conversion
from Zone_Generation.optimization.data.conversion import (
    LevelConverter,
    base_area_assignment,
)
from Zone_Generation.optimization.levels import LevelSpec
from Zone_Generation.optimization.tests.synthetic import make_path_graphs

BG0 = LevelSpec("BlockGroup", 0)
BG1 = LevelSpec("BlockGroup", 1)
BLOCK0 = LevelSpec("Block", 0)


def test_base_area_assignment_expands_blocks():
    _, coarse = make_path_graphs()
    area = base_area_assignment(coarse, {0: 0, 1: 1})
    assert area == {10: 0, 11: 0, 12: 1, 13: 1}


def test_fine_to_coarse():
    base, coarse = make_path_graphs()
    conv = LevelConverter()
    fine_assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    coarse_assignment = conv.between(base, fine_assignment, BG0, coarse, BG1)
    assert coarse_assignment == {0: 0, 1: 1}


def test_coarse_to_fine_roundtrip():
    base, coarse = make_path_graphs()
    conv = LevelConverter()
    coarse_assignment = {0: 0, 1: 1}
    fine = conv.between(coarse, coarse_assignment, BG1, base, BG0)
    assert fine == {0: 0, 1: 0, 2: 1, 3: 1}
    # round-trip back to coarse
    back = conv.between(base, fine, BG0, coarse, BG1)
    assert back == coarse_assignment


def test_blockgroup_to_block_uses_injected_crosswalk():
    source = nx.Graph()
    source.add_nodes_from([(0, {"area_id": 100}), (1, {"area_id": 200})])
    target = nx.Graph()
    target.add_nodes_from(
        [
            (0, {"area_id": 1001}),
            (1, {"area_id": 1002}),
            (2, {"area_id": 2001}),
        ]
    )
    conv = LevelConverter({1001: 100, 1002: 100, 2001: 200})

    result = conv.between(source, {0: 3, 1: 7}, BG0, target, BLOCK0)

    assert result == {0: 3, 1: 3, 2: 7}


def test_block_to_blockgroup_uses_majority_zone():
    source = nx.Graph()
    source.add_nodes_from(
        [
            (0, {"area_id": 1001}),
            (1, {"area_id": 1002}),
            (2, {"area_id": 1003}),
        ]
    )
    target = nx.Graph()
    target.add_node(0, area_id=100)
    conv = LevelConverter({1001: 100, 1002: 100, 1003: 100})

    result = conv.between(source, {0: 3, 1: 7, 2: 7}, BLOCK0, target, BG0)

    assert result == {0: 7}


def test_load_block_to_blockgroup(tmp_path, monkeypatch):
    optimization_dir = tmp_path / "Optimization"
    optimization_dir.mkdir()
    pd.DataFrame({"Block": [1001, 1002], "BlockGroup": [100, 100]}).to_csv(
        optimization_dir / "block_blockgroup_tract.csv", index=False
    )
    monkeypatch.setattr(conversion, "get_dropbox_path", lambda _: str(tmp_path))

    crosswalk = conversion._load_block_to_blockgroup()

    assert crosswalk == {1001: 100, 1002: 100}

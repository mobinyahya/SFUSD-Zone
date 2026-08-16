import networkx as nx
import pandas as pd

from optimization.data import conversion
from optimization.data.conversion import (
    LevelConverter,
    base_area_assignment,
)
from optimization.levels import LevelSpec
from optimization.tests.synthetic import make_path_graphs

BG0 = LevelSpec("BlockGroup", 0)
BG1 = LevelSpec("BlockGroup", 1)
BLOCK0 = LevelSpec("Block", 0)
TRACT0 = LevelSpec("Tract", 0)


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


def test_tract_and_block_conversion_uses_injected_crosswalk():
    tract = nx.Graph()
    tract.add_nodes_from([(0, {"area_id": 10}), (1, {"area_id": 20})])
    blocks = nx.Graph()
    blocks.add_nodes_from(
        [(0, {"area_id": 1001}), (1, {"area_id": 1002}), (2, {"area_id": 2001})]
    )
    conv = LevelConverter(
        {1001: 100, 1002: 100, 2001: 200},
        block_to_tract={1001: 10, 1002: 10, 2001: 20},
    )

    expanded = conv.between(tract, {0: 3, 1: 7}, TRACT0, blocks, BLOCK0)
    collapsed = conv.between(blocks, expanded, BLOCK0, tract, TRACT0)

    assert expanded == {0: 3, 1: 3, 2: 7}
    assert collapsed == {0: 3, 1: 7}


def test_load_block_to_blockgroup_uses_scenario_role(tmp_path, scenario_factory):
    crosswalk_path = tmp_path / "crosswalk.csv"
    pd.DataFrame({"Block": [1001, 1002], "BlockGroup": [100, 100]}).to_csv(
        crosswalk_path, index=False
    )
    scenario = scenario_factory(
        sources={"optimization.crosswalk": {"path": str(crosswalk_path)}}
    )

    crosswalk = conversion._load_block_to_blockgroup(scenario)

    assert crosswalk == {1001: 100, 1002: 100}

import networkx as nx
import pandas as pd
import pytest

from optimization.config import OptimizationConfig
from optimization.data import loaders
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec


def _config(tmp_path, **overrides):
    params = {
        "centroids_type": "6-zone-3",
        "levels": ["Block_0"],
        "graphs_dir": str(tmp_path),
    }
    params.update(overrides)
    return OptimizationConfig(**params)


@pytest.mark.parametrize(
    "override",
    [
        {"years": [14]},
        {"population_type": "All"},
        {"drop_optout": False},
        {"capacity_scenario": "B"},
        {"new_schools": False},
        {"include_k8": True},
        {"level_to_split": {1: 1, 2: 1}},
    ],
)
def test_graph_cache_path_changes_for_graph_data_parameters(tmp_path, override):
    level = LevelSpec.parse("Block_0")

    baseline = Dataset(_config(tmp_path))
    changed = Dataset(_config(tmp_path, **override))

    assert baseline._graph_path(level) != changed._graph_path(level)


def test_graph_cache_path_ignores_centroid_choice(tmp_path):
    level = LevelSpec.parse("Block_0")

    baseline = Dataset(_config(tmp_path, centroids_type="6-zone-3"))
    changed = Dataset(_config(tmp_path, centroids_type="8-zone-22"))

    assert baseline._graph_path(level) == changed._graph_path(level)


def test_centroids_fallback_to_raw_school_locations_for_aggregated_graph(
    tmp_path,
    monkeypatch,
):
    G = nx.Graph()
    G.add_node(10, school_ids=[], block_ids=[1000, 1001])
    G.add_node(20, school_ids=[], block_ids=[1002])

    dataset = Dataset(
        _config(
            tmp_path,
            centroids_type="raw-location-centroid",
            levels=["Block_1"],
        )
    )
    dataset._graphs["Block_1"] = G

    monkeypatch.setattr(
        loaders,
        "load_centroid_schools",
        lambda centroids_type: [999],
    )
    monkeypatch.setattr(
        loaders,
        "load_school_locations",
        lambda cfg: pd.DataFrame({"school_id": [999], "Block": [1001]}),
    )

    assert dataset.centroids_for("Block_1") == [10]
    assert G.nodes[10]["school_ids"] == []

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
        {"remove_city_wide": True},
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


def test_default_graph_root_uses_shared_optimization_graph_directory():
    config = OptimizationConfig(levels=["Block_0"])

    assert config.graphs_dir == "/share/data/school_choice/Zones/Optimization/Graphs"


def test_dataset_builds_each_level_from_its_immediate_parent(tmp_path, monkeypatch):
    base = nx.path_graph(4)
    middle = nx.path_graph(3)
    coarse = nx.path_graph(2)
    generated_from = []

    monkeypatch.setattr(loaders, "load_students", lambda cfg: None)
    monkeypatch.setattr(
        "optimization.data.graph_builder.build_base_graph",
        lambda cfg: base,
    )

    def fake_aggregate(parent, target, population_type):
        generated_from.append((parent, target, population_type))
        return middle if parent is base else coarse

    monkeypatch.setattr(
        "optimization.data.graph_builder.aggregate_level",
        fake_aggregate,
    )

    dataset = Dataset(_config(tmp_path, levels=["BlockGroup_2"]))
    result = dataset.graph_for("BlockGroup_2")

    assert result is coarse
    assert generated_from == [(base, 250, "GE"), (middle, 125, "GE")]


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


def test_problem_for_accepts_explicit_centroid_school_ids(tmp_path, monkeypatch):
    G = nx.path_graph(2)
    for node, school_id in enumerate((100, 200)):
        G.nodes[node].update(
            {
                "school_ids": [school_id],
                "num_schools": 1,
                "area_id": 1000 + node,
            }
        )
    dataset = Dataset(_config(tmp_path))
    dataset._graphs["Block_0"] = G
    monkeypatch.setattr(
        dataset._closer_neighbor_store,
        "attach_to_graph",
        lambda level, graph: None,
    )

    problem = dataset.problem_for("Block_0", centroid_school_ids=[200])

    assert dataset.school_ids_for("Block_0") == [100, 200]
    assert problem.centroids == [1]
    assert problem.centroid_school_ids == [200]

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from optimization.config import OptimizationConfig
from optimization.data import loaders
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec


def _data(cache_root, *, filters=None, sources=None):
    overrides = {"roots": {"cache": str(cache_root)}}
    if filters:
        overrides["filters"] = {"optimization": filters}
    if sources:
        overrides["sources"] = sources
    return {"scenario": "legacy", "overrides": overrides}


def _config(tmp_path, *, data_filters=None, data_sources=None, **overrides):
    params = {
        "centroids_type": "6-zone-3",
        "levels": ["Block_0"],
        "data": _data(
            tmp_path / "cache",
            filters=data_filters,
            sources=data_sources,
        ),
    }
    params.update(overrides)
    return OptimizationConfig(**params)


@pytest.mark.parametrize(
    "data_filters",
    [
        {"years": ["1415"]},
        {"grades": ["01"]},
        {"student_population": "applicant"},
        {"rounds": [1, 2, 4]},
        {"special_programs": "exclude_any_special"},
        {"program_population": "All"},
        {"capacity_scenario": "B"},
        {"include_k8": True},
        {"include_citywide": True},
        {"include_mission_bay": False},
    ],
)
def test_graph_cache_path_changes_for_graph_data_parameters(tmp_path, data_filters):
    level = LevelSpec.parse("Block_0")

    baseline = Dataset(_config(tmp_path))
    changed = Dataset(_config(tmp_path, data_filters=data_filters))

    assert baseline._graph_path(level) != changed._graph_path(level)


def test_graph_cache_path_ignores_centroid_choice(tmp_path):
    level = LevelSpec.parse("Block_0")

    baseline = Dataset(_config(tmp_path, centroids_type="6-zone-3"))
    changed = Dataset(_config(tmp_path, centroids_type="8-zone-22"))

    assert baseline._graph_path(level) == changed._graph_path(level)


def test_default_graph_root_uses_v12_shared_cache_namespace():
    config = OptimizationConfig(levels=["Block_0"])
    dataset = Dataset(config)

    assert dataset._graph_namespace.schema_version == 12
    assert dataset._graph_namespace.version_dir == Path(
        "/soalnas/share/data/school_choice/Data/caches/graphs/v12"
    )
    assert Path(dataset.graph_cache_dir).parent == dataset._graph_namespace.version_dir


def test_graph_cache_key_ignores_cache_root(tmp_path):
    first = OptimizationConfig(
        levels=["Block_0"], data=_data(tmp_path / "cache-one")
    )
    second = OptimizationConfig(
        levels=["Block_0"], data=_data(tmp_path / "cache-two")
    )

    first_dataset = Dataset(first)
    second_dataset = Dataset(second)

    assert first_dataset._graph_cache_namespace() == (
        second_dataset._graph_cache_namespace()
    )
    assert first_dataset.graph_cache_dir != second_dataset.graph_cache_dir


def test_graph_cache_key_changes_when_source_bytes_change(tmp_path):
    students = tmp_path / "enrolled_2122.csv"
    students.write_text("value\nfirst\n", encoding="utf-8")
    data = _data(
        tmp_path / "cache",
        filters={"years": ["2122"]},
        sources={"optimization.students": [{"path": str(students)}]},
    )
    first = Dataset(OptimizationConfig(levels=["Block_0"], data=data))

    students.write_text("value\nsecond-longer\n", encoding="utf-8")
    second = Dataset(OptimizationConfig(levels=["Block_0"], data=data))

    assert first._graph_cache_namespace() != second._graph_cache_namespace()


def test_graph_cache_key_tracks_selected_program_capacity_bytes(tmp_path):
    programs = tmp_path / "programs.csv"
    programs.write_text("program_id,capacity\n10-GE-KG,5\n", encoding="utf-8")
    data = _data(
        tmp_path / "cache",
        sources={"optimization.programs": {"path": str(programs)}},
    )
    first = Dataset(OptimizationConfig(levels=["Block_0"], data=data))

    programs.write_text("program_id,capacity\n10-GE-KG,6\n", encoding="utf-8")
    second = Dataset(OptimizationConfig(levels=["Block_0"], data=data))

    assert first._graph_cache_namespace() != second._graph_cache_namespace()


def test_program_capacity_mode_ignores_unused_scenario_source_bytes(tmp_path):
    capacities = tmp_path / "capacities.csv"
    capacities.write_text("Scenario_A_Capacity\n5\n", encoding="utf-8")
    sources = {"optimization.capacity": {"path": str(capacities)}}
    programs_first = Dataset(
        _config(tmp_path, data_sources=sources)
    )._graph_cache_namespace()
    scenario_first = Dataset(
        _config(
            tmp_path,
            data_filters={"capacity_scenario": "A"},
            data_sources=sources,
        )
    )._graph_cache_namespace()

    capacities.write_text("Scenario_A_Capacity\n6\n", encoding="utf-8")
    programs_second = Dataset(
        _config(tmp_path, data_sources=sources)
    )._graph_cache_namespace()
    scenario_second = Dataset(
        _config(
            tmp_path,
            data_filters={"capacity_scenario": "A"},
            data_sources=sources,
        )
    )._graph_cache_namespace()

    assert programs_first == programs_second
    assert scenario_first != scenario_second


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

    def fake_aggregate(parent, target, program_population):
        generated_from.append((parent, target, program_population))
        return middle if parent is base else coarse

    monkeypatch.setattr(
        "optimization.data.graph_builder.aggregate_level",
        fake_aggregate,
    )

    dataset = Dataset(_config(tmp_path, levels=["BlockGroup_2"]))
    result = dataset.graph_for("BlockGroup_2")

    assert result is coarse
    assert generated_from == [(base, 250, "GE"), (middle, 125, "GE")]


def test_graph_payload_is_saved_and_loaded_through_validated_manifest(
    tmp_path, monkeypatch
):
    graph = nx.path_graph(3)
    config = _config(tmp_path)
    monkeypatch.setattr(
        "optimization.data.graph_builder.build_base_graph", lambda cfg: graph
    )
    first = Dataset(config)

    assert first.graph_for("Block_0") is graph
    manifest = first._graph_namespace.manifest()
    assert manifest is not None
    assert manifest["schema_version"] == 12
    assert manifest["payloads"]["Block_0.pickle"]["format"] == "pickle"

    monkeypatch.setattr(
        "optimization.data.graph_builder.build_base_graph",
        lambda cfg: (_ for _ in ()).throw(
            AssertionError("validated graph payload should be reused")
        ),
    )
    loaded = Dataset(config).graph_for("Block_0")

    assert list(loaded.edges()) == list(graph.edges())


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
        lambda centroids_type, data: [999],
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

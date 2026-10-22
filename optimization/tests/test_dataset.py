from pathlib import Path
from copy import copy

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
        {"include_citywide_zoning": True},
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


def test_graph_cache_path_changes_for_weighted_edges(tmp_path):
    baseline = Dataset(_config(tmp_path, weight_edges=False))
    weighted = Dataset(_config(tmp_path, weight_edges=True))

    assert baseline._graph_cache_namespace() != weighted._graph_cache_namespace()
    assert (
        "edge_weighting" not in baseline._graph_namespace.parameters["partition_policy"]
    )
    assert "edge_weighting" in weighted._graph_namespace.parameters["partition_policy"]


def test_default_graph_root_uses_v14_shared_cache_namespace():
    config = OptimizationConfig(levels=["Block_0"])
    dataset = Dataset(config)

    assert dataset._graph_namespace.schema_version == 14
    assert dataset._graph_namespace.version_dir == Path(
        "/soalnas/share/data/school_choice/Data/caches/graphs/v14"
    )
    assert Path(dataset.graph_cache_dir).parent == dataset._graph_namespace.version_dir


def test_graph_cache_key_ignores_cache_root(tmp_path):
    first = OptimizationConfig(levels=["Block_0"], data=_data(tmp_path / "cache-one"))
    second = OptimizationConfig(levels=["Block_0"], data=_data(tmp_path / "cache-two"))

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
    # Real coarsenings carry area ids and a membership, which the dataset
    # publishes as the portable partition artifact.
    for index, node in enumerate(base):
        base.nodes[node]["area_id"] = 6075000000000 + index
    middle.graph["partition"] = {0: 0, 1: 1, 2: 1, 3: 2}
    for index, node in enumerate(middle):
        middle.nodes[node]["block_ids"] = [6075000000000 + index]
    coarse.graph["partition"] = {0: 0, 1: 0, 2: 1}

    monkeypatch.setattr(loaders, "load_students", lambda cfg: None)
    monkeypatch.setattr(
        "optimization.data.graph_builder.build_base_graph",
        lambda cfg, **kwargs: base,
    )

    def fake_aggregate(parent, target, program_population, *, partition=None):
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
        "optimization.data.graph_builder.build_base_graph", lambda cfg, **kwargs: graph
    )
    first = Dataset(config)

    assert first.graph_for("Block_0") is graph
    manifest = first._graph_namespace.manifest()
    assert manifest is not None
    assert manifest["schema_version"] == 14
    assert manifest["payloads"]["Block_0.pickle"]["format"] == "pickle"

    monkeypatch.setattr(
        "optimization.data.graph_builder.build_base_graph",
        lambda cfg, **kwargs: (_ for _ in ()).throw(
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
    dataset = Dataset(_config(tmp_path, weight_edges=True))
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
    assert problem.weight_edges is True


def test_problem_for_resolves_auto_max_distance(tmp_path, monkeypatch):
    G = nx.path_graph(3)
    G.graph["distance_dict"] = {
        0: {0: 0.0, 1: 2.5, 2: 7.5},
        2: {0: 7.5, 1: 5.0, 2: 0.0},
    }
    for node, school_id in enumerate((100, None, 200)):
        G.nodes[node].update(
            {
                "school_ids": [school_id] if school_id else [],
                "num_schools": 1 if school_id else 0,
                "area_id": 1000 + node,
            }
        )
    dataset = Dataset(_config(tmp_path, max_distance="auto"))
    dataset._graphs["Block_0"] = G
    monkeypatch.setattr(
        dataset._closer_neighbor_store,
        "attach_to_graph",
        lambda level, graph: None,
    )

    problem = dataset.problem_for("Block_0", centroid_school_ids=[100, 200])

    assert problem.centroids == [0, 2]
    assert problem.max_distance == 3.0
    assert problem.candidate_zones(1) == {0}


def _pipeline_dataset(tmp_path, monkeypatch, **overrides):
    graph = nx.path_graph(2)
    for node, school_id in enumerate((100, 200)):
        graph.nodes[node].update(school_ids=[school_id], area_id=1000 + node)
    dataset = Dataset(_config(tmp_path, **overrides))
    dataset._graphs["Block_0"] = graph
    attached = []
    monkeypatch.setattr(
        dataset._closer_neighbor_store,
        "attach_to_graph",
        lambda level, graph: attached.append(graph),
    )
    return dataset, graph, attached


def test_perimeter_and_unit_cut_steps_reuse_weighted_graph_data(tmp_path, monkeypatch):
    from optimization.pipeline import step_config

    dataset, graph, attached = _pipeline_dataset(
        tmp_path,
        monkeypatch,
        steps=[
            {"level": "Block_0", "objective": "boundary_perimeter"},
            {"level": "Block_0", "objective": "cut_edges"},
        ],
    )
    config = dataset.config
    problems = [
        dataset.problem_for(
            step.level,
            centroid_school_ids=[100],
            config=step_config(config, step, index),
        )
        for index, step in enumerate(config.steps)
    ]
    assert problems[0].G is problems[1].G is graph
    assert [problem.weight_edges for problem in problems] == [True, False]
    assert attached == [graph]
    assert dataset._graph_namespace.parameters["partition_policy"]["edge_weighting"]
    assert (
        dataset._graph_cache_namespace()
        != Dataset(_config(tmp_path))._graph_cache_namespace()
    )
    generated = []
    monkeypatch.setattr(
        "optimization.data.graph_builder.build_base_graph",
        lambda ingest, **kwargs: generated.append(kwargs) or graph,
    )
    assert dataset._generate(LevelSpec.parse("Block_0")) is graph
    assert generated == [{"weight_edges": True}]


@pytest.mark.parametrize(
    "objective,needs_areas",
    [("average_distance", False), ("average_squared_distance", True)],
)
def test_named_heuristic_distance_objectives_load_only_needed_geometry(
    tmp_path, monkeypatch, objective, needs_areas
):
    from optimization.pipeline import step_config

    dataset, graph, _ = _pipeline_dataset(
        tmp_path,
        monkeypatch,
        steps=[
            {
                "level": "Block_0",
                "method": "adaptive_short_bursts",
                "objective": objective,
            }
        ],
    )
    attachments = []
    monkeypatch.setattr(
        "optimization.data.node_areas.attach_land_areas",
        lambda *args: attachments.append(args),
    )
    config = step_config(dataset.config, dataset.config.steps[0])
    for _ in range(2):
        dataset.problem_for("Block_0", centroid_school_ids=[100], config=config)
    assert len(attachments) == int(needs_areas)
    if needs_areas:
        assert attachments[0] == (graph, "Block", dataset.data)


def test_pipeline_problems_share_graph_and_honor_effective_constraints(
    tmp_path, monkeypatch
):
    dataset, graph, attached = _pipeline_dataset(tmp_path, monkeypatch)
    first = dataset.problem_for("Block_0", centroid_school_ids=[100])
    step_config = copy(dataset.config)
    step_config.frl_dev = 0.1
    step_config.racial_dev = 0.2
    step_config.overage = -1.0
    step_config.boundary_prop = 0.4
    step_config.max_distance = 3.0
    step_config.solver = "cp_bool"

    second = dataset.problem_for(
        "Block_0", centroid_school_ids=[100], config=step_config
    )

    assert first.G is second.G is graph
    assert attached == [graph]
    assert first.frl_dev == 0.3
    assert second.frl_dev == 0.1
    assert second.racial_dev == 0.2
    assert second.overage == -1.0
    assert second.shortage == first.shortage
    assert second.boundary_prop == 0.4
    assert second.max_distance == 3.0
    assert second.optimization_config is step_config
    assert second.optimization_config.data_scenario is dataset.data


def test_pipeline_school_distance_attaches_areas_only_when_needed(
    tmp_path, monkeypatch
):
    dataset, graph, _ = _pipeline_dataset(tmp_path, monkeypatch)
    attachments = []
    monkeypatch.setattr(
        "optimization.data.node_areas.attach_land_areas",
        lambda graph, unit, data: attachments.append((graph, unit, data)),
    )
    dataset.problem_for("Block_0", centroid_school_ids=[100])
    assert attachments == []
    step_config = copy(dataset.config)
    step_config.compactness_objective = "school_distance"
    for _ in range(2):
        problem = dataset.problem_for(
            "Block_0", centroid_school_ids=[100], config=step_config
        )
        assert problem.compactness_objective == "school_distance"
    assert attachments == [(graph, "Block", dataset.data)]


@pytest.mark.parametrize(
    "overrides",
    [
        {"data_filters": {"include_k8": True}},
        {"weight_edges": True},
        {"levels": ["BlockGroup_0"]},
    ],
)
def test_pipeline_problem_rejects_changed_dataset_inputs(
    tmp_path, monkeypatch, overrides
):
    dataset, _, attached = _pipeline_dataset(tmp_path, monkeypatch)
    changed = _config(tmp_path, **overrides)
    with pytest.raises(ValueError, match="dataset"):
        dataset.problem_for("Block_0", centroid_school_ids=[100], config=changed)
    assert attached == []


def test_pipeline_centroid_definition_loaded_once(tmp_path, monkeypatch):
    dataset, _, _ = _pipeline_dataset(tmp_path, monkeypatch)
    loaded = []

    def load(centroids_type, data):
        loaded.append((centroids_type, data))
        return [100]

    monkeypatch.setattr(loaders, "load_centroid_schools", load)
    assert dataset.centroids_for("Block_0") == [0]
    assert dataset.problem_for("Block_0").centroids == [0]
    assert dataset.problem_for("Block_0").centroid_school_ids == [100]
    assert loaded == [(dataset.config.centroids_type, dataset.data)]


def test_pipeline_area_hint_cache_accounts_for_file_and_centroids(
    tmp_path, monkeypatch
):
    dataset, graph, _ = _pipeline_dataset(tmp_path, monkeypatch)
    loaded = []
    projected = []

    def load(path):
        loaded.append(path)
        return {1000: 0, 1001: 0}

    def project(assignment, source_level, graph, level, centroids, **kwargs):
        projected.append((level.name, tuple(centroids)))
        return {0: 0, 1: 0}

    monkeypatch.setattr("optimization.data.conversion.load_area_assignment", load)
    monkeypatch.setattr(
        "optimization.data.conversion.hint_from_area_assignment", project
    )
    for centroids in ([0], [1], [0]):
        hint = dataset.area_hint_for("Block_0", graph, centroids, path="first.json")
        hint[0] = 99
    assert loaded == ["first.json"]
    assert projected == [("Block_0", (0,)), ("Block_0", (1,))]
    assert dataset.area_hint_for("Block_0", graph, [0], path="first.json")[0] == 0
    dataset.area_hint_for("Block_0", graph, [0], path="second.json")
    assert loaded == ["first.json", "second.json"]

import kahip
import networkx as nx
import pandas as pd
import pytest

from optimization.data import graph_builder
from optimization.data.edge_weights import (
    BOUNDARY_WEIGHT_ATTR,
    MANUAL_EDGE_ATTR,
    SHARED_BOUNDARY_ATTR,
)
from optimization.data.loaders import IngestConfig
from optimization.tests.synthetic import make_grid_graph


def test_aggregate_level_excludes_school_nodes_and_reattaches_edges(monkeypatch):
    G = make_grid_graph(3, 3)
    captured = {}

    def fake_partition(graph, target_partition_count, population_attr):
        captured["nodes"] = set(graph)
        captured["target"] = target_partition_count
        captured["population_attr"] = population_attr
        return (
            {node: 0 if node in {1, 2, 3} else 1 for node in graph.nodes()},
            [0.4],
        )

    monkeypatch.setattr(
        graph_builder,
        "_partition_non_school_nodes",
        fake_partition,
    )

    coarse = graph_builder.aggregate_level(G, 4, "GE")

    assert captured == {
        "nodes": set(range(1, 8)),
        "target": 2,
        "population_attr": "ge_students",
    }
    assert len(coarse) == 4
    assert coarse.graph["target_node_count"] == 4
    assert coarse.graph["partition_imbalance"] == 0.4
    assert coarse.graph["school_singleton_count"] == 2

    school_nodes = {
        tuple(attrs["school_ids"]): (node, attrs)
        for node, attrs in coarse.nodes(data=True)
        if attrs["school_ids"]
    }
    assert set(school_nodes) == {(100,), (200,)}
    assert school_nodes[(100,)][1]["block_ids"] == [1000]
    assert school_nodes[(200,)][1]["block_ids"] == [1008]

    partition = coarse.graph["partition"]
    expected_edges = {
        tuple(sorted((partition[u], partition[v])))
        for u, v in G.edges()
        if partition[u] != partition[v]
    }
    assert {tuple(sorted(edge)) for edge in coarse.edges()} == expected_edges


def test_aggregate_can_chain_and_preserves_base_ids():
    base = make_grid_graph(2, 3)
    middle = graph_builder.aggregate(
        base,
        {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2},
    )
    coarse = graph_builder.aggregate(middle, {0: 0, 1: 0, 2: 1})

    assert coarse.nodes[0]["block_ids"] == [1000, 1001, 1002, 1003]
    assert coarse.nodes[1]["block_ids"] == [1004, 1005]
    assert sum(len(attrs["block_ids"]) for _, attrs in coarse.nodes(data=True)) == 6


def test_aggregate_sums_crossing_boundary_weights():
    graph = make_grid_graph(1, 3)
    graph.add_edge(0, 2)
    graph.graph.update(
        {
            "weight_edges": True,
            "boundary_crs": "EPSG:32610",
            "boundary_weight_unit": "meter",
            "manual_edge_weight_m": 4.0,
        }
    )
    graph.edges[0, 2].update(
        {
            SHARED_BOUNDARY_ATTR: 4.5,
            BOUNDARY_WEIGHT_ATTR: 5,
            MANUAL_EDGE_ATTR: False,
        }
    )
    graph.edges[1, 2].update(
        {
            SHARED_BOUNDARY_ATTR: 6.5,
            BOUNDARY_WEIGHT_ATTR: 7,
            MANUAL_EDGE_ATTR: True,
        }
    )

    coarse = graph_builder.aggregate(graph, {0: 0, 1: 0, 2: 1})

    assert coarse.edges[0, 1][SHARED_BOUNDARY_ATTR] == 11
    assert coarse.edges[0, 1][BOUNDARY_WEIGHT_ATTR] == 12
    assert coarse.edges[0, 1][MANUAL_EDGE_ATTR] is True
    assert coarse.graph["weight_edges"] is True


def test_kahip_partition_uses_population_weights_and_strong_mode(monkeypatch):
    G = make_grid_graph(2, 2)
    captured = {}

    def fake_kaffpa(*args):
        captured["args"] = args
        return 1, [0, 0, 1, 1]

    monkeypatch.setattr(kahip, "kaffpa", fake_kaffpa)

    groups, imbalance = graph_builder._partition_graph_kahip(
        G,
        2,
        "ge_students",
    )

    args = captured["args"]
    assert args[0] == [1000, 1000, 1000, 1000]
    assert len(args[1]) == len(G) + 1
    assert len(args[2]) == len(args[3]) == 2 * G.number_of_edges()
    assert args[4] == 2
    assert args[5] == 0.8
    assert args[7] == graph_builder.PARTITION_SEED
    assert args[8] == kahip.STRONG
    assert groups == {0: [0, 1], 1: [2, 3]}
    assert imbalance == 0.8


def test_kahip_partition_uses_boundary_weights(monkeypatch):
    graph = nx.path_graph(4)
    for node in graph:
        graph.nodes[node]["ge_students"] = 1.0
    graph.graph["weight_edges"] = True
    for edge, weight in zip(graph.edges(), [3, 5, 7]):
        graph.edges[edge][BOUNDARY_WEIGHT_ATTR] = weight
    captured = {}

    def fake_kaffpa(*args):
        captured["edge_weights"] = args[2]
        return 1, [0, 0, 1, 1]

    monkeypatch.setattr(kahip, "kaffpa", fake_kaffpa)

    graph_builder._partition_graph_kahip(graph, 2, "ge_students")

    assert captured["edge_weights"] == [3, 3, 5, 5, 7, 7]


def test_kahip_partition_relaxes_imbalance_until_valid(monkeypatch):
    G = make_grid_graph(1, 4)
    for node in G:
        G.nodes[node]["ge_students"] = 10.0 if node == 0 else 0.0
    attempted = []

    def fake_kaffpa(*args):
        attempted.append(args[5])
        return 1, [0, 1, 1, 1]

    monkeypatch.setattr(kahip, "kaffpa", fake_kaffpa)

    groups, imbalance = graph_builder._partition_graph_kahip(
        G,
        2,
        "ge_students",
    )

    assert groups == {0: [0], 1: [1, 2, 3]}
    assert attempted == [0.8, 1.6]
    assert imbalance == 1.6


def test_kahip_partition_accepts_fewer_nonempty_parts(monkeypatch):
    G = make_grid_graph(1, 4)
    for node in G:
        G.nodes[node]["ge_students"] = 0.0

    monkeypatch.setattr(kahip, "kaffpa", lambda *args: (1, [0, 0, 2, 2]))

    groups, imbalance = graph_builder._partition_graph_kahip(
        G,
        3,
        "ge_students",
    )

    assert groups == {0: [0, 1], 1: [2, 3]}
    assert imbalance == 0.8


def test_partition_population_attribute_follows_program_population():
    assert graph_builder.population_attribute("GE") == "ge_students"
    assert graph_builder.population_attribute("All") == "all_prog_students"
    assert graph_builder.population_attribute("SB") == "all_prog_students"


def test_base_graph_district_frl_uses_selected_population(
    monkeypatch, scenario_factory
):
    rows = []
    for index, (ge_students, all_students, frl) in enumerate(
        [(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)]
    ):
        row = {
            "BlockGroup": 100 + index,
            "ge_students": ge_students,
            "ge_capacity": 0.0,
            "all_prog_students": all_students,
            "all_prog_capacity": 0.0,
            "num_schools": 0,
            "FRL": frl,
            "school_ids": [],
            "Lat": 0.0,
            "Lon": float(index),
        }
        row.update({ethnicity: 0.0 for ethnicity in graph_builder.AREA_ETHNICITIES})
        rows.append(row)
    area = pd.DataFrame(rows)

    monkeypatch.setattr(graph_builder.loaders, "load_area_table", lambda cfg: area)
    monkeypatch.setattr(
        graph_builder.loaders,
        "load_distance_dict",
        lambda cfg, area2idx: {node: {} for node in area.index},
    )
    monkeypatch.setattr(
        graph_builder.loaders,
        "load_neighbors",
        lambda cfg, area2idx: {},
    )
    monkeypatch.setattr(graph_builder, "_school_data", lambda cfg: {})

    graph = graph_builder.build_base_graph(
        IngestConfig(
            unit="BlockGroup",
            data=scenario_factory(
                filters={"optimization": {"program_population": "All"}}
            ),
        )
    )

    assert graph.graph["F"] == pytest.approx(0.5)


def test_component_partitioning_keeps_disconnected_inputs_separate(monkeypatch):
    G = make_grid_graph(1, 5)
    G.remove_edge(3, 4)
    calls = []

    def fake_partition(graph, target_partition_count, population_attr):
        calls.append((set(graph), target_partition_count, population_attr))
        return {0: list(graph)}, 0.2

    monkeypatch.setattr(graph_builder, "_partition_graph_kahip", fake_partition)

    partition, _ = graph_builder._partition_non_school_nodes(
        G,
        3,
        "ge_students",
    )

    assert calls == [
        ({0, 1, 2, 3}, 2, "ge_students"),
        ({4}, 1, "ge_students"),
    ]
    assert partition[0] != partition[4]


def test_component_partitioning_keeps_distance_exempt_nodes_separate():
    G = make_grid_graph(1, 4)
    for node in G:
        G.nodes[node]["max_distance_exempt"] = node < 2

    partition, _ = graph_builder._partition_non_school_nodes(
        G,
        2,
        "ge_students",
    )

    exempt_parts = {partition[node] for node in (0, 1)}
    regular_parts = {partition[node] for node in (2, 3)}
    assert exempt_parts.isdisjoint(regular_parts)


def test_aggregate_propagates_distance_exemption():
    G = make_grid_graph(1, 3)
    G.nodes[1]["max_distance_exempt"] = True

    coarse = graph_builder.aggregate(G, {0: 0, 1: 0, 2: 1})

    assert coarse.nodes[0]["max_distance_exempt"] is True
    assert coarse.nodes[1]["max_distance_exempt"] is False


def test_component_partition_counts_follow_population():
    G = make_grid_graph(2, 4)
    G.remove_edges_from((u, v) for u, v in list(G.edges()) if u < 4 <= v)
    for node in G:
        G.nodes[node]["ge_students"] = 100.0 if node < 4 else 1.0
    components = [list(range(4)), list(range(4, 8))]

    counts = graph_builder._component_partition_counts(
        G,
        components,
        4,
        "ge_students",
    )

    assert counts == [3, 1]


def test_aggregate_level_rejects_more_school_nodes_than_target():
    G = make_grid_graph(1, 3)
    for node in G:
        G.nodes[node]["school_ids"] = [100 + node]
        G.nodes[node]["num_schools"] = 1

    with pytest.raises(ValueError, match="3 school nodes"):
        graph_builder.aggregate_level(G, 2, "GE")


def test_canonical_partition_is_invariant_to_kahip_part_labels():
    """Relabelling KaHIP's parts must not change the canonical numbering."""
    G = make_grid_graph(3, 3)
    partition = {node: (0 if node < 4 else 1) for node in G}
    relabelled = {node: (7 if part == 0 else 3) for node, part in partition.items()}

    canonical = graph_builder.canonical_partition(G, partition)
    canonical_relabelled = graph_builder.canonical_partition(G, relabelled)

    assert canonical == canonical_relabelled
    assert set(canonical.values()) == {0, 1}


def test_aggregate_level_node_numbering_survives_part_relabelling(monkeypatch):
    """The coarse graph is identical when KaHIP labels the same parts differently."""

    def partition_as(labels):
        def fake_partition(graph, target_partition_count, population_attr):
            return {
                node: labels[0] if node in {1, 2, 3} else labels[1]
                for node in graph.nodes()
            }, [0.4]

        return fake_partition

    coarse_by_labels = []
    for labels in [(0, 1), (5, 2)]:
        monkeypatch.setattr(
            graph_builder, "_partition_non_school_nodes", partition_as(labels)
        )
        coarse = graph_builder.aggregate_level(make_grid_graph(3, 3), 4, "GE")
        coarse_by_labels.append(
            {node: sorted(coarse.nodes[node]["block_ids"]) for node in coarse}
        )

    assert coarse_by_labels[0] == coarse_by_labels[1]


def test_portable_partition_round_trips_through_area_ids():
    G = make_grid_graph(3, 3)
    partition = graph_builder.canonical_partition(
        G, {node: (0 if node < 4 else 1) for node in G}
    )

    artifact = graph_builder.partition_to_portable(G, partition)
    assert artifact["artifact_version"] == graph_builder.PARTITION_ARTIFACT_VERSION

    assert graph_builder.partition_from_portable(G, artifact) == partition


def test_replayed_partition_reproduces_the_coarse_graph(monkeypatch):
    """Replay must yield the same graph without consulting KaHIP."""

    def fake_partition(graph, target_partition_count, population_attr):
        return {node: (0 if node in {1, 2, 3} else 1) for node in graph.nodes()}, [0.4]

    monkeypatch.setattr(graph_builder, "_partition_non_school_nodes", fake_partition)
    built = graph_builder.aggregate_level(make_grid_graph(3, 3), 4, "GE")
    artifact = graph_builder.partition_to_portable(
        make_grid_graph(3, 3), built.graph["partition"]
    )

    def exploding_partition(*args, **kwargs):
        raise AssertionError("replay must not re-partition")

    monkeypatch.setattr(
        graph_builder, "_partition_non_school_nodes", exploding_partition
    )
    parent = make_grid_graph(3, 3)
    replayed = graph_builder.aggregate_level(
        parent,
        4,
        "GE",
        partition=graph_builder.partition_from_portable(parent, artifact),
    )

    def edge_set(graph):
        return {tuple(sorted(edge)) for edge in graph.edges()}

    assert built.graph["partition"] == replayed.graph["partition"]
    assert edge_set(built) == edge_set(replayed)
    assert replayed.graph["partition_backend"] == "replayed"
    assert {n: sorted(built.nodes[n]["block_ids"]) for n in built} == {
        n: sorted(replayed.nodes[n]["block_ids"]) for n in replayed
    }


def test_portable_partition_rejects_a_foreign_artifact():
    G = make_grid_graph(3, 3)
    artifact = graph_builder.partition_to_portable(
        G, {node: (0 if node < 4 else 1) for node in G}
    )
    artifact["parts"][0] = artifact["parts"][0][:-1]

    with pytest.raises(ValueError, match="missing area ids"):
        graph_builder.partition_from_portable(G, artifact)

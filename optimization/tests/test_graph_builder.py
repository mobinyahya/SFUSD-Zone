import kahip
import pytest

from optimization.data import graph_builder
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


def test_partition_population_attribute_follows_population_type():
    assert graph_builder.population_attribute("GE") == "ge_students"
    assert graph_builder.population_attribute("All") == "all_prog_students"
    assert graph_builder.population_attribute("SB") == "all_prog_students"


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

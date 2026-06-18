from Zone_Generation.optimization.data import graph_builder
from Zone_Generation.optimization.tests.synthetic import make_grid_graph


def test_aggregate_level_uses_local_partition_helper(monkeypatch):
    G = make_grid_graph(3, 3)

    def fake_partition(graph, target_partition_count):
        nodes = list(graph.nodes())
        midpoint = len(nodes) // 2
        return {0: nodes[:midpoint], 1: nodes[midpoint:]}

    monkeypatch.setattr(
        graph_builder,
        "_partition_graph_metis_partial_constraint",
        fake_partition,
    )

    coarse = graph_builder.aggregate_level(G, split_depth=1, split_base=9)

    assert set(coarse.nodes()) == {0, 1}
    assert coarse.graph["partition"] == {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
    }
    assert coarse.nodes[0]["block_ids"] == [1000, 1001, 1002, 1003]
    assert coarse.nodes[1]["block_ids"] == [1004, 1005, 1006, 1007, 1008]
    assert coarse.nodes[0]["school_ids"] == [100]
    assert coarse.nodes[1]["school_ids"] == [200]
    assert coarse.has_edge(0, 1)
    assert coarse.graph["F"] == G.graph["F"]
    assert coarse.graph["R"] == G.graph["R"]
    assert set(coarse.graph["distance_dict"]) == {0, 1}


def test_metis_partition_helper_groups_all_nodes():
    G = make_grid_graph(3, 3)

    super_nodes = graph_builder._partition_graph_metis_partial_constraint(G, 2)

    assigned = {node for nodes in super_nodes.values() for node in nodes}
    assert assigned == set(G.nodes())
    assert len(super_nodes) == 2


def test_partition_helper_skips_metis_when_target_exceeds_nodes():
    G = make_grid_graph(2, 2)

    super_nodes = graph_builder._partition_graph_metis_partial_constraint(G, 9)

    assert super_nodes == {0: [0], 1: [1], 2: [2], 3: [3]}

import networkx as nx
import pytest

from optimization.data import contiguity
from optimization.tests.synthetic import make_grid_graph


def test_contiguous_assignment_passes():
    G = make_grid_graph(3, 3)
    centroids = [0, 8]
    # Left column + middle to zone 0, right column to zone 1 -> both connected.
    assignment = {i: (1 if i % 3 == 2 else 0) for i in G.nodes()}
    assert contiguity.is_contiguous(G, assignment, centroids)


def test_noncontiguous_assignment_flagged():
    G = make_grid_graph(3, 3)
    centroids = [0, 8]
    assignment = {i: 0 for i in G.nodes()}
    # Plant an isolated zone-1 island at node 4 (center), surrounded by zone 0.
    assignment[4] = 1
    assignment[8] = 1  # centroid of zone 1, not adjacent to node 4
    assert not contiguity.is_contiguous(G, assignment, centroids)


def test_repair_makes_contiguous():
    G = make_grid_graph(3, 3)
    centroids = [0, 8]
    assignment = {i: 0 for i in G.nodes()}
    assignment[4] = 1
    assignment[8] = 1
    repaired = contiguity.repair(G, assignment, centroids)
    assert contiguity.is_contiguous(G, repaired, centroids)
    # centroids keep their zones
    assert repaired[0] == 0
    assert repaired[8] == 1


def test_relax_unsupported_candidates_opens_closer_path():
    G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (3, 4)])
    G.graph["closer_neighbors"] = {
        0: {100: frozenset(), 200: frozenset({3})},
        1: {100: frozenset({2}), 200: frozenset({2})},
        2: {100: frozenset({3}), 200: frozenset({4})},
        3: {100: frozenset({0}), 200: frozenset({4})},
        4: {100: frozenset({3}), 200: frozenset()},
    }
    assignment = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
    candidates = {node: {zone} for node, zone in assignment.items()}

    relaxed = contiguity.relax_unsupported_candidates(
        G, assignment, [0, 4], [100, 200], candidates
    )

    assert 1 not in relaxed
    assert relaxed[2] == {0, 1}
    assert relaxed[3] == {0, 1}
    assert relaxed[0] == {0}
    assert relaxed[4] == {1}


def test_shortest_release_opens_one_path_instead_of_every_path():
    # Block 1 (zone 0) has two closer neighbours toward school 100, both in
    # zone 1: 2 leads straight to the centroid 0, 3 leads there through 5.
    # "all" opens zone 0 on both; "shortest" only on the one-block path via 2.
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.graph["closer_neighbors"] = {
        0: {100: frozenset(), 200: frozenset({2})},
        1: {100: frozenset({2, 3}), 200: frozenset({2})},
        2: {100: frozenset({0}), 200: frozenset({4})},
        3: {100: frozenset({5}), 200: frozenset({4})},
        4: {100: frozenset({3}), 200: frozenset()},
        5: {100: frozenset({0}), 200: frozenset({3})},
    }
    assignment = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 0}
    candidates = {node: {zone} for node, zone in assignment.items()}
    args = (G, assignment, [0, 4], [100, 200], candidates)

    every = contiguity.relax_unsupported_candidates(*args)
    shortest = contiguity.relax_unsupported_candidates(*args, paths="shortest")

    assert every[2] == {0, 1} and every[3] == {0, 1}
    assert shortest[2] == {0, 1} and shortest[3] == {1}
    assert 1 not in every and 1 not in shortest
    with pytest.raises(ValueError, match="paths must be one of"):
        contiguity.relax_unsupported_candidates(*args, paths="cone")


def test_shortest_release_stops_at_the_first_supported_block_of_its_zone():
    # From 1 the walk reaches 5, already zone 0 and supported by 0, after one
    # zone-1 block (3); it stops there rather than continuing to the centroid.
    G = nx.Graph()
    G.add_nodes_from(range(6))
    G.graph["closer_neighbors"] = {
        0: {100: frozenset(), 200: frozenset({2})},
        1: {100: frozenset({3}), 200: frozenset({2})},
        2: {100: frozenset({0}), 200: frozenset({4})},
        3: {100: frozenset({5}), 200: frozenset({4})},
        4: {100: frozenset({3}), 200: frozenset()},
        5: {100: frozenset({2}), 200: frozenset({3})},
    }
    # 5 is zone 0 but its only closer neighbour 2 is zone 1, so 5 is itself
    # unsupported and the walk must go on through 2 to the centroid.
    assignment = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 0}
    candidates = {node: {zone} for node, zone in assignment.items()}
    relaxed = contiguity.relax_unsupported_candidates(
        G, assignment, [0, 4], [100, 200], candidates, paths="shortest"
    )
    assert relaxed[3] == {0, 1} and relaxed[2] == {0, 1}

    # Give 5 its own support (closer neighbour 0): the walk from 1 now ends at 5.
    G.graph["closer_neighbors"][5] = {100: frozenset({0}), 200: frozenset({3})}
    relaxed = contiguity.relax_unsupported_candidates(
        G, assignment, [0, 4], [100, 200], candidates, paths="shortest"
    )
    assert relaxed[3] == {0, 1} and relaxed[2] == {1}


def test_contiguity_supports_do_not_fall_back_for_geometry_local_minimum():
    G = nx.path_graph(3)
    G.graph["closer_neighbors"] = {
        0: {100: frozenset()},
        1: {100: frozenset({0})},
        2: {100: frozenset()},
    }

    supports = contiguity.contiguity_supports(G, [0], [100], lambda node: {0})

    assert supports[(2, 0)] == []


def test_closer_supports_require_precomputed_geometry_relation():
    G = nx.path_graph(2)
    G.graph["distance_dict"] = {0: {0: 0.0, 1: 1.0}}

    with pytest.raises(ValueError, match="precomputed geometry-based"):
        contiguity.closer_supports(G, [0], [100], lambda node: {0})

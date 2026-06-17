from Zone_Generation.pipeline.data import contiguity
from Zone_Generation.pipeline.tests.synthetic import make_grid_graph


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


def test_boundary_candidates_pins_interior():
    G = make_grid_graph(3, 3)
    centroids = [0, 8]
    assignment = {i: (1 if i % 3 == 2 else 0) for i in G.nodes()}
    cands = contiguity.boundary_candidates(G, assignment, centroids, radius=1)
    # centroids pinned
    assert cands[0] == {0}
    assert cands[8] == {1}
    # every candidate set contains the node's current zone
    for node, zones in cands.items():
        assert assignment[node] in zones

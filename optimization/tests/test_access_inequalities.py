from __future__ import annotations

from optimization.access_inequalities import (
    cardinality_cliques,
    completion_pairs,
    pair_key,
    transitivity_triples,
)


def test_centroids_form_one_cardinality_clique_per_anchor():
    """Each centroid is pinned to its own zone, so no two can share one."""
    # Node 10 reaches centroids 0, 1, 2, each fixed to a distinct zone.
    zones = {0: {0}, 1: {1}, 2: {2}, 10: {0, 1, 2}}
    pairs = [pair_key(10, 0), pair_key(10, 1), pair_key(10, 2)]

    constraints = cardinality_cliques(pairs, lambda node: zones[node])

    assert (10, (0, 1, 2)) in constraints


def test_zone_compatible_neighbours_are_not_capped_together():
    """Two nodes that can share a zone may both be co-zoned with the anchor."""
    zones = {0: {0, 1}, 1: {0, 1}, 10: {0, 1}}
    pairs = [pair_key(10, 0), pair_key(10, 1)]

    constraints = cardinality_cliques(pairs, lambda node: zones[node])

    # 0 and 1 overlap on both zones, so {0, 1} is not a valid cap.
    assert all(set(members) != {0, 1} for _, members in constraints)


def test_one_anchor_can_yield_several_disjoint_cliques():
    """The pool is covered, so a second valid cap is not left on the table."""
    # {0,1} and {2,3} are each mutually incompatible, but every cross pair
    # overlaps, so no clique spans both and two constraints are needed.
    zones = {
        0: {1, 2},
        1: {3, 4},
        2: {1, 3},
        3: {2, 4},
        10: {1, 2, 3, 4},
    }
    pairs = [pair_key(10, node) for node in (0, 1, 2, 3)]

    constraints = cardinality_cliques(pairs, lambda node: zones[node])

    assert constraints == [(10, (0, 1)), (10, (2, 3))]


def test_leftover_singletons_are_dropped():
    """A one-member cap says ``a <= 1``, which the variable already satisfies."""
    zones = {0: {0}, 1: {1}, 2: {0, 2}, 10: {0, 1, 2}}
    pairs = [pair_key(10, node) for node in (0, 1, 2)]

    constraints = cardinality_cliques(pairs, lambda node: zones[node])

    assert constraints == [(10, (0, 1))]
    assert all(len(members) >= 2 for _, members in constraints)


def test_cliques_are_deterministic():
    zones = {n: {n} for n in range(6)} | {10: set(range(6))}
    pairs = [pair_key(10, node) for node in range(6)]

    first = cardinality_cliques(pairs, lambda node: zones[node])
    second = cardinality_cliques(pairs, lambda node: zones[node])

    assert first == second


def test_transitivity_needs_all_three_sides_present():
    # Triangle 0-1-2 is complete; node 3 only touches 0.
    pairs = [(0, 1), (1, 2), (0, 2), (0, 3)]

    triples = list(transitivity_triples(pairs))

    assert triples == [(0, 1, 2)]


def test_transitivity_respects_its_limit():
    pairs = [(a, b) for a in range(6) for b in range(a + 1, 6)]

    assert len(list(transitivity_triples(pairs))) == 20  # C(6, 3)
    assert len(list(transitivity_triples(pairs, limit=7))) == 7


def test_completion_creates_the_missing_side_of_a_triangle():
    """A bipartite pair set has no triangles until the third side is added."""
    zones = {0: {0, 1}, 1: {0, 1}, 10: {0, 1}}
    pairs = [pair_key(10, 0), pair_key(10, 1)]

    assert list(transitivity_triples(pairs)) == []

    added = completion_pairs(pairs, lambda node: zones[node])

    assert added == [(0, 1)]
    assert list(transitivity_triples(pairs + added)) == [(0, 1, 10)]


def test_completion_skips_pairs_that_cannot_share_a_zone():
    """Those are constants the solver fixes to zero; the cap family covers them."""
    zones = {0: {0}, 1: {1}, 10: {0, 1}}
    pairs = [pair_key(10, 0), pair_key(10, 1)]

    assert completion_pairs(pairs, lambda node: zones[node]) == []


def test_completion_respects_its_limit():
    zones = {n: {0, 1} for n in range(8)} | {10: {0, 1}}
    pairs = [pair_key(10, n) for n in range(8)]

    assert len(completion_pairs(pairs, lambda n: zones[n])) == 28  # C(8, 2)
    assert len(completion_pairs(pairs, lambda n: zones[n], limit=5)) == 5

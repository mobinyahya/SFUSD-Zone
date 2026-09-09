"""Valid inequalities for the co-zoning indicators ``a[u, v]``.

``a[u, v]`` is 1 when nodes ``u`` and ``v`` share a zone. The solvers linearize
it through per-zone conjunctions, and that linearization is weak: spreading the
assignment variables evenly over ``Z`` zones lets every ``both[u, v, z]`` reach
``1/Z`` and therefore every ``a[u, v]`` reach 1 at once. Under a choice
objective, whose cuts read ``total <= constant + sum(g * a)`` with positive
``g``, that makes every cut slack, so the master's dual bound never improves on
the a-priori constant it was declared with.

Two families cut that off. Both are stated purely in terms of which node pairs
*could* share a zone, so they hold for any objective.

Transitivity
    Sharing a zone is an equivalence relation, so for any triple

        a[u,v] + a[v,w] - a[u,w] <= 1

    together with its two rotations. These are the triangle facets of the clique
    partitioning polytope.

Cardinality
    If a set ``C`` of nodes is pairwise unable to share a zone, then a single
    zone holds at most one of them, so for any anchor ``u``

        sum(a[u, c] for c in C) <= 1.

    This is the clique-strengthened form of the transitivity inequalities whose
    third term is structurally zero, and it is much stronger than the pairwise
    version: over ``|C|`` nodes, pairwise triangles only force the sum below
    ``|C|/2``. Centroids are always such a set, since each is pinned to its own
    zone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import combinations

AccessPair = tuple[int, int]


def pair_key(first: int, second: int) -> AccessPair:
    """Normalize a node pair the way the solvers key their access variables."""
    return (min(first, second), max(first, second))


def _neighbours(pairs: Iterable[AccessPair]) -> dict[int, list[int]]:
    adjacency: dict[int, set[int]] = {}
    for first, second in pairs:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    return {node: sorted(others) for node, others in sorted(adjacency.items())}


def cardinality_cliques(
    pairs: Iterable[AccessPair],
    candidate_zones: Callable[[int], set[int]],
    *,
    min_size: int = 2,
) -> list[tuple[int, tuple[int, ...]]]:
    """Anchors paired with sets of mutually zone-incompatible neighbours.

    Returns ``(anchor, members)`` such that ``sum(a[anchor, m] for m in
    members) <= 1`` is valid. Cliques are grown greedily and the pool is
    covered, so one anchor can contribute several disjoint constraints. Node
    order is fixed, so the output is deterministic.
    """
    zones: dict[int, set[int]] = {}

    def zones_for(node: int) -> set[int]:
        if node not in zones:
            zones[node] = set(candidate_zones(node))
        return zones[node]

    def incompatible(first: int, second: int) -> bool:
        return not (zones_for(first) & zones_for(second))

    constraints: list[tuple[int, tuple[int, ...]]] = []
    for anchor, others in _neighbours(pairs).items():
        pool = list(others)
        while len(pool) >= min_size:
            clique = [pool[0]]
            for candidate in pool[1:]:
                if all(incompatible(candidate, member) for member in clique):
                    clique.append(candidate)
            if len(clique) >= min_size:
                constraints.append((anchor, tuple(clique)))
            # Removing the clique -- even a singleton -- guarantees progress.
            chosen = set(clique)
            pool = [node for node in pool if node not in chosen]
    return constraints


def completion_pairs(
    pairs: Iterable[AccessPair],
    candidate_zones: Callable[[int], set[int]],
    *,
    limit: int | None = None,
) -> list[AccessPair]:
    """Pairs to introduce so that transitivity has triples to bind on.

    The cuts reference ``(student, school)`` pairs, which makes the pair graph
    close to bipartite -- and a bipartite graph has no triangles at all, so the
    transitivity facets have almost nothing to attach to. Adding the missing
    ``(school, school)`` side for every pair of neighbours of a common anchor
    creates those triangles. Only zone-compatible pairs are worth adding; an
    incompatible one is a constant the solver already fixes to zero, and is
    covered by :func:`cardinality_cliques` instead.
    """
    zones: dict[int, set[int]] = {}

    def zones_for(node: int) -> set[int]:
        if node not in zones:
            zones[node] = set(candidate_zones(node))
        return zones[node]

    known = {pair_key(*pair) for pair in pairs}
    missing: set[AccessPair] = set()
    for _, others in _neighbours(known).items():
        for first, second in combinations(others, 2):
            key = pair_key(first, second)
            if key in known or key in missing:
                continue
            if not (zones_for(first) & zones_for(second)):
                continue
            missing.add(key)
            if limit is not None and len(missing) >= limit:
                return sorted(missing)
    return sorted(missing)


def transitivity_triples(
    pairs: Iterable[AccessPair], *, limit: int | None = None
) -> Iterator[tuple[int, int, int]]:
    """Triples whose three access variables all exist, for the triangle facets.

    Only triples that are fully represented by variables are produced; a triple
    with a structurally zero side is already covered, more strongly, by
    :func:`cardinality_cliques`.
    """
    known = {pair_key(*pair) for pair in pairs}
    produced = 0
    for anchor, others in _neighbours(known).items():
        for first, second in combinations(others, 2):
            if pair_key(first, second) not in known:
                continue
            # Emit each triple once, from its smallest node.
            if anchor > min(first, second):
                continue
            if limit is not None and produced >= limit:
                return
            produced += 1
            yield (anchor, first, second)

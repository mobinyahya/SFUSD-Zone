"""Contiguity primitives.

A zone is *contiguous* if the subgraph it induces is connected. We enforce this
with a centroid-rooted support formulation: a non-centroid node may belong to
zone ``z`` only if at least one supported neighbor also belongs to ``z``. The
preferred support moves strictly closer to the centroid; a fallback handles
fine-graph local minima where a physically adjacent block is the only way out.

This module provides:

* :func:`contiguity_supports` -- the per-(node, zone) support sets the solvers
  turn into linear constraints,
* :func:`is_contiguous` / :func:`boundary_edges` -- validators / objective
  helpers,
* :func:`repair` -- a post-hoc fixer used by local search,
* :func:`boundary_candidates` -- the candidate-narrowing used by the recursive
  strategy to relax only near zone borders.
"""

from __future__ import annotations

import networkx as nx


def _distance(G: nx.Graph, centroid: int, node: int) -> float:
    return float(G.graph["distance_dict"][centroid][node])


def closer_supports(
    G: nx.Graph,
    centroids: list[int],
    candidate_zones,
) -> dict[tuple[int, int], list[int]]:
    """Support sets for the contiguity constraint.

    Parameters
    ----------
    candidate_zones:
        Callable ``node -> set[zone]`` giving the zones each node may take.

    Returns
    -------
    ``{(node, zone): [neighbor, ...]}`` where each neighbor is strictly closer
    to the zone's centroid and is itself a candidate for that zone. A
    non-centroid node may join ``zone`` only if at least one listed neighbor
    does too. An empty list means the assignment is infeasible (the solver must
    forbid it).
    """
    supports: dict[tuple[int, int], list[int]] = {}
    for node in G.nodes():
        for z in candidate_zones(node):
            centroid = centroids[z]
            if node == centroid:
                continue
            d_node = _distance(G, centroid, node)
            supports[(node, z)] = [
                nb
                for nb in G.neighbors(node)
                if _distance(G, centroid, nb) < d_node
                and z in candidate_zones(nb)
            ]
    return supports


def contiguity_supports(
    G: nx.Graph,
    centroids: list[int],
    candidate_zones,
) -> dict[tuple[int, int], list[int]]:
    """Support sets with a fallback for fine-graph distance local minima.

    The preferred support for ``(node, zone)`` is a same-zone candidate neighbor
    that is strictly closer to the zone centroid and can itself continue toward
    the centroid. Some fine Block-level geometries have small local minima where
    every adjacent block is slightly farther from the centroid even though the
    area is physically connected. For those cases, fall back to any same-zone
    candidate neighbor that can continue toward the centroid. This mirrors the
    legacy CP contiguity behavior and avoids forbidding otherwise valid leaf
    blocks.
    """
    closer = closer_supports(G, centroids, candidate_zones)
    adjacent: dict[tuple[int, int], list[int]] = {}
    for node in G.nodes():
        for z in candidate_zones(node):
            centroid = centroids[z]
            if node == centroid:
                continue
            adjacent[(node, z)] = [
                nb for nb in G.neighbors(node) if z in candidate_zones(nb)
            ]

    supports: dict[tuple[int, int], list[int]] = {}
    for key, closer_nodes in closer.items():
        node, z = key
        centroid = centroids[z]
        good = [
            nb
            for nb in closer_nodes
            if nb == centroid or closer.get((nb, z), [])
        ]
        if good:
            supports[key] = good
            continue
        supports[key] = [
            nb
            for nb in adjacent[key]
            if nb == centroid or closer.get((nb, z), [])
        ]
    return supports


def is_contiguous(
    G: nx.Graph, assignment: dict[int, int], centroids: list[int]
) -> bool:
    """True iff every zone induces a single connected component."""
    zones: dict[int, list[int]] = {}
    for node, z in assignment.items():
        zones.setdefault(z, []).append(node)
    for z, nodes in zones.items():
        if not nodes:
            continue
        sub = G.subgraph(nodes)
        if not nx.is_connected(sub):
            return False
    return True


def boundary_edges(G: nx.Graph, assignment: dict[int, int]) -> int:
    """Number of edges whose endpoints fall in different zones."""
    return sum(
        1
        for u, v in G.edges()
        if assignment.get(u) != assignment.get(v)
    )


def repair(
    G: nx.Graph, assignment: dict[int, int], centroids: list[int]
) -> dict[int, int]:
    """Make ``assignment`` contiguous by reabsorbing orphaned fragments.

    For each zone we keep only the connected component containing its centroid.
    Every other node is repeatedly reassigned to the most common zone among its
    already-settled neighbors until all nodes are placed. Used by the
    local-search solver and as a post-processing pass.
    """
    result = dict(assignment)
    centroid_zone = {centroids[z]: z for z in range(len(centroids))}

    # Drop nodes not in their centroid's component.
    orphans: set[int] = set()
    zones: dict[int, list[int]] = {}
    for node, z in result.items():
        zones.setdefault(z, []).append(node)
    for z, nodes in zones.items():
        centroid = centroids[z]
        if centroid not in nodes:
            orphans.update(nodes)
            continue
        comp = nx.node_connected_component(G.subgraph(nodes), centroid)
        orphans.update(n for n in nodes if n not in comp)

    for node in orphans:
        result.pop(node, None)
    # Centroids are always anchored.
    for centroid, z in centroid_zone.items():
        result[centroid] = z

    # Greedily grow zones into the orphan set along edges.
    changed = True
    while orphans and changed:
        changed = False
        for node in list(orphans):
            counts: dict[int, int] = {}
            for nb in G.neighbors(node):
                if nb in result:
                    counts[result[nb]] = counts.get(result[nb], 0) + 1
            if counts:
                result[node] = max(counts, key=counts.get)
                orphans.discard(node)
                changed = True
    # Anything still unreachable keeps its original zone as a last resort.
    for node in orphans:
        result[node] = assignment[node]
    return result


def boundary_candidates(
    G: nx.Graph,
    assignment: dict[int, int],
    centroids: list[int],
    radius: int = 1,
) -> dict[int, set[int]]:
    """Candidate sets that relax only near zone borders.

    Nodes within ``radius`` hops of a zone boundary may switch to any zone
    present in that neighborhood; interior nodes are pinned to their current
    zone. This is the clean re-derivation of the legacy ``drop_boundary`` /
    ``trim_noncontiguity`` heuristics, used to seed a finer level from a coarser
    solution in the recursive strategy.
    """
    anchored = dict(assignment)
    for z, centroid in enumerate(centroids):
        anchored[centroid] = z

    candidates: dict[int, set[int]] = {
        node: {zone} for node, zone in anchored.items() if node in G
    }
    boundary_sources: list[tuple[int, set[int]]] = []
    for u, v in G.edges():
        zu = anchored.get(u)
        zv = anchored.get(v)
        if zu is None or zv is None or zu == zv:
            continue
        zones = {zu, zv}
        boundary_sources.append((u, zones))
        boundary_sources.append((v, zones))

    # Let adjacent-zone labels propagate into the boundary band. The previous
    # implementation expanded the band but still only inspected immediate
    # neighbors, so radius > 1 rarely changed the candidate sets.
    cutoff = max(0, int(radius))
    for source, zones in boundary_sources:
        for node in nx.single_source_shortest_path_length(
            G, source, cutoff=cutoff
        ):
            if node in anchored:
                candidates.setdefault(node, {anchored[node]}).update(zones)

    # Pin centroids regardless of coverage.
    for z, centroid in enumerate(centroids):
        candidates[centroid] = {z}
    return candidates

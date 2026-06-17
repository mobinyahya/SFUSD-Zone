"""Strict contiguity primitives.

A zone is *contiguous* if the subgraph it induces is connected. We enforce this
with the classic shortest-path-tree formulation: a non-centroid node may belong
to zone ``z`` only if at least one of its neighbors that is *strictly closer to
``z``'s centroid* also belongs to ``z``. Because every such "support" edge moves
strictly closer to the centroid (whose distance is 0, the unique minimum), the
support relation cannot cycle, so every node in a zone has a strictly-decreasing
path to the centroid -- i.e. the zone is connected.

This module provides:

* :func:`closer_supports` -- the per-(node, zone) support sets the solvers turn
  into linear constraints,
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
    boundary: set[int] = set()
    for u, v in G.edges():
        if assignment.get(u) != assignment.get(v):
            boundary.add(u)
            boundary.add(v)

    # Expand the boundary band by `radius` hops.
    band = set(boundary)
    frontier = set(boundary)
    for _ in range(max(0, radius - 1)):
        nxt = set()
        for node in frontier:
            nxt.update(G.neighbors(node))
        band |= nxt
        frontier = nxt

    candidates: dict[int, set[int]] = {}
    for node in G.nodes():
        if node not in assignment:
            # Not covered by the projection: leave it out so the problem falls
            # back to its distance-based candidacy.
            continue
        if node in band:
            candidates[node] = {assignment[node]} | {
                assignment[nb]
                for nb in G.neighbors(node)
                if nb in assignment
            }
        else:
            candidates[node] = {assignment[node]}
    # Pin centroids regardless of coverage.
    for z, centroid in enumerate(centroids):
        candidates[centroid] = {z}
    return candidates

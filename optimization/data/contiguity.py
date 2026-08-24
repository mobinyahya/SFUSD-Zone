"""Contiguity primitives.

A zone is *contiguous* if the subgraph it induces is connected. The math
programming solvers enforce this with a centroid-rooted support formulation and
reject a non-centroid assignment when the block has no candidate neighbor that
is strictly closer to the zone's school point under the precomputed polygon
geometry relation.

This module provides:

* :func:`closer_supports` -- strictly closer candidate neighbors,
* :func:`contiguity_supports` -- the support sets the solvers turn into linear
  constraints,
* :func:`is_contiguous` / :func:`boundary_edges` -- validators / objective
  helpers,
* :func:`repair` -- a post-hoc assignment fixer,
* :func:`boundary_candidates` -- the candidate-narrowing used by the recursive
  strategy to relax only near zone borders,
* :func:`relax_unsupported_candidates` -- targeted expansion for projected
  blocks without monotone support toward their centroid.
"""

from __future__ import annotations

import networkx as nx

from optimization.data.closer_neighbors import CLOSER_NEIGHBORS_GRAPH_KEY
from optimization.data.edge_weights import boundary_cost as weighted_boundary_cost


def closer_supports(
    G: nx.Graph,
    centroids: list[int],
    centroid_school_ids: list[int],
    candidate_zones,
) -> dict[tuple[int, int], list[int]]:
    """Support sets for the contiguity constraint.

    Parameters
    ----------
    candidate_zones:
        Callable ``node -> set[zone]`` giving the zones each node may take.

    Returns
    -------
    ``{(node, zone): [neighbor, ...]}`` where each neighbor's geometry is
    strictly closer to the zone's school point and is itself a candidate for
    that zone. A
    non-centroid node may join ``zone`` only if at least one listed neighbor
    does too. An empty list means the assignment is infeasible (the solver must
    forbid it).
    """
    if len(centroid_school_ids) != len(centroids):
        raise ValueError("Closer-neighbor supports require one school ID per centroid.")
    relation = G.graph.get(CLOSER_NEIGHBORS_GRAPH_KEY)
    if relation is None:
        raise ValueError(
            "Graph has no precomputed geometry-based closer-neighbor relation."
        )

    supports: dict[tuple[int, int], list[int]] = {}
    for node in G.nodes():
        for z in candidate_zones(node):
            centroid = centroids[z]
            if node == centroid:
                continue
            school_id = int(centroid_school_ids[z])
            try:
                precomputed = relation[node][school_id]
            except KeyError as exc:
                raise ValueError(
                    "Precomputed closer-neighbor relation is missing "
                    f"node {node}, school {school_id}."
                ) from exc
            supports[(node, z)] = [
                neighbor
                for neighbor in sorted(precomputed)
                if z in candidate_zones(neighbor)
            ]
    return supports


def contiguity_supports(
    G: nx.Graph,
    centroids: list[int],
    centroid_school_ids: list[int],
    candidate_zones,
) -> dict[tuple[int, int], list[int]]:
    """Closer supports that can continue monotonically toward the school."""
    closer = closer_supports(
        G, centroids, centroid_school_ids, candidate_zones
    )

    supports: dict[tuple[int, int], list[int]] = {}
    for key, closer_nodes in closer.items():
        node, z = key
        centroid = centroids[z]
        good = [nb for nb in closer_nodes if nb == centroid or closer.get((nb, z), [])]
        if good:
            supports[key] = good
            continue
        supports[key] = []
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
    return sum(1 for u, v in G.edges() if assignment.get(u) != assignment.get(v))


def boundary_cost(
    G: nx.Graph,
    assignment: dict[int, int],
    *,
    weight_edges: bool,
) -> int:
    """Cut-edge count or integer-metre weighted boundary cost."""
    return weighted_boundary_cost(G, assignment, weighted=weight_edges)


def repair(
    G: nx.Graph, assignment: dict[int, int], centroids: list[int]
) -> dict[int, int]:
    """Make ``assignment`` contiguous by reabsorbing orphaned fragments.

    For each zone we keep only the connected component containing its centroid.
    Every other node is repeatedly reassigned to the most common zone among its
    already-settled neighbors until all nodes are placed. Used for warm starts
    and post-processing.
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
        for node in nx.single_source_shortest_path_length(G, source, cutoff=cutoff):
            if node in anchored:
                candidates.setdefault(node, {anchored[node]}).update(zones)

    # Pin centroids regardless of coverage.
    for z, centroid in enumerate(centroids):
        candidates[centroid] = {z}
    return candidates


def relax_unsupported_candidates(
    G: nx.Graph,
    assignment: dict[int, int],
    centroids: list[int],
    centroid_school_ids: list[int],
    candidates: dict[int, set[int]],
) -> dict[int, set[int]]:
    """Relax projected blocks without a same-zone closer neighbor.

    The unsupported block itself falls back to distance-based candidacy. Its
    projected zone is also opened along closer-neighbor paths so a reassignment
    can retain monotone support toward the centroid.
    """
    if len(centroid_school_ids) != len(centroids):
        raise ValueError("Candidate relaxation requires one school ID per centroid.")
    relation = G.graph.get(CLOSER_NEIGHBORS_GRAPH_KEY)
    if relation is None:
        raise ValueError(
            "Graph has no precomputed geometry-based closer-neighbor relation."
        )

    result = {node: set(zones) for node, zones in candidates.items()}
    unsupported: dict[int, set[int]] = {}
    for node, zone in assignment.items():
        if node == centroids[zone]:
            continue
        school_id = int(centroid_school_ids[zone])
        closer = relation[node][school_id]
        if not any(assignment.get(neighbor) == zone for neighbor in closer):
            unsupported.setdefault(zone, set()).add(node)
            result.pop(node, None)

    centroid_set = set(centroids)
    for zone, sources in unsupported.items():
        school_id = int(centroid_school_ids[zone])
        stack = [
            neighbor
            for source in sources
            for neighbor in relation[source][school_id]
        ]
        seen = set(sources)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in centroid_set and node != centroids[zone]:
                continue
            if node in result:
                result[node].add(zone)
            stack.extend(relation[node][school_id])

    for zone, centroid in enumerate(centroids):
        result[centroid] = {zone}
    return result

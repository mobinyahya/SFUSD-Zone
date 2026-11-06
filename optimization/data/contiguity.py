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
* :func:`relax_unsupported_candidates` -- targeted expansion for projected
  blocks without monotone support toward their centroid.
"""

from __future__ import annotations

from collections import deque

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
    closer = closer_supports(G, centroids, centroid_school_ids, candidate_zones)

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


#: How many closer-neighbor paths :func:`relax_unsupported_candidates` opens.
RELEASE_PATHS = ("all", "shortest")


def _shortest_support_path(
    relation, assignment, source, zone, school_id, centroid, centroid_set, unsupported
) -> list[int]:
    """Blocks strictly between ``source`` and the nearest support for ``zone``.

    Breadth-first along closer-neighbor links (fewest blocks), stopping at the
    zone's centroid or at a block the assignment already puts in ``zone`` that
    has support of its own. Other zones' centroids are never crossed. Empty if
    no such path exists.
    """
    parent = {source: None}
    queue = deque([source])
    end = None
    while queue:
        node = queue.popleft()
        if node != source and (
            node == centroid
            or (assignment.get(node) == zone and node not in unsupported)
        ):
            end = node
            break
        for neighbor in sorted(relation[node][school_id]):
            if neighbor in parent or (
                neighbor in centroid_set and neighbor != centroid
            ):
                continue
            parent[neighbor] = node
            queue.append(neighbor)
    path = []
    node = parent.get(end) if end is not None else None
    while node is not None and node != source:
        path.append(node)
        node = parent[node]
    return path


def relax_unsupported_candidates(
    G: nx.Graph,
    assignment: dict[int, int],
    centroids: list[int],
    centroid_school_ids: list[int],
    candidates: dict[int, set[int]],
    paths: str = "all",
) -> dict[int, set[int]]:
    """Relax projected blocks without a same-zone closer neighbor.

    The unsupported block itself falls back to distance-based candidacy. Its
    projected zone is also opened along closer-neighbor paths so a reassignment
    can retain monotone support toward the centroid: with ``paths="all"`` every
    block on every such path to the school, with ``paths="shortest"`` only one
    path with the fewest blocks, ending at the first block already in that
    zone with its own support (or at the centroid).
    """
    if paths not in RELEASE_PATHS:
        raise ValueError(f"paths must be one of: {', '.join(RELEASE_PATHS)}.")
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
    if paths == "shortest":
        for zone, sources in unsupported.items():
            school_id = int(centroid_school_ids[zone])
            for source in sorted(sources):
                path = _shortest_support_path(
                    relation,
                    assignment,
                    source,
                    zone,
                    school_id,
                    centroids[zone],
                    centroid_set,
                    sources,
                )
                for node in path:
                    if node in result:
                        result[node].add(zone)
        for zone, centroid in enumerate(centroids):
            result[centroid] = {zone}
        return result

    for zone, sources in unsupported.items():
        school_id = int(centroid_school_ids[zone])
        stack = [
            neighbor for source in sources for neighbor in relation[source][school_id]
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

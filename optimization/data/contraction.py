"""Contract connected groups of fixed nodes into single nodes, exactly.

A refinement step fixes most of the graph: only a band along the source
zoning's boundaries stays free. Every fixed node still costs the CP model its
assignment literals, contiguity rows and balance terms, and a single-worker
CP-SAT solve has to complete a hint over all of them. :func:`contract_fixed`
builds a smaller :class:`ZoneProblem` whose zonings expand one-to-one to the
original problem's zonings with fixed nodes held:

* per zone, each connected component (in ``G``) of nodes forced to that zone
  -- ``fixed`` nodes and the zone's centroid whose candidate set is exactly
  that zone -- becomes one node. A group holding the centroid becomes the
  zone's centroid node.
* a fixed non-centroid node none of whose closer-neighbour supports for its
  zone's school lies in its group is *held out*: it stays its own fixed node,
  so its contiguity row (at least one support in its zone) is kept as is.
  Every remaining member has a support in the original group, which is fixed
  to the zone, so its row always holds and dropping it is exact. Holding
  nodes out can split a group; groups are recomputed on what remains.
* additive node attributes (students, capacities, FRL, ethnicities, schools,
  land area, the capacity/FRL reference counts) are summed, so every zone sum
  -- balance rows, school counts, resident floor -- is unchanged.
* an edge from a group to an outside node carries the sum of its members'
  edge weights to that node (unit weight per member edge when the solve is
  unweighted), so every cut cost is unchanged. The reduced problem therefore
  always solves with ``weight_edges=True``.
* closer neighbours are mapped through the contraction, never recomputed. A
  free node's support on a member becomes the group's node. A group's
  supports are its members' supports outside the group, mapped. The closer
  relation is strict in geometry distance, so a centroid-free group's member
  nearest the school has its support outside the group: on a held-out node,
  fixed to the zone. The group's own row therefore always holds, as the
  members' rows did.
* distances: a group is represented by its centroid when it holds one,
  otherwise by its member nearest its zone's centroid. Free nodes' distances
  to every centroid, and so their candidate zones and distance costs, are
  unchanged. A group's own distance cost is a constant (it is fixed), so the
  reduced distance objective differs from the original by a constant; callers
  recompute the objective on the original problem after :meth:`expand`.
  School-geometry distances follow the same rule (the centroid's, otherwise
  the minimum over members); nothing in a solve reads them.

A free or held-out node keeps its id and a group takes its smallest member's,
so reduced ids are a subset of the original ones (not ``0..A'-1``).

The transform is in-memory only: no geometry, no cache, one pass over nodes
and edges. CP-SAT's integer rounding of a summed coefficient is never
stricter than the rounding of its parts, and expansions are checked with
:func:`optimization.data.feasibility.check_zoning` on the original problem.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import networkx as nx

from optimization.data.closer_neighbors import (
    CLOSER_NEIGHBORS_GRAPH_KEY,
    SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY,
)
from optimization.data.edge_weights import (
    BOUNDARY_WEIGHT_ATTR,
    MANUAL_EDGE_ATTR,
    SHARED_BOUNDARY_ATTR,
    edge_weight,
)
from optimization.problem import ZoneProblem

# Numeric node attributes that are positions or identifiers, never summed.
_NOT_ADDITIVE = frozenset({"area_id", "lat", "lon"})
# Non-additive attributes a merged node does not carry.
_DROPPED = frozenset({"geometry", "area_id", "block_ids", "school_ids"})
_REFERENCE_COUNTS = (
    "capacity_student_counts",
    "capacity_seat_counts",
    "frl_reference_student_counts",
    "frl_reference_seat_counts",
)


@dataclass
class Contraction:
    """A reduced problem and the node maps between it and the original."""

    original: ZoneProblem
    reduced: ZoneProblem
    node_map: dict[int, int]  # original node -> reduced node
    members: dict[int, list[int]]  # reduced node -> original nodes
    held_out: int

    @property
    def nodes_before(self) -> int:
        return self.original.A

    @property
    def nodes_after(self) -> int:
        return self.reduced.A

    def expand(self, assignment: Mapping[int, int]) -> dict[int, int]:
        """Every original node takes its reduced node's zone."""
        return {
            node: int(assignment[reduced])
            for node, reduced in self.node_map.items()
            if reduced in assignment
        }

    def expand_tuple(self, zones) -> tuple[int, ...]:
        """A zone tuple in reduced node order, in original node order."""
        by_node = dict(zip(self.reduced.nodes, zones))
        return tuple(int(by_node[self.node_map[node]]) for node in self.original.nodes)

    def metadata(self) -> dict:
        return {
            "contracted_nodes_before": self.nodes_before,
            "contracted_nodes_after": self.nodes_after,
            "contracted_held_out": self.held_out,
        }


class _RowMap(Mapping):
    """Distances from one representative node, keyed by reduced node."""

    def __init__(self, row, reps):
        self._row = row
        self._reps = reps

    def __getitem__(self, node):
        return self._row[self._reps[node]]

    def __iter__(self):
        return iter(self._reps)

    def __len__(self):
        return len(self._reps)


class _DistanceMap(Mapping):
    """``distance_dict`` of the reduced graph, read through representatives."""

    def __init__(self, distances, reps):
        self._distances = distances
        self._reps = reps

    def __getitem__(self, node):
        return _RowMap(self._distances[self._reps[node]], self._reps)

    def __iter__(self):
        return iter(self._reps)

    def __len__(self):
        return len(self._reps)


def forced_zones(problem: ZoneProblem) -> dict[int, int]:
    """``{node: zone}`` for every node whose only candidate is that zone."""
    forced = {}
    for node in set(problem.fixed or {}) | set(problem.centroids):
        zones = problem.candidate_zones(node)
        if len(zones) == 1:
            forced[node] = next(iter(zones))
    return forced


def _groups(problem: ZoneProblem, forced: dict[int, int]):
    """Contractible groups (lists of nodes, len > 1) and the held-out count."""
    relation = problem.G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    centroids = set(problem.centroids)
    by_zone: dict[int, set[int]] = {}
    for node, zone in forced.items():
        by_zone.setdefault(zone, set()).add(node)
    groups, held_out = [], 0
    for zone, nodes in by_zone.items():
        school = int(problem.centroid_school_ids[zone])
        for component in nx.connected_components(problem.G.subgraph(nodes)):
            if len(component) < 2:
                continue
            kept = {
                node
                for node in component
                if node in centroids or not relation[node][school].isdisjoint(component)
            }
            held_out += len(component) - len(kept)
            for piece in nx.connected_components(problem.G.subgraph(kept)):
                if len(piece) > 1:
                    groups.append((zone, sorted(piece)))
    return groups, held_out


def contract_fixed(problem: ZoneProblem, *, weighted: bool) -> Contraction | None:
    """The reduced problem, or ``None`` when no group has two or more nodes.

    ``weighted`` is whether the solve measures cut edges by boundary weight
    (``weight_edges`` or a ``boundary_perimeter`` objective); the reduced
    edges carry those weights, or member-edge counts when unweighted.
    """
    G = problem.G
    forced = forced_zones(problem)
    groups, held_out = _groups(problem, forced)
    if not groups:
        return None

    group_of: dict[int, int] = {}
    for index, (_, members) in enumerate(groups):
        for node in members:
            group_of[node] = index

    # A free or held-out node keeps its id, and a group takes its smallest
    # member's, so hints, candidates and most closer-neighbour rows carry
    # over unchanged.
    node_map: dict[int, int] = {}
    members: dict[int, list[int]] = {}
    for node in sorted(G.nodes()):
        reduced = groups[group_of[node]][1][0] if node in group_of else node
        node_map[node] = reduced
        members.setdefault(reduced, []).append(node)

    centroid_set = set(problem.centroids)
    centroid_zone = {c: z for z, c in enumerate(problem.centroids)}
    distances = G.graph["distance_dict"]
    reps: dict[int, int] = {}
    for reduced, nodes in members.items():
        if len(nodes) == 1:
            reps[reduced] = nodes[0]
            continue
        inside = [n for n in nodes if n in centroid_set]
        if inside:
            reps[reduced] = inside[0]
        else:
            zone = forced[nodes[0]]
            row = distances[problem.centroids[zone]]
            reps[reduced] = min(nodes, key=lambda n: (row[n], n))

    H = nx.Graph()
    for reduced, nodes in members.items():
        if len(nodes) == 1:
            H.add_node(reduced, **G.nodes[nodes[0]])
            continue
        H.add_node(reduced, **_merged_attributes(G, nodes, reps[reduced]))

    for u, v, attrs in G.edges(data=True):
        ru, rv = node_map[u], node_map[v]
        if ru == rv:
            continue
        weight = edge_weight(G, u, v, weighted=weighted)
        if not H.has_edge(ru, rv):
            H.add_edge(
                ru,
                rv,
                **{
                    BOUNDARY_WEIGHT_ATTR: 0,
                    SHARED_BOUNDARY_ATTR: 0.0,
                    MANUAL_EDGE_ATTR: False,
                },
            )
        edge = H.edges[ru, rv]
        edge[BOUNDARY_WEIGHT_ATTR] += weight
        edge[SHARED_BOUNDARY_ATTR] += float(attrs.get(SHARED_BOUNDARY_ATTR, 0.0))
        edge[MANUAL_EDGE_ATTR] |= bool(attrs.get(MANUAL_EDGE_ATTR, False))

    H.graph.update(G.graph)
    H.graph.pop("partition", None)
    H.graph["weight_edges"] = True
    H.graph["distance_dict"] = _DistanceMap(distances, reps)
    H.graph[CLOSER_NEIGHBORS_GRAPH_KEY] = _closer_relation(
        G, members, node_map, group_of
    )
    if SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY in G.graph:
        H.graph[SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY] = _school_distances(
            G.graph[SCHOOL_GEOMETRY_DISTANCES_GRAPH_KEY], members, centroid_set
        )

    fixed, hint = {}, {}
    for reduced, nodes in members.items():
        if len(nodes) > 1:
            zone = forced[nodes[0]]
            if not any(n in centroid_zone for n in nodes):
                fixed[reduced] = zone
            hint[reduced] = zone
            continue
        node = nodes[0]
        if problem.fixed and node in problem.fixed:
            fixed[reduced] = problem.fixed[node]
        if problem.hint and node in problem.hint:
            hint[reduced] = problem.hint[node]
    candidates = (
        {
            node_map[node]: set(zones)
            for node, zones in problem.candidates.items()
            if node not in group_of
        }
        if problem.candidates
        else None
    )
    counts = {
        name: _summed(getattr(problem, name), members)
        for name in _REFERENCE_COUNTS
        if getattr(problem, name) is not None
    }
    reduced_problem = replace(
        problem,
        G=H,
        centroids=[node_map[c] for c in problem.centroids],
        weight_edges=True,
        fixed=fixed or None,
        candidates=candidates or None,
        hint=hint or None,
        _candidates=None,
        **counts,
    )
    return Contraction(problem, reduced_problem, node_map, members, held_out)


def _merged_attributes(G: nx.Graph, nodes: list[int], rep: int) -> dict:
    merged: dict = {}
    school_ids, base_ids = [], []
    for node in nodes:
        attrs = G.nodes[node]
        for key, value in attrs.items():
            if key in _DROPPED or key in _NOT_ADDITIVE:
                continue
            if isinstance(value, bool):
                merged[key] = merged.get(key, False) or value
            elif isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
        school_ids.extend(attrs.get("school_ids", []))
        base_ids.extend(
            [attrs["area_id"]] if "area_id" in attrs else attrs.get("block_ids", [])
        )
    rep_attrs = G.nodes[rep]
    for key in ("lat", "lon"):
        if key in rep_attrs:
            merged[key] = rep_attrs[key]
    merged["school_ids"] = school_ids
    merged["block_ids"] = base_ids
    return merged


def _closer_relation(G, members, node_map, group_of):
    relation = G.graph[CLOSER_NEIGHBORS_GRAPH_KEY]
    reduced = {}
    for node_id, nodes in members.items():
        if len(nodes) == 1:
            node = nodes[0]
            schools = relation[node]
            # Only a node next to a group names a member; the rest share
            # their original (immutable) rows.
            if any(nb in group_of for nb in G.neighbors(node)):
                schools = {
                    school: frozenset(node_map[nb] for nb in supports)
                    for school, supports in schools.items()
                }
            reduced[node_id] = schools
            continue
        inside = set(nodes)
        union: dict[int, set[int]] = {}
        for node in nodes:
            for school, supports in relation[node].items():
                bucket = union.setdefault(school, set())
                bucket.update(node_map[nb] for nb in supports if nb not in inside)
        reduced[node_id] = {
            school: frozenset(supports) for school, supports in union.items()
        }
    return reduced


def _school_distances(distances, members, centroid_set):
    reduced = {}
    for node_id, nodes in members.items():
        if len(nodes) == 1:
            reduced[node_id] = distances[nodes[0]]
            continue
        inside = [n for n in nodes if n in centroid_set]
        if inside:
            reduced[node_id] = distances[inside[0]]
            continue
        schools = set().union(*(distances[n] for n in nodes))
        reduced[node_id] = {
            school: min(distances[n][school] for n in nodes if school in distances[n])
            for school in schools
        }
    return reduced


def _summed(counts, members):
    return {
        reduced: sum(counts[node] for node in nodes)
        for reduced, nodes in members.items()
    }

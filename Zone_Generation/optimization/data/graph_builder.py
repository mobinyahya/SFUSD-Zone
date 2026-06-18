"""Graph generation and aggregation.

Builds the node-attributed adjacency graphs the optimizer consumes, for *either*
unit (Block or BlockGroup) at *any* depth. This replaces the legacy
``create_larger_areas.py``, which hardcoded ``level='BlockGroup'`` and a fixed
two-level hierarchy.

* :func:`build_base_graph` -- the depth-0 graph from the per-area tables.
* :func:`aggregate`        -- collapse a base graph under a partition into a
  coarser graph (adjacency rederived from base edges, so no reliance on
  re-dissolving shapefiles).
* :func:`build_hierarchy`  -- METIS recursive split + aggregate to produce a
  ``{depth: graph}`` hierarchy.

Graph/node attribute schema matches CLAUDE.md.
"""

from __future__ import annotations

import networkx as nx

from Helper_Functions.util import calculate_euc_distance
from Zone_Generation.Config.Constants import AREA_ETHNICITIES
from Zone_Generation.optimization.data import loaders
from Zone_Generation.optimization.data.loaders import IngestConfig

# Node attributes summed when aggregating.
_SUM_ATTRS = [
    "ge_students",
    "ge_capacity",
    "all_prog_students",
    "all_prog_capacity",
    "num_schools",
    "FRL",
]


# ====================================================================== #
# Base graph
# ====================================================================== #
def build_base_graph(cfg: IngestConfig) -> nx.Graph:
    """Depth-0 graph for ``cfg.unit``."""
    area = loaders.load_area_table(cfg)
    area2idx = {int(a): idx for idx, a in zip(area.index, area[cfg.unit])}

    G = nx.Graph()
    distance_dict = loaders.load_distance_dict(cfg, area2idx)
    neighbors = loaders.load_neighbors(cfg, area2idx)

    total_students = 0.0
    total_frl = 0.0
    total_eth = {eth: 0.0 for eth in AREA_ETHNICITIES}

    for idx, row in area.iterrows():
        attrs = {
            "area_id": int(row[cfg.unit]),
            "ge_students": float(row["ge_students"]),
            "ge_capacity": float(row["ge_capacity"]),
            "all_prog_students": float(row["all_prog_students"]),
            "all_prog_capacity": float(row["all_prog_capacity"]),
            "num_schools": int(row["num_schools"]),
            "FRL": float(row["FRL"]),
            "school_ids": list(row["school_ids"]),
            "lat": float(row["Lat"]),
            "lon": float(row["Lon"]),
        }
        for eth in AREA_ETHNICITIES:
            attrs[eth] = float(row[eth])
            total_eth[eth] += attrs[eth]
        total_students += attrs["ge_students"]
        total_frl += attrs["FRL"]
        G.add_node(idx, **attrs)

    for idx in area.index:
        for nb in neighbors.get(idx, []):
            G.add_edge(idx, nb)

    G.graph["distance_dict"] = distance_dict
    G.graph["school_data"] = _school_data(cfg)
    G.graph["F"] = (total_frl / total_students) if total_students else 0.0
    G.graph["R"] = {
        eth: (total_eth[eth] / total_students if total_students else 0.0)
        for eth in AREA_ETHNICITIES
    }
    return G


def _school_data(cfg: IngestConfig) -> dict:
    schools = loaders.load_schools(cfg)
    data = {}
    for _, row in schools.iterrows():
        info = row.to_dict()
        sid = info.pop("school_id", None)
        data[sid] = info
    return data


# ====================================================================== #
# Aggregation
# ====================================================================== #
def aggregate(base_G: nx.Graph, partition: dict[int, int]) -> nx.Graph:
    """Collapse ``base_G`` under ``partition`` (``{base_node: part_id}``).

    Sums per-area attributes, rederives adjacency from base edges that cross
    parts, recomputes centroids/distances, and records the mapping in
    ``G.graph['partition']`` (and each node's ``block_ids``).
    """
    new_G = nx.Graph()

    for node, part in partition.items():
        if part not in new_G:
            new_G.add_node(
                part,
                **{a: 0.0 for a in _SUM_ATTRS},
                **{e: 0.0 for e in AREA_ETHNICITIES},
                school_ids=[],
                block_ids=[],
                lat=0.0,
                lon=0.0,
            )
        n = new_G.nodes[part]
        b = base_G.nodes[node]
        for a in _SUM_ATTRS:
            n[a] += float(b[a])
        for e in AREA_ETHNICITIES:
            n[e] += float(b[e])
        n["school_ids"].extend(b.get("school_ids", []))
        n["block_ids"].append(b["area_id"])

    # population-weighted centroids
    acc: dict[int, list[float]] = {p: [0.0, 0.0, 0.0] for p in new_G.nodes()}
    for node, part in partition.items():
        b = base_G.nodes[node]
        w = float(b["ge_students"]) + 1e-9
        acc[part][0] += w * float(b["lat"])
        acc[part][1] += w * float(b["lon"])
        acc[part][2] += w
    for part, (slat, slon, w) in acc.items():
        new_G.nodes[part]["lat"] = slat / w if w else 0.0
        new_G.nodes[part]["lon"] = slon / w if w else 0.0

    # adjacency from crossing base edges
    for u, v in base_G.edges():
        pu, pv = partition[u], partition[v]
        if pu != pv:
            new_G.add_edge(pu, pv)

    # distances between aggregated centroids
    distance_dict: dict[int, dict[int, float]] = {}
    for i in new_G.nodes():
        distance_dict[i] = {}
        lat_i, lon_i = new_G.nodes[i]["lat"], new_G.nodes[i]["lon"]
        for j in new_G.nodes():
            if i == j:
                distance_dict[i][j] = 0.0
            else:
                distance_dict[i][j] = calculate_euc_distance(
                    lat_i, lon_i, new_G.nodes[j]["lat"], new_G.nodes[j]["lon"]
                )
    new_G.graph["distance_dict"] = distance_dict
    new_G.graph["F"] = base_G.graph["F"]
    new_G.graph["R"] = base_G.graph["R"]
    new_G.graph["school_data"] = base_G.graph["school_data"]
    new_G.graph["partition"] = partition
    return new_G


# ====================================================================== #
# Hierarchy
# ====================================================================== #
def _recursive_split(
    G: nx.Graph, cur_size: int, depth: int, offset: int = 0
) -> tuple[dict[int, int], int]:
    """METIS recursive partition; returns ``({node: part}, next_part_id)``."""
    if depth == 0 or cur_size <= 4:
        return {node: offset for node in G.nodes()}, offset + 1
    # Imported lazily: METIS is only needed when actually building a hierarchy,
    # so the package (and its tests) import without pymetis installed.
    from Zone_Generation.Optimization.graph_utils import (
        partition_graph_metis_partial_constraint,
    )

    supers = partition_graph_metis_partial_constraint(G, cur_size)
    zone_dict: dict[int, int] = {}
    cur = offset
    for nodes in supers.values():
        sub = G.subgraph(nodes).copy()
        sub_zones, cur = _recursive_split(sub, cur_size // 3, depth - 1, cur)
        zone_dict.update(sub_zones)
    return zone_dict, cur


def aggregate_level(
    base_G: nx.Graph, split_depth: int, split_base: int = 3 ** 3
) -> nx.Graph:
    """Aggregate ``base_G`` once using a METIS recursion of ``split_depth``.

    Larger ``split_depth`` = finer aggregation (more nodes). Used by the dataset
    to materialize a single coarser level on demand from a cached base graph.
    """
    partition, _ = _recursive_split(base_G, split_base, split_depth)
    return aggregate(base_G, partition)


def build_hierarchy(
    cfg: IngestConfig,
    level_to_split: dict[int, int] | None = None,
    split_base: int = 3 ** 3,
) -> dict[int, nx.Graph]:
    """Build ``{depth: graph}`` for ``cfg.unit``.

    ``level_to_split`` maps a LevelSpec depth (>=1) to the METIS recursion depth
    used to partition the base graph. Larger recursion depth = finer
    aggregation, so the default ``{1: 2, 2: 1}`` makes depth 1 finer than
    depth 2 (mirroring the legacy two-level hierarchy).
    """
    if level_to_split is None:
        level_to_split = {1: 2, 2: 1}

    base = build_base_graph(cfg)
    graphs: dict[int, nx.Graph] = {0: base}
    for depth, split_depth in sorted(level_to_split.items()):
        partition, _ = _recursive_split(base, split_base, split_depth)
        graphs[depth] = aggregate(base, partition)
    return graphs

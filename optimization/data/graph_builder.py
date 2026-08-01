"""Build base graphs and school-preserving KaHIP graph hierarchies."""

from __future__ import annotations

import math
from importlib.metadata import version

import networkx as nx

from Config.Constants import AREA_ETHNICITIES
from optimization.data import edge_overrides, loaders
from optimization.data.geography import great_circle_miles
from optimization.data.loaders import IngestConfig
from optimization.levels import LEVEL_NODE_TARGETS

# Node attributes summed when aggregating.
_SUM_ATTRS = [
    "ge_students",
    "ge_capacity",
    "all_prog_students",
    "all_prog_capacity",
    "num_schools",
    "FRL",
]

# Version 3 moves Mission Bay ES (999) from Block 60750607001031 to 60750607001053.
# GRAPH_CACHE_SCHEMA_VERSION changed from 4 to 5 after moving its distance-cache
# source row from Block 60750607001031 to Block 60750607001053.
# /home/kumarc/sfusd-local-data/zones/SFUSD/Optimization/distances_b2b_schools.before_mission_bay_row_update.csv
GRAPH_CACHE_SCHEMA_VERSION = 7
PARTITION_INITIAL_IMBALANCE = 0.8
PARTITION_MAX_ATTEMPTS = 14
PARTITION_SEED = 42
PARTITION_WEIGHT_SCALE = 1000
PARTITION_MODE = "strong"


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

    population_attr = population_attribute(cfg.population_type)
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
        total_students += attrs[population_attr]
        total_frl += attrs["FRL"]
        G.add_node(idx, **attrs)

    for idx in area.index:
        for nb in neighbors.get(idx, []):
            G.add_edge(idx, nb)

    if cfg.unit == "Block":
        edge_overrides.apply_block_edge_overrides(G)

    G.graph["distance_dict"] = distance_dict
    G.graph["school_data"] = _school_data(cfg)
    G.graph["population_type"] = cfg.population_type
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
def aggregate(parent_G: nx.Graph, partition: dict[int, int]) -> nx.Graph:
    """Collapse ``parent_G`` under ``partition`` (``{parent_node: part_id}``).

    Sums per-area attributes, rederives adjacency from parent edges that cross
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
        b = parent_G.nodes[node]
        for a in _SUM_ATTRS:
            n[a] += float(b[a])
        for e in AREA_ETHNICITIES:
            n[e] += float(b[e])
        n["school_ids"].extend(b.get("school_ids", []))
        if "area_id" in b:
            n["block_ids"].append(b["area_id"])
        else:
            n["block_ids"].extend(b["block_ids"])

    # population-weighted centroids
    acc: dict[int, list[float]] = {p: [0.0, 0.0, 0.0] for p in new_G.nodes()}
    for node, part in partition.items():
        b = parent_G.nodes[node]
        w = float(b["ge_students"]) + 1e-9
        acc[part][0] += w * float(b["lat"])
        acc[part][1] += w * float(b["lon"])
        acc[part][2] += w
    for part, (slat, slon, w) in acc.items():
        new_G.nodes[part]["lat"] = slat / w if w else 0.0
        new_G.nodes[part]["lon"] = slon / w if w else 0.0

    # Adjacency from crossing parent edges also reattaches school singletons.
    for u, v in parent_G.edges():
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
                distance_dict[i][j] = great_circle_miles(
                    lat_i, lon_i, new_G.nodes[j]["lat"], new_G.nodes[j]["lon"]
                )
    new_G.graph["distance_dict"] = distance_dict
    new_G.graph["F"] = parent_G.graph["F"]
    new_G.graph["R"] = parent_G.graph["R"]
    new_G.graph["school_data"] = parent_G.graph["school_data"]
    new_G.graph["population_type"] = parent_G.graph.get("population_type", "GE")
    new_G.graph["partition"] = dict(partition)
    if "manual_block_edges" in parent_G.graph:
        new_G.graph["manual_block_edges"] = parent_G.graph["manual_block_edges"]
        new_G.graph["manual_block_edge_fingerprint"] = parent_G.graph[
            "manual_block_edge_fingerprint"
        ]
    return new_G


# ====================================================================== #
# Hierarchy partitioning
# ====================================================================== #
def partition_cache_policy(unit: str) -> dict:
    """Return the partition policy included in graph cache namespaces."""
    return {
        "backend": "kahip",
        "backend_version": version("kahip"),
        "mode": PARTITION_MODE,
        "seed": PARTITION_SEED,
        "initial_imbalance": PARTITION_INITIAL_IMBALANCE,
        "weight_scale": PARTITION_WEIGHT_SCALE,
        "school_nodes_are_singletons": True,
        "hierarchical": True,
        "node_targets": {
            str(depth): target
            for depth, target in sorted(LEVEL_NODE_TARGETS.get(unit, {}).items())
        },
    }


def population_attribute(population_type: str) -> str:
    """Node attribute representing the population selected during ingestion."""
    return "ge_students" if population_type == "GE" else "all_prog_students"


def _is_school_node(attrs: dict) -> bool:
    return bool(attrs.get("school_ids")) or float(attrs.get("num_schools", 0)) > 0


def _integer_population_weights(
    G: nx.Graph, nodes: list[int], population_attr: str
) -> list[int]:
    weights = []
    for node in nodes:
        population = float(G.nodes[node][population_attr])
        if not math.isfinite(population) or population < 0:
            raise ValueError(
                f"Invalid {population_attr}={population!r} for node {node}."
            )
        weights.append(round(population * PARTITION_WEIGHT_SCALE))
    # KaHIP needs a meaningful balance dimension even for an empty population.
    return weights if any(weights) else [1] * len(nodes)


def _partition_graph_kahip(
    G: nx.Graph,
    target_partition_count: int,
    population_attr: str,
) -> tuple[dict[int, list[int]], float]:
    """Partition one connected graph, relaxing balance until output is valid."""
    nodes = sorted(G.nodes())
    if not nodes:
        return {}, PARTITION_INITIAL_IMBALANCE
    if not nx.is_connected(G):
        raise ValueError("_partition_graph_kahip requires a connected graph.")
    if target_partition_count <= 1:
        return {0: nodes}, PARTITION_INITIAL_IMBALANCE
    if target_partition_count >= len(nodes):
        return (
            {idx: [node] for idx, node in enumerate(nodes)},
            PARTITION_INITIAL_IMBALANCE,
        )

    import kahip

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    xadj = [0]
    adjncy: list[int] = []
    for node in nodes:
        adjncy.extend(sorted(node_to_idx[neighbor] for neighbor in G.neighbors(node)))
        xadj.append(len(adjncy))
    vertex_weights = _integer_population_weights(G, nodes, population_attr)
    edge_weights = [1] * len(adjncy)

    imbalance = PARTITION_INITIAL_IMBALANCE
    last_reason = "KaHIP returned no result"
    for _ in range(PARTITION_MAX_ATTEMPTS):
        try:
            _, membership = kahip.kaffpa(
                vertex_weights,
                xadj,
                edge_weights,
                adjncy,
                target_partition_count,
                imbalance,
                True,
                PARTITION_SEED,
                kahip.STRONG,
            )
        except Exception as exc:
            raise RuntimeError(
                "KaHIP failed while partitioning "
                f"{len(nodes)} nodes into at most {target_partition_count} parts."
            ) from exc

        if len(membership) != len(nodes):
            raise RuntimeError(
                f"KaHIP returned {len(membership)} memberships for {len(nodes)} nodes."
            )
        if any(part < 0 or part >= target_partition_count for part in membership):
            raise RuntimeError("KaHIP returned an out-of-range partition id.")

        groups: dict[int, list[int]] = {}
        totals: dict[int, int] = {}
        for node, weight, part in zip(nodes, vertex_weights, membership):
            part = int(part)
            groups.setdefault(part, []).append(node)
            totals[part] = totals.get(part, 0) + weight

        average = math.ceil(sum(vertex_weights) / target_partition_count)
        upper_bound = (1 + imbalance) * average
        overweight = max(totals.values(), default=0) > upper_bound
        disconnected = any(
            not nx.is_connected(G.subgraph(group)) for group in groups.values()
        )
        if not overweight and not disconnected:
            compact = {
                new_part: groups[old_part]
                for new_part, old_part in enumerate(sorted(groups))
            }
            return compact, imbalance

        failures = []
        if overweight:
            failures.append("population imbalance")
        if disconnected:
            failures.append("disconnected aggregate")
        last_reason = " and ".join(failures)
        imbalance *= 2

    raise RuntimeError(
        f"KaHIP could not produce a valid partition after "
        f"{PARTITION_MAX_ATTEMPTS} attempts: {last_reason}."
    )


def _component_partition_counts(
    G: nx.Graph,
    components: list[list[int]],
    target_partition_count: int,
    population_attr: str,
) -> list[int]:
    """Allocate parts across components in proportion to selected population."""
    if target_partition_count < len(components):
        raise ValueError(
            f"Cannot form at most {target_partition_count} connected aggregates from "
            f"{len(components)} connected components."
        )

    counts = [1] * len(components)
    populations = [
        sum(float(G.nodes[node][population_attr]) for node in component)
        for component in components
    ]
    if not any(populations):
        populations = [float(len(component)) for component in components]
    remaining = min(target_partition_count, sum(map(len, components))) - len(components)
    while remaining:
        candidates = [
            idx
            for idx, component in enumerate(components)
            if counts[idx] < len(component)
        ]
        chosen = max(
            candidates,
            key=lambda idx: (populations[idx] / counts[idx], -idx),
        )
        counts[chosen] += 1
        remaining -= 1
    return counts


def _partition_non_school_nodes(
    G: nx.Graph,
    target_partition_count: int,
    population_attr: str,
) -> tuple[dict[int, int], list[float]]:
    if not G:
        return {}, []
    components = [sorted(component) for component in nx.connected_components(G)]
    components.sort(key=lambda component: (-len(component), component[0]))
    counts = _component_partition_counts(
        G,
        components,
        target_partition_count,
        population_attr,
    )

    partition: dict[int, int] = {}
    imbalances = []
    offset = 0
    for component, count in zip(components, counts):
        groups, imbalance = _partition_graph_kahip(
            G.subgraph(component).copy(), count, population_attr
        )
        imbalances.append(imbalance)
        for part, nodes in groups.items():
            for node in nodes:
                partition[node] = offset + part
        offset += len(groups)
    return partition, imbalances


def aggregate_level(
    parent_G: nx.Graph,
    target_node_count: int,
    population_type: str,
) -> nx.Graph:
    """Build one coarse graph from its immediate finer parent graph."""
    school_nodes = sorted(
        node for node, attrs in parent_G.nodes(data=True) if _is_school_node(attrs)
    )
    if len(school_nodes) > target_node_count:
        raise ValueError(
            f"Target {target_node_count} cannot preserve {len(school_nodes)} school "
            "nodes as singleton vertices."
        )
    school_node_set = set(school_nodes)
    non_school_nodes = [node for node in parent_G if node not in school_node_set]
    non_school_target = target_node_count - len(school_nodes)
    if non_school_nodes and non_school_target < 1:
        raise ValueError(
            f"Target {target_node_count} cannot preserve {len(school_nodes)} school "
            "nodes and aggregate the remaining nodes."
        )

    population_attr = population_attribute(population_type)
    partition, imbalances = _partition_non_school_nodes(
        parent_G.subgraph(non_school_nodes).copy(),
        non_school_target,
        population_attr,
    )
    next_part = max(partition.values(), default=-1) + 1
    for node in school_nodes:
        partition[node] = next_part
        next_part += 1

    coarse = aggregate(parent_G, partition)
    coarse.graph.update(
        {
            "partition_backend": "kahip",
            "partition_mode": PARTITION_MODE,
            "partition_seed": PARTITION_SEED,
            "partition_population_attribute": population_attr,
            "partition_initial_imbalance": PARTITION_INITIAL_IMBALANCE,
            "partition_imbalance": max(imbalances, default=PARTITION_INITIAL_IMBALANCE),
            "target_node_count": target_node_count,
            "actual_node_count": len(coarse),
            "school_singleton_count": len(school_nodes),
        }
    )
    return coarse


def build_hierarchy(cfg: IngestConfig) -> dict[int, nx.Graph]:
    """Build every predefined level sequentially from its immediate parent."""
    graphs: dict[int, nx.Graph] = {0: build_base_graph(cfg)}
    for depth, target in sorted(LEVEL_NODE_TARGETS[cfg.unit].items()):
        graphs[depth] = aggregate_level(graphs[depth - 1], target, cfg.population_type)
    return graphs

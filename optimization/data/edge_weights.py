"""Shared-boundary edge weights for graph partitioning and optimization."""

from __future__ import annotations

import math
from collections.abc import Mapping

import networkx as nx


BOUNDARY_WEIGHT_ATTR = "boundary_weight"
SHARED_BOUNDARY_ATTR = "shared_boundary_m"
MANUAL_EDGE_ATTR = "manual_edge"
BOUNDARY_CRS = "EPSG:32610"
MANUAL_EDGE_QUANTILE = 0.1
MIN_BOUNDARY_WEIGHT = 1


def assign_boundary_weights(
    G: nx.Graph,
    geometry,
    unit: str,
    *,
    source_edges: set[tuple[int, int]],
    manual_area_edges: list[tuple[int, int]],
) -> None:
    """Measure shared boundaries and attach integer-metre edge costs."""
    projected = geometry.to_crs(BOUNDARY_CRS)
    boundaries = {
        int(row[unit]): row.geometry.boundary
        for _, row in projected[[unit, "geometry"]].iterrows()
    }
    area_to_node = {int(attrs["area_id"]): node for node, attrs in G.nodes(data=True)}
    missing = sorted(set(area_to_node) - set(boundaries))
    if missing:
        raise ValueError(f"Boundary geometry is missing {unit} GEOIDs: {missing[:5]}.")

    manual_edges = {
        _edge_key(area_to_node[first], area_to_node[second])
        for first, second in manual_area_edges
    }
    lengths = {}
    for u, v in G.edges():
        area_u = int(G.nodes[u]["area_id"])
        area_v = int(G.nodes[v]["area_id"])
        length = float(boundaries[area_u].intersection(boundaries[area_v]).length)
        if not math.isfinite(length) or length < 0:
            raise ValueError(f"Invalid shared boundary length for edge {(u, v)}.")
        lengths[_edge_key(u, v)] = length

    synthetic_manual_edges = [
        edge for edge in manual_edges if lengths.get(edge, 0.0) == 0.0
    ]
    manual_weight_m = None
    if synthetic_manual_edges:
        reference = sorted(
            length
            for edge, length in lengths.items()
            if edge in source_edges and length > 0
        )
        if not reference:
            raise ValueError(
                "Cannot weight synthetic manual edges without positive source "
                "boundary lengths."
            )
        rank = max(0, math.ceil(MANUAL_EDGE_QUANTILE * len(reference)) - 1)
        manual_weight_m = reference[rank]

    for u, v in G.edges():
        edge = _edge_key(u, v)
        shared_length = lengths[edge]
        effective_length = (
            manual_weight_m
            if edge in manual_edges and shared_length == 0.0
            else shared_length
        )
        attrs = G.edges[u, v]
        attrs[SHARED_BOUNDARY_ATTR] = shared_length
        attrs[BOUNDARY_WEIGHT_ATTR] = max(
            MIN_BOUNDARY_WEIGHT, int(round(effective_length))
        )
        attrs[MANUAL_EDGE_ATTR] = edge in manual_edges

    G.graph.update(
        {
            "weight_edges": True,
            "boundary_crs": BOUNDARY_CRS,
            "boundary_weight_unit": "meter",
            "manual_edge_weight_m": manual_weight_m,
        }
    )


def edge_weight(
    G: nx.Graph,
    u: int,
    v: int,
    *,
    weighted: bool | None = None,
) -> int:
    """Return one validated edge coefficient."""
    enabled = bool(G.graph.get("weight_edges", False)) if weighted is None else weighted
    if not enabled:
        return 1
    value = G.edges[u, v].get(BOUNDARY_WEIGHT_ATTR)
    if isinstance(value, bool):
        raise ValueError(f"Invalid boundary weight {value!r} for edge {(u, v)}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid boundary weight {value!r} for edge {(u, v)}."
        ) from exc
    if (
        not math.isfinite(number)
        or number < MIN_BOUNDARY_WEIGHT
        or not number.is_integer()
    ):
        raise ValueError(f"Invalid boundary weight {value!r} for edge {(u, v)}.")
    return int(number)


def boundary_cost(
    G: nx.Graph,
    assignment: Mapping[int, int],
    *,
    weighted: bool,
) -> int:
    """Return the cut-edge count or weighted boundary cost."""
    return sum(
        edge_weight(G, u, v, weighted=weighted)
        for u, v in G.edges()
        if assignment.get(u) != assignment.get(v)
    )


def weighting_policy() -> dict:
    """Return the complete policy used in weighted graph cache keys."""
    return {
        "attribute": BOUNDARY_WEIGHT_ATTR,
        "measurement": "shared_polygon_boundary",
        "crs": BOUNDARY_CRS,
        "unit": "meter",
        "rounding": "nearest_integer_minimum_one",
        "manual_edge_quantile": MANUAL_EDGE_QUANTILE,
    }


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)

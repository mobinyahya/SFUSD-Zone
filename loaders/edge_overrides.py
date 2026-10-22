"""Stable, manually reviewed overrides for Census Block adjacency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import yaml


def load_block_edge_overrides(path: str | Path) -> list[tuple[int, int]]:
    """Load one manual Block-edge file as normalized undirected GEOID pairs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required manual Block edge file is missing: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    rows = payload.get("edges", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Manual Block edges in {path} must be a list.")

    edges: set[tuple[int, int]] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, list | tuple) or len(row) != 2:
            raise ValueError(
                f"Manual Block edge {index} in {path} must have two GEOIDs."
            )
        if any(isinstance(value, bool) for value in row):
            raise ValueError(f"Manual Block edge {index} in {path} contains a Boolean.")
        try:
            first, second = (int(value) for value in row)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Manual Block edge {index} in {path} has invalid GEOIDs."
            ) from exc
        if first == second:
            raise ValueError(f"Manual Block edge {index} in {path} is a self-edge.")
        edges.add(tuple(sorted((first, second))))
    return sorted(edges)


def apply_block_edge_overrides(
    G: nx.Graph,
    edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Add reviewed Block edges to a graph by resolving stable area IDs."""
    normalized = sorted(
        {tuple(sorted((int(first), int(second)))) for first, second in edges}
    )
    area_to_node = {
        int(attributes["area_id"]): node
        for node, attributes in G.nodes(data=True)
        if "area_id" in attributes
    }
    missing = sorted(
        {
            area_id
            for edge in normalized
            for area_id in edge
            if area_id not in area_to_node
        }
    )
    if missing:
        raise ValueError(
            "Manual Block edges reference GEOIDs absent from the base graph: "
            f"{missing}."
        )

    for first, second in normalized:
        G.add_edge(area_to_node[first], area_to_node[second])
    G.graph["manual_block_edges"] = normalized
    G.graph["manual_block_edge_fingerprint"] = hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return normalized


__all__ = [
    "apply_block_edge_overrides",
    "load_block_edge_overrides",
]

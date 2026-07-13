"""Stable, manually reviewed overrides for Census Block adjacency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import yaml


DEFAULT_BLOCK_EDGE_OVERRIDES = (
    Path(__file__).resolve().parents[2] / "Config" / "manual_block_edges.yaml"
)


def load_block_edge_overrides(
    path: str | Path = DEFAULT_BLOCK_EDGE_OVERRIDES,
) -> list[tuple[int, int]]:
    """Load normalized undirected edge pairs expressed as Block GEOIDs."""
    override_path = Path(path)
    if not override_path.exists():
        return []

    with override_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    rows = payload.get("edges", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Manual Block edges in {override_path} must be a list.")

    edges = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(
                f"Manual Block edge {index} in {override_path} must have two GEOIDs."
            )
        if any(isinstance(value, bool) for value in row):
            raise ValueError(
                f"Manual Block edge {index} in {override_path} contains a Boolean."
            )
        try:
            u, v = (int(value) for value in row)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Manual Block edge {index} in {override_path} has invalid GEOIDs."
            ) from exc
        if u == v:
            raise ValueError(
                f"Manual Block edge {index} in {override_path} is a self-edge."
            )
        edges.add(tuple(sorted((u, v))))
    return sorted(edges)


def block_edge_override_fingerprint(
    path: str | Path = DEFAULT_BLOCK_EDGE_OVERRIDES,
) -> str:
    """Return a content fingerprint for normalized manual Block edges."""
    encoded = json.dumps(
        load_block_edge_overrides(path), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def apply_block_edge_overrides(
    G: nx.Graph,
    edges: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Add reviewed Block edges to ``G`` by resolving stable area IDs."""
    edges = load_block_edge_overrides() if edges is None else edges
    normalized = sorted({tuple(sorted((int(u), int(v)))) for u, v in edges})
    area_to_node = {
        int(attrs["area_id"]): node
        for node, attrs in G.nodes(data=True)
        if "area_id" in attrs
    }
    missing = sorted(
        {area_id for edge in normalized for area_id in edge if area_id not in area_to_node}
    )
    if missing:
        raise ValueError(
            "Manual Block edges reference GEOIDs absent from the base graph: "
            f"{missing}."
        )

    for area_u, area_v in normalized:
        G.add_edge(area_to_node[area_u], area_to_node[area_v])
    G.graph["manual_block_edges"] = normalized
    G.graph["manual_block_edge_fingerprint"] = hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return normalized

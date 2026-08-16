"""Stable, manually reviewed overrides for Census Block adjacency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import yaml

_CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
DEFAULT_BLOCK_EDGE_OVERRIDES = _CONFIG_ROOT / "manual_block_edges.yaml"
DEFAULT_BLOCK_EDGE_ADDITIONS = _CONFIG_ROOT / "manual_block_edge_additions.yaml"


def load_block_edge_overrides(
    path: str | Path | None = None,
) -> list[tuple[int, int]]:
    """Load normalized undirected edge pairs expressed as Block GEOIDs."""
    override_paths = (
        (DEFAULT_BLOCK_EDGE_OVERRIDES, DEFAULT_BLOCK_EDGE_ADDITIONS)
        if path is None
        else (Path(path),)
    )
    edges: set[tuple[int, int]] = set()
    for override_path in override_paths:
        edges.update(_load_block_edge_override_file(override_path))
    return sorted(edges)


def _load_block_edge_override_file(path: Path) -> set[tuple[int, int]]:
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
            raise ValueError(
                f"Manual Block edge {index} in {path} contains a Boolean."
            )
        try:
            first, second = (int(value) for value in row)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Manual Block edge {index} in {path} has invalid GEOIDs."
            ) from exc
        if first == second:
            raise ValueError(
                f"Manual Block edge {index} in {path} is a self-edge."
            )
        edges.add(tuple(sorted((first, second))))
    return edges


def block_edge_override_fingerprint(path: str | Path | None = None) -> str:
    """Return a content fingerprint for normalized manual Block edges."""
    encoded = json.dumps(
        load_block_edge_overrides(path), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def apply_block_edge_overrides(
    G: nx.Graph,
    edges: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Add reviewed Block edges to a graph by resolving stable area IDs."""
    configured = load_block_edge_overrides() if edges is None else edges
    normalized = sorted(
        {tuple(sorted((int(first), int(second)))) for first, second in configured}
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
    "DEFAULT_BLOCK_EDGE_ADDITIONS",
    "DEFAULT_BLOCK_EDGE_OVERRIDES",
    "apply_block_edge_overrides",
    "block_edge_override_fingerprint",
    "load_block_edge_overrides",
]

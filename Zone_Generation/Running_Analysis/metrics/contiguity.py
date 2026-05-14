"""
Contiguity metric: 1 if every zone forms a single connected component
anchored on its centroid school, 0 otherwise.

Mirrors the logic of the standalone check_contiguity.py post-processing
script so results stay backwards-compatible.
"""

from pathlib import Path

import networkx as nx
import yaml

from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.Optimzation_Heuristics.zone_eval import trim_noncontiguity_soft


_CENTROIDS_YAML = Path(__file__).resolve().parents[2] / "Config" / "centroids.yaml"
_centroids_cache: dict | None = None


def _load_centroid_configs() -> dict:
    global _centroids_cache
    if _centroids_cache is None:
        with open(_CENTROIDS_YAML, "r") as f:
            _centroids_cache = yaml.safe_load(f)
    return _centroids_cache


def _school_to_node(G: nx.Graph, centroid_schools: list[int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for node_id, attrs in G.nodes(data=True):
        for s in attrs.get("school_ids", []):
            if s in centroid_schools and s not in mapping:
                mapping[s] = node_id
    return mapping


def compute_contiguity_metrics(
    zone_dict: dict[int, int],
    G: nx.Graph,
    centroids_type: str | None,
) -> dict[str, int]:
    if not zone_dict or not centroids_type:
        return {MetricColumns.CONTIGUOUS: 0}

    configs = _load_centroid_configs()
    centroid_schools = configs.get(centroids_type)
    if not centroid_schools:
        return {MetricColumns.CONTIGUOUS: 0}

    school_to_node = _school_to_node(G, centroid_schools)

    combined: dict[int, int | None] = {n: None for n in zone_dict}
    for cs in centroid_schools:
        if cs not in school_to_node:
            continue
        sub = {n: v for n, v in zone_dict.items() if v == cs}
        if not sub:
            continue
        trimmed = trim_noncontiguity_soft(sub, G, [school_to_node[cs]])
        for n, v in trimmed.items():
            combined[n] = v

    contiguous = 0 if any(v is None for v in combined.values()) else 1
    return {MetricColumns.CONTIGUOUS: contiguous}

"""The :class:`ZoneSolution` contract.

A ``ZoneSolution`` is what every solver returns and what every strategy passes
around. It pairs a node->zone assignment with solver metadata and the
``ZoneProblem`` it solved, and knows how to validate its own contiguity, expand
to geographic-unit ids, and serialize (to ``optimization.run_archive``). Most assignments cover the full graph;
single-zone selection assignments omit nodes outside the selected zone and set
``metadata["partial_assignment"]``.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

import yaml

from optimization.data.edge_weights import edge_weight
from optimization.problem import ZoneProblem
from optimization.progress import SolverProgressEntry


def graph_fingerprint(G) -> str:
    """Stable hash of graph nodes, finest geographic units, and topology."""
    digest = hashlib.sha256()
    for node in sorted(G.nodes()):
        attrs = G.nodes[node]
        area_ids = (
            [attrs["area_id"]] if "area_id" in attrs else attrs.get("block_ids", [])
        )
        digest.update(str(int(node)).encode("utf-8"))
        digest.update(b":")
        for area_id in sorted(area_ids):
            digest.update(str(int(area_id)).encode("utf-8"))
            digest.update(b",")
        digest.update(b";")
    digest.update(b"|edges|")
    weighted = bool(G.graph.get("weight_edges", False))
    for u, v in sorted(tuple(sorted((int(u), int(v)))) for u, v in G.edges()):
        if weighted:
            digest.update(f"{u},{v},{edge_weight(G, u, v)};".encode("utf-8"))
        else:
            digest.update(f"{u},{v};".encode("utf-8"))
    return digest.hexdigest()[:16]


@dataclass
class ZoneSolution:
    """Result of solving a :class:`ZoneProblem`."""

    problem: ZoneProblem
    assignment: dict[int, int]
    status: str
    objective: Optional[float] = None
    wall_time: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    solver_progress: list[SolverProgressEntry] = field(default_factory=list)

    @property
    def level(self):
        return self.problem.level

    @property
    def feasible(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE")

    # ------------------------------------------------------------------ #
    # views
    # ------------------------------------------------------------------ #
    def area_assignment(self) -> dict[int, int]:
        """Expand the node->zone mapping to ``{area_id: zone}``.

        For base graphs each node carries an ``area_id``; for aggregated graphs
        each node carries ``block_ids`` (the finest area ids it absorbed). This
        finest-unit dict is the lingua franca for cross-level conversion.
        """
        return self._area_assignment_for(self.assignment)

    def _area_assignment_for(self, assignment: dict[int, int]) -> dict[int, int]:
        G = self.problem.G
        out: dict[int, int] = {}
        for node, zone in assignment.items():
            attrs = G.nodes[node]
            if "area_id" in attrs:
                out[attrs["area_id"]] = zone
            else:
                for block_id in attrs["block_ids"]:
                    out[block_id] = zone
        return out

    def is_contiguous(self) -> bool:
        # Imported lazily to avoid a data-layer import cycle.
        from optimization.data.contiguity import is_contiguous

        return is_contiguous(self.problem.G, self.assignment, self.problem.centroids)

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def save(
        self,
        folder: str,
        config: Optional[Mapping] = None,
        stage_name: Optional[str] = None,
    ) -> str:
        """Write this zoning as a one-stage ``zonings.json.xz`` in ``folder``.

        ``config`` is the snapshot of the config that produced this zoning. It
        is embedded in the archive and written beside it as
        ``zoning_config.yaml``, which ``optimization.run`` accepts as is, so a
        zoning copied out of its run still says how it was made. A run with
        several stages saves them together through
        :func:`optimization.run_archive.save_zonings` instead.
        """
        from optimization.run_archive import save_zonings

        name = stage_name or self.level.name
        path = save_zonings(folder, [self], [name], config=config, final_stage=name)
        if config is not None:
            save_zoning_config(config, folder)
        return path


ZONING_CONFIG_FILENAME = "zoning_config.yaml"


def save_zoning_config(config: Mapping, folder: str) -> str:
    """Write ``config`` as a YAML ``OptimizationConfig.from_yaml`` can load.

    ``unit`` is dropped because it is derived from ``levels``, not a key.
    """
    body = {k: v for k, v in config.items() if k != "unit"}
    path = os.path.join(folder, ZONING_CONFIG_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Config that produced the zoning in this folder.\n")
        f.write("# Rerun: uv run python -m optimization.run <this file>\n")
        yaml.safe_dump(_yaml_ready(body), f, sort_keys=False)
    return path


def _json_ready(value):
    """Spell infinities as ``benchmark.config.json_ready`` does, so the copy
    here matches the run manifest's and stays strict JSON."""
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def _yaml_ready(value):
    if isinstance(value, Mapping):
        return {str(k): _yaml_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_yaml_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value

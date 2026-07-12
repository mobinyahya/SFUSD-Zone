"""The :class:`ZoneSolution` contract.

A ``ZoneSolution`` is what every solver returns and what every strategy passes
around. It pairs a node->zone assignment with solver metadata and the
``ZoneProblem`` it solved, and knows how to validate its own contiguity, expand
to geographic-unit ids, and serialize. Most assignments cover the full graph;
single-zone selection assignments omit nodes outside the selected zone and set
``metadata["partial_assignment"]``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from optimization.progress import SolverProgressEntry
from optimization.problem import ZoneProblem


def graph_fingerprint(G) -> str:
    """Stable hash of graph node labels and their finest geographic units."""
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
    def save(self, folder: str) -> None:
        """Write ``zone_dict_<level>.json`` and ``solution_<level>.json``."""
        os.makedirs(folder, exist_ok=True)
        level = self.level.name

        zone_dict_path = os.path.join(folder, f"zone_dict_{level}.json")
        with open(zone_dict_path, "w") as f:
            json.dump({str(k): int(v) for k, v in self.assignment.items()}, f)

        area_path = os.path.join(folder, f"zone_dict_area_{level}.json")
        with open(area_path, "w") as f:
            json.dump({str(k): int(v) for k, v in self.area_assignment().items()}, f)

        self._save_solver_progress(folder, level)

        info = {
            "level": level,
            "graph_fingerprint": graph_fingerprint(self.problem.G),
            "status": self.status,
            "objective": self.objective,
            "wall_time": self.wall_time,
            "num_zones": self.problem.Z,
            "centroids": list(self.problem.centroids),
            "contiguous": self.is_contiguous() if self.feasible else None,
            "metadata": self.metadata,
        }
        info_path = os.path.join(folder, f"solution_{level}.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def _save_solver_progress(self, folder: str, level: str) -> None:
        if not self.solver_progress:
            if self.metadata.get("solver_progress_enabled"):
                self.metadata["solver_progress_count"] = 0
            return

        progress_id = str(
            self.metadata.get("solver_progress_id")
            or _safe_filename(f"{level}_{self.metadata.get('solver', 'solver')}")
        )
        rel_dir = os.path.join("solver_progress", progress_id)
        progress_dir = os.path.join(folder, rel_dir)
        os.makedirs(progress_dir, exist_ok=True)

        log_name = "progress.jsonl"
        log_path = os.path.join(progress_dir, log_name)
        nodes = list(self.problem.nodes)
        with open(log_path, "w", encoding="utf-8") as log_file:
            for idx, entry in enumerate(self.solver_progress):
                if len(entry.assignment) != len(nodes):
                    raise ValueError(
                        "Solver progress assignment length does not match problem nodes."
                    )
                assignment = {
                    int(node): int(zone) for node, zone in zip(nodes, entry.assignment)
                }
                zone_name = f"zone_dict_{level}_{idx:04d}.json"
                area_name = f"zone_dict_area_{level}_{idx:04d}.json"
                with open(os.path.join(progress_dir, zone_name), "w") as f:
                    json.dump({str(k): int(v) for k, v in assignment.items()}, f)
                area_assignment = self._area_assignment_for(assignment)
                with open(os.path.join(progress_dir, area_name), "w") as f:
                    json.dump({str(k): int(v) for k, v in area_assignment.items()}, f)

                row = {
                    "solution_index": idx,
                    "objective": entry.objective,
                    "elapsed_seconds": entry.elapsed_seconds,
                    "assignment_path": zone_name,
                    "area_assignment_path": area_name,
                }
                if entry.iteration is not None:
                    row["iteration"] = entry.iteration
                json.dump(row, log_file, sort_keys=True)
                log_file.write("\n")

        self.metadata.update(
            {
                "solver_progress_enabled": True,
                "solver_progress_id": progress_id,
                "solver_progress_count": len(self.solver_progress),
                "solver_progress_format": "jsonl",
                "solver_progress_path": os.path.join(rel_dir, log_name),
                "solver_progress_dir": rel_dir,
            }
        )


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)
    return safe.strip("_") or "solver_progress"

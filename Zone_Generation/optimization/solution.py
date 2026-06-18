"""The :class:`ZoneSolution` contract.

A ``ZoneSolution`` is what every solver returns and what every strategy passes
around. It pairs a node->zone assignment with solver metadata and the
``ZoneProblem`` it solved, and knows how to validate its own contiguity, expand
to geographic-unit ids, and serialize.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from Zone_Generation.pipeline.problem import ZoneProblem


@dataclass
class ZoneSolution:
    """Result of solving a :class:`ZoneProblem`."""

    problem: ZoneProblem
    assignment: dict[int, int]
    status: str
    objective: Optional[float] = None
    wall_time: Optional[float] = None
    metadata: dict = field(default_factory=dict)

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
        G = self.problem.G
        out: dict[int, int] = {}
        for node, zone in self.assignment.items():
            attrs = G.nodes[node]
            if "area_id" in attrs:
                out[attrs["area_id"]] = zone
            else:
                for block_id in attrs["block_ids"]:
                    out[block_id] = zone
        return out

    def is_contiguous(self) -> bool:
        # Imported lazily to avoid a data-layer import cycle.
        from Zone_Generation.pipeline.data.contiguity import is_contiguous

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
            json.dump(
                {str(k): int(v) for k, v in self.area_assignment().items()}, f
            )

        info = {
            "level": level,
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

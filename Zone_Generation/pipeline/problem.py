"""The :class:`ZoneProblem` contract.

A ``ZoneProblem`` is a *solver-agnostic* description of a single optimization
instance at one level. Solvers read only from this object; they never touch the
data layer or the strategy that produced them. That decoupling is what lets the
solver and strategy layers be swapped independently.

Construct these via :meth:`Dataset.problem_for`; they are not meant to be built
by hand outside the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from Zone_Generation.Config.Constants import AREA_ETHNICITIES, AALPI_ETHNICITIES
from Zone_Generation.pipeline.levels import LevelSpec


@dataclass
class ZoneProblem:
    """One optimization instance.

    Attributes
    ----------
    G:
        Node-attributed adjacency graph for ``level`` (see CLAUDE.md for the
        node/graph attribute schema). Nodes are integer indices ``0..A-1``.
    level:
        The granularity this instance lives at.
    centroids:
        Node indices anchoring each zone. ``len(centroids) == Z`` and zone ``z``
        is anchored at ``centroids[z]``.
    frl_dev, racial_dev:
        Maximum allowed deviation of a zone's FRL / per-ethnicity proportion
        from the district-wide proportion.
    overage, shortage:
        Capacity tolerance: a zone's seats must lie within
        ``[(1 - shortage), (1 + overage)]`` times its students (proportions).
    max_distance:
        Areas farther than this (miles) from a centroid are not candidates for
        that centroid's zone.
    fixed:
        Optional ``{node: zone}`` assignments forced by a strategy (e.g. a
        coarse solution projected down in recursive zoning).
    candidates:
        Optional ``{node: set[zone]}`` restricting which zones each node may
        take. When absent, candidacy is derived from ``max_distance``.
    hint:
        Optional ``{node: zone}`` warm-start passed to the solver.
    """

    G: nx.Graph
    level: LevelSpec
    centroids: list[int]

    frl_dev: float = 0.3
    racial_dev: float = 0.3
    overage: float = 0.8
    shortage: float = 0.2
    max_distance: float = float("inf")

    fixed: Optional[dict[int, int]] = None
    candidates: Optional[dict[int, set[int]]] = None
    hint: Optional[dict[int, int]] = None

    # cached candidate structures (populated lazily by `candidate_zones`)
    _candidates: Optional[dict[int, set[int]]] = field(
        default=None, repr=False, compare=False
    )

    # ------------------------------------------------------------------ #
    # basic dimensions
    # ------------------------------------------------------------------ #
    @property
    def Z(self) -> int:
        return len(self.centroids)

    @property
    def nodes(self) -> list[int]:
        return list(self.G.nodes())

    @property
    def A(self) -> int:
        return self.G.number_of_nodes()

    def neighbors(self, node: int) -> list[int]:
        return list(self.G.neighbors(node))

    # ------------------------------------------------------------------ #
    # node / graph attributes
    # ------------------------------------------------------------------ #
    def students(self, node: int) -> float:
        return float(self.G.nodes[node]["ge_students"])

    def capacity(self, node: int) -> float:
        return float(self.G.nodes[node]["ge_capacity"])

    def frl(self, node: int) -> float:
        return float(self.G.nodes[node]["FRL"])

    def num_schools(self, node: int) -> int:
        return int(self.G.nodes[node].get("num_schools", 0))

    def ethnicity(self, node: int, ethnicity: str) -> float:
        return float(self.G.nodes[node][ethnicity])

    @property
    def ethnicities(self) -> list[str]:
        return list(AREA_ETHNICITIES)

    @property
    def aalpi_ethnicities(self) -> list[str]:
        return list(AALPI_ETHNICITIES)

    @property
    def district_frl(self) -> float:
        """District-wide FRL proportion (0-1)."""
        return float(self.G.graph["F"])

    @property
    def district_racial(self) -> dict[str, float]:
        """District-wide per-ethnicity proportions (0-1)."""
        return dict(self.G.graph["R"])

    def distance(self, centroid: int, node: int) -> float:
        """Distance from ``centroid`` to ``node`` (miles)."""
        return float(self.G.graph["distance_dict"][centroid][node])

    # ------------------------------------------------------------------ #
    # candidacy
    # ------------------------------------------------------------------ #
    def candidate_zones(self, node: int) -> set[int]:
        """Set of zone indices ``node`` may be assigned to.

        Priority: centroid anchors, then an explicit per-node ``candidates``
        entry, then a forced ``fixed`` assignment, then the distance-derived
        default. Centroids are checked first so strategy-level relaxations can
        never unassign an anchor node.
        """
        centroid_zones = {
            z for z, centroid in enumerate(self.centroids) if centroid == node
        }
        if len(centroid_zones) > 1:
            raise ValueError(
                f"Node {node} is used as multiple centroids: "
                f"{sorted(centroid_zones)}."
            )
        if centroid_zones:
            return centroid_zones
        if self.candidates is not None and node in self.candidates:
            return set(self.candidates[node])
        if self.fixed is not None and node in self.fixed:
            return {self.fixed[node]}
        return self._distance_candidates()[node]

    def _distance_candidates(self) -> dict[int, set[int]]:
        if self._candidates is None:
            cand: dict[int, set[int]] = {}
            for node in self.G.nodes():
                allowed = {
                    z
                    for z, centroid in enumerate(self.centroids)
                    if self.distance(centroid, node) <= self.max_distance
                }
                # A node always remains a candidate for its own centroid's zone
                # so the instance stays feasible even with a tight max_distance.
                cand[node] = allowed
            for z, centroid in enumerate(self.centroids):
                cand[centroid] = {z}
            self._candidates = cand
        return self._candidates

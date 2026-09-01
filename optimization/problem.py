"""The :class:`ZoneProblem` contract.

A ``ZoneProblem`` is a *solver-agnostic* description of a single optimization
instance at one level. Solvers read only from this object; they never touch the
data layer or the strategy that produced them. That decoupling is what lets the
solver and strategy layers be swapped independently.

Construct these via :meth:`Dataset.problem_for`; they are not meant to be built
by hand outside the data layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import networkx as nx

from Config.Constants import AREA_ETHNICITIES, AALPI_ETHNICITIES
from choice.objective import ChoiceObjective
from optimization.data.edge_weights import edge_weight
from optimization.levels import LevelSpec

if TYPE_CHECKING:
    from optimization.config import OptimizationConfig


class DuplicateCentroidError(ValueError):
    """Raised when multiple zones resolve to the same centroid node."""

    def __init__(self, node: int, zones) -> None:
        self.node = int(node)
        self.zones = tuple(sorted(int(zone) for zone in zones))
        super().__init__(
            f"Node {self.node} is used as multiple centroids: {list(self.zones)}."
        )


class NoCandidateZonesError(ValueError):
    """Raised when a node cannot be legally assigned to any zone."""

    def __init__(self, problem, node: int) -> None:
        self.node = int(node)
        super().__init__(problem.no_candidate_zones_message(node))


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
    centroid_school_ids:
        School IDs corresponding one-to-one with ``centroids``. Geometry-based
        closer-neighbor relations are keyed by these IDs rather than by the
        centroid nodes that contain them.
    frl_dev, racial_dev:
        Maximum allowed deviation of a zone's FRL / per-ethnicity proportion
        from the district-wide proportion. A negative value disables the
        corresponding constraint.
    overage, shortage:
        Capacity tolerance: a zone's seats must lie within
        ``[(1 - shortage), (1 + overage)]`` times its students (proportions).
        A negative value disables its corresponding bound.
    max_distance:
        Areas farther than this (miles) from a centroid are not candidates for
        that centroid's zone. Can also be set to ``"auto"`` to automatically
        use 1.2x the maximum distance to the closest centroid across all
        geography units. Nodes marked ``max_distance_exempt`` are candidates
        for every zone regardless of distance.
    boundary_prop:
        Maximum proportion of graph edges whose endpoints may be assigned to
        different zones. A negative value disables the constraint.
    weight_edges:
        Whether boundary objectives use integer-metre shared-boundary weights
        instead of one unit per cut edge.
    fixed:
        Optional ``{node: zone}`` assignments forced by a strategy (e.g. a
        coarse solution projected down in recursive zoning).
    candidates:
        Optional ``{node: set[zone]}`` restricting which zones each node may
        take. When absent, candidacy is derived from ``max_distance``.
    hint:
        Optional ``{node: zone}`` warm-start passed to the solver.
    choice_objective:
        Optional choice-utility objective with accumulated linearization cuts.
    optimization_config:
        Strict originating config retained for scenario-backed downstream work.
    """

    G: nx.Graph
    level: LevelSpec
    centroids: list[int]
    centroid_school_ids: list[int]
    program_population: str = "GE"

    frl_dev: float = 0.3
    racial_dev: float = 0.3
    overage: float = 0.8
    shortage: float = 0.2
    max_distance: float | str = float("inf")
    boundary_prop: float = -1.0
    weight_edges: bool = False

    fixed: Optional[dict[int, int]] = None
    candidates: Optional[dict[int, set[int]]] = None
    hint: Optional[dict[int, int]] = None
    choice_objective: Optional[ChoiceObjective] = None
    optimization_config: Optional["OptimizationConfig"] = field(
        default=None, repr=False, compare=False
    )

    # cached candidate structures (populated lazily by `candidate_zones`)
    _candidates: Optional[dict[int, set[int]]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if len(self.centroid_school_ids) != len(self.centroids):
            raise ValueError(
                "centroid_school_ids must have one school ID per centroid node."
            )
        if not isinstance(self.weight_edges, bool):
            raise ValueError("weight_edges must be a Boolean.")
        if isinstance(self.max_distance, str) and self.max_distance.strip().lower() == "auto":
            if self.centroids and self.G.number_of_nodes() > 0:
                self.max_distance = float(
                    1.2
                    * max(
                        min(self.distance(centroid, node) for centroid in self.centroids)
                        for node in self.G.nodes()
                    )
                )
            else:
                self.max_distance = 0.0
        elif isinstance(self.max_distance, bool) or not isinstance(
            self.max_distance, (int, float)
        ):
            raise ValueError(
                "max_distance must be a non-negative float, 'inf', or 'auto'."
            )
        else:
            self.max_distance = float(self.max_distance)
            if math.isnan(self.max_distance) or self.max_distance < 0:
                raise ValueError(
                    "max_distance must be a non-negative float, 'inf', or 'auto'."
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

    def boundary_weight(self, u: int, v: int) -> int:
        return edge_weight(self.G, u, v, weighted=self.weight_edges)

    # ------------------------------------------------------------------ #
    # node / graph attributes
    # ------------------------------------------------------------------ #
    def students(self, node: int) -> float:
        return float(self.G.nodes[node][self.student_attribute])

    def capacity(self, node: int) -> float:
        return float(self.G.nodes[node][self.capacity_attribute])

    @property
    def student_attribute(self) -> str:
        return (
            "ge_students"
            if self.program_population == "GE"
            else "all_prog_students"
        )

    @property
    def capacity_attribute(self) -> str:
        return (
            "ge_capacity"
            if self.program_population == "GE"
            else "all_prog_capacity"
        )

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
            raise DuplicateCentroidError(node, centroid_zones)
        if centroid_zones:
            return centroid_zones
        if self.candidates is not None and node in self.candidates:
            return set(self.candidates[node])
        if self.fixed is not None and node in self.fixed:
            return {self.fixed[node]}
        return self._distance_candidates()[node]

    def no_candidate_zones_error(self, node: int) -> NoCandidateZonesError:
        return NoCandidateZonesError(self, node)

    def no_candidate_zones_message(self, node: int) -> str:
        attrs = self.G.nodes[node]
        area_id = attrs.get("area_id")
        label = f"Node {node}"
        if area_id is not None:
            label += f" (area_id={area_id})"

        if self.candidates is not None and node in self.candidates:
            reason = "explicit candidate restrictions leave no legal zones"
            advice = (
                "Relax the candidate restrictions or add a legal zone for this node."
            )
        else:
            reason = (
                f"max_distance={self.max_distance:g} excludes all {self.Z} centroids"
            )
            advice = "Increase max_distance or choose centroid schools that cover every node."

        nearest = self._nearest_centroid_description(node)
        nearest_text = f" {nearest}" if nearest else ""
        return (
            f"{label} has no candidate zones for {self.level.name}: "
            f"{reason}.{nearest_text} {advice}"
        )

    def _nearest_centroid_description(self, node: int) -> str:
        if not self.centroids:
            return ""
        distance, zone, centroid = min(
            (
                (self.distance(centroid, node), zone, centroid)
                for zone, centroid in enumerate(self.centroids)
            ),
            key=lambda item: item[0],
        )
        return (
            f"Nearest centroid is zone {zone} at node {centroid} "
            f"({distance:.3f} miles away)."
        )

    def _distance_candidates(self) -> dict[int, set[int]]:
        if self._candidates is None:
            cand: dict[int, set[int]] = {}
            for node in self.G.nodes():
                if self.G.nodes[node].get("max_distance_exempt", False):
                    allowed = set(range(self.Z))
                else:
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

"""Exact zone-column data contract."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import TYPE_CHECKING, TypeAlias

import networkx as nx

if TYPE_CHECKING:
    from optimization.problem import ZoneProblem

PatternKey: TypeAlias = tuple[int, frozenset[int]]


@dataclass(frozen=True, slots=True)
class ZonePattern:
    """One exact-valued column, uniquely identified by label and full node set."""

    label: int
    nodes: frozenset[int]
    raw_welfare: int = field(compare=False)
    perimeter: int = field(compare=False)

    def __post_init__(self) -> None:
        label = _as_int("label", self.label)
        nodes = frozenset(_as_int("node", node) for node in self.nodes)
        raw_welfare = _as_int("raw_welfare", self.raw_welfare)
        perimeter = _as_int("perimeter", self.perimeter)
        if label < 0:
            raise ValueError("Pattern label must be nonnegative.")
        if not nodes:
            raise ValueError("A zone pattern must contain at least one node.")
        if raw_welfare < 0:
            raise ValueError("Pattern raw_welfare must be nonnegative.")
        if perimeter < 0:
            raise ValueError("Pattern perimeter must be nonnegative.")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "raw_welfare", raw_welfare)
        object.__setattr__(self, "perimeter", perimeter)

    @property
    def key(self) -> PatternKey:
        """Stable column identity; welfare diagnostics are not part of the key."""
        return self.label, self.nodes

    @classmethod
    def from_graph(
        cls,
        *,
        label: int,
        nodes: frozenset[int] | set[int],
        raw_welfare: int,
        graph: nx.Graph,
    ) -> ZonePattern:
        full_nodes = frozenset(nodes)
        return cls(
            label=label,
            nodes=full_nodes,
            raw_welfare=raw_welfare,
            perimeter=zone_perimeter(graph, full_nodes),
        )


def zone_perimeter(graph: nx.Graph, nodes: frozenset[int] | set[int]) -> int:
    """Return the number of graph edges with exactly one endpoint in ``nodes``."""
    selected = set(nodes)
    return sum((left in selected) != (right in selected) for left, right in graph.edges)


class ZonePatternValidator:
    """Reusable exact local validator with precomputed support structures."""

    def __init__(
        self,
        problem: ZoneProblem,
        *,
        centroid_neighbor_radius: int = 0,
    ) -> None:
        from optimization.data import contiguity
        from optimization.solvers.balance import enforced_balance_constraints

        if (
            isinstance(centroid_neighbor_radius, bool)
            or not isinstance(centroid_neighbor_radius, int)
            or centroid_neighbor_radius < 0
        ):
            raise ValueError("centroid_neighbor_radius must be a non-negative integer.")
        self.problem = problem
        self.graph_nodes = set(problem.nodes)
        self.centroids = set(problem.centroids)
        self.neighborhoods = {
            label: set(
                nx.single_source_shortest_path_length(
                    problem.G,
                    centroid,
                    cutoff=centroid_neighbor_radius,
                )
            )
            for label, centroid in enumerate(problem.centroids)
        }
        self.closer = contiguity.closer_supports(
            problem.G,
            problem.centroids,
            problem.centroid_school_ids,
            problem.candidate_zones,
        )
        self.supports = {
            key: frozenset(nodes)
            for key, nodes in contiguity.contiguity_supports(
                problem.G,
                problem.centroids,
                problem.centroid_school_ids,
                problem.candidate_zones,
            ).items()
        }
        self.constraints = tuple(enforced_balance_constraints(problem))
        total_schools = sum(problem.num_schools(node) for node in problem.nodes)
        if total_schools:
            average = total_schools / problem.Z
            self.school_bounds = (
                round(100 * max(0.0, average - 1.0)),
                round(100 * (average + 1.0)),
            )
        else:
            self.school_bounds = None
        self.max_cut_edges = (
            math.floor(problem.boundary_prop * problem.G.number_of_edges())
            if problem.boundary_prop >= 0
            else None
        )

    def __call__(self, pattern: ZonePattern) -> None:
        """Validate one column against the zoning constraints encoded by CP-SAT."""
        self.validate_membership(
            label=pattern.label,
            nodes=pattern.nodes,
            perimeter=pattern.perimeter,
        )

    def validate_membership(
        self,
        *,
        label: int,
        nodes: frozenset[int] | set[int],
        perimeter: int,
    ) -> None:
        """Validate objective-independent complete-zone membership data."""
        problem = self.problem
        label = _as_int("label", label)
        nodes = frozenset(_as_int("node", node) for node in nodes)
        perimeter = _as_int("perimeter", perimeter)
        if not nodes:
            raise ValueError("A zone pattern must contain at least one node.")
        if perimeter < 0:
            raise ValueError("Pattern perimeter must be nonnegative.")
        if label not in range(problem.Z):
            raise ValueError(f"Unknown pattern label {label}.")
        if not nodes <= self.graph_nodes:
            raise ValueError("Pattern contains nodes outside the problem graph.")
        centroid = problem.centroids[label]
        if nodes & self.centroids != {centroid}:
            raise ValueError("Pattern must contain exactly its labeled centroid.")
        if any(
            label not in problem.candidate_zones(node) for node in nodes
        ):
            raise ValueError("Pattern violates node-zone candidate restrictions.")
        if not self.neighborhoods[label] <= nodes:
            raise ValueError("Pattern omits a required centroid-neighborhood node.")
        if not nx.is_connected(problem.G.subgraph(nodes)):
            raise ValueError("Pattern nodes must induce a connected subgraph.")

        for node in nodes - {centroid}:
            key = (node, label)
            if not self.closer.get(key) or not (
                self.supports.get(key, frozenset()) & nodes
            ):
                raise ValueError(
                    "Pattern violates centroid-monotone contiguity supports."
                )

        for constraint in self.constraints:
            if constraint.lower_ratio is not None:
                lower = sum(
                    round(
                        100
                        * (
                            constraint.value(node)
                            - constraint.lower_ratio * problem.students(node)
                        )
                    )
                    for node in nodes
                )
                if lower < 0:
                    raise ValueError(
                        f"Pattern violates the {constraint.kind} lower bound."
                    )
            if constraint.upper_ratio is not None:
                upper = sum(
                    round(
                        100
                        * (
                            constraint.value(node)
                            - constraint.upper_ratio * problem.students(node)
                        )
                    )
                    for node in nodes
                )
                if upper > 0:
                    raise ValueError(
                        f"Pattern violates the {constraint.kind} upper bound."
                    )

        if self.school_bounds is not None:
            school_count = 100 * sum(
                problem.num_schools(node) for node in nodes
            )
            if not self.school_bounds[0] <= school_count <= self.school_bounds[1]:
                raise ValueError("Pattern violates graph-school count bounds.")

        exact_perimeter = zone_perimeter(problem.G, nodes)
        if perimeter != exact_perimeter:
            raise ValueError(
                f"Pattern perimeter is {perimeter}, expected {exact_perimeter}."
            )
        if self.max_cut_edges is not None and perimeter > self.max_cut_edges:
            raise ValueError("Pattern exceeds the necessary per-zone perimeter bound.")


def validate_zone_pattern(
    problem: ZoneProblem,
    pattern: ZonePattern,
    *,
    centroid_neighbor_radius: int = 0,
) -> None:
    """Validate one pattern without retaining precomputed support structures."""
    ZonePatternValidator(
        problem,
        centroid_neighbor_radius=centroid_neighbor_radius,
    )(pattern)


def _as_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Pattern {name} must be an integer, got {value!r}.")
    return int(value)

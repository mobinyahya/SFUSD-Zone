"""Floating restricted LP and MIP masters for analytical zone patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from optimization.branch_price.analytical_patterns import (
    AnalyticalPatternKey,
    AnalyticalZonePattern,
)
from optimization.branch_price.patterns import zone_perimeter


@dataclass(frozen=True, slots=True)
class AnalyticalMasterDuals:
    convexity: dict[int, float]
    coverage: dict[int, float]
    boundary: float

    def pricing_value(self, pattern: AnalyticalZonePattern) -> float:
        return (
            pattern.shi_welfare
            - sum(self.coverage.get(node, 0.0) for node in pattern.nodes)
            - self.boundary * pattern.perimeter
        )

    def reduced_cost(self, pattern: AnalyticalZonePattern) -> float:
        return self.pricing_value(pattern) - self.convexity[pattern.label]


@dataclass(frozen=True, slots=True)
class AnalyticalMasterResult:
    status: str
    objective: float | None
    upper_bound: float | None
    values: dict[AnalyticalPatternKey, float]
    selected_patterns: tuple[AnalyticalZonePattern, ...]
    assignment: dict[int, int] | None
    perimeter: float | None
    duals: AnalyticalMasterDuals | None = None
    seed_fallback_used: bool = False


class RestrictedAnalyticalPatternMaster:
    """Set partitioning over complete labeled analytical zone columns."""

    def __init__(
        self,
        graph: nx.Graph,
        centroids: Sequence[int],
        patterns: Sequence[AnalyticalZonePattern],
        *,
        max_cut_edges: int | None,
        pattern_validator: Callable[[AnalyticalZonePattern], None] | None = None,
        welfare_tolerance: float = 1e-7,
    ) -> None:
        self.graph = graph
        self.nodes = tuple(map(int, graph.nodes))
        self.centroids = tuple(map(int, centroids))
        self.labels = tuple(range(len(self.centroids)))
        self.coverage_nodes = tuple(
            node for node in self.nodes if node not in set(self.centroids)
        )
        if len(set(self.centroids)) != len(self.centroids):
            raise ValueError("Analytical master centroids must be distinct.")
        if not set(self.centroids) <= set(self.nodes):
            raise ValueError("Every analytical-master centroid must be a graph node.")
        if max_cut_edges is not None:
            if isinstance(max_cut_edges, bool) or int(max_cut_edges) < 0:
                raise ValueError("max_cut_edges must be nonnegative or None.")
            max_cut_edges = int(max_cut_edges)
        self.max_cut_edges = max_cut_edges
        self.zone_perimeter_cap = None if max_cut_edges is None else 2 * max_cut_edges
        self.pattern_validator = pattern_validator
        self.welfare_tolerance = float(welfare_tolerance)
        self.patterns = self._validated_patterns(patterns)

    def solve_lp(
        self,
        *,
        feasibility_tolerance: float = 1e-8,
        optimality_tolerance: float = 1e-6,
        threads: int = 1,
        time_limit: float | None = None,
    ) -> AnalyticalMasterResult:
        model, variables, convexity, coverage, boundary = self._build_model(
            binary=False,
            feasibility_tolerance=feasibility_tolerance,
            optimality_tolerance=optimality_tolerance,
            threads=threads,
        )
        if time_limit is not None:
            model.Params.TimeLimit = max(0.0, float(time_limit))
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            return AnalyticalMasterResult(
                status=_status_name(model.Status),
                objective=None,
                upper_bound=None,
                values={},
                selected_patterns=(),
                assignment=None,
                perimeter=None,
            )
        values = {
            pattern.key: float(variables[pattern.key].X) for pattern in self.patterns
        }
        selected, assignment = self._integral_selection(
            values,
            tolerance=max(feasibility_tolerance, 1e-7),
        )
        duals = AnalyticalMasterDuals(
            convexity={label: float(row.Pi) for label, row in convexity.items()},
            coverage={node: float(row.Pi) for node, row in coverage.items()},
            boundary=float(boundary.Pi) if boundary is not None else 0.0,
        )
        if duals.boundary < -max(feasibility_tolerance, 1e-9):
            raise RuntimeError("Analytical master returned a negative boundary price.")
        dual_objective = sum(duals.convexity.values()) + sum(duals.coverage.values())
        if self.zone_perimeter_cap is not None:
            dual_objective += self.zone_perimeter_cap * duals.boundary
        if not math.isclose(
            dual_objective,
            float(model.ObjVal),
            rel_tol=max(optimality_tolerance, 1e-8),
            abs_tol=max(feasibility_tolerance, 1e-7),
        ):
            raise RuntimeError(
                "Analytical master primal and reconstructed dual objectives disagree."
            )
        perimeter = sum(
            pattern.perimeter * values[pattern.key] for pattern in self.patterns
        )
        return AnalyticalMasterResult(
            status="OPTIMAL",
            objective=float(model.ObjVal),
            upper_bound=float(model.ObjVal),
            values=values,
            selected_patterns=selected,
            assignment=assignment,
            perimeter=float(perimeter),
            duals=duals,
        )

    def solve_mip(
        self,
        *,
        time_limit: float,
        workers: int,
        random_seed: int,
        seed_assignment: Mapping[int, int] | None = None,
        feasibility_tolerance: float = 1e-8,
        optimality_tolerance: float = 1e-6,
    ) -> AnalyticalMasterResult:
        model, variables, _, _, _ = self._build_model(
            binary=True,
            feasibility_tolerance=feasibility_tolerance,
            optimality_tolerance=optimality_tolerance,
            threads=workers,
        )
        model.Params.TimeLimit = max(0.0, float(time_limit))
        model.Params.Seed = int(random_seed)
        model.Params.MIPGap = 0.0
        if seed_assignment is not None:
            seed_keys = {
                (
                    label,
                    frozenset(
                        node
                        for node, assigned in seed_assignment.items()
                        if int(assigned) == label
                    ),
                )
                for label in self.labels
            }
            for pattern in self.patterns:
                variables[pattern.key].Start = float(pattern.key in seed_keys)
        model.optimize()
        upper_bound = (
            math.nextafter(float(model.ObjBound), math.inf)
            if math.isfinite(float(model.ObjBound))
            else None
        )
        if model.SolCount > 0:
            values = {
                pattern.key: float(variables[pattern.key].X)
                for pattern in self.patterns
            }
            selected = tuple(
                pattern for pattern in self.patterns if variables[pattern.key].X > 0.5
            )
            assignment = self.reconstruct_assignment(selected)
            return AnalyticalMasterResult(
                status=_status_name(model.Status),
                objective=sum(pattern.shi_welfare for pattern in selected),
                upper_bound=upper_bound,
                values=values,
                selected_patterns=selected,
                assignment=assignment,
                perimeter=float(sum(pattern.perimeter for pattern in selected)),
            )
        if seed_assignment is None:
            return AnalyticalMasterResult(
                status=_status_name(model.Status),
                objective=None,
                upper_bound=upper_bound,
                values={},
                selected_patterns=(),
                assignment=None,
                perimeter=None,
            )
        selected = self.patterns_for_assignment(seed_assignment)
        assignment = self.reconstruct_assignment(selected)
        return AnalyticalMasterResult(
            status=f"{_status_name(model.Status)}_SEED_FALLBACK",
            objective=sum(pattern.shi_welfare for pattern in selected),
            upper_bound=upper_bound,
            values={
                pattern.key: float(pattern in selected) for pattern in self.patterns
            },
            selected_patterns=selected,
            assignment=assignment,
            perimeter=float(sum(pattern.perimeter for pattern in selected)),
            seed_fallback_used=True,
        )

    def patterns_for_assignment(
        self, assignment: Mapping[int, int]
    ) -> tuple[AnalyticalZonePattern, ...]:
        keys = {
            label: (
                label,
                frozenset(
                    node for node, zone in assignment.items() if int(zone) == label
                ),
            )
            for label in self.labels
        }
        by_key = {pattern.key: pattern for pattern in self.patterns}
        missing = [key for key in keys.values() if key not in by_key]
        if missing:
            raise ValueError("Seed assignment patterns are absent from the master.")
        return tuple(by_key[keys[label]] for label in self.labels)

    def reconstruct_assignment(
        self, selected_patterns: Sequence[AnalyticalZonePattern]
    ) -> dict[int, int]:
        by_label = {pattern.label: pattern for pattern in selected_patterns}
        if set(by_label) != set(self.labels) or len(by_label) != len(selected_patterns):
            raise ValueError("An analytical master must select one pattern per label.")
        assignment: dict[int, int] = {}
        for label in self.labels:
            pattern = by_label[label]
            if self.centroids[label] not in pattern.nodes:
                raise ValueError("Selected pattern omits its centroid.")
            for node in pattern.nodes:
                if node in assignment:
                    raise ValueError(
                        f"Selected analytical patterns overlap at node {node}."
                    )
                assignment[node] = label
        if set(assignment) != set(self.nodes):
            missing = sorted(set(self.nodes) - set(assignment))
            raise ValueError(
                f"Selected analytical patterns omit graph nodes {missing}."
            )
        perimeters = sum(pattern.perimeter for pattern in selected_patterns)
        cut_edges = sum(
            assignment[left] != assignment[right] for left, right in self.graph.edges
        )
        if perimeters != 2 * cut_edges:
            raise ValueError(
                "Analytical pattern perimeters violate the factor-two identity."
            )
        if self.zone_perimeter_cap is not None and perimeters > self.zone_perimeter_cap:
            raise ValueError("Selected analytical patterns violate the boundary cap.")
        return assignment

    def _build_model(
        self,
        *,
        binary: bool,
        feasibility_tolerance: float,
        optimality_tolerance: float,
        threads: int,
    ):
        model = gp.Model("analytical_pattern_master")
        model.Params.OutputFlag = 0
        model.Params.Threads = max(1, int(threads))
        model.Params.FeasibilityTol = max(1e-9, float(feasibility_tolerance))
        model.Params.OptimalityTol = max(1e-9, float(optimality_tolerance))
        variables = {
            pattern.key: model.addVar(
                lb=0.0,
                ub=1.0 if binary else GRB.INFINITY,
                vtype=GRB.BINARY if binary else GRB.CONTINUOUS,
                obj=pattern.shi_welfare,
                name=f"pattern_{index}",
            )
            for index, pattern in enumerate(self.patterns)
        }
        convexity = {
            label: model.addConstr(
                gp.quicksum(
                    variables[pattern.key]
                    for pattern in self.patterns
                    if pattern.label == label
                )
                == 1.0,
                name=f"convexity_{label}",
            )
            for label in self.labels
        }
        coverage = {
            node: model.addConstr(
                gp.quicksum(
                    variables[pattern.key]
                    for pattern in self.patterns
                    if node in pattern.nodes
                )
                == 1.0,
                name=f"coverage_{node}",
            )
            for node in self.coverage_nodes
        }
        boundary = None
        if self.zone_perimeter_cap is not None:
            boundary = model.addConstr(
                gp.quicksum(
                    pattern.perimeter * variables[pattern.key]
                    for pattern in self.patterns
                )
                <= self.zone_perimeter_cap,
                name="perimeter",
            )
        model.ModelSense = GRB.MAXIMIZE
        model.update()
        return model, variables, convexity, coverage, boundary

    def _validated_patterns(
        self, patterns: Sequence[AnalyticalZonePattern]
    ) -> tuple[AnalyticalZonePattern, ...]:
        graph_nodes = set(self.nodes)
        centroids = set(self.centroids)
        by_key: dict[AnalyticalPatternKey, AnalyticalZonePattern] = {}
        for pattern in patterns:
            if pattern.label not in self.labels:
                raise ValueError(f"Unknown analytical pattern label {pattern.label}.")
            if not pattern.nodes <= graph_nodes:
                raise ValueError("Analytical pattern contains nodes outside the graph.")
            own_centroid = self.centroids[pattern.label]
            if pattern.nodes & centroids != {own_centroid}:
                raise ValueError("Analytical pattern has invalid centroid membership.")
            if pattern.perimeter != zone_perimeter(self.graph, pattern.nodes):
                raise ValueError("Analytical pattern has an incorrect perimeter.")
            if not nx.is_connected(self.graph.subgraph(pattern.nodes)):
                raise ValueError("Analytical pattern nodes must be connected.")
            if self.pattern_validator is not None:
                self.pattern_validator(pattern)
            previous = by_key.get(pattern.key)
            if previous is not None:
                if previous.perimeter != pattern.perimeter or not math.isclose(
                    previous.shi_welfare,
                    pattern.shi_welfare,
                    rel_tol=self.welfare_tolerance,
                    abs_tol=self.welfare_tolerance,
                ):
                    raise ValueError(
                        "Duplicate analytical pattern key has conflicting closed data."
                    )
                continue
            by_key[pattern.key] = pattern
        for label in self.labels:
            if not any(pattern.label == label for pattern in by_key.values()):
                raise ValueError(f"Analytical master has no pattern for label {label}.")
        return tuple(by_key.values())

    def _integral_selection(
        self,
        values: Mapping[AnalyticalPatternKey, float],
        *,
        tolerance: float,
    ) -> tuple[tuple[AnalyticalZonePattern, ...], dict[int, int] | None]:
        if any(tolerance < value < 1.0 - tolerance for value in values.values()):
            return (), None
        selected = tuple(
            pattern
            for pattern in self.patterns
            if values[pattern.key] >= 1.0 - tolerance
        )
        try:
            assignment = self.reconstruct_assignment(selected)
        except ValueError:
            return (), None
        return selected, assignment


def _status_name(status: int) -> str:
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
    }.get(status, f"STATUS_{status}")

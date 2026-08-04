"""Restricted LP and integer set-partitioning masters for zone patterns."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, Mapping, Sequence

import networkx as nx
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from optimization.branch_price.patterns import PatternKey, ZonePattern, zone_perimeter
from optimization.welfare_oracle import MAX_EXACT_CP_SAT_OBJECTIVE


@dataclass(frozen=True, slots=True)
class PatternMasterDuals:
    """LP row multipliers in the maximization-master sign convention."""

    convexity: dict[int, float]
    coverage: dict[int, float]
    boundary: float

    def reduced_cost(self, pattern: ZonePattern) -> float:
        return (
            pattern.raw_welfare
            - self.convexity[pattern.label]
            - sum(self.coverage.get(node, 0.0) for node in pattern.nodes)
            - self.boundary * pattern.perimeter
        )


@dataclass(frozen=True, slots=True)
class PatternMasterResult:
    """A restricted-master solve, including an assignment when it is integral."""

    status: str
    objective: float | int | None
    values: dict[PatternKey, float]
    selected_patterns: tuple[ZonePattern, ...]
    assignment: dict[int, int] | None
    perimeter: float | int | None
    duals: PatternMasterDuals | None = None


class RestrictedPatternMaster:
    """Set partitioning over complete labeled zones.

    Convexity rows cover every label. Coverage rows deliberately omit centroid
    nodes because each centroid row is identical to its label's convexity row.
    """

    def __init__(
        self,
        graph: nx.Graph,
        centroids: Sequence[int],
        patterns: Sequence[ZonePattern],
        *,
        max_cut_edges: int,
        pattern_validator: Callable[[ZonePattern], None] | None = None,
    ) -> None:
        self.graph = graph
        self.nodes = tuple(int(node) for node in graph.nodes)
        self.centroids = tuple(int(node) for node in centroids)
        self.labels = tuple(range(len(self.centroids)))
        self.max_cut_edges = _nonnegative_int("max_cut_edges", max_cut_edges)
        self.zone_perimeter_cap = 2 * self.max_cut_edges
        self.pattern_validator = pattern_validator
        if len(set(self.centroids)) != len(self.centroids):
            raise ValueError("Pattern master centroids must be distinct.")
        if not set(self.centroids) <= set(self.nodes):
            raise ValueError("Every pattern-master centroid must be a graph node.")
        self.coverage_nodes = tuple(
            node for node in self.nodes if node not in set(self.centroids)
        )
        self.patterns = self._validated_patterns(patterns)

    @property
    def convexity_row_count(self) -> int:
        return len(self.labels)

    @property
    def coverage_row_count(self) -> int:
        return len(self.coverage_nodes)

    def solve_lp(self) -> PatternMasterResult:
        """Solve the continuous restricted master and return rank-correct duals."""
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:  # pragma: no cover - packaged OR-Tools includes GLOP
            raise RuntimeError("OR-Tools GLOP is unavailable.")
        variables = {
            pattern.key: solver.NumVar(0.0, solver.infinity(), f"pattern_{index}")
            for index, pattern in enumerate(self.patterns)
        }
        convexity_rows = {
            label: solver.Constraint(1.0, 1.0, f"convexity_{label}")
            for label in self.labels
        }
        coverage_rows = {
            node: solver.Constraint(1.0, 1.0, f"coverage_{node}")
            for node in self.coverage_nodes
        }
        boundary_row = solver.Constraint(
            -solver.infinity(), float(self.zone_perimeter_cap), "perimeter"
        )
        objective = solver.Objective()
        objective.SetMaximization()
        for pattern in self.patterns:
            variable = variables[pattern.key]
            convexity_rows[pattern.label].SetCoefficient(variable, 1.0)
            for node in pattern.nodes:
                row = coverage_rows.get(node)
                if row is not None:
                    row.SetCoefficient(variable, 1.0)
            boundary_row.SetCoefficient(variable, float(pattern.perimeter))
            objective.SetCoefficient(variable, float(pattern.raw_welfare))

        status = solver.Solve()
        status_name = _linear_status_name(status)
        if status != pywraplp.Solver.OPTIMAL:
            return PatternMasterResult(
                status=status_name,
                objective=None,
                values={},
                selected_patterns=(),
                assignment=None,
                perimeter=None,
            )
        values = {
            pattern.key: variables[pattern.key].solution_value()
            for pattern in self.patterns
        }
        selected, assignment = self._integral_selection(values)
        perimeter = sum(
            pattern.perimeter * values[pattern.key] for pattern in self.patterns
        )
        duals = PatternMasterDuals(
            convexity={
                label: row.dual_value() for label, row in convexity_rows.items()
            },
            coverage={node: row.dual_value() for node, row in coverage_rows.items()},
            boundary=boundary_row.dual_value(),
        )
        return PatternMasterResult(
            status=status_name,
            objective=objective.Value(),
            values=values,
            selected_patterns=selected,
            assignment=assignment,
            perimeter=perimeter,
            duals=duals,
        )

    def solve_mip(
        self,
        *,
        time_limit: float = 60.0,
        workers: int = 1,
        random_seed: int = 0,
    ) -> PatternMasterResult:
        """Solve the integer restricted master and reconstruct every node label."""
        model = cp_model.CpModel()
        variables = {
            pattern.key: model.NewBoolVar(f"pattern_{index}")
            for index, pattern in enumerate(self.patterns)
        }
        for label in self.labels:
            model.AddExactlyOne(
                variables[pattern.key]
                for pattern in self.patterns
                if pattern.label == label
            )
        for node in self.coverage_nodes:
            model.AddExactlyOne(
                variables[pattern.key]
                for pattern in self.patterns
                if node in pattern.nodes
            )
        model.Add(
            sum(pattern.perimeter * variables[pattern.key] for pattern in self.patterns)
            <= self.zone_perimeter_cap
        )
        model.Maximize(
            sum(
                pattern.raw_welfare * variables[pattern.key]
                for pattern in self.patterns
            )
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.num_search_workers = int(workers)
        solver.parameters.random_seed = int(random_seed)
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return PatternMasterResult(
                status=status_name,
                objective=None,
                values={},
                selected_patterns=(),
                assignment=None,
                perimeter=None,
            )
        selected = tuple(
            pattern
            for pattern in self.patterns
            if solver.Value(variables[pattern.key]) == 1
        )
        assignment = self.reconstruct_assignment(selected)
        values = {
            pattern.key: float(solver.Value(variables[pattern.key]))
            for pattern in self.patterns
        }
        return PatternMasterResult(
            status=status_name,
            objective=sum(pattern.raw_welfare for pattern in selected),
            values=values,
            selected_patterns=selected,
            assignment=assignment,
            perimeter=sum(pattern.perimeter for pattern in selected),
        )

    def reconstruct_assignment(
        self, selected_patterns: Sequence[ZonePattern]
    ) -> dict[int, int]:
        """Reconstruct and validate the complete original-node assignment."""
        by_label = {pattern.label: pattern for pattern in selected_patterns}
        if set(by_label) != set(self.labels) or len(by_label) != len(selected_patterns):
            raise ValueError(
                "An integer master solution must select one pattern per label."
            )
        assignment: dict[int, int] = {}
        for label in self.labels:
            for node in by_label[label].nodes:
                if node in assignment:
                    raise ValueError(f"Selected patterns overlap at node {node}.")
                assignment[node] = label
        if set(assignment) != set(self.nodes):
            missing = sorted(set(self.nodes) - set(assignment))
            raise ValueError(f"Selected patterns do not cover graph nodes {missing}.")
        if (
            sum(pattern.perimeter for pattern in selected_patterns)
            > self.zone_perimeter_cap
        ):
            raise ValueError("Selected patterns violate the perimeter cap.")
        cut_edges = sum(
            assignment[left] != assignment[right] for left, right in self.graph.edges
        )
        if sum(pattern.perimeter for pattern in selected_patterns) != 2 * cut_edges:
            raise ValueError(
                "Selected pattern perimeters do not match graph cut edges."
            )
        return assignment

    def _validated_patterns(
        self, patterns: Sequence[ZonePattern]
    ) -> tuple[ZonePattern, ...]:
        graph_nodes = set(self.nodes)
        centroid_set = set(self.centroids)
        by_key: dict[PatternKey, ZonePattern] = {}
        for pattern in patterns:
            if pattern.label not in self.labels:
                raise ValueError(f"Unknown pattern label {pattern.label}.")
            if not pattern.nodes <= graph_nodes:
                raise ValueError("Pattern contains nodes outside the master graph.")
            own_centroid = self.centroids[pattern.label]
            if own_centroid not in pattern.nodes:
                raise ValueError("Pattern does not contain its labeled centroid.")
            if (pattern.nodes & centroid_set) != {own_centroid}:
                raise ValueError("Pattern contains another label's centroid.")
            exact_perimeter = zone_perimeter(self.graph, pattern.nodes)
            if pattern.perimeter != exact_perimeter:
                raise ValueError(
                    f"Pattern {pattern.key} perimeter is {pattern.perimeter}, "
                    f"expected {exact_perimeter}."
                )
            if not nx.is_connected(self.graph.subgraph(pattern.nodes)):
                raise ValueError("Pattern nodes must induce a connected subgraph.")
            if self.pattern_validator is not None:
                self.pattern_validator(pattern)
            previous = by_key.get(pattern.key)
            if previous is not None and (
                previous.raw_welfare != pattern.raw_welfare
                or previous.perimeter != pattern.perimeter
            ):
                raise ValueError("Duplicate pattern key has conflicting exact data.")
            by_key[pattern.key] = pattern
        for label in self.labels:
            if not any(pattern.label == label for pattern in by_key.values()):
                raise ValueError(f"Restricted master has no pattern for label {label}.")
        objective_bound = sum(
            max(
                pattern.raw_welfare
                for pattern in by_key.values()
                if pattern.label == label
            )
            for label in self.labels
        )
        if objective_bound > MAX_EXACT_CP_SAT_OBJECTIVE:
            raise ValueError("Pattern-master objective exceeds exact reporting range.")
        return tuple(by_key.values())

    def _integral_selection(
        self, values: Mapping[PatternKey, float], *, tolerance: float = 1e-7
    ) -> tuple[tuple[ZonePattern, ...], dict[int, int] | None]:
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


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _linear_status_name(status: int) -> str:
    return {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status, f"STATUS_{status}")

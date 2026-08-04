"""Low-treewidth max-plus bounds for one labeled zone's local geography."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from optimization.data import contiguity
from optimization.problem import ZoneProblem


@dataclass(frozen=True)
class LocalGeographyResult:
    """Exact optimum over local support/fix factors for supplied node weights."""

    objective: float
    selected_nodes: frozenset[int]
    perimeter: int
    elimination_width: int
    max_table_entries: int
    generated_table_entries: int
    solve_seconds: float


@dataclass(frozen=True)
class _Factor:
    scope: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class _Backpointer:
    variable: int
    remaining_scope: tuple[int, ...]
    choices: np.ndarray


class LocalGeographyDP:
    """Compile and repeatedly solve a one-label binary factor graph."""

    def __init__(self, problem: ZoneProblem, label: int) -> None:
        if label not in range(problem.Z):
            raise ValueError(f"Unknown zone label {label}.")
        self.problem = problem
        self.label = label
        self.nodes = tuple(sorted(problem.nodes))
        self.node_set = set(self.nodes)
        supports = contiguity.contiguity_supports(
            problem.G,
            problem.centroids,
            problem.centroid_school_ids,
            problem.candidate_zones,
        )
        self.supports = {
            node: tuple(sorted(supports.get((node, label), ())))
            for node in self.nodes
            if node != problem.centroids[label]
        }
        scopes = [tuple(sorted(edge)) for edge in problem.G.edges]
        scopes.extend(
            tuple(sorted((node, *parents)))
            for node, parents in self.supports.items()
            if parents
        )
        self.elimination_order, self.elimination_width = _min_fill_order(
            self.nodes, scopes
        )

    def solve(
        self,
        node_weights: Mapping[int, float],
        *,
        perimeter_price: float = 0.0,
        fixes: Mapping[int, int] | None = None,
    ) -> LocalGeographyResult:
        """Maximize node weight minus priced perimeter over local legal sets."""
        started = time.monotonic()
        if not math.isfinite(perimeter_price) or perimeter_price < 0:
            raise ValueError("perimeter_price must be finite and non-negative.")
        unknown_weights = set(node_weights) - self.node_set
        if unknown_weights:
            raise ValueError(f"Unknown weighted nodes: {sorted(unknown_weights)}.")
        fixes = {int(node): int(value) for node, value in (fixes or {}).items()}
        if set(fixes) - self.node_set:
            raise ValueError("fixes contains an unknown node.")
        if any(value not in {0, 1} for value in fixes.values()):
            raise ValueError("fix values must be zero or one.")

        required_fixes = dict(fixes)
        required_fixes[self.problem.centroids[self.label]] = 1
        for other_label, centroid in enumerate(self.problem.centroids):
            if other_label != self.label:
                required_fixes[centroid] = 0
        for node in self.nodes:
            if self.label not in self.problem.candidate_zones(node):
                required_fixes[node] = 0
            elif (
                node != self.problem.centroids[self.label]
                and not self.supports.get(node)
            ):
                required_fixes[node] = 0
        for node, value in fixes.items():
            if required_fixes.get(node, value) != value:
                raise ValueError(f"fix for node {node} conflicts with local geography.")

        factors = []
        for node in self.nodes:
            weight = float(node_weights.get(node, 0.0))
            if not math.isfinite(weight):
                raise ValueError("node weights must be finite.")
            values = np.asarray([0.0, weight])
            if node in required_fixes:
                values[1 - required_fixes[node]] = -math.inf
            factors.append(_Factor((node,), values))
        edge_values = np.asarray(
            [[0.0, -perimeter_price], [-perimeter_price, 0.0]]
        )
        factors.extend(
            _Factor(tuple(sorted((left, right))), edge_values.copy())
            for left, right in self.problem.G.edges
        )
        for node, parents in self.supports.items():
            if not parents:
                continue
            scope = tuple(sorted((node, *parents)))
            values = np.zeros((2,) * len(scope), dtype=float)
            invalid = [slice(None)] * len(scope)
            invalid[scope.index(node)] = 1
            for parent in parents:
                invalid[scope.index(parent)] = 0
            values[tuple(invalid)] = -math.inf
            factors.append(_Factor(scope, values))

        active = list(factors)
        backpointers = []
        max_entries = 0
        generated_entries = 0
        for variable in self.elimination_order:
            selected = [factor for factor in active if variable in factor.scope]
            active = [factor for factor in active if variable not in factor.scope]
            union = tuple(sorted({item for factor in selected for item in factor.scope}))
            shape = (2,) * len(union)
            joint = np.zeros(shape, dtype=float)
            for factor in selected:
                aligned_shape = [1] * len(union)
                for item in factor.scope:
                    aligned_shape[union.index(item)] = 2
                joint += factor.values.reshape(aligned_shape)
            axis = union.index(variable)
            choices = np.argmax(joint, axis=axis).astype(np.int8)
            reduced = np.max(joint, axis=axis)
            remaining = tuple(item for item in union if item != variable)
            backpointers.append(_Backpointer(variable, remaining, choices))
            entries = int(joint.size)
            max_entries = max(max_entries, entries)
            generated_entries += entries
            active.append(_Factor(remaining, reduced))

        objective = sum(float(factor.values) for factor in active)
        assignment = {}
        for pointer in reversed(backpointers):
            index = tuple(assignment[item] for item in pointer.remaining_scope)
            assignment[pointer.variable] = int(pointer.choices[index])
        selected_nodes = frozenset(
            node for node, selected_value in assignment.items() if selected_value
        )
        perimeter = sum(
            (left in selected_nodes) != (right in selected_nodes)
            for left, right in self.problem.G.edges
        )
        return LocalGeographyResult(
            objective=objective,
            selected_nodes=selected_nodes,
            perimeter=perimeter,
            elimination_width=self.elimination_width,
            max_table_entries=max_entries,
            generated_table_entries=generated_entries,
            solve_seconds=time.monotonic() - started,
        )


def _min_fill_order(
    nodes: tuple[int, ...], scopes: list[tuple[int, ...]]
) -> tuple[tuple[int, ...], int]:
    adjacency = {node: set() for node in nodes}
    for scope in scopes:
        for index, left in enumerate(scope):
            adjacency[left].update(scope[:index])
            adjacency[left].update(scope[index + 1 :])
    order = []
    width = 0
    while adjacency:
        best = None
        for node, neighbors in adjacency.items():
            neighbor_list = tuple(neighbors)
            missing = sum(
                right not in adjacency[left]
                for index, left in enumerate(neighbor_list)
                for right in neighbor_list[index + 1 :]
            )
            score = (missing, len(neighbors), node)
            if best is None or score < best[0]:
                best = (score, node)
        node = best[1]
        neighbors = tuple(adjacency[node])
        width = max(width, len(neighbors))
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        for neighbor in neighbors:
            adjacency[neighbor].discard(node)
        del adjacency[node]
        order.append(node)
    return tuple(order), width

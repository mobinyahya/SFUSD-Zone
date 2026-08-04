"""One-zone geographic constraints shared by pricing formulations."""

from __future__ import annotations

import math

import networkx as nx
from ortools.sat.python import cp_model

from optimization.data import contiguity
from optimization.problem import ZoneProblem
from optimization.solvers.balance import balance_constraints


def add_zone_pattern_constraints(
    model: cp_model.CpModel,
    problem: ZoneProblem,
    label: int,
    selected: dict[int, cp_model.IntVar],
    *,
    centroid_neighbor_radius: int = 0,
) -> cp_model.LinearExpr:
    """Constrain ``selected`` to one locally legal labeled zone."""
    for node in problem.nodes:
        if label not in problem.candidate_zones(node):
            model.Add(selected[node] == 0)
    centroid = problem.centroids[label]
    model.Add(selected[centroid] == 1)
    for other_label, other_centroid in enumerate(problem.centroids):
        if other_label != label:
            model.Add(selected[other_centroid] == 0)
    neighborhood = nx.single_source_shortest_path_length(
        problem.G, centroid, cutoff=centroid_neighbor_radius
    )
    for node in neighborhood:
        model.Add(selected[node] == 1)

    closer = contiguity.closer_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    supports = contiguity.contiguity_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    for node in problem.nodes:
        if node == centroid or label not in problem.candidate_zones(node):
            continue
        support_nodes = supports[(node, label)]
        if not closer[(node, label)] or not support_nodes:
            model.Add(selected[node] == 0)
        else:
            model.Add(selected[node] <= sum(selected[other] for other in support_nodes))

    for constraint in balance_constraints(problem):
        if problem.cutoff_market is not None and constraint.kind == "capacity":
            continue
        if constraint.lower_ratio is not None:
            model.Add(
                sum(
                    round(
                        100
                        * (
                            constraint.value(node)
                            - constraint.lower_ratio * problem.students(node)
                        )
                    )
                    * selected[node]
                    for node in problem.nodes
                )
                >= 0
            )
        if constraint.upper_ratio is not None:
            model.Add(
                sum(
                    round(
                        100
                        * (
                            constraint.value(node)
                            - constraint.upper_ratio * problem.students(node)
                        )
                    )
                    * selected[node]
                    for node in problem.nodes
                )
                <= 0
            )

    total_schools = sum(problem.num_schools(node) for node in problem.nodes)
    if total_schools:
        average = total_schools / problem.Z
        school_count = sum(
            100 * problem.num_schools(node) * selected[node] for node in problem.nodes
        )
        model.Add(school_count >= round(100 * max(0.0, average - 1.0)))
        model.Add(school_count <= round(100 * (average + 1.0)))

    boundary_variables = []
    for left, right in problem.G.edges:
        boundary = model.NewBoolVar(f"perimeter_{label}_{left}_{right}")
        model.Add(selected[left] != selected[right]).OnlyEnforceIf(boundary)
        model.Add(selected[left] == selected[right]).OnlyEnforceIf(boundary.Not())
        boundary_variables.append(boundary)
    perimeter = sum(boundary_variables)
    if problem.boundary_prop >= 0:
        model.Add(
            perimeter <= math.floor(problem.boundary_prop * problem.G.number_of_edges())
        )
    return perimeter

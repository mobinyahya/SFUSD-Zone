"""Tests for the one-label max-plus geographic dynamic program."""

from __future__ import annotations

import itertools

import pytest

from optimization.analytical_geography_dp import LocalGeographyDP
from optimization.data import contiguity
from optimization.tests.synthetic import make_grid_problem


def test_local_geography_dp_matches_brute_force_support_objective():
    problem = make_grid_problem(
        2,
        3,
        population_type="All",
        frl_dev=-1,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
    )
    weights = {node: (node - 2.0) / 3.0 for node in problem.nodes}
    price = 0.7
    compiled = LocalGeographyDP(problem, 0)
    result = compiled.solve(weights, perimeter_price=price)
    supports = contiguity.contiguity_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    values = []
    for bits in itertools.product((0, 1), repeat=problem.A):
        selected = {node for node, value in enumerate(bits) if value}
        if problem.centroids[0] not in selected or problem.centroids[1] in selected:
            continue
        if any(
            node != problem.centroids[0]
            and not (set(supports[node, 0]) & selected)
            for node in selected
        ):
            continue
        perimeter = sum(
            (left in selected) != (right in selected)
            for left, right in problem.G.edges
        )
        values.append((sum(weights[node] for node in selected) - price * perimeter, selected))

    expected, expected_nodes = max(values, key=lambda item: item[0])
    assert result.objective == pytest.approx(expected)
    assert result.selected_nodes == expected_nodes
    assert result.perimeter == sum(
        (left in expected_nodes) != (right in expected_nodes)
        for left, right in problem.G.edges
    )


def test_local_geography_dp_honors_compatible_fixes():
    problem = make_grid_problem(2, 3)
    compiled = LocalGeographyDP(problem, 0)

    result = compiled.solve({node: 1.0 for node in problem.nodes}, fixes={1: 0, 3: 1})

    assert 1 not in result.selected_nodes
    assert 3 in result.selected_nodes

"""Tests for fixed-cutoff Lagrangian geographic resource bounds."""

from __future__ import annotations

import itertools

import pytest

from optimization.analytical_resource_bound import FixedCutoffResourceBounder
from optimization.analytical_welfare_oracle import AnalyticalNodeValues
from optimization.tests.synthetic import make_grid_problem


def test_resource_bound_dominates_brute_force_feasible_value():
    problem = make_grid_problem(
        2,
        3,
        population_type="All",
        frl_dev=1.0,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=1.0,
    )
    welfare = {node: float(node + 1) for node in problem.nodes}
    demands = {node: {100: 0.4 + node / 10} for node in problem.nodes}
    bounder = FixedCutoffResourceBounder(
        problem,
        0,
        AnalyticalNodeValues(welfare, demands),
        {100: 2},
        frozenset({0}),
    )
    result = bounder.solve(max_rounds=100)

    feasible_values = []
    for bits in itertools.product((0, 1), repeat=problem.A):
        selected = {node for node, value in enumerate(bits) if value}
        if 0 not in selected or problem.centroids[1] in selected:
            continue
        if any(problem.num_schools(node) and node != 0 for node in selected):
            continue
        if sum(demands[node][100] for node in selected) > 2 + 1e-9:
            continue
        try:
            local = bounder.dp.solve(
                {node: 0.0 for node in problem.nodes},
                fixes={node: int(node in selected) for node in problem.nodes},
            )
        except ValueError:
            continue
        if local.selected_nodes != selected:
            continue
        feasible_values.append(sum(welfare[node] for node in selected))

    assert feasible_values
    assert result.upper_bound + 1e-7 >= max(feasible_values)
    assert result.max_separation_violation <= 1e-8


def test_exact_resource_mip_matches_brute_force_feasible_value():
    pytest.importorskip("gurobipy")
    problem = make_grid_problem(
        2,
        3,
        population_type="All",
        frl_dev=1.0,
        racial_dev=-1,
        overage=-1,
        shortage=-1,
        boundary_prop=1.0,
    )
    welfare = {node: float(node + 1) for node in problem.nodes}
    demands = {node: {100: 0.4 + node / 10} for node in problem.nodes}
    bounder = FixedCutoffResourceBounder(
        problem,
        0,
        AnalyticalNodeValues(welfare, demands),
        {100: 2},
        frozenset({0}),
    )

    result = bounder.solve_exact_geography(time_limit=10)
    lagrangian = bounder.solve(max_rounds=100)

    assert result.status == "OPTIMAL_FLOATING"
    assert lagrangian.upper_bound + 1e-7 >= result.objective

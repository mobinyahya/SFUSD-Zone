"""Restricted analytical complete-zone column generation experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import gurobipy as gp
import numpy as np
from scipy.optimize import linprog

from optimization.analytical_resource_bound import FixedCutoffResourceBounder
from optimization.analytical_welfare_oracle import (
    ZonedAnalyticalWelfareResult,
    aggregate_analytical_node_values,
    evaluate_zoned_analytical_welfare,
)
from optimization.branch_price.patterns import zone_perimeter
from optimization.problem import AnalyticalWelfareMarket, ZoneProblem


@dataclass(frozen=True)
class AnalyticalZoneColumn:
    label: int
    nodes: frozenset[int]
    fixed_state_welfare: float
    perimeter: int


@dataclass(frozen=True)
class RestrictedColumnGenerationResult:
    columns: tuple[AnalyticalZoneColumn, ...]
    selected_columns: tuple[AnalyticalZoneColumn, ...]
    assignment: dict[int, int]
    q20_result: ZonedAnalyticalWelfareResult
    lp_objective: float
    integer_fixed_state_objective: float
    rounds: int
    pricing_calls: int
    max_reduced_cost: float
    closed: bool
    timing_seconds: float


def run_fixed_state_column_generation(
    problem: ZoneProblem,
    market: AnalyticalWelfareMarket,
    initial_assignment: Mapping[int, int],
    *,
    max_rounds: int = 100,
    pricing_time_limit: float = 30.0,
    reduced_cost_tolerance: float = 1e-7,
    columns_per_price: int = 10,
) -> RestrictedColumnGenerationResult:
    """Close a column universe restricted to the initial six bundle/cutoff states."""
    started = time.monotonic()
    if columns_per_price <= 0:
        raise ValueError("columns_per_price must be positive.")
    if set(initial_assignment) != set(problem.nodes):
        raise ValueError("initial_assignment must contain every problem node.")
    initial = evaluate_zoned_analytical_welfare(
        market,
        initial_assignment,
        num_zones=problem.Z,
        cutoff_grid=market.lottery_scale,
    )
    states = {}
    columns = []
    seen = set()
    for label in range(problem.Z):
        schools = tuple(
            school
            for school, node in market.school_nodes.items()
            if int(initial_assignment[node]) == label
        )
        cutoffs = {school: initial.school_cutoffs[school] for school in schools}
        node_values = aggregate_analytical_node_values(
            market.segments, schools, cutoffs, beta=market.beta
        )
        graph_school_nodes = frozenset(
            node
            for node in problem.nodes
            if problem.num_schools(node) > 0
            and int(initial_assignment[node]) == label
        )
        bounder = FixedCutoffResourceBounder(
            problem,
            label,
            node_values,
            {school: market.school_capacities[school] for school in schools},
            graph_school_nodes,
        )
        states[label] = (bounder, node_values)
        nodes = frozenset(
            node for node, zone in initial_assignment.items() if int(zone) == label
        )
        value = sum(node_values.welfare.get(node, 0.0) for node in nodes)
        if not math.isclose(
            value, initial.zones[label].normalized_welfare, abs_tol=1e-7
        ):
            raise RuntimeError("Node aggregation does not reconstruct initial Q20 value.")
        column = AnalyticalZoneColumn(
            label, nodes, value, zone_perimeter(problem.G, nodes)
        )
        columns.append(column)
        seen.add((label, nodes))

    pricing_calls = 0
    max_reduced_cost = math.inf
    lp_objective = initial.normalized_welfare
    closed = False
    for round_number in range(1, max_rounds + 1):
        lp = _solve_restricted_lp(problem, columns)
        lp_objective = -float(lp.fun)
        equality_duals = lp.eqlin.marginals
        perimeter_dual = float(lp.ineqlin.marginals[0])
        convexity_duals = equality_duals[: problem.Z]
        coverage_duals = equality_duals[problem.Z :]
        noncentroids = tuple(
            node for node in problem.nodes if node not in set(problem.centroids)
        )
        node_prices = {
            node: -float(coverage_duals[index])
            for index, node in enumerate(noncentroids)
        }
        perimeter_price = -perimeter_dual
        additions = []
        max_reduced_cost = -math.inf
        all_prices_optimal = True
        for label in range(problem.Z):
            bounder, node_values = states[label]
            bounder.node_prices = node_prices
            bounder.perimeter_price = perimeter_price
            priced_pool = bounder.solve_exact_geography_pool(
                time_limit=pricing_time_limit,
                pool_solutions=columns_per_price,
            )
            pricing_calls += 1
            all_prices_optimal &= priced_pool[0].status == "OPTIMAL_FLOATING"
            sigma = -float(convexity_duals[label])
            reduced_cost = priced_pool[0].objective - sigma
            max_reduced_cost = max(max_reduced_cost, reduced_cost)
            for priced in priced_pool:
                pool_reduced_cost = priced.objective - sigma
                key = (label, priced.selected_nodes)
                if pool_reduced_cost <= reduced_cost_tolerance or key in seen:
                    continue
                value = sum(
                    node_values.welfare.get(node, 0.0)
                    for node in priced.selected_nodes
                )
                additions.append(
                    AnalyticalZoneColumn(
                        label,
                        priced.selected_nodes,
                        value,
                        zone_perimeter(problem.G, priced.selected_nodes),
                    )
                )
                seen.add(key)
        if not additions:
            closed = (
                all_prices_optimal and max_reduced_cost <= reduced_cost_tolerance
            )
            break
        columns.extend(additions)

    selected_columns, integer_objective = _solve_restricted_integer_master(
        problem, columns
    )
    assignment = {
        node: column.label for column in selected_columns for node in column.nodes
    }
    if set(assignment) != set(problem.nodes):
        raise RuntimeError("Restricted integer master did not produce a partition.")
    q20 = evaluate_zoned_analytical_welfare(
        market,
        assignment,
        num_zones=problem.Z,
        cutoff_grid=market.lottery_scale,
    )
    return RestrictedColumnGenerationResult(
        columns=tuple(columns),
        selected_columns=selected_columns,
        assignment=assignment,
        q20_result=q20,
        lp_objective=lp_objective,
        integer_fixed_state_objective=integer_objective,
        rounds=round_number,
        pricing_calls=pricing_calls,
        max_reduced_cost=max_reduced_cost,
        closed=closed,
        timing_seconds=time.monotonic() - started,
    )


def _solve_restricted_lp(problem: ZoneProblem, columns: list[AnalyticalZoneColumn]):
    noncentroids = tuple(
        node for node in problem.nodes if node not in set(problem.centroids)
    )
    equalities = np.zeros((problem.Z + len(noncentroids), len(columns)))
    for column_index, column in enumerate(columns):
        equalities[column.label, column_index] = 1.0
        for node_index, node in enumerate(noncentroids):
            if node in column.nodes:
                equalities[problem.Z + node_index, column_index] = 1.0
    perimeter = np.asarray([[column.perimeter for column in columns]], dtype=float)
    result = linprog(
        -np.asarray([column.fixed_state_welfare for column in columns]),
        A_ub=perimeter,
        b_ub=np.asarray([2 * math.floor(problem.boundary_prop * problem.G.number_of_edges())]),
        A_eq=equalities,
        b_eq=np.ones(problem.Z + len(noncentroids)),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Restricted analytical LP failed: {result.message}")
    return result


def _solve_restricted_integer_master(
    problem: ZoneProblem, columns: list[AnalyticalZoneColumn]
) -> tuple[tuple[AnalyticalZoneColumn, ...], float]:
    model = gp.Model("analytical_restricted_integer_master")
    model.Params.OutputFlag = 0
    choose = [
        model.addVar(vtype=gp.GRB.BINARY, name=f"column_{index}")
        for index in range(len(columns))
    ]
    for label in range(problem.Z):
        model.addConstr(
            gp.quicksum(
                choose[index]
                for index, column in enumerate(columns)
                if column.label == label
            )
            == 1
        )
    for node in problem.nodes:
        model.addConstr(
            gp.quicksum(
                choose[index]
                for index, column in enumerate(columns)
                if node in column.nodes
            )
            == 1
        )
    model.addConstr(
        gp.quicksum(
            column.perimeter * choose[index]
            for index, column in enumerate(columns)
        )
        <= 2 * math.floor(problem.boundary_prop * problem.G.number_of_edges())
    )
    model.setObjective(
        gp.quicksum(
            column.fixed_state_welfare * choose[index]
            for index, column in enumerate(columns)
        ),
        gp.GRB.MAXIMIZE,
    )
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Restricted analytical integer master status {model.Status}.")
    selected = tuple(
        column
        for index, column in enumerate(columns)
        if choose[index].X > 0.5
    )
    return selected, float(model.ObjVal)

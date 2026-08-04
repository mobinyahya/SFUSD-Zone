"""Global compact MIP for fixed analytical school-bundle/cutoff states."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import gurobipy as gp

from optimization.analytical_welfare_oracle import (
    ZonedAnalyticalWelfareResult,
    aggregate_analytical_node_values,
    evaluate_zoned_analytical_welfare,
)
from optimization.data import contiguity
from optimization.problem import AnalyticalWelfareMarket, ZoneProblem
from optimization.solvers.balance import balance_constraints


@dataclass(frozen=True)
class FixedStateMasterResult:
    assignment: dict[int, int]
    fixed_state_objective: float
    fixed_state_upper_bound: float
    q20_result: ZonedAnalyticalWelfareResult
    status: str
    solve_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class IterativeFixedStateResult:
    assignment: dict[int, int]
    q20_result: ZonedAnalyticalWelfareResult
    history: tuple[dict, ...]
    iterations: int
    timing_seconds: float


def optimize_fixed_school_bundles(
    problem: ZoneProblem,
    market: AnalyticalWelfareMarket,
    initial_assignment: Mapping[int, int],
    *,
    time_limit: float = 120.0,
    max_iterations: int = 100,
    improvement_tolerance: float = 1e-8,
) -> IterativeFixedStateResult:
    """Monotonically alternate exact fixed-cutoff geography and least cutoffs."""
    started = time.monotonic()
    assignment = dict(initial_assignment)
    current = evaluate_zoned_analytical_welfare(
        market,
        assignment,
        num_zones=problem.Z,
        cutoff_grid=market.lottery_scale,
    )
    history = [
        {
            "iteration": 0,
            "q20_welfare": current.normalized_welfare,
            "fixed_state_objective": None,
            "fixed_state_upper_bound": None,
            "status": "INITIAL",
            "elapsed_seconds": time.monotonic() - started,
        }
    ]
    seen = {tuple(sorted(assignment.items()))}
    for iteration in range(1, max_iterations + 1):
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0:
            break
        result = solve_fixed_analytical_states(
            problem,
            market,
            assignment,
            time_limit=remaining,
        )
        improvement = result.q20_result.normalized_welfare - current.normalized_welfare
        history.append(
            {
                "iteration": iteration,
                "q20_welfare": result.q20_result.normalized_welfare,
                "improvement": improvement,
                "fixed_state_objective": result.fixed_state_objective,
                "fixed_state_upper_bound": result.fixed_state_upper_bound,
                "status": result.status,
                "solve_seconds": result.solve_seconds,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if improvement < -improvement_tolerance:
            raise RuntimeError("Fixed-state iteration decreased verified Q20 welfare.")
        signature = tuple(sorted(result.assignment.items()))
        assignment = result.assignment
        current = result.q20_result
        if improvement <= improvement_tolerance or signature in seen:
            break
        seen.add(signature)
    return IterativeFixedStateResult(
        assignment=assignment,
        q20_result=current,
        history=tuple(history),
        iterations=len(history) - 1,
        timing_seconds=time.monotonic() - started,
    )


def solve_fixed_analytical_states(
    problem: ZoneProblem,
    market: AnalyticalWelfareMarket,
    initial_assignment: Mapping[int, int],
    *,
    time_limit: float = 540.0,
    enforce_market_capacities: bool = True,
) -> FixedStateMasterResult:
    """Globally optimize geography for the initial six bundles and Q20 cutoffs."""
    started = time.monotonic()
    if set(initial_assignment) != set(problem.nodes):
        raise ValueError("initial_assignment must contain every problem node.")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite.")
    initial = evaluate_zoned_analytical_welfare(
        market,
        initial_assignment,
        num_zones=problem.Z,
        cutoff_grid=market.lottery_scale,
    )
    states = {}
    for label in range(problem.Z):
        schools = tuple(
            school
            for school, node in market.school_nodes.items()
            if int(initial_assignment[node]) == label
        )
        states[label] = (
            schools,
            aggregate_analytical_node_values(
                market.segments,
                schools,
                {school: initial.school_cutoffs[school] for school in schools},
                beta=market.beta,
            ),
        )

    model = gp.Model("analytical_fixed_state_master")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.0
    selected = {
        (node, label): model.addVar(
            vtype=gp.GRB.BINARY, name=f"selected_{node}_{label}"
        )
        for node in problem.nodes
        for label in range(problem.Z)
    }
    for node in problem.nodes:
        model.addConstr(
            gp.quicksum(selected[node, label] for label in range(problem.Z)) == 1
        )
    for label, centroid in enumerate(problem.centroids):
        model.addConstr(selected[centroid, label] == 1)
    for node in problem.nodes:
        for label in range(problem.Z):
            if label not in problem.candidate_zones(node):
                model.addConstr(selected[node, label] == 0)

    supports = contiguity.contiguity_supports(
        problem.G,
        problem.centroids,
        problem.centroid_school_ids,
        problem.candidate_zones,
    )
    for node in problem.nodes:
        for label in range(problem.Z):
            if node == problem.centroids[label] or label not in problem.candidate_zones(
                node
            ):
                continue
            parents = supports.get((node, label), ())
            if parents:
                model.addConstr(
                    selected[node, label]
                    <= gp.quicksum(selected[parent, label] for parent in parents)
                )
            else:
                model.addConstr(selected[node, label] == 0)

    # This restricted master keeps every graph school in its initial zone.
    for node in problem.nodes:
        if problem.num_schools(node) <= 0:
            continue
        initial_label = int(initial_assignment[node])
        model.addConstr(selected[node, initial_label] == 1)

    for constraint in balance_constraints(problem):
        if constraint.kind == "capacity":
            continue
        for label in range(problem.Z):
            if constraint.lower_ratio is not None:
                model.addConstr(
                    gp.quicksum(
                        round(
                            100
                            * (
                                constraint.value(node)
                                - constraint.lower_ratio * problem.students(node)
                            )
                        )
                        * selected[node, label]
                        for node in problem.nodes
                    )
                    >= 0
                )
            if constraint.upper_ratio is not None:
                model.addConstr(
                    gp.quicksum(
                        round(
                            100
                            * (
                                constraint.value(node)
                                - constraint.upper_ratio * problem.students(node)
                            )
                        )
                        * selected[node, label]
                        for node in problem.nodes
                    )
                    <= 0
                )

    if enforce_market_capacities:
        for label, (schools, node_values) in states.items():
            for school in schools:
                model.addConstr(
                    gp.quicksum(
                        node_values.demands.get(node, {}).get(school, 0.0)
                        * selected[node, label]
                        for node in problem.nodes
                    )
                    <= market.school_capacities[school]
                )

    boundary = []
    for left, right in problem.G.edges:
        edge = model.addVar(vtype=gp.GRB.BINARY, name=f"boundary_{left}_{right}")
        for label in range(problem.Z):
            model.addConstr(edge >= selected[left, label] - selected[right, label])
            model.addConstr(edge >= selected[right, label] - selected[left, label])
        boundary.append(edge)
    model.addConstr(
        gp.quicksum(boundary)
        <= math.floor(problem.boundary_prop * problem.G.number_of_edges())
    )
    model.setObjective(
        gp.quicksum(
            states[label][1].welfare.get(node, 0.0) * selected[node, label]
            for node in problem.nodes
            for label in range(problem.Z)
        ),
        gp.GRB.MAXIMIZE,
    )
    solve_started = time.monotonic()
    model.optimize()
    solve_seconds = time.monotonic() - solve_started
    if model.SolCount <= 0:
        raise RuntimeError(f"Fixed-state master found no solution: {model.Status}.")
    assignment = {
        node: next(
            label
            for label in range(problem.Z)
            if selected[node, label].X > 0.5
        )
        for node in problem.nodes
    }
    q20 = evaluate_zoned_analytical_welfare(
        market,
        assignment,
        num_zones=problem.Z,
        cutoff_grid=market.lottery_scale,
    )
    statuses = {
        gp.GRB.OPTIMAL: (
            "OPTIMAL_FLOATING_FIXED_STATE"
            if enforce_market_capacities
            else "OPTIMAL_FLOATING_CAPACITY_RELAXED"
        ),
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    }
    return FixedStateMasterResult(
        assignment=assignment,
        fixed_state_objective=float(model.ObjVal),
        fixed_state_upper_bound=float(model.ObjBound),
        q20_result=q20,
        status=statuses.get(model.Status, f"STATUS_{model.Status}"),
        solve_seconds=solve_seconds,
        total_seconds=time.monotonic() - started,
    )

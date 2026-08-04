"""Lagrangian fixed-bundle geographic bounds using the local max-plus DP."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import linprog

from optimization.analytical_geography_dp import LocalGeographyDP
from optimization.analytical_welfare_oracle import AnalyticalNodeValues
from optimization.problem import ZoneProblem
from optimization.solvers.balance import balance_constraints


@dataclass(frozen=True)
class ResourceBoundResult:
    """Closed cutting-plane value of a boxed Lagrangian resource dual."""

    upper_bound: float
    selected_nodes: frozenset[int]
    lower_frl_total: int
    upper_frl_total: int
    school_demands: dict[int, float]
    perimeter: int
    multipliers: dict[str, float]
    rounds: int
    generated_sets: int
    max_separation_violation: float
    timing_seconds: float


@dataclass(frozen=True)
class ExactResourceResult:
    """Floating Gurobi result for an exact fixed-state geographic completion."""

    objective: float
    upper_bound: float
    selected_nodes: frozenset[int]
    perimeter: int
    status: str
    solve_seconds: float


class FixedCutoffResourceBounder:
    """Optimize resource multipliers around a fixed school/cutoff state."""

    def __init__(
        self,
        problem: ZoneProblem,
        label: int,
        node_values: AnalyticalNodeValues,
        school_capacities: Mapping[int, int],
        graph_school_nodes: frozenset[int],
        *,
        node_prices: Mapping[int, float] | None = None,
        perimeter_price: float = 0.0,
    ) -> None:
        self.problem = problem
        self.label = label
        self.node_values = node_values
        self.schools = tuple(map(int, school_capacities))
        self.capacities = {
            int(school): float(capacity)
            for school, capacity in school_capacities.items()
        }
        self.node_prices = {
            int(node): float(value) for node, value in (node_prices or {}).items()
        }
        if not math.isfinite(perimeter_price) or perimeter_price < 0:
            raise ValueError("perimeter_price must be finite and non-negative.")
        self.perimeter_price = perimeter_price
        all_graph_school_nodes = {
            node for node in problem.nodes if problem.num_schools(node) > 0
        }
        if not graph_school_nodes <= all_graph_school_nodes:
            raise ValueError("graph_school_nodes contains a non-school node.")
        self.fixes = {
            node: int(node in graph_school_nodes) for node in all_graph_school_nodes
        }
        self.dp = LocalGeographyDP(problem, label)
        frl = next(
            constraint
            for constraint in balance_constraints(problem)
            if constraint.kind == "frl"
        )
        self.lower_frl = {
            node: round(
                100
                * (
                    frl.value(node)
                    - frl.lower_ratio * problem.students(node)
                )
            )
            for node in problem.nodes
        }
        self.upper_frl = {
            node: round(
                100
                * (
                    frl.value(node)
                    - frl.upper_ratio * problem.students(node)
                )
            )
            for node in problem.nodes
        }
        self.perimeter_cap = math.floor(
            problem.boundary_prop * problem.G.number_of_edges()
        )

    def solve(
        self,
        *,
        tolerance: float = 1e-8,
        max_rounds: int = 200,
        multiplier_bound: float = 100.0,
    ) -> ResourceBoundResult:
        started = time.monotonic()
        if tolerance <= 0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be positive and finite.")
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive.")
        if multiplier_bound <= 0 or not math.isfinite(multiplier_bound):
            raise ValueError("multiplier_bound must be positive and finite.")

        resource_count = 2 + len(self.schools) + 1
        eta_index = resource_count
        objective = np.zeros(resource_count + 1)
        objective[2 : 2 + len(self.schools)] = [
            self.capacities[school] for school in self.schools
        ]
        objective[2 + len(self.schools)] = self.perimeter_cap
        objective[eta_index] = 1.0
        rows = []
        rhs = []
        generated = set()
        multipliers = np.zeros(resource_count)
        eta = 0.0
        max_violation = math.inf

        for round_number in range(1, max_rounds + 1):
            dp_result, base, lower_total, upper_total, demands = self._separate(
                multipliers
            )
            expression = self._lagrangian_expression(
                base,
                lower_total,
                upper_total,
                demands,
                dp_result.perimeter,
                multipliers,
            )
            max_violation = expression - eta
            signature = dp_result.selected_nodes
            if max_violation <= tolerance and rows:
                break
            if signature in generated:
                raise RuntimeError(
                    "Resource-dual separation repeated a violating local set."
                )
            generated.add(signature)
            row = np.zeros(resource_count + 1)
            row[0] = lower_total
            row[1] = -upper_total
            row[2 : 2 + len(self.schools)] = [
                -demands[school] for school in self.schools
            ]
            row[2 + len(self.schools)] = -dp_result.perimeter
            row[eta_index] = -1.0
            rows.append(row)
            rhs.append(-base)
            solution = linprog(
                objective,
                A_ub=np.asarray(rows),
                b_ub=np.asarray(rhs),
                bounds=[(0.0, multiplier_bound)] * resource_count
                + [(None, None)],
                method="highs",
            )
            if not solution.success:
                raise RuntimeError(f"Resource-dual LP failed: {solution.message}")
            multipliers = solution.x[:resource_count]
            eta = solution.x[eta_index]
        else:
            raise RuntimeError(
                f"Resource-dual separation did not close after {max_rounds} rounds."
            )

        dp_result, base, lower_total, upper_total, demands = self._separate(
            multipliers
        )
        expression = self._lagrangian_expression(
            base,
            lower_total,
            upper_total,
            demands,
            dp_result.perimeter,
            multipliers,
        )
        max_violation = max(0.0, expression - eta)
        upper_bound = float(objective[:-1] @ multipliers + expression)
        names = ["frl_lower", "frl_upper"] + [
            f"capacity_{school}" for school in self.schools
        ] + ["perimeter"]
        return ResourceBoundResult(
            upper_bound=upper_bound,
            selected_nodes=dp_result.selected_nodes,
            lower_frl_total=lower_total,
            upper_frl_total=upper_total,
            school_demands=demands,
            perimeter=dp_result.perimeter,
            multipliers={
                name: float(value) for name, value in zip(names, multipliers, strict=True)
            },
            rounds=round_number,
            generated_sets=len(generated),
            max_separation_violation=max_violation,
            timing_seconds=time.monotonic() - started,
        )

    def solve_exact_geography(self, *, time_limit: float = 60.0) -> ExactResourceResult:
        """Solve exact local resources for this fixed bundle/cutoff state."""
        return self.solve_exact_geography_pool(
            time_limit=time_limit, pool_solutions=1
        )[0]

    def solve_exact_geography_pool(
        self,
        *,
        time_limit: float = 60.0,
        pool_solutions: int = 10,
    ) -> tuple[ExactResourceResult, ...]:
        """Return several best fixed-state completions from one exact price."""
        import gurobipy as gp

        if not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("time_limit must be positive and finite.")
        if pool_solutions <= 0:
            raise ValueError("pool_solutions must be positive.")
        started = time.monotonic()
        model = gp.Model(f"analytical_fixed_geography_{self.label}")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        model.Params.MIPGap = 0.0
        model.Params.PoolSolutions = pool_solutions
        model.Params.PoolSearchMode = 2 if pool_solutions > 1 else 0
        selected = {
            node: model.addVar(vtype=gp.GRB.BINARY, name=f"selected_{node}")
            for node in self.problem.nodes
        }
        for node, value in self.fixes.items():
            model.addConstr(selected[node] == value)
        centroid = self.problem.centroids[self.label]
        model.addConstr(selected[centroid] == 1)
        for other_label, other_centroid in enumerate(self.problem.centroids):
            if other_label != self.label:
                model.addConstr(selected[other_centroid] == 0)
        for node in self.problem.nodes:
            if self.label not in self.problem.candidate_zones(node):
                model.addConstr(selected[node] == 0)
            elif node != centroid:
                parents = self.dp.supports.get(node, ())
                if parents:
                    model.addConstr(
                        selected[node] <= gp.quicksum(selected[parent] for parent in parents)
                    )
                else:
                    model.addConstr(selected[node] == 0)
        model.addConstr(
            gp.quicksum(
                self.lower_frl[node] * selected[node] for node in self.problem.nodes
            )
            >= 0
        )
        model.addConstr(
            gp.quicksum(
                self.upper_frl[node] * selected[node] for node in self.problem.nodes
            )
            <= 0
        )
        for school in self.schools:
            model.addConstr(
                gp.quicksum(
                    self.node_values.demands.get(node, {}).get(school, 0.0)
                    * selected[node]
                    for node in self.problem.nodes
                )
                <= self.capacities[school]
            )
        boundary = []
        for left, right in self.problem.G.edges:
            edge = model.addVar(vtype=gp.GRB.BINARY, name=f"boundary_{left}_{right}")
            model.addConstr(edge >= selected[left] - selected[right])
            model.addConstr(edge >= selected[right] - selected[left])
            boundary.append(edge)
        model.addConstr(gp.quicksum(boundary) <= self.perimeter_cap)
        model.setObjective(
            gp.quicksum(
                (
                    self.node_values.welfare.get(node, 0.0)
                    - self.node_prices.get(node, 0.0)
                )
                * selected[node]
                for node in self.problem.nodes
            )
            - self.perimeter_price * gp.quicksum(boundary),
            gp.GRB.MAXIMIZE,
        )
        model.optimize()
        if model.SolCount <= 0:
            raise RuntimeError(f"Fixed geographic MIP found no solution: {model.Status}.")
        statuses = {
            gp.GRB.OPTIMAL: "OPTIMAL_FLOATING",
            gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        }
        results = []
        for solution_number in range(min(pool_solutions, model.SolCount)):
            model.Params.SolutionNumber = solution_number
            selected_nodes = frozenset(
                node for node, variable in selected.items() if variable.Xn > 0.5
            )
            perimeter = sum(
                (left in selected_nodes) != (right in selected_nodes)
                for left, right in self.problem.G.edges
            )
            results.append(
                ExactResourceResult(
                    objective=float(model.PoolObjVal),
                    upper_bound=float(model.ObjBound),
                    selected_nodes=selected_nodes,
                    perimeter=perimeter,
                    status=statuses.get(model.Status, f"STATUS_{model.Status}"),
                    solve_seconds=time.monotonic() - started,
                )
            )
        return tuple(results)

    def _separate(self, multipliers: np.ndarray):
        lower_multiplier = multipliers[0]
        upper_multiplier = multipliers[1]
        capacity_multipliers = multipliers[2 : 2 + len(self.schools)]
        perimeter_multiplier = multipliers[2 + len(self.schools)]
        weights = {}
        for node in self.problem.nodes:
            demand_penalty = sum(
                multiplier
                * self.node_values.demands.get(node, {}).get(school, 0.0)
                for school, multiplier in zip(
                    self.schools, capacity_multipliers, strict=True
                )
            )
            weights[node] = (
                self.node_values.welfare.get(node, 0.0)
                - self.node_prices.get(node, 0.0)
                + lower_multiplier * self.lower_frl[node]
                - upper_multiplier * self.upper_frl[node]
                - demand_penalty
            )
        result = self.dp.solve(
            weights,
            perimeter_price=self.perimeter_price + perimeter_multiplier,
            fixes=self.fixes,
        )
        selected = result.selected_nodes
        base = sum(
            self.node_values.welfare.get(node, 0.0)
            - self.node_prices.get(node, 0.0)
            for node in selected
        ) - self.perimeter_price * result.perimeter
        lower_total = sum(self.lower_frl[node] for node in selected)
        upper_total = sum(self.upper_frl[node] for node in selected)
        demands = {
            school: sum(
                self.node_values.demands.get(node, {}).get(school, 0.0)
                for node in selected
            )
            for school in self.schools
        }
        return result, base, lower_total, upper_total, demands

    def _lagrangian_expression(
        self,
        base,
        lower_total,
        upper_total,
        demands,
        perimeter,
        multipliers,
    ):
        return (
            base
            + multipliers[0] * lower_total
            - multipliers[1] * upper_total
            - sum(
                multipliers[2 + index] * demands[school]
                for index, school in enumerate(self.schools)
            )
            - multipliers[2 + len(self.schools)] * perimeter
        )

"""Gurobi MIP solver for zoning problems.

The implementation reads :class:`ZoneProblem` directly and builds a native
Gurobi model with Boolean assignment variables ``x[z, i]``.
"""

from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB

from Zone_Generation.choice.objective import ChoiceCut
from Zone_Generation.optimization.data import contiguity
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers.base import Solver, register

_SENSE = {"<=", ">=", "=="}

# A term is (coefficient, zone, node), referencing coefficient * x[zone][node].
_Term = tuple[float, int, int]
_AssignmentVars = dict[tuple[int, int], gp.Var]


@register("mip")
class MipSolver(Solver):
    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        m = gp.Model("zoning")
        m.Params.OutputFlag = int(self.options.get("verbose", 0))
        m.Params.TimeLimit = float(self.options.get("solve_time_limit", 60))
        m.Params.MIPGap = float(self.options.get("relative_gap_limit", 0.0))
        m.Params.Seed = int(self.options.get("seed", 42))
        if "workers" in self.options:
            m.Params.Threads = int(self.options["workers"])

        x = self._build_assignment_vars(m, problem)
        self._add_core_constraints(m, problem, x)

        if problem.choice_objective is None:
            self._add_boundary_objective(m, problem, x)
        else:
            self._add_choice_objective(m, problem, x)

        self._add_hints(problem, x)

        start = time.time()
        m.optimize()
        wall = time.time() - start

        if m.Status == GRB.OPTIMAL:
            status = "OPTIMAL"
        elif m.SolCount > 0:
            status = "FEASIBLE"
        elif m.Status == GRB.INFEASIBLE:
            status = "INFEASIBLE"
        else:
            status = "UNKNOWN"

        assignment = {}
        objective = None
        if m.SolCount > 0:
            for i in problem.nodes:
                for z in problem.candidate_zones(i):
                    if x[(z, i)].X > 0.5:
                        assignment[i] = z
                        break
            objective = m.ObjVal

        metadata = {"solver": self.name}
        if problem.choice_objective is not None:
            metadata.update(
                {
                    "objective_kind": "choice_utility",
                    "choice_cuts": len(problem.choice_objective.cuts),
                }
            )
        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=status,
            objective=objective,
            wall_time=wall,
            metadata=metadata,
        )

    def _build_assignment_vars(
        self, m: gp.Model, problem: ZoneProblem
    ) -> _AssignmentVars:
        x = {}
        for i in problem.nodes:
            for z in problem.candidate_zones(i):
                x[(z, i)] = m.addVar(vtype=GRB.BINARY, name=f"x_{z}_{i}")
        m.update()
        return x

    # ------------------------------------------------------------------ #
    # Core constraints
    # ------------------------------------------------------------------ #
    def _add_core_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        self._add_assignment_constraints(m, problem, x)
        self._add_centroid_constraints(m, problem, x)
        self._add_contiguity_constraints(m, problem, x)
        self._add_capacity_constraints(m, problem, x)
        self._add_diversity_constraints(m, problem, x)
        self._add_school_count_constraints(m, problem, x)

    def _add_assignment_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        for node in problem.nodes:
            choices = [(z, node) for z in problem.candidate_zones(node)]
            if not choices:
                raise ValueError(f"Node {node} has no candidate zones (infeasible).")
            m.addConstr(gp.quicksum(x[(z, i)] for (z, i) in choices) == 1)

    def _add_centroid_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        for z, centroid in enumerate(problem.centroids):
            self._fix_assignment(m, z, centroid, x)

    def _add_contiguity_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        supports = contiguity.contiguity_supports(
            problem.G, problem.centroids, problem.candidate_zones
        )
        forbidden_by_node: dict[int, set[int]] = {}
        for (node, z), support_nodes in supports.items():
            if not support_nodes:
                # If another zone remains available, forbid unsupported choices.
                # If this is the only candidate, leave it to avoid contradictory
                # constraints in boundary-relaxed edge cases.
                forbidden = forbidden_by_node.setdefault(node, set())
                remaining = problem.candidate_zones(node) - forbidden
                if len(remaining) > 1:
                    self._forbid_assignment(m, z, node, x)
                    forbidden.add(z)
                continue

            terms: list[_Term] = [(1.0, z, node)]
            terms += [(-1.0, z, n) for n in support_nodes]
            self._add_linear_constraint(m, x, terms, "<=", 0.0)

    def _add_capacity_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        lo = 1.0 - problem.shortage
        hi = 1.0 + problem.overage
        for z in range(problem.Z):
            nodes = self._candidate_nodes(problem, z)
            ge = [
                (problem.capacity(n) - lo * problem.students(n), z, n)
                for n in nodes
            ]
            self._add_linear_constraint(m, x, ge, ">=", 0.0)

            le = [
                (problem.capacity(n) - hi * problem.students(n), z, n)
                for n in nodes
            ]
            self._add_linear_constraint(m, x, le, "<=", 0.0)

    def _add_diversity_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        def balance(value_fn, ratio: float, dev: float) -> None:
            for z in range(problem.Z):
                nodes = self._candidate_nodes(problem, z)
                upper = [
                    (value_fn(n) - (ratio + dev) * problem.students(n), z, n)
                    for n in nodes
                ]
                self._add_linear_constraint(m, x, upper, "<=", 0.0)

                lower = [
                    (value_fn(n) - (ratio - dev) * problem.students(n), z, n)
                    for n in nodes
                ]
                self._add_linear_constraint(m, x, lower, ">=", 0.0)

        balance(problem.frl, problem.district_frl, problem.frl_dev)
        racial = problem.district_racial
        for eth in problem.ethnicities:
            balance(
                lambda n, e=eth: problem.ethnicity(n, e),
                racial[eth],
                problem.racial_dev,
            )

    def _add_school_count_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        total = sum(problem.num_schools(n) for n in problem.nodes)
        if total == 0:
            return
        avg = total / problem.Z
        for z in range(problem.Z):
            nodes = self._candidate_nodes(problem, z)
            terms = [(float(problem.num_schools(n)), z, n) for n in nodes]
            self._add_linear_constraint(m, x, terms, ">=", max(0.0, avg - 1.0))
            self._add_linear_constraint(m, x, terms, "<=", avg + 1.0)

    def _add_linear_constraint(
        self,
        m: gp.Model,
        x: _AssignmentVars,
        terms: list[_Term],
        sense: str,
        rhs: float,
    ) -> None:
        if sense not in _SENSE:
            raise ValueError(f"Bad sense {sense!r}.")
        expr = gp.quicksum(
            c * x[(z, i)] for (c, z, i) in terms if (z, i) in x
        )
        if sense == "<=":
            m.addConstr(expr <= rhs)
        elif sense == ">=":
            m.addConstr(expr >= rhs)
        else:
            m.addConstr(expr == rhs)

    def _fix_assignment(
        self, m: gp.Model, zone: int, node: int, x: _AssignmentVars
    ) -> None:
        if (zone, node) in x:
            m.addConstr(x[(zone, node)] == 1)

    def _forbid_assignment(
        self, m: gp.Model, zone: int, node: int, x: _AssignmentVars
    ) -> None:
        if (zone, node) in x:
            m.addConstr(x[(zone, node)] == 0)

    def _candidate_nodes(self, problem: ZoneProblem, zone: int) -> list[int]:
        return [n for n in problem.nodes if zone in problem.candidate_zones(n)]

    # ------------------------------------------------------------------ #
    # Objective and hints
    # ------------------------------------------------------------------ #
    def _add_boundary_objective(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        boundary = []
        for u, v in problem.G.edges():
            b = m.addVar(vtype=GRB.BINARY, name=f"bnd_{u}_{v}")
            for z in problem.candidate_zones(u) | problem.candidate_zones(v):
                xu = x.get((z, u))
                xv = x.get((z, v))
                if xu is not None and xv is not None:
                    m.addConstr(b >= xu - xv)
                    m.addConstr(b >= xv - xu)
                elif xu is not None:
                    m.addConstr(b >= xu)
                elif xv is not None:
                    m.addConstr(b >= xv)
            boundary.append(b)
        m.setObjective(gp.quicksum(boundary), GRB.MINIMIZE)

    def _add_choice_objective(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        choice = problem.choice_objective
        utilities = {
            node: m.addVar(
                lb=choice.lower_bound,
                ub=choice.upper_bound,
                vtype=GRB.CONTINUOUS,
                name=f"choice_u_{node}",
            )
            for node in problem.nodes
        }
        for cut in choice.cuts:
            self._add_choice_cut(m, x, utilities, cut)

        total = m.addVar(
            lb=choice.lower_bound * len(problem.nodes),
            ub=choice.upper_bound * len(problem.nodes),
            vtype=GRB.CONTINUOUS,
            name="choice_total_utility",
        )
        m.addConstr(total == gp.quicksum(utilities.values()))
        m.setObjective(total, GRB.MAXIMIZE)

    def _add_choice_cut(
        self,
        m: gp.Model,
        x: _AssignmentVars,
        utilities: dict[int, gp.Var],
        cut: ChoiceCut,
    ) -> None:
        indicator = x.get((cut.zone, cut.node))
        if indicator is None or cut.node not in utilities:
            return
        expr = cut.constant + gp.quicksum(
            term.coefficient * x[(term.zone, term.node)]
            for term in cut.terms
            if (term.zone, term.node) in x
        )
        m.addGenConstrIndicator(indicator, True, utilities[cut.node] <= expr)

    def _add_hints(self, problem: ZoneProblem, x: _AssignmentVars) -> None:
        if not problem.hint:
            return
        for (z, i), var in x.items():
            if i in problem.hint:
                var.Start = 1 if problem.hint[i] == z else 0

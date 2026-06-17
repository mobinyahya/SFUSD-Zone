"""Gurobi MIP backend.

Reads the same :class:`ZoneProblem` and applies the same shared constraints as
the CP-SAT solvers -- no separate constraint math, no ``DesignZones`` coupling.
Gurobi consumes float coefficients directly, so no scaling is needed.
"""

from __future__ import annotations

import time

import gurobipy as gp
from gurobipy import GRB

from Zone_Generation.pipeline.problem import ZoneProblem
from Zone_Generation.pipeline.solution import ZoneSolution
from Zone_Generation.pipeline.solvers import constraints
from Zone_Generation.pipeline.solvers.base import Solver, register


class _GurobiBackend(constraints.ModelBackend):
    def __init__(self, model: gp.Model, problem: ZoneProblem):
        self.m = model
        self.problem = problem
        self.x: dict[tuple[int, int], gp.Var] = {}
        for i in problem.nodes:
            for z in problem.candidate_zones(i):
                self.x[(z, i)] = model.addVar(vtype=GRB.BINARY, name=f"x_{z}_{i}")
        model.update()

    def add_exactly_one(self, choices):
        self.m.addConstr(gp.quicksum(self.x[(z, i)] for (z, i) in choices) == 1)

    def add_linear(self, terms, sense, rhs):
        expr = gp.quicksum(
            c * self.x[(z, i)] for (c, z, i) in terms if (z, i) in self.x
        )
        if sense == "<=":
            self.m.addConstr(expr <= rhs)
        elif sense == ">=":
            self.m.addConstr(expr >= rhs)
        else:
            self.m.addConstr(expr == rhs)

    def fix(self, zone, node):
        if (zone, node) in self.x:
            self.m.addConstr(self.x[(zone, node)] == 1)

    def forbid(self, zone, node):
        if (zone, node) in self.x:
            self.m.addConstr(self.x[(zone, node)] == 0)


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

        backend = _GurobiBackend(m, problem)
        constraints.add_all(problem, backend)

        # Boundary objective: b_uv = 1 iff endpoints differ.
        boundary = []
        for u, v in problem.G.edges():
            b = m.addVar(vtype=GRB.BINARY, name=f"bnd_{u}_{v}")
            for z in problem.candidate_zones(u) | problem.candidate_zones(v):
                xu = backend.x.get((z, u))
                xv = backend.x.get((z, v))
                if xu is not None and xv is not None:
                    m.addConstr(b >= xu - xv)
                    m.addConstr(b >= xv - xu)
                elif xu is not None:
                    m.addConstr(b >= xu)
                elif xv is not None:
                    m.addConstr(b >= xv)
            boundary.append(b)
        m.setObjective(gp.quicksum(boundary), GRB.MINIMIZE)

        if problem.hint:
            for (z, i), var in backend.x.items():
                if i in problem.hint:
                    var.Start = 1 if problem.hint[i] == z else 0

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
                    if backend.x[(z, i)].X > 0.5:
                        assignment[i] = z
                        break
            objective = m.ObjVal

        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=status,
            objective=objective,
            wall_time=wall,
            metadata={"solver": self.name},
        )

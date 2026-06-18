"""OR-Tools CP-SAT backends.

A single model builder creates the assignment variables ``x[z][i]`` and applies
the shared constraint set. The two registered solvers differ *only* in how they
encode the boundary-minimization objective:

* ``cp_bool`` -- boundary bools derived directly from the ``x`` edge differences,
* ``cp_int``  -- an integer zone variable ``y[i]`` per node and ``y[u] != y[v]``
  boundary bools.

Float constraint coefficients are scaled to integers here (CP-SAT requires
integer coefficients); the shared constraint code stays in floats.
"""

from __future__ import annotations

import time

from ortools.sat.python import cp_model

from Zone_Generation.pipeline.problem import ZoneProblem
from Zone_Generation.pipeline.solution import ZoneSolution
from Zone_Generation.pipeline.solvers import constraints
from Zone_Generation.pipeline.solvers.base import Solver, register

_SCALE = 100  # integer scaling for float coefficients
_SENSE = {"<=", ">=", "=="}


class _CpBackend(constraints.ModelBackend):
    """Holds the CpModel and ``x`` vars; implements the shared primitives."""

    def __init__(self, model: cp_model.CpModel, problem: ZoneProblem):
        self.m = model
        self.problem = problem
        self.x: dict[tuple[int, int], cp_model.IntVar] = {}
        for i in problem.nodes:
            for z in problem.candidate_zones(i):
                self.x[(z, i)] = model.NewBoolVar(f"x_{z}_{i}")

    # -- primitives ---------------------------------------------------- #
    def add_exactly_one(self, choices):
        self.m.AddExactlyOne(self.x[(z, i)] for (z, i) in choices)

    def add_linear(self, terms, sense, rhs):
        if sense not in _SENSE:
            raise ValueError(f"Bad sense {sense!r}.")
        expr = sum(
            int(round(c * _SCALE)) * self.x[(z, i)]
            for (c, z, i) in terms
            if (z, i) in self.x
        )
        r = int(round(rhs * _SCALE))
        if sense == "<=":
            self.m.Add(expr <= r)
        elif sense == ">=":
            self.m.Add(expr >= r)
        else:
            self.m.Add(expr == r)

    def fix(self, zone, node):
        if (zone, node) in self.x:
            self.m.Add(self.x[(zone, node)] == 1)

    def forbid(self, zone, node):
        if (zone, node) in self.x:
            self.m.Add(self.x[(zone, node)] == 0)


class _CpSatSolver(Solver):
    """Common build/solve; subclasses add the boundary objective."""

    def _add_objective(self, m, backend, problem):  # pragma: no cover - abstract
        raise NotImplementedError

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        m = cp_model.CpModel()
        backend = _CpBackend(m, problem)
        constraints.add_all(problem, backend)
        self._add_objective(m, backend, problem)

        if problem.hint:
            for (z, i), var in backend.x.items():
                if i in problem.hint:
                    m.AddHint(var, 1 if problem.hint[i] == z else 0)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(
            self.options.get("solve_time_limit", 60)
        )
        if "relative_gap_limit" in self.options:
            solver.parameters.relative_gap_limit = float(
                self.options["relative_gap_limit"]
            )
        solver.parameters.num_search_workers = int(self.options.get("workers", 8))
        solver.parameters.random_seed = int(self.options.get("seed", 42))

        start = time.time()
        status = solver.Solve(m)
        wall = time.time() - start

        status_name = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
        }.get(status, "UNKNOWN")

        assignment = {}
        objective = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in problem.nodes:
                for z in problem.candidate_zones(i):
                    if solver.Value(backend.x[(z, i)]) == 1:
                        assignment[i] = z
                        break
            objective = solver.ObjectiveValue()

        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=status_name,
            objective=objective,
            wall_time=wall,
            metadata={"solver": self.name},
        )


@register("cp_bool")
class CpBoolSolver(_CpSatSolver):
    """Boundary minimized over ``x`` edge differences."""

    def _add_objective(self, m, backend, problem):
        boundary_vars = []
        for u, v in problem.G.edges():
            zones = problem.candidate_zones(u) | problem.candidate_zones(v)
            b = m.NewBoolVar(f"bnd_{u}_{v}")
            for z in zones:
                xu = backend.x.get((z, u))
                xv = backend.x.get((z, v))
                if xu is not None and xv is not None:
                    m.Add(b >= xu - xv)
                    m.Add(b >= xv - xu)
                elif xu is not None:
                    m.Add(b >= xu)
                elif xv is not None:
                    m.Add(b >= xv)
            boundary_vars.append(b)
        m.Minimize(sum(boundary_vars))


@register("cp_int")
class CpIntSolver(_CpSatSolver):
    """Boundary minimized over an integer zone variable per node."""

    def _add_objective(self, m, backend, problem):
        y = {}
        for i in problem.nodes:
            zones = sorted(problem.candidate_zones(i))
            y[i] = m.NewIntVarFromDomain(
                cp_model.Domain.FromValues(zones), f"y_{i}"
            )
            for z in zones:
                m.Add(y[i] == z).OnlyEnforceIf(backend.x[(z, i)])

        boundary_vars = []
        for u, v in problem.G.edges():
            b = m.NewBoolVar(f"bnd_{u}_{v}")
            m.Add(y[u] != y[v]).OnlyEnforceIf(b)
            m.Add(y[u] == y[v]).OnlyEnforceIf(b.Not())
            boundary_vars.append(b)
        m.Minimize(sum(boundary_vars))

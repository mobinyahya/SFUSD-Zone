"""OR-Tools CP-SAT backends.

The two registered solvers use different assignment encodings:

* ``cp_bool`` -- Boolean assignment variables ``x[z][i]`` plus an explicit
  exactly-one constraint per node,
* ``cp_int``  -- one integer zone variable ``y[i]`` per node, with Boolean
  indicators ``x[z][i]`` derived from ``y[i]`` for the shared linear constraints.
  This avoids adding the explicit exactly-one assignment constraint.

Float constraint coefficients are scaled to integers here (CP-SAT requires
integer coefficients); the shared constraint code stays in floats.
"""

from __future__ import annotations

import time
import math

from ortools.sat.python import cp_model

from Zone_Generation.choice.objective import ChoiceCut
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers import constraints
from Zone_Generation.optimization.solvers.base import Solver, register

_SCALE = 100  # integer scaling for float coefficients
_SENSE = {"<=", ">=", "=="}


class _CpLinearBackend(constraints.ModelBackend):
    """Shared CP-SAT linear primitives over assignment indicators."""

    def __init__(self, model: cp_model.CpModel, problem: ZoneProblem):
        self.m = model
        self.problem = problem
        self.x: dict[tuple[int, int], cp_model.IntVar] = {}

    # -- primitives ---------------------------------------------------- #
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


class _CpBoolBackend(_CpLinearBackend):
    """Boolean assignment backend for ``cp_bool``."""

    def __init__(self, model: cp_model.CpModel, problem: ZoneProblem):
        super().__init__(model, problem)
        for i in problem.nodes:
            for z in problem.candidate_zones(i):
                self.x[(z, i)] = model.NewBoolVar(f"x_{z}_{i}")

    # -- primitives ---------------------------------------------------- #
    def add_exactly_one(self, choices):
        self.m.AddExactlyOne(self.x[(z, i)] for (z, i) in choices)

    def fix(self, zone, node):
        if (zone, node) in self.x:
            self.m.Add(self.x[(zone, node)] == 1)

    def forbid(self, zone, node):
        if (zone, node) in self.x:
            self.m.Add(self.x[(zone, node)] == 0)


class _CpIntBackend(_CpLinearBackend):
    """Integer assignment backend for ``cp_int``."""

    def __init__(self, model: cp_model.CpModel, problem: ZoneProblem):
        super().__init__(model, problem)
        self.y: dict[int, cp_model.IntVar] = {}
        for i in problem.nodes:
            zones = sorted(problem.candidate_zones(i))
            if not zones:
                raise ValueError(f"Node {i} has no candidate zones (infeasible).")
            self.y[i] = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(zones), f"y_{i}"
            )
            for z in zones:
                indicator = model.NewBoolVar(f"x_{z}_{i}")
                self.x[(z, i)] = indicator
                model.Add(self.y[i] == z).OnlyEnforceIf(indicator)
                model.Add(self.y[i] != z).OnlyEnforceIf(indicator.Not())

    # -- primitives ---------------------------------------------------- #
    def add_exactly_one(self, choices):
        # ``y[i]`` has a candidate-zone domain, and each ``x[z, i]`` is fully
        # reified to ``y[i] == z``. Exactly one indicator is therefore implied.
        pass

    def fix(self, zone, node):
        if (zone, node) in self.x:
            self.m.Add(self.y[node] == zone)

    def forbid(self, zone, node):
        if (zone, node) in self.x:
            self.m.Add(self.y[node] != zone)


class _CpSatSolver(Solver):
    """Common solve flow; subclasses provide the assignment encoding."""

    backend_cls = _CpBoolBackend

    def _build_backend(self, m, problem):
        return self.backend_cls(m, problem)

    def _add_objective(self, m, backend, problem):  # pragma: no cover - abstract
        raise NotImplementedError

    def _add_hints(self, m, backend, problem):
        if not problem.hint:
            return
        for (z, i), var in backend.x.items():
            if i in problem.hint:
                m.AddHint(var, 1 if problem.hint[i] == z else 0)

    def _extract_assignment(self, solver, backend, problem):
        assignment = {}
        for i in problem.nodes:
            for z in problem.candidate_zones(i):
                if solver.Value(backend.x[(z, i)]) == 1:
                    assignment[i] = z
                    break
        return assignment

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        m = cp_model.CpModel()
        backend = self._build_backend(m, problem)
        constraints.add_all(problem, backend)
        if problem.choice_objective is None:
            self._add_objective(m, backend, problem)
        else:
            self._add_choice_objective(m, backend, problem)
        self._add_hints(m, backend, problem)

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
            assignment = self._extract_assignment(solver, backend, problem)
            objective = solver.ObjectiveValue()
            if problem.choice_objective is not None:
                objective /= problem.choice_objective.scale

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
            status=status_name,
            objective=objective,
            wall_time=wall,
            metadata=metadata,
        )

    def _add_choice_objective(self, m, backend, problem):
        choice = problem.choice_objective
        scale = float(choice.scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("choice_utility_scale must be a positive finite value.")

        lb = _scaled(choice.lower_bound, scale)
        ub = _scaled(choice.upper_bound, scale)
        if lb > ub:
            raise ValueError("Choice utility lower_bound exceeds upper_bound.")

        utilities = {
            node: m.NewIntVar(lb, ub, f"choice_u_{node}") for node in problem.nodes
        }
        for cut in choice.cuts:
            self._add_choice_cut(m, backend, utilities, cut, scale)

        total_lb = lb * len(problem.nodes)
        total_ub = ub * len(problem.nodes)
        total = m.NewIntVar(total_lb, total_ub, "choice_total_utility")
        m.Add(total == sum(utilities.values()))
        m.Maximize(total)

    def _add_choice_cut(self, m, backend, utilities, cut: ChoiceCut, scale: float):
        indicator = backend.x.get((cut.zone, cut.node))
        if indicator is None or cut.node not in utilities:
            return
        terms = []
        coeffs = []
        for term in cut.terms:
            var = backend.x.get((term.zone, term.node))
            if var is None:
                continue
            coeff = _scaled(term.coefficient, scale)
            if coeff == 0:
                continue
            terms.append(var)
            coeffs.append(coeff)
        expr = _scaled(cut.constant, scale)
        if terms:
            expr += cp_model.LinearExpr.WeightedSum(terms, coeffs)
        m.Add(utilities[cut.node] <= expr).OnlyEnforceIf(indicator)


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
class CpIntSolver(CpBoolSolver):
    """Boundary minimized over an integer zone variable per node."""

    backend_cls = _CpIntBackend

    def _add_hints(self, m, backend, problem):
        if not problem.hint:
            return
        for node, zone in problem.hint.items():
            if node in backend.y and zone in problem.candidate_zones(node):
                m.AddHint(backend.y[node], zone)

    def _extract_assignment(self, solver, backend, problem):
        return {i: int(solver.Value(backend.y[i])) for i in problem.nodes}

    def _add_objective(self, m, backend, problem):
        boundary_vars = []
        for u, v in problem.G.edges():
            b = m.NewBoolVar(f"bnd_{u}_{v}")
            m.Add(backend.y[u] != backend.y[v]).OnlyEnforceIf(b)
            m.Add(backend.y[u] == backend.y[v]).OnlyEnforceIf(b.Not())
            boundary_vars.append(b)
        m.Minimize(sum(boundary_vars))


def _scaled(value: float, scale: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError(f"Choice objective contains non-finite value: {value!r}")
    return int(round(float(value) * scale))

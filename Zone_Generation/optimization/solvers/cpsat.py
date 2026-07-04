"""OR-Tools CP-SAT solvers.

The two registered solvers use different assignment encodings:

* ``cp_bool`` -- Boolean assignment variables ``x[z][i]`` plus an explicit
  exactly-one constraint per node,
* ``cp_int``  -- one integer zone variable ``y[i]`` per node, with Boolean
  indicators ``x[z][i]`` reified from ``y[i]`` for linear constraints. This
  avoids adding the explicit exactly-one assignment constraint.

CP-SAT requires integer coefficients, so float coefficients are scaled locally
when constraints are added.
"""

from __future__ import annotations

import math
import time

from ortools.sat.python import cp_model

from Zone_Generation.choice.objective import ChoiceCut
from Zone_Generation.optimization.data import contiguity
from Zone_Generation.optimization.progress import SolverProgressTracker
from Zone_Generation.optimization.problem import ZoneProblem
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers.balance import (
    balance_constraints,
    balance_terms,
)
from Zone_Generation.optimization.solvers.base import Solver, register

_SCALE = 100  # integer scaling for float coefficients
_SENSE = {"<=", ">=", "=="}
_CP_SAT_INT_PARAMETERS = (
    "linearization_level",
    "cp_model_probing_level",
    "symmetry_level",
)

# A term is (coefficient, zone, node), referencing coefficient * x[zone][node].
_Term = tuple[float, int, int]
_AssignmentVars = dict[tuple[int, int], cp_model.IntVar]
_ZoneVars = dict[int, cp_model.IntVar]


class _CpSatProgressCallback(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
        progress: SolverProgressTracker,
        start: float,
    ) -> None:
        super().__init__()
        self._nodes = tuple(problem.nodes)
        self._progress = progress
        self._start = start
        if y:
            self._y_vars = tuple(y[node] for node in self._nodes)
            self._x_vars = None
        else:
            self._y_vars = None
            self._x_vars = tuple(
                tuple((z, x[(z, node)]) for z in sorted(problem.candidate_zones(node)))
                for node in self._nodes
            )

    def on_solution_callback(self) -> None:
        objective = self.ObjectiveValue()
        if not self._progress.is_improvement(objective):
            return
        self._progress.add(
            objective,
            time.time() - self._start,
            self._assignment(),
        )

    def _assignment(self) -> tuple[int, ...]:
        if self._y_vars is not None:
            return tuple(int(self.Value(var)) for var in self._y_vars)

        out = []
        for candidates in self._x_vars or ():
            selected = candidates[0][0]
            for zone, var in candidates:
                if self.Value(var) == 1:
                    selected = zone
                    break
            out.append(selected)
        return tuple(out)


class _CpSatSolver(Solver):
    """Common CP-SAT solve flow; subclasses own assignment encoding details."""

    def _build_assignment_vars(
        self, m: cp_model.CpModel, problem: ZoneProblem
    ) -> tuple[_AssignmentVars, _ZoneVars]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _add_assignment_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _add_boundary_objective(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _fix_assignment(
        self,
        m: cp_model.CpModel,
        zone: int,
        node: int,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        if (zone, node) in x:
            m.Add(x[(zone, node)] == 1)

    def _forbid_assignment(
        self,
        m: cp_model.CpModel,
        zone: int,
        node: int,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        if (zone, node) in x:
            m.Add(x[(zone, node)] == 0)

    def _add_hints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        if not problem.hint:
            return
        for (z, i), var in x.items():
            if i in problem.hint:
                m.AddHint(var, 1 if problem.hint[i] == z else 0)

    def _extract_assignment(
        self,
        solver: cp_model.CpSolver,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> dict[int, int]:
        assignment = {}
        for i in problem.nodes:
            for z in problem.candidate_zones(i):
                if solver.Value(x[(z, i)]) == 1:
                    assignment[i] = z
                    break
        return assignment

    def _configure_solver_parameters(self, solver: cp_model.CpSolver) -> None:
        solver.parameters.max_time_in_seconds = float(
            self.options.get("solve_time_limit", 60)
        )
        if "relative_gap_limit" in self.options:
            solver.parameters.relative_gap_limit = float(
                self.options["relative_gap_limit"]
            )
        solver.parameters.num_search_workers = int(self.options.get("workers", 8))
        solver.parameters.random_seed = int(self.options.get("seed", 42))
        for parameter_name in _CP_SAT_INT_PARAMETERS:
            value = self.options.get(parameter_name)
            if value is not None:
                setattr(solver.parameters, parameter_name, int(value))

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        m = cp_model.CpModel()
        x, y = self._build_assignment_vars(m, problem)
        self._add_core_constraints(m, problem, x, y)
        if problem.choice_objective is None:
            self._add_boundary_objective(m, problem, x, y)
            progress = self._new_solver_progress_tracker(problem, maximize=False)
        else:
            self._add_choice_objective(m, problem, x)
            progress = self._new_solver_progress_tracker(
                problem,
                maximize=True,
                objective_scale=problem.choice_objective.scale,
            )
        self._add_hints(m, problem, x, y)

        solver = cp_model.CpSolver()
        self._configure_solver_parameters(solver)
        log_path = self._next_solver_log_path(problem)
        log_file = None
        if log_path:
            solver.parameters.log_search_progress = True
            solver.parameters.log_to_stdout = False
            log_file = open(log_path, "w", encoding="utf-8")

            def write_log(text: str) -> None:
                log_file.write(text)
                log_file.flush()

            solver.log_callback = write_log

        start = time.time()
        progress_callback = (
            _CpSatProgressCallback(problem, x, y, progress, start)
            if progress is not None
            else None
        )
        try:
            if progress_callback is None:
                status = solver.Solve(m)
            else:
                status = solver.Solve(m, progress_callback)
            wall = time.time() - start
        finally:
            if log_file is not None:
                log_file.close()

        status_name = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
        }.get(status, "UNKNOWN")

        assignment = {}
        objective = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assignment = self._extract_assignment(solver, problem, x, y)
            objective = solver.ObjectiveValue()
            if problem.choice_objective is not None:
                objective /= problem.choice_objective.scale

        metadata = {
            "solver": self.name,
            **self._solver_log_metadata(log_path),
            **self._solver_progress_metadata(progress),
        }
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
            solver_progress=list(progress.entries) if progress is not None else [],
        )

    # ------------------------------------------------------------------ #
    # Core constraints
    # ------------------------------------------------------------------ #
    def _add_core_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        self._add_assignment_constraints(m, problem, x, y)
        self._add_centroid_constraints(m, problem, x, y)
        self._add_contiguity_constraints(m, problem, x, y)
        self._add_balance_constraints(m, problem, x)
        self._add_school_count_constraints(m, problem, x)

    def _add_centroid_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        for z, centroid in enumerate(problem.centroids):
            self._fix_assignment(m, z, centroid, x, y)

    def _add_contiguity_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
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
                    self._forbid_assignment(m, z, node, x, y)
                    forbidden.add(z)
                continue

            # x[z, node] => at least one supported neighbor is also in z.
            m.AddBoolOr([x[(z, node)].Not(), *[x[(z, n)] for n in support_nodes]])

    def _add_balance_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
    ) -> None:
        constraints = balance_constraints(problem)
        for z in range(problem.Z):
            nodes = self._candidate_nodes(problem, z)
            for constraint in constraints:
                lower, upper = balance_terms(problem, constraint, z, nodes)
                self._add_linear_constraint(m, x, lower, ">=", 0.0)
                self._add_linear_constraint(m, x, upper, "<=", 0.0)

    def _add_school_count_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
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
        m: cp_model.CpModel,
        x: _AssignmentVars,
        terms: list[_Term],
        sense: str,
        rhs: float,
    ) -> None:
        if sense not in _SENSE:
            raise ValueError(f"Bad sense {sense!r}.")
        expr = sum(
            int(round(c * _SCALE)) * x[(z, i)]
            for (c, z, i) in terms
            if (z, i) in x
        )
        r = int(round(rhs * _SCALE))
        if sense == "<=":
            m.Add(expr <= r)
        elif sense == ">=":
            m.Add(expr >= r)
        else:
            m.Add(expr == r)

    def _candidate_nodes(self, problem: ZoneProblem, zone: int) -> list[int]:
        return [n for n in problem.nodes if zone in problem.candidate_zones(n)]

    # ------------------------------------------------------------------ #
    # Choice objective
    # ------------------------------------------------------------------ #
    def _add_choice_objective(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
    ) -> None:
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
            self._add_choice_cut(m, x, utilities, cut, scale)

        total_lb = lb * len(problem.nodes)
        total_ub = ub * len(problem.nodes)
        total = m.NewIntVar(total_lb, total_ub, "choice_total_utility")
        m.Add(total == sum(utilities.values()))
        m.Maximize(total)

    def _add_choice_cut(
        self,
        m: cp_model.CpModel,
        x: _AssignmentVars,
        utilities: dict[int, cp_model.IntVar],
        cut: ChoiceCut,
        scale: float,
    ) -> None:
        indicator = x.get((cut.zone, cut.node))
        if indicator is None or cut.node not in utilities:
            return
        terms = []
        coeffs = []
        for term in cut.terms:
            var = x.get((term.zone, term.node))
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
    """CP-SAT solver with Boolean assignment variables."""

    def _build_assignment_vars(
        self, m: cp_model.CpModel, problem: ZoneProblem
    ) -> tuple[_AssignmentVars, _ZoneVars]:
        x = {}
        for i in problem.nodes:
            for z in problem.candidate_zones(i):
                x[(z, i)] = m.NewBoolVar(f"x_{z}_{i}")
        return x, {}

    def _add_assignment_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        for node in problem.nodes:
            choices = [(z, node) for z in problem.candidate_zones(node)]
            if not choices:
                raise ValueError(f"Node {node} has no candidate zones (infeasible).")
            m.AddExactlyOne(x[(z, i)] for (z, i) in choices)

    def _add_boundary_objective(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        boundary_vars = []
        for u, v in problem.G.edges():
            zones = problem.candidate_zones(u) | problem.candidate_zones(v)
            b = m.NewBoolVar(f"bnd_{u}_{v}")
            for z in zones:
                xu = x.get((z, u))
                xv = x.get((z, v))
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
    """CP-SAT solver with one integer zone variable per node."""

    def _build_assignment_vars(
        self, m: cp_model.CpModel, problem: ZoneProblem
    ) -> tuple[_AssignmentVars, _ZoneVars]:
        x = {}
        y = {}
        for i in problem.nodes:
            zones = sorted(problem.candidate_zones(i))
            if not zones:
                raise ValueError(f"Node {i} has no candidate zones (infeasible).")
            y[i] = m.NewIntVarFromDomain(cp_model.Domain.FromValues(zones), f"y_{i}")
            for z in zones:
                indicator = m.NewBoolVar(f"x_{z}_{i}")
                x[(z, i)] = indicator
                m.Add(y[i] == z).OnlyEnforceIf(indicator)
                m.Add(y[i] != z).OnlyEnforceIf(indicator.Not())
        return x, y

    def _add_assignment_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        # ``y[i]`` has a candidate-zone domain, and each ``x[z, i]`` is fully
        # reified to ``y[i] == z``. Exactly one indicator is therefore implied.
        return

    def _fix_assignment(
        self,
        m: cp_model.CpModel,
        zone: int,
        node: int,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        if (zone, node) in x:
            m.Add(y[node] == zone)

    def _forbid_assignment(
        self,
        m: cp_model.CpModel,
        zone: int,
        node: int,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        if (zone, node) in x:
            m.Add(y[node] != zone)

    def _add_hints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        if not problem.hint:
            return
        for node, zone in problem.hint.items():
            if node in y and zone in problem.candidate_zones(node):
                m.AddHint(y[node], zone)

    def _extract_assignment(
        self,
        solver: cp_model.CpSolver,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> dict[int, int]:
        return {i: int(solver.Value(y[i])) for i in problem.nodes}

    def _add_boundary_objective(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        boundary_vars = []
        for u, v in problem.G.edges():
            b = m.NewBoolVar(f"bnd_{u}_{v}")
            m.Add(y[u] != y[v]).OnlyEnforceIf(b)
            m.Add(y[u] == y[v]).OnlyEnforceIf(b.Not())
            boundary_vars.append(b)
        m.Minimize(sum(boundary_vars))


def _scaled(value: float, scale: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError(f"Choice objective contains non-finite value: {value!r}")
    return int(round(float(value) * scale))

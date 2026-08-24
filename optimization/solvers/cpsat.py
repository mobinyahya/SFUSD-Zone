"""OR-Tools CP-SAT solvers.

The registered solvers use different assignment encodings:

* ``cp_bool`` -- Boolean assignment variables ``x[z][i]`` plus an explicit
  exactly-one constraint per node,
* ``cp_int``  -- one integer zone variable ``y[i]`` per node, with Boolean
  indicators ``x[z][i]`` reified from ``y[i]`` for linear constraints. This
  avoids adding the explicit exactly-one assignment constraint,
* ``cp_single_zone`` -- one Boolean membership variable per node, selecting a
  connected subset around a single school centroid.

CP-SAT requires integer coefficients, so float coefficients are scaled locally
when constraints are added.
"""

from __future__ import annotations

import math
import time

import networkx as nx
from ortools.sat.python import cp_model

from choice.objective import ChoiceCut
from optimization.data import contiguity
from optimization.data.initial_solutions import initial_solution
from optimization.progress import SolverProgressTracker
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution
from optimization.solvers.balance import (
    balance_constraints,
    balance_terms,
)
from optimization.solvers.base import Solver, register

CP_SAT_SCALE = 100  # integer scaling for float coefficients
_SENSE = {"<=", ">=", "=="}
_CP_SAT_INT_PARAMETERS = (
    "linearization_level",
    "cp_model_probing_level",
    "symmetry_level",
)
_CP_SAT_SEARCH_STRATEGY_DISTANCE = "distance_to_centroid"

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


class _CpSatEnumerationCallback(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
        limit: int,
        start: float,
    ) -> None:
        super().__init__()
        self._nodes = tuple(problem.nodes)
        self._limit = limit
        self._start = start
        self._seen: set[tuple[int, ...]] = set()
        self.assignments: list[tuple[int, ...]] = []
        self.elapsed_seconds: list[float] = []
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
        assignment = self._assignment()
        if assignment in self._seen:
            return
        self._seen.add(assignment)
        self.assignments.append(assignment)
        self.elapsed_seconds.append(time.time() - self._start)
        if len(self.assignments) >= self._limit:
            self.StopSearch()

    def _assignment(self) -> tuple[int, ...]:
        if self._y_vars is not None:
            return tuple(int(self.Value(var)) for var in self._y_vars)

        out = []
        for candidates in self._x_vars or ():
            for zone, var in candidates:
                if self.Value(var) == 1:
                    out.append(zone)
                    break
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
        hint = self._hint_assignment(problem)
        if not hint:
            return
        for (z, i), var in x.items():
            if i in hint:
                m.AddHint(var, 1 if hint[i] == z else 0)

    def _hint_assignment(self, problem: ZoneProblem) -> dict[int, int] | None:
        if problem.hint:
            return problem.hint
        if "hints" not in self.options:
            return None
        initial = initial_solution(
            problem,
            self.options.get("hints"),
            solver_options=self.options,
        )
        return initial.assignment if initial is not None else None

    def _add_search_strategy(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        strategy = _normalized_cp_sat_search_strategy(
            self.options.get("cp_sat_search_strategy")
        )
        if strategy is None:
            return
        if strategy == _CP_SAT_SEARCH_STRATEGY_DISTANCE:
            variables = self._distance_to_centroid_search_vars(problem, x, y)
            if variables:
                m.AddDecisionStrategy(
                    variables,
                    cp_model.CHOOSE_FIRST,
                    self._distance_to_centroid_domain_strategy(),
                )

    def _distance_to_centroid_search_vars(
        self,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> list[cp_model.IntVar]:
        keys = sorted(
            x,
            key=lambda zone_node: (
                problem.distance(problem.centroids[zone_node[0]], zone_node[1]),
                zone_node[1],
                zone_node[0],
            ),
        )
        return [x[key] for key in keys]

    def _distance_to_centroid_domain_strategy(self):
        return cp_model.SELECT_MAX_VALUE

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

    def _configure_solver_parameters(
        self, solver: cp_model.CpSolver, *, enumerate_solutions: bool = False
    ) -> None:
        solver.parameters.max_time_in_seconds = float(
            self.options.get("solve_time_limit", 60)
        )
        if "relative_gap_limit" in self.options:
            solver.parameters.relative_gap_limit = float(
                self.options["relative_gap_limit"]
            )
        solver.parameters.num_search_workers = (
            1 if enumerate_solutions else int(self.options.get("workers", 5))
        )
        solver.parameters.enumerate_all_solutions = enumerate_solutions
        solver.parameters.random_seed = int(self.options.get("seed", 42))
        for parameter_name in _CP_SAT_INT_PARAMETERS:
            value = self.options.get(parameter_name)
            if value is not None:
                setattr(solver.parameters, parameter_name, int(value))
        if _normalized_cp_sat_search_strategy(
            self.options.get("cp_sat_search_strategy")
        ):
            solver.parameters.search_branching = cp_model.PARTIAL_FIXED_SEARCH

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        solution = self._solve(problem)
        assert isinstance(solution, ZoneSolution)
        return solution

    def find_feasible_solution(self, problem: ZoneProblem) -> ZoneSolution:
        solution = self._solve(problem, feasibility_only=True)
        assert isinstance(solution, ZoneSolution)
        return solution

    def enumerate_solutions(
        self, problem: ZoneProblem, limit: int
    ) -> list[ZoneSolution]:
        if self.name not in {"cp_bool", "cp_int"}:
            return super().enumerate_solutions(problem, limit)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("Solution enumeration limit must be a positive integer.")
        solutions = self._solve(problem, solution_limit=limit)
        assert isinstance(solutions, list)
        return solutions

    def _solve(
        self,
        problem: ZoneProblem,
        *,
        solution_limit: int | None = None,
        feasibility_only: bool = False,
    ) -> ZoneSolution | list[ZoneSolution]:
        self._centroid_neighbor_radius()
        m = cp_model.CpModel()
        x, y = self._build_assignment_vars(m, problem)
        self._add_core_constraints(m, problem, x, y)
        progress = None
        objective_scale = 1.0
        if solution_limit is None and not feasibility_only:
            maximize, objective_scale = self._add_model_objective(m, problem, x, y)
            progress = self._new_solver_progress_tracker(
                problem,
                maximize=maximize,
                objective_scale=objective_scale,
            )
        self._add_search_strategy(m, problem, x, y)
        self._add_hints(m, problem, x, y)

        solver = cp_model.CpSolver()
        self._configure_solver_parameters(
            solver, enumerate_solutions=solution_limit is not None
        )
        log_path = self._next_solver_log_path(problem)
        log_file = None
        if log_path:
            solver.parameters.log_search_progress = True
            solver.parameters.log_to_stdout = False
            log_file = open(log_path, "w", encoding="utf-8")

            def write_log(text: str) -> None:
                log_file.write(text)
                if not text.endswith("\n"):
                    log_file.write("\n")
                log_file.flush()

            solver.log_callback = write_log

        start = time.time()
        callback = (
            _CpSatEnumerationCallback(problem, x, y, solution_limit, start)
            if solution_limit is not None
            else (
                _CpSatProgressCallback(problem, x, y, progress, start)
                if progress is not None
                else None
            )
        )
        try:
            if callback is None:
                status = solver.Solve(m)
            else:
                status = solver.Solve(m, callback)
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

        if solution_limit is not None:
            assert isinstance(callback, _CpSatEnumerationCallback)
            return self._enumerated_zone_solutions(
                problem,
                callback,
                status_name,
                wall,
                solution_limit,
                log_path,
            )

        assignment = {}
        objective = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assignment = self._extract_assignment(solver, problem, x, y)
            if not feasibility_only:
                objective = solver.ObjectiveValue() / objective_scale

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
        elif problem.weight_edges:
            metadata.update(
                {
                    "objective_kind": "weighted_boundary_length",
                    "objective_unit": "meter",
                }
            )
        metadata.update(self._additional_solution_metadata(solver, m, status))
        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=status_name,
            objective=objective,
            wall_time=wall,
            metadata=metadata,
            solver_progress=list(progress.entries) if progress is not None else [],
        )

    def _enumerated_zone_solutions(
        self,
        problem: ZoneProblem,
        callback: _CpSatEnumerationCallback,
        status: str,
        wall_time: float,
        limit: int,
        log_path: str | None,
    ) -> list[ZoneSolution]:
        common_metadata = {
            "solver": self.name,
            "objective_kind": "none",
            "enumerated_solutions_limit": limit,
            "enumerated_solutions_found": len(callback.assignments),
            "enumeration_wall_time_seconds": wall_time,
            **self._solver_log_metadata(log_path),
        }
        if not callback.assignments:
            return [
                ZoneSolution(
                    problem=problem,
                    assignment={},
                    status=status,
                    objective=None,
                    wall_time=wall_time,
                    metadata=common_metadata,
                )
            ]

        solution_status = status if status in {"OPTIMAL", "FEASIBLE"} else "FEASIBLE"
        nodes = tuple(problem.nodes)
        return [
            ZoneSolution(
                problem=problem,
                assignment=dict(zip(nodes, assignment)),
                status=solution_status,
                objective=None,
                wall_time=0.0,
                metadata={
                    **common_metadata,
                    "enumerated_solution_index": index,
                    "enumeration_discovery_time_seconds": callback.elapsed_seconds[
                        index
                    ],
                },
            )
            for index, assignment in enumerate(callback.assignments)
        ]

    def _add_model_objective(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> tuple[bool, float]:
        if problem.choice_objective is None:
            self._add_boundary_objective(m, problem, x, y)
            return False, 1.0
        self._add_choice_objective(m, problem, x)
        return True, float(problem.choice_objective.scale)

    def _additional_solution_metadata(
        self, solver: cp_model.CpSolver, model: cp_model.CpModel, status: int
    ) -> dict[str, object]:
        return {}

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
        self._boundary_limit_vars = {}
        self._add_assignment_constraints(m, problem, x, y)
        self._add_centroid_constraints(m, problem, x, y)
        self._add_contiguity_constraints(m, problem, x, y)
        self._add_balance_constraints(m, problem, x)
        self._add_school_count_constraints(m, problem, x)
        self._add_boundary_constraint(m, problem, x)

    def _add_boundary_constraint(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
    ) -> None:
        if problem.boundary_prop < 0:
            return

        boundary_vars = []
        for u, v in problem.G.edges():
            boundary = m.NewBoolVar(f"boundary_limit_{u}_{v}")
            self._boundary_limit_vars[(u, v)] = boundary
            if self.name == "cp_single_zone":
                m.Add(x[(0, u)] != x[(0, v)]).OnlyEnforceIf(boundary)
                m.Add(x[(0, u)] == x[(0, v)]).OnlyEnforceIf(boundary.Not())
                boundary_vars.append(boundary)
                continue

            candidates_u = problem.candidate_zones(u)
            candidates_v = problem.candidate_zones(v)
            common = candidates_u & candidates_v
            cost_u = 2 * len(common) + len(candidates_u - candidates_v)
            cost_v = 2 * len(common) + len(candidates_v - candidates_u)
            selected, other = (u, v) if cost_u <= cost_v else (v, u)
            for zone in problem.candidate_zones(selected):
                selector = x[(zone, selected)]
                other_zone = x.get((zone, other))
                if other_zone is None:
                    m.AddImplication(selector, boundary)
                    continue
                m.AddBoolOr([selector.Not(), other_zone, boundary])
                m.AddBoolOr([selector.Not(), other_zone.Not(), boundary.Not()])
            boundary_vars.append(boundary)

        max_cut_edges = math.floor(problem.boundary_prop * problem.G.number_of_edges())
        m.Add(sum(boundary_vars) <= max_cut_edges)

    def _add_centroid_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        for zone, neighborhood in self._centroid_neighborhoods(problem).items():
            for node in neighborhood:
                self._fix_assignment(m, zone, node, x, y)
                for other_zone in problem.candidate_zones(node) - {zone}:
                    self._forbid_assignment(m, other_zone, node, x, y)

    def _add_contiguity_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        closer_supports = contiguity.closer_supports(
            problem.G,
            problem.centroids,
            problem.centroid_school_ids,
            problem.candidate_zones,
        )
        supports = contiguity.contiguity_supports(
            problem.G,
            problem.centroids,
            problem.centroid_school_ids,
            problem.candidate_zones,
        )
        for (node, z), support_nodes in supports.items():
            if not closer_supports[(node, z)] or not support_nodes:
                self._forbid_assignment(m, z, node, x, y)
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
                if lower:
                    self._add_linear_constraint(m, x, lower, ">=", 0.0)
                if upper:
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
            int(round(c * CP_SAT_SCALE)) * x[(z, i)]
            for (c, z, i) in terms
            if (z, i) in x
        )
        r = int(round(rhs * CP_SAT_SCALE))
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
                raise problem.no_candidate_zones_error(node)
            m.AddExactlyOne(x[(z, i)] for (z, i) in choices)

    def _add_boundary_objective(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        if self.options.get("secondary_objective", False):
            boundary_terms = []
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
                boundary_terms.append(problem.boundary_weight(u, v) * b)
            m.Minimize(sum(boundary_terms))
            return

        boundary_terms = []
        for u, v in problem.G.edges():
            zones = problem.candidate_zones(u) | problem.candidate_zones(v)
            b = m.NewBoolVar(f"bnd_{u}_{v}")
            for z in zones:
                xu = x.get((z, u))
                xv = x.get((z, v))

                if xu is not None and xv is not None:
                    # Direction 1: u is assigned to zone z, but v is not
                    m.Add(b == 1).OnlyEnforceIf([xu, xv.Not()])
                    # Direction 2: v is assigned to zone z, but u is not
                    m.Add(b == 1).OnlyEnforceIf([xv, xu.Not()])
                elif xu is not None:
                    # z is only a candidate for u. If u takes it, v cannot, so edge is cut.
                    m.Add(b == 1).OnlyEnforceIf(xu)
                elif xv is not None:
                    # z is only a candidate for v. If v takes it, u cannot, so edge is cut.
                    m.Add(b == 1).OnlyEnforceIf(xv)

            boundary_terms.append(problem.boundary_weight(u, v) * b)
        m.Minimize(sum(boundary_terms))


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
                raise problem.no_candidate_zones_error(i)
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
        hint = self._hint_assignment(problem)
        if not hint:
            return
        for node, zone in hint.items():
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

    def _distance_to_centroid_search_vars(
        self,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> list[cp_model.IntVar]:
        nodes = sorted(
            y,
            key=lambda node: (
                min(
                    problem.distance(problem.centroids[z], node)
                    for z in problem.candidate_zones(node)
                ),
                node,
            ),
        )
        return [y[node] for node in nodes]

    def _distance_to_centroid_domain_strategy(self):
        return cp_model.SELECT_MIN_VALUE

    def _add_boundary_objective(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        boundary_terms = []
        for u, v in problem.G.edges():
            b = m.NewBoolVar(f"bnd_{u}_{v}")
            m.Add(y[u] != y[v]).OnlyEnforceIf(b)
            m.Add(y[u] == y[v]).OnlyEnforceIf(b.Not())
            boundary_terms.append(problem.boundary_weight(u, v) * b)
        m.Minimize(sum(boundary_terms))


@register("cp_single_zone")
class CpSingleZoneSolver(CpBoolSolver):
    """Select one connected, balanced zone around one school centroid."""

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        self._validate_problem(problem)
        solution = super().solve(problem)
        solution.metadata.update(
            {
                "partial_assignment": True,
                "objective_kind": (
                    "selected_zone_weighted_boundary_length"
                    if problem.weight_edges
                    else "selected_zone_boundary"
                ),
                "centroid_node": problem.centroids[0],
                "centroid_school_id": problem.centroid_school_ids[0],
                "selected_node_count": len(solution.assignment),
                "omitted_node_count": problem.A - len(solution.assignment),
            }
        )
        return solution

    def _validate_problem(self, problem: ZoneProblem) -> None:
        if problem.Z != 1:
            raise ValueError(
                "cp_single_zone requires centroids_type to resolve to exactly "
                "one centroid."
            )
        centroid = problem.centroids[0]
        school_ids = self._school_ids(problem, centroid)
        school_count = problem.num_schools(centroid)
        if len(school_ids) != 1 or school_count != 1:
            raise ValueError(
                "cp_single_zone requires its centroid node to contain exactly one "
                f"school; node {centroid} has {len(school_ids)} school_ids and "
                f"num_schools={school_count}."
            )
        if problem.choice_objective is not None:
            raise ValueError("cp_single_zone does not support choice objectives.")
        if self.options.get("save_solver_progress"):
            raise ValueError(
                "cp_single_zone does not support save_solver_progress because "
                "its assignments omit nodes outside the selected zone."
            )
        self._centroid_neighbor_radius()

    @staticmethod
    def _school_ids(problem: ZoneProblem, node: int) -> list[int]:
        return [
            int(school_id) for school_id in problem.G.nodes[node].get("school_ids", [])
        ]

    def _is_school_node(self, problem: ZoneProblem, node: int) -> bool:
        return bool(self._school_ids(problem, node)) or problem.num_schools(node) > 0

    def _build_assignment_vars(
        self, m: cp_model.CpModel, problem: ZoneProblem
    ) -> tuple[_AssignmentVars, _ZoneVars]:
        return {(0, node): m.NewBoolVar(f"x_0_{node}") for node in problem.nodes}, {}

    def _add_assignment_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        centroid = problem.centroids[0]
        radius = self._centroid_neighbor_radius()
        other_school_neighbors = set()
        for school_node in problem.nodes:
            if school_node == centroid or not self._is_school_node(
                problem, school_node
            ):
                continue
            other_school_neighbors.update(
                nx.single_source_shortest_path_length(
                    problem.G, school_node, cutoff=radius
                )
            )

        for node in problem.nodes:
            var = x[(0, node)]
            if 0 not in problem.candidate_zones(node):
                m.Add(var == 0)
            if node in other_school_neighbors:
                m.Add(var == 0)
            if problem.fixed is not None and problem.fixed.get(node) == 0:
                m.Add(var == 1)

    def _add_school_count_constraints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
    ) -> None:
        # The centroid is fixed in and every other school node is fixed out.
        return

    def _add_hints(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        if problem.hint is None:
            return
        for node in problem.nodes:
            m.AddHint(x[(0, node)], int(problem.hint.get(node) == 0))

    def _extract_assignment(
        self,
        solver: cp_model.CpSolver,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> dict[int, int]:
        return {node: 0 for node in problem.nodes if solver.Value(x[(0, node)]) == 1}

    def _add_boundary_objective(
        self,
        m: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        boundary_terms = []
        for u, v in problem.G.edges():
            b = m.NewBoolVar(f"bnd_{u}_{v}")
            m.Add(x[(0, u)] != x[(0, v)]).OnlyEnforceIf(b)
            m.Add(x[(0, u)] == x[(0, v)]).OnlyEnforceIf(b.Not())
            boundary_terms.append(problem.boundary_weight(u, v) * b)
        m.Minimize(sum(boundary_terms))


def _scaled(value: float, scale: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError(f"Choice objective contains non-finite value: {value!r}")
    return int(round(float(value) * scale))


def _normalized_cp_sat_search_strategy(value: object) -> str | None:
    if value is None:
        return None
    strategy = str(value).strip().lower()
    if strategy in {"", "default"}:
        return None
    if strategy == _CP_SAT_SEARCH_STRATEGY_DISTANCE:
        return strategy
    raise ValueError(
        "cp_sat_search_strategy must be one of: default, distance_to_centroid."
    )

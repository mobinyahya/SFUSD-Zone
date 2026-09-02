"""Gurobi MIP solver for zoning problems.

The implementation reads :class:`ZoneProblem` directly and builds a native
Gurobi model with Boolean assignment variables ``x[z, i]``.
"""

from __future__ import annotations

import math
import time

import gurobipy as gp
from gurobipy import GRB

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

_SENSE = {"<=", ">=", "=="}

# A term is (coefficient, zone, node), referencing coefficient * x[zone][node].
_Term = tuple[float, int, int]
_AssignmentVars = dict[tuple[int, int], gp.Var]
_ProgressCaptureData = tuple[list[gp.Var], list[tuple[int, int, tuple[int, ...]]]]


def add_gurobi_zoning_geography(
    model: gp.Model,
    problem: ZoneProblem,
    *,
    centroid_neighbor_radius: int = 0,
) -> _AssignmentVars:
    """Add the canonical complete-zoning variables and constraints to a model."""
    builder = MipSolver(centroid_neighbor_radius=centroid_neighbor_radius)
    assignment = builder._build_assignment_vars(model, problem)
    builder._add_core_constraints(model, problem, assignment)
    return assignment


@register("mip")
class MipSolver(Solver):
    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        self._centroid_neighbor_radius()
        with gp.Env() as env:
            with gp.Model("zoning", env=env) as m:
                log_path = self._next_solver_log_path(problem)
                if log_path:
                    m.Params.OutputFlag = 1
                    m.Params.LogToConsole = 0
                    m.Params.LogFile = log_path
                else:
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
                    progress = self._new_solver_progress_tracker(problem, maximize=False)
                else:
                    self._add_choice_objective(m, problem, x)
                    progress = self._new_solver_progress_tracker(problem, maximize=True)

                self._add_hints(problem, x)

                start = time.time()
                if progress is None:
                    m.optimize()
                else:
                    capture_data = self._progress_capture_data(problem, x)

                    def progress_callback(model, where):
                        if where == GRB.Callback.MIPSOL:
                            self._capture_progress(
                                model,
                                progress,
                                capture_data,
                                start,
                            )

                    m.optimize(progress_callback)
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
                return ZoneSolution(
                    problem=problem,
                    assignment=assignment,
                    status=status,
                    objective=objective,
                    wall_time=wall,
                    metadata=metadata,
                    solver_progress=list(progress.entries) if progress is not None else [],
                )

    def _progress_capture_data(
        self, problem: ZoneProblem, x: _AssignmentVars
    ) -> _ProgressCaptureData:
        variables: list[gp.Var] = []
        node_slices: list[tuple[int, int, tuple[int, ...]]] = []
        for node in problem.nodes:
            zones = tuple(sorted(problem.candidate_zones(node)))
            offset = len(variables)
            for zone in zones:
                variables.append(x[(zone, node)])
            node_slices.append((offset, len(zones), zones))
        return variables, node_slices

    def _capture_progress(
        self,
        model: gp.Model,
        progress: SolverProgressTracker,
        capture_data: _ProgressCaptureData,
        start: float,
    ) -> None:
        objective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if not progress.is_improvement(objective):
            return

        variables, node_slices = capture_data
        values = model.cbGetSolution(variables)
        assignment = []
        for offset, count, zones in node_slices:
            selected = zones[0]
            best_value = values[offset]
            for idx in range(1, count):
                value = values[offset + idx]
                if value > best_value:
                    selected = zones[idx]
                    best_value = value
            assignment.append(selected)
        progress.add(objective, time.time() - start, assignment)

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
        self._add_balance_constraints(m, problem, x)
        self._add_school_count_constraints(m, problem, x)
        self._add_boundary_constraint(m, problem, x)

    def _add_boundary_constraint(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        if problem.boundary_prop < 0:
            return

        boundary_vars = []
        for u, v in problem.G.edges():
            boundary = m.addVar(vtype=GRB.BINARY, name=f"boundary_limit_{u}_{v}")
            for zone in problem.candidate_zones(u) | problem.candidate_zones(v):
                xu = x.get((zone, u))
                xv = x.get((zone, v))
                if xu is not None and xv is not None:
                    m.addConstr(boundary >= xu - xv)
                    m.addConstr(boundary >= xv - xu)
                elif xu is not None:
                    m.addConstr(boundary >= xu)
                elif xv is not None:
                    m.addConstr(boundary >= xv)
            boundary_vars.append(boundary)

        max_cut_edges = math.floor(problem.boundary_prop * problem.G.number_of_edges())
        m.addConstr(gp.quicksum(boundary_vars) <= max_cut_edges)

    def _add_assignment_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        for node in problem.nodes:
            choices = [(z, node) for z in problem.candidate_zones(node)]
            if not choices:
                raise problem.no_candidate_zones_error(node)
            m.addConstr(gp.quicksum(x[(z, i)] for (z, i) in choices) == 1)

    def _add_centroid_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
    ) -> None:
        for zone, neighborhood in self._centroid_neighborhoods(problem).items():
            for node in neighborhood:
                self._fix_assignment(m, zone, node, x)
                for other_zone in problem.candidate_zones(node) - {zone}:
                    self._forbid_assignment(m, other_zone, node, x)

    def _add_contiguity_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
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
                self._forbid_assignment(m, z, node, x)
                continue

            terms: list[_Term] = [(1.0, z, node)]
            terms += [(-1.0, z, n) for n in support_nodes]
            self._add_linear_constraint(m, x, terms, "<=", 0.0)

    def _add_balance_constraints(
        self, m: gp.Model, problem: ZoneProblem, x: _AssignmentVars
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
        expr = gp.quicksum(c * x[(z, i)] for (c, z, i) in terms if (z, i) in x)
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
        if self.options.get("secondary_objective", False):
            boundary = []
            for u, v in problem.G.edges():
                u_ord, v_ord = (u, v) if u < v else (v, u)
                b = m.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    name=f"bnd_{u_ord}_{v_ord}",
                )

                z_vars = []
                all_zones = problem.candidate_zones(u_ord) | problem.candidate_zones(
                    v_ord
                )
                for z in all_zones:
                    z_var = m.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=0.0,
                        name=f"z_{u_ord}_{v_ord}_{z}",
                    )
                    xu = x.get((z, u_ord))
                    xv = x.get((z, v_ord))
                    if xu is not None and xv is not None:
                        m.addConstr(xu - xv <= z_var)
                    elif xu is not None:
                        m.addConstr(xu <= z_var)
                    elif xv is not None:
                        m.addConstr(-xv <= z_var)
                    z_vars.append(z_var)

                m.addConstr(b == gp.quicksum(z_vars))
                boundary.append(problem.boundary_weight(u, v) * b)

            m.setObjective(gp.quicksum(boundary), GRB.MINIMIZE)
            return

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
            boundary.append(problem.boundary_weight(u, v) * b)
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
        hint = self._hint_assignment(problem)
        if not hint:
            return
        for (z, i), var in x.items():
            if i in hint:
                var.Start = 1 if hint[i] == z else 0

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

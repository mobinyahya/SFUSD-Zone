"""MIP and CP-SAT zoning masters for SAA dual cuts."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
from ortools.sat.python import cp_model

from optimization.data.saa import SaaMarket, SaaSample
from optimization.problem import ZoneProblem
from optimization.saa_oracle import (
    AccessPair,
    SaaCut,
    required_access_pairs,
    restricted_access_state,
)
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.solvers.cpsat import (
    CP_SAT_SCALE,
    CpBoolSolver,
    _AssignmentVars,
    _ZoneVars,
)
from optimization.solvers.mip import add_gurobi_zoning_geography


_CP_INTEGER_LIMIT = (2**63 - 1) // 2


@dataclass
class _CpSaaVariables:
    access: dict[AccessPair, cp_model.IntVar]
    access_joint: dict[tuple[int, int, int], cp_model.IntVar]
    fixed_access_count: int
    eta: dict[int, cp_model.IntVar]
    scaled_upper_bound: int
    max_rounding_slack: float


class SaaMipSolver(Solver):
    """Gurobi zoning master with one outer-approximation value per sample."""

    name = "saa_mip"

    def __init__(
        self,
        market: SaaMarket,
        samples: tuple[SaaSample, ...],
        cuts: tuple[SaaCut, ...],
        *,
        preprocessing_seconds: float = 0.0,
        master_index: int = 0,
        **options,
    ) -> None:
        super().__init__(**options)
        self._solve_count = master_index
        self.market = market
        self.samples = samples
        self.cuts = cuts
        self.preprocessing_seconds = preprocessing_seconds

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        model = gp.Model("saa_zoning_master")
        log_path = self._next_solver_log_path(problem)
        if log_path:
            model.Params.OutputFlag = 1
            model.Params.LogToConsole = 0
            model.Params.LogFile = log_path
        else:
            model.Params.OutputFlag = int(self.options.get("verbose", 0))
        model.Params.TimeLimit = float(self.options.get("solve_time_limit", 60.0))
        model.Params.MIPGap = float(self.options.get("relative_gap_limit", 0.0))
        model.Params.Seed = int(self.options.get("seed", 42))
        if "workers" in self.options:
            model.Params.Threads = int(self.options["workers"])

        zoning = add_gurobi_zoning_geography(
            model,
            problem,
            centroid_neighbor_radius=int(
                self.options.get("centroid_neighbor_radius", 0)
            ),
        )
        access, access_joint, fixed_access = _add_mip_access_variables(
            model, self.market, problem, zoning
        )
        upper_bound = self.market.welfare_upper_bound
        eta = {
            sample_index: model.addVar(
                lb=0.0,
                ub=upper_bound,
                vtype=GRB.CONTINUOUS,
                name=f"saa_eta_{sample_index}",
            )
            for sample_index in range(len(self.samples))
        }
        for cut_index, cut in enumerate(self.cuts):
            expression = cut.constant + gp.quicksum(
                coefficient * access[pair] for pair, coefficient in cut.coefficients
            )
            model.addConstr(
                eta[cut.sample_index] <= expression,
                name=f"saa_cut_{cut.sample_index}_{cut_index}",
            )
        model.setObjective(gp.quicksum(eta.values()) / len(self.samples), GRB.MAXIMIZE)
        _add_mip_hints(problem, zoning, access, access_joint)

        progress = self._new_solver_progress_tracker(problem, maximize=True)
        start = time.time()
        if progress is None:
            model.optimize()
        else:
            variables, node_slices = _mip_progress_capture_data(problem, zoning)

            def progress_callback(callback_model, where):
                if where != GRB.Callback.MIPSOL:
                    return
                objective = callback_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if not progress.is_improvement(objective):
                    return
                values = callback_model.cbGetSolution(variables)
                assignment = []
                for offset, count, zones in node_slices:
                    selected = zones[0]
                    best_value = values[offset]
                    for index in range(1, count):
                        value = values[offset + index]
                        if value > best_value:
                            selected = zones[index]
                            best_value = value
                    assignment.append(selected)
                progress.add(objective, time.time() - start, tuple(assignment))

            model.optimize(progress_callback)
        wall_time = time.time() - start
        if model.Status == GRB.OPTIMAL:
            status = "OPTIMAL"
        elif model.SolCount > 0:
            status = "FEASIBLE"
        elif model.Status == GRB.INFEASIBLE:
            status = "INFEASIBLE"
        else:
            status = "UNKNOWN"

        assignment = {}
        objective = None
        best_bound = None
        if model.SolCount > 0:
            objective = float(model.ObjVal)
            best_bound = float(model.ObjBound)
            for node in problem.nodes:
                for zone in problem.candidate_zones(node):
                    if zoning[(zone, node)].X > 0.5:
                        assignment[node] = zone
                        break
        return ZoneSolution(
            problem=problem,
            assignment=assignment,
            status=status,
            objective=objective,
            wall_time=wall_time,
            metadata={
                "solver": self.name,
                "formulation": "saa_stable_matching_outer_approximation",
                "objective_kind": "saa_expected_welfare_upper_bound",
                "saa_master_backend": "mip",
                "saa_num_seeds": len(self.samples),
                "saa_cut_count": len(self.cuts),
                "saa_master_best_bound": best_bound,
                "saa_access_pair_count": len(access) + len(fixed_access),
                "saa_access_indicator_count": len(access),
                "saa_access_joint_count": len(access_joint),
                "saa_preprocessing_seconds": self.preprocessing_seconds,
                "aggregate_capacity_overage_disabled": True,
                "aggregate_capacity_shortage_disabled": True,
                **self._solver_log_metadata(log_path),
                **self._solver_progress_metadata(progress),
            },
            solver_progress=list(progress.entries) if progress is not None else [],
        )


class SaaCpSatSolver(CpBoolSolver):
    """CP-SAT zoning master consuming the shared LP oracle's dual cuts."""

    def __init__(
        self,
        market: SaaMarket,
        samples: tuple[SaaSample, ...],
        cuts: tuple[SaaCut, ...],
        *,
        preprocessing_seconds: float = 0.0,
        master_index: int = 0,
        utility_scale: int = CP_SAT_SCALE,
        **options,
    ) -> None:
        super().__init__(**options)
        if (
            isinstance(utility_scale, bool)
            or not isinstance(utility_scale, int)
            or utility_scale <= 0
        ):
            raise ValueError("SAA CP-SAT utility scale must be a positive integer.")
        self._solve_count = master_index
        self.market = market
        self.samples = samples
        self.cuts = cuts
        self.preprocessing_seconds = preprocessing_seconds
        self.utility_scale = utility_scale
        self._saa_variables: _CpSaaVariables | None = None

    def _add_model_objective(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> tuple[bool, float]:
        access, access_joint, fixed_access = _add_cp_access_variables(
            model, self.market, problem, x
        )
        upper_bound = math.ceil(self.market.welfare_upper_bound * self.utility_scale)
        if upper_bound < 0 or upper_bound * len(self.samples) > _CP_INTEGER_LIMIT:
            raise ValueError("SAA CP-SAT objective exceeds integer limits.")
        eta = {
            sample_index: model.NewIntVar(
                0,
                upper_bound,
                f"saa_eta_{sample_index}",
            )
            for sample_index in range(len(self.samples))
        }
        max_rounding_slack = 0.0
        for cut_index, cut in enumerate(self.cuts):
            constant, coefficients = scaled_saa_cut(cut, self.utility_scale)
            activity_bound = abs(constant) + sum(
                abs(coefficient) for _, coefficient in coefficients
            )
            if activity_bound > _CP_INTEGER_LIMIT:
                raise ValueError("SAA CP-SAT cut exceeds integer limits.")
            expression = constant + sum(
                coefficient * access[pair] for pair, coefficient in coefficients
            )
            model.Add(eta[cut.sample_index] <= expression)
            anchor = dict(cut.anchor_access)
            scaled_anchor = constant + sum(
                coefficient * anchor.get(pair, 0) for pair, coefficient in coefficients
            )
            max_rounding_slack = max(
                max_rounding_slack,
                scaled_anchor / self.utility_scale - cut.value(anchor),
            )
        model.Maximize(sum(eta.values()))
        self._saa_variables = _CpSaaVariables(
            access=access,
            access_joint=access_joint,
            fixed_access_count=len(fixed_access),
            eta=eta,
            scaled_upper_bound=upper_bound,
            max_rounding_slack=max_rounding_slack,
        )
        return True, float(self.utility_scale * len(self.samples))

    def _add_hints(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        super()._add_hints(model, problem, x, y)
        variables = self._saa_variables
        if variables is None or not problem.hint:
            return
        for (student_node, school_node), variable in variables.access.items():
            model.AddHint(
                variable,
                int(problem.hint[student_node] == problem.hint[school_node]),
            )
        for (
            student_node,
            school_node,
            zone,
        ), variable in variables.access_joint.items():
            model.AddHint(
                variable,
                int(
                    problem.hint[student_node] == zone
                    and problem.hint[school_node] == zone
                ),
            )

    def _additional_solution_metadata(
        self,
        solver: cp_model.CpSolver,
        model: cp_model.CpModel,
        status: int,
    ) -> dict[str, object]:
        variables = self._saa_variables
        if variables is None:
            return {}
        metadata: dict[str, object] = {
            "formulation": "saa_stable_matching_outer_approximation",
            "objective_kind": "saa_expected_welfare_upper_bound",
            "saa_master_backend": "cp_bool",
            "saa_num_seeds": len(self.samples),
            "saa_cut_count": len(self.cuts),
            "saa_access_pair_count": len(variables.access)
            + variables.fixed_access_count,
            "saa_access_indicator_count": len(variables.access),
            "saa_access_joint_count": len(variables.access_joint),
            "saa_cp_sat_utility_scale": self.utility_scale,
            "saa_cp_sat_max_cut_rounding_slack": variables.max_rounding_slack,
            "saa_model_variable_count": len(model.Proto().variables),
            "saa_model_constraint_count": len(model.Proto().constraints),
            "saa_preprocessing_seconds": self.preprocessing_seconds,
            "aggregate_capacity_overage_disabled": True,
            "aggregate_capacity_shortage_disabled": True,
        }
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            objective_scale = self.utility_scale * len(self.samples)
            metadata["saa_master_best_bound"] = (
                solver.BestObjectiveBound() / objective_scale
            )
            metadata["saa_raw_master_objective"] = int(solver.ObjectiveValue())
            metadata["saa_raw_master_best_bound"] = solver.BestObjectiveBound()
        else:
            metadata["saa_master_best_bound"] = None
        return metadata


def scaled_saa_cut(
    cut: SaaCut,
    scale: int,
) -> tuple[int, tuple[tuple[AccessPair, int], ...]]:
    """Round a cut outward while keeping it tight at its anchor access."""
    anchor = dict(cut.anchor_access)
    coefficients = []
    adjusted_constant = cut.constant * scale
    for pair, coefficient in cut.coefficients:
        scaled = coefficient * scale
        rounded = math.floor(scaled) if anchor.get(pair, 0) else math.ceil(scaled)
        coefficients.append((pair, rounded))
        if anchor.get(pair, 0):
            adjusted_constant += scaled - rounded
    return math.ceil(adjusted_constant), tuple(coefficients)


def _add_mip_access_variables(model, market, problem, zoning):
    access = {}
    access_joint = {}
    fixed = {}
    for student_node, school_node in sorted(required_access_pairs(market)):
        pair, value = restricted_access_state(problem, student_node, school_node)
        if pair is None:
            fixed[(student_node, school_node)] = value
            continue
        same_zone = model.addVar(
            vtype=GRB.BINARY,
            name=f"saa_access_{student_node}_{school_node}",
        )
        joints = []
        for zone in sorted(
            problem.candidate_zones(student_node) & problem.candidate_zones(school_node)
        ):
            both = model.addVar(
                vtype=GRB.BINARY,
                name=f"saa_access_joint_{student_node}_{school_node}_{zone}",
            )
            model.addConstr(both <= zoning[(zone, student_node)])
            model.addConstr(both <= zoning[(zone, school_node)])
            model.addConstr(
                both >= zoning[(zone, student_node)] + zoning[(zone, school_node)] - 1
            )
            joints.append(both)
            access_joint[(student_node, school_node, zone)] = both
        model.addConstr(same_zone == gp.quicksum(joints))
        access[pair] = same_zone
    return access, access_joint, fixed


def _add_cp_access_variables(model, market, problem, zoning):
    access = {}
    access_joint = {}
    fixed = {}
    for student_node, school_node in sorted(required_access_pairs(market)):
        pair, value = restricted_access_state(problem, student_node, school_node)
        if pair is None:
            fixed[(student_node, school_node)] = value
            continue
        same_zone = model.NewBoolVar(f"saa_access_{student_node}_{school_node}")
        joints = []
        for zone in sorted(
            problem.candidate_zones(student_node) & problem.candidate_zones(school_node)
        ):
            both = model.NewBoolVar(
                f"saa_access_joint_{student_node}_{school_node}_{zone}"
            )
            model.AddMultiplicationEquality(
                both,
                [zoning[(zone, student_node)], zoning[(zone, school_node)]],
            )
            joints.append(both)
            access_joint[(student_node, school_node, zone)] = both
        model.Add(same_zone == sum(joints))
        access[pair] = same_zone
    return access, access_joint, fixed


def _add_mip_hints(problem, zoning, access, access_joint) -> None:
    if not problem.hint or set(problem.hint) != set(problem.nodes):
        return
    for (zone, node), variable in zoning.items():
        variable.Start = int(problem.hint[node] == zone)
    for (student_node, school_node), variable in access.items():
        variable.Start = int(problem.hint[student_node] == problem.hint[school_node])
    for (student_node, school_node, zone), variable in access_joint.items():
        variable.Start = int(
            problem.hint[student_node] == zone and problem.hint[school_node] == zone
        )


def _mip_progress_capture_data(problem, zoning):
    variables = []
    node_slices = []
    for node in problem.nodes:
        zones = tuple(sorted(problem.candidate_zones(node)))
        offset = len(variables)
        variables.extend(zoning[(zone, node)] for zone in zones)
        node_slices.append((offset, len(zones), zones))
    return variables, node_slices

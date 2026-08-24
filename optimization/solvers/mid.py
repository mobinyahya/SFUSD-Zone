"""Joint Boolean zoning and finite-grid MID welfare model."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from optimization.data.mid import MidMarket
from optimization.mid_oracle import (
    continuum_oracle,
    evaluate_cutoffs,
    finite_grid_oracle,
)
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution
from optimization.solvers.cpsat import (
    CP_SAT_SCALE,
    CpBoolSolver,
    _AssignmentVars,
    _ZoneVars,
)


@dataclass
class _MidVariables:
    cutoffs: dict[str, cp_model.IntVar]
    thresholds: dict[tuple[str, int], cp_model.IntVar]
    access: dict[tuple[int, int], cp_model.IntVar]
    effective: dict[tuple[int, str, int], cp_model.IntVar]
    remaining: tuple[tuple[cp_model.IntVar, ...], ...]
    objective: cp_model.IntVar
    access_pair_count: int


class MidCpSatSolver(CpBoolSolver):
    """Extend `cp_bool` with program cutoffs and assignment welfare."""

    def __init__(
        self,
        market: MidMarket,
        lottery_scale: int,
        *,
        preprocessing_seconds: float = 0.0,
        **options,
    ) -> None:
        super().__init__(**options)
        if (
            isinstance(lottery_scale, bool)
            or not isinstance(lottery_scale, int)
            or lottery_scale <= 0
        ):
            raise ValueError("MID lottery scale must be a positive integer.")
        self.market = market
        self.lottery_scale = lottery_scale
        self.preprocessing_seconds = preprocessing_seconds
        self._mid_variables: _MidVariables | None = None

    def _add_model_objective(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> tuple[bool, float]:
        self._mid_variables = self._add_mid_model(model, problem, x)
        return True, float(self.lottery_scale * CP_SAT_SCALE)

    def _add_mid_model(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
    ) -> _MidVariables:
        scale = self.lottery_scale
        program_by_id = self.market.program_by_id
        program_number = {
            program.program_id: index
            for index, program in enumerate(self.market.programs)
        }
        observed_priorities: dict[str, set[int]] = {
            program_id: set() for program_id in program_by_id
        }
        for student_type in self.market.types:
            for program_id, priority in zip(
                student_type.programs, student_type.priorities
            ):
                observed_priorities[program_id].add(priority)

        upper_bounds = {
            program_id: (max(priorities) + 1) * scale if priorities else 0
            for program_id, priorities in observed_priorities.items()
        }
        integer_limit = (2**63 - 1) // 2
        if (
            max(upper_bounds.values(), default=0) > integer_limit
            or self.market.student_count * scale > integer_limit
            or max(
                (program.capacity * scale for program in self.market.programs),
                default=0,
            )
            > integer_limit
        ):
            raise ValueError("MID mass or cutoff domain exceeds CP-SAT integer limits.")
        cutoffs = {
            program_id: model.NewIntVar(
                0, upper_bounds[program_id], f"mid_cutoff_{program_number[program_id]}"
            )
            for program_id in program_by_id
        }
        thresholds = {}
        for program_id, priorities in observed_priorities.items():
            upper = upper_bounds[program_id]
            for priority in sorted(priorities):
                threshold = model.NewIntVar(
                    0,
                    upper,
                    f"mid_threshold_{program_number[program_id]}_{priority}",
                )
                model.AddMaxEquality(
                    threshold,
                    [cutoffs[program_id] - priority * scale, 0],
                )
                thresholds[(program_id, priority)] = threshold

        required_access = {
            (student_type.node, program_by_id[program_id].school_node)
            for student_type in self.market.types
            for program_id in student_type.programs
            if not program_by_id[program_id].citywide
        }
        access = {}
        fixed_access = {}
        for student_node, school_node in sorted(required_access):
            if student_node == school_node:
                fixed_access[(student_node, school_node)] = True
                continue
            common_zones = sorted(
                problem.candidate_zones(student_node)
                & problem.candidate_zones(school_node)
            )
            if not common_zones:
                fixed_access[(student_node, school_node)] = False
                continue
            same_zone = model.NewBoolVar(f"mid_access_{student_node}_{school_node}")
            joint = []
            for zone in common_zones:
                both = model.NewBoolVar(
                    f"mid_access_joint_{student_node}_{school_node}_{zone}"
                )
                model.AddMultiplicationEquality(
                    both,
                    [x[(zone, student_node)], x[(zone, school_node)]],
                )
                joint.append(both)
            model.Add(same_zone == sum(joint))
            access[(student_node, school_node)] = same_zone

        effective = {}
        effective_value = {}
        for student_type in self.market.types:
            for program_id, priority in zip(
                student_type.programs, student_type.priorities
            ):
                key = (student_type.node, program_id, priority)
                if key in effective_value:
                    continue
                program = program_by_id[program_id]
                threshold = thresholds[(program_id, priority)]
                if program.citywide:
                    effective_value[key] = threshold
                    continue
                access_key = (student_type.node, program.school_node)
                if fixed_access.get(access_key) is True:
                    effective_value[key] = threshold
                    continue
                if fixed_access.get(access_key) is False:
                    effective_value[key] = scale
                    continue
                value = model.NewIntVar(
                    0,
                    max(scale, upper_bounds[program_id]),
                    f"mid_effective_{student_type.node}_"
                    f"{program_number[program_id]}_{priority}",
                )
                indicator = access[access_key]
                model.Add(value == threshold).OnlyEnforceIf(indicator)
                model.Add(value == scale).OnlyEnforceIf(indicator.Not())
                effective[key] = value
                effective_value[key] = value

        capacity_terms: dict[str, list] = {
            program_id: [] for program_id in program_by_id
        }
        objective_terms = []
        remaining_rows = []
        for type_index, student_type in enumerate(self.market.types):
            previous = scale
            row = []
            for rank, (program_id, priority) in enumerate(
                zip(student_type.programs, student_type.priorities)
            ):
                remaining = model.NewIntVar(
                    0, scale, f"mid_remaining_{type_index}_{rank}"
                )
                model.AddMinEquality(
                    remaining,
                    [
                        previous,
                        effective_value[(student_type.node, program_id, priority)],
                    ],
                )
                mass = previous - remaining
                capacity_terms[program_id].append(student_type.count * mass)
                objective_terms.append(student_type.scaled_utility_sums[rank] * mass)
                row.append(remaining)
                previous = remaining
            remaining_rows.append(tuple(row))

        for program in self.market.programs:
            model.Add(
                sum(capacity_terms[program.program_id]) <= scale * program.capacity
            )

        objective_bound = scale * sum(
            sum(student_type.scaled_utility_sums) for student_type in self.market.types
        )
        if objective_bound > integer_limit:
            raise ValueError("MID fixed-point objective exceeds CP-SAT integer limits.")
        objective = model.NewIntVar(0, objective_bound, "mid_total_welfare")
        model.Add(objective == sum(objective_terms))
        model.Maximize(objective)

        variables = _MidVariables(
            cutoffs=cutoffs,
            thresholds=thresholds,
            access=access,
            effective=effective,
            remaining=tuple(remaining_rows),
            objective=objective,
            access_pair_count=len(required_access),
        )
        self._add_mid_hints(model, problem, variables)
        return variables

    def _add_mid_hints(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        variables: _MidVariables,
    ) -> None:
        if not problem.hint or set(problem.hint) != set(problem.nodes):
            return
        hint = finite_grid_oracle(self.market, problem.hint, self.lottery_scale)
        for program_id, variable in variables.cutoffs.items():
            model.AddHint(variable, round(hint.cutoffs[program_id]))
        for row, values in zip(variables.remaining, hint.remaining_masses):
            for variable, value in zip(row, values):
                model.AddHint(variable, round(value))

    def _additional_solution_metadata(
        self, solver: cp_model.CpSolver, model: cp_model.CpModel, status: int
    ) -> dict[str, object]:
        variables = self._mid_variables
        if variables is None:
            return {}
        metadata: dict[str, object] = {
            "formulation": "mid_finite_grid",
            "objective_kind": "mid_program_welfare",
            "mid_lottery_scale": self.lottery_scale,
            "mid_utility_scale": CP_SAT_SCALE,
            "mid_utility_handling": self.market.utility_handling,
            "mid_student_count": self.market.student_count,
            "mid_utility_student_count": self.market.utility_student_count,
            "mid_outside_only_student_count": self.market.outside_only_student_count,
            "mid_program_count": len(self.market.programs),
            "mid_restricted_program_count": sum(
                not program.citywide for program in self.market.programs
            ),
            "mid_citywide_program_count": sum(
                program.citywide for program in self.market.programs
            ),
            "mid_type_count": len(self.market.types),
            "mid_compression_ratio": (
                len(self.market.types) / self.market.student_count
                if self.market.student_count
                else 0.0
            ),
            "mid_preference_count": self.market.preference_count,
            "mid_access_pair_count": variables.access_pair_count,
            "mid_access_indicator_count": len(variables.access),
            "mid_threshold_count": len(variables.thresholds),
            "mid_effective_threshold_count": len(variables.effective),
            "mid_remaining_variable_count": sum(
                len(row) for row in variables.remaining
            ),
            "mid_model_variable_count": len(model.Proto().variables),
            "mid_model_constraint_count": len(model.Proto().constraints),
            "mid_preprocessing_seconds": self.preprocessing_seconds,
            "aggregate_capacity_overage_disabled": True,
            "aggregate_capacity_shortage_disabled": True,
        }
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raw_objective = int(solver.Value(variables.objective))
            metadata["mid_raw_solver_objective"] = raw_objective
            metadata["mid_solver_fixed_point_welfare"] = raw_objective / (
                self.lottery_scale * CP_SAT_SCALE
            )
            metadata["mid_solver_cutoffs"] = {
                program_id: int(solver.Value(variable))
                for program_id, variable in variables.cutoffs.items()
            }
        return metadata

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        solution = super().solve(problem)
        solution.metadata["objective_kind"] = "mid_program_welfare"
        if not solution.feasible:
            return solution

        oracle_start = time.perf_counter()
        finite = finite_grid_oracle(
            self.market, solution.assignment, self.lottery_scale
        )
        continuous = continuum_oracle(self.market, solution.assignment)
        oracle_seconds = time.perf_counter() - oracle_start
        solver_cutoffs = solution.metadata["mid_solver_cutoffs"]
        solver_market = evaluate_cutoffs(
            self.market,
            solution.assignment,
            solver_cutoffs,
            self.lottery_scale,
        )
        solution.objective = finite.welfare
        solution.metadata.update(
            {
                "mid_solver_market_welfare": solver_market.welfare,
                "mid_solver_market_fixed_point_welfare": solver_market.fixed_point_welfare,
                "mid_solver_market_stable": solver_market.stable,
                "mid_finite_grid_welfare": finite.welfare,
                "mid_finite_grid_fixed_point_welfare": finite.fixed_point_welfare,
                "mid_finite_grid_cutoffs": {
                    key: int(round(value)) for key, value in finite.cutoffs.items()
                },
                "mid_finite_grid_demands": finite.demands,
                "mid_finite_grid_outside_mass": finite.outside_mass,
                "mid_finite_grid_stable": finite.stable,
                "mid_finite_grid_minimal": finite.minimal,
                "mid_continuum_welfare": continuous.welfare,
                "mid_continuum_cutoffs": continuous.cutoffs,
                "mid_continuum_demands": continuous.demands,
                "mid_continuum_outside_mass": continuous.outside_mass,
                "mid_continuum_stable": continuous.stable,
                "mid_continuum_minimal": continuous.minimal,
                "mid_solver_cutoff_agreement": all(
                    solver_cutoffs[program_id] == round(cutoff)
                    for program_id, cutoff in finite.cutoffs.items()
                ),
                "mid_oracle_seconds": oracle_seconds,
            }
        )
        if not math.isclose(
            solver_market.fixed_point_welfare,
            solution.metadata["mid_solver_fixed_point_welfare"],
            rel_tol=0,
            abs_tol=1e-8,
        ):
            raise RuntimeError("MID solver objective does not match cutoff evaluation.")
        return solution

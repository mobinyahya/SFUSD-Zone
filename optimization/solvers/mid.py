"""Joint Boolean zoning and finite-grid MID welfare model."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from optimization.data.mid import MidMarket, preprocess_mid_market
from optimization.mid_oracle import (
    continuum_oracle,
    evaluate_cutoffs,
    finite_grid_oracle,
)
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution
from optimization.solvers.cpsat import (
    CpBoolSolver,
    _AssignmentVars,
    _ZoneVars,
)


@dataclass
class _MidVariables:
    cutoffs: dict[str, cp_model.IntVar]
    thresholds: dict[tuple[str, int], cp_model.IntVar]
    access: dict[tuple[int, int], cp_model.IntVar]
    access_joint: dict[tuple[int, int, int], cp_model.IntVar]
    effective: dict[tuple[int, str, int], cp_model.IntVar]
    remaining: tuple[tuple[cp_model.IntVar, ...], ...]
    transport: tuple[tuple[tuple[int, cp_model.IntVar], ...], ...]
    objective: cp_model.IntVar
    objective_bound: int
    active_prefix_lengths: tuple[int, ...]
    access_pair_count: int


class MidCpSatSolver(CpBoolSolver):
    """Extend `cp_bool` with program cutoffs and assignment welfare."""

    def __init__(
        self,
        market: MidMarket,
        lottery_scale: int,
        *,
        preprocessing_seconds: float = 0.0,
        active_prefix_lengths: dict[int, int] | None = None,
        preprocessed: bool = False,
        **options,
    ) -> None:
        super().__init__(**options)
        if (
            isinstance(lottery_scale, bool)
            or not isinstance(lottery_scale, int)
            or lottery_scale <= 0
        ):
            raise ValueError("MID lottery scale must be a positive integer.")
        self.source_market = market
        self.market = market
        self.lottery_scale = lottery_scale
        self.preprocessing_seconds = preprocessing_seconds
        self.active_prefix_lengths = (
            None if active_prefix_lengths is None else dict(active_prefix_lengths)
        )
        self.preprocessed = preprocessed
        self._mid_variables: _MidVariables | None = None
        self.master_assignment_masses: tuple[tuple[int, ...], ...] | None = None

    def _add_model_objective(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> tuple[bool, float]:
        self.market = (
            self.source_market
            if self.preprocessed
            else preprocess_mid_market(self.source_market, problem)
        )
        self._mid_variables = self._add_mid_model(model, problem, x)
        return True, float(self.lottery_scale * self.market.utility_scale)

    def _add_mid_model(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
    ) -> _MidVariables:
        scale = self.lottery_scale
        active_prefix_lengths = self._validated_prefix_lengths()
        program_by_id = self.market.program_by_id
        program_number = {
            program.program_id: index
            for index, program in enumerate(self.market.programs)
        }
        observed_priorities: dict[str, set[int]] = {
            program_id: set() for program_id in program_by_id
        }
        active_priorities: dict[str, set[int]] = {
            program_id: set() for program_id in program_by_id
        }
        for type_index, student_type in enumerate(self.market.types):
            for program_id, priority in zip(
                student_type.programs, student_type.priorities
            ):
                observed_priorities[program_id].add(priority)
            prefix_length = active_prefix_lengths[type_index]
            for program_id, priority in zip(
                student_type.programs[:prefix_length],
                student_type.priorities[:prefix_length],
            ):
                active_priorities[program_id].add(priority)

        upper_bounds = {
            program_id: (max(priorities) + 1) * scale if priorities else 0
            for program_id, priorities in observed_priorities.items()
        }
        cp_integer_limit = 2**63 - 1
        integer_limit = cp_integer_limit // 2
        max_tail_mass_activity = scale * max(
            (len(student_type.programs) for student_type in self.market.types),
            default=0,
        )
        if (
            max(upper_bounds.values(), default=0) > integer_limit
            or self.market.student_count * scale > integer_limit
            or max_tail_mass_activity > integer_limit
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
        for program_id, priorities in active_priorities.items():
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
        access_joint = {}
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
                access_joint[(student_node, school_node, zone)] = both
            model.Add(same_zone == sum(joint))
            access[(student_node, school_node)] = same_zone

        effective = {}
        effective_value = {}
        for type_index, student_type in enumerate(self.market.types):
            prefix_length = active_prefix_lengths[type_index]
            for program_id, priority in zip(
                student_type.programs[:prefix_length],
                student_type.priorities[:prefix_length],
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
        transport_rows = []
        for type_index, student_type in enumerate(self.market.types):
            prefix_length = active_prefix_lengths[type_index]
            previous = scale
            row = []
            for rank, (program_id, priority) in enumerate(
                zip(
                    student_type.programs[:prefix_length],
                    student_type.priorities[:prefix_length],
                )
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

            tail = []
            for rank in range(prefix_length, len(student_type.programs)):
                program_id = student_type.programs[rank]
                program = program_by_id[program_id]
                access_key = (student_type.node, program.school_node)
                if not program.citywide and fixed_access.get(access_key) is False:
                    continue
                mass = model.NewIntVar(
                    0, scale, f"mid_transport_{type_index}_{rank}"
                )
                if not program.citywide and access_key in access:
                    model.Add(mass <= scale * access[access_key])
                capacity_terms[program_id].append(student_type.count * mass)
                objective_terms.append(student_type.scaled_utility_sums[rank] * mass)
                tail.append((rank, mass))
            if tail:
                model.Add(sum(variable for _, variable in tail) <= previous)
            transport_rows.append(tuple(tail))

        for program in self.market.programs:
            terms = capacity_terms[program.program_id]
            if terms:
                model.Add(sum(terms) <= scale * program.capacity)

        objective_bound = scale * sum(
            max(student_type.scaled_utility_sums, default=0)
            for student_type in self.market.types
        )
        if objective_bound > integer_limit:
            raise ValueError("MID fixed-point objective exceeds CP-SAT integer limits.")
        objective_expression_bound = scale * sum(
            sum(student_type.scaled_utility_sums)
            for student_type in self.market.types
        )
        if objective_bound + objective_expression_bound > cp_integer_limit:
            raise ValueError(
                "MID objective expression exceeds CP-SAT integer limits."
            )
        objective = model.NewIntVar(0, objective_bound, "mid_total_welfare")
        model.Add(objective == sum(objective_terms))
        model.Maximize(objective)

        variables = _MidVariables(
            cutoffs=cutoffs,
            thresholds=thresholds,
            access=access,
            access_joint=access_joint,
            effective=effective,
            remaining=tuple(remaining_rows),
            transport=tuple(transport_rows),
            objective=objective,
            objective_bound=objective_bound,
            active_prefix_lengths=active_prefix_lengths,
            access_pair_count=len(required_access),
        )
        self._add_mid_hints(model, problem, variables)
        return variables

    def _validated_prefix_lengths(self) -> tuple[int, ...]:
        if self.active_prefix_lengths is None:
            return tuple(len(student_type.programs) for student_type in self.market.types)
        for type_index, prefix_length in self.active_prefix_lengths.items():
            if isinstance(type_index, bool) or not isinstance(type_index, int):
                raise ValueError("MID active-prefix type indices must be integers.")
            if type_index < 0 or type_index >= len(self.market.types):
                raise ValueError(f"Unknown MID type index: {type_index}.")
            if isinstance(prefix_length, bool) or not isinstance(prefix_length, int):
                raise ValueError("MID active-prefix lengths must be integers.")
            if not 0 <= prefix_length <= len(self.market.types[type_index].programs):
                raise ValueError(
                    f"Invalid MID prefix length {prefix_length} for type {type_index}."
                )
        return tuple(
            self.active_prefix_lengths.get(type_index, 0)
            for type_index in range(len(self.market.types))
        )

    def _add_mid_hints(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        variables: _MidVariables,
    ) -> None:
        if not problem.hint or set(problem.hint) != set(problem.nodes):
            return
        hint = finite_grid_oracle(self.market, problem.hint, self.lottery_scale)
        cutoffs = {key: int(round(value)) for key, value in hint.cutoffs.items()}
        for program_id, variable in variables.cutoffs.items():
            model.AddHint(variable, cutoffs[program_id])
        for (program_id, priority), variable in variables.thresholds.items():
            model.AddHint(
                variable,
                max(cutoffs[program_id] - priority * self.lottery_scale, 0),
            )
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
        programs = self.market.program_by_id
        for (
            student_node,
            program_id,
            priority,
        ), variable in variables.effective.items():
            program = programs[program_id]
            accessible = program.citywide or (
                problem.hint[student_node] == problem.hint[program.school_node]
            )
            threshold = max(
                cutoffs[program_id] - priority * self.lottery_scale,
                0,
            )
            model.AddHint(variable, threshold if accessible else self.lottery_scale)
        for row, values in zip(variables.remaining, hint.remaining_masses):
            for variable, value in zip(row, values):
                model.AddHint(variable, int(round(value)))
        for tail, values in zip(variables.transport, hint.assignment_masses):
            for rank, variable in tail:
                model.AddHint(variable, int(round(values[rank])))
        raw_objective = 0
        for type_index, (student_type, values) in enumerate(
            zip(self.market.types, hint.remaining_masses)
        ):
            previous = self.lottery_scale
            prefix_length = variables.active_prefix_lengths[type_index]
            for utility, value in zip(
                student_type.scaled_utility_sums[:prefix_length],
                values[:prefix_length],
            ):
                remaining = int(round(value))
                raw_objective += utility * (previous - remaining)
                previous = remaining
            raw_objective += sum(
                student_type.scaled_utility_sums[rank]
                * int(round(hint.assignment_masses[type_index][rank]))
                for rank, _ in variables.transport[type_index]
            )
        model.AddHint(variables.objective, raw_objective)

    def _add_hints(
        self,
        model: cp_model.CpModel,
        problem: ZoneProblem,
        x: _AssignmentVars,
        y: _ZoneVars,
    ) -> None:
        super()._add_hints(model, problem, x, y)
        if not problem.hint or set(problem.hint) != set(problem.nodes):
            return
        for (u, v), variable in self._boundary_limit_vars.items():
            model.AddHint(variable, int(problem.hint[u] != problem.hint[v]))

    def _additional_solution_metadata(
        self, solver: cp_model.CpSolver, model: cp_model.CpModel, status: int
    ) -> dict[str, object]:
        variables = self._mid_variables
        if variables is None:
            return {}
        activated_type_count = sum(
            prefix_length > 0 for prefix_length in variables.active_prefix_lengths
        )
        fully_activated_type_count = sum(
            prefix_length == len(student_type.programs)
            for prefix_length, student_type in zip(
                variables.active_prefix_lengths, self.market.types
            )
        )
        activated_preference_count = sum(variables.active_prefix_lengths)
        metadata: dict[str, object] = {
            "formulation": (
                "mid_finite_grid"
                if self.active_prefix_lengths is None
                else "mid_generated_utility_decomposition"
            ),
            "objective_kind": "mid_program_welfare",
            "mid_lottery_scale": self.lottery_scale,
            "mid_utility_scale": self.market.utility_scale,
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
            "mid_transport_variable_count": sum(
                len(row) for row in variables.transport
            ),
            "mid_objective_upper_bound": variables.objective_bound,
            "mid_activated_type_count": activated_type_count,
            "mid_fully_activated_type_count": fully_activated_type_count,
            "mid_inactive_type_count": (
                len(self.market.types) - fully_activated_type_count
            ),
            "mid_activated_preference_count": activated_preference_count,
            "mid_inactive_preference_count": (
                self.market.preference_count - activated_preference_count
            ),
            "mid_active_prefix_lengths": {
                str(type_index): prefix_length
                for type_index, prefix_length in enumerate(
                    variables.active_prefix_lengths
                )
                if prefix_length
            },
            "mid_model_variable_count": len(model.Proto().variables),
            "mid_model_constraint_count": len(model.Proto().constraints),
            "mid_model_hint_count": len(model.Proto().solution_hint.vars),
            "mid_preprocessing_seconds": self.preprocessing_seconds,
            "aggregate_capacity_overage_disabled": True,
            "aggregate_capacity_shortage_disabled": True,
        }
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raw_objective = int(solver.Value(variables.objective))
            raw_bound = float(solver.BestObjectiveBound())
            metadata["mid_raw_solver_objective"] = raw_objective
            metadata["mid_master_candidate_objective"] = raw_objective / (
                self.lottery_scale * self.market.utility_scale
            )
            metadata["mid_solver_fixed_point_welfare"] = raw_objective / (
                self.lottery_scale * self.market.utility_scale
            )
            metadata["mid_master_raw_best_objective_bound"] = raw_bound
            metadata["mid_master_best_objective_bound"] = raw_bound / (
                self.lottery_scale * self.market.utility_scale
            )
            metadata["mid_solver_cutoffs"] = {
                program_id: int(solver.Value(variable))
                for program_id, variable in variables.cutoffs.items()
            }
            rows = []
            for type_index, student_type in enumerate(self.market.types):
                masses = [0] * len(student_type.programs)
                previous = self.lottery_scale
                for rank, variable in enumerate(variables.remaining[type_index]):
                    remaining = int(solver.Value(variable))
                    masses[rank] = previous - remaining
                    previous = remaining
                for rank, variable in variables.transport[type_index]:
                    masses[rank] = int(solver.Value(variable))
                rows.append(tuple(masses))
            self.master_assignment_masses = tuple(rows)
        return metadata

    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        solution = super().solve(problem)
        solution.metadata["objective_kind"] = "mid_program_welfare"
        if self.active_prefix_lengths is not None:
            return solution
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
        solution.objective = finite.fixed_point_welfare
        solution.metadata.update(
            {
                "mid_solver_market_welfare": solver_market.welfare,
                "mid_solver_market_fixed_point_welfare": solver_market.fixed_point_welfare,
                "mid_solver_market_stable": solver_market.stable,
                "mid_finite_grid_welfare": finite.welfare,
                "mid_finite_grid_fixed_point_value": finite.fixed_point_value,
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
        if (
            solver_market.fixed_point_value
            != solution.metadata["mid_raw_solver_objective"]
        ):
            raise RuntimeError("MID solver objective does not match cutoff evaluation.")
        return solution

"""Generated compressed-type decomposition for MID assignment welfare."""

from __future__ import annotations

import math
import time

from optimization.data.initial_solutions import initial_solution
from optimization.data.mid import build_mid_market, preprocess_mid_market
from optimization.levels import LevelSpec
from optimization.mid_oracle import (
    MidOracleResult,
    continuum_oracle,
    evaluate_cutoffs,
    finite_grid_oracle,
    separate_mid_prefixes,
)
from optimization.solution import ZoneSolution
from optimization.solvers.mid import MidCpSatSolver
from optimization.strategies.base import Strategy, register


_BUDGET_POLICY = "linearly_increasing_with_carry_forward"


@register("mid_decomp")
class MidDecompositionStrategy(Strategy):
    """Solve MID welfare by iteratively activating compressed market types."""

    def run(self, dataset, solver):
        if getattr(solver, "name", None) != "cp_bool":
            raise ValueError("mid_decomp requires solver='cp_bool'.")
        if dataset.config.program_population != "All":
            raise ValueError("mid_decomp requires program_population='All'.")

        max_iterations = int(self.options.get("max_iterations", 5))
        tolerance = float(self.options.get("tolerance", 1e-6))
        if max_iterations <= 0:
            raise ValueError("mid_decomp max_iterations must be positive.")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("mid_decomp tolerance must be non-negative.")

        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        problem = dataset.problem_for(levels[-1])
        problem.overage = -1.0
        problem.shortage = -1.0
        problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))

        hint = initial_solution(
            problem,
            self.options.get("hints", "voronoi"),
            solver_options=solver.options,
        )
        if hint is not None:
            problem.hint = hint.assignment

        preprocessing_start = time.perf_counter()
        market = preprocess_mid_market(
            build_mid_market(problem, dataset.config),
            problem,
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_start

        lottery_scale = int(self.options.get("mid_lottery_scale", 20))
        transport_bounds = self.options.get("mid_transport_bounds", True)
        if not isinstance(transport_bounds, bool):
            raise ValueError("mid_decomp mid_transport_bounds must be a Boolean.")
        objective_scale = lottery_scale * market.utility_scale
        time_limit = _final_value(
            self.options.get("solve_time_limits"),
            solver.options.get("solve_time_limit", 60.0),
        )
        relative_tolerance = _final_value(
            self.options.get("gap_limits"),
            solver.options.get("relative_gap_limit", 0.0),
        )
        if time_limit < 0 or not math.isfinite(time_limit):
            raise ValueError(
                "mid_decomp solve time limit must be finite and non-negative."
            )
        if relative_tolerance < 0 or not math.isfinite(relative_tolerance):
            raise ValueError("mid_decomp relative gap limit must be non-negative.")

        started = time.perf_counter()
        deadline = started + time_limit
        active: dict[int, int] = {}
        stages: list[ZoneSolution] = []
        iteration_records = []
        incumbent: ZoneSolution | None = None
        incumbent_result: MidOracleResult | None = None
        incumbent_value: int | None = None
        incumbent_iteration: int | None = None
        upper_bound_value: int | None = None
        master_seconds = 0.0
        oracle_seconds = 0.0

        if hint is not None and hint.metadata.get("hints") == "feasible":
            oracle_start = time.perf_counter()
            finite = finite_grid_oracle(market, hint.assignment, lottery_scale)
            oracle_seconds += time.perf_counter() - oracle_start
            incumbent = ZoneSolution(
                problem=problem,
                assignment=dict(hint.assignment),
                status="FEASIBLE",
                objective=finite.welfare,
                wall_time=0.0,
                metadata={
                    "solver": "cp_bool",
                    "formulation": "mid_generated_utility_decomposition",
                    "objective_kind": "mid_program_welfare",
                    **hint.metadata,
                },
            )
            _add_finite_metadata(incumbent, finite)
            incumbent_result = finite
            incumbent_value = int(finite.fixed_point_value)
            incumbent_iteration = -1

        termination_reason = "iteration_limit"
        base_options = dict(solver.options)
        for iteration in range(max_iterations):
            remaining_seconds = deadline - time.perf_counter()
            if remaining_seconds <= 0:
                termination_reason = "time_limit"
                break
            master_time_limit = _master_time_limit(
                remaining_seconds,
                iteration,
                max_iterations,
            )

            if incumbent is not None:
                problem.hint = incumbent.assignment
            master_options = {
                **base_options,
                "solve_time_limit": master_time_limit,
                "relative_gap_limit": relative_tolerance,
            }
            master = MidCpSatSolver(
                market,
                lottery_scale,
                preprocessing_seconds=preprocessing_seconds,
                active_prefix_lengths=active,
                transport_bounds=transport_bounds,
                preprocessed=True,
                **master_options,
            )
            master._solve_count = iteration
            master._progress_count = iteration
            active_types_before = len(active)
            active_preferences_before = sum(active.values())
            master_start = time.perf_counter()
            solution = master.solve(problem)
            iteration_master_seconds = time.perf_counter() - master_start
            master_seconds += iteration_master_seconds
            solution.metadata.update(
                {
                    "mid_decomp_iteration": iteration,
                    "mid_decomp_active_types_before": active_types_before,
                    "mid_decomp_active_preferences_before": (active_preferences_before),
                    "mid_decomp_master_time_limit_seconds": master_time_limit,
                }
            )

            if not solution.feasible:
                stages.append(solution)
                iteration_records.append(
                    {
                        "iteration": iteration,
                        "status": solution.status,
                        "activated_types_before": active_types_before,
                        "activated_types_after": len(active),
                        "activated_preferences_before": active_preferences_before,
                        "activated_preferences_after": sum(active.values()),
                        "master_time_limit_seconds": master_time_limit,
                        "master_seconds": iteration_master_seconds,
                    }
                )
                termination_reason = f"master_{solution.status.lower()}"
                break

            raw_objective = int(solution.metadata["mid_raw_solver_objective"])
            raw_bound = float(solution.metadata["mid_master_raw_best_objective_bound"])
            certified_bound = _certified_integer_upper_bound(
                raw_bound,
                raw_objective,
            )
            upper_bound_value = (
                certified_bound
                if upper_bound_value is None
                else min(upper_bound_value, certified_bound)
            )
            cutoffs = solution.metadata["mid_solver_cutoffs"]

            oracle_start = time.perf_counter()
            candidate = evaluate_cutoffs(
                market,
                solution.assignment,
                cutoffs,
                lottery_scale,
            )
            finite = finite_grid_oracle(
                market,
                solution.assignment,
                lottery_scale,
            )
            iteration_oracle_seconds = time.perf_counter() - oracle_start
            oracle_seconds += iteration_oracle_seconds
            _add_finite_metadata(solution, finite, candidate=candidate)

            finite_value = int(finite.fixed_point_value)
            if incumbent_value is None or finite_value > incumbent_value:
                incumbent = solution
                incumbent_result = finite
                incumbent_value = finite_value
                incumbent_iteration = iteration

            if master.master_assignment_masses is None:
                raise RuntimeError("MID master did not expose assignment masses.")
            separation = separate_mid_prefixes(
                market,
                candidate,
                active,
                master.master_assignment_masses,
                lottery_scale,
            )
            overload_prefixes = dict(separation.overload_prefixes)
            utility_gap_prefixes = dict(separation.utility_gap_prefixes)
            updates = dict(overload_prefixes)
            for type_index, prefix_length in utility_gap_prefixes.items():
                updates[type_index] = max(updates.get(type_index, 0), prefix_length)
            for type_index, prefix_length in updates.items():
                active[type_index] = max(active.get(type_index, 0), prefix_length)
            activated_preferences = sum(active.values()) - active_preferences_before
            newly_active = activated_preferences > 0
            if separation.overloaded_programs and not newly_active:
                raise RuntimeError("MID overload separation made no progress.")

            lower_bound, upper_bound, absolute_gap, relative_gap = _bounds(
                incumbent_value,
                upper_bound_value,
                objective_scale,
            )
            record = {
                "iteration": iteration,
                "status": solution.status,
                "activated_types_before": active_types_before,
                "activated_types_after": len(active),
                "activated_preferences_before": active_preferences_before,
                "activated_preferences_after": sum(active.values()),
                "overloaded_programs": list(separation.overloaded_programs),
                "overload_activated_prefixes": sorted(overload_prefixes.items()),
                "utility_gap_activated_prefixes": sorted(utility_gap_prefixes.items()),
                "newly_activated_preference_count": activated_preferences,
                "master_raw_objective": raw_objective,
                "master_candidate_objective": raw_objective / objective_scale,
                "master_raw_best_objective_bound": raw_bound,
                "master_best_objective_bound": raw_bound / objective_scale,
                "master_certified_upper_bound": certified_bound / objective_scale,
                "candidate_fixed_point_welfare": candidate.fixed_point_welfare,
                "incumbent_fixed_point_welfare": finite.fixed_point_welfare,
                "global_lower_bound": lower_bound,
                "global_upper_bound": upper_bound,
                "absolute_gap": absolute_gap,
                "relative_gap": relative_gap,
                "candidate_cutoffs": dict(cutoffs),
                "transport_bounds": transport_bounds,
                "model_variable_count": solution.metadata["mid_model_variable_count"],
                "model_constraint_count": solution.metadata[
                    "mid_model_constraint_count"
                ],
                "remaining_variable_count": solution.metadata[
                    "mid_remaining_variable_count"
                ],
                "transport_variable_count": solution.metadata[
                    "mid_transport_variable_count"
                ],
                "threshold_count": solution.metadata["mid_threshold_count"],
                "effective_threshold_count": solution.metadata[
                    "mid_effective_threshold_count"
                ],
                "master_time_limit_seconds": master_time_limit,
                "master_seconds": iteration_master_seconds,
                "oracle_seconds": iteration_oracle_seconds,
            }
            iteration_records.append(record)
            solution.metadata.update(
                {
                    "mid_decomp_active_types_after": len(active),
                    "mid_decomp_active_preferences_after": sum(active.values()),
                    "mid_decomp_master_time_limit_seconds": master_time_limit,
                    "mid_decomp_overload_activated_count": len(overload_prefixes),
                    "mid_decomp_utility_gap_activated_count": len(utility_gap_prefixes),
                    "mid_decomp_candidate_cutoffs": dict(cutoffs),
                    "mid_decomp_global_lower_bound": lower_bound,
                    "mid_decomp_global_upper_bound": upper_bound,
                    "mid_decomp_absolute_gap": absolute_gap,
                    "mid_decomp_relative_gap": relative_gap,
                }
            )
            stages.append(solution)

            if certified_bound == raw_objective and not newly_active:
                termination_reason = (
                    "all_preferences_active"
                    if sum(active.values()) == market.preference_count
                    else "no_separation"
                )
                break
            if _gap_reached(
                absolute_gap,
                relative_gap,
                tolerance,
                relative_tolerance,
            ):
                termination_reason = "bound_gap"
                break
            if time.perf_counter() >= deadline:
                termination_reason = "time_limit"
                break

        if incumbent is None or incumbent_result is None or incumbent_value is None:
            if not stages:
                stages.append(
                    ZoneSolution(
                        problem=problem,
                        assignment={},
                        status="UNKNOWN",
                        wall_time=0.0,
                        metadata={
                            "solver": "cp_bool",
                            "formulation": "mid_generated_utility_decomposition",
                            "objective_kind": "mid_program_welfare",
                        },
                    )
                )
            _add_summary_metadata(
                stages[-1],
                market=market,
                active=active,
                records=iteration_records,
                incumbent_iteration=None,
                incumbent_result=None,
                incumbent_value=None,
                upper_bound_value=upper_bound_value,
                objective_scale=objective_scale,
                termination_reason=termination_reason,
                transport_bounds=transport_bounds,
                total_budget_seconds=time_limit,
                preprocessing_seconds=preprocessing_seconds,
                master_seconds=master_seconds,
                oracle_seconds=oracle_seconds,
            )
            return stages

        continuous_start = time.perf_counter()
        continuous = continuum_oracle(market, incumbent.assignment)
        oracle_seconds += time.perf_counter() - continuous_start
        _add_continuous_metadata(incumbent, continuous)

        _, _, absolute_gap, _ = _bounds(
            incumbent_value,
            upper_bound_value,
            objective_scale,
        )
        certified_optimal = (
            termination_reason
            in {
                "all_types_active",
                "all_preferences_active",
                "no_separation",
            }
            or absolute_gap == 0
        )
        final = incumbent
        if not stages or stages[-1] is not incumbent:
            final = ZoneSolution(
                problem=incumbent.problem,
                assignment=dict(incumbent.assignment),
                status="OPTIMAL" if certified_optimal else "FEASIBLE",
                objective=incumbent.objective,
                wall_time=0.0,
                metadata=dict(incumbent.metadata),
                solver_progress=[],
            )
            stages.append(final)
        elif certified_optimal:
            final.status = "OPTIMAL"
        else:
            final.status = "FEASIBLE"

        _add_summary_metadata(
            final,
            market=market,
            active=active,
            records=iteration_records,
            incumbent_iteration=incumbent_iteration,
            incumbent_result=incumbent_result,
            incumbent_value=incumbent_value,
            upper_bound_value=upper_bound_value,
            objective_scale=objective_scale,
            termination_reason=termination_reason,
            transport_bounds=transport_bounds,
            total_budget_seconds=time_limit,
            preprocessing_seconds=preprocessing_seconds,
            master_seconds=master_seconds,
            oracle_seconds=oracle_seconds,
        )
        return stages


def _final_value(values, default) -> float:
    return float(values[-1] if values else default)


def _certified_integer_upper_bound(raw_bound: float, objective: int) -> int:
    if not math.isfinite(raw_bound):
        raise RuntimeError("MID master returned a non-finite objective bound.")
    if abs(raw_bound) < 2**53:
        bound = math.ceil(raw_bound)
    else:
        bound = math.ceil(math.nextafter(raw_bound, math.inf))
    return max(objective, bound)


def _master_time_limit(
    remaining_seconds: float,
    iteration: int,
    max_iterations: int,
) -> float:
    current_weight = iteration + 1
    remaining_weight = sum(range(current_weight, max_iterations + 1))
    return remaining_seconds * current_weight / remaining_weight


def _bounds(
    lower_value: int | None,
    upper_value: int | None,
    objective_scale: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    lower = lower_value / objective_scale if lower_value is not None else None
    upper = upper_value / objective_scale if upper_value is not None else None
    if lower_value is None or upper_value is None:
        return lower, upper, None, None
    if upper_value < lower_value:
        raise RuntimeError("MID decomposition upper bound is below its incumbent.")
    difference = upper_value - lower_value
    absolute_gap = difference / objective_scale
    relative_gap = difference / max(abs(lower_value), 1)
    return lower, upper, absolute_gap, relative_gap


def _gap_reached(
    absolute_gap: float | None,
    relative_gap: float | None,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    return (
        absolute_gap is not None
        and relative_gap is not None
        and (absolute_gap <= absolute_tolerance or relative_gap <= relative_tolerance)
    )


def _add_finite_metadata(
    solution: ZoneSolution,
    finite: MidOracleResult,
    *,
    candidate: MidOracleResult | None = None,
) -> None:
    solution.objective = finite.fixed_point_welfare
    solution.metadata.update(
        {
            "mid_finite_grid_welfare": finite.welfare,
            "mid_finite_grid_fixed_point_value": finite.fixed_point_value,
            "mid_finite_grid_fixed_point_welfare": finite.fixed_point_welfare,
            "mid_finite_grid_cutoffs": dict(finite.cutoffs),
            "mid_finite_grid_demands": dict(finite.demands),
            "mid_finite_grid_outside_mass": finite.outside_mass,
            "mid_finite_grid_stable": finite.stable,
            "mid_finite_grid_minimal": finite.minimal,
        }
    )
    if candidate is None:
        return
    solver_cutoffs = solution.metadata["mid_solver_cutoffs"]
    solution.metadata.update(
        {
            "mid_solver_market_welfare": candidate.welfare,
            "mid_solver_market_fixed_point_value": candidate.fixed_point_value,
            "mid_solver_market_fixed_point_welfare": (candidate.fixed_point_welfare),
            "mid_solver_market_stable": candidate.stable,
            "mid_solver_cutoff_agreement": all(
                solver_cutoffs[program_id] == cutoff
                for program_id, cutoff in finite.cutoffs.items()
            ),
        }
    )


def _add_continuous_metadata(
    solution: ZoneSolution,
    continuous: MidOracleResult,
) -> None:
    solution.metadata.update(
        {
            "mid_continuum_welfare": continuous.welfare,
            "mid_continuum_cutoffs": dict(continuous.cutoffs),
            "mid_continuum_demands": dict(continuous.demands),
            "mid_continuum_outside_mass": continuous.outside_mass,
            "mid_continuum_stable": continuous.stable,
            "mid_continuum_minimal": continuous.minimal,
        }
    )


def _add_summary_metadata(
    solution: ZoneSolution,
    *,
    market,
    active: dict[int, int],
    records: list[dict],
    incumbent_iteration: int | None,
    incumbent_result: MidOracleResult | None,
    incumbent_value: int | None,
    upper_bound_value: int | None,
    objective_scale: int,
    termination_reason: str,
    transport_bounds: bool,
    total_budget_seconds: float,
    preprocessing_seconds: float,
    master_seconds: float,
    oracle_seconds: float,
) -> None:
    lower, upper, absolute_gap, relative_gap = _bounds(
        incumbent_value,
        upper_bound_value,
        objective_scale,
    )
    programs = market.program_by_id
    solution.metadata.update(
        {
            "formulation": "mid_generated_utility_decomposition",
            "objective_kind": "mid_program_welfare",
            "mid_lottery_scale": objective_scale // market.utility_scale,
            "mid_utility_scale": market.utility_scale,
            "mid_utility_handling": market.utility_handling,
            "mid_transport_bounds": transport_bounds,
            "mid_student_count": market.student_count,
            "mid_utility_student_count": market.utility_student_count,
            "mid_outside_only_student_count": market.outside_only_student_count,
            "mid_program_count": len(market.programs),
            "mid_restricted_program_count": sum(
                not program.citywide for program in market.programs
            ),
            "mid_citywide_program_count": sum(
                program.citywide for program in market.programs
            ),
            "mid_type_count": len(market.types),
            "mid_compression_ratio": (
                len(market.types) / market.student_count
                if market.student_count
                else 0.0
            ),
            "mid_preference_count": market.preference_count,
            "mid_access_pair_count": len(
                {
                    (student_type.node, programs[program_id].school_node)
                    for student_type in market.types
                    for program_id in student_type.programs
                    if not programs[program_id].citywide
                }
            ),
            "mid_decomp_iteration_count": len(records),
            "mid_decomp_activated_type_count": len(active),
            "mid_decomp_fully_activated_type_count": sum(
                active.get(type_index, 0) == len(student_type.programs)
                for type_index, student_type in enumerate(market.types)
            ),
            "mid_decomp_total_type_count": len(market.types),
            "mid_decomp_activated_preference_count": sum(active.values()),
            "mid_decomp_total_preference_count": market.preference_count,
            "mid_decomp_active_prefix_lengths": {
                str(type_index): prefix_length
                for type_index, prefix_length in sorted(active.items())
            },
            "mid_decomp_overload_activated_count": sum(
                len(record.get("overload_activated_prefixes", ())) for record in records
            ),
            "mid_decomp_utility_gap_activated_count": sum(
                len(record.get("utility_gap_activated_prefixes", ()))
                for record in records
            ),
            "mid_decomp_iterations": records,
            "mid_decomp_global_lower_bound": lower,
            "mid_decomp_global_upper_bound": upper,
            "mid_decomp_absolute_gap": absolute_gap,
            "mid_decomp_relative_gap": relative_gap,
            "mid_decomp_best_incumbent_iteration": incumbent_iteration,
            "mid_decomp_termination_reason": termination_reason,
            "mid_decomp_total_budget_seconds": total_budget_seconds,
            "mid_decomp_budget_policy": _BUDGET_POLICY,
            "mid_decomp_total_master_seconds": master_seconds,
            "mid_decomp_total_oracle_seconds": oracle_seconds,
            "mid_preprocessing_seconds": preprocessing_seconds,
            "aggregate_capacity_overage_disabled": True,
            "aggregate_capacity_shortage_disabled": True,
        }
    )
    if records:
        solution.metadata["mid_decomp_candidate_cutoffs"] = records[-1].get(
            "candidate_cutoffs"
        )
    if incumbent_result is not None:
        solution.metadata["mid_decomp_incumbent_least_cutoffs"] = dict(
            incumbent_result.cutoffs
        )

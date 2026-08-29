"""Sample-average approximation for stable-assignment welfare."""

from __future__ import annotations

import math
import time

from optimization.data.initial_solutions import initial_solution, normalize_hints
from optimization.data.saa import build_saa_market, sample_school_preferences
from optimization.levels import LevelSpec
from optimization.saa_oracle import SaaOracle
from optimization.solution import ZoneSolution
from optimization.solvers.saa import SaaCpSatSolver, SaaMipSolver
from optimization.strategies.base import Strategy, register


@register("saa")
class SaaStrategy(Strategy):
    def run(self, dataset, solver):
        backend = getattr(solver, "name", None)
        if backend not in {"cp_bool", "mip"}:
            raise ValueError("saa requires solver='cp_bool' or solver='mip'.")
        if dataset.config.program_population != "All":
            raise ValueError("saa requires program_population='All'.")

        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        target = levels[-1]
        max_iterations = int(self.options.get("max_iterations", 5))
        tolerance = float(self.options.get("tolerance", 1e-6))
        num_seeds = int(self.options.get("saa_num_seeds", 5))
        tie_breaking_method = str(
            self.options.get("saa_tie_breaking_method", "MTB")
        ).upper()
        if max_iterations <= 0:
            raise ValueError("saa max_iterations must be positive.")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("saa tolerance must be finite and non-negative.")
        _apply_final_limit(solver.options, self.options.get("solve_time_limits"))
        _apply_final_limit(
            solver.options,
            self.options.get("gap_limits"),
            key="relative_gap_limit",
        )

        base_problem = dataset.problem_for(target)
        _configure_problem(base_problem, self.options)
        hint = initial_solution(
            base_problem,
            self.options.get("hints", "voronoi"),
            solver_options=solver.options,
        )
        if hint is not None:
            base_problem.hint = hint.assignment

        start = time.perf_counter()
        market = build_saa_market(base_problem, dataset.config)
        samples = sample_school_preferences(
            market,
            num_seeds,
            tie_breaking_method,
            int(self.options.get("seed", 42)),
        )
        oracles = tuple(
            SaaOracle(
                market,
                sample,
                sample_index,
                base_problem,
                workers=int(solver.options.get("workers", 1)),
            )
            for sample_index, sample in enumerate(samples)
        )
        preprocessing_seconds = time.perf_counter() - start

        cuts = []
        stages = []
        incumbent = None
        incumbent_welfare = float("-inf")
        incumbent_iteration = None
        global_upper_bound = float("inf")
        termination_reason = "iteration_limit"
        iterations_completed = 0
        apply_hints = normalize_hints(self.options.get("hints", "voronoi")) != "none"

        if hint is not None and hint.metadata.get("hints") == "feasible":
            evaluation_start = time.perf_counter()
            results = tuple(oracle.solve(hint.assignment) for oracle in oracles)
            evaluation_seconds = time.perf_counter() - evaluation_start
            incumbent_welfare = sum(result.welfare for result in results) / len(results)
            cuts.extend(result.cut for result in results)
            incumbent = ZoneSolution(
                problem=base_problem,
                assignment=dict(hint.assignment),
                status="FEASIBLE",
                objective=incumbent_welfare,
                wall_time=evaluation_seconds,
                metadata={
                    **hint.metadata,
                    "solver": backend,
                    "formulation": "saa_stable_matching_outer_approximation",
                    "objective_kind": "saa_expected_stable_matching_welfare",
                    "saa_master_backend": backend,
                    "saa_num_seeds": len(samples),
                    "saa_tie_breaking_method": tie_breaking_method,
                    "saa_sample_seeds": [sample.seed for sample in samples],
                    "saa_utility_handling": market.utility_handling,
                    "saa_student_count": len(market.students),
                    "saa_utility_student_count": market.utility_student_count,
                    "saa_outside_only_student_count": sum(
                        not student.programs for student in market.students
                    ),
                    "saa_program_count": len(market.programs),
                    "saa_preference_count": market.preference_count,
                    "saa_sample_welfares": [result.welfare for result in results],
                    "saa_welfare": incumbent_welfare,
                    "saa_cuts_added": len(results),
                    "saa_cuts_total": len(cuts),
                    "saa_recourse_seconds": evaluation_seconds,
                    "aggregate_capacity_overage_disabled": True,
                    "aggregate_capacity_shortage_disabled": True,
                },
            )

        for iteration in range(max_iterations):
            problem = dataset.problem_for(
                target,
                hint=(
                    incumbent.assignment
                    if incumbent is not None and apply_hints
                    else base_problem.hint
                ),
            )
            _configure_problem(problem, self.options)
            master_class = SaaCpSatSolver if backend == "cp_bool" else SaaMipSolver
            master = master_class(
                market,
                samples,
                tuple(cuts),
                preprocessing_seconds=preprocessing_seconds,
                master_index=iteration,
                **solver.options,
            )
            solution = master.solve(problem)
            iterations_completed += 1
            solution.metadata.update(
                {
                    "saa_iteration": iteration,
                    "saa_tie_breaking_method": tie_breaking_method,
                    "saa_sample_seeds": [sample.seed for sample in samples],
                    "saa_utility_handling": market.utility_handling,
                    "saa_student_count": len(market.students),
                    "saa_utility_student_count": market.utility_student_count,
                    "saa_outside_only_student_count": sum(
                        not student.programs for student in market.students
                    ),
                    "saa_program_count": len(market.programs),
                    "saa_preference_count": market.preference_count,
                    "saa_cuts_before": len(cuts),
                }
            )
            stages.append(solution)
            if not solution.feasible:
                termination_reason = "master_status"
                break

            evaluation_start = time.perf_counter()
            results = tuple(oracle.solve(solution.assignment) for oracle in oracles)
            evaluation_seconds = time.perf_counter() - evaluation_start
            solution.wall_time = float(solution.wall_time or 0.0) + evaluation_seconds
            welfare = sum(result.welfare for result in results) / len(results)
            cuts.extend(result.cut for result in results)
            if welfare > incumbent_welfare:
                incumbent = solution
                incumbent_welfare = welfare
                incumbent_iteration = iteration

            master_bound = solution.metadata.get("saa_master_best_bound")
            if master_bound is not None:
                global_upper_bound = min(global_upper_bound, float(master_bound))
            absolute_gap = max(0.0, global_upper_bound - incumbent_welfare)
            solution.metadata.update(
                {
                    "saa_master_candidate_upper_bound": solution.objective,
                    "saa_sample_welfares": [result.welfare for result in results],
                    "saa_welfare": welfare,
                    "saa_incumbent_welfare": incumbent_welfare,
                    "saa_global_upper_bound": global_upper_bound,
                    "saa_absolute_gap": absolute_gap,
                    "saa_cuts_added": len(results),
                    "saa_cuts_total": len(cuts),
                    "saa_recourse_seconds": evaluation_seconds,
                }
            )
            if absolute_gap <= tolerance:
                termination_reason = "bound_gap"
                break

        if incumbent is None:
            if stages:
                stages[-1].metadata.update(
                    {
                        "saa_iteration_count": iterations_completed,
                        "saa_certified_optimal": False,
                        "saa_global_upper_bound": (
                            global_upper_bound
                            if math.isfinite(global_upper_bound)
                            else None
                        ),
                        "saa_absolute_gap": None,
                        "saa_termination_reason": termination_reason,
                    }
                )
            return stages

        certified = (
            termination_reason == "bound_gap"
            and global_upper_bound <= incumbent_welfare
        )
        final = incumbent
        if stages[-1] is not incumbent:
            metadata = dict(incumbent.metadata)
            metadata.pop("saa_iteration", None)
            final = ZoneSolution(
                problem=incumbent.problem,
                assignment=dict(incumbent.assignment),
                status="OPTIMAL" if certified else "FEASIBLE",
                objective=incumbent_welfare,
                wall_time=(
                    0.0
                    if any(stage is incumbent for stage in stages)
                    else incumbent.wall_time
                ),
                metadata=metadata,
                solver_progress=list(incumbent.solver_progress),
            )
            stages.append(final)
        else:
            final.status = "OPTIMAL" if certified else "FEASIBLE"
            final.objective = incumbent_welfare
        final.metadata.update(
            {
                "objective_kind": "saa_expected_stable_matching_welfare",
                "saa_selected_incumbent": True,
                "saa_incumbent_iteration": incumbent_iteration,
                "saa_incumbent_welfare": incumbent_welfare,
                "saa_global_upper_bound": (
                    global_upper_bound if math.isfinite(global_upper_bound) else None
                ),
                "saa_absolute_gap": (
                    max(0.0, global_upper_bound - incumbent_welfare)
                    if math.isfinite(global_upper_bound)
                    else None
                ),
                "saa_iteration_count": iterations_completed,
                "saa_certified_optimal": certified,
                "saa_termination_reason": termination_reason,
            }
        )
        return stages


def _configure_problem(problem, options) -> None:
    problem.overage = -1.0
    problem.shortage = -1.0
    problem.boundary_prop = float(options.get("boundary_prop", -1.0))


def _apply_final_limit(options: dict, values, *, key: str = "solve_time_limit") -> None:
    if values:
        options[key] = float(values[-1])

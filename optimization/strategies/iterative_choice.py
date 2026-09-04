"""Iterative choice strategy.

The strategy solves a zoning model with explicit utility variables, evaluates the
returned zoning against the real choice model, then adds linearized utility cuts
so the next solve has a more accurate objective approximation.
"""

from __future__ import annotations

import time

from choice.objective import ChoiceObjective
from optimization.data.initial_solutions import normalize_hints
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register
from optimization.strategies.budget import Budget, make_budget
from optimization.strategies.choice_model import build_mnl_choice_model


def _budget_metadata(budget: Budget, evaluation_seconds: float) -> dict:
    """Time-budget accounting shared by every iterative-choice stage."""

    return {
        **budget.metadata("choice"),
        "choice_total_evaluation_seconds": evaluation_seconds,
    }


@register("iterative_choice")
class IterativeChoiceStrategy(Strategy):
    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        target = levels[-1]
        max_iterations = int(self.options.get("max_iterations", 5))
        tolerance = float(self.options.get("tolerance", 1e-6))
        if max_iterations <= 0:
            raise ValueError("iterative_choice max_iterations must be positive.")
        apply_hints = normalize_hints(self.options.get("hints", "voronoi")) != "none"
        scale = float(self.options.get("choice_utility_scale", 100.0))
        use_choice_utility_hints = bool(self.options.get("choice_utility_hints", False))

        budget, relative_tolerance = make_budget(
            self.options,
            solver.options,
            max_iterations,
            label="iterative_choice",
        )
        started = time.perf_counter()
        evaluation_seconds = 0.0

        model = build_mnl_choice_model(
            dataset.data,
            method=str(self.options.get("choice_model_method", "logsum")),
        )

        base_problem = dataset.problem_for(target)
        lower_bound, upper_bound = model.utility_bounds(base_problem)
        cuts = []
        hint_cut_count = 0
        if use_choice_utility_hints:
            raw_hint_cuts = model.choice_utility_hint_cuts(base_problem)
            cuts.extend(raw_hint_cuts)
            hint_cut_count = len(cuts)
        preprocessing_seconds = time.perf_counter() - started

        solutions: list[ZoneSolution] = []
        best_choice_solution: ZoneSolution | None = None
        best_choice_utility = float("-inf")
        last_feasible: ZoneSolution | None = None
        previous_model_utility: float | None = None
        termination_reason = "iteration_limit"
        iterations_completed = 0

        for iteration in range(max_iterations):
            if budget.exhausted():
                termination_reason = "time_limit"
                break
            iteration_time_limit = budget.iteration_limit(iteration)

            choice_objective = ChoiceObjective(
                cuts=tuple(cuts),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                scale=scale,
                aggregate_cuts=False,
            )
            hint_solution = best_choice_solution or last_feasible
            hint = (
                hint_solution.assignment
                if hint_solution is not None and apply_hints
                else None
            )
            # Without choice_utility_hints, the first iteration intentionally has
            # no cuts, matching the legacy unconstrained seed.
            problem = dataset.problem_for(
                target,
                hint=hint,
                choice_objective=choice_objective,
            )
            problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
            solver.options["solve_time_limit"] = iteration_time_limit
            solver.options["relative_gap_limit"] = relative_tolerance
            master_start = time.perf_counter()
            sol = solver.solve(problem)
            budget.charge(time.perf_counter() - master_start)
            iterations_completed += 1
            sol.metadata["choice_iteration"] = iteration
            sol.metadata["choice_objective_cuts"] = len(cuts)
            sol.metadata["choice_master_time_limit_seconds"] = iteration_time_limit
            sol.metadata["choice_preprocessing_seconds"] = preprocessing_seconds
            if use_choice_utility_hints:
                sol.metadata["choice_utility_hint_cuts"] = hint_cut_count
            if not sol.feasible:
                sol.metadata.update(_budget_metadata(budget, evaluation_seconds))
                solutions.append(sol)
                termination_reason = f"master_{sol.status.lower()}"
                break

            evaluation_start = time.perf_counter()
            evaluated = model.evaluate_with_cuts(problem, sol.assignment)
            evaluation_seconds += time.perf_counter() - evaluation_start
            utility = evaluated.utility
            model_utility = sol.objective
            cuts_to_add = list(evaluated.cuts)
            sol.metadata.update(
                {
                    "choice_model_utility": model_utility,
                    "choice_model_utility_gap": (
                        model_utility - utility if model_utility is not None else None
                    ),
                    "choice_utility": utility,
                    "choice_cuts_added": len(cuts_to_add),
                    "choice_cuts_total": len(cuts) + len(cuts_to_add),
                    **_budget_metadata(budget, evaluation_seconds),
                }
            )
            solutions.append(sol)
            last_feasible = sol
            if utility > best_choice_utility:
                best_choice_utility = utility
                best_choice_solution = sol

            if iteration > 0:
                if model_utility is None or previous_model_utility is None:
                    termination_reason = "missing_objective"
                    break
                model_utility_change = abs(model_utility - previous_model_utility)
                sol.metadata["choice_model_utility_change"] = model_utility_change
                if model_utility_change <= tolerance:
                    termination_reason = "objective_change"
                    break

            previous_model_utility = model_utility

            cuts.extend(cuts_to_add)

            if not cuts_to_add:
                termination_reason = "no_separation"
                break

        if solutions:
            solutions[-1].metadata.update(
                {
                    "choice_iteration_count": iterations_completed,
                    "choice_termination_reason": termination_reason,
                    **_budget_metadata(budget, evaluation_seconds),
                }
            )
        return solutions

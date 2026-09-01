"""Iterative choice strategy.

The strategy solves a zoning model with explicit utility variables, evaluates the
returned zoning against the real choice model, then adds linearized utility cuts
so the next solve has a more accurate objective approximation.
"""

from __future__ import annotations

from choice.objective import ChoiceObjective
from optimization.data.initial_solutions import normalize_hints
from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register
from optimization.strategies.choice_model import (
    get_configured_choice_model,
)


@register("iterative_choice")
class IterativeChoiceStrategy(Strategy):
    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        target = levels[-1]
        max_iterations = int(self.options.get("max_iterations", 5))
        tolerance = float(self.options.get("tolerance", 1e-6))
        apply_hints = normalize_hints(self.options.get("hints", "voronoi")) != "none"
        scale = float(self.options.get("choice_utility_scale", 100.0))
        use_choice_utility_hints = bool(self.options.get("choice_utility_hints", False))
        model = get_configured_choice_model(self.options, dataset.data)

        base_problem = dataset.problem_for(target)
        lower_bound, upper_bound = model.utility_bounds(base_problem)
        cuts = []
        hint_cut_count = 0
        if use_choice_utility_hints:
            cuts.extend(model.choice_utility_hint_cuts(base_problem))
            hint_cut_count = len(cuts)

        solutions: list[ZoneSolution] = []
        best_choice_solution: ZoneSolution | None = None
        best_choice_utility = float("-inf")
        last_feasible: ZoneSolution | None = None
        previous_model_utility: float | None = None

        for iteration in range(max_iterations):
            choice_objective = ChoiceObjective(
                cuts=tuple(cuts),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                scale=scale,
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
            sol = solver.solve(problem)
            sol.metadata["choice_iteration"] = iteration
            sol.metadata["choice_objective_cuts"] = len(cuts)
            if use_choice_utility_hints:
                sol.metadata["choice_utility_hint_cuts"] = hint_cut_count
            if not sol.feasible:
                solutions.append(sol)
                break

            evaluated = model.evaluate_with_cuts(problem, sol.assignment)
            utility = evaluated.utility
            model_utility = sol.objective
            new_cuts = list(evaluated.cuts)
            sol.metadata.update(
                {
                    "choice_model_utility": model_utility,
                    "choice_model_utility_gap": (
                        model_utility - utility if model_utility is not None else None
                    ),
                    "choice_utility": utility,
                    "choice_cuts_added": len(new_cuts),
                    "choice_cuts_total": len(cuts) + len(new_cuts),
                }
            )
            solutions.append(sol)
            last_feasible = sol
            if utility > best_choice_utility:
                best_choice_utility = utility
                best_choice_solution = sol

            if iteration > 0:
                if model_utility is None or previous_model_utility is None:
                    break
                model_utility_change = abs(model_utility - previous_model_utility)
                sol.metadata["choice_model_utility_change"] = model_utility_change
                if model_utility_change <= tolerance:
                    break

            previous_model_utility = model_utility

            cuts.extend(new_cuts)

            if not new_cuts:
                break

        return solutions

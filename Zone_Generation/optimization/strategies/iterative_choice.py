"""Iterative choice strategy.

The strategy solves a zoning model with explicit utility variables, evaluates the
returned zoning against the real choice model, then adds linearized utility cuts
so the next solve has a more accurate objective approximation.
"""

from __future__ import annotations

from Zone_Generation.choice.objective import ChoiceObjective
from Zone_Generation.optimization.data.dataset import Dataset
from Zone_Generation.optimization.levels import LevelSpec
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers.base import Solver
from Zone_Generation.optimization.strategies.base import Strategy, register
from Zone_Generation.optimization.strategies.choice_model import get_choice_model


@register("iterative_choice")
class IterativeChoiceStrategy(Strategy):
    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        levels = [LevelSpec.parse(l) for l in self.options["levels"]]
        target = levels[-1]
        max_iterations = int(self.options.get("max_iterations", 5))
        tolerance = float(self.options.get("tolerance", 1e-6))
        use_hints = bool(self.options.get("use_hints", True))
        scale = float(self.options.get("choice_utility_scale", 100.0))
        model = get_choice_model(
            self.options.get("choice_model", "distance"),
            **self.options.get("choice_model_options", {}),
        )

        base_problem = dataset.problem_for(target)
        lower_bound, upper_bound = model.utility_bounds(base_problem)
        cuts = []

        solutions: list[ZoneSolution] = []
        best: ZoneSolution | None = None
        best_utility = float("-inf")

        for iteration in range(max_iterations):
            choice_objective = ChoiceObjective(
                cuts=tuple(cuts),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                scale=scale,
            )
            hint = best.assignment if best is not None and use_hints else None
            # The first iteration intentionally has no cuts, matching the legacy
            # unconstrained seed. A boundary-minimized seed is a reasonable
            # alternative if we later want a less arbitrary starting zoning.
            problem = dataset.problem_for(
                target,
                hint=hint,
                choice_objective=choice_objective,
            )
            sol = solver.solve(problem)
            sol.metadata["choice_iteration"] = iteration
            sol.metadata["choice_objective_cuts"] = len(cuts)
            if not sol.feasible:
                solutions.append(sol)
                break

            evaluated = model.evaluate_with_cuts(problem, sol.assignment)
            utility = evaluated.utility
            new_cuts = list(evaluated.cuts)
            sol.metadata.update(
                {
                    "choice_utility": utility,
                    "choice_cuts_added": len(new_cuts),
                    "choice_cuts_total": len(cuts) + len(new_cuts),
                }
            )
            solutions.append(sol)

            improved = utility > best_utility + tolerance
            if improved:
                best_utility = utility
                best = sol
            elif iteration > 0:
                break

            cuts.extend(new_cuts)

        return solutions

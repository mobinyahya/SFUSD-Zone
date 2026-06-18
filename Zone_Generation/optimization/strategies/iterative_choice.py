"""Iterative-choice strategy.

Alternates between solving a zoning and evaluating its school-choice utility,
keeping the best zoning found. Each iteration re-solves with the incumbent as a
warm-start hint and relaxes only the candidate zones near zone boundaries, so
the search explores neighboring zonings while staying feasible and contiguous.
Stops when the utility stops improving (by more than ``tolerance``) or after
``max_iterations``.

This is the interchangeable rewrite of the legacy ``iterative_choice`` loop.
The utility evaluation is delegated to a :class:`ChoiceModel`, and the
boundary-relaxation hook is exactly where MNL impact-gradient cuts would attach
in a richer implementation.
"""

from __future__ import annotations

from Zone_Generation.optimization.data import contiguity
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
        radius = self.options.get("boundary_radius", 1)
        model = get_choice_model(
            self.options.get("choice_model", "distance"),
            **self.options.get("choice_model_options", {}),
        )

        centroids = dataset.centroids_for(target)
        G = dataset.graph_for(target)

        solutions: list[ZoneSolution] = []
        best: ZoneSolution | None = None
        best_utility = float("-inf")

        for _ in range(max_iterations):
            if best is None:
                problem = dataset.problem_for(target)
            else:
                candidates = contiguity.boundary_candidates(
                    G, best.assignment, centroids, radius=radius
                )
                problem = dataset.problem_for(
                    target, candidates=candidates, hint=best.assignment
                )

            sol = solver.solve(problem)
            if not sol.feasible and sol.status != "STUB":
                solutions.append(sol)
                break

            utility = model.evaluate(problem, sol.assignment)
            sol.metadata["choice_utility"] = utility
            solutions.append(sol)

            if utility <= best_utility + tolerance:
                # No meaningful improvement; converged.
                if best is not None:
                    break
            if utility > best_utility:
                best_utility, best = utility, sol

        return solutions

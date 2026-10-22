"""Single-shot strategy: build one problem, solve it once."""

from __future__ import annotations

import random

from optimization.data.dataset import Dataset
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register


@register("single")
class SingleShotStrategy(Strategy):
    """Solve only the finest configured level, under its own limits."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        solver = solver.with_options(**self.final_limits(solver))
        problem = self.problem_for(dataset, self.options["levels"][-1])
        self.add_initial_hint(problem, solver)
        limit = self.options.get("enumerated_solutions", -1)
        if limit > 0:
            solutions = solver.enumerate_solutions(problem, limit)
            feasible = [solution for solution in solutions if solution.feasible]
            if not feasible:
                return solutions

            selected = random.Random(self.options.get("seed", 42)).choice(feasible)
            selected_index = next(
                index
                for index, solution in enumerate(solutions)
                if solution is selected
            )
            solutions.append(solutions.pop(selected_index))
            wall_time = selected.metadata["enumeration_wall_time_seconds"]
            for solution in solutions:
                solution.wall_time = wall_time if solution is selected else 0.0
                solution.metadata["enumerated_solution_selected"] = solution is selected
                solution.metadata["enumeration_selection_seed"] = self.options.get(
                    "seed", 42
                )
            return solutions
        return [solver.solve(problem)]

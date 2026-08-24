"""Single-shot strategy: build one problem, solve it once."""

from __future__ import annotations

import random

from optimization.data.dataset import Dataset
from optimization.data.initial_solutions import initial_solution
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.strategies.base import Strategy, register


@register("single")
class SingleShotStrategy(Strategy):
    """Solve only the finest configured level directly."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        levels = [LevelSpec.parse(level) for level in self.options["levels"]]
        target = levels[-1]
        problem = dataset.problem_for(target)
        problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
        _add_math_programming_initial_hint(problem, solver, self.options)
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


def _add_math_programming_initial_hint(problem, solver: Solver, options: dict) -> None:
    if getattr(solver, "name", None) not in {"cp_int", "cp_bool", "mip"}:
        return
    if problem.hint is not None:
        return
    hints = options.get("hints", solver.options.get("hints", "voronoi"))
    initial = initial_solution(problem, hints)
    if initial is not None:
        problem.hint = initial.assignment

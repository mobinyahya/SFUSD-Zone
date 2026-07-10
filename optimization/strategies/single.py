"""Single-shot strategy: build one problem, solve it once."""

from __future__ import annotations

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
        _add_math_programming_initial_hint(problem, solver, self.options)
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

"""Single-shot strategy: build one problem, solve it once."""

from __future__ import annotations

from Zone_Generation.optimization.data.dataset import Dataset
from Zone_Generation.optimization.data.initial_solutions import math_prog_initial_hint
from Zone_Generation.optimization.levels import LevelSpec
from Zone_Generation.optimization.solution import ZoneSolution
from Zone_Generation.optimization.solvers.base import Solver
from Zone_Generation.optimization.strategies.base import Strategy, register


@register("single")
class SingleShotStrategy(Strategy):
    """Solve only the finest configured level directly."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        levels = [LevelSpec.parse(l) for l in self.options["levels"]]
        target = levels[-1]
        problem = dataset.problem_for(target)
        if _use_math_prog_initialization(solver, problem, self.options):
            problem.hint = math_prog_initial_hint(dataset, problem, solver.options)
        return [solver.solve(problem)]


def _use_math_prog_initialization(solver: Solver, problem, options: dict) -> bool:
    if (
        getattr(solver, "name", None) not in {"recom", "relaxed_recom"}
        or problem.hint is not None
    ):
        return False
    method = options.get(
        "initialization_method", solver.options.get("initialization_method", "gerrychain")
    )
    return method == "math_prog"

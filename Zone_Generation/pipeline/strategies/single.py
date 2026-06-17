"""Single-shot strategy: build one problem, solve it once."""

from __future__ import annotations

from Zone_Generation.pipeline.data.dataset import Dataset
from Zone_Generation.pipeline.levels import LevelSpec
from Zone_Generation.pipeline.solution import ZoneSolution
from Zone_Generation.pipeline.solvers.base import Solver
from Zone_Generation.pipeline.strategies.base import Strategy, register


@register("single")
class SingleShotStrategy(Strategy):
    """Solve only the finest configured level directly."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        levels = [LevelSpec.parse(l) for l in self.options["levels"]]
        target = levels[-1]
        problem = dataset.problem_for(target)
        return [solver.solve(problem)]

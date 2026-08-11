"""Direct cumulative-threshold approximation to assignment welfare."""

from __future__ import annotations

from optimization.data.dataset import Dataset
from optimization.solution import ZoneSolution
from optimization.solvers.base import Solver
from optimization.solvers.welfare import ApproximateWelfareSolver
from optimization.strategies.base import Strategy, register
from optimization.strategies.welfare import _build_isolated_welfare_problem


@register("approximate_welfare")
class ApproximateWelfareStrategy(Strategy):
    """Jointly optimize feasible zoning and cumulative assignment welfare."""

    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        problem = _build_isolated_welfare_problem(
            dataset,
            solver,
            self.options,
            strategy_name=self.name,
        )
        return [
            ApproximateWelfareSolver(
                solver,
                utility_scale=int(self.options["welfare_utility_scale"]),
            ).solve(problem)
        ]

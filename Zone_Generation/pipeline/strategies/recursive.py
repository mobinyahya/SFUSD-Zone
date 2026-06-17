"""Recursive strategy: solve coarse, then refine at each finer level.

For each level in ``levels`` (coarse->fine, any mix of Block/BlockGroup depths)
the previous solution is projected onto the current graph via
:class:`LevelConverter`, used as a warm-start hint, and used to narrow each
node's candidate zones to those seen near zone boundaries -- so the finer solve
only re-decides the borders. Per-level time and gap limits are applied to the
solver before each solve.
"""

from __future__ import annotations

from Zone_Generation.pipeline.data import contiguity
from Zone_Generation.pipeline.data.conversion import LevelConverter
from Zone_Generation.pipeline.data.dataset import Dataset
from Zone_Generation.pipeline.levels import LevelSpec
from Zone_Generation.pipeline.solution import ZoneSolution
from Zone_Generation.pipeline.solvers.base import Solver
from Zone_Generation.pipeline.strategies.base import Strategy, register


@register("recursive")
class RecursiveStrategy(Strategy):
    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]:
        levels = [LevelSpec.parse(l) for l in self.options["levels"]]
        time_limits = self.options.get("solve_time_limits")
        gap_limits = self.options.get("gap_limits")
        use_hints = self.options.get("use_hints", True)
        radius = self.options.get("boundary_radius", 1)
        converter = LevelConverter()

        solutions: list[ZoneSolution] = []
        prev: ZoneSolution | None = None
        prev_level: LevelSpec | None = None

        for i, level in enumerate(levels):
            self._apply_limits(solver, time_limits, gap_limits, i)

            if prev is None:
                problem = dataset.problem_for(level)
            else:
                dst_G = dataset.graph_for(level)
                centroids = dataset.centroids_for(level)
                projected = converter.between(
                    prev.problem.G, prev.assignment, prev_level, dst_G, level
                )
                candidates = contiguity.boundary_candidates(
                    dst_G, projected, centroids, radius=radius
                )
                problem = dataset.problem_for(
                    level,
                    candidates=candidates,
                    hint=projected if use_hints else None,
                )

            sol = solver.solve(problem)
            solutions.append(sol)
            prev, prev_level = sol, level

        return solutions

    @staticmethod
    def _apply_limits(solver, time_limits, gap_limits, i):
        if time_limits and i < len(time_limits):
            solver.options["solve_time_limit"] = time_limits[i]
        if gap_limits and i < len(gap_limits):
            solver.options["relative_gap_limit"] = gap_limits[i]

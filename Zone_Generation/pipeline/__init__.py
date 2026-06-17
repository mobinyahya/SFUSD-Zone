"""SFUSD zoning optimization pipeline.

A three-layer, fully interchangeable architecture:

    Data layer      -> Solver layer      -> Strategy layer
    (Dataset)          (Solver.solve)       (Strategy.run)

The layers communicate through two solver-agnostic contracts defined in this
package: :class:`~Zone_Generation.pipeline.problem.ZoneProblem` (a single
optimization instance) and
:class:`~Zone_Generation.pipeline.solution.ZoneSolution` (its result).

Nothing here depends on the legacy ``Zone_Generation.Optimization`` package or
on the benchmark/website consumers; it is a standalone replacement that can be
migrated onto gradually.
"""

from Zone_Generation.pipeline.levels import LevelSpec
from Zone_Generation.pipeline.problem import ZoneProblem
from Zone_Generation.pipeline.solution import ZoneSolution

__all__ = ["LevelSpec", "ZoneProblem", "ZoneSolution"]

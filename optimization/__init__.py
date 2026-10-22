"""SFUSD zoning optimization optimization.

A three-layer, fully interchangeable architecture:

    Data layer      -> Solver layer      -> Strategy layer
    (Dataset)          (Solver.solve)       (Strategy.run)

The layers communicate through two solver-agnostic contracts defined in this
package: :class:`~optimization.problem.ZoneProblem` (a single
optimization instance) and
:class:`~optimization.solution.ZoneSolution` (its result).

Nothing here depends on the legacy optimization package or
on benchmark consumers; it is a standalone replacement that can be migrated onto
gradually.
"""

from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution

__all__ = ["LevelSpec", "ZoneProblem", "ZoneSolution"]

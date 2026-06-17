"""Strategy layer: interchangeable orchestration around a solver.

A strategy decides *what problems to solve and in what order*, delegating the
actual solving to any :class:`~Zone_Generation.pipeline.solvers.base.Solver`.
Single-shot, recursive (coarse->fine) and iterative-choice strategies all
implement the same :class:`~Zone_Generation.pipeline.strategies.base.Strategy`
interface and register themselves by name.
"""

from Zone_Generation.pipeline.strategies.base import Strategy, get_strategy, register

from Zone_Generation.pipeline.strategies import (  # noqa: E402,F401
    single,
    recursive,
    iterative_choice,
)

__all__ = ["Strategy", "get_strategy", "register"]

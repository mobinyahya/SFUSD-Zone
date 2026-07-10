"""Solver layer: interchangeable algorithms that solve a ZoneProblem.

Every solver implements :class:`~optimization.solvers.base.Solver`
(``solve(problem) -> ZoneSolution``) and registers itself by name so the config
layer can build one from a string. Importing this package registers the
built-in implementations.
"""

from optimization.solvers.base import Solver, get_solver, register

# Importing the modules triggers their @register decorators.
from optimization.solvers import (  # noqa: E402,F401
    cpsat,
    local_search,
    recom,
    relaxed_recom,
    short_bursts_recom,
)

# Gurobi is optional; only register it if the package is importable.
try:  # pragma: no cover - depends on optional dependency
    from optimization.solvers import mip  # noqa: F401
except ImportError:
    pass

__all__ = ["Solver", "get_solver", "register"]

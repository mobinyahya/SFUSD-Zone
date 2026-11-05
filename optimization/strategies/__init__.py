"""Strategy layer: interchangeable orchestration around a solver.

A strategy decides *what problems to solve and in what order*, delegating the
actual solving to any :class:`~optimization.solvers.base.Solver`. Single-level
and coarse-to-fine (``single``/``recursive``) runs are pipelines of solve steps
(:mod:`optimization.strategies.pipeline`); the iterative-choice and welfare
strategies implement the same :class:`~optimization.strategies.base.Strategy`
interface and register themselves by name.
"""

from optimization.strategies.base import (
    Strategy,
    get_strategy,
    register,
)

from optimization.strategies import (  # noqa: E402,F401
    iterative_choice,
    mid,
    mid_decomp,
    saa,
    direct_samples,
    priced_access,
    stable_cutoff,
    dantzig_wolfe,
    pipeline,
)

__all__ = ["Strategy", "get_strategy", "register"]

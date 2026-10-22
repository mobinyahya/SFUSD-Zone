"""Strategy interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from optimization.data.dataset import Dataset
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.solvers.base import Solver
from optimization.solution import ZoneSolution
from optimization.strategies.budget import final_value


class Strategy(ABC):
    """Orchestrates one or more solves over a :class:`Dataset`.

    ``run`` returns the list of solutions produced, finest level last (so
    ``run(...)[-1]`` is the primary result).

    A strategy never mutates the solver it is handed: a solve that needs its
    own limits gets a :meth:`Solver.with_options` copy.
    """

    name: str = "strategy"
    #: Whether ``run`` builds the assignment-side market (``build_mid_market``
    #: or ``build_saa_market``), so a benchmark task id must hash its inputs.
    reads_assignment_market: bool = False
    #: Whether the aggregate capacity rows (``overage``/``shortage``) stay on.
    #: The welfare strategies turn them off; their module docstrings say why.
    capacity_balance: bool = True

    def __init__(self, **options):
        self.options = options

    @abstractmethod
    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]: ...

    def problem_for(self, dataset: Dataset, level, **kwargs) -> ZoneProblem:
        """``dataset.problem_for`` with this strategy's feasible set applied.

        Every problem a strategy solves comes from here, and so do the problems
        ``analysis/warm_feasible_hints.py`` warms hints for, so a warmed hint is
        keyed on exactly the feasible set the strategy will ask for.
        """

        return self.configure_problem(
            dataset.problem_for(LevelSpec.parse(level), **kwargs)
        )

    def configure_problem(self, problem: ZoneProblem) -> ZoneProblem:
        """Apply ``boundary_prop`` and, if off, the capacity band to ``problem``."""

        problem.boundary_prop = float(self.options.get("boundary_prop", -1.0))
        if not self.capacity_balance:
            problem.overage = -1.0
            problem.shortage = -1.0
        return problem

    def final_limits(self, solver: Solver) -> dict[str, float]:
        """The last configured time and gap limits, for a single final solve."""

        return {
            "solve_time_limit": final_value(
                self.options.get("solve_time_limits"),
                solver.options.get("solve_time_limit", 60.0),
            ),
            "relative_gap_limit": final_value(
                self.options.get("gap_limits"),
                solver.options.get("relative_gap_limit", 0.0),
            ),
        }

    def add_initial_hint(self, problem: ZoneProblem, solver: Solver) -> None:
        """Warm-start a CP/MIP solve with the configured ``hints`` kind.

        A hint the problem already carries -- a projected coarser level, the
        best zoning so far -- is kept.
        """

        if solver.name not in HINTED_SOLVERS or problem.hint is not None:
            return
        problem.hint = solver.hint_assignment(
            problem, self.options.get("hints", solver.options.get("hints", "voronoi"))
        )


# The solvers whose model takes a complete-assignment warm start.
HINTED_SOLVERS = frozenset({"cp_int", "cp_bool", "mip"})


_REGISTRY: dict[str, type[Strategy]] = {}


def register(name: str):
    def deco(cls: type[Strategy]) -> type[Strategy]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_strategy(name: str, **options) -> Strategy:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy {name!r}. Registered: {sorted(_REGISTRY)}.")
    return _REGISTRY[name](**options)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)

"""Strategy interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from Zone_Generation.optimization.data.dataset import Dataset
from Zone_Generation.optimization.solvers.base import Solver
from Zone_Generation.optimization.solution import ZoneSolution


class Strategy(ABC):
    """Orchestrates one or more solves over a :class:`Dataset`.

    ``run`` returns the list of solutions produced, finest level last (so
    ``run(...)[-1]`` is the primary result).
    """

    name: str = "strategy"

    def __init__(self, **options):
        self.options = options

    @abstractmethod
    def run(self, dataset: Dataset, solver: Solver) -> list[ZoneSolution]: ...


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

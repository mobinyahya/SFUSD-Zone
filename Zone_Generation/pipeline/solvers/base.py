"""Solver interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from Zone_Generation.pipeline.problem import ZoneProblem
from Zone_Generation.pipeline.solution import ZoneSolution


class Solver(ABC):
    """Maps a :class:`ZoneProblem` to a :class:`ZoneSolution`.

    Implementations read *only* from the problem (graph, centroids, params,
    candidates, hint). They must honor ``problem.candidate_zones(node)``, fix
    each centroid to its zone, and produce a contiguous assignment.
    """

    name: str = "solver"

    def __init__(self, **options):
        self.options = options

    @abstractmethod
    def solve(self, problem: ZoneProblem) -> ZoneSolution:
        ...


_REGISTRY: dict[str, type[Solver]] = {}


def register(name: str):
    """Class decorator registering a solver under ``name``."""

    def deco(cls: type[Solver]) -> type[Solver]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_solver(name: str, **options) -> Solver:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown solver {name!r}. Registered: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name](**options)


def available_solvers() -> list[str]:
    return sorted(_REGISTRY)

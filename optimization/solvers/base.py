"""Solver interface and registry."""

from __future__ import annotations

import copy
import os
import re
from abc import ABC, abstractmethod

import networkx as nx

from optimization.data.initial_solutions import initial_solution
from optimization.progress import SolverProgressTracker
from optimization.problem import ZoneProblem
from optimization.solution import ZoneSolution


class Solver(ABC):
    """Maps a :class:`ZoneProblem` to a :class:`ZoneSolution`.

    Implementations read *only* from the problem (graph, centroids, params,
    candidates, hint). They must honor ``problem.candidate_zones(node)``, fix
    each centroid to its zone, and produce a contiguous assignment.
    """

    name: str = "solver"

    def __init__(self, **options):
        self.options = options
        # Shared with every ``with_options`` copy, so their log and progress
        # files keep numbering on from this solver's instead of colliding.
        self._counts = {"solve": 0, "progress": 0}

    @abstractmethod
    def solve(self, problem: ZoneProblem) -> ZoneSolution: ...

    def with_options(self, **overrides) -> "Solver":
        """A copy of this solver with ``overrides`` applied; ``self`` is untouched.

        This is how a strategy gives one solve its own time or gap limit: the
        solver it was handed is shared, so it is never mutated.
        """

        solver = copy.copy(self)
        solver.options = {**self.options, **overrides}
        return solver

    def hint_assignment(
        self, problem: ZoneProblem, hints: str | None = None
    ) -> dict[int, int] | None:
        """``problem.hint``, else a fresh initial solution of kind ``hints``.

        ``hints`` defaults to this solver's own ``hints`` option; with neither,
        there is no hint.
        """

        if problem.hint:
            return problem.hint
        hints = self.options.get("hints") if hints is None else hints
        if hints is None:
            return None
        initial = initial_solution(problem, hints, solver_options=self.options)
        return initial.assignment if initial is not None else None

    @property
    def _solve_count(self) -> int:
        return self._counts["solve"]

    @_solve_count.setter
    def _solve_count(self, value: int) -> None:
        self._counts["solve"] = value

    @property
    def _progress_count(self) -> int:
        return self._counts["progress"]

    @_progress_count.setter
    def _progress_count(self, value: int) -> None:
        self._counts["progress"] = value

    def enumerate_solutions(
        self, problem: ZoneProblem, limit: int
    ) -> list[ZoneSolution]:
        raise ValueError(f"{self.name} does not support solution enumeration.")

    def _centroid_neighbor_radius(self) -> int:
        return checked_centroid_neighbor_radius(
            self.options.get("centroid_neighbor_radius", 0)
        )

    def _centroid_neighborhoods(self, problem: ZoneProblem) -> dict[int, set[int]]:
        return centroid_neighborhoods(problem, self._centroid_neighbor_radius())

    def _next_solver_log_path(
        self, problem: ZoneProblem, *, suffix: str = ".log"
    ) -> str | None:
        if not self.options.get("save_solver_logs"):
            return None
        log_dir = self.options.get("solver_log_dir")
        if not log_dir:
            output_dir = self.options.get("output_dir")
            if not output_dir:
                return None
            log_dir = os.path.join(str(output_dir), "solver_logs")
        log_dir = os.path.expanduser(str(log_dir))
        os.makedirs(log_dir, exist_ok=True)
        level_name = getattr(getattr(problem, "level", None), "name", "unknown_level")
        filename = _safe_filename(
            f"solver_{self._solve_count:02d}_{level_name}_{self.name}{suffix}"
        )
        self._solve_count += 1
        return os.path.join(log_dir, filename)

    def _solver_log_metadata(
        self, log_path: str | None, *, log_format: str | None = None
    ) -> dict[str, str]:
        if not log_path:
            return {}
        output_dir = self.options.get("output_dir")
        if output_dir:
            display_path = os.path.relpath(
                log_path, os.path.expanduser(str(output_dir))
            )
        else:
            display_path = log_path
        metadata = {"solver_log_path": display_path}
        if log_format:
            metadata["solver_log_format"] = log_format
        return metadata

    def _new_solver_progress_tracker(
        self,
        problem: ZoneProblem,
        *,
        maximize: bool = False,
        objective_scale: float = 1.0,
    ) -> SolverProgressTracker | None:
        if not self.options.get("save_solver_progress"):
            return None
        level_name = getattr(getattr(problem, "level", None), "name", "unknown_level")
        progress_id = _safe_filename(
            f"solver_{self._progress_count:02d}_{level_name}_{self.name}"
        )
        self._progress_count += 1
        return SolverProgressTracker(
            progress_id=progress_id,
            maximize=maximize,
            objective_scale=objective_scale,
        )

    def _solver_progress_metadata(
        self, progress: SolverProgressTracker | None
    ) -> dict[str, object]:
        if progress is None:
            return {}
        return {
            "solver_progress_enabled": True,
            "solver_progress_id": progress.progress_id,
            "solver_progress_count": len(progress.entries),
        }


def checked_centroid_neighbor_radius(radius) -> int:
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("centroid_neighbor_radius must be a non-negative integer.")
    return radius


def centroid_neighborhoods(problem: ZoneProblem, radius: int) -> dict[int, set[int]]:
    """Each zone's nodes within ``radius`` hops of its centroid, owned outright."""

    radius = checked_centroid_neighbor_radius(radius)
    return {
        zone: set(
            nx.single_source_shortest_path_length(problem.G, centroid, cutoff=radius)
        )
        for zone, centroid in enumerate(problem.centroids)
    }


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "solver.log"


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
        raise ValueError(f"Unknown solver {name!r}. Registered: {sorted(_REGISTRY)}.")
    return _REGISTRY[name](**options)


def available_solvers() -> list[str]:
    return sorted(_REGISTRY)

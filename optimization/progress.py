"""Lightweight solver incumbent-progress capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class SolverProgressEntry:
    """One improving incumbent, stored in node order."""

    objective: float
    elapsed_seconds: float
    assignment: tuple[int, ...]
    iteration: int | None = None


@dataclass(slots=True)
class SolverProgressTracker:
    """Append-only incumbent tracker with objective-sense filtering."""

    progress_id: str
    maximize: bool = False
    objective_scale: float = 1.0
    tolerance: float = 1e-9
    entries: list[SolverProgressEntry] = field(default_factory=list)
    best_objective: float | None = None

    def scaled_objective(self, objective: float) -> float:
        return float(objective) / float(self.objective_scale)

    def is_improvement(self, objective: float) -> bool:
        value = self.scaled_objective(objective)
        if self.best_objective is None:
            return True
        if self.maximize:
            return value > self.best_objective + self.tolerance
        return value < self.best_objective - self.tolerance

    def add(
        self,
        objective: float,
        elapsed_seconds: float,
        assignment: Iterable[int],
        *,
        iteration: int | None = None,
    ) -> bool:
        value = self.scaled_objective(objective)
        if self.best_objective is not None:
            if self.maximize and value <= self.best_objective + self.tolerance:
                return False
            if not self.maximize and value >= self.best_objective - self.tolerance:
                return False
        self.entries.append(
            SolverProgressEntry(
                objective=value,
                elapsed_seconds=float(elapsed_seconds),
                assignment=tuple(int(zone) for zone in assignment),
                iteration=iteration,
            )
        )
        self.best_objective = value
        return True


def assignment_tuple(
    nodes: Iterable[int], assignment: Mapping[int, int]
) -> tuple[int, ...]:
    return tuple(int(assignment[node]) for node in nodes)

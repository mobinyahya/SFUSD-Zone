"""Solver-agnostic choice objective primitives."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChoiceTerm:
    """One coefficient times an assignment indicator ``x[zone, node]``."""

    coefficient: float
    zone: int
    node: int


@dataclass(frozen=True)
class ChoiceCut:
    """Conditional upper bound for one node utility variable.

    If ``node`` is assigned to ``zone``, then ``u_node`` is constrained by
    ``constant + sum(coefficient * x[term.zone, term.node])``.
    """

    node: int
    zone: int
    constant: float
    terms: tuple[ChoiceTerm, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ChoiceObjective:
    """Choice utility objective data consumed by swappable solvers."""

    cuts: tuple[ChoiceCut, ...] = field(default_factory=tuple)
    lower_bound: float = -1_000_000_000.0
    upper_bound: float = 1_000_000_000.0
    scale: float = 100.0


@dataclass(frozen=True)
class ChoiceEvaluation:
    """Real utility evaluation plus cuts linearized around that solution."""

    utility: float
    cuts: tuple[ChoiceCut, ...] = field(default_factory=tuple)

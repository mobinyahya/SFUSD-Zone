"""Solver-agnostic choice objective primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


AccessPair = tuple[int, int]


@dataclass(frozen=True)
class ChoiceTerm:
    """One coefficient times a pairwise access indicator ``a[student, school]``."""

    coefficient: float
    node: int
    student_node: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient):
            raise ValueError("Choice term coefficients must be finite.")
        if isinstance(self.node, bool) or not isinstance(self.node, int):
            raise ValueError("Choice term school nodes must be integers.")
        if self.student_node is not None and (
            isinstance(self.student_node, bool)
            or not isinstance(self.student_node, int)
        ):
            raise ValueError("Choice term student nodes must be integers.")


@dataclass(frozen=True)
class ChoiceCut:
    """Upper bound for node utility (if ``node`` is set) or total utility (if ``node is None``).

    Constrained by ``constant + sum(coefficient * a[student, school])``.
    """

    node: int | None = None
    constant: float = 0.0
    terms: tuple[ChoiceTerm, ...] = field(default_factory=tuple)
    anchor_access: tuple[tuple[AccessPair, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.node is not None and (
            isinstance(self.node, bool) or not isinstance(self.node, int)
        ):
            raise ValueError("Choice cut nodes must be integers.")
        if not math.isfinite(self.constant):
            raise ValueError("Choice cut constants must be finite.")
        if not isinstance(self.terms, tuple) or not all(
            isinstance(term, ChoiceTerm) for term in self.terms
        ):
            raise ValueError("Choice cut terms must be a tuple of ChoiceTerm values.")

        aggregate = self.node is None
        if aggregate and any(term.student_node is None for term in self.terms):
            raise ValueError(
                "Aggregate choice cuts require student_node on every term."
            )
        if not aggregate and any(term.student_node is not None for term in self.terms):
            raise ValueError(
                "Per-node choice cuts must infer the student from ChoiceCut.node."
            )

        seen: set[AccessPair] = set()
        term_pairs = {
            (
                term.student_node if term.student_node is not None else self.node,
                term.node,
            )
            for term in self.terms
        }
        for pair, value in self.anchor_access:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(
                    isinstance(node, bool) or not isinstance(node, int) for node in pair
                )
            ):
                raise ValueError("Choice cut anchor keys must be integer node pairs.")
            if pair in seen:
                raise ValueError("Choice cut anchor keys must be unique.")
            if isinstance(value, bool) or value not in {0, 1}:
                raise ValueError("Choice cut anchor values must be 0 or 1.")
            if pair not in term_pairs:
                raise ValueError("Choice cut anchors must correspond to cut terms.")
            seen.add(pair)


@dataclass(frozen=True)
class ChoiceObjective:
    """Choice utility objective data consumed by swappable solvers."""

    cuts: tuple[ChoiceCut, ...] = field(default_factory=tuple)
    lower_bound: float = -1_000_000_000.0
    upper_bound: float = 1_000_000_000.0
    scale: float = 100.0
    aggregate_cuts: bool = False
    total_lower_bound: float | None = None
    total_upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cuts, tuple) or not all(
            isinstance(cut, ChoiceCut) for cut in self.cuts
        ):
            raise ValueError(
                "Choice objective cuts must be a tuple of ChoiceCut values."
            )
        if not isinstance(self.aggregate_cuts, bool):
            raise ValueError("aggregate_cuts must be a Boolean.")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("choice_utility_scale must be a positive finite value.")
        if not math.isfinite(self.lower_bound) or not math.isfinite(self.upper_bound):
            raise ValueError("Choice utility bounds must be finite.")
        if self.lower_bound > self.upper_bound:
            raise ValueError("Choice utility lower_bound exceeds upper_bound.")

        if self.aggregate_cuts:
            if any(cut.node is not None for cut in self.cuts):
                raise ValueError("Aggregate choice objectives require aggregate cuts.")
        else:
            if any(cut.node is None for cut in self.cuts):
                raise ValueError("Per-node choice objectives require per-node cuts.")
            if self.total_lower_bound is not None or self.total_upper_bound is not None:
                raise ValueError(
                    "Per-node choice objectives cannot override total utility bounds."
                )

        for value in (self.total_lower_bound, self.total_upper_bound):
            if value is not None and not math.isfinite(value):
                raise ValueError("Choice total utility bounds must be finite.")
        if (
            self.total_lower_bound is not None
            and self.total_upper_bound is not None
            and self.total_lower_bound > self.total_upper_bound
        ):
            raise ValueError("Choice total lower bound exceeds upper bound.")


@dataclass(frozen=True)
class ChoiceEvaluation:
    """Real utility evaluation plus cuts linearized around that solution."""

    utility: float
    cuts: tuple[ChoiceCut, ...] = field(default_factory=tuple)

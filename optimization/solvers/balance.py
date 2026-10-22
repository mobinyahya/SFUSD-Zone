"""Shared balance constraint definitions for solver backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from optimization.problem import ZoneProblem


@dataclass(frozen=True)
class BalanceConstraint:
    """A zone-level ratio bound of ``value / students``."""

    kind: str
    value: Callable[[int], float]
    lower_ratio: float | None
    upper_ratio: float | None


Term = tuple[float, int, int]


def balance_constraints(problem: ZoneProblem) -> list[BalanceConstraint]:
    """Return all capacity and diversity balance constraints for a problem."""

    constraints = [
        BalanceConstraint(
            kind="frl",
            value=problem.frl,
            lower_ratio=(
                problem.district_frl - problem.frl_dev if problem.frl_dev >= 0 else None
            ),
            upper_ratio=(
                problem.district_frl + problem.frl_dev if problem.frl_dev >= 0 else None
            ),
        ),
    ]

    if problem.shortage >= 0 or problem.overage >= 0:
        constraints.insert(
            0,
            BalanceConstraint(
                kind="capacity",
                value=problem.capacity,
                lower_ratio=1.0 - problem.shortage if problem.shortage >= 0 else None,
                upper_ratio=1.0 + problem.overage if problem.overage >= 0 else None,
            ),
        )

    if problem.racial_dev >= 0:
        racial = problem.district_racial
        for ethnicity in problem.ethnicities:
            constraints.append(
                BalanceConstraint(
                    kind="racial",
                    value=lambda node, e=ethnicity: problem.ethnicity(node, e),
                    lower_ratio=racial[ethnicity] - problem.racial_dev,
                    upper_ratio=racial[ethnicity] + problem.racial_dev,
                )
            )

    return constraints


def balance_terms(
    problem: ZoneProblem,
    constraint: BalanceConstraint,
    zone: int,
    nodes: list[int],
) -> tuple[list[Term], list[Term]]:
    """Build lower and upper linear terms for one zone balance constraint."""

    lower = (
        [
            (
                constraint.value(node)
                - constraint.lower_ratio * problem.students(node),
                zone,
                node,
            )
            for node in nodes
        ]
        if constraint.lower_ratio is not None
        else []
    )
    upper = (
        [
            (
                constraint.value(node)
                - constraint.upper_ratio * problem.students(node),
                zone,
                node,
            )
            for node in nodes
        ]
        if constraint.upper_ratio is not None
        else []
    )
    return lower, upper

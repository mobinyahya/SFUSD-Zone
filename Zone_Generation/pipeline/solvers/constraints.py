"""Constraint definitions -- the single source of truth.

Each constraint is written once here in terms of a small
:class:`ModelBackend` interface of linear primitives over the assignment
variables ``x[z][i]`` (1 iff node ``i`` is in zone ``z``). The CP-SAT and Gurobi
backends each implement those primitives, so the *mathematics* lives in exactly
one place -- fixing the legacy problem where the same constraints were
duplicated (and drifted) across three solvers.

All coefficients are plain floats. Backends that need integer coefficients
(CP-SAT) scale internally; Gurobi consumes the floats directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from Zone_Generation.pipeline.data import contiguity
from Zone_Generation.pipeline.problem import ZoneProblem

# A term is (coefficient, zone, node), referencing coefficient * x[zone][node].
Term = tuple[float, int, int]


class ModelBackend(ABC):
    """Linear-model primitives a solver must provide for the shared constraints."""

    @abstractmethod
    def add_exactly_one(self, choices: list[tuple[int, int]]) -> None:
        """Exactly one of the given ``(zone, node)`` assignment vars is 1."""

    @abstractmethod
    def add_linear(self, terms: list[Term], sense: str, rhs: float) -> None:
        """``sum(coef * x[z][i]) <sense> rhs`` with ``sense`` in ``<=,>=,==``."""

    @abstractmethod
    def fix(self, zone: int, node: int) -> None:
        """Force ``x[zone][node] == 1``."""

    @abstractmethod
    def forbid(self, zone: int, node: int) -> None:
        """Force ``x[zone][node] == 0``."""


# ====================================================================== #
# Individual constraints
# ====================================================================== #
def assignment(problem: ZoneProblem, backend: ModelBackend) -> None:
    """Each node belongs to exactly one of its candidate zones."""
    for node in problem.nodes:
        choices = [(z, node) for z in problem.candidate_zones(node)]
        if not choices:
            raise ValueError(f"Node {node} has no candidate zones (infeasible).")
        backend.add_exactly_one(choices)


def centroids(problem: ZoneProblem, backend: ModelBackend) -> None:
    """Each centroid anchors its own zone."""
    for z, centroid in enumerate(problem.centroids):
        backend.fix(z, centroid)


def contiguity_constraints(problem: ZoneProblem, backend: ModelBackend) -> None:
    """Strict contiguity via the shortest-path-tree support relation."""
    supports = contiguity.closer_supports(
        problem.G, problem.centroids, problem.candidate_zones
    )
    for (node, z), support_nodes in supports.items():
        if not support_nodes:
            # No strictly-closer candidate neighbor -> cannot be reached.
            backend.forbid(z, node)
            continue
        # x[z][node] <= sum(x[z][n] for n in support_nodes)
        terms: list[Term] = [(1.0, z, node)]
        terms += [(-1.0, z, n) for n in support_nodes]
        backend.add_linear(terms, "<=", 0.0)


def capacity(problem: ZoneProblem, backend: ModelBackend) -> None:
    """Per-zone seats within ``[(1-shortage), (1+overage)]`` x students."""
    lo = 1.0 - problem.shortage
    hi = 1.0 + problem.overage
    for z in range(problem.Z):
        nodes = [n for n in problem.nodes if z in problem.candidate_zones(n)]
        # seats - lo*students >= 0
        ge = [
            (problem.capacity(n) - lo * problem.students(n), z, n) for n in nodes
        ]
        backend.add_linear(ge, ">=", 0.0)
        # seats - hi*students <= 0
        le = [
            (problem.capacity(n) - hi * problem.students(n), z, n) for n in nodes
        ]
        backend.add_linear(le, "<=", 0.0)


def diversity(problem: ZoneProblem, backend: ModelBackend) -> None:
    """Per-zone FRL and per-ethnicity proportions within their deviations."""

    def balance(value_fn, ratio: float, dev: float) -> None:
        for z in range(problem.Z):
            nodes = [n for n in problem.nodes if z in problem.candidate_zones(n)]
            # value - (ratio+dev)*students <= 0
            upper = [
                (value_fn(n) - (ratio + dev) * problem.students(n), z, n)
                for n in nodes
            ]
            backend.add_linear(upper, "<=", 0.0)
            # value - (ratio-dev)*students >= 0
            lower = [
                (value_fn(n) - (ratio - dev) * problem.students(n), z, n)
                for n in nodes
            ]
            backend.add_linear(lower, ">=", 0.0)

    balance(problem.frl, problem.district_frl, problem.frl_dev)
    racial = problem.district_racial
    for eth in problem.ethnicities:
        balance(lambda n, e=eth: problem.ethnicity(n, e), racial[eth], problem.racial_dev)


def school_count(problem: ZoneProblem, backend: ModelBackend) -> None:
    """Each zone gets roughly the average number of schools (+/- 1)."""
    total = sum(problem.num_schools(n) for n in problem.nodes)
    if total == 0:
        return
    avg = total / problem.Z
    for z in range(problem.Z):
        nodes = [n for n in problem.nodes if z in problem.candidate_zones(n)]
        terms = [(float(problem.num_schools(n)), z, n) for n in nodes]
        backend.add_linear(terms, ">=", max(0.0, avg - 1.0))
        backend.add_linear(terms, "<=", avg + 1.0)


# ====================================================================== #
# Aggregate
# ====================================================================== #
def add_all(problem: ZoneProblem, backend: ModelBackend) -> None:
    """Add the full constraint set in dependency order."""
    assignment(problem, backend)
    centroids(problem, backend)
    contiguity_constraints(problem, backend)
    capacity(problem, backend)
    diversity(problem, backend)
    school_count(problem, backend)

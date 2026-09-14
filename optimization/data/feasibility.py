"""Exact feasibility check for a complete zoning, independent of any solver.

Each backend decides feasibility in its own arithmetic: CP-SAT reasons about
integer-scaled coefficients, ReCom keeps running float sums against a 1e-6
tolerance, and the MIP backends inherit the LP feasibility tolerance. So a
zoning one backend calls feasible is not automatically feasible to another, and
"is this assignment feasible?" is a question no solver can answer on another's
behalf.

That is exactly the question a shared warm start needs answered. This module
answers it once, in exact float arithmetic, reading only the problem. It is
deliberately outside every solver's search loop -- it is a gate for assignments
crossing a trust boundary (cache write, cache read, hand-authored zonings),
where being slow and right beats being fast.

Every check here is implied by the constraints the solvers post, never stricter:

* coverage -- one zone per problem node, in ``range(problem.Z)``
* candidates -- ``problem.candidate_zones``, which already folds in the
  centroid anchors, explicit per-node candidates, ``fixed``, and
  ``max_distance``
* contiguity -- each zone induces one connected component. The CP-SAT
  closer-neighbour support formulation is strictly stronger than this, so a
  CP-SAT solution always passes; plain connectivity is what every downstream
  consumer actually requires.
* balance -- the ``frl`` / ``capacity`` / ``racial`` rows of
  :func:`optimization.solvers.balance.balance_constraints`
* school count -- every zone within one school of the district average
* boundary -- cut edges within ``boundary_prop`` of the graph's edge count,
  counted unweighted, as :meth:`_add_boundary_constraint` posts it

Solver *options* that shrink the feasible set further -- ``centroid_neighbor_
radius`` is the only one today -- are invisible here, because they are not
properties of the problem. Callers that set them own that part of the contract;
the feasible-hint cache does so through its key.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from optimization.data import contiguity
from optimization.problem import ZoneProblem

# Matches ``optimization.solvers.recom._EPS``, the tightest consumer tolerance.
FEASIBILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FeasibilityViolation:
    """One broken constraint, with the amount it is broken by."""

    kind: str
    detail: str
    amount: float = 0.0

    def __str__(self) -> str:
        if self.amount:
            return f"{self.kind}: {self.detail} (by {self.amount:.6g})"
        return f"{self.kind}: {self.detail}"


@dataclass(frozen=True)
class FeasibilityReport:
    """The verdict for one assignment, plus every violation behind it."""

    violations: tuple[FeasibilityViolation, ...] = ()
    tolerance: float = FEASIBILITY_TOLERANCE

    @property
    def feasible(self) -> bool:
        return not self.violations

    def describe(self, limit: int = 6) -> str:
        if not self.violations:
            return "feasible"
        shown = [str(violation) for violation in self.violations[:limit]]
        if len(self.violations) > limit:
            shown.append(f"... and {len(self.violations) - limit} more")
        return "; ".join(shown)


def check_zoning(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
    *,
    tolerance: float = FEASIBILITY_TOLERANCE,
) -> FeasibilityReport:
    """Report every way ``assignment`` fails to be a feasible zoning."""

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Feasibility tolerance must be non-negative and finite.")

    structural = _structural_violations(problem, assignment)
    if structural:
        # The numeric checks below would report noise on top of a broken
        # assignment, so stop at the structure.
        return FeasibilityReport(tuple(structural), tolerance)

    zones = [[] for _ in range(problem.Z)]
    for node, zone in assignment.items():
        zones[int(zone)].append(int(node))

    violations: list[FeasibilityViolation] = []
    violations.extend(_contiguity_violations(problem, assignment))
    violations.extend(_balance_violations(problem, zones, tolerance))
    violations.extend(_school_count_violations(problem, zones, tolerance))
    violations.extend(_boundary_violations(problem, assignment))
    return FeasibilityReport(tuple(violations), tolerance)


def assert_feasible(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
    *,
    tolerance: float = FEASIBILITY_TOLERANCE,
    label: str = "zoning",
) -> None:
    """Raise :class:`InfeasibleZoningError` unless ``assignment`` is feasible."""

    report = check_zoning(problem, assignment, tolerance=tolerance)
    if not report.feasible:
        raise InfeasibleZoningError(label, report)


class InfeasibleZoningError(ValueError):
    """An assignment failed the exact feasibility check."""

    def __init__(self, label: str, report: FeasibilityReport):
        self.report = report
        super().__init__(f"{label} is not feasible -- {report.describe()}")


# ---------------------------------------------------------------------- #
# individual checks
# ---------------------------------------------------------------------- #
def _structural_violations(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
) -> list[FeasibilityViolation]:
    violations: list[FeasibilityViolation] = []
    nodes = set(problem.nodes)
    assigned = {int(node) for node in assignment}

    missing = nodes - assigned
    if missing:
        violations.append(
            FeasibilityViolation(
                "coverage",
                f"{len(missing)} problem node(s) unassigned, e.g. {sorted(missing)[:5]}",
                float(len(missing)),
            )
        )
    extra = assigned - nodes
    if extra:
        violations.append(
            FeasibilityViolation(
                "coverage",
                f"{len(extra)} assigned node(s) outside the problem, "
                f"e.g. {sorted(extra)[:5]}",
                float(len(extra)),
            )
        )

    for node in sorted(nodes & assigned):
        raw = assignment[node]
        if isinstance(raw, bool) or not isinstance(raw, (int,)):
            violations.append(
                FeasibilityViolation(
                    "zone", f"node {node} has non-integer zone {raw!r}"
                )
            )
            continue
        zone = int(raw)
        if not 0 <= zone < problem.Z:
            violations.append(
                FeasibilityViolation(
                    "zone", f"node {node} has out-of-range zone {zone}"
                )
            )
            continue
        if zone not in problem.candidate_zones(node):
            violations.append(
                FeasibilityViolation(
                    "candidate", f"node {node} is not a candidate for zone {zone}"
                )
            )

    if violations:
        return violations

    for zone, centroid in enumerate(problem.centroids):
        if int(assignment[centroid]) != zone:
            violations.append(
                FeasibilityViolation(
                    "centroid",
                    f"centroid {centroid} of zone {zone} sits in zone "
                    f"{int(assignment[centroid])}",
                )
            )
    return violations


def _contiguity_violations(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
) -> list[FeasibilityViolation]:
    plain = {int(node): int(zone) for node, zone in assignment.items()}
    if contiguity.is_contiguous(problem.G, plain, problem.centroids):
        return []
    return [
        FeasibilityViolation("contiguity", "a zone induces more than one component")
    ]


def _balance_violations(
    problem: ZoneProblem,
    zones: list[list[int]],
    tolerance: float,
) -> list[FeasibilityViolation]:
    # Imported here so a pure validator does not pull in the solver package
    # (and, through it, this module's own importers).
    from optimization.solvers.balance import balance_constraints

    violations: list[FeasibilityViolation] = []
    for constraint in balance_constraints(problem):
        for zone, nodes in enumerate(zones):
            students = sum(problem.students(node) for node in nodes)
            value = sum(constraint.value(node) for node in nodes)
            if constraint.lower_ratio is not None:
                short = constraint.lower_ratio * students - value
                if short > tolerance:
                    violations.append(
                        FeasibilityViolation(
                            f"{constraint.kind}_lower",
                            f"zone {zone} holds {value:.6g} against a floor of "
                            f"{constraint.lower_ratio * students:.6g}",
                            short,
                        )
                    )
            if constraint.upper_ratio is not None:
                over = value - constraint.upper_ratio * students
                if over > tolerance:
                    violations.append(
                        FeasibilityViolation(
                            f"{constraint.kind}_upper",
                            f"zone {zone} holds {value:.6g} against a ceiling of "
                            f"{constraint.upper_ratio * students:.6g}",
                            over,
                        )
                    )
    return violations


def _school_count_violations(
    problem: ZoneProblem,
    zones: list[list[int]],
    tolerance: float,
) -> list[FeasibilityViolation]:
    total = sum(problem.num_schools(node) for node in problem.nodes)
    if total == 0:
        return []
    average = total / problem.Z
    floor_, ceiling = max(0.0, average - 1.0), average + 1.0
    violations: list[FeasibilityViolation] = []
    for zone, nodes in enumerate(zones):
        schools = float(sum(problem.num_schools(node) for node in nodes))
        if floor_ - schools > tolerance:
            violations.append(
                FeasibilityViolation(
                    "schools_lower",
                    f"zone {zone} holds {schools:g} schools against a floor of "
                    f"{floor_:.6g}",
                    floor_ - schools,
                )
            )
        if schools - ceiling > tolerance:
            violations.append(
                FeasibilityViolation(
                    "schools_upper",
                    f"zone {zone} holds {schools:g} schools against a ceiling of "
                    f"{ceiling:.6g}",
                    schools - ceiling,
                )
            )
    return violations


def _boundary_violations(
    problem: ZoneProblem,
    assignment: Mapping[int, int],
) -> list[FeasibilityViolation]:
    if problem.boundary_prop < 0:
        return []
    plain = {int(node): int(zone) for node, zone in assignment.items()}
    cut = contiguity.boundary_edges(problem.G, plain)
    # Mirrors the solver's cap exactly: an unweighted count against a floored
    # proportion of the edge count.
    limit = math.floor(problem.boundary_prop * problem.G.number_of_edges())
    if cut <= limit:
        return []
    return [
        FeasibilityViolation(
            "boundary",
            f"{cut} cut edges against a cap of {limit}",
            float(cut - limit),
        )
    ]

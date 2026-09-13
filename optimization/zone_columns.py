"""Whole-zone columns and the restricted master they feed.

Everything here is maximized. Boundary scores are minus half a zone's
perimeter, so summing them over a partition counts each cut edge once and the
master's boundary row is the district boundary cost.

The master is a set-partitioning LP and is therefore massively degenerate:
many columns can enter with a strictly positive reduced cost and a step length
of zero, which is a degenerate pivot, not progress. Measured on
:math:`\\text{BlockGroup}_0` with a 404-column pool, three to six such columns
entered every round and the LP value did not move for eight rounds. The
standard remedies are both implemented here and both are about the *duals*
rather than the columns:

``method``
    Which algorithm solves the LP, and hence which dual solution it lands on.
    Dual simplex returns a vertex of the dual polyhedron, and with only 132 of
    579 cover rows carrying a nonzero dual the pricing problem gets almost no
    guidance -- it maximizes raw welfare and returns oversized zones. Barrier
    without crossover returns an *interior* dual point instead, which spreads
    the prices over the rows and is the default here. The cost is that the
    duals are no longer bit-for-bit reproducible across runs.

:func:`smooth`
    Wentges dual smoothing: price against a convex combination of this LP's
    duals and the previous round's point rather than against the raw LP duals.
    Proposition 9's bound is valid at *any* dual point -- it only needs
    ``mu >= 0`` and a global pricing bound at that point -- so smoothing costs
    nothing in rigour. The bound is simply computed where the pricing happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from optimization.dw_options import DW_MASTER_METHODS as MASTER_METHODS
from optimization.zone_family import (
    ZoneFamily,
    boundary_limit,
    build_zone_family,
    perimeter as zone_perimeter,
)
from optimization.zone_welfare import BoundaryZoneObjective, ZoneObjective


__all__ = [
    "DualPoint",
    "MASTER_METHODS",
    "MasterResult",
    "ZoneColumn",
    "ZonePool",
    "boundary_limit",
    "smooth",
    "solve_master",
]


@dataclass(frozen=True)
class ZoneColumn:
    zone: int
    nodes: frozenset[int]
    score: float
    perimeter: int


class ZonePool:
    """Deduplicate, validate, and cache the exact value of zone memberships.

    Admissibility is :class:`~optimization.zone_family.ZoneFamily`'s and
    nothing else, so the pool, the seeding filter and the pricing model agree
    by construction rather than by two implementations of the same rows.
    """

    def __init__(
        self,
        problem,
        objective: ZoneObjective | None = None,
        *,
        family: ZoneFamily | None = None,
        centroid_neighbor_radius: int = 0,
    ):
        self.problem = problem
        self.objective = objective if objective is not None else BoundaryZoneObjective()
        self.family = (
            family
            if family is not None
            else build_zone_family(
                problem, centroid_neighbor_radius=centroid_neighbor_radius
            )
        )
        self.nodes = frozenset(problem.nodes)
        self.columns: dict[tuple[int, frozenset[int]], ZoneColumn] = {}
        self.scores: dict[frozenset[int], tuple[float, int]] = {}
        self.objective.validate(self.nodes)

    def feasible(self, zone, nodes) -> bool:
        return self.family.feasible(zone, frozenset(nodes))

    def column(self, zone, nodes) -> ZoneColumn:
        nodes = frozenset(nodes)
        if not self.feasible(zone, nodes):
            raise ValueError("Cannot create an inadmissible zone column.")
        if nodes not in self.scores:
            edge_cut = zone_perimeter(self.problem, nodes)
            score = float(self.objective.score(nodes, edge_cut))
            if not math.isfinite(score):
                raise ValueError("Zone objective returned a non-finite value.")
            self.scores[nodes] = (score, edge_cut)
        score, edge_cut = self.scores[nodes]
        return ZoneColumn(zone, nodes, score, edge_cut)

    def admit(self, zone, nodes) -> ZoneColumn | None:
        """The column for ``nodes``, or ``None`` if it is not admissible.

        Rejection is the normal case, not an error: a ReCom sample is
        connected but knows nothing about anchors or closer-neighbour support,
        and an interrupted pricing solve reports to a tolerance.
        """

        nodes = frozenset(nodes)
        if not self.feasible(zone, nodes):
            return None
        return self.column(zone, nodes)

    def add(self, column) -> bool:
        key = (column.zone, column.nodes)
        if key in self.columns:
            return False
        self.columns[key] = column
        return True

    def add_partition(self, assignment) -> tuple[ZoneColumn, ...]:
        if set(assignment) != self.nodes or set(assignment.values()) != set(
            range(self.problem.Z)
        ):
            raise ValueError("A seed partition must cover all nodes and zones.")
        columns = tuple(
            self.column(z, frozenset(n for n, a in assignment.items() if a == z))
            for z in range(self.problem.Z)
        )
        for column in columns:
            self.add(column)
        return columns

    def admit_partition(self, assignment) -> tuple[ZoneColumn, ...] | None:
        """Add every admissible zone of ``assignment``; return it if all are.

        This is the rejection filter the ReCom seeder runs through. A partition
        with one bad zone still contributes its other ``Z - 1`` zones, which is
        most of what recombination produces: a ReCom step rewrites two zones
        and leaves the rest of a previously admissible partition alone.
        """

        columns = []
        complete = True
        for zone in range(self.problem.Z):
            nodes = frozenset(
                node for node, label in assignment.items() if label == zone
            )
            column = self.admit(zone, nodes)
            if column is None:
                complete = False
                continue
            self.add(column)
            columns.append(column)
        return tuple(columns) if complete else None


@dataclass(frozen=True)
class DualPoint:
    """A dual solution the pricing problem can be aimed at.

    The pricer reads only these three fields, so a smoothed point and a raw LP
    solution are interchangeable to it.
    """

    node_duals: dict[int, float]
    zone_duals: dict[int, float]
    boundary_dual: float = 0.0
    phase_one: bool = False

    def reduced_cost(self, column) -> float:
        return (
            (0.0 if self.phase_one else column.score)
            - sum(self.node_duals[n] for n in column.nodes)
            - self.zone_duals[column.zone]
            - self.boundary_dual * column.perimeter / 2
        )

    def dual_objective(self, problem) -> float:
        """The dual objective, which Proposition 9 corrects into a bound."""

        value = sum(self.node_duals.values()) + sum(self.zone_duals.values())
        if problem.boundary_prop >= 0:
            value += self.boundary_dual * boundary_limit(problem)
        return value


def smooth(previous: DualPoint | None, current: DualPoint, alpha: float) -> DualPoint:
    """Wentges smoothing of ``current`` toward ``previous``.

    ``alpha = 1`` is no smoothing. The boundary multiplier is clamped
    non-negative because Proposition 9 needs ``mu >= 0`` to read the master's
    boundary row as a dual inequality; a convex combination of non-negative
    multipliers is non-negative, so the clamp only ever catches a numerical
    artefact.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("dw_dual_smoothing must lie in (0, 1].")
    if previous is None or alpha == 1.0:
        return DualPoint(
            dict(current.node_duals),
            dict(current.zone_duals),
            max(0.0, current.boundary_dual),
            current.phase_one,
        )
    blend = lambda new, old: alpha * new + (1.0 - alpha) * old  # noqa: E731
    return DualPoint(
        {
            node: blend(value, previous.node_duals.get(node, 0.0))
            for node, value in current.node_duals.items()
        },
        {
            zone: blend(value, previous.zone_duals.get(zone, 0.0))
            for zone, value in current.zone_duals.items()
        },
        max(0.0, blend(current.boundary_dual, previous.boundary_dual)),
        current.phase_one,
    )


@dataclass(frozen=True)
class MasterResult:
    status: str
    objective: float | None = None
    selected: tuple[ZoneColumn, ...] = ()
    node_duals: dict[int, float] | None = None
    zone_duals: dict[int, float] | None = None
    boundary_dual: float = 0.0
    values: tuple[float, ...] = ()
    artificial_mass: float = 0.0
    phase_one: bool = False

    def duals(self) -> DualPoint:
        return DualPoint(
            dict(self.node_duals or {}),
            dict(self.zone_duals or {}),
            self.boundary_dual,
            self.phase_one,
        )

    def reduced_cost(self, column) -> float:
        return self.duals().reduced_cost(column)


def solve_master(
    problem,
    columns,
    seconds,
    *,
    integer=False,
    phase_one=False,
    method="barrier",
) -> MasterResult:
    """Cover every node once and choose exactly one column per zone label."""

    if seconds <= 0:
        return MasterResult("TIME_LIMIT")
    if method not in MASTER_METHODS:
        raise ValueError(
            f"dw_master_method must be one of: {sorted(MASTER_METHODS)}."
        )
    columns = tuple(columns)
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("dw_master", env=env) as m:
        if math.isfinite(seconds):
            m.Params.TimeLimit = max(1e-3, float(seconds))
        m.Params.Seed = 0
        m.Params.MIPGap = 0.0
        if not integer:
            m.Params.Method = MASTER_METHODS[method]
            if method == "barrier":
                # Stop at the barrier iterate. Crossover would walk it to a
                # vertex, which is exactly the degenerate dual solution the
                # interior point is chosen to avoid.
                m.Params.Crossover = 0

        variables = []
        for i, column in enumerate(columns):
            # No upper bound in the LP: implied by the cover rows, and leaving
            # it off keeps reduced costs entirely in the structural row duals.
            variables.append(
                m.addVar(vtype=GRB.BINARY, name=f"lambda_{i}")
                if integer
                else m.addVar(lb=0.0, ub=GRB.INFINITY, name=f"lambda_{i}")
            )

        cover_terms: dict[int, list] = {n: [] for n in problem.nodes}
        convexity_terms: dict[int, list] = {z: [] for z in range(problem.Z)}
        boundary_terms = []
        objective = gp.LinExpr()
        for column, variable in zip(columns, variables):
            for n in column.nodes:
                cover_terms[n].append(variable)
            convexity_terms[column.zone].append(variable)
            boundary_terms.append((column.perimeter / 2, variable))
            if not phase_one:
                objective.addTerms(float(column.score), variable)

        artificials = []
        if phase_one:
            for terms in (*cover_terms.values(), *convexity_terms.values()):
                # Deficit-only artificials keep structural coverage <= 1. An
                # empty pool is always Phase-I feasible, and zero deficit
                # recovers the exact master. Increasing a convexity dual stays
                # valid for them, which is what Proposition 9 needs in Phase I.
                variable = m.addVar(lb=0.0, ub=GRB.INFINITY)
                terms.append(variable)
                objective.addTerms(-1.0, variable)
                artificials.append(variable)

        cover = {
            n: m.addConstr(gp.quicksum(terms) == 1, name=f"cover_{n}")
            for n, terms in cover_terms.items()
        }
        convexity = {
            z: m.addConstr(gp.quicksum(terms) == 1, name=f"zone_{z}")
            for z, terms in convexity_terms.items()
        }
        boundary = None
        if problem.boundary_prop >= 0:
            boundary = m.addConstr(
                gp.quicksum(weight * variable for weight, variable in boundary_terms)
                <= boundary_limit(problem),
                name="boundary",
            )
        m.setObjective(objective, GRB.MAXIMIZE)
        m.optimize()

        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            return MasterResult("INFEASIBLE")
        if m.SolCount == 0:
            return MasterResult(
                "TIME_LIMIT" if m.Status == GRB.TIME_LIMIT else "ERROR"
            )
        status = "OPTIMAL" if m.Status == GRB.OPTIMAL else "FEASIBLE"
        if not integer and status != "OPTIMAL":
            return MasterResult("LP_NOT_OPTIMAL")
        return MasterResult(
            status,
            m.ObjVal,
            tuple(c for c, v in zip(columns, variables) if integer and v.X > 0.5),
            None if integer else {n: r.Pi for n, r in cover.items()},
            None if integer else {z: r.Pi for z, r in convexity.items()},
            boundary.Pi if boundary is not None and not integer else 0.0,
            tuple(v.X for v in variables),
            sum(v.X for v in artificials),
            phase_one,
        )

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

Neither remedy was sufficient, and the measured reason is that the primal has
no freedom rather than that the duals are badly chosen: with ``|V| + Z``
equality rows against a pool that holds one tiling, the LP's feasible set is a
single point and *every* entering column has a zero step length. Changing which
dual you read off that point cannot help.

``overlap_budget``
    The repair that addresses the primal. Each cover row becomes
    ``sum(lambda) + d_v - e_v = 1`` with ``d, e >= 0``, and one extra row
    rations the total mismatch: ``sum_v w_v (d_v + e_v) <= K``. The LP is then
    full-dimensional in ``lambda``, so a priced zone that collides with the
    incumbent's other zones can enter with a positive step length, and the
    cover duals are set by where mismatch is contested rather than left at
    zero by degeneracy.

    Budgeting rather than penalizing is the point. The classical elastic master
    penalizes mismatch at a hand-chosen ``M``, and ``M`` has to be guessed in
    welfare-per-node units: above ``max_v |alpha_v|`` the elastic variables
    price themselves out and the degenerate single point is back, below it the
    LP drifts toward zones that can never tile. The budget row's dual *is* that
    ``M``, re-chosen by the LP every round and bounded by the elastic pair's own
    dual-feasibility rows, ``|alpha_v| <= w_v mu_K``. What is left to choose is
    ``K``, in student-equivalents, which is the same kind of knob as
    ``boundary_prop``.

    It stays a relaxation of the exact master -- every tiling has ``d = e = 0``
    and the same objective -- so Proposition 9 survives and the bound stays
    valid at any ``K``; a weaker ``K`` costs bound quality, never correctness.
    ``K = 0`` is the exact master.
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
    "overlap_limit",
    "overlap_weight",
    "smooth",
    "solve_master",
]


def overlap_weight(problem, node) -> float:
    """What one unit of coverage mismatch at ``node`` costs the budget.

    Student mass, floored at one unit. The floor is not cosmetic: a node with
    no students would otherwise be double-claimable for free, and a free node
    is exactly what lets two labels each run their own support chain through it
    without either paying for it. Weighting by students rather than uniformly
    matters because block mass spans an order of magnitude -- a uniform weight
    charges the same for double-claiming a 2-student block and a 40-student
    one, so the LP would spend its whole budget where the welfare payoff is and
    the duals would be least informative exactly there.
    """

    return max(1.0, float(problem.students(node)))


def overlap_limit(problem, proportion) -> float:
    """``K``, from a proportion of the district's total weighted mass."""

    proportion = float(proportion)
    if proportion <= 0:
        return 0.0
    return proportion * sum(overlap_weight(problem, n) for n in problem.nodes)


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
    # Dual of the overlap-budget row, and the ``M`` the elastic pair is priced
    # at. It is deliberately absent from ``reduced_cost``: the elastic
    # variables are always in the master and are never priced, so they change
    # no zone column's reduced cost. It enters the *bound*, as ``mu_K * K``.
    overlap_dual: float = 0.0

    def reduced_cost(self, column) -> float:
        return (
            (0.0 if self.phase_one else column.score)
            - sum(self.node_duals[n] for n in column.nodes)
            - self.zone_duals[column.zone]
            - self.boundary_dual * column.perimeter / 2
        )

    def dual_objective(self, problem, overlap_limit: float = 0.0) -> float:
        """The dual objective, which Proposition 9 corrects into a bound.

        ``overlap_limit`` is ``K``. Omitting it would understate the dual
        objective of a budgeted-elastic master and so report a bound *below*
        the relaxation's own optimum, which is the one direction that is not
        merely weak but wrong.
        """

        value = sum(self.node_duals.values()) + sum(self.zone_duals.values())
        if problem.boundary_prop >= 0:
            value += self.boundary_dual * boundary_limit(problem)
        return value + self.overlap_dual * float(overlap_limit)


def smooth(previous: DualPoint | None, current: DualPoint, alpha: float) -> DualPoint:
    """Wentges smoothing of ``current`` toward ``previous``.

    ``alpha = 1`` is no smoothing. The boundary and overlap multipliers are
    clamped non-negative because Proposition 9 needs ``mu >= 0`` to read those
    rows as dual inequalities; a convex combination of non-negative multipliers
    is non-negative, so the clamp only ever catches a numerical artefact.

    Smoothing is safe for the elastic pair's dual-feasibility rows as well.
    ``|alpha_v| <= w_v mu_K`` cuts out a convex set, and both endpoints satisfy
    it, so the blend does too -- provided ``mu_K`` is blended along with
    ``alpha``, which is why it lives on :class:`DualPoint` rather than being
    folded into the bound at the call site. The rows do not mention ``K``, so a
    point smoothed across a change in ``K`` stays dual-feasible.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("dw_dual_smoothing must lie in (0, 1].")
    if previous is None or alpha == 1.0:
        return DualPoint(
            dict(current.node_duals),
            dict(current.zone_duals),
            max(0.0, current.boundary_dual),
            current.phase_one,
            max(0.0, current.overlap_dual),
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
        max(0.0, blend(current.overlap_dual, previous.overlap_dual)),
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
    overlap_dual: float = 0.0
    #: Weighted coverage mismatch the budget actually spent. Zero means this
    #: LP solution is a partition of the pool's columns after all, so the
    #: elasticity bought nothing; equal to ``K`` means the budget is binding.
    elastic_mass: float = 0.0
    #: How many nodes carry any mismatch, which is the shape of it.
    elastic_nodes: int = 0
    #: Cover duals sitting exactly on ``|alpha_v| <= w_v mu_K``. A large count
    #: means the budget row, not the columns, is setting the prices -- the
    #: signature of a ``K`` too small to guide pricing.
    duals_pinned: int = 0

    def duals(self) -> DualPoint:
        return DualPoint(
            dict(self.node_duals or {}),
            dict(self.zone_duals or {}),
            self.boundary_dual,
            self.phase_one,
            self.overlap_dual,
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
    overlap_budget=0.0,
) -> MasterResult:
    """Cover every node once and choose exactly one column per zone label.

    ``overlap_budget`` is ``K``: the weighted coverage mismatch the cover rows
    may carry between them, rationed by a single row rather than priced by a
    penalty. ``0`` is the exact master and is the default.
    """

    if seconds <= 0:
        return MasterResult("TIME_LIMIT")
    if method not in MASTER_METHODS:
        raise ValueError(
            f"dw_master_method must be one of: {sorted(MASTER_METHODS)}."
        )
    overlap_budget = float(overlap_budget)
    if overlap_budget < 0 or math.isnan(overlap_budget):
        raise ValueError("dw_overlap_prop must be nonnegative.")
    if overlap_budget > 0 and integer:
        raise ValueError(
            "The integer master must be exact. It is the only thing here that "
            "produces an incumbent, and an incumbent has to be a tiling."
        )
    if overlap_budget > 0 and phase_one:
        raise ValueError(
            "Phase I already carries deficit artificials and its completion "
            "test is zero deficit, so a surplus variable would let it pass by "
            "double-claiming a node instead of by covering the graph."
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

        elastic: dict[int, tuple] = {}
        if overlap_budget > 0:
            for n in cover_terms:
                # Both directions are needed. Raising a new label-z column
                # pushes the incumbent's label-z column down by the same
                # amount, which under-covers the nodes only the incumbent held
                # (deficit) and over-covers the ones only the newcomer holds
                # (surplus). Deficit alone -- which is what Phase I carries --
                # still leaves the step length at zero, so the surplus
                # variable is the load-bearing half.
                #
                # No objective coefficient and no upper bound. The price of a
                # unit of mismatch is the budget row's dual, and a bound whose
                # own reduced cost the dual objective did not account for
                # would understate the bound rather than merely weaken it. The
                # budget row bounds them anyway: w_v >= 1, so e_v <= K.
                elastic[n] = (
                    m.addVar(lb=0.0, ub=GRB.INFINITY, name=f"deficit_{n}"),
                    m.addVar(lb=0.0, ub=GRB.INFINITY, name=f"surplus_{n}"),
                )

        cover = {}
        for n, terms in cover_terms.items():
            row = gp.quicksum(terms)
            if n in elastic:
                deficit, surplus = elastic[n]
                row = row + deficit - surplus
            cover[n] = m.addConstr(row == 1, name=f"cover_{n}")
        convexity = {
            z: m.addConstr(gp.quicksum(terms) == 1, name=f"zone_{z}")
            for z, terms in convexity_terms.items()
        }
        overlap = None
        if elastic:
            overlap = m.addConstr(
                gp.quicksum(
                    overlap_weight(problem, n) * (deficit + surplus)
                    for n, (deficit, surplus) in elastic.items()
                )
                <= overlap_budget,
                name="overlap",
            )
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
        overlap_dual = 0.0 if overlap is None or integer else float(overlap.Pi)
        elastic_mass = 0.0
        elastic_nodes = 0
        for n, (deficit, surplus) in elastic.items():
            units = deficit.X + surplus.X
            if units > 1e-9:
                elastic_nodes += 1
                elastic_mass += overlap_weight(problem, n) * units
        # A cover dual sitting *at* ``w_v mu_K`` is one the budget row is
        # holding down rather than one the columns set, so counting them says
        # whether K is large enough for the prices to mean anything.
        duals_pinned = (
            sum(
                1
                for n, row in cover.items()
                if abs(abs(row.Pi) - overlap_weight(problem, n) * overlap_dual)
                <= 1e-6 * max(1.0, abs(row.Pi))
            )
            if overlap_dual > 1e-9
            else 0
        )
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
            overlap_dual,
            elastic_mass,
            elastic_nodes,
            duals_pinned,
        )

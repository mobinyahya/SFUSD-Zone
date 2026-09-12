"""Whole-zone columns and restricted masters for branch-and-price.

All objectives are maximized internally. Boundary scores are minus half the
zone perimeter, so summing over a partition counts each cut edge once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from optimization.data.mid import MidMarket
from optimization.mid_oracle import finite_grid_oracle
from optimization.solvers.balance import balance_constraints


ZONE_FEASIBILITY_TOLERANCE = 1e-6


def boundary_limit(problem):
    """Shared cap, including the partition validator's feasibility slack."""
    total = sum(problem.boundary_weight(u, v) for u, v in problem.G.edges)
    return problem.boundary_prop * total + ZONE_FEASIBILITY_TOLERANCE


def restrict_market(market: MidMarket, nodes=None) -> MidMarket:
    """Remove citywide programs, optionally keeping only one zone's market.

    Students remain in the cohort even when all their preferences disappear.
    Utilities and priority tiers retain their original values and rank order.
    """
    programs = tuple(
        p
        for p in market.programs
        if not p.citywide and (nodes is None or p.school_node in nodes)
    )
    available = {p.program_id for p in programs}
    types = []
    for student in market.types:
        if nodes is not None and student.node not in nodes:
            continue
        ranks = [i for i, p in enumerate(student.programs) if p in available]
        types.append(
            replace(
                student,
                programs=tuple(student.programs[i] for i in ranks),
                priorities=tuple(student.priorities[i] for i in ranks),
                utility_sums=tuple(student.utility_sums[i] for i in ranks),
                scaled_utility_sums=tuple(
                    student.scaled_utility_sums[i] for i in ranks
                ),
            )
        )
    count = sum(t.count for t in types)
    return replace(
        market,
        programs=programs,
        types=tuple(types),
        student_count=count,
        outside_only_student_count=sum(t.count for t in types if not t.programs),
        # The source only stores the district utility-cohort count, not a flag
        # per type. This diagnostic is not used by the oracle.
        utility_student_count=(
            market.utility_student_count
            if nodes is None
            else sum(t.count for t in types if t.programs)
        ),
    )


@dataclass(frozen=True)
class ZoneColumn:
    zone: int
    nodes: frozenset[int]
    score: float
    perimeter: int


class ZonePool:
    """Deduplicate, validate, and cache exact scores of zone memberships.

    Geography follows ReCom: connected zones, unanchored centroids, no implicit
    distance restriction. Explicit candidate and fixed assignments are hard.
    """

    def __init__(self, problem, market=None, lottery_scale=20):
        self.problem = problem
        self.market = restrict_market(market) if market is not None else None
        self.lottery_scale = lottery_scale
        self.columns: dict[tuple[int, frozenset[int]], ZoneColumn] = {}
        self.scores: dict[frozenset[int], tuple[float, int]] = {}
        self.nodes = frozenset(problem.nodes)
        self.constraints = balance_constraints(problem)
        self.school_total = sum(problem.num_schools(n) for n in self.nodes)
        if self.market is not None:
            if any(t.node not in self.nodes for t in self.market.types):
                raise ValueError("DW market students must belong to the graph.")
            if any(p.school_node not in self.nodes for p in self.market.programs):
                raise ValueError("DW market schools must belong to the graph.")

    def feasible(self, zone, nodes) -> bool:
        p = self.problem
        if not nodes or not nodes <= self.nodes or not 0 <= zone < p.Z:
            return False
        for node, fixed in (p.fixed or {}).items():
            if (node in nodes) != (fixed == zone):
                return False
        if any(
            zone not in p.candidates[n]
            for n in nodes
            if p.candidates is not None and n in p.candidates
        ):
            return False
        if not nx.is_connected(p.G.subgraph(nodes)):
            return False
        students = sum(p.students(n) for n in nodes)
        for row in self.constraints:
            value = sum(row.value(n) for n in nodes)
            if (
                row.lower_ratio is not None
                and value < row.lower_ratio * students - ZONE_FEASIBILITY_TOLERANCE
            ):
                return False
            if (
                row.upper_ratio is not None
                and value > row.upper_ratio * students + ZONE_FEASIBILITY_TOLERANCE
            ):
                return False
        if self.school_total:
            schools = sum(p.num_schools(n) for n in nodes)
            average = self.school_total / p.Z
            if (
                not max(0, average - 1) - ZONE_FEASIBILITY_TOLERANCE
                <= schools
                <= average + 1 + ZONE_FEASIBILITY_TOLERANCE
            ):
                return False
        return True

    def column(self, zone, nodes) -> ZoneColumn:
        nodes = frozenset(nodes)
        if not self.feasible(zone, nodes):
            raise ValueError("Cannot create an infeasible zone column.")
        if nodes not in self.scores:
            perimeter = sum(
                self.problem.boundary_weight(u, v)
                for u, v in self.problem.G.edges
                if (u in nodes) != (v in nodes)
            )
            if self.market is None:
                score = -perimeter / 2
            else:
                local = restrict_market(self.market, nodes)
                score = finite_grid_oracle(
                    local,
                    {n: 0 for n in nodes},
                    self.lottery_scale,
                    check_minimality=False,
                ).welfare
            if not math.isfinite(score):
                raise ValueError("Zone oracle returned a non-finite welfare.")
            self.scores[nodes] = (float(score), perimeter)
        score, perimeter = self.scores[nodes]
        return ZoneColumn(zone, nodes, score, perimeter)

    def add(self, column) -> bool:
        key = (column.zone, column.nodes)
        if key in self.columns:
            return False
        self.columns[key] = column
        return True

    def add_partition(self, assignment):
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

    def reduced_cost(self, column):
        return (
            (0.0 if self.phase_one else column.score)
            - sum(self.node_duals[n] for n in column.nodes)
            - self.zone_duals[column.zone]
            - self.boundary_dual * column.perimeter / 2
        )


def solve_master(
    problem, columns, seconds, *, integer=False, phase_one=False
) -> MasterResult:
    """Cover every node once and choose exactly one column per zone label."""
    if seconds <= 0:
        return MasterResult("TIME_LIMIT")
    columns = tuple(columns)
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("dw_master", env=env) as m:
        if math.isfinite(seconds):
            m.Params.TimeLimit = max(1e-3, float(seconds))
        # Dual simplex rather than the default concurrent method: these duals
        # are the pricing objective, so the same pool has to produce the same
        # duals on every run. Concurrent returns whichever algorithm finishes
        # first, and barrier would return interior rather than vertex duals.
        #
        # REVISIT -- those interior duals may be exactly what this needs.
        # Vertex duals of a set-partitioning master are massively degenerate
        # (132 of 579 nonzero on BlockGroup_0), which is what freezes column
        # generation; see the note in ``optimization/branch_price.py``. Trying
        # ``Method = 2`` costs reproducibility, so measure before switching.
        m.Params.Method = 1
        m.Params.Seed = 0
        m.Params.MIPGap = 0.0

        variables = []
        for i, column in enumerate(columns):
            # No upper bound in the LP: implied by cover rows, and avoiding it
            # leaves reduced costs entirely in the structural row duals.
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
                # Deficit-only artificials keep structural coverage <= 1. Empty
                # columns are always feasible, and zero deficit recovers the
                # exact master. Increasing a convexity dual remains valid for
                # artificials.
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

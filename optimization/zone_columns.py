"""Whole-zone columns and restricted masters for branch-and-price.

All objectives are maximized internally. Boundary scores are minus half the
zone perimeter, so summing over a partition counts each cut edge once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import networkx as nx
from ortools.linear_solver import pywraplp

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
    solver = pywraplp.Solver.CreateSolver("SCIP" if integer else "GLOP")
    if solver is None:
        raise RuntimeError("DW requires the OR-Tools GLOP and SCIP backends.")
    if math.isfinite(seconds):
        solver.SetTimeLimit(max(1, int(seconds * 1000)))
    cover = {n: solver.Constraint(1, 1) for n in problem.nodes}
    convexity = {z: solver.Constraint(1, 1) for z in range(problem.Z)}
    boundary = None
    if problem.boundary_prop >= 0:
        boundary = solver.Constraint(-solver.infinity(), boundary_limit(problem))
    objective = solver.Objective()
    objective.SetMaximization()
    artificials = []
    if phase_one:
        for row in (*cover.values(), *convexity.values()):
            # Deficit-only artificials keep structural coverage <= 1. Empty
            # columns are always feasible, and zero deficit recovers the exact
            # master. Increasing a convexity dual remains valid for artificials.
            for sign in (1,):
                var = solver.NumVar(0, solver.infinity(), "")
                row.SetCoefficient(var, sign)
                objective.SetCoefficient(var, -1)
                artificials.append(var)
    variables = []
    for i, column in enumerate(columns):
        # No upper bound in the LP: implied by cover rows, and avoiding it
        # leaves reduced costs entirely in the structural row duals.
        var = (
            solver.BoolVar(f"lambda_{i}")
            if integer
            else solver.NumVar(0, solver.infinity(), f"lambda_{i}")
        )
        variables.append(var)
        for n in column.nodes:
            cover[n].SetCoefficient(var, 1)
        convexity[column.zone].SetCoefficient(var, 1)
        if boundary is not None:
            boundary.SetCoefficient(var, column.perimeter / 2)
        objective.SetCoefficient(var, 0 if phase_one else column.score)
    status = solver.Solve()
    if status not in (solver.OPTIMAL, solver.FEASIBLE):
        return MasterResult(
            {solver.INFEASIBLE: "INFEASIBLE", solver.NOT_SOLVED: "TIME_LIMIT"}.get(
                status, "ERROR"
            )
        )
    if not integer and status != solver.OPTIMAL:
        return MasterResult("LP_NOT_OPTIMAL")
    return MasterResult(
        "OPTIMAL" if status == solver.OPTIMAL else "FEASIBLE",
        objective.Value(),
        tuple(
            c
            for c, v in zip(columns, variables)
            if integer and v.solution_value() > 0.5
        ),
        None if integer else {n: r.dual_value() for n, r in cover.items()},
        None if integer else {z: r.dual_value() for z, r in convexity.items()},
        boundary.dual_value() if boundary is not None and not integer else 0.0,
        tuple(v.solution_value() for v in variables),
        sum(v.solution_value() for v in artificials),
        phase_one,
    )

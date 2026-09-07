"""Global MIP pricing for one connected zone and its MID market.

The binary disjunctions encode the cutoff/minimum recurrence exactly. Lottery
masses are integral; utility coefficients are never rounded.
SCIP supplies a global reduced-cost bound, including on interrupted solves.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ortools.linear_solver import pywraplp

from optimization.zone_columns import ZONE_FEASIBILITY_TOLERANCE


@dataclass(frozen=True)
class PricingResult:
    status: str
    bound: float
    nodes: frozenset[int] = frozenset()
    score: float | None = None
    reduced_cost: float | None = None


def compatible(column, decisions):
    return all(
        (node in column.nodes) == value
        for (node, zone), value in decisions.items()
        if zone == column.zone
    )


def _solver(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("Exact zone pricing requires OR-Tools SCIP.")
    if math.isfinite(remaining):
        solver.SetTimeLimit(max(1, int(remaining * 1000)))
    solver.SetSolverSpecificParametersAsString(
        "limits/gap = 0\nlimits/absgap = 0\nnumerics/feastol = 1e-9"
    )
    return solver


def _minimum(solver, a, b, upper, big_m, variable):
    result = variable(0, upper, "")
    select = solver.BoolVar("")
    solver.Add(result <= a)
    solver.Add(result <= b)
    solver.Add(result >= a - big_m * select)
    solver.Add(result >= b - big_m * (1 - select))
    return result


def _welfare_model(solver, market, membership, lottery_scale):
    """Maximizing over capacity-clearing cutoffs yields least-cutoff welfare.

    Omitted students and schools have zero access even when co-located. All
    preference prefixes are represented, without optimistic tail relaxations.
    """
    scale = lottery_scale
    variable = solver.IntVar
    programs = market.program_by_id
    priorities = {p: set() for p in programs}
    for student in market.types:
        for p, priority in zip(student.programs, student.priorities):
            priorities[p].add(priority)
    thresholds = {}
    for p in programs:
        upper = (max(priorities[p], default=0) + 1) * scale
        cutoff = variable(0, upper, "")
        for priority in priorities[p]:
            positive = solver.BoolVar("")
            raw = variable(0, upper, "")
            delta = cutoff - priority * scale
            solver.Add(raw >= delta)
            solver.Add(raw <= delta + upper * (1 - positive))
            solver.Add(raw <= upper * positive)
            thresholds[p, priority] = _minimum(
                solver, raw, scale, scale, upper, variable
            )
    access = {}
    capacities = {p: [] for p in programs}
    welfare = []
    for student in market.types:
        previous = scale
        for p, priority, utility in zip(
            student.programs, student.priorities, student.utility_sums
        ):
            program = programs[p]
            if program.citywide:
                raise ValueError("Exact DW pricing forbids citywide programs.")
            pair = (student.node, program.school_node)
            if pair not in access:
                both = solver.BoolVar("")
                a, b = membership[pair[0]], membership[pair[1]]
                solver.Add(both <= a)
                solver.Add(both <= b)
                solver.Add(both >= a + b - 1)
                access[pair] = both
            both = access[pair]
            threshold = thresholds[p, priority]
            effective = variable(0, scale, "")
            solver.Add(effective >= threshold)
            solver.Add(effective <= threshold + scale * (1 - both))
            solver.Add(effective >= scale * (1 - both))
            remaining = _minimum(solver, previous, effective, scale, scale, variable)
            mass = previous - remaining
            capacities[p].append(student.count * mass)
            welfare.append((utility / scale) * mass)
            previous = remaining
    for p, terms in capacities.items():
        solver.Add(solver.Sum(terms) <= programs[p].capacity * scale)
    return solver.Sum(welfare)


def price_zone(
    pool,
    zone,
    master,
    decisions,
    *,
    deadline,
    model="grid",
    phase_one=False,
    fixed_nodes=None,
):
    """Globally optimize reduced welfare over ALL feasible zones for one label.

    `fixed_nodes` bypasses geometric constraints for scoring an already validated
    zone. Phase I prices geometric feasibility with zero original column cost.
    """
    if model != "grid":
        raise ValueError("Exact DW pricing currently certifies only finite-grid MID.")
    solver = _solver(deadline)
    if solver is None:
        return PricingResult("TIME_LIMIT", math.inf)
    p = pool.problem
    x = {n: solver.BoolVar(f"member_{n}") for n in p.nodes}
    if fixed_nodes is not None:
        for n in p.nodes:
            solver.Add(x[n] == int(n in fixed_nodes))
    else:
        for n, fixed in (p.fixed or {}).items():
            solver.Add(x[n] == int(fixed == zone))
        for n, allowed in (p.candidates or {}).items():
            if zone not in allowed:
                solver.Add(x[n] == 0)
        for (n, label), value in decisions.items():
            if label == zone:
                solver.Add(x[n] == value)
        # A freely selected root supplies one unit to each selected vertex.
        # Flow is possible only on edges whose endpoints are selected.
        roots = {n: solver.BoolVar("") for n in p.nodes}
        supplies = {n: solver.NumVar(0, p.A, "") for n in p.nodes}
        flows = {
            (a, b): solver.NumVar(0, p.A, "")
            for u, v in p.G.edges
            for a, b in ((u, v), (v, u))
        }
        solver.Add(solver.Sum(roots.values()) == 1)
        for n in p.nodes:
            solver.Add(roots[n] <= x[n])
            solver.Add(supplies[n] <= p.A * roots[n])
            solver.Add(
                supplies[n]
                + solver.Sum(flows[v, n] - flows[n, v] for v in p.G.neighbors(n))
                == x[n]
            )
        for (u, v), flow in flows.items():
            solver.Add(flow <= p.A * x[u])
            solver.Add(flow <= p.A * x[v])
        students = solver.Sum(p.students(n) * x[n] for n in p.nodes)
        for row in pool.constraints:
            value = solver.Sum(row.value(n) * x[n] for n in p.nodes)
            if row.lower_ratio is not None:
                solver.Add(
                    value >= row.lower_ratio * students - ZONE_FEASIBILITY_TOLERANCE
                )
            if row.upper_ratio is not None:
                solver.Add(
                    value <= row.upper_ratio * students + ZONE_FEASIBILITY_TOLERANCE
                )
        if pool.school_total:
            schools = solver.Sum(p.num_schools(n) * x[n] for n in p.nodes)
            average = pool.school_total / p.Z
            solver.Add(schools >= max(0, average - 1) - ZONE_FEASIBILITY_TOLERANCE)
            solver.Add(schools <= average + 1 + ZONE_FEASIBILITY_TOLERANCE)
    edges = []
    for u, v in p.G.edges:
        cut = solver.BoolVar("")
        solver.Add(cut >= x[u] - x[v])
        solver.Add(cut >= x[v] - x[u])
        solver.Add(cut <= x[u] + x[v])
        solver.Add(cut <= 2 - x[u] - x[v])
        edges.append(p.boundary_weight(u, v) * cut / 2)
    perimeter = solver.Sum(edges)
    score = (
        solver.Sum([])
        if phase_one
        else -perimeter
        if pool.market is None
        else _welfare_model(solver, pool.market, x, pool.lottery_scale)
    )
    reduced = score
    if master is not None:
        reduced = (
            score
            - solver.Sum(master.node_duals[n] * x[n] for n in p.nodes)
            - master.zone_duals[zone]
            - master.boundary_dual * perimeter
        )
    solver.Maximize(reduced)
    if time.monotonic() >= deadline:
        return PricingResult("TIME_LIMIT", math.inf)
    if math.isfinite(deadline):
        solver.SetTimeLimit(max(1, int((deadline - time.monotonic()) * 1000)))
    status = solver.Solve()
    if status == solver.INFEASIBLE:
        return PricingResult("INFEASIBLE", -math.inf)
    if status not in (solver.OPTIMAL, solver.FEASIBLE):
        return PricingResult(
            "TIME_LIMIT" if status == solver.NOT_SOLVED else "ERROR", math.inf
        )
    nodes = frozenset(n for n in p.nodes if x[n].solution_value() > 0.5)
    return PricingResult(
        "OPTIMAL" if status == solver.OPTIMAL else "FEASIBLE",
        solver.Objective().BestBound(),
        nodes,
        score.solution_value(),
        solver.Objective().Value(),
    )

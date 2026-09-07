"""Complete whole-zone branch-and-price, with Phase-I feasibility pricing.

At each node, pricing sees the same membership fixings as the restricted LP.
Only pricing-certified bounds prune the search. Branching on x[n,z] is complete:
if every such marginal is integral, every positive column of each label has the
same membership, so the deduplicated column solution is integral too.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from optimization.zone_columns import boundary_limit, solve_master
from optimization.zone_pricing import compatible, price_zone


@dataclass
class BranchPriceResult:
    selected: tuple
    status: str
    upper_bound: float | None
    reason: str
    nodes: int
    pricing_calls: int
    lp_iterations: int
    history: list


def _dual_objective(problem, lp):
    value = sum(lp.node_duals.values()) + sum(lp.zone_duals.values())
    if problem.boundary_prop >= 0:
        value += lp.boundary_dual * boundary_limit(problem)
    return value


def branch_and_price(
    pool,
    *,
    deadline=math.inf,
    model="grid",
    tolerance=1e-6,
    incumbent=(),
    pricer=price_zone,
):
    p = pool.problem
    if model != "grid":
        raise ValueError("Exact DW search currently certifies only finite-grid MID.")
    best = tuple(incumbent)
    lower = sum(c.score for c in best) if best else -math.inf
    initial_bound = (
        sum(max(t.utility_sums, default=0.0) for t in pool.market.types)
        if pool.market is not None
        else 0.0
    )
    pending = [({}, initial_bound)]
    closed_bound = -math.inf
    visited = pricing_calls = lp_iterations = 0
    history = []
    reason = "tree_exhausted"

    def consider(columns):
        nonlocal best, lower
        if len(columns) != p.Z or {c.zone for c in columns} != set(range(p.Z)):
            return
        if sum(len(c.nodes) for c in columns) != p.A:
            return
        if frozenset().union(*(c.nodes for c in columns)) != pool.nodes:
            return
        if p.boundary_prop >= 0:
            if sum(c.perimeter for c in columns) / 2 > boundary_limit(p):
                return
        value = sum(c.score for c in columns)
        if value > lower:
            best, lower = tuple(columns), value

    while pending and time.monotonic() < deadline:
        decisions, node_bound = pending.pop()
        if best and node_bound <= lower + tolerance:
            closed_bound = max(closed_bound, node_bound)
            continue
        visited += 1
        phase_one = True
        finished = False
        while time.monotonic() < deadline:
            columns = tuple(
                c for c in pool.columns.values() if compatible(c, decisions)
            )
            lp = solve_master(
                p, columns, deadline - time.monotonic(), phase_one=phase_one
            )
            lp_iterations += 1
            if lp.status != "OPTIMAL":
                reason = "master_" + lp.status.lower()
                break
            if phase_one and lp.artificial_mass <= tolerance:
                phase_one = False
                continue
            added = 0
            bounds = []
            complete = True
            for zone in range(p.Z):
                result = pricer(
                    pool,
                    zone,
                    lp,
                    decisions,
                    deadline=deadline,
                    model=model,
                    phase_one=phase_one,
                )
                pricing_calls += 1
                if result.status == "INFEASIBLE":
                    # One label has no possible zone: this branch is infeasible.
                    finished = True
                    break
                bounds.append(result.bound)
                if result.status != "OPTIMAL":
                    complete = False
                if result.status == "OPTIMAL" and result.reduced_cost > tolerance:
                    # Independent least-cutoff scoring also handles Phase I,
                    # where the pricing objective contains no welfare terms.
                    column = pool.column(zone, result.nodes)
                    if lp.reduced_cost(column) > tolerance:
                        added += pool.add(column)
            if finished:
                break
            # Correct the LP dual objective by the global pricing bound for
            # EACH label. This is valid even before column generation ends.
            certified_bound = (
                _dual_objective(p, lp) + sum(max(0.0, b) for b in bounds)
                if len(bounds) == p.Z
                else math.inf
            )
            if not phase_one:
                node_bound = min(node_bound, certified_bound)
            history.append(
                {
                    "node": visited,
                    "phase_one": phase_one,
                    "restricted_lp": lp.objective,
                    "columns_added": added,
                    "pricing_complete": complete,
                    "node_bound": node_bound if math.isfinite(node_bound) else None,
                }
            )
            if not complete:
                reason = "pricing_incomplete"
                break
            if added:
                continue
            if phase_one:
                if certified_bound < -tolerance:
                    finished = True
                else:
                    reason = "phase_one_numerical_stall"
                break
            # A positive unexplained residual must not be called convergence.
            if max(bounds, default=0.0) > tolerance:
                reason = "pricing_numerical_stall"
                break
            # The integer restricted master is only an incumbent heuristic.
            if time.monotonic() < deadline:
                integer = solve_master(
                    p,
                    columns,
                    min(1.0, (deadline - time.monotonic()) / 10),
                    integer=True,
                )
                if integer.selected:
                    consider(integer.selected)
            if best and node_bound <= lower + tolerance:
                closed_bound = max(closed_bound, node_bound)
                finished = True
                break
            marginals = {
                (n, z): sum(
                    value
                    for c, value in zip(columns, lp.values)
                    if c.zone == z and n in c.nodes
                )
                for n in p.nodes
                for z in range(p.Z)
            }
            fractional = [
                (key, value)
                for key, value in marginals.items()
                if key not in decisions and tolerance < value < 1 - tolerance
            ]
            if not fractional:
                selected = tuple(
                    c for c, value in zip(columns, lp.values) if value > 0.5
                )
                consider(selected)
                if best and node_bound <= lower + tolerance:
                    closed_bound = max(closed_bound, node_bound)
                    finished = True
                else:
                    reason = "integrality_numerical_stall"
                break
            (node, label), _ = min(fractional, key=lambda item: abs(item[1] - 0.5))
            excluded = {**decisions, (node, label): 0}
            included = {**decisions, **{(node, z): int(z == label) for z in range(p.Z)}}
            pending.extend(((excluded, node_bound), (included, node_bound)))
            finished = True
            break
        if not finished:
            pending.append((decisions, node_bound))
            if time.monotonic() >= deadline:
                reason = "time_limit"
            break
    if pending and time.monotonic() >= deadline:
        reason = "time_limit"
    upper = max([lower, closed_bound, *(bound for _, bound in pending)])
    status = (
        ("FEASIBLE" if best else "UNKNOWN")
        if pending
        else ("OPTIMAL" if best else "INFEASIBLE")
    )
    return BranchPriceResult(
        best,
        status,
        upper if math.isfinite(upper) else None,
        reason,
        visited,
        pricing_calls,
        lp_iterations,
        history,
    )

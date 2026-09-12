"""Complete whole-zone branch-and-price, with Phase-I feasibility pricing.

At each node, pricing sees the same membership fixings as the restricted LP.
Only pricing-certified bounds prune the search. Branching on x[n,z] is complete:
if every such marginal is integral, every positive column of each label has the
same membership, so the deduplicated column solution is integral too.

REVISIT -- plain column generation is the wrong solving method here, and the
pricer is no longer what is holding this back. Measured on BlockGroup_0 with a
200-partition ReCom pool (404 columns, 132 of 579 node duals nonzero) and a
short per-label slice so eight rounds fit: three to six columns of genuinely
positive reduced cost entered every round and the master LP stayed at
11,142.80 for all eight. A positive reduced cost that does not improve the
objective is a **degenerate pivot** -- the entering column's step length is
zero. Set partitioning is massively degenerate, and pricing each label
independently against near-zero duals returns *oversized* zones (340 to 392
nodes out of 579), so no Z of them can tile the graph and none can enter a
cover.

Things to try, roughly in order of effort:

* Dual stabilization. Smoothing (Wentges/Pessoa), a trust region or a penalty
  box on the duals, or interior rather than vertex duals -- ``solve_master``
  pins ``Method = 1`` for reproducibility, so ``Method = 2`` (barrier) is a
  one-line experiment.
* Pricing that knows its columns must be mutually completable, rather than Z
  independent maximizations, e.g. a cardinality or student-mass target per
  label taken from the incumbent.
* A cheaper recourse, so that many more rounds fit per hour: see the note in
  ``optimization/zone_pricing.py`` about pricing the SAA stable-admissions LP
  instead of the finite-grid MID cutoff recurrence.

Note also that 600s at BlockGroup_0 buys exactly one column-generation round at
the even per-label split below, so nothing about convergence is observable at
that budget.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from optimization.zone_columns import boundary_limit, solve_master
from optimization.zone_pricing import ZonePricer, compatible


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
    # One model per zone label for the whole search, however many nodes and
    # column-generation rounds it took. ``None`` when the caller supplied its
    # own pricer.
    pricing_models: int | None = None


def _column_or_none(pool, zone, nodes):
    """A time-limited solve can return a zone the pool's exact test rejects.

    The MIP works to a feasibility tolerance; ``ZonePool.feasible`` re-checks
    connectedness and the balance rows exactly. A hair's-breadth violation is
    not a column, and it is not an error either.
    """
    try:
        return pool.column(zone, nodes)
    except ValueError:
        return None


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
    pricer=None,
    workers=0,
):
    if model != "grid":
        raise ValueError("Exact DW search currently certifies only finite-grid MID.")
    # One live model per label for the whole search, unless the caller brought
    # its own pricer (the tests use enumerative ones).
    owned = ZonePricer(pool, model=model, workers=workers) if pricer is None else None
    pricer = owned if pricer is None else pricer
    try:
        return _search(
            pool, deadline, model, tolerance, incumbent, pricer, owned
        )
    finally:
        if owned is not None:
            owned.close()


def _search(pool, deadline, model, tolerance, incumbent, pricer, owned):
    p = pool.problem
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
            for priced, zone in enumerate(range(p.Z)):
                # Split what is left of the round evenly over the labels still
                # to price. Handing the first label the whole budget is what
                # made a certified bound unreachable: it needs a finite bound
                # from EVERY label, and one label would consume everything and
                # leave the rest unpriced. Infinity divides to infinity, so an
                # unlimited search is unaffected.
                share = (deadline - time.monotonic()) / (p.Z - priced)
                result = pricer(
                    pool,
                    zone,
                    lp,
                    decisions,
                    deadline=min(deadline, time.monotonic() + share),
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
                # Optimality is needed to *bound*, never to add a column: any
                # zone with positive reduced cost improves the master, however
                # it was found. Independent least-cutoff scoring also handles
                # Phase I, where the pricing objective has no welfare terms.
                if result.reduced_cost is not None and result.reduced_cost > tolerance:
                    column = _column_or_none(pool, zone, result.nodes)
                    if column is not None and lp.reduced_cost(column) > tolerance:
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
            if added:
                # Progress is progress. An incomplete round that still found
                # improving columns is an ordinary column-generation round, so
                # re-solve the master rather than abandoning the node. The node
                # still cannot be *closed* without a complete round, because
                # closing happens only past the check below.
                continue
            if not complete:
                reason = "pricing_incomplete"
                break
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
        owned.models_built if owned is not None else None,
    )

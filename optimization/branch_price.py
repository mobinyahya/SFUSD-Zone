"""Whole-zone branch-and-price, with Phase-I feasibility pricing.

At every node of the tree, pricing sees the same membership fixings as the
restricted LP, and only pricing-certified bounds prune. Branching on
``x[v,z]`` is complete: if every such marginal is integral then every
positive-weight column of a label has the same membership, so the
deduplicated LP solution is an integer partition.

What changed, and why
---------------------
The previous build priced correctly and still never converged, and the reason
was on the dual side rather than the pricing side. Measured on
:math:`\\text{BlockGroup}_0` with a 404-column pool, three to six columns of
genuinely positive reduced cost entered every round and the master LP sat at
11,142.80 for eight consecutive rounds. A positive reduced cost that does not
improve the objective is a *degenerate pivot*: the entering column's step
length is zero. Set partitioning is massively degenerate, only 132 of 579
cover rows carried a nonzero dual, and pricing six labels independently
against near-zero prices returned six zones of 340 to 392 vertices out of 579
-- no six of which can tile the graph, so none could enter a cover.

Three changes address the dual side of that, and all three are shipped:

* The prices are different. The master is solved by barrier without crossover,
  so the duals are an interior point of the dual polyhedron rather than a
  degenerate vertex, and they are smoothed across rounds (Wentges). Both are
  free in rigour: Proposition 9's bound holds at any dual point with
  ``mu >= 0``.
* The zones are anchored. Each label prices over sets containing its own
  centroid and drawn from its own candidate set, so it can no longer return a
  zone covering two thirds of the district.
* Each pricing call contributes several columns rather than one, and each round
  gets a bounded pricing budget that doubles only when a round neither adds a
  column nor proves a bound. Uncapped, the first round on a real instance ate
  the whole budget: 240s bought two master LPs.

REVISIT -- none of that is sufficient, and the reason is *not* the dual
degeneracy those remedies were chosen for. Measured on ``Block_2`` (501 units,
six anchors, 4,154 applicants) seeded with one ``cp_bool`` feasibility hint,
both welfare objectives ran 21 rounds and admitted 542 to 555 columns of
strictly positive reduced cost in 600s while the restricted LP stayed at the
hint value to four decimal places on every round, leaving a pool that still
admits exactly one tiling. The same on a district-sized synthetic instance
(576 vertices, 4,032 applicants, |Gamma| = 12,096): all four combinations of
{vertex, interior} x {alpha = 1, alpha = 0.5} admitted 400 to 424 such columns
in ten rounds and left the LP at the seed value 13,161.9302, and deleting the Z
seed columns from the resulting 413-column pool makes the integer master
INFEASIBLE -- **the pool contains exactly one tiling.**

The obstruction is a dimension count, not a choice of basis. The restricted LP
carries ``|V| + Z`` equality rows -- 582 here -- against however many columns
have been generated, so while the pool is small relative to ``|V|`` the system
is over-determined and its solution set is essentially the tilings it happens
to contain. Pricing each label independently maximizes reduced welfare with no
reference to the other labels, so Z priced zones are under no pressure to be
mutually complementary, and the pool can grow by hundreds of individually
improving zones without acquiring a second tiling. What this module then
delivers is recombination of *seeded* partitions through a set-partitioning
master: a useful primal heuristic, not column generation.

Two repairs, in order of effort, neither implemented:

* An **elastic master**: replace the cover equality by
  ``sum(lambda) + d_v - e_v = 1`` with ``d, e >= 0`` penalized at a large M,
  i.e. Phase I extended to Phase II. The LP becomes full-dimensional so a
  column can enter with a positive step length and the duals become
  informative; it stays a relaxation of the exact master, so Proposition 9
  survives, and a zero-slack optimum is still a partition.
* **Completability-targeted pricing**: a cardinality or student-mass window per
  label read off the incumbent, imposed on the column-generating pass while the
  bound-producing pass still ranges over all of ``F_z``.

Until one of them lands, the bound this module reports does not improve on the
a-priori welfare constants. Note that pricing is no longer what holds it back:
anchoring plus the CP-SAT formulation took a label's pricing solve from ~600s
for a 0.62% gap to 0.1-8s proved optimal.

Certification is also stated more carefully than before. CP-SAT reports a
proven objective bound even on an interrupted solve, so a label contributes a
valid bound whenever that number is finite -- strictly more often than "the
solve finished". What a *closed* node needs is different: every label proved
optimal, and no label found a zone improving at the LP's own duals.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from optimization.zone_columns import boundary_limit, smooth, solve_master
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
    columns_added: int = 0
    bound_slack: float = 0.0
    pricing_models: int | None = None


def _price_labels(pricer, pool, duals, decisions, deadline, phase_one):
    """Price every label, using the pricer's own parallel entry point if it has one."""

    price_all = getattr(pricer, "price_all", None)
    if price_all is not None:
        return price_all(
            pool, duals, decisions, deadline=deadline, phase_one=phase_one
        )
    labels = tuple(range(pool.problem.Z))
    results = {}
    for priced, zone in enumerate(labels):
        share = (deadline - time.monotonic()) / (len(labels) - priced)
        results[zone] = pricer(
            pool,
            zone,
            duals,
            decisions,
            deadline=min(deadline, time.monotonic() + share),
            phase_one=phase_one,
        )
    return results


def branch_and_price(
    pool,
    *,
    deadline=math.inf,
    tolerance=1e-6,
    incumbent=(),
    pricer=None,
    workers=0,
    seed=42,
    master_method="barrier",
    dual_smoothing=1.0,
    pricing_scale=1000,
    pricing_columns_per_call=8,
    pricing_parallel=True,
    pricing_time_limit=math.inf,
    redraw=None,
    redraw_time_limit=math.inf,
):
    owned = (
        ZonePricer(
            pool,
            workers=workers,
            seed=seed,
            scale=pricing_scale,
            columns_per_call=pricing_columns_per_call,
            parallel=pricing_parallel,
        )
        if pricer is None
        else None
    )
    pricer = owned if pricer is None else pricer
    try:
        return _search(
            pool,
            deadline,
            tolerance,
            incumbent,
            pricer,
            owned,
            master_method,
            dual_smoothing,
            pricing_time_limit,
            redraw,
            redraw_time_limit,
        )
    finally:
        if owned is not None:
            owned.close()


def _search(
    pool,
    deadline,
    tolerance,
    incumbent,
    pricer,
    owned,
    master_method,
    dual_smoothing,
    pricing_time_limit=math.inf,
    redraw=None,
    redraw_time_limit=math.inf,
):
    p = pool.problem
    best = tuple(incumbent)
    lower = sum(c.score for c in best) if best else -math.inf
    pending = [({}, pool.objective.upper_bound())]
    closed_bound = -math.inf
    visited = pricing_calls = lp_iterations = added_total = 0
    history = []
    reason = "tree_exhausted"
    # How far above the true reduced-welfare optimum a proved pricing bound can
    # sit, purely from the integer scaling of the pricing objective. Every node
    # bound carries it, so the incumbent comparison has to as well or a node
    # whose LP is already solved never closes. This is the search's absolute
    # optimality tolerance and is reported as such.
    slack = 0.0
    # How long one column-generation round may spend pricing. Without a cap the
    # first round on a real instance consumes the whole budget -- measured: 240s
    # bought exactly two master LPs -- so nothing about convergence is
    # observable. The cap doubles whenever a round neither adds a column nor
    # proves a bound, which is the only situation in which it was the binding
    # constraint. Short rounds therefore do the column generation and long ones
    # do the certification, without either being configured by hand.
    cap = float(pricing_time_limit)
    if cap <= 0:
        raise ValueError("dw_pricing_time_limit must be positive or infinity.")

    def closable(bound) -> bool:
        return bool(best) and bound <= lower + tolerance + slack

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

    def harvest(lp, point, results):
        """Store every admissible improving zone; report what it means.

        A column improving at the *smoothed* prices is worth keeping -- that is
        where stabilization earns its keep -- but only one improving at the
        LP's own prices is evidence that the node's LP is unsolved.
        """

        added = improving = 0
        for zone, result in sorted(results.items()):
            for nodes in result.candidates:
                column = pool.admit(zone, nodes)
                if column is None:
                    continue
                exact = lp.reduced_cost(column)
                if max(exact, point.reduced_cost(column)) <= tolerance:
                    continue
                if exact > tolerance:
                    improving += 1
                if pool.add(column):
                    added += 1
        return added, improving

    while pending and time.monotonic() < deadline:
        decisions, node_bound = pending.pop()
        if closable(node_bound):
            closed_bound = max(closed_bound, node_bound)
            continue
        visited += 1
        phase_one = True
        finished = False
        # Smoothing is per node and per phase: a Phase-I dual point satisfies
        # the artificials' dual rows and a Phase-II point need not, so mixing
        # them would break the convex-combination argument.
        centre = None
        previous_objective = None
        while time.monotonic() < deadline:
            columns = tuple(
                c for c in pool.columns.values() if compatible(c, decisions)
            )
            lp = solve_master(
                p,
                columns,
                deadline - time.monotonic(),
                phase_one=phase_one,
                method=master_method,
            )
            lp_iterations += 1
            if lp.status != "OPTIMAL":
                reason = "master_" + lp.status.lower()
                break
            if phase_one and lp.artificial_mass <= tolerance:
                phase_one = False
                centre = None
                previous_objective = None
                continue

            # Price at the smoothed point first; if it mis-prices -- finds no
            # improving zone -- fall back to the LP's own duals before drawing
            # any conclusion, and move the stabilization centre there. Without
            # that fallback a smoothed point can report convergence the LP does
            # not have, and its dual objective gives a needlessly weak bound.
            raw = lp.duals()
            centre = smooth(centre, raw, dual_smoothing)
            attempts = [centre] if dual_smoothing >= 1.0 else [centre, raw]
            infeasible = False
            round_deadline = min(deadline, time.monotonic() + cap)
            for index, point in enumerate(attempts):
                results = _price_labels(
                    pricer, pool, point, decisions, round_deadline, phase_one
                )
                pricing_calls += len(results)
                if any(r.status == "INFEASIBLE" for r in results.values()):
                    # One label admits no zone at all under this branch.
                    infeasible = True
                    break
                added, improving = harvest(lp, point, results)
                if added or improving or index == len(attempts) - 1:
                    break
            if infeasible:
                finished = True
                break
            if point is raw:
                centre = raw
            added_total += added
            bounds = [result.bound for result in results.values()]
            proved = all(r.status == "OPTIMAL" for r in results.values())
            slack = max(slack, sum(r.allowance for r in results.values()))
            certified = (
                point.dual_objective(p) + sum(max(0.0, b) for b in bounds)
                if all(math.isfinite(b) for b in bounds)
                else math.inf
            )
            if not phase_one:
                node_bound = min(node_bound, certified)
            history.append(
                {
                    "node": visited,
                    "phase_one": phase_one,
                    "restricted_lp": lp.objective,
                    "columns": len(columns),
                    "columns_added": added,
                    "stabilized": point is not raw,
                    "pricing_proved": proved,
                    "pricing_improving": improving,
                    "pricing_residual": max(bounds) if bounds else None,
                    "pricing_cap": cap if math.isfinite(cap) else None,
                    "bound_slack": slack,
                    "node_bound": node_bound if math.isfinite(node_bound) else None,
                }
            )
            # Test the pruning rule *before* spending another round on column
            # generation. Once the node's certified bound cannot beat the
            # incumbent, nothing a further column could do would change that,
            # and pricing on a district-sized instance keeps finding improving
            # columns indefinitely -- so a node whose incumbent already meets
            # its bound would otherwise burn the whole budget and report a time
            # limit instead of optimality. Measured: the incumbent reached the
            # certified bound in round 3 of 12 and the search ran to the wall.
            if closable(node_bound):
                closed_bound = max(closed_bound, node_bound)
                finished = True
                break

            # The measured pathology is a restricted LP that does not move
            # although pricing keeps finding columns of strictly positive
            # reduced cost: the pool holds one tiling, so every entering
            # column has a zero step length. That is the signal to redraw a
            # pair of zones, and waiting for pricing to be *exhausted* is too
            # late -- on a district-sized instance under a finite budget it
            # never is. Measured without this trigger: ten rounds, 380
            # columns, one tiling, the LP still at the seed value. The redraw
            # writes no bound, so firing it early costs only its own time, and
            # its ``(pair, territory)`` memo makes every sweep after the first
            # nearly free until the incumbent moves. On Block_2 this trigger is
            # worth +14.7% (mid) and +17.0% (stable_matching) on the incumbent
            # against +0.000% without it.
            stalled = (
                previous_objective is not None
                and lp.objective <= previous_objective + tolerance
            )
            previous_objective = lp.objective
            if (
                redraw is not None
                and best
                and not phase_one
                and (stalled or not added)
                and time.monotonic() < deadline
            ):
                _, stats = redraw.sweep(
                    best,
                    deadline=min(deadline, time.monotonic() + redraw_time_limit),
                    decisions=decisions,
                    offer=consider,
                )
                history[-1]["redraw"] = {
                    key: value
                    for key, value in stats.items()
                    if key != "dw_redraw_records"
                }
                if stats["dw_redraw_columns_added"]:
                    added_total += stats["dw_redraw_columns_added"]
                    continue
            if added:
                # Progress is progress. An unproved round that still found
                # improving zones is an ordinary column-generation round, so
                # re-solve the master rather than abandoning the node. The node
                # still cannot be *closed* without a proved round, because
                # closing happens only past the checks below.
                continue
            if not proved:
                if cap < deadline - time.monotonic():
                    # The cap, not the model, is what stopped the round. Spend
                    # more on the next one rather than abandoning the node.
                    cap *= 2
                    continue
                reason = "pricing_incomplete"
                break
            if improving:
                # Pricing found an exactly improving column the pool already
                # held, so the LP reported optimality with that column present.
                # That is numerically impossible and not something to loop on.
                reason = "column_generation_stall"
                break
            if phase_one:
                if certified < -tolerance:
                    # A strictly negative certified Phase-I bound proves that
                    # no admissible partition satisfies this branch.
                    finished = True
                else:
                    reason = "phase_one_numerical_stall"
                break
            # The node's LP is solved as far as pricing can tell. The residual
            # ``max(bounds)`` need not reach zero: the pricing objective is
            # scaled and directionally rounded, so its proven optimum sits
            # slightly *above* the true reduced welfare. That residual stays in
            # ``node_bound`` -- it is never discarded and never called zero --
            # and ``slack`` is what lets the node close in spite of it.
            #
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
            if closable(node_bound):
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
                if closable(node_bound):
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
        added_total,
        slack,
        owned.models_built if owned is not None else None,
    )

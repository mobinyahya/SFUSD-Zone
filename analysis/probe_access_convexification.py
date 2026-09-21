#!/usr/bin/env python3
"""Can any valid inequality in co-zoning space close the joint MID bound?

The joint model's root relaxation sits at 17,377 against a best-known zoning of
about 14,950, and the leak is not the all-access corner: at the root optimum the
mean co-zoning weight is 2.34 of an available 10.67 and the access pairs the
welfare model uses average ``a = 0.68``.  The relaxation wants *fractional*
access, because a fractional ``a[v, u]`` buys a pro-rata slice of a school no
integral zoning could hand out.

That raises a decidable question, and it decides the whole architecture.  Write
``L(a)`` for the welfare LP with the co-zoning matrix held fixed at ``a``.  Take
feasible zonings ``x^1..x^N`` with co-zoning matrices ``a^1..a^N``, and let
``abar`` be their average.  ``abar`` is by construction a point of the convex
hull of feasible co-zoning matrices, so *every* valid inequality in ``a``-space
holds at ``abar``.  Therefore if

    L(abar)  >>  mean_k L(a^k)

the excess is a convexification gap that no valid inequality in ``a``-space can
remove -- adding co-zoning cuts to the master cannot help, and the welfare model
itself has to change.  If instead ``L(abar)`` is close to the mean, the leak is
an ``a``-space consistency failure and cutting is the right response.

The same run also splits the total gap into its two independent parts, by
evaluating each zoning three ways: the exact finite-grid oracle, the welfare LP
at that zoning's integral ``a``, and the joint root bound.  The middle term
isolates how much of the gap is the welfare LP's own slack at fixed ``x`` (the
min-recurrence relaxation, documented at +0.8%) and how much is the zoning.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from analysis.probe_mid_root_bound import build_instance
from optimization.mid_oracle import cutoff_upper_bounds, finite_grid_oracle
from optimization.solvers import get_solver


def feasible_zonings(problem, count: int, seconds: float, workers: int) -> list[dict]:
    """Distinct feasible zonings, enumerated by the Boolean model itself.

    Enumeration keeps every core constraint -- contiguity, centroids, the FRL
    band, the school-count ceiling -- so each matrix really is a vertex of the
    co-zoning polytope and the average of them really is inside its hull.
    """
    solver = get_solver(
        "cp_bool",
        solve_time_limit=seconds,
        num_workers=workers,
        log_search_progress=False,
    )
    solutions = solver.enumerate_solutions(problem, count)
    seen: dict[tuple, dict] = {}
    for solution in solutions:
        if not solution.assignment:
            continue
        key = tuple(sorted(solution.assignment.items()))
        seen.setdefault(key, dict(solution.assignment))
    return list(seen.values())


class FixedAccessWelfare:
    """The finite-grid welfare LP with the co-zoning matrix supplied, not solved.

    Same rows as the joint model's welfare block: global cutoffs, thresholds
    clamped into the lottery interval, the ``min`` recurrence relaxed to its two
    ``<=`` sides plus the ``R >= R_prev + e - L`` tightening, and per-program
    capacity.  Only ``a`` changes between calls, so the model is built once and
    the access coefficients are re-stamped per evaluation.
    """

    def __init__(self, problem, market, lottery_scale: int):
        self.problem = problem
        self.market = market
        self.scale = float(lottery_scale)
        L = self.scale
        m = gp.Model()
        m.Params.OutputFlag = 0
        self.model = m

        bounds = cutoff_upper_bounds(market, lottery_scale)
        cutoffs = {
            program.program_id: m.addVar(
                lb=0.0, ub=min(float(bounds[program.program_id]), L * 40)
            )
            for program in market.programs
        }
        thresholds: dict[tuple[str, int], object] = {}

        def threshold(program_id: str, priority: int):
            key = (program_id, priority)
            if key not in thresholds:
                var = m.addVar(lb=0.0, ub=L)
                m.addConstr(var >= cutoffs[program_id] - priority * L)
                thresholds[key] = var
            return thresholds[key]

        # One access row per (student node, school node) pair in use; its
        # right-hand side is what gets re-stamped.
        self.access_rows: dict[tuple[int, int], list] = {}
        capacity_terms: dict[str, list] = {
            program.program_id: [] for program in market.programs
        }
        objective = gp.LinExpr()
        for student_type in market.types:
            previous = None
            for rank, (program_id, priority) in enumerate(
                zip(student_type.programs, student_type.priorities)
            ):
                program = market.program_by_id[program_id]
                thresh = threshold(program_id, priority)
                previous_expr = L if previous is None else previous
                remaining = m.addVar(lb=0.0, ub=L)
                m.addConstr(remaining <= previous_expr)
                if program.citywide or program.school_node == student_type.node:
                    m.addConstr(remaining <= thresh)
                    m.addConstr(remaining >= previous_expr + thresh - L)
                else:
                    effective = m.addVar(lb=0.0, ub=L)
                    m.addConstr(effective >= thresh)
                    # e >= L (1 - a): the access row, stamped per evaluation.
                    row = m.addConstr(effective >= 0.0)
                    self.access_rows.setdefault(
                        (student_type.node, program.school_node), []
                    ).append((row, effective))
                    m.addConstr(remaining <= effective)
                    m.addConstr(remaining >= previous_expr + effective - L)
                mass = previous_expr - remaining
                capacity_terms[program_id].append(student_type.count * mass)
                objective += student_type.scaled_utility_sums[rank] * mass
                previous = remaining

        for program in market.programs:
            terms = capacity_terms[program.program_id]
            if terms:
                m.addConstr(gp.quicksum(terms) <= L * program.capacity)

        self.denominator = L * float(market.utility_scale)
        m.setObjective(objective, GRB.MAXIMIZE)

    def value(self, access) -> float:
        """Welfare LP value with ``access(v, u)`` in [0, 1] supplied."""
        L = self.scale
        for (node, school_node), rows in self.access_rows.items():
            share = float(access(node, school_node))
            for row, effective in rows:
                # effective - L*(1-a) >= 0  ->  effective >= L - L*a
                row.RHS = L * (1.0 - share)
                self.model.chgCoeff(row, effective, 1.0)
        self.model.optimize()
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"fixed-access LP status {self.model.Status}")
        return float(self.model.ObjVal) / self.denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="benchmark/configs/priced_access_seeds.yaml"
    )
    parser.add_argument("--centroids", default="6-zone-3")
    parser.add_argument("--lottery-scale", type=int, default=20)
    parser.add_argument("--zonings", type=int, default=40)
    parser.add_argument("--enumerate-seconds", type=float, default=600.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()

    problem, market = build_instance(args.config, args.centroids)
    start = time.perf_counter()
    zonings = feasible_zonings(
        problem, args.zonings, args.enumerate_seconds, args.workers
    )
    print(
        f"enumerated {len(zonings)} distinct feasible zonings "
        f"in {time.perf_counter() - start:.0f}s",
        flush=True,
    )
    if len(zonings) < 2:
        raise SystemExit("Need at least two feasible zonings to average.")

    welfare = FixedAccessWelfare(problem, market, args.lottery_scale)

    def integral_access(assignment):
        def access(node, school_node):
            return 1.0 if assignment[node] == assignment[school_node] else 0.0

        return access

    exact, relaxed = [], []
    counts: dict[tuple[int, int], float] = {pair: 0.0 for pair in welfare.access_rows}
    for index, assignment in enumerate(zonings):
        oracle = finite_grid_oracle(market, assignment, args.lottery_scale)
        exact.append(oracle.welfare)
        relaxed.append(welfare.value(integral_access(assignment)))
        for node, school_node in counts:
            if assignment[node] == assignment[school_node]:
                counts[(node, school_node)] += 1.0
        print(
            f"  zoning {index:3d}: exact={oracle.welfare:11,.2f} "
            f"LP(a)={relaxed[-1]:11,.2f}  slack={relaxed[-1] - oracle.welfare:8,.2f}",
            flush=True,
        )

    shares = {pair: value / len(zonings) for pair, value in counts.items()}
    averaged = welfare.value(lambda node, school: shares[(node, school)])

    record = {
        "zonings": len(zonings),
        "exact_mean": statistics.mean(exact),
        "exact_max": max(exact),
        "lp_at_integral_mean": statistics.mean(relaxed),
        "lp_at_integral_max": max(relaxed),
        "lp_at_averaged_access": averaged,
        "mean_access_share": statistics.mean(shares.values()),
        "fixed_x_slack_mean": statistics.mean(r - e for r, e in zip(relaxed, exact)),
        "convexification_gap": averaged - statistics.mean(relaxed),
    }
    print("\n" + "=" * 72)
    print(
        f"exact MID welfare        mean {record['exact_mean']:12,.2f}"
        f"   max {record['exact_max']:12,.2f}"
    )
    print(
        f"welfare LP at integral a mean {record['lp_at_integral_mean']:12,.2f}"
        f"   max {record['lp_at_integral_max']:12,.2f}"
    )
    print(f"  => LP slack at fixed x      {record['fixed_x_slack_mean']:12,.2f}")
    print(
        f"welfare LP at averaged a      {record['lp_at_averaged_access']:12,.2f}"
        f"   (mean share {record['mean_access_share']:.3f})"
    )
    print(f"  => convexification gap      {record['convexification_gap']:12,.2f}")
    print("=" * 72)
    print(
        "A large convexification gap is unreachable by co-zoning cuts: the\n"
        "averaged matrix satisfies every valid inequality in a-space."
    )

    if args.output:
        Path(args.output).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

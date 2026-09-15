"""Sweep ``K`` for the budgeted-elastic DW master on the real instance.

Three stages, cheapest first, because the first two do not price at all:

0. Build the *diagnostic* pool -- the one the measured pathology produces.
   Seed one ``cp_bool`` feasible hint and run column generation with the
   redraw off, so the pool grows by hundreds of individually-improving zones
   and still holds exactly one tiling. That one-tiling property is asserted
   rather than assumed: delete the ``Z`` seed columns and the integer master
   must go INFEASIBLE.
1. Ladder ``K`` over that fixed pool with ``solve_master`` alone. No pricing,
   no CP-SAT, so this is seconds per point and it brackets the window.
2. For each ``K`` in the bracket, one master plus one pricing round, scored on
   the certified bound ``elastic_LP(K) + sum_z max(0, b_z)`` against the
   a-priori constant. That -- not the LP value -- is the acceptance metric,
   because the two terms move in opposite directions as ``K`` falls.

    uv run python -m analysis.probe_overlap_budget [--level Block_2]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import yaml

from optimization.branch_price import branch_and_price
from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.data.initial_solutions import initial_solution
from optimization.strategies.dantzig_wolfe import build_objective
from optimization.zone_columns import (
    ZonePool,
    overlap_limit,
    solve_master,
)
from optimization.zone_family import build_zone_family
from optimization.zone_pricing import ZonePricer

CONFIG = "optimization/dantzig_wolfe.example.yaml"
PROPS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5)


def build(level: str, objective: str):
    raw = yaml.safe_load(open(CONFIG))
    raw["levels"] = [level]
    raw["dw_objective"] = objective
    config = OptimizationConfig(**raw)
    dataset = config.make_dataset()
    solver = config.make_solver(output_dir="./optimization_output")
    problem = dataset.problem_for(LevelSpec.parse(level))
    problem.boundary_prop = float(config.boundary_prop)
    radius = int(solver.options.get("centroid_neighbor_radius", 0))
    family = build_zone_family(problem, centroid_neighbor_radius=radius)
    pool = ZonePool(
        problem,
        build_objective(problem, dataset.config, config.make_strategy().options),
        family=family,
    )
    return config, solver, problem, pool


def seed(problem, pool, solver):
    hint = initial_solution(problem, "feasible", solver_options=solver.options)
    problem.hint = hint.assignment
    columns = pool.admit_partition(hint.assignment)
    if columns is None:
        raise SystemExit("The feasible hint is not admissible; nothing to seed.")
    return columns


def one_tiling(problem, pool, seeds) -> bool:
    """The diagnostic: without the seed columns, can the pool cover V at all?"""

    rest = tuple(c for c in pool.columns.values() if c not in set(seeds))
    return solve_master(problem, rest, 120, integer=True).status == "INFEASIBLE"


def ladder(problem, pool, props, out):
    columns = tuple(pool.columns.values())
    rows = []
    for prop in props:
        K = overlap_limit(problem, prop)
        lp = solve_master(problem, columns, 300, method="dual", overlap_budget=K)
        rows.append(
            {
                "prop": prop,
                "K": K,
                "status": lp.status,
                "lp": lp.objective,
                "elastic_mass": lp.elastic_mass,
                "elastic_nodes": lp.elastic_nodes,
                "overlap_dual": lp.overlap_dual,
                "duals_pinned": lp.duals_pinned,
            }
        )
        print(
            f"  prop={prop:<6} K={K:9.2f} LP={lp.objective if lp.objective is None else round(lp.objective, 4)!s:>12}"
            f" mass={lp.elastic_mass:9.2f} nodes={lp.elastic_nodes:4d}"
            f" mu_K={lp.overlap_dual:9.5f} pinned={lp.duals_pinned:4d}",
            flush=True,
        )
    out["ladder"] = rows
    return rows


def certify(problem, pool, pricer, props, seconds, constant, out):
    columns = tuple(pool.columns.values())
    rows = []
    for prop in props:
        K = overlap_limit(problem, prop)
        lp = solve_master(problem, columns, 300, method="dual", overlap_budget=K)
        if lp.status != "OPTIMAL":
            print(f"  prop={prop}: master {lp.status}", flush=True)
            continue
        duals = lp.duals()
        results = pricer.price_all(
            pool, duals, {}, deadline=time.monotonic() + seconds, phase_one=False
        )
        bounds = [r.bound for r in results.values()]
        residual = sum(max(0.0, b) for b in bounds if math.isfinite(b))
        dual_objective = duals.dual_objective(problem, K)
        certified = dual_objective + residual
        rows.append(
            {
                "prop": prop,
                "K": K,
                "lp": lp.objective,
                "dual_objective": dual_objective,
                "residual": residual,
                "per_label": sorted(bounds),
                "certified": certified,
                "beats_constant": certified < constant,
                "proved": [r.status for r in results.values()],
            }
        )
        print(
            f"  prop={prop:<6} LP={lp.objective:12.4f} dual_obj={dual_objective:12.4f}"
            f" residual={residual:11.2f} certified={certified:12.2f}"
            f" {'BEATS' if certified < constant else 'no'} ({constant:.2f})",
            flush=True,
        )
    out["certified"] = rows
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="Block_2")
    ap.add_argument("--objective", default="mid")
    ap.add_argument("--pool-seconds", type=float, default=300.0)
    ap.add_argument("--price-seconds", type=float, default=120.0)
    ap.add_argument("--out", default="overlap_budget_sweep.json")
    args = ap.parse_args()

    out: dict = {"level": args.level, "objective": args.objective}
    t0 = time.monotonic()
    config, solver, problem, pool = build(args.level, args.objective)
    constant = pool.objective.upper_bound()
    print(
        f"{args.level}: nodes={problem.A} Z={problem.Z} "
        f"students={sum(problem.students(n) for n in problem.nodes):.0f} "
        f"weight={overlap_limit(problem, 1.0):.0f} constant={constant:.4f} "
        f"({time.monotonic() - t0:.1f}s)",
        flush=True,
    )
    out["nodes"] = problem.A
    out["constant"] = constant
    out["total_weight"] = overlap_limit(problem, 1.0)

    seeds = seed(problem, pool, solver)
    print(f"seeded {len(pool.columns)} columns, hint = {sum(c.score for c in seeds):.4f}", flush=True)

    print(f"\n[0] building the diagnostic pool ({args.pool_seconds:.0f}s, redraw off)", flush=True)
    search = branch_and_price(
        pool,
        deadline=time.monotonic() + args.pool_seconds,
        incumbent=seeds,
        workers=max(1, int(solver.options.get("workers", 1))),
        seed=int(config.seed),
        master_method="dual",
        overlap_prop=0.0,
        pricing_time_limit=float(config.dw_pricing_time_limit),
        redraw=None,
    )
    print(
        f"    columns={len(pool.columns)} added={search.columns_added} "
        f"rounds={search.lp_iterations} reason={search.reason}",
        flush=True,
    )
    out["pool_columns"] = len(pool.columns)
    out["pool_added"] = search.columns_added
    out["pool_reason"] = search.reason
    out["lp_trace"] = [h.get("restricted_lp") for h in search.history]
    out["one_tiling"] = one_tiling(problem, pool, seeds)
    print(f"    one tiling only: {out['one_tiling']}", flush=True)

    print("\n[1] K ladder on the fixed pool (no pricing)", flush=True)
    rows = ladder(problem, pool, PROPS, out)

    window = [
        r["prop"]
        for r in rows
        if r["prop"] > 0 and r["overlap_dual"] > 1e-9
    ] or [p for p in PROPS if p > 0][:3]
    print(f"\n[2] certified bound over the window {window}", flush=True)
    with ZonePricer(
        pool,
        workers=max(1, int(solver.options.get("workers", 1))),
        seed=int(config.seed),
        scale=int(config.dw_pricing_scale),
        columns_per_call=int(config.dw_pricing_columns_per_call),
        parallel=bool(config.dw_pricing_parallel),
    ) as pricer:
        certify(problem, pool, pricer, [0.0, *window], args.price_seconds, constant, out)

    out["wall_seconds"] = time.monotonic() - t0
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {args.out} in {out['wall_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

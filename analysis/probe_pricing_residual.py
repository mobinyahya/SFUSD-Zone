"""Does one label's pricing problem prove optimality on the real instance?

This is what separates the two explanations for the ~8,200-per-label residual
that keeps the DW bound pinned to the a-priori constant:

* **unproved** -- ``bound`` is a loose CP-SAT dual bound, the residual is
  slack rather than optimism, and a larger per-label budget shrinks it. A
  compute problem.
* **proved** -- ``bound`` is the true maximum reduced welfare over ``F_z``, the
  residual is real per-label optimism, and no amount of compute touches it.
  A formulation problem.

The docs' 0.1-8s-to-proved-optimal figure is from a *synthetic* instance. This
measures the real one, one label at a time, against a long budget, and reports
the bound trajectory so a plateau is visible even if optimality is never
proved.

    uv run python -m analysis.probe_pricing_residual --seconds 600 --labels 0,1
"""

from __future__ import annotations

import argparse
import json
import math
import time

import yaml

from optimization.config import OptimizationConfig
from optimization.data.initial_solutions import initial_solution
from optimization.levels import LevelSpec
from optimization.strategies.dantzig_wolfe import build_objective
from optimization.zone_columns import ZonePool, solve_master
from optimization.zone_family import build_zone_family
from optimization.zone_pricing import ZonePricer

CONFIG = "optimization/dantzig_wolfe.example.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="Block_2")
    ap.add_argument("--objective", default="mid")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", default="pricing_residual.json")
    args = ap.parse_args()

    raw = yaml.safe_load(open(CONFIG))
    raw["levels"] = [args.level]
    raw["dw_objective"] = args.objective
    config = OptimizationConfig(**raw)
    dataset = config.make_dataset()
    solver = config.make_solver(output_dir="./optimization_output")
    problem = dataset.problem_for(LevelSpec.parse(args.level))
    problem.boundary_prop = float(config.boundary_prop)
    family = build_zone_family(
        problem,
        centroid_neighbor_radius=int(solver.options.get("centroid_neighbor_radius", 0)),
    )
    pool = ZonePool(
        problem,
        build_objective(problem, dataset.config, config.make_strategy().options),
        family=family,
    )
    hint = initial_solution(problem, "feasible", solver_options=solver.options)
    problem.hint = hint.assignment
    seeds = pool.admit_partition(hint.assignment)
    if seeds is None:
        raise SystemExit("hint not admissible")

    lp = solve_master(problem, tuple(pool.columns.values()), 300, method="dual")
    duals = lp.duals()
    labels = (
        [int(x) for x in args.labels.split(",") if x != ""]
        if args.labels
        else list(range(problem.Z))
    )
    print(
        f"{args.level} seeded LP={lp.objective:.4f} constant="
        f"{pool.objective.upper_bound():.4f} labels={labels} "
        f"budget={args.seconds:.0f}s each",
        flush=True,
    )
    print(
        "  nonzero cover duals: "
        f"{sum(1 for v in lp.node_duals.values() if abs(v) > 1e-9)} of {problem.A}",
        flush=True,
    )

    rows = []
    with ZonePricer(
        pool,
        workers=max(1, int(solver.options.get("workers", 1))),
        seed=int(config.seed),
        scale=int(config.dw_pricing_scale),
        columns_per_call=int(config.dw_pricing_columns_per_call),
        parallel=False,
    ) as pricer:
        for zone in labels:
            t = time.monotonic()
            result = pricer(
                pool,
                zone,
                duals,
                {},
                deadline=time.monotonic() + args.seconds,
                phase_one=False,
            )
            elapsed = time.monotonic() - t
            incumbent = (
                duals.reduced_cost(pool.column(zone, result.nodes))
                if result.nodes and pool.feasible(zone, result.nodes)
                else None
            )
            gap = (
                None
                if incumbent is None or not math.isfinite(result.bound)
                else result.bound - incumbent
            )
            rows.append(
                {
                    "zone": zone,
                    "status": result.status,
                    "bound": result.bound,
                    "incumbent_reduced_cost": incumbent,
                    "gap": gap,
                    "allowance": result.allowance,
                    "seconds": elapsed,
                    "candidates": len(result.candidates),
                    "best_size": len(result.nodes) if result.nodes else 0,
                    "candidate_set": len(family.candidates[zone]),
                }
            )
            print(
                f"  label {zone}: {result.status:<11} bound={result.bound:12.2f}"
                f" incumbent={'n/a' if incumbent is None else f'{incumbent:12.2f}'}"
                f" gap={'n/a' if gap is None else f'{gap:11.2f}'}"
                f" {elapsed:6.1f}s size={len(result.nodes) if result.nodes else 0}"
                f"/{len(family.candidates[zone])}",
                flush=True,
            )

    proved = [r for r in rows if r["status"] == "OPTIMAL"]
    print(
        f"\nproved optimal: {len(proved)}/{len(rows)}; "
        f"sum of bounds = {sum(max(0.0, r['bound']) for r in rows):.2f}",
        flush=True,
    )
    with open(args.out, "w") as fh:
        json.dump(
            {"level": args.level, "seeded_lp": lp.objective, "labels": rows},
            fh,
            indent=2,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

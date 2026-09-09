#!/usr/bin/env python3
"""Does the exact cutoff formulation out-branch the MID monolith?

Everything measured so far says the welfare model is no longer the problem. At a
fixed zoning the cutoff formulation of `probe_cutoff_formulation` is exact --
its integral optimum reproduces deferred acceptance to 2.4e-11 -- and its linear
relaxation sits only 0.73-0.91% above the truth, against 3.2% for the MID
min-recurrence model and 1.8-2.3% for the stable-admissions LP the SAA oracle
solves. Yet all three joint root bounds land within 70 of each other, around
17,310-17,377, because the leak is the fractional co-zoning that access hands
the relaxation, not the welfare block.

Root bounds are not the whole story though: the `mid` strategy descends from
17,377 to 16,572.83 over 1800s of branch-and-bound. So the question this script
answers is whether a formulation with a tighter welfare block and a *fast* fixed
zoning subproblem -- 0.2-0.3s with (S1), against 8-55s without it -- converts
that into a better bound in the same budget.

The reference numbers, `mid` on 6-zone-3 / Block_2 at 1800s with 16 workers:
dual bound 16,572.83, incumbent 14,075.95, relative gap 17.74%. Single-seed DA
welfare and MID continuum welfare agree to about 6 units on this instance, so
the two are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from gurobipy import GRB

from analysis.probe_cutoff_formulation import (
    CutoffModel,
    build_individual_instance,
)
from optimization.data.saa import sample_school_preferences

MID_DUAL_BOUND = 16_572.83
MID_INCUMBENT = 14_075.95


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="benchmark/configs/priced_access_seeds.yaml"
    )
    parser.add_argument("--centroids", default="6-zone-3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--extras", default="S1,S2")
    parser.add_argument("--output")
    args = parser.parse_args()

    problem, market = build_individual_instance(args.config, args.centroids)
    sample = sample_school_preferences(market, 1, "STB", args.seed)[0]
    extras = tuple(part for part in args.extras.split(",") if part)

    model = CutoffModel(problem, market, sample, extras=extras)
    model.set_integral()
    # The zoning is the decision the bound is actually about; leaving it
    # continuous would just re-measure the root.
    for var in model.x.values():
        var.VType = GRB.BINARY
    # `a` and `both` are pinned by McCormick once `x` is integral, so they
    # need no branching of their own.

    model.model.Params.OutputFlag = 1
    model.model.Params.Threads = args.workers
    model.model.Params.MIPFocus = 0
    print(
        f"extras={extras or '(none)'}  rows={model.model.NumConstrs:,}  "
        f"cols={model.model.NumVars:,}  binaries={model.model.NumBinVars:,}",
        flush=True,
    )

    start = time.perf_counter()
    model.model.Params.TimeLimit = args.seconds
    model.model.optimize()
    elapsed = time.perf_counter() - start

    status = int(model.model.Status)
    solutions = int(model.model.SolCount)
    record = {
        "extras": list(extras),
        "status": status,
        "seconds": elapsed,
        "solution_count": solutions,
        "bound": float(model.model.ObjBound) / model.denominator,
        "incumbent": (
            float(model.model.ObjVal) / model.denominator if solutions else None
        ),
        "raw_incumbent": model.raw_value() if solutions else None,
        "relative_gap": float(model.model.MIPGap) if solutions else None,
        "rows": int(model.model.NumConstrs),
        "columns": int(model.model.NumVars),
        "binaries": int(model.model.NumBinVars),
        "mid_dual_bound": MID_DUAL_BOUND,
        "mid_incumbent": MID_INCUMBENT,
    }
    print("\n" + "=" * 76)
    print(f"status={status}  solutions={solutions}  wall={elapsed:,.0f}s")
    print(f"  dual bound      {record['bound']:12,.2f}   "
          f"(mid: {MID_DUAL_BOUND:,.2f} -> "
          f"{record['bound'] - MID_DUAL_BOUND:+,.2f})")
    if solutions:
        print(f"  incumbent       {record['incumbent']:12,.2f}   "
              f"(mid: {MID_INCUMBENT:,.2f} -> "
              f"{record['incumbent'] - MID_INCUMBENT:+,.2f})")
        print(f"  raw incumbent   {record['raw_incumbent']:12,.2f}")
        print(f"  relative gap    {100 * record['relative_gap']:11.2f}%   "
              f"(mid: 17.74%)")
    else:
        print("  no feasible zoning found")
    print("=" * 76)

    if args.output:
        Path(args.output).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

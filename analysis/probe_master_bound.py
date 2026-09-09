#!/usr/bin/env python3
"""Does the master's dual bound come from its cuts, or from the constant?

Solves the same cut-laden master twice, changing only the declared upper bound
on the objective variable. If the reported ``choice_best_bound`` tracks that
constant, the cuts and the assignment structure contribute nothing to the
relaxation and no amount of restructuring the cuts can tighten the bound.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark.slurm import create_plan
from choice.objective import ChoiceObjective
from optimization.data.initial_solutions import initial_solution
from optimization.data.saa import build_saa_market, sample_school_preferences
from optimization.levels import LevelSpec
from optimization.saa_oracle import SaaOracle
from optimization.solvers import get_solver
from optimization.strategies.saa import _configure_problem
from optimization.welfare_bounds import welfare_upper_bound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--centroids", default="6-zone-3")
    parser.add_argument("--cut-rounds", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan = create_plan(args.config)
    config = plan.tasks[0].optimization_config()
    config.centroids_type = args.centroids
    options = config.make_strategy().options
    solver_options = dict(config.make_solver().options)
    target = LevelSpec.parse(config.levels[-1])
    dataset = config.make_dataset()
    base = dataset.problem_for(target)
    _configure_problem(base, options)

    market = build_saa_market(base, dataset.config)
    sample = sample_school_preferences(
        market, 1, str(options.get("saa_tie_breaking_method", "MTB")).upper(),
        int(options.get("seed", 42)),
    )[0]
    oracle = SaaOracle(market, sample, 0, base, workers=args.workers)
    transport = welfare_upper_bound("transport", market.programs, market.students)
    hint = initial_solution(base, options.get("hints", "voronoi"), solver_options=solver_options)
    assignment = hint.assignment if hint is not None else None
    print(f"transport constant: {transport:,.4f}", flush=True)

    # Accumulate real cuts first, so the comparison is made on a master that
    # actually has an outer approximation to work with.
    cuts = []
    for _ in range(args.cut_rounds):
        if assignment is None:
            break
        result = oracle.solve(assignment)
        cuts.append(result.cut.to_choice_cut())
        objective = ChoiceObjective(
            cuts=tuple(cuts), scale=float(options.get("choice_utility_scale", 100.0)),
            aggregate_cuts=True, total_lower_bound=0.0, total_upper_bound=transport,
        )
        problem = dataset.problem_for(target, hint=assignment, choice_objective=objective)
        _configure_problem(problem, options)
        solution = get_solver(
            config.solver, solve_time_limit=args.time_limit, workers=args.workers
        ).solve(problem)
        if solution.feasible:
            assignment = solution.assignment
    print(f"accumulated {len(cuts)} cuts", flush=True)

    rows = []
    for multiplier in (1.0, 2.0, 10.0):
        declared = transport * multiplier
        objective = ChoiceObjective(
            cuts=tuple(cuts), scale=float(options.get("choice_utility_scale", 100.0)),
            aggregate_cuts=True, total_lower_bound=0.0, total_upper_bound=declared,
        )
        problem = dataset.problem_for(target, hint=assignment, choice_objective=objective)
        _configure_problem(problem, options)
        start = time.perf_counter()
        solution = get_solver(
            config.solver, solve_time_limit=args.time_limit, workers=args.workers
        ).solve(problem)
        bound = solution.metadata.get("choice_best_bound")
        rows.append(
            {
                "multiplier": multiplier,
                "declared_upper_bound": declared,
                "reported_best_bound": bound,
                "objective": solution.objective,
                "status": solution.status,
                "seconds": time.perf_counter() - start,
                # If this stays ~0 the constant is the entire bound.
                "proved_below_constant": None if bound is None else declared - bound,
            }
        )
        print(
            f"declared {declared:,.2f} -> best bound "
            f"{bound if bound is None else format(bound, ',.2f')}  "
            f"(proved below constant: "
            f"{rows[-1]['proved_below_constant'] if bound is None else format(rows[-1]['proved_below_constant'], ',.2f')})",
            flush=True,
        )

    payload = {
        "centroids_type": args.centroids,
        "cuts": len(cuts),
        "transport_upper_bound": transport,
        "time_limit": args.time_limit,
        "probes": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

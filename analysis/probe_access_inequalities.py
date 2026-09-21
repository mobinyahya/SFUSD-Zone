#!/usr/bin/env python3
"""Measure what the triangle and cardinality inequalities do to the cut bound.

The reported SAA bound is ``min(a-priori constant, what the cuts can prove)``,
and on these instances the constant binds. So to see whether the inequalities
help, the declared constant is deliberately loosened (``--bound-multiplier``)
until the cuts are what binds; the reported ``choice_best_bound`` is then a
direct read on the strength of the relaxation.

The same accumulated cuts are reused for every variant, so the only thing that
changes between rows is which valid inequalities are present.
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

VARIANTS = {
    "baseline": {},
    "cardinality": {"choice_access_cardinality": True},
    "triangle": {"choice_access_triangle": True},
    "both": {"choice_access_cardinality": True, "choice_access_triangle": True},
    # The cut pairs are near-bipartite, so plain transitivity finds few triples;
    # this creates the missing side first.
    "triangle_completed": {
        "choice_access_triangle": True,
        "choice_access_complete": True,
    },
    "all": {
        "choice_access_cardinality": True,
        "choice_access_triangle": True,
        "choice_access_complete": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--centroids", default="6-zone-3")
    parser.add_argument("--cut-rounds", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--bound-multiplier", type=float, default=10.0)
    parser.add_argument("--triangle-limit", type=int, default=200_000)
    parser.add_argument("--completion-limit", type=int, default=20_000)
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
        market,
        1,
        str(options.get("saa_tie_breaking_method", "MTB")).upper(),
        int(options.get("seed", 42)),
    )[0]
    oracle = SaaOracle(market, sample, 0, base, workers=args.workers)
    transport = welfare_upper_bound("transport", market.programs, market.students)
    scale = float(options.get("choice_utility_scale", 100.0))
    hint = initial_solution(
        base, options.get("hints", "voronoi"), solver_options=solver_options
    )
    assignment = hint.assignment if hint is not None else None
    print(f"transport constant: {transport:,.4f}", flush=True)

    cuts = []
    for _ in range(args.cut_rounds):
        if assignment is None:
            break
        result = oracle.solve(assignment)
        cuts.append(result.cut.to_choice_cut())
        objective = ChoiceObjective(
            cuts=tuple(cuts),
            scale=scale,
            aggregate_cuts=True,
            total_lower_bound=0.0,
            total_upper_bound=transport,
        )
        problem = dataset.problem_for(
            target, hint=assignment, choice_objective=objective
        )
        _configure_problem(problem, options)
        solution = get_solver(
            config.solver, solve_time_limit=args.time_limit, workers=args.workers
        ).solve(problem)
        if solution.feasible:
            assignment = solution.assignment
    print(f"accumulated {len(cuts)} cuts", flush=True)

    declared = transport * args.bound_multiplier
    rows = []
    for label, flags in VARIANTS.items():
        objective = ChoiceObjective(
            cuts=tuple(cuts),
            scale=scale,
            aggregate_cuts=True,
            total_lower_bound=0.0,
            total_upper_bound=declared,
        )
        problem = dataset.problem_for(
            target, hint=assignment, choice_objective=objective
        )
        _configure_problem(problem, options)
        start = time.perf_counter()
        solution = get_solver(
            config.solver,
            solve_time_limit=args.time_limit,
            workers=args.workers,
            choice_access_triangle_limit=args.triangle_limit,
            choice_access_completion_limit=args.completion_limit,
            **flags,
        ).solve(problem)
        meta = solution.metadata
        rows.append(
            {
                "variant": label,
                "declared_upper_bound": declared,
                "cut_implied_bound": meta.get("choice_best_bound"),
                "objective": solution.objective,
                "status": solution.status,
                "access_pairs": meta.get("choice_access_pairs"),
                "cardinality_constraints": meta.get("choice_access_cardinality"),
                "triangle_triples": meta.get("choice_access_triangle"),
                "completed_pairs": meta.get("choice_access_completed"),
                "seconds": time.perf_counter() - start,
            }
        )
        bound = rows[-1]["cut_implied_bound"]
        print(
            f"{label:<12s} bound {'-' if bound is None else format(bound, ',.2f')}  "
            f"obj {'-' if solution.objective is None else format(solution.objective, ',.2f')}  "
            f"cliques {rows[-1]['cardinality_constraints']}  "
            f"triples {rows[-1]['triangle_triples']}  "
            f"completed {rows[-1]['completed_pairs']}  "
            f"pairs {rows[-1]['access_pairs']}  "
            f"({rows[-1]['seconds']:.0f}s)",
            flush=True,
        )

    payload = {
        "centroids_type": args.centroids,
        "cuts": len(cuts),
        "transport_upper_bound": transport,
        "declared_upper_bound": declared,
        "time_limit": args.time_limit,
        "variants": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

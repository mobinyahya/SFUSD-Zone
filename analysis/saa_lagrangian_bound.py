#!/usr/bin/env python3
"""Bound the SAA optimum by decomposing over scenarios instead of over cuts.

The SAA master reports an optimality gap measured against its own dual bound,
and on these instances that bound never leaves the a-priori constant the model
was handed -- so the reported gap is the distance to "every student gets their
first choice", not to the optimum. This script computes bounds that do not ask
the joint master to prove anything.

Writing ``W_s(x)`` for scenario ``s`` welfare under zoning ``x`` and
``a(x)`` for the same-zone indicators, with ``S`` scenarios:

  Perfect information (``--lambda-steps 0``)

      max_x (1/S) sum_s W_s(x)  <=  (1/S) sum_s max_x W_s(x)

  because the right side may pick a different zoning per scenario. Each inner
  problem is a single-scenario cutting-plane maximisation, and *its* dual bound
  is enough -- truncating it keeps the average valid.

  Lagrangian dual decomposition (``--lambda-steps k``)

      D(lambda) = (1/S) sum_s max_x [ W_s(x) + <lambda_s, a(x)> ],
      subject to sum_s lambda_s = 0.

  For any such lambda and any feasible x, setting every scenario copy to x makes
  the prices cancel, so ``D(lambda)`` bounds the optimum for *every* lambda --
  no convergence in lambda is required for validity. ``D(0)`` is the
  perfect-information bound, so minimising over lambda can only improve on it.
  ``D`` is convex in lambda with subgradient ``(1/S) a(x_s*)``; projected onto
  the zero-sum subspace that is ``(1/S)(a_s - mean_r a_r)``, which is exactly a
  price update that discourages scenarios from disagreeing with the consensus.

Every bound printed is valid. The question the script answers is whether any of
them is *tighter* than the a-priori constant.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from benchmark.slurm import create_plan
from choice.objective import ChoiceObjective
from optimization.data.saa import build_saa_market, sample_school_preferences
from optimization.data.initial_solutions import initial_solution
from optimization.levels import LevelSpec
from optimization.saa_oracle import SaaOracle
from optimization.solvers import get_solver
from optimization.welfare_bounds import first_choice_upper_bound, welfare_upper_bound


def access_pattern(assignment: dict[int, int], pairs) -> dict[tuple[int, int], int]:
    """Which of ``pairs`` share a zone under ``assignment``."""
    return {
        pair: int(assignment[pair[0]] == assignment[pair[1]])
        for pair in pairs
        if pair[0] in assignment and pair[1] in assignment
    }


def scenario_inner_max(
    dataset,
    options,
    target,
    oracle,
    welfare_bound,
    prices,
    *,
    backend: str,
    iterations: int,
    time_limit: float,
    workers: int,
    scale: float,
    hint: dict[int, int] | None,
) -> dict[str, Any]:
    """Cutting-plane maximisation of ``W_s(x) + <prices, a(x)>`` for one scenario.

    Returns the best *dual* bound seen (valid however early we stop), the best
    primal value found, and the access pattern of the best zoning -- the last is
    the subgradient ingredient.
    """
    from optimization.strategies.saa import _configure_problem

    cuts: list = []
    access_terms = tuple(prices.items())
    best_bound = math.inf
    best_value = -math.inf
    best_access: dict[tuple[int, int], int] = {}
    best_welfare = -math.inf
    current_hint = hint
    for iteration in range(iterations):
        objective = ChoiceObjective(
            cuts=tuple(cuts),
            scale=scale,
            aggregate_cuts=True,
            total_lower_bound=0.0,
            total_upper_bound=welfare_bound,
            access_terms=access_terms,
        )
        problem = dataset.problem_for(
            target, hint=current_hint, choice_objective=objective
        )
        _configure_problem(problem, options)
        solver = get_solver(
            backend,
            solve_time_limit=time_limit,
            workers=workers,
            linearization_level=int(options.get("linearization_level", 1)),
        )
        solver._solve_count = iteration
        solution = solver.solve(problem)
        if not solution.feasible:
            continue
        bound = solution.metadata.get("choice_best_bound")
        if bound is None:
            bound = solution.objective
        # Every master is a relaxation of the inner max, so its dual bound is a
        # valid bound on the inner max and the best one so far is what counts.
        best_bound = min(best_bound, float(bound))

        result = oracle.solve(solution.assignment)
        cuts.append(result.cut.to_choice_cut())
        current_hint = solution.assignment

        access = access_pattern(solution.assignment, prices)
        value = result.welfare + sum(
            price * access.get(pair, 0) for pair, price in prices.items()
        )
        if value > best_value:
            best_value = value
            best_access = access
            best_welfare = result.welfare
    return {
        "dual_bound": best_bound,
        "primal_value": best_value,
        "welfare": best_welfare,
        "access": best_access,
        "cuts": len(cuts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Sweep YAML to take the problem from.")
    parser.add_argument("--centroids", default=None, help="Override centroids_type.")
    parser.add_argument("--scenarios", type=int, default=None, help="How many scenarios to use (default: all).")
    parser.add_argument("--inner-iterations", type=int, default=4, help="Cutting-plane iterations per scenario.")
    parser.add_argument("--master-time-limit", type=float, default=120.0, help="Seconds per master solve.")
    parser.add_argument("--lambda-steps", type=int, default=0, help="Subgradient steps; 0 gives the perfect-information bound.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--incumbent", type=float, default=None, help="Known feasible welfare, used for the Polyak step.")
    parser.add_argument(
        "--max-price-step",
        type=float,
        default=None,
        help=(
            "Cap the largest single-coordinate price change per step. The Polyak "
            "step is scaled by the gap, which over thousands of priced pairs can "
            "put more price mass into the objective than the welfare itself; "
            "capping it gives the dual its best chance of improving on D(0)."
        ),
    )
    parser.add_argument("--output", required=True, help="Where to write the JSON result.")
    args = parser.parse_args()

    plan = create_plan(args.config)
    config = plan.tasks[0].optimization_config()
    if args.centroids:
        config.centroids_type = args.centroids
    options = config.make_strategy().options
    solver_options = dict(config.make_solver().options)
    backend = config.solver
    target = LevelSpec.parse(config.levels[-1])

    dataset = config.make_dataset()
    base_problem = dataset.problem_for(target)
    from optimization.strategies.saa import _configure_problem

    _configure_problem(base_problem, options)

    start = time.perf_counter()
    market = build_saa_market(base_problem, dataset.config)
    total_scenarios = int(options.get("saa_num_seeds", 15))
    samples = sample_school_preferences(
        market,
        total_scenarios,
        str(options.get("saa_tie_breaking_method", "MTB")).upper(),
        int(options.get("seed", 42)),
    )
    used = samples if args.scenarios is None else samples[: args.scenarios]
    oracles = [
        SaaOracle(market, sample, index, base_problem, workers=args.workers)
        for index, sample in enumerate(used)
    ]
    first_choice = first_choice_upper_bound(market.students)
    transport = welfare_upper_bound("transport", market.programs, market.students)
    setup_seconds = time.perf_counter() - start
    print(
        f"market: {len(market.students)} students, {len(market.programs)} programs, "
        f"{len(used)}/{total_scenarios} scenarios, setup {setup_seconds:.1f}s",
        flush=True,
    )
    print(f"a-priori bounds: first_choice={first_choice:,.2f}  transport={transport:,.2f}", flush=True)

    # Reuses the shared feasibility cache, so this is normally a lookup.
    hint = initial_solution(
        base_problem, options.get("hints", "voronoi"), solver_options=solver_options
    )
    hint_assignment = hint.assignment if hint is not None else None

    # Prices live only on the access pairs the cuts actually touch, so no new
    # access variables are created for their sake.
    probe = [
        oracle.solve(hint_assignment) for oracle in oracles
    ] if hint_assignment is not None else []
    # The solvers key access variables on the unordered pair, so prices must be
    # deduplicated the same way; a self-pair is a constant and cannot
    # discriminate between scenarios, so it is dropped.
    pairs = sorted(
        {
            (min(pair), max(pair))
            for result in probe
            for pair, _ in result.cut.coefficients
            if pair[0] != pair[1]
        }
    )
    print(f"priced access pairs: {len(pairs)}", flush=True)

    prices = [dict.fromkeys(pairs, 0.0) for _ in oracles]
    history = []
    best_bound = math.inf
    for step in range(args.lambda_steps + 1):
        step_start = time.perf_counter()
        results = []
        for index, oracle in enumerate(oracles):
            outcome = scenario_inner_max(
                dataset,
                options,
                target,
                oracle,
                transport,
                prices[index],
                backend=backend,
                iterations=args.inner_iterations,
                time_limit=args.master_time_limit,
                workers=args.workers,
                scale=float(options.get("choice_utility_scale", 100.0)),
                hint=hint_assignment,
            )
            results.append(outcome)
            print(
                f"  step {step} scenario {index}: dual {outcome['dual_bound']:,.2f} "
                f"primal {outcome['primal_value']:,.2f} welfare {outcome['welfare']:,.2f}",
                flush=True,
            )
        bound = sum(r["dual_bound"] for r in results) / len(results)
        best_bound = min(best_bound, bound)
        elapsed = time.perf_counter() - step_start
        label = "perfect_information" if step == 0 else f"lagrangian_step_{step}"
        history.append(
            {
                "step": step,
                "label": label,
                "bound": bound,
                "best_bound": best_bound,
                "scenario_duals": [r["dual_bound"] for r in results],
                "scenario_welfares": [r["welfare"] for r in results],
                "seconds": elapsed,
            }
        )
        print(f"  D = {bound:,.2f}   best {best_bound:,.2f}   ({elapsed:.0f}s)", flush=True)

        if step == args.lambda_steps:
            break

        # Projected subgradient: discourage each scenario from disagreeing with
        # the consensus access pattern. The mean subtraction keeps sum_s
        # lambda_s at zero, which is what makes every D(lambda) valid.
        count = len(results)
        consensus = {
            pair: sum(r["access"].get(pair, 0) for r in results) / count
            for pair in pairs
        }
        gradient = [
            {pair: (r["access"].get(pair, 0) - consensus[pair]) / count for pair in pairs}
            for r in results
        ]
        norm_sq = sum(value * value for g in gradient for value in g.values())
        if norm_sq <= 0:
            print("  scenarios already agree; no subgradient direction", flush=True)
            break
        target_value = args.incumbent if args.incumbent is not None else 0.0
        stepsize = max(0.0, (bound - target_value)) / norm_sq
        if args.max_price_step is not None:
            largest = max(abs(value) for g in gradient for value in g.values())
            if largest > 0:
                stepsize = min(stepsize, args.max_price_step / largest)
        print(f"  subgradient norm^2 {norm_sq:.4g}, step {stepsize:.4g}", flush=True)
        for index, g in enumerate(gradient):
            for pair, value in g.items():
                prices[index][pair] -= stepsize * value

    payload = {
        "config": args.config,
        "centroids_type": config.centroids_type,
        "level": str(target),
        "solver": backend,
        "scenarios": len(used),
        "inner_iterations": args.inner_iterations,
        "master_time_limit": args.master_time_limit,
        "first_choice_upper_bound": first_choice,
        "transport_upper_bound": transport,
        "priced_pairs": len(pairs),
        "history": history,
        "best_bound": best_bound,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

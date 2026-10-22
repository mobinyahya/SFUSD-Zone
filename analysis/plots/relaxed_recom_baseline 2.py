#!/usr/bin/env python3
"""Score an unguided relaxed-ReCom chain on the benchmark's three yardsticks.

Every strategy in the choice benchmark spends its budget steering the search.
This script produces the reference point they should be read against: one
relaxed-ReCom random walk that optimizes nothing at all, with choice welfare,
budget welfare, and cut edges computed for *every* iteration rather than only
for a final zoning.

The chain is the same kernel ``short_bursts_choice`` uses - ``relaxed_recom``,
started from the same ``hints: feasible`` initial solution and run on the same
problem - but with no scorer and no burst restarts, so nothing selects among the
zonings it visits. Its configuration is read from a benchmarked run so the
problem, the balance constraints, and the lottery scale match the runs it is
compared against.

Recombinations that would break a balance constraint are rejected, the way
``recom`` rejects them, so every zoning scored here is one the benchmarked
strategies were also allowed to return and the comparison stays like-for-like.
Roughly half of the proposals survive that test on these instances, and each
costs milliseconds, so the rejection is not what the run spends its time on.

Each visited zoning is scored the way the comparison figure scores the
benchmarked ones:

* choice welfare - exact MNL logsum, the same call ``metrics.choice`` makes;
* budget welfare - ``finite_grid_oracle`` at base-block resolution, the same
  call ``recompute_mid_welfare --level base`` makes;
* cut edges - crossings of the ``Block_0`` adjacency graph, the same count
  ``metrics.spatial`` reports.

The walk itself runs on the run's own coarse level and is expanded to base
blocks for scoring, which is exact: a coarse node's blocks all carry its zone.
Both the oracle and the MNL evaluation depend only on that block-level zoning,
so the numbers do not depend on the local KaHIP coarsening.

Writes ``relaxed_recom_baseline.csv`` with one row per iteration, carrying
``max_violation`` - the worst balance residual in students - so the feasibility
of every scored zoning stays auditable rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from choice.models import build_mnl_choice_model
from loaders import load_scenario
from optimization.config import OptimizationConfig
from optimization.data.mid import MidMarket, build_mid_market, preprocess_mid_market
from optimization.levels import LevelSpec
from optimization.mid_oracle import finite_grid_oracle
from optimization.problem import ZoneProblem
from optimization.solvers import get_solver

# The chain is driven directly rather than through ``RelaxedReComSolver.solve``,
# which reports only the best zoning it saw. These are the same handles that
# solver uses, so the walk is the benchmark's walk and not a re-implementation.
from optimization.solvers.recom import _NoProposal, _ReComKernel

DEFAULT_BENCHMARK_DIR = (
    "/soalnas/share/data/school_choice/local_runs/benchmark_choice_v4"
)

# Whose config to copy. This strategy runs the same ReCom kernel, so its runs
# define the problem the baseline should walk over.
CONFIG_DONOR_STRATEGY = "short_bursts_choice"

# Proposals per requested iteration before the walk is declared stuck. Around
# half of them are rejected as infeasible on these instances, so the headroom
# is generous on purpose: a proposal costs milliseconds, scoring one does not.
PROPOSAL_ATTEMPT_FACTOR = 50

CONFIG_FIELDS = {field.name for field in fields(OptimizationConfig)}


def available_cpus() -> int:
    """CPUs this process may actually use, not the machine's core count.

    ``os.cpu_count()`` reports the whole node. Under Slurm a job that asked for
    one CPU still sees every core that way, so defaulting to it would start one
    CP-SAT worker per core inside a one-CPU allocation -- oversubscribing the
    node and slowing down the very search the workers were meant to speed up.
    ``SLURM_CPUS_PER_TASK`` is what the rest of this repo reads (see
    ``benchmark.slurm._validate_worker_cpus``); the affinity mask catches
    cgroup-confined runs that do not set it, and is absent on macOS.
    """
    limits = []
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and allocated.isdigit():
        limits.append(int(allocated))
    if hasattr(os, "sched_getaffinity"):
        limits.append(len(os.sched_getaffinity(0)))
    limits.append(os.cpu_count() or 1)
    return max(1, min(limits))


@dataclass(frozen=True)
class ChainInputs:
    """Everything the walk and the scoring of one configuration need."""

    config: OptimizationConfig
    problem: ZoneProblem
    base_problem: ZoneProblem
    base_market: MidMarket
    choice_model: Any
    # Base graph nodes, the coarse node covering each of them, and the base
    # adjacency as node pairs: enough to expand a coarse zoning and count cuts.
    base_nodes: tuple[int, ...]
    coarse_of_base: tuple[int, ...]
    base_edges: tuple[tuple[int, int], ...]


def config_from_snapshot(snapshot: dict[str, Any]) -> OptimizationConfig:
    """Rebuild a run's config from the snapshot stored in ``result.json``."""
    return OptimizationConfig(
        **{k: v for k, v in snapshot.items() if k in CONFIG_FIELDS}
    )


def donor_snapshots(benchmark_dir: Path) -> dict[str, dict[str, Any]]:
    """One config snapshot per centroid configuration, lowest seed first."""
    best: dict[str, tuple[int, dict[str, Any]]] = {}
    for path in sorted(benchmark_dir.rglob("result.json")):
        with path.open() as handle:
            snapshot = json.load(handle)["config"]
        if snapshot["strategy"] != CONFIG_DONOR_STRATEGY:
            continue
        centroid = str(snapshot["centroids_type"])
        seed = int(snapshot["seed"])
        if centroid not in best or seed < best[centroid][0]:
            best[centroid] = (seed, snapshot)
    if not best:
        raise SystemExit(
            f"No {CONFIG_DONOR_STRATEGY} run under {benchmark_dir} to copy a config from."
        )
    return {centroid: snapshot for centroid, (_, snapshot) in sorted(best.items())}


def build_inputs(config: OptimizationConfig) -> ChainInputs:
    """Build the walk's problem and the scoring machinery for one config."""
    level = LevelSpec.parse(config.levels[-1])
    dataset = config.make_dataset()

    problem = dataset.problem_for(level)
    base_problem = dataset.problem_for(level.base())
    for target in (problem, base_problem):
        # The MID strategies drop capacity slack the same way before scoring.
        target.overage = -1.0
        target.shortage = -1.0
        target.boundary_prop = float(config.boundary_prop)

    start = time.perf_counter()
    base_market = preprocess_mid_market(
        build_mid_market(base_problem, dataset.config), base_problem
    )
    print(f"  built base MID market in {time.perf_counter() - start:.1f} s")

    area_to_coarse: dict[int, int] = {}
    for node in problem.G:
        attrs = problem.G.nodes[node]
        area_ids = (
            [int(attrs["area_id"])]
            if "area_id" in attrs
            else [int(area_id) for area_id in attrs["block_ids"]]
        )
        for area_id in area_ids:
            area_to_coarse[area_id] = node

    base_nodes = tuple(base_problem.G)
    missing = [
        node
        for node in base_nodes
        if int(base_problem.G.nodes[node]["area_id"]) not in area_to_coarse
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} base block(s) belong to no coarse node; the two levels "
            "do not describe the same district."
        )
    coarse_of_base = tuple(
        area_to_coarse[int(base_problem.G.nodes[node]["area_id"])]
        for node in base_nodes
    )
    return ChainInputs(
        config=config,
        problem=problem,
        base_problem=base_problem,
        base_market=base_market,
        choice_model=build_mnl_choice_model(
            load_scenario(config.data), method=config.choice_model_method
        ),
        base_nodes=base_nodes,
        coarse_of_base=coarse_of_base,
        base_edges=tuple(base_problem.G.edges()),
    )


def score_zoning(inputs: ChainInputs, assignment: dict[int, int]) -> dict[str, float]:
    """Score one coarse zoning on all three measures of the comparison figure."""
    base_zoning = {
        node: assignment[coarse]
        for node, coarse in zip(inputs.base_nodes, inputs.coarse_of_base)
    }
    oracle = finite_grid_oracle(
        inputs.base_market, base_zoning, int(inputs.config.mid_lottery_scale)
    )
    return {
        "choice_welfare": float(
            inputs.choice_model.preassignment_utility(inputs.problem, assignment)
        ),
        "budget_welfare": float(oracle.welfare),
        "cut_edges": sum(
            base_zoning[u] != base_zoning[v] for u, v in inputs.base_edges
        ),
    }


def walk(inputs: ChainInputs, iterations: int, seed: int) -> Iterator[dict[str, Any]]:
    """Yield one scored row per visited zoning, the initial solution included.

    Iteration 0 is the ``hints: feasible`` starting point; iteration *i* is the
    zoning after the *i*-th accepted recombination. A recombination is accepted
    only if it leaves every balance constraint satisfied, so every scored zoning
    is one the benchmarked strategies were also allowed to return. The starting
    point is the one exception: a CP-feasible hint can carry a rounding-scale
    residual under ReCom's own float check, which the first accepted move
    clears. ``max_violation`` records it either way.
    """
    solver = get_solver(
        "relaxed_recom",
        **{
            **inputs.config.make_solver().options,
            "recom_iterations": iterations,
            # The chain is bounded by iterations only; the run's time limit
            # would otherwise cut it short.
            "solve_time_limit": None,
            "seed": seed,
        },
    )
    setup = solver._initialize(inputs.problem, time.monotonic())
    context, state = setup.context, setup.state
    kernel = _ReComKernel(context, setup.rng, setup.deadline)
    print(
        f"  initial solution: {setup.hint_metadata.get('hints')} "
        f"(cache {setup.hint_metadata.get('hint_cache', 'n/a')})"
    )

    attempt_limit = max(PROPOSAL_ATTEMPT_FACTOR * iterations, PROPOSAL_ATTEMPT_FACTOR)
    attempted = 0
    rejected = 0
    failures = 0
    iteration = 0
    stalled = ""
    while True:
        yield {
            "iteration": iteration,
            "attempted_moves": attempted,
            "rejected_moves": rejected,
            **score_zoning(inputs, context.assignment_dict(state.assignment)),
            "coarse_boundary_cost": int(state.boundary_cost),
            "max_violation": float(max(state.violations, default=0.0)),
            "feasible": bool(state.feasible),
        }
        if iteration >= iterations:
            break
        move = None
        while move is None and attempted < attempt_limit:
            attempted += 1
            try:
                move = kernel.propose(state, "relaxed")
            except _NoProposal as exc:
                failures += 1
                if exc.reason == "no_adjacent_zone_pairs":
                    stalled = exc.reason
                    break
                continue
            if not move.globally_feasible:
                rejected += 1
                move = None
        if move is None:
            stalled = stalled or "proposal_attempt_limit"
            break
        kernel.apply(state, move)
        iteration += 1
    if stalled:
        print(f"  chain stopped after {iteration} iteration(s): {stalled}")
    print(
        f"  {attempted} proposal attempt(s), {rejected} rejected as infeasible, "
        f"{failures} failure(s)"
    )


def run_baseline(
    benchmark_dir: Path,
    iterations: int,
    seed: int,
    centroids: list[str] | None,
    workers: int,
) -> pd.DataFrame:
    """Walk and score one chain per centroid configuration."""
    snapshots = donor_snapshots(benchmark_dir)
    if centroids:
        unknown = sorted(set(centroids) - set(snapshots))
        if unknown:
            raise SystemExit(
                f"No {CONFIG_DONOR_STRATEGY} run for centroid(s) {unknown}; "
                f"available: {sorted(snapshots)}"
            )
        snapshots = {name: snapshots[name] for name in centroids}

    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    for centroid, snapshot in snapshots.items():
        # The donor is a short_bursts_choice run, and that sweep task gives it
        # one worker. The ReCom walk is single-threaded either way, so the only
        # thing the count reaches here is the CP-SAT feasible-hint search --
        # where it is the difference between a hint in half a minute and a
        # 600-second timeout on the tighter centroid configurations.
        config = config_from_snapshot({**snapshot, "workers": workers})
        level = LevelSpec.parse(config.levels[-1])
        print(f"{centroid}: {iterations} relaxed-ReCom iterations on {level}")
        start = time.perf_counter()
        try:
            for row in walk(build_inputs(config), iterations, seed):
                rows.append(
                    {
                        "centroid": centroid,
                        "seed": seed,
                        "level": str(level),
                        "scoring_level": str(level.base()),
                        **row,
                    }
                )
                if row["iteration"] % 100 == 0:
                    print(
                        f"  [{row['iteration']:5d}] choice {row['choice_welfare']:11,.1f}  "
                        f"budget {row['budget_welfare']:11,.1f}  "
                        f"cut edges {row['cut_edges']:5d}  "
                        f"({time.perf_counter() - start:.0f} s)",
                        flush=True,
                    )
        except Exception as error:
            # The chain starts from the same ``hints: feasible`` solution the
            # benchmark used, and on a tight configuration that hint search
            # times out. One centroid failing is a missing baseline row, not a
            # reason to discard the chains already walked -- which is what an
            # uncaught raise did, since the CSV is written once at the end.
            # Rows already collected for this centroid are dropped: a chain
            # that stopped early is not the 1,000-iteration walk the figure
            # says the row is.
            print(f"  skipped after {time.perf_counter() - start:.0f} s: "
                  f"{type(error).__name__}: {error}")
            rows = [row for row in rows if row["centroid"] != centroid]
            failed.append(centroid)
            continue
        chain = pd.DataFrame([row for row in rows if row["centroid"] == centroid])
        print(
            f"  {len(chain)} iteration(s) in {time.perf_counter() - start:.0f} s:\n"
            + "\n".join(
                f"    {label:14s} mean {chain[column].mean():11,.1f}  "
                f"min {chain[column].min():11,.1f}  max {chain[column].max():11,.1f}"
                for column, label in (
                    ("choice_welfare", "choice welfare"),
                    ("budget_welfare", "budget welfare"),
                    ("cut_edges", "cut edges"),
                )
            )
            + f"\n    balance-feasible {int(chain['feasible'].sum())}/{len(chain)}, "
            f"median violation {chain['max_violation'].median():.3f} student(s)"
        )
    if failed:
        print(f"\nno chain for {failed}; those panels get no baseline row")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score an unguided relaxed-ReCom chain iteration by iteration."
    )
    parser.add_argument(
        "benchmark_dir",
        nargs="?",
        default=DEFAULT_BENCHMARK_DIR,
        help=(
            "Benchmark directory whose run configs define the problem "
            f"(default: {DEFAULT_BENCHMARK_DIR})"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Destination CSV (default: analysis/plots/output/relaxed_recom_baseline.csv)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Recombination steps to walk after the initial solution (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Chain seed (default: 1)")
    parser.add_argument(
        "--workers",
        type=int,
        default=available_cpus(),
        help=(
            "CP-SAT workers for the feasible-hint search only; the walk itself "
            "is single-threaded (default: the CPUs allocated to this process, "
            "which under Slurm is SLURM_CPUS_PER_TASK, not the node's core count)"
        ),
    )
    parser.add_argument(
        "--centroids",
        nargs="*",
        default=None,
        help="Centroid configurations to walk (default: every one in the benchmark)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative.")
    output = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "output" / "relaxed_recom_baseline.csv"
    )
    table = run_baseline(
        Path(args.benchmark_dir), args.iterations, args.seed, args.centroids,
        args.workers,
    )
    if table.empty:
        raise SystemExit("No iterations were scored.")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nwrote {output}  ({len(table)} iteration rows)")


if __name__ == "__main__":
    main()

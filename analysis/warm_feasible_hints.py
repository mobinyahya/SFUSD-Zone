#!/usr/bin/env python3
"""Pre-compute the zoning-feasible hint for one centroid configuration.

Every task in a sweep starts from ``hints: feasible``, and on the tighter
centroid configurations that CP-SAT search is the difference between a run and
a ``FeasibleHintError``. The hints are content-addressed and shared, so solving
each one once up front -- with enough workers to actually find it -- lets every
later run read it back instead of re-searching under whatever worker count its
own sweep task happens to allocate.

The cache key is the *feasible set*: the problem fingerprint plus the
feasibility-affecting options. ``workers``, ``seed`` and
``feasible_hint_time_limit`` are excluded by construction (see
``optimization.data.initial_solutions``), so a hint found here with 8 workers
in 5 minutes is served to a sweep task configured with 1 worker and 600
seconds. That is the whole point of warming them separately.

The problem is built from the sweep's own generated tasks rather than
hand-assembled, and ``boundary_prop`` is applied the way every strategy applies
it -- after ``problem_for``, which leaves it disabled. Getting that wrong would
warm a hint for a different, strictly easier feasible set and the sweep would
miss the cache.
"""

from __future__ import annotations

import argparse
import time

from benchmark.config import SimulationSweep
from optimization.data.feasibility import check_zoning
from optimization.data.initial_solutions import (
    FeasibleHintError,
    feasible_initial_solution,
)
from optimization.levels import LevelSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", help="Path to the sweep YAML")
    parser.add_argument("centroid", help="Centroid configuration to warm")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Seconds for the CP-SAT hint search (default: 300)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CP-SAT workers for the hint search (default: 8)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sweep = SimulationSweep.from_yaml(args.sweep)

    # Any task at this centroid defines the feasible set: the hint model reads
    # the problem and the feasibility options, neither of which varies with the
    # strategy or solver that a task happens to use.
    tasks = [
        task
        for task in sweep.generate_tasks()
        if str(task.config.get("centroids_type")) == args.centroid
    ]
    if not tasks:
        raise SystemExit(f"No task in {args.sweep} uses centroid {args.centroid!r}.")
    config = tasks[0].optimization_config()

    level = LevelSpec.parse(config.levels[-1])
    problem = config.make_dataset().problem_for(level)
    # problem_for leaves boundary_prop at -1.0; every strategy sets it from its
    # own options, and it is part of the feasible set the hint is keyed on.
    problem.boundary_prop = float(getattr(config, "boundary_prop", -1.0))

    print(
        f"centroid={args.centroid} level={level} nodes={problem.G.number_of_nodes()} "
        f"zones={problem.Z} boundary_prop={problem.boundary_prop} "
        f"frl_dev={problem.frl_dev} max_distance={problem.max_distance}",
        flush=True,
    )
    print(
        f"searching with {args.workers} worker(s), limit {args.time_limit:.0f}s",
        flush=True,
    )

    start = time.perf_counter()
    try:
        solution = feasible_initial_solution(
            problem,
            solver_options={
                "feasible_hint_time_limit": args.time_limit,
                "workers": args.workers,
                "centroid_neighbor_radius": getattr(
                    config, "centroid_neighbor_radius", 0
                ),
            },
        )
    except FeasibleHintError as error:
        print(f"FAILED after {time.perf_counter() - start:.0f}s: {error}", flush=True)
        return 1

    elapsed = time.perf_counter() - start
    report = check_zoning(problem, solution.assignment)
    meta = solution.metadata
    print(
        f"OK in {elapsed:.0f}s  status={meta.get('hint_solver_status')}  "
        f"cache={meta.get('hint_cache', 'stored')}  "
        f"key={meta.get('hint_cache_key', 'n/a')}  "
        f"independently_feasible={report.feasible}",
        flush=True,
    )
    if not report.feasible:
        print("hint failed the independent validator; not usable", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

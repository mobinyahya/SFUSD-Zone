"""Time every objective's batch scorer on saved ReCom walk states.

A sizing probe for scoring the walk ensemble: for each walk file given, it
scores a few of one walk's states (every ``--every``-th) with every registered
objective (``weighted_sum`` aside, which only combines the others) and
reports the scorer's build time, the time per state, and the first values.

    uv run python analysis/time_walk_objectives.py CONFIG WALK.npz [WALK.npz ...] \
        [--objectives NAME ...]
"""

from __future__ import annotations

import argparse
import time
import traceback
import warnings
from dataclasses import replace

import numpy as np

from optimization.config import OptimizationConfig
from optimization.data.contiguity import boundary_cost
from optimization.objectives import OBJECTIVES, heuristic_scorer


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("walks", nargs="+")
    parser.add_argument("--level", default="Block_1")
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument(
        "--objectives", nargs="+", default=None, help="Default: every objective."
    )
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")

    config = OptimizationConfig.from_yaml(args.config)
    dataset = config.make_dataset()
    options = config.make_solver().options
    for path in args.walks:
        walk = np.load(path, allow_pickle=True)
        school_ids = [int(s) for s in walk["centroid_school_ids"]]
        problem = dataset.problem_for(args.level, centroid_school_ids=school_ids)
        nodes = [int(n) for n in walk["node_ids"]]
        if nodes != list(problem.nodes):
            raise ValueError(f"{path}: node order differs from the problem's.")
        states = walk["states"][0, :: args.every]
        assignments = tuple(dict(zip(nodes, map(int, s))) for s in states)
        print(f"== {path}: Z={problem.Z}, {len(assignments)} states", flush=True)
        for name in args.objectives or OBJECTIVES:
            if name == "weighted_sum":
                continue
            try:
                start = time.monotonic()
                if name in {"cut_edges", "boundary_perimeter"}:
                    build = 0.0
                    values = [
                        boundary_cost(
                            problem.G, a, weight_edges=name == "boundary_perimeter"
                        )
                        for a in assignments
                    ]
                else:
                    scorer = heuristic_scorer(
                        replace(problem, objective=name), name, options
                    )
                    build = time.monotonic() - start
                    start = time.monotonic()
                    values = scorer(assignments, None)
                per_state = (time.monotonic() - start) / len(assignments)
                print(
                    f"{name:20s} build {build:7.1f} s  per state "
                    f"{per_state * 1000:10.1f} ms  first {np.round(values[:3], 3)}",
                    flush=True,
                )
            except Exception as exc:
                print(f"{name:20s} ERROR {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    main()

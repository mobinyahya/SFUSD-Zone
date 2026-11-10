"""Record plain ReCom chains: every state, for later scoring on any objective.

Each walk starts from the ``--hints`` zoning of a centroid set (default
``voronoi``) and takes
``--iterations`` attempted moves of the classic chain (``PRESETS["chain"]``:
one continuous chain, a ``--pair-selection`` zone pair, the configured random
spanning tree, a uniform cut over the cuts that keep the rules). The config
turns every balance row off except ``population_balance: auto``, so the rules
are: zones stay contiguous, each holds its centroid, and no zone's students
stray from the average zone by more than the start's worst deviation plus 5
points. The even school count per zone, which no config turns off, is dropped
here (``NoSchoolCountProblem``). A move that breaks a rule is rejected and the
chain records the unchanged state again, so a walk is always ``iterations + 1``
states, one per attempted move.

Output, one file per centroid set, ``<output>/<centroids_type>.npz``:

    states      uint8 [seeds, iterations + 1, nodes]  zone of each node
    seeds       int   [seeds]
    node_ids    int   [nodes]  graph node index, the states' column order
    block_ids   object [nodes] Census block GEOIDs each node aggregates
    centroid_school_ids int [zones]  zone z is anchored at this school
    centroid_nodes      int [zones]
    population_balance_dev float [seeds]  the walk's population band (``auto``:
                                          its start's worst deviation + 0.05)
    moves_accepted      int [seeds]  states that differ from their predecessor
                                     (``iterations`` minus rejected moves)

plus ``<output>/<centroids_type>.json`` with the run's settings and timing.

    uv run python analysis/recom_walk_ensemble.py CONFIG --level Block_1 \
        --centroids 6-zone-1 all --seeds 1-100 --iterations 1000 -o OUT
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np

from optimization.config import OptimizationConfig
from optimization.problem import ZoneProblem
from optimization.solvers import get_solver


class NoSchoolCountProblem(ZoneProblem):
    """A problem without the even school count per zone.

    ``ZoneProblem.school_count_bounds`` is a row every solver enforces whatever
    the config, so it is the one row the config cannot turn off. The baseline
    walk does not want it, so it drops it here. The class survives
    ``dataclasses.replace``, which ReCom uses to anchor centroids.
    """

    def school_count_bounds(self):
        return None


def _seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = (int(part) for part in text.split("-", 1))
        return list(range(lo, hi + 1))
    return [int(part) for part in text.split(",")]


def walk(
    problem, seed: int, iterations: int, tree_sampler: str, hints: str, pair_selection: str
):
    """One chain's states, initial first, as zone indices over ``problem.nodes``."""
    solver = get_solver(
        "recom",
        seed=seed,
        hints=hints,
        recom_iterations=iterations,
        recom_tree_sampler=tree_sampler,
        solve_time_limit=None,
        save_solver_logs=False,
    )
    nodes = list(problem.nodes)
    states: list[np.ndarray] = []

    def visit(assignment):
        states.append(np.fromiter((assignment[n] for n in nodes), np.uint8, len(nodes)))

    solution = solver.sample_feasible(
        problem, visit, pair_selection=pair_selection, visit_rejected=True
    )
    if solution.status == "ERROR":
        raise RuntimeError(solution.metadata.get("error_message"))
    return np.stack(states), solution.metadata


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--level", default="Block_1")
    parser.add_argument("--centroids", nargs="+", required=True)
    parser.add_argument("--seeds", default="1-100")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--tree-sampler", default="random_mst")
    parser.add_argument("--hints", default="voronoi")
    parser.add_argument("--pair-selection", default="uniform_boundary")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    config = OptimizationConfig.from_yaml(args.config)
    load_start = time.monotonic()
    dataset = config.make_dataset()
    seeds = _seeds(args.seeds)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    for centroids_type in args.centroids:
        start = time.monotonic()
        school_ids = dataset.centroid_school_ids(centroids_type)
        problem = dataset.problem_for(args.level, centroid_school_ids=school_ids)
        problem.__class__ = NoSchoolCountProblem
        if problem.Z > 255:
            raise ValueError("uint8 states hold at most 255 zones.")
        position = {node: i for i, node in enumerate(problem.nodes)}
        centroid_cols = np.asarray([position[c] for c in problem.centroids])
        setup = time.monotonic() - start
        walks, accepted, walk_seconds, population_devs = [], [], [], []
        for seed in seeds:
            t = time.monotonic()
            states, meta = walk(
                problem,
                seed,
                args.iterations,
                args.tree_sampler,
                args.hints,
                args.pair_selection,
            )
            walk_seconds.append(time.monotonic() - t)
            if len(states) != args.iterations + 1:
                raise RuntimeError(
                    f"{centroids_type} seed {seed}: {len(states)} states for "
                    f"{args.iterations} moves ({meta.get('stop_reason')}, "
                    f"{meta.get('proposal_failures')} proposal failures)."
                )
            if not (states[:, centroid_cols] == np.arange(problem.Z)).all():
                raise RuntimeError(f"{centroids_type} seed {seed}: a centroid left its zone.")
            walks.append(states)
            accepted.append(int(meta["accepted_moves"]))
            # The population band, recomputed from the saved states.
            dev = meta.get("population_balance_dev")
            if dev is not None:
                students = np.asarray([problem.students(n) for n in problem.nodes])
                average = students.sum() / problem.Z
                per_zone = np.stack(
                    [(students * (states == z)).sum(1) for z in range(problem.Z)], 1
                )
                slack = 1e-6 * max(average, 1.0)
                if (np.abs(per_zone - average) > dev * average + slack).any():
                    raise RuntimeError(
                        f"{centroids_type} seed {seed}: population band broke."
                    )
            population_devs.append(dev)
        G = problem.G
        nodes = list(problem.nodes)
        block_ids = np.empty(len(nodes), dtype=object)
        for i, n in enumerate(nodes):
            attrs = G.nodes[n]
            ids = [attrs["area_id"]] if "area_id" in attrs else attrs["block_ids"]
            block_ids[i] = np.asarray(sorted(int(a) for a in ids), dtype=np.int64)
        path = os.path.join(args.output, f"{centroids_type}.npz")
        np.savez_compressed(
            path,
            states=np.stack(walks),
            seeds=np.asarray(seeds),
            node_ids=np.asarray(nodes),
            block_ids=block_ids,
            centroid_school_ids=np.asarray(school_ids),
            centroid_nodes=np.asarray(problem.centroids),
            moves_accepted=np.asarray(accepted),
            population_balance_dev=np.asarray(
                [np.nan if d is None else d for d in population_devs]
            ),
        )
        summary = {
            "config": os.path.abspath(args.config),
            "commit": commit,
            "level": args.level,
            "centroids_type": centroids_type,
            "num_zones": problem.Z,
            "num_nodes": len(nodes),
            "iterations": args.iterations,
            "tree_sampler": args.tree_sampler,
            "chain": "PRESETS['chain'] (uniform cut, no bursts, rejections repeat)",
            "hints": args.hints,
            "pair_selection": args.pair_selection,
            "seeds": seeds,
            "dataset_load_seconds": start - load_start,
            "problem_setup_seconds": setup,
            "walk_seconds": walk_seconds,
            "moves_accepted": accepted,
            "population_balance": config.population_balance,
            "population_balance_dev": population_devs,
            "bytes": os.path.getsize(path),
        }
        with open(os.path.join(args.output, f"{centroids_type}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"{centroids_type}: Z={problem.Z} nodes={len(nodes)} "
            f"walk mean {np.mean(walk_seconds):.2f}s file {os.path.getsize(path)/1e6:.2f} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()

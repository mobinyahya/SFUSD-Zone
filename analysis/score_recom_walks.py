"""Score saved ReCom walk states: three constraints and every objective.

For every state of every walk in a walk file (``analysis/recom_walk_ensemble.py``),
this records

* three constraint checks, each as a pass/fail flag and the worst zone's
  value, in the same linear form as the repo's balance rows
  (:mod:`optimization.solvers.balance`):

  - ``frl``: every zone's FRL share within ``--frl-dev`` (default 0.15) of
    the district's, ``F - dev <= FRL_z / students_z <= F + dev``;
  - ``schools``: every zone's school count within ``--school-slack``
    (default 1) of the average, ``|schools_z - S / Z| <= 1``;
  - ``capacity``: every zone's seats within ``--capacity-dev`` (default 0.3)
    of its students, ``1 - dev <= seats_z / students_z <= 1 + dev``, with
    ``capacity_seats`` and ``capacity_students`` as the overage/shortage
    rows count them;

  and the cell, the subset of the three met (bit 0 frl, 1 schools,
  2 capacity, so ``cell == 7`` meets all three);
* every objective in ``--objectives`` (default: all of the registry but
  ``weighted_sum``), each in its own units from its batch scorer.

Output: ``<output>/<walk stem>.npz`` with one row per state: ``seed``,
``step``, the flags and worst values, ``cell``, and one array per objective.
A file whose output exists is skipped, so a rerun finishes what failed.

    uv run python analysis/score_recom_walks.py CONFIG WALK.npz [...] -o OUT
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import replace

import numpy as np

from optimization.config import OptimizationConfig
from optimization.data.contiguity import boundary_cost
from optimization.objectives import OBJECTIVES, heuristic_scorer

DEFAULT_OBJECTIVES = tuple(name for name in OBJECTIVES if name != "weighted_sum")
# These read the full MID market and share one match per state; the
# attendance-area match (aa_capacity_match) is a different market.
SHARED_MARKET = {
    "matching_welfare": "welfare",
    "capacity_match": "capacity_match",
    "fill_polarization": "fill_polarization",
    "frl_dissimilarity": "frl_dissimilarity",
}


def walk_problem(dataset, config, level, school_ids):
    """The walk's problem, with the land areas ``school_distance`` weighs by.

    The dataset attaches land areas only for a config whose objective reads
    them, so the problem is built through one; the graph (and so the node
    order of the saved states) is the same.
    """
    return dataset.problem_for(
        level,
        centroid_school_ids=school_ids,
        config=replace(config, objective="school_distance"),
    )
# Slack on the linear rows, as check_zoning's default feasibility tolerance.
_TOL = 1e-6


def zone_sums(states: np.ndarray, values: np.ndarray, zones: int) -> np.ndarray:
    """``[T, Z]`` sums of a per-node quantity over each state's zones."""
    T = states.shape[0]
    index = states.astype(np.int64) + zones * np.arange(T)[:, None]
    return np.bincount(
        index.ravel(), weights=np.broadcast_to(values, states.shape).ravel(),
        minlength=T * zones,
    ).reshape(T, zones)


def constraint_columns(problem, states, *, frl_dev, school_slack, capacity_dev):
    """Flags and worst-zone values of the three constraints, one per state."""
    nodes = list(problem.nodes)
    Z = problem.Z
    students = np.asarray([problem.students(n) for n in nodes])
    frl = np.asarray([problem.frl(n) for n in nodes])
    seats = np.asarray([problem.capacity_seats(n) for n in nodes])
    cap_students = np.asarray([problem.capacity_students(n) for n in nodes])
    schools = np.asarray([float(problem.num_schools(n)) for n in nodes])

    s = zone_sums(states, students, Z)
    f = zone_sums(states, frl, Z)
    c = zone_sums(states, seats, Z)
    cs = zone_sums(states, cap_students, Z)
    k = zone_sums(states, schools, Z)

    F = problem.district_frl
    # Linear rows: value - ratio * students, >= 0 (lower) and <= 0 (upper).
    frl_ok = (
        (f - (F - frl_dev) * s >= -_TOL) & (f - (F + frl_dev) * s <= _TOL)
    ).all(1)
    cap_ok = (
        (c - (1 - capacity_dev) * cs >= -_TOL) & (c - (1 + capacity_dev) * cs <= _TOL)
    ).all(1)
    average = schools.sum() / Z
    school_dev = np.abs(k - average).max(1)
    schools_ok = school_dev <= school_slack + _TOL

    with np.errstate(divide="ignore", invalid="ignore"):
        frl_share = np.where(s > 0, f / s, np.nan)
        cap_ratio = np.where(cs > 0, c / cs, np.where(c > 0, np.inf, np.nan))
    return {
        "frl_ok": frl_ok,
        "schools_ok": schools_ok,
        "capacity_ok": cap_ok,
        "cell": frl_ok.astype(np.int8)
        + 2 * schools_ok.astype(np.int8)
        + 4 * cap_ok.astype(np.int8),
        # Worst zone: largest |FRL share - F| (zones with students), largest
        # |schools - average|, and the extreme seats/students ratios.
        "frl_max_dev": np.nanmax(np.abs(frl_share - F), axis=1),
        "schools_max_dev": school_dev,
        "capacity_ratio_min": np.nanmin(cap_ratio, axis=1),
        "capacity_ratio_max": np.nanmax(cap_ratio, axis=1),
        "empty_zones": (s <= 0).sum(1),
    }


def build_scorers(problem, names, options) -> dict:
    """One batch scorer per objective; the full-market ones share a match."""
    from optimization.solvers.burst_objectives import mid_burst_scorers

    scorers: dict = {}
    shared = [n for n in names if n in SHARED_MARKET]
    if shared:
        market = mid_burst_scorers(
            replace(problem, objective=shared[0]),
            tuple(SHARED_MARKET[n] for n in shared),
            options,
        )
        scorers.update({n: market[SHARED_MARKET[n]] for n in shared})
    for name in names:
        if name in scorers:
            continue
        scorers[name] = (
            None
            if name in {"cut_edges", "boundary_perimeter"}
            else heuristic_scorer(replace(problem, objective=name), name, options)
        )
    return scorers


def score(problem, name, scorer, assignments) -> np.ndarray:
    if scorer is None:
        weighted = name == "boundary_perimeter"
        return np.asarray(
            [boundary_cost(problem.G, a, weight_edges=weighted) for a in assignments],
            dtype=float,
        )
    return np.asarray(scorer(tuple(assignments), None), dtype=float)


def close_all(scorers) -> None:
    for scorer in {id(s): s for s in scorers.values() if s is not None}.values():
        close = getattr(scorer, "close", None)
        if close is not None:
            close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("walks", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--level", default="Block_1")
    parser.add_argument("--objectives", nargs="+", default=list(DEFAULT_OBJECTIVES))
    parser.add_argument("--every", type=int, default=1, help="Score every k-th state.")
    parser.add_argument("--frl-dev", type=float, default=0.15)
    parser.add_argument("--school-slack", type=float, default=1.0)
    parser.add_argument("--capacity-dev", type=float, default=0.3)
    parser.add_argument("--batch", type=int, default=200)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    os.makedirs(args.output, exist_ok=True)

    config = OptimizationConfig.from_yaml(args.config)
    dataset = config.make_dataset()
    options = config.make_solver().options
    for path in args.walks:
        stem = os.path.splitext(os.path.basename(path))[0]
        tag = "" if args.objectives == list(DEFAULT_OBJECTIVES) else (
            "__" + "+".join(args.objectives)
        )
        out = os.path.join(args.output, f"{stem}{tag}.npz")
        if os.path.exists(out):
            print(f"skip {out}", flush=True)
            continue
        start = time.monotonic()
        walk = np.load(path, allow_pickle=True)
        school_ids = [int(s) for s in walk["centroid_school_ids"]]
        problem = walk_problem(dataset, config, args.level, school_ids)
        nodes = [int(n) for n in walk["node_ids"]]
        if nodes != list(problem.nodes):
            raise ValueError(f"{path}: node order differs from the problem's.")
        states = walk["states"][:, :: args.every]  # [seeds, steps, nodes]
        seeds_n, steps_n, _ = states.shape
        flat = states.reshape(-1, states.shape[-1])
        columns = {
            "seed": np.repeat(walk["seeds"], steps_n),
            "step": np.tile(np.arange(steps_n) * args.every, seeds_n),
            **constraint_columns(
                problem,
                flat,
                frl_dev=args.frl_dev,
                school_slack=args.school_slack,
                capacity_dev=args.capacity_dev,
            ),
        }
        timings = {name: 0.0 for name in args.objectives}
        values = {name: np.empty(len(flat)) for name in args.objectives}
        t = time.monotonic()
        scorers = build_scorers(problem, args.objectives, options)
        build_seconds = time.monotonic() - t
        try:
            for lo in range(0, len(flat), args.batch):
                chunk = flat[lo : lo + args.batch]
                assignments = [dict(zip(nodes, map(int, row))) for row in chunk]
                for name in args.objectives:
                    t = time.monotonic()
                    values[name][lo : lo + len(chunk)] = score(
                        problem, name, scorers[name], assignments
                    )
                    timings[name] += time.monotonic() - t
        finally:
            close_all(scorers)
        columns.update(values)
        np.savez_compressed(
            out,
            **columns,
            centroids_type=os.path.basename(path).split("_seeds")[0],
            num_zones=problem.Z,
            objectives=np.asarray(args.objectives),
            timings=np.asarray([timings[n] for n in args.objectives]),
            build_seconds=build_seconds,
            thresholds=np.asarray([args.frl_dev, args.school_slack, args.capacity_dev]),
        )
        print(
            f"{stem}: {len(flat)} states in {time.monotonic() - start:.0f} s "
            f"(scorers built in {build_seconds:.0f} s) "
            + " ".join(f"{n}={t:.0f}s" for n, t in timings.items()),
            flush=True,
        )


if __name__ == "__main__":
    main()

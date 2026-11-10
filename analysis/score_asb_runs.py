"""Score benchmark runs' final zonings exactly as the walk states are scored.

For every ``zonings.json.xz`` under the given run directories, the final
stage's zoning is put through :mod:`analysis.score_recom_walks`: the same
three constraint checks (FRL, schools, capacity) and the same five objectives,
on the same Block_1 problem, so a run lands on the same axes as the naive
walk ensemble.

Output: a CSV with one row per run: ``centroids_type``, ``run_objective``
(what the run optimized; ``boundary_perimeter`` for a run whose config
weighted edges, whatever it was called), ``status``, the flags, the cell
and every objective.

    uv run python -m analysis.score_asb_runs CONFIG RUN_DIR [...] -o OUT.csv \
        [--centroids SET ...] [--exclude all]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import lzma
import os
import warnings

import numpy as np

from analysis.score_recom_walks import (
    DEFAULT_OBJECTIVES,
    build_scorers,
    close_all,
    constraint_columns,
    score,
    walk_problem,
)
from optimization.config import OptimizationConfig

FLAG_COLUMNS = (
    "frl_ok",
    "schools_ok",
    "capacity_ok",
    "cell",
    "frl_max_dev",
    "schools_max_dev",
    "capacity_ratio_min",
    "capacity_ratio_max",
)


def run_records(run_dirs, include, exclude):
    for run_dir in run_dirs:
        for path in sorted(
            glob.glob(os.path.join(run_dir, "**", "zonings.json.xz"), recursive=True)
        ):
            archive = json.load(lzma.open(path))
            config = archive["config"]
            centroids = config["centroids_type"]
            if centroids in exclude or (include and centroids not in include):
                continue
            stage = archive["stages"][-1]
            objective = config.get("objective")
            # A weighted-edge run costs boundaries in metres, whatever its name.
            if objective == "cut_edges" and config.get("weight_edges"):
                objective = "boundary_perimeter"
            yield {
                "path": path,
                "centroids_type": centroids,
                "run_objective": objective,
                "status": stage["status"],
                "level": stage["level"],
                "node_ids": [int(n) for n in stage.get("node_ids") or []],
                "node_zones": stage.get("node_zones"),
            }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--centroids", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=["all"])
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")

    config = OptimizationConfig.from_yaml(args.config)
    dataset = config.make_dataset()
    options = config.make_solver().options
    records = list(run_records(args.run_dirs, set(args.centroids or ()), set(args.exclude)))
    by_set: dict[str, list[dict]] = {}
    for record in records:
        by_set.setdefault(record["centroids_type"], []).append(record)

    rows = []
    for centroids, group in sorted(by_set.items()):
        problem = walk_problem(
            dataset, config, "Block_1", dataset.centroid_school_ids(centroids)
        )
        nodes = list(problem.nodes)
        position = {node: i for i, node in enumerate(nodes)}
        scored = [r for r in group if r["status"] in {"FEASIBLE", "OPTIMAL"}]
        for record in group:
            if record not in scored:
                rows.append({**_meta(record), "note": "no zoning"})
        if not scored:
            continue
        states = np.zeros((len(scored), len(nodes)), dtype=np.uint8)
        for row, record in enumerate(scored):
            if record["level"] != "Block_1" or set(record["node_ids"]) != set(nodes):
                raise ValueError(f"{record['path']}: not this problem's Block_1 nodes.")
            for node, zone in zip(record["node_ids"], record["node_zones"]):
                states[row, position[node]] = int(zone)
        flags = constraint_columns(
            problem, states, frl_dev=0.15, school_slack=1.0, capacity_dev=0.3
        )
        assignments = [dict(zip(nodes, map(int, s))) for s in states]
        scorers = build_scorers(problem, DEFAULT_OBJECTIVES, options)
        try:
            values = {
                name: score(problem, name, scorers[name], assignments)
                for name in DEFAULT_OBJECTIVES
            }
        finally:
            close_all(scorers)
        for row, record in enumerate(scored):
            rows.append(
                {
                    **_meta(record),
                    **{key: _plain(flags[key][row]) for key in FLAG_COLUMNS},
                    **{name: float(values[name][row]) for name in DEFAULT_OBJECTIVES},
                }
            )
        print(f"{centroids}: scored {len(scored)} run(s)", flush=True)

    fields = ["centroids_type", "run_objective", "status", "path", "note",
              *FLAG_COLUMNS, *DEFAULT_OBJECTIVES]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} run(s) -> {args.output}")


def _meta(record):
    return {k: record[k] for k in ("centroids_type", "run_objective", "status", "path")}


def _plain(value):
    return value.item() if hasattr(value, "item") else value


if __name__ == "__main__":
    main()

"""Combine scored walk states and summarize each constraint cell's objectives.

Reads every ``<centroids>_seed<s>.npz`` that ``analysis/score_recom_walks.py``
wrote and produces, in ``--output`` (default: the scores directory):

``states.npz``
    Every state as one row, columns as arrays: ``centroids_type``,
    ``num_zones``, ``seed``, ``step``, the three flags and worst-zone values,
    ``cell`` and every objective.
``cells.csv``
    One row per centroid set (and ``ALL`` for the pooled ensemble) and cell:
    the cell's state count and share, then for every objective its mean,
    standard deviation and the 0/5/25/50/75/95/100th percentiles.
``cell_counts.csv``
    The state count of every centroid set x cell, cells as columns.

A cell is the subset of the three constraints a state meets: bit 0 FRL,
bit 1 schools, bit 2 capacity, so cell 7 meets all three and cell 0 none.

    uv run python -m analysis.summarize_walk_scores SCORES_DIR [-o OUT]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import numpy as np

CELL_NAMES = {
    cell: "+".join(
        name
        for bit, name in enumerate(("frl", "schools", "capacity"))
        if cell >> bit & 1
    )
    or "none"
    for cell in range(8)
}
QUANTILES = (0, 5, 25, 50, 75, 95, 100)


def load(scores_dir: str) -> dict[str, np.ndarray]:
    paths = sorted(
        p
        for p in glob.glob(os.path.join(scores_dir, "*_seed*.npz"))
        if re.search(r"_seed\d+\.npz$", p)
    )
    if not paths:
        raise SystemExit(f"No scored walks in {scores_dir}.")
    parts: dict[str, list[np.ndarray]] = {}
    objectives = None
    for path in paths:
        z = np.load(path)
        names = [str(o) for o in z["objectives"]]
        if objectives is None:
            objectives = names
        elif names != objectives:
            raise ValueError(f"{path} scores {names}, not {objectives}.")
        n = len(z["cell"])
        row = {
            "centroids_type": np.full(n, str(z["centroids_type"])),
            "num_zones": np.full(n, int(z["num_zones"])),
        }
        for key in z.files:
            if key in {"centroids_type", "num_zones", "objectives", "timings",
                       "thresholds", "build_seconds"}:
                continue
            row[key] = z[key]
        for key, value in row.items():
            parts.setdefault(key, []).append(value)
    table = {key: np.concatenate(values) for key, values in parts.items()}
    table["objectives"] = np.asarray(objectives)
    return table


def summarize(table, output: str) -> None:
    objectives = [str(o) for o in table["objectives"]]
    sets = sorted(set(table["centroids_type"]), key=_set_order)
    header = ["centroids_type", "num_zones", "cell", "cell_name", "states", "share"]
    for name in objectives:
        header += [f"{name}_mean", f"{name}_std"] + [f"{name}_p{q}" for q in QUANTILES]
    rows = []
    counts = []
    for group in [*sets, "ALL"]:
        mask = np.ones(len(table["cell"]), bool) if group == "ALL" else (
            table["centroids_type"] == group
        )
        total = int(mask.sum())
        zones = "" if group == "ALL" else int(table["num_zones"][mask][0])
        cells = table["cell"][mask]
        counts.append([group, zones, *[int((cells == c).sum()) for c in range(8)], total])
        for cell in range(8):
            sel = mask & (table["cell"] == cell)
            n = int(sel.sum())
            row = [group, zones, cell, CELL_NAMES[cell], n, n / total if total else 0]
            for name in objectives:
                values = table[name][sel]
                values = values[np.isfinite(values)]
                if len(values):
                    row += [values.mean(), values.std()] + list(
                        np.percentile(values, QUANTILES)
                    )
                else:
                    row += [""] * (2 + len(QUANTILES))
            rows.append(row)
    with open(os.path.join(output, "cells.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    with open(os.path.join(output, "cell_counts.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["centroids_type", "num_zones", *[CELL_NAMES[c] for c in range(8)], "states"]
        )
        writer.writerows(counts)


def _set_order(name: str):
    match = re.match(r"(\d+)-zone-(\d+)", name)
    return (int(match.group(1)), int(match.group(2))) if match else (10**6, 0)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scores_dir")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)
    output = args.output or args.scores_dir
    os.makedirs(output, exist_ok=True)
    table = load(args.scores_dir)
    np.savez_compressed(os.path.join(output, "states.npz"), **table)
    summarize(table, output)
    print(
        f"{len(table['cell'])} states from "
        f"{len(set(table['centroids_type']))} centroid sets -> {output}"
    )


if __name__ == "__main__":
    main()

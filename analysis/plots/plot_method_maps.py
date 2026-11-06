#!/usr/bin/env python3
"""Final zoning of every method for one instance, side by side.

Finds, across one or more sweeps, the runs of one centroid set, seed and
target level (later sweeps replace earlier runs of the same task), and maps
each method's final zoning. Methods that found no feasible zoning get an empty
panel saying so.

Zone numbers are only labels, and ReCom does not keep a zone on its centroid,
so every zoning is relabelled to match a reference zoning (by default the
multi-level CP one) by largest block overlap. The same colour and number
then mean the same area in every panel::

    uv run python -m analysis.plots.plot_method_maps ROOT [ROOT ...] \
        --centroids 6-zone-3 --seed 1 --level Block_0 -o maps.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from analysis.plots.plot_method_grid import METHOD_ORDER, method_label
from analysis.plots.plot_multilevel_stages import (
    INK,
    INK_2,
    PROJECTED_CRS,
    SURFACE,
    _label_zones,
    _zone_color,
)
from optimization.config import OptimizationConfig
from optimization.data.loaders import load_census_shapefile
from optimization.levels import LevelSpec
from optimization.run_archive import area_assignment, get_stage, read_zonings

SUCCESS = {"FEASIBLE", "OPTIMAL"}
EMPTY = "#e6e5e0"
REFERENCE = "CP (Multi-Level)"


def _label(config: dict) -> str:
    return method_label(
        pd.Series(
            {
                "config_solver": config["solver"],
                "config_levels": json.dumps(config["levels"]),
                "config_recom_burst_length": config.get("recom_burst_length", 25),
                "config_recom_reject_infeasible": config.get("recom_reject_infeasible"),
                "config_recom_pair_selection": config.get("recom_pair_selection"),
                "config_recom_cut_selection": config.get("recom_cut_selection"),
            }
        )
    )


def find_runs(roots, centroids: str, seed: int, level: str) -> dict[str, dict]:
    """``{method: {status, cut_edges, run_dir}}``; later roots win per task."""
    runs: dict[str, dict] = {}
    for root in roots:
        for result in sorted(
            Path(root).glob(f"{centroids}/seed{seed}/*/*/*/result.json")
        ):
            r = json.loads(result.read_text())
            config = r["config"]
            if config["levels"][-1] != level:
                continue
            runs[_label(config)] = {
                "status": r["status"],
                "cut_edges": (r.get("metrics") or {}).get("cut_edges"),
                "run_dir": result.parent,
            }
    return runs


def relabel(assignment: dict[int, int], reference: dict[int, int]) -> dict[int, int]:
    """``assignment`` with zone labels matched to ``reference`` by overlap."""
    zones = sorted(set(assignment.values()) | set(reference.values()))
    pos = {z: i for i, z in enumerate(zones)}
    overlap = np.zeros((len(zones), len(zones)))
    for area, zone in assignment.items():
        if area in reference:
            overlap[pos[zone], pos[reference[area]]] += 1
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {zones[r]: zones[c] for r, c in zip(rows, cols)}
    return {area: mapping[zone] for area, zone in assignment.items()}


def plot(runs: dict[str, dict], output: Path, *, ncols: int = 4) -> None:
    feasible = {m: r for m, r in runs.items() if r["status"] in SUCCESS}
    if not feasible:
        raise ValueError("No method found a feasible zoning for this instance.")
    zonings = {}
    for method, run in feasible.items():
        archive = read_zonings(run["run_dir"])
        zonings[method] = area_assignment(archive, get_stage(archive))
    reference = REFERENCE if REFERENCE in zonings else next(iter(zonings))
    zonings = {m: relabel(z, zonings[reference]) for m, z in zonings.items()}

    some_run = next(iter(feasible.values()))["run_dir"]
    config = OptimizationConfig.from_yaml(str(some_run / "zoning_config.yaml"))
    unit = LevelSpec.parse(config.levels[-1]).unit
    blocks = load_census_shapefile(unit, config.data_scenario)[[unit, "geometry"]]
    blocks = blocks.dropna().astype({unit: "int64"})
    blocks = blocks.dissolve(by=unit, as_index=False).to_crs(PROJECTED_CRS)
    # Only blocks some zoning covers (the graph drops e.g. offshore islands),
    # and one shared extent so every panel draws the city at the same scale.
    zoned = set().union(*(z.keys() for z in zonings.values()))
    blocks = blocks[blocks[unit].isin(zoned)]
    outline = blocks.dissolve()
    xmin, ymin, xmax, ymax = blocks.total_bounds
    pad = 0.02 * max(xmax - xmin, ymax - ymin)

    methods = [m for m in METHOD_ORDER if m in runs]
    best = min(r["cut_edges"] for r in feasible.values())
    nrows = -(-len(methods) // ncols)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.9 * ncols, 4.3 * nrows), facecolor=SURFACE
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, method in zip(axes, methods):
        run = runs[method]
        if method in zonings:
            shown = blocks.assign(zone=blocks[unit].map(zonings[method])).dropna(
                subset=["zone"]
            )
            shown["zone"] = shown["zone"].astype(int)
            shown.plot(ax=ax, color=shown["zone"].map(_zone_color), linewidth=0)
            zones = shown.dissolve(by="zone", as_index=False)[["zone", "geometry"]]
            zones.boundary.plot(ax=ax, color=INK, linewidth=0.7)
            _label_zones(ax, zones, "zone")
            cut = int(run["cut_edges"])
            subtitle = f"{cut:,} cut edges" + ("  (best)" if cut == best else "")
            weight = "bold" if cut == best else "normal"
        else:
            outline.plot(ax=ax, color=EMPTY, linewidth=0)
            ax.text(
                0.5,
                0.5,
                "no feasible zoning\nwithin the budget",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=INK_2,
            )
            subtitle, weight = "–", "normal"
        ax.set_title(method, loc="left", fontsize=10.5, color=INK, pad=16)
        ax.text(
            0.0,
            1.0,
            subtitle,
            transform=ax.transAxes,
            fontsize=8.5,
            color=INK_2,
            fontweight=weight,
            va="bottom",
        )
    for ax in axes:
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots", nargs="+", help="Sweep folders; later ones override earlier runs."
    )
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--level", default="Block_0")
    parser.add_argument("-o", "--output", default="method_maps.png")
    args = parser.parse_args(argv)
    runs = find_runs(args.roots, args.centroids, args.seed, args.level)
    if not runs:
        raise SystemExit("No runs match that centroid set, seed and level.")
    plot(runs, Path(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

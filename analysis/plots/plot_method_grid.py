#!/usr/bin/env python3
"""Method comparison heatmaps: feasibility and cut edges by level and zones.

Reads the ``summary.csv`` of one or more benchmark sweeps and draws two
figures, each a row of heatmaps with one panel per target level (coarse to
fine), methods as rows and zone counts as columns:

* ``<stem>_feasible.png``: feasible runs out of all runs in the cell
  (seeds x centroid sets);
* ``<stem>_cut_edges.png``: the median cut edges of the feasible runs,
  coloured by how far it sits above the best median for that zone count
  across every method and level. Cut edges are counted on Block_0 for every
  level, so cells compare across levels.

Darker is better in both. Cells where fewer than ``--min-runs`` runs were
feasible are hatched, since their median rests on a handful of runs, and they
never set the reference best.

Later sweeps override earlier ones run by run (matched on ``task_id``), so a
rerun of some tasks replaces the original runs::

    uv run python -m analysis.plots.plot_method_grid \
        /soalnas/.../local_runs/benchmark_methods_2324 \
        /soalnas/.../local_runs/benchmark_methods_2324_mipfix \
        -o figures/method_grid.png

The aggregated table is written next to the figures as ``<stem>.csv``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUCCESS = {"FEASIBLE", "OPTIMAL"}
# Display names and order shared with analysis/plots/plot_benchmark_success_heatmap.py
# and benchmark/plot_edges.py.
METHOD_ORDER = [
    "Recom",
    "Relaxed Recom",
    "Short Bursts",
    "Adaptive Short Bursts",
    "MIP",
    "CP",
    "CP (Int)",
    "MIP (Multi-Level)",
    "CP (Multi-Level)",
    "CP (Int, Multi-Level)",
]
LEVEL_ORDER = ["Block_4", "Block_3", "Block_2", "Block_1", "Block_0"]

# Reference palette (dataviz skill): two sequential contexts, blue then orange.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
EMPTY = "#f0efec"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fde4d6", "#f9c3a6", "#f39f77", "#eb6834", "#c8521f", "#9c3f16", "#6e2c0f"]
# Gaps at or beyond this are drawn as the lightest step.
GAP_CAP = 1.0


def method_label(row: pd.Series) -> str:
    """Canonical display name for one summary row."""
    solver = str(row["config_solver"])
    levels = _levels(row["config_levels"])
    if solver == "recom":
        burst = int(float(row.get("config_recom_burst_length", 25)))
        uniform = (
            row.get("config_recom_pair_selection") == "uniform"
            and row.get("config_recom_cut_selection") == "uniform"
        )
        if burst == 0:
            # One continuous chain, with or without rejecting infeasible moves.
            return "Recom"
        return "Short Bursts" if uniform else "Adaptive Short Bursts"
    base = {"mip": "MIP", "cp_bool": "CP", "cp_int": "CP (Int)"}[solver]
    if len(levels) > 1:
        return (
            f"{base[:-1]}, Multi-Level)"
            if base.endswith(")")
            else f"{base} (Multi-Level)"
        )
    return base


def _levels(value) -> list[str]:
    if isinstance(value, list):
        return value
    return list(json.loads(str(value).replace("'", '"')))


def load_runs(roots: Sequence[str | Path]) -> pd.DataFrame:
    """One row per run; a later root replaces an earlier one's run of the same task."""
    frames = []
    for order, root in enumerate(roots):
        path = Path(root)
        path = path if path.suffix == ".csv" else path / "summary.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["_order"] = order
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True)
    runs = runs.sort_values("_order").drop_duplicates("task_id", keep="last")
    status = runs["final_status"].where(runs["final_status"] != "", runs["status"])
    out = pd.DataFrame(
        {
            "method": runs.apply(method_label, axis=1),
            "level": runs["config_levels"].map(lambda v: _levels(v)[-1]),
            "zones": runs["config_centroids_type"].map(
                lambda v: int(str(v).split("-")[0])
            ),
            "feasible": status.isin(SUCCESS),
            "cut_edges": pd.to_numeric(runs["cut_edges"], errors="coerce"),
        }
    )
    out.loc[~out["feasible"], "cut_edges"] = np.nan
    return out.reset_index(drop=True)


def aggregate(runs: pd.DataFrame, min_runs: int) -> pd.DataFrame:
    """Per (method, level, zones): runs, feasible runs, median cut edges, gap."""
    cells = (
        runs.groupby(["method", "level", "zones"])
        .agg(
            runs=("feasible", "size"),
            feasible=("feasible", "sum"),
            median_cut_edges=("cut_edges", "median"),
        )
        .reset_index()
    )
    reliable = cells[cells["feasible"] >= min_runs]
    best = reliable.groupby("zones")["median_cut_edges"].min()
    cells["best_for_zones"] = cells["zones"].map(best)
    cells["gap"] = cells["median_cut_edges"] / cells["best_for_zones"] - 1.0
    cells["reliable"] = cells["feasible"] >= min_runs
    return cells


def _ink_for(rgb) -> str:
    r, g, b = mcolors.to_rgb(rgb)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 0.45 else INK


METRICS = {
    "feasible": "Feasible Runs",
    "cut_edges": "Number of Cut Edges",
}


def _cell(c: pd.Series, kind: str, feas_cmap, gap_cmap):
    """``(fill, text, subtext)`` for one cell of the ``kind`` figure."""
    if kind == "feasible":
        fill = feas_cmap(c["feasible"] / c["runs"]) if c["feasible"] else EMPTY
        return fill, f"{int(c['feasible'])}/{int(c['runs'])}", None
    if not c["feasible"]:
        return EMPTY, "–", None
    gap = max(float(c["gap"]), 0.0)
    fill = gap_cmap(1.0 - min(gap, GAP_CAP) / GAP_CAP)
    sub = "best" if gap < 0.005 else f"+{gap:.0%}"
    return fill, f"{int(round(c['median_cut_edges'])):,}", sub


def plot(
    cells: pd.DataFrame,
    output: Path,
    kind: str,
    *,
    title: str | None = None,
) -> None:
    """One row of heatmaps (a panel per level) for ``kind`` in :data:`METRICS`."""
    methods = [m for m in METHOD_ORDER if m in set(cells["method"])]
    levels = [lv for lv in LEVEL_ORDER if lv in set(cells["level"])]
    zones = sorted(cells["zones"].unique())
    feas_cmap = mcolors.LinearSegmentedColormap.from_list("feas", BLUE)
    gap_cmap = mcolors.LinearSegmentedColormap.from_list("gap", ORANGE)
    header = METRICS[kind]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK,
            "hatch.linewidth": 0.7,
        }
    )
    # Laid out in inches so headings, level titles and tick labels never
    # collide, whatever the number of methods, zones or levels.
    cell_w, cell_h = 0.62, 0.42
    left, right, panel_gap = 1.75, 0.15, 0.3
    title_h = 0.4 if title else 0.0
    header_h, level_h, bottom_h = 0.35, 0.3, 0.5
    panel_w, panel_h = len(zones) * cell_w, len(methods) * cell_h
    width = left + len(levels) * panel_w + (len(levels) - 1) * panel_gap + right
    height = title_h + header_h + level_h + panel_h + bottom_h
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    index = cells.set_index(["method", "level", "zones"])

    for col, level in enumerate(levels):
        ax = fig.add_axes(
            (
                (left + col * (panel_w + panel_gap)) / width,
                bottom_h / height,
                panel_w / width,
                panel_h / height,
            )
        )
        ax.set_facecolor(SURFACE)
        for i, method in enumerate(methods):
            for j, z in enumerate(zones):
                key = (method, level, z)
                if key not in index.index:
                    continue
                c = index.loc[key]
                fill, text, sub = _cell(c, kind, feas_cmap, gap_cmap)
                box = dict(xy=(j + 0.04, i + 0.06), width=0.92, height=0.88)
                ax.add_patch(plt.Rectangle(**box, facecolor=fill, edgecolor="none"))
                if kind == "cut_edges" and c["feasible"] and not c["reliable"]:
                    ax.add_patch(
                        plt.Rectangle(
                            **box,
                            facecolor="none",
                            edgecolor=(1, 1, 1, 0.45),
                            hatch="///",
                            linewidth=0,
                            zorder=2,
                        )
                    )
                ink = _ink_for(fill) if fill != EMPTY else MUTED
                if sub is None:
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=ink,
                    )
                    continue
                weight = "bold" if sub == "best" else "normal"
                ax.text(
                    j + 0.5,
                    i + 0.38,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=ink,
                    fontweight=weight,
                )
                ax.text(
                    j + 0.5,
                    i + 0.70,
                    sub,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=ink,
                    fontweight=weight,
                )
        ax.set_xlim(0, len(zones))
        ax.set_ylim(len(methods), 0)
        ax.set_xticks(np.arange(len(zones)) + 0.5)
        ax.set_xticklabels([str(z) for z in zones])
        ax.set_yticks(np.arange(len(methods)) + 0.5)
        ax.set_yticklabels(methods if col == 0 else [])
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(level.replace("_", " "), fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("zones", fontsize=8.5)

    x_left = 0.12 / width
    if title:
        fig.text(
            x_left,
            (height - 0.08) / height,
            title,
            fontsize=11.5,
            fontweight="bold",
            ha="left",
            va="top",
        )
    fig.text(
        x_left,
        (height - title_h - 0.05) / height,
        header,
        fontsize=9.5,
        color=INK,
        fontweight="bold" if not title else "normal",
        ha="left",
        va="top",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def output_paths(output: Path) -> dict[str, Path]:
    """``<stem>_feasible.png`` and ``<stem>_cut_edges.png`` beside ``output``."""
    return {kind: output.with_name(f"{output.stem}_{kind}.png") for kind in METRICS}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots",
        nargs="+",
        help="Sweep folders (or summary.csv files); later ones override earlier runs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="method_grid.png",
        help="Base path: writes <stem>_feasible.png, <stem>_cut_edges.png and <stem>.csv.",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=5,
        help="Feasible runs a cell needs before its median can be the best (default 5).",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    output = Path(args.output)
    cells = aggregate(load_runs(args.roots), args.min_runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = output.with_suffix(".csv")
    cells.sort_values(["level", "zones", "method"]).to_csv(table, index=False)
    for kind, path in output_paths(output).items():
        plot(cells, path, kind, title=args.title)
        print(f"Saved {path}")
    print(f"Saved {table}")


if __name__ == "__main__":
    main()

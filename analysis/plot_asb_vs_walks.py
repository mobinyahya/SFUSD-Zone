"""Where adaptive short bursts land against the naive ReCom walk ensemble.

Inputs: the walk ensemble's scored states (``states.npz`` from
``analysis/summarize_walk_scores.py``) and the ASB runs scored the same way
(``analysis/score_asb_runs.py``). Figures, in ``--output``:

``objective_<name>.png`` (one per objective)
    A 5 x 4 grid, one panel per centroid set (rows: 6/8/10/13/18 zones). Each
    panel is the set's 100,100 walk states as step histograms on a log count
    axis, split by how many of the three constraints a state meets (FRL
    +/-0.15, schools within 1 of the average, capacity +/-30%), with the ASB
    zoning, which meets all three, as an orange line. The text gives the share
    of walk states meeting one or more and two or more constraints, and the
    share of all walk states the ASB zoning beats.
``asb_rank.png``
    The share of walk states each ASB zoning beats, per set and objective.
``constraint_shares.png``
    The share of walk states meeting one or more, two or more and all three
    constraints, per set, on a log axis.

    uv run python -m analysis.plot_asb_vs_walks STATES.npz ASB.csv -o FIGDIR
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Reference palette (dataviz skill), light mode. The three blue steps are an
# ordinal ramp (0 < 1 < 2 constraints met); orange is categorical slot 2.
# Validated all-pairs: normal-vision dE >= 19.5, CVD dE >= 19.0.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MET_COLORS = {0: "#86b6ef", 1: "#2a78d6", 2: "#104281"}
ASB = "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

OBJECTIVES = {
    # name: (label, unit, lower_is_better)
    "cut_edges": ("Cut edges", "edges", True),
    "average_distance": ("Average distance to school", "miles", True),
    "matching_welfare": ("Matching welfare", "MID welfare", False),
    "capacity_match": ("Unassigned students (capacity match)", "students", True),
    "frl_dissimilarity": ("FRL dissimilarity", "percentage points", True),
}
MET_LABELS = {0: "0 of 3 constraints met", 1: "1 of 3 met", 2: "2 of 3 met"}
RUN_LABELS = {
    "boundary_perimeter": "ASB, boundary-length objective",
    "capacity_match": "ASB, capacity-match objective",
}


def set_order(name):
    m = re.match(r"(\d+)-zone-(\d+)", name)
    return (int(m.group(1)), int(m.group(2)))


def style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "font.size": 8,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "savefig.facecolor": SURFACE,
        }
    )


def load_states(path):
    z = np.load(path, allow_pickle=True)
    met = (
        z["frl_ok"].astype(int) + z["schools_ok"].astype(int) + z["capacity_ok"].astype(int)
    )
    data = {"centroids_type": z["centroids_type"].astype(str), "met": met}
    for name in OBJECTIVES:
        data[name] = z[name]
    return data


def load_asb(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get(next(iter(OBJECTIVES)))]
    for r in rows:
        for name in OBJECTIVES:
            r[name] = float(r[name])
        r["met"] = sum(r[k] == "True" for k in ("frl_ok", "schools_ok", "capacity_ok"))
    return rows


def share_text(mask) -> str:
    """A share as a percentage, or as a count of states when under 1%."""
    n, total = int(mask.sum()), len(mask)
    if n == 0:
        return "none"
    if n / total < 0.01:
        return f"{n:,} state{'s' if n != 1 else ''}"
    return f"{n / total:.1%}"


def beats(values, x, lower_is_better):
    """Share of ``values`` that ``x`` is strictly better than."""
    return float(np.mean(values > x) if lower_is_better else np.mean(values < x))


def objective_figure(data, asb, name, sets, output):
    label, unit, lower = OBJECTIVES[name]
    rows = sorted({set_order(s)[0] for s in sets})
    by_row = {r: [s for s in sets if set_order(s)[0] == r] for r in rows}
    ncols = max(len(v) for v in by_row.values())
    fig, axes = plt.subplots(
        len(rows), ncols, figsize=(3.0 * ncols, 2.2 * len(rows) + 0.9), squeeze=False
    )
    for i, zones in enumerate(rows):
        row_sets = by_row[zones]
        masks = {s: data["centroids_type"] == s for s in row_sets}
        row_values = np.concatenate([data[name][masks[s]] for s in row_sets])
        row_values = row_values[np.isfinite(row_values)]
        marks = [r[name] for r in asb if r["centroids_type"] in row_sets]
        lo, hi = np.percentile(row_values, [0.05, 99.95])
        lo, hi = min([lo, *marks]), max([hi, *marks])
        pad = 0.03 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, 61)
        for j in range(ncols):
            ax = axes[i, j]
            if j >= len(row_sets):
                ax.set_visible(False)
                continue
            s = row_sets[j]
            values = data[name][masks[s]]
            met = data["met"][masks[s]]
            ok = np.isfinite(values)
            values, met = values[ok], met[ok]
            for k in (0, 1, 2):
                sel = values[met == k]
                if not len(sel):
                    continue
                counts, _ = np.histogram(sel, edges)
                ax.stairs(
                    counts, edges, color=MET_COLORS[k], linewidth=1.2,
                    fill=False, zorder=2 + k,
                )
                ax.stairs(
                    counts, edges, color=MET_COLORS[k], alpha=0.18 if k == 0 else 0.30,
                    fill=True, linewidth=0, zorder=1 + k,
                )
            ax.set_yscale("log")
            peak = max(np.histogram(values, edges)[0].max(), 1)
            ax.set_ylim(0.7, peak * 3)
            ax.set_xlim(edges[0], edges[-1])
            text = [f"≥1 met: {share_text(met >= 1)}  ·  ≥2 met: {share_text(met >= 2)}"]
            ranks = []
            for r in sorted(
                (r for r in asb if r["centroids_type"] == s), key=lambda r: r["run_objective"]
            ):
                dashed = r["run_objective"] != "boundary_perimeter"
                ax.axvline(
                    r[name], color=SURFACE, linewidth=3.6, zorder=8,
                )
                ax.axvline(
                    r[name], color=ASB, linewidth=1.8, zorder=9,
                    linestyle=(0, (3, 1.6)) if dashed else "-",
                )
                tag = "cap.-match ASB" if dashed else "ASB"
                ranks.append(f"{tag} beats {beats(values, r[name], lower):.1%}")
            # One line per ASB run, so two never run into the set name.
            text += [f"{rank} of walk states" if i == 0 else rank for i, rank in enumerate(ranks)]
            # Set name left, its shares and the ASB rank right, both above the
            # plot area so no text ever crosses a mark.
            ax.set_title(s, loc="left", fontsize=8.5, fontweight="bold", pad=4)
            ax.set_title(
                "\n".join(text), loc="right", fontsize=6.3, color=INK_2, pad=4,
                linespacing=1.35,
            )
            ax.tick_params(labelsize=6.5, length=2)
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(f"{zones} zones\nwalk states", fontsize=7.5)
            if i == len(rows) - 1 or j >= len(by_row.get(rows[min(i + 1, len(rows) - 1)], [])):
                ax.set_xlabel(unit, fontsize=7)
    direction = "lower is better" if lower else "higher is better"
    fig.suptitle(
        f"{label}: ASB zoning vs. 100,100 naive ReCom walk states per centroid set",
        x=0.012, y=0.993, ha="left", fontsize=11, fontweight="bold", color=INK,
    )
    fig.text(
        0.012, 0.968,
        f"{direction}  ·  log count  ·  no walk state meets all three constraints "
        "(FRL ±0.15, schools within 1 of the average, capacity ±30%); every ASB zoning does",
        ha="left", fontsize=7.8, color=INK_2,
    )
    handles = [
        Line2D([0], [0], color=MET_COLORS[k], linewidth=6, alpha=0.6, label=MET_LABELS[k])
        for k in (0, 1, 2)
    ] + [
        Line2D([0], [0], color=ASB, linewidth=1.8, label=RUN_LABELS["boundary_perimeter"]),
        Line2D([0], [0], color=ASB, linewidth=1.8, linestyle=(0, (3, 1.6)),
               label=RUN_LABELS["capacity_match"] + " (10-zone-1 only)"),
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.960), ncol=5,
        frameon=False, fontsize=7.2, labelcolor=INK_2, handlelength=2.2,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.952), h_pad=1.1, w_pad=0.8)
    path = os.path.join(output, f"objective_{name}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def rank_figure(data, asb, sets, output):
    main = {r["centroids_type"]: r for r in asb if r["run_objective"] == "boundary_perimeter"}
    names = list(OBJECTIVES)
    grid = np.full((len(sets), len(names)), np.nan)
    for i, s in enumerate(sets):
        mask = data["centroids_type"] == s
        for j, name in enumerate(names):
            if s in main:
                values = data[name][mask]
                grid[i, j] = beats(values[np.isfinite(values)], main[s][name], OBJECTIVES[name][2])
    fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(sets) + 1.6))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(sets)):
        for j in range(len(names)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1%}" if v < 0.01 or v >= 0.995 else f"{v:.0%}", ha="center",
                        va="center", fontsize=7,
                        color="#ffffff" if v > 0.55 else INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{OBJECTIVES[n][0].split(' (')[0]}\n({'lower' if OBJECTIVES[n][2] else 'higher'} is better)"
         for n in names], fontsize=7, color=INK_2,
    )
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels(sets, fontsize=7.5, color=INK_2)
    ax.tick_params(length=0)
    for k in range(1, len(sets)):
        if set_order(sets[k])[0] != set_order(sets[k - 1])[0]:
            ax.axhline(k - 0.5, color=SURFACE, linewidth=3)
    ax.set_title(
        "Share of naive walk states the ASB zoning beats",
        loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=34,
    )
    fig.text(
        0.012, 0.012,
        "ASB run: hedge preset, 300 s, boundary-length objective, all three constraints held. "
        "Walk states: 100,100 per set, none meeting all three.",
        fontsize=6.6, color=MUTED, ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(output, "asb_rank.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def shares_figure(data, sets, output):
    fig, ax = plt.subplots(figsize=(6.4, 0.27 * len(sets) + 1.4))
    y = np.arange(len(sets))[::-1]
    floor = 1 / 100100 / 3
    for k, (thresh, color, label) in enumerate(
        ((1, MET_COLORS[0], "≥1 constraint met"),
         (2, MET_COLORS[1], "≥2 met"),
         (3, MET_COLORS[2], "all 3 met"))
    ):
        shares = np.array([np.mean(data["met"][data["centroids_type"] == s] >= thresh) for s in sets])
        zero = shares == 0
        ax.scatter(shares[~zero], y[~zero], s=34, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2, label=label)
        if zero.any():
            # Open markers in the "none" column, one row offset per series.
            ax.scatter(np.full(zero.sum(), floor), y[zero] + (1 - k) * 0.24, s=26,
                       facecolor=SURFACE, edgecolor=color, linewidth=1.3, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(floor / 1.8, 1.2)
    ax.axvline(floor, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(sets, fontsize=7.5, color=INK_2)
    ax.grid(axis="y", visible=False)
    ticks = [floor, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["none\n(open marks)"] + [f"{t * 100:g}%" for t in ticks[1:]])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("share of the set's 100,100 walk states (log scale)", fontsize=7.5)
    ax.set_title("How often a naive ReCom walk meets the constraints",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
              fontsize=7.2, labelcolor=INK_2, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    path = os.path.join(output, "constraint_shares.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("states")
    parser.add_argument("asb")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--exclude", nargs="*", default=["all"])
    args = parser.parse_args(argv)
    os.makedirs(args.output, exist_ok=True)
    style()
    data = load_states(args.states)
    asb = [r for r in load_asb(args.asb) if r["centroids_type"] not in args.exclude]
    sets = sorted(
        (s for s in set(data["centroids_type"]) if s not in args.exclude), key=set_order
    )
    for name in OBJECTIVES:
        print(objective_figure(data, asb, name, sets, args.output))
    print(rank_figure(data, asb, sets, args.output))
    print(shares_figure(data, sets, args.output))


if __name__ == "__main__":
    main()

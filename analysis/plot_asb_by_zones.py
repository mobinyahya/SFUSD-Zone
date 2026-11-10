"""ASB against the naive walk ensemble, pooled by zone count, like for like.

Each objective is compared only with the ASB runs that optimized it, and each
zone count's ASB runs only with that zone count's walk states: the four
centroid sets of a zone count pool into one distribution (400,400 states), and
every ASB run on one of those sets that optimized the objective is a marker on
it. A run that optimized something else (e.g. boundary length) appears on no
panel. Figures, in ``--output``:

``by_zones_overview.png``
    Objectives as rows, zone counts as columns. Each panel: the pooled walk
    states as log-count step histograms by constraints met, and the ASB runs
    that optimized that row's objective as orange lines, with the share of the
    pooled walk states each beats.
``by_zones_rank.png``
    Per zone count and objective, the share of the pooled walk states the
    objective's ASB runs beat: median, and the range over centroid sets.
``by_zones_shares.png``
    Per zone count, the share of walk states meeting >=1, >=2 and all 3
    constraints.

    uv run python -m analysis.plot_asb_by_zones STATES.npz ASB.csv -o FIGDIR
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analysis.plot_asb_vs_walks import (  # noqa: E402
    ASB,
    AXIS,
    INK,
    INK_2,
    MET_COLORS,
    MET_LABELS,
    MUTED,
    OBJECTIVES,
    SEQ,
    SURFACE,
    beats,
    load_asb,
    load_states,
    set_order,
    share_text,
    style,
)


def zone_counts(data, exclude):
    sets = {s for s in set(data["centroids_type"]) if s not in exclude}
    zones = sorted({set_order(s)[0] for s in sets})
    return zones, {z: sorted((s for s in sets if set_order(s)[0] == z), key=set_order) for z in zones}


def pooled(data, sets_of_zone):
    return np.isin(data["centroids_type"], sets_of_zone)


def matching_runs(asb, name, sets_of_zone):
    """The ASB runs on this zone count's sets that optimized ``name``."""
    return sorted(
        (r for r in asb if r["run_objective"] == name and r["centroids_type"] in sets_of_zone),
        key=lambda r: set_order(r["centroids_type"]),
    )


def overview_figure(data, asb, zones, by_zone, output):
    names = list(OBJECTIVES)
    fig, axes = plt.subplots(
        len(names), len(zones), figsize=(3.05 * len(zones), 2.35 * len(names) + 1.0),
        squeeze=False,
    )
    for i, name in enumerate(names):
        label, unit, lower = OBJECTIVES[name]
        for j, zones_n in enumerate(zones):
            ax = axes[i, j]
            sets = by_zone[zones_n]
            mask = pooled(data, sets)
            values, met = data[name][mask], data["met"][mask]
            ok = np.isfinite(values)
            values, met = values[ok], met[ok]
            runs = matching_runs(asb, name, sets)
            lo, hi = np.percentile(values, [0.05, 99.95])
            lo = min([lo, *(r[name] for r in runs)])
            hi = max([hi, *(r[name] for r in runs)])
            pad = 0.03 * (hi - lo)
            edges = np.linspace(lo - pad, hi + pad, 61)
            for k in (0, 1, 2):
                sel = values[met == k]
                if not len(sel):
                    continue
                counts, _ = np.histogram(sel, edges)
                ax.stairs(counts, edges, color=MET_COLORS[k], linewidth=1.2, zorder=2 + k)
                ax.stairs(counts, edges, color=MET_COLORS[k], fill=True, linewidth=0,
                          alpha=0.18 if k == 0 else 0.30, zorder=1 + k)
            peak = max(np.histogram(values, edges)[0].max(), 1)
            ax.set_yscale("log")
            ax.set_ylim(0.7, peak * 3)
            ax.set_xlim(edges[0], edges[-1])
            for r in runs:
                ax.axvline(r[name], color=SURFACE, linewidth=3.6, zorder=8)
                ax.axvline(r[name], color=ASB, linewidth=1.8, zorder=9)
            text = [f"≥1 met: {share_text(met >= 1)}  ·  ≥2 met: {share_text(met >= 2)}"]
            if runs:
                shares = [beats(values, r[name], lower) for r in runs]
                listed = ", ".join(f"{s:.1%}" for s in shares)
                text.append(f"{len(runs)} ASB run{'s' if len(runs) > 1 else ''} beat {listed}")
            else:
                text.append("no ASB run optimized this yet")
            ax.set_title(f"{zones_n} zones", loc="left", fontsize=8.5, fontweight="bold", pad=4)
            ax.set_title("\n".join(text), loc="right", fontsize=6.2, color=INK_2, pad=4,
                         linespacing=1.35)
            ax.tick_params(labelsize=6.5, length=2)
            ax.grid(axis="x", visible=False)
            if j == 0:
                direction = "lower is better" if lower else "higher is better"
                ax.set_ylabel(f"{label}\n({direction})\nwalk states", fontsize=7.3)
            ax.set_xlabel(unit, fontsize=6.8)
    fig.suptitle(
        "ASB vs. the naive ReCom walk ensemble, by zone count, each objective against "
        "the ASB runs that optimized it",
        x=0.01, y=0.995, ha="left", fontsize=11, fontweight="bold", color=INK,
    )
    fig.text(
        0.01, 0.976,
        "Each panel pools the zone count's four centroid sets (400,400 walk states, log "
        "count). No walk state meets all three constraints (FRL ±0.15, schools within 1 "
        "of the average, capacity ±30%); every ASB zoning does.",
        ha="left", fontsize=7.6, color=INK_2,
    )
    handles = [
        Line2D([0], [0], color=MET_COLORS[k], linewidth=6, alpha=0.6, label=MET_LABELS[k])
        for k in (0, 1, 2)
    ] + [Line2D([0], [0], color=ASB, linewidth=1.8,
                label="ASB run that optimized this row's objective (one per centroid set)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.004, 0.97), ncol=4,
               frameon=False, fontsize=7.2, labelcolor=INK_2, handlelength=2.2,
               columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.953), h_pad=1.2, w_pad=0.8)
    path = os.path.join(output, "by_zones_overview.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def rank_figure(data, asb, zones, by_zone, output):
    names = list(OBJECTIVES)
    median = np.full((len(zones), len(names)), np.nan)
    labels = [["" for _ in names] for _ in zones]
    for i, zones_n in enumerate(zones):
        sets = by_zone[zones_n]
        mask = pooled(data, sets)
        for j, name in enumerate(names):
            runs = matching_runs(asb, name, sets)
            values = data[name][mask]
            values = values[np.isfinite(values)]
            if not runs:
                labels[i][j] = "no ASB run yet"
                continue
            shares = np.array([beats(values, r[name], OBJECTIVES[name][2]) for r in runs])
            median[i, j] = np.median(shares)
            fmt = lambda v: f"{v:.1%}" if v < 0.01 or v >= 0.995 else f"{v:.0%}"  # noqa: E731
            labels[i][j] = (
                f"{fmt(median[i, j])}\n{fmt(shares.min())}–{fmt(shares.max())}  ({len(runs)} runs)"
                if len(runs) > 1 else f"{fmt(median[i, j])}\n(1 run)"
            )
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(zones) + 1.7))
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    cmap.set_bad("#f0efec")
    ax.imshow(np.ma.masked_invalid(median), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(zones)):
        for j in range(len(names)):
            v = median[i, j]
            dark = np.isfinite(v) and v > 0.55
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=7.4 if np.isfinite(v) else 6.6, linespacing=1.3,
                    color="#ffffff" if dark else (INK if np.isfinite(v) else MUTED))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{OBJECTIVES[n][0].split(' (')[0]}\n({'lower' if OBJECTIVES[n][2] else 'higher'} is better)"
         for n in names], fontsize=7, color=INK_2)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels([f"{z} zones" for z in zones], fontsize=8, color=INK_2)
    ax.tick_params(length=0)
    for k in range(1, len(zones)):
        ax.axhline(k - 0.5, color=SURFACE, linewidth=2)
    ax.set_title("Share of the zone count's walk states beaten by the ASB runs that "
                 "optimized each objective", loc="left", fontsize=10, fontweight="bold",
                 color=INK, pad=34)
    fig.text(0.01, 0.015,
             "Median over the zone count's centroid sets, with the range. Walk states: the "
             "four sets pooled, 400,400, none meeting all three constraints.",
             fontsize=6.6, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    path = os.path.join(output, "by_zones_rank.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def shares_figure(data, zones, by_zone, output):
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(zones) + 1.4))
    y = np.arange(len(zones))[::-1]
    totals = [pooled(data, by_zone[z]).sum() for z in zones]
    floor = 1 / max(totals) / 3
    for k, (thresh, color, label) in enumerate(
        ((1, MET_COLORS[0], "≥1 constraint met"), (2, MET_COLORS[1], "≥2 met"),
         (3, MET_COLORS[2], "all 3 met"))
    ):
        shares = np.array([np.mean(data["met"][pooled(data, by_zone[z])] >= thresh) for z in zones])
        zero = shares == 0
        ax.scatter(shares[~zero], y[~zero], s=40, color=color, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2, label=label)
        if zero.any():
            ax.scatter(np.full(zero.sum(), floor), y[zero] + (1 - k) * 0.18, s=30,
                       facecolor=SURFACE, edgecolor=color, linewidth=1.3, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(floor / 1.8, 1.2)
    ax.axvline(floor, color=AXIS, linewidth=0.8, zorder=1)
    ticks = [floor, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["none\n(open marks)"] + [f"{t * 100:g}%" for t in ticks[1:]])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(y)
    ax.set_yticklabels([f"{z} zones" for z in zones], fontsize=8, color=INK_2)
    ax.set_ylim(-0.6, len(zones) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of the zone count's 400,400 walk states (log scale)", fontsize=7.5)
    ax.set_title("How often a naive ReCom walk meets the constraints, by zone count",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
              fontsize=7.2, labelcolor=INK_2, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    path = os.path.join(output, "by_zones_shares.png")
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
    zones, by_zone = zone_counts(data, set(args.exclude))
    print(overview_figure(data, asb, zones, by_zone, args.output))
    print(rank_figure(data, asb, zones, by_zone, args.output))
    print(shares_figure(data, zones, by_zone, args.output))


if __name__ == "__main__":
    main()

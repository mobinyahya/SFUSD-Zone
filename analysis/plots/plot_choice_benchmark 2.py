#!/usr/bin/env python3
"""Plot choice-benchmark strategy comparisons and model-vs-actual utility traces.

Reads the per-run ``result.json`` files written by ``benchmark.run`` and produces:

1. ``choice_strategy_comparison.png`` - which strategy is best, on one common
   yardstick. Each run is represented by its best saved zoning, chosen on budget
   welfare by ``analysis.plots.recompute_mid_welfare`` rather than by the
   strategy's own surrogate, and that zoning's budget welfare and cut edges are
   what the two panels show. Budget welfare is the only welfare reported here
   and the only measure rows are ranked on: the MNL logsum is a surrogate
   internal to ``iterative_choice``, not a cross-strategy yardstick, so it is
   deliberately absent from this figure. It still drives figure 2, which is a
   convergence diagnostic for that strategy rather than a comparison. The
   welfare of the discrete MID program is labelled *budget welfare* throughout
   the figure, after the budget sets it clears; the data columns keep the ``mid``
   names of the code that computes them. When
   ``analysis.plots.relaxed_recom_baseline`` has written its CSV, the feasible
   relaxed-ReCom chain it walked joins the strategy axis as a baseline row.
2. ``choice_model_vs_actual.png`` - for ``iterative_choice``, the master
   problem's model utility against the exact evaluation of the same zoning,
   per iteration.
3. ``saa_model_vs_actual.png`` - the same for ``saa``, plus the global upper
   bound, which stays on scale for this strategy.
4. ``priced_access_model_vs_actual.png`` - the same for ``priced_access``. Its
   "actual" series is the exact value of the priced bound rather than realized
   welfare, because that strategy evaluates its surrogate in closed form and
   never solves a recourse LP; the figure is therefore a convergence diagnostic
   for the cutting plane and says nothing about realized welfare. Since the
   priced-access sweeps replicate over seeds and price arms, only one run is
   traced per panel (see ``single_trace_per_centroid``).

MIP-backed runs are excluded throughout (see ``EXCLUDED_SOLVERS``), as are runs
the benchmark marked ``ERROR`` (see ``iter_results``, and ``--keep-failed`` to
put them back). No figure reports compute cost - solve time is deliberately not
part of this comparison.

The figures describe what they plot and state no conclusions about it: every
caption is either methodology or a count read off the data being drawn. Nothing
here interprets the result, by design.

Conventions: color encodes the centroid configuration in figure 1 and
the utility series (model / actual / bound) in figures 2-4; identity is always
carried by a legend plus direct labels, never by color alone.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DEFAULT_BENCHMARK_DIR = (
    "/soalnas/share/data/school_choice/local_runs/benchmark_choice_v4"
)

# Solver backends skipped everywhere in this analysis.
EXCLUDED_SOLVERS = frozenset({"mip"})

# Validated categorical slots (light surface); see the dataviz palette reference.
SURFACE = "#fcfcfb"
BAND = "#f4f3ef"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e3e2dd"
# The utility channel of figures 2-4: model, actual, bound. Exactly three
# things are ever encoded here, so this list is deliberately three long.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua

# The centroid channel of figure 1. Unlike SERIES this has to stretch to
# however many centroid configurations the sweep happened to run, so it carries
# its own slots rather than borrowing the three utility ones.
CENTROID_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#9b5cc4",  # violet
    "#b38600",  # ochre
    "#41748d",  # slate
]
CENTROID_MARKER_CYCLE = ["o", "s", "D", "^", "v", "P"]

# Display names and marker shapes for strategy/solver variants.
VARIANT_LABELS = {
    ("saa", "cp_bool"): "SAA",
    ("iterative_choice", "cp_bool"): "Iterative Choice",
    ("short_bursts_choice", "short_bursts"): "Short Bursts (Relaxed ReCom)",
    ("mid", "cp_bool"): "MID",
    ("mid_decomp", "cp_bool"): "MID Decomp",
    ("priced_access", "cp_bool"): "Priced Access",
}

STRATEGY_LABELS = {
    "iterative_choice": "Iterative Choice",
    "saa": "SAA",
    "priced_access": "Priced Access",
}

# Strategies that expose a per-iteration model/actual pair to trace.
TRACED_STRATEGIES = ("iterative_choice", "saa", "priced_access")

# Row label for the unguided relaxed-ReCom chain.
BASELINE_LABEL = "Relaxed ReCom (baseline)"

# Above this many replicates in a row, one marker per point paints a solid bar,
# so the row shows its range, this inner band, and its mean instead.
DISTRIBUTION_MIN_POINTS = 25
DISTRIBUTION_BAND = (25.0, 75.0)

# Fraction of a strategy row that its per-centroid sub-rows span, leaving the
# rest as the gap to the next strategy. The row height needed to keep the value
# labels from overprinting is derived from this, so the two cannot drift apart.
SUBROW_SPAN = 0.52


def centroid_channel(
    centroids: Sequence[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Colour and marker for each centroid configuration on display.

    Both channels are assigned here rather than indexed inline, so a sweep with
    more centroid configurations than the palette has slots degrades to reused
    slots with a warning instead of an ``IndexError`` mid-figure. Identity in
    these figures is carried by a legend and direct labels as well, so a reused
    slot costs discriminability rather than correctness -- but it is still a
    figure that wants a wider palette, which is what the warning says.
    """
    if len(centroids) > len(CENTROID_COLORS):
        print(
            f"  note: {len(centroids)} centroid configurations but only "
            f"{len(CENTROID_COLORS)} palette slots; colours repeat"
        )
    colors = {
        centroid: CENTROID_COLORS[i % len(CENTROID_COLORS)]
        for i, centroid in enumerate(centroids)
    }
    markers = {
        centroid: CENTROID_MARKER_CYCLE[i % len(CENTROID_MARKER_CYCLE)]
        for i, centroid in enumerate(centroids)
    }
    return colors, markers


def _variant_label(config: dict[str, Any]) -> str:
    """Row identity for the strategy axis.

    ``priced_access`` arms differ only in their congestion price multiplier,
    which is the thing under test, so it has to be part of the row identity --
    otherwise several distinct configurations average into a single row.
    """
    label = VARIANT_LABELS.get(
        (config["strategy"], config["solver"]),
        f"{config['strategy']} ({config['solver']})",
    )
    if config["strategy"] != "priced_access":
        return label
    scale = float(config.get("priced_access_price_scale", 1.0))
    return f"{label} (no pricing)" if scale == 0.0 else f"{label} (price x{scale:g})"


def style() -> None:
    """Apply shared figure styling: recessive grid/axes, text-token ink."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 9.5,
        }
    )


def recede(ax: plt.Axes, axis: str = "both") -> None:
    """Hide top/right spines and put a light grid behind the marks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=axis, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def fmt_gap(gap: float, actual: float) -> str:
    """Format an absolute/relative optimism gap without losing small magnitudes."""
    pct = 100.0 * gap / actual
    abs_txt = f"{gap:,.0f}" if abs(gap) >= 10 else f"{gap:.3g}"
    pct_txt = f"{pct:.1f}%" if abs(pct) >= 0.1 else f"{pct:.2g}%"
    return f"final gap {abs_txt}  ({pct_txt})"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _matches(config: dict[str, Any], key: str, wanted: str) -> bool:
    """Whether a run's config sets ``key`` to ``wanted``, compared as written."""
    value = config.get(key)
    if str(value) == wanted:
        return True
    try:  # 10 and "10" and 10.0 are the same arm
        return float(value) == float(wanted)
    except (TypeError, ValueError):
        return False


def iter_results(
    benchmark_dir: Path,
    keep_failed: bool = False,
    config_filters: Sequence[tuple[str, str]] = (),
) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield ``(path, parsed)`` for every kept ``result.json`` under the benchmark.

    Runs whose solver is in ``EXCLUDED_SOLVERS`` are skipped here so no figure or
    table downstream can pick them up.

    So are runs the benchmark marked ``ERROR``. A run that raised part-way can
    still have written a zoning from the stage it completed, and nothing
    downstream distinguishes that zoning from one a healthy run spent its whole
    budget on -- it lands in its strategy's row as an ordinary replicate. That
    is a point about a strategy stopped after a fraction of its budget, so it
    is dropped by default; ``keep_failed`` puts it back for diagnosing the
    failures themselves.

    ``config_filters`` keeps only runs whose config matches every given
    ``(key, value)``. A sweep arm that ``_variant_label`` does not split on --
    ``saa_num_seeds`` for ``direct_samples``, say -- otherwise averages several
    distinct configurations into one row; this restricts the figure to one of
    them rather than silently pooling.
    """
    kept_failures = dropped_failures = dropped_filtered = 0
    for path in sorted(benchmark_dir.rglob("result.json")):
        with path.open() as handle:
            result = json.load(handle)
        if result["config"]["solver"] in EXCLUDED_SOLVERS:
            continue
        if any(
            not _matches(result["config"], key, value)
            for key, value in config_filters
        ):
            dropped_filtered += 1
            continue
        if "stages" not in result.get("run", {}):
            # The optimization phase writes a metrics-free payload, so a run
            # still in flight carries no welfare to plot yet.
            continue
        if result.get("status") == "ERROR":
            if not keep_failed:
                dropped_failures += 1
                continue
            kept_failures += 1
        yield path, result
    if dropped_failures:
        print(
            f"  note: {dropped_failures} errored run(s) excluded; "
            "pass --keep-failed to include them"
        )
    if kept_failures:
        print(f"  note: {kept_failures} errored run(s) kept by --keep-failed")
    if dropped_filtered:
        terms = ", ".join(f"{key}={value}" for key, value in config_filters)
        print(f"  note: {dropped_filtered} run(s) excluded by --filter {terms}")


def attach_best_stage(runs: pd.DataFrame, stage_csv: Path | None) -> pd.DataFrame:
    """Score each run by its best saved zoning instead of the one it returned.

    ``analysis.plots.recompute_mid_welfare`` replays every zoning a run saved -
    each iteration of ``iterative_choice`` and ``saa``, each stage of
    ``mid_decomp`` - and scores all of them on the same three measures. Every
    strategy picks its answer with its own surrogate, so this comparison picks
    instead: the saved zoning with the highest budget welfare, then reports that
    zoning's choice welfare and cut edges alongside. Ties keep the stage the run
    itself returned.

    Without the replay CSV the run's own final zoning stands and the budget
    panel is simply omitted, since the benchmark stores no MID welfare for
    ``iterative_choice`` at all.
    """
    if stage_csv is None or not stage_csv.exists():
        return runs
    stages = pd.read_csv(stage_csv)
    if not {"path", "mid_welfare"}.issubset(stages.columns):
        print(f"  note: {stage_csv.name} has no budget welfare; using final zonings")
        return runs

    ranking = ["mid_welfare"] + (["is_selected"] if "is_selected" in stages else [])
    best = (
        stages.sort_values(ranking, ascending=False)
        .drop_duplicates("path")
        .set_index("path")
    )
    # The join key is the run path relative to the benchmark directory, so a
    # CSV replayed from a *different* benchmark matches nothing at all. That
    # used to surface three steps later as "no budget welfare to plot", which
    # reads as "you forgot to run the replay" when the replay was in fact run,
    # just against another sweep.
    if not set(runs["path"]) & set(best.index):
        raise SystemExit(
            f"{stage_csv} shares no run with this benchmark directory "
            f"({len(best)} replayed run(s), {len(runs)} here). It was almost "
            "certainly replayed from a different sweep; re-run "
            "analysis.plots.recompute_mid_welfare against this one."
        )
    merged = runs.copy()
    merged["mid_welfare"] = merged["path"].map(best["mid_welfare"])
    # The selected zoning's own measures, so all three panels describe one zoning.
    replayed_columns = (("welfare", "logsum_welfare"), ("cut_edges", "cut_edges"))
    for column, replayed in replayed_columns:
        if replayed in best.columns:
            merged[column] = merged["path"].map(best[replayed]).fillna(merged[column])
    if "stage_name" in best.columns:
        merged["best_stage"] = merged["path"].map(best["stage_name"])

    best_of = merged.groupby("centroid")["mid_welfare"].transform("max")
    merged["budget_pct_of_best"] = 100.0 * merged["mid_welfare"] / best_of

    scored = merged["mid_welfare"].notna()
    if not scored.all():
        print(f"  note: {int((~scored).sum())} run(s) have no replayed zoning")
    if "is_selected" in stages.columns:
        moved = [
            path
            for path in merged.loc[scored, "path"]
            if path in best.index and not bool(best.loc[path, "is_selected"])
        ]
        print(
            f"  best zoning: {int(scored.sum())} run(s) rescored from "
            f"{len(stages)} saved zoning(s); {len(moved)} run(s) beat the zoning "
            "they returned"
        )
    return merged


def append_baseline(runs: pd.DataFrame, baseline_csv: Path | None) -> pd.DataFrame:
    """Add the relaxed-ReCom chain to the comparison as one extra row.

    ``analysis.plots.relaxed_recom_baseline`` writes one row per zoning the
    chain visited, each scored on the same three measures as a benchmarked run.
    Iteration 0 is the ``hints: feasible`` starting point rather than something
    the chain produced, so it is left out; so is any zoning the chain flagged as
    balance-infeasible, since the row is only comparable to rows of solutions.
    Welfare is normalized against the benchmark's own per-centroid best, so the
    row sorts among the strategies without moving that yardstick.
    """
    if baseline_csv is None or not baseline_csv.exists():
        return runs
    chain = pd.read_csv(baseline_csv)
    columns = {"centroid", "iteration", "choice_welfare", "budget_welfare", "cut_edges"}
    if not columns.issubset(chain.columns):
        print(
            f"  note: {baseline_csv.name} lacks "
            f"{sorted(columns - set(chain.columns))}; baseline row omitted"
        )
        return runs
    chain = chain[chain["iteration"] > 0]
    if "feasible" in chain.columns:
        infeasible = int((~chain["feasible"].astype(bool)).sum())
        if infeasible:
            print(f"  note: {infeasible} baseline iteration(s) are not feasible")
            chain = chain[chain["feasible"].astype(bool)]
    known = set(runs["centroid"].unique())
    walked = set(chain["centroid"].unique())
    unknown = sorted(walked - known)
    if unknown:
        print(f"  note: baseline centroid(s) {unknown} are not in the benchmark")
        chain = chain[chain["centroid"].isin(known)]
    # This CSV joins on the centroid name alone -- there is nothing in it that
    # ties it to a benchmark directory. A chain walked for an earlier sweep
    # therefore attaches happily to whichever centroids the two share, and the
    # figure grows a baseline row covering only some of its panels. Missing
    # centroids are the visible symptom, so say so rather than drawing it.
    missing = sorted(known - walked)
    if missing:
        print(
            f"  note: {baseline_csv.name} has no chain for {missing}; those "
            "panels get no baseline row. Re-run analysis.plots."
            "relaxed_recom_baseline against this benchmark to cover them."
        )
    if chain.empty:
        return runs

    if "mid_welfare" not in runs.columns or runs["mid_welfare"].isna().all():
        print(f"  note: {baseline_csv.name} skipped; no budget welfare to rank it against")
        return runs
    best = runs.groupby("centroid")["mid_welfare"].max()
    rows = pd.DataFrame(
        {
            "path": str(baseline_csv.name),
            "strategy": "relaxed_recom",
            "solver": "relaxed_recom",
            "variant": BASELINE_LABEL,
            "centroid": chain["centroid"].to_numpy(),
            "seed": chain["seed"].to_numpy() if "seed" in chain else np.nan,
            "welfare": chain["choice_welfare"].to_numpy(dtype=float),
            "mid_welfare": chain["budget_welfare"].to_numpy(dtype=float),
            "cut_edges": chain["cut_edges"].to_numpy(dtype=float),
            "status": "CHAIN",
        }
    )
    rows["budget_pct_of_best"] = (
        100.0 * rows["mid_welfare"] / rows["centroid"].map(best)
    )
    print(
        f"  baseline: {len(rows)} chain iteration(s) over "
        f"{rows['centroid'].nunique()} centroid configuration(s)"
    )
    return pd.concat([runs, rows], ignore_index=True)


def load_runs(
    benchmark_dir: Path,
    keep_failed: bool = False,
    config_filters: Sequence[tuple[str, str]] = (),
) -> pd.DataFrame:
    """One row per benchmark run, scored on the common choice-utility metric."""
    rows: list[dict[str, Any]] = []
    for path, result in iter_results(benchmark_dir, keep_failed, config_filters):
        config = result["config"]
        run = result["run"]
        limits = config.get("solve_time_limits") or [float("nan")]
        rows.append(
            {
                "path": str(path.parent.relative_to(benchmark_dir)),
                "strategy": config["strategy"],
                "solver": config["solver"],
                "variant": _variant_label(config),
                "centroid": config["centroids_type"],
                "seed": config["seed"],
                "complementary_slackness": config.get(
                    "mid_complementary_slackness", False
                ),
                "time_limit": sum(limits),
                # Exact MNL logsum welfare of the final zoning: the one metric
                # computed identically for every strategy.
                "welfare": (run.get("choice_preassignment_utility") or {}).get(
                    "utility", float("nan")
                ),
                "wall_time": result["total_wall_time"],
                "cut_edges": run["final_cut_edges"],
                "status": result["status"],
                "n_stages": len(run["stages"]),
            }
        )
    if not rows:
        raise SystemExit(f"No result.json files found under {benchmark_dir}")
    runs = pd.DataFrame(rows)
    return runs


def load_iterations(
    benchmark_dir: Path,
    keep_failed: bool = False,
    config_filters: Sequence[tuple[str, str]] = (),
) -> pd.DataFrame:
    """One row per iteration for the iterative strategies.

    ``model`` is the master problem's own utility for the candidate zoning;
    ``actual`` is that same zoning re-scored by the exact oracle. Both are read
    from what the run *recorded*, not recomputed here: the master's objective
    is an internal solver quantity that no replay can reproduce from a saved
    zoning. So these traces sit on whatever market the benchmark ran against,
    which is not necessarily the one ``recompute_mid_welfare`` replays figure 1
    on -- changing a market selector such as ``include_citywide_choice_opt``
    moves figure 1 and leaves these where they were. What "exact"
    means is strategy-specific and the figures have to say so: for SAA it is
    realized welfare averaged over the tie-breaking samples, for
    ``iterative_choice`` the exact logsum, and for ``priced_access`` the exact
    value of the priced bound itself, since that strategy has no recourse LP to
    realize welfare with. All three are the quantity the master is being
    optimistic about, which is what the optimism gap measures. The final stage
    of SAA and ``priced_access`` is a zero-time replay of the selected
    incumbent, so it is flagged rather than treated as a fresh iteration.

    ``seed`` and ``price_scale`` identify the run a trace came from; a sweep
    with replicates holds several runs per centroid, which
    :func:`single_trace_per_centroid` reduces before plotting.
    """
    rows: list[dict[str, Any]] = []
    for path, result in iter_results(benchmark_dir, keep_failed, config_filters):
        config = result["config"]
        if config["strategy"] not in TRACED_STRATEGIES:
            continue
        run = result["run"]
        for stage in run["stages"]:
            meta = stage["metadata"]
            model = meta.get(
                "choice_model_utility", meta.get("saa_master_candidate_upper_bound")
            )
            actual = stage.get("choice_utility", meta.get("saa_welfare"))
            if model is None or actual is None:
                continue
            rows.append(
                {
                    "strategy": config["strategy"],
                    "solver": config["solver"],
                    "centroid": config["centroids_type"],
                    # Full run identity. Seed alone does not separate runs that
                    # differ only in a sweep arm, so it cannot be what reduces
                    # a panel to one trace.
                    "run": str(path.parent.relative_to(benchmark_dir)),
                    "seed": config.get("seed"),
                    "price_scale": config.get("priced_access_price_scale"),
                    # How many tie-breaking draws the SAA recourse averaged, so
                    # the figure can name it instead of hardcoding a count that
                    # goes stale the next time the sweep changes it.
                    "saa_num_seeds": config.get("saa_num_seeds"),
                    "iteration": stage["index"],
                    "model": model,
                    "actual": actual,
                    "upper_bound": meta.get(
                        "choice_best_bound", meta.get("saa_global_upper_bound")
                    ),
                    "wall_time": stage["wall_time"],
                    "is_replay": stage["wall_time"] == 0.0,
                    "is_selected": stage["index"] == run["final_stage_index"],
                }
            )
    iterations = pd.DataFrame(rows)
    if iterations.empty:
        return iterations
    iterations["gap"] = iterations["model"] - iterations["actual"]
    iterations["gap_pct"] = 100.0 * iterations["gap"] / iterations["actual"]
    iterations = iterations.sort_values(
        ["strategy", "solver", "centroid", "iteration"]
    ).reset_index(drop=True)
    # Grouped per run, not per centroid: a replicated sweep interleaves several
    # runs at the same centroid, and summing their stage times together would
    # report a budget no single run ever spent.
    iterations["cum_time"] = iterations.groupby("run", dropna=False)[
        "wall_time"
    ].cumsum()
    return iterations


def single_trace_per_centroid(
    iterations: pd.DataFrame, strategy: str, **filters: Any
) -> pd.DataFrame:
    """Reduce a replicated sweep to one iteration trace per centroid.

    :func:`plot_model_vs_actual` draws a single line per panel, so feeding it a
    sweep of four seeds crossed with three price arms stacks twelve runs into
    one zigzag and reports their pooled iteration count. Keep the arm named in
    ``filters`` and, within it, one run at each centroid. The pick is an
    arbitrary one and the figure says which; it is defensible only because the
    quantity on display -- how far the master's model sits above the exact
    evaluation of its own candidate -- lands in the same place for every
    replicate. Anything that varies across replicates belongs in the comparison
    figure, not here.

    Selection is on the run identity, lowest seed first, rather than on the
    seed alone: two runs of one strategy can share a seed and differ only in a
    sweep arm -- ``saa_multicut``, say -- and filtering to the lowest seed then
    still leaves two traces to interleave.
    """
    data = iterations[iterations["strategy"] == strategy]
    for column, value in filters.items():
        data = data[data[column] == value]
    if data.empty:
        return data
    ordered = data.sort_values(["seed", "run"], kind="stable")
    keep = ordered.groupby("centroid")["run"].first()
    return data[data["run"].isin(set(keep))]


# --------------------------------------------------------------------------- #
# Figure 1: strategy comparison
# --------------------------------------------------------------------------- #


def _measure_panel(
    ax: plt.Axes,
    runs: pd.DataFrame,
    column: str,
    order: Sequence[str],
    centroids: Sequence[str],
    channel: tuple[dict[str, str], dict[str, str]],
    label_fmt: str,
    title: str,
    xlabel: str,
    force_zero: bool = False,
) -> None:
    """Horizontal per-run dot plot of one measure, one row per strategy variant."""
    y_of = {variant: i for i, variant in enumerate(order)}
    offsets = np.linspace(SUBROW_SPAN / 2, -SUBROW_SPAN / 2, len(centroids))
    colors, markers = channel
    for row in range(len(order)):
        ax.axhspan(
            row - 0.5, row + 0.5, color=BAND if row % 2 == 0 else SURFACE, zorder=0, lw=0
        )
    labels: list[tuple[float, str]] = []
    all_values: list[float] = []
    for c_idx, centroid in enumerate(centroids):
        color = colors[centroid]
        marker = markers[centroid]
        subset = runs[runs["centroid"] == centroid]
        for variant, group in subset.groupby("variant"):
            y = y_of[variant] + offsets[c_idx]
            values = group[column].to_numpy(dtype=float)
            many = len(values) > 1
            dense = len(values) >= DISTRIBUTION_MIN_POINTS
            low, high = values.min(), values.max()
            if many:
                ax.plot(
                    [low, high], [y, y], color=color, lw=2.0,
                    alpha=0.30, solid_capstyle="round", zorder=2,
                )
            if dense:
                # A marker per point would paint a solid bar, so the middle half
                # carries the shape and the mean marker carries the location.
                ax.plot(
                    np.percentile(values, DISTRIBUTION_BAND), [y, y], color=color,
                    lw=5.5, alpha=0.35, solid_capstyle="butt", zorder=3,
                )
            else:
                ax.plot(
                    values, np.full(len(values), y), marker=marker, ls="none",
                    ms=5.0 if many else 8.0, mfc=color, mec=SURFACE, mew=1.2,
                    alpha=0.55 if many else 1.0, zorder=3,
                )
            mean = float(values.mean())
            if many:
                ax.plot(
                    [mean], [y], marker=marker, ms=9.0, mfc=color, mec=SURFACE,
                    mew=1.6, zorder=4,
                )
            # Deferred: the labels are drawn as one right-hand column once the
            # panel's extent is known. Anchoring each to its own row's maximum
            # staggers them horizontally, and at four centroids per row the
            # stagger is smaller than the text, so they overprint each other.
            labels.append(
                (y, format(mean, label_fmt) + (f"  (n={len(values)})" if many else ""))
            )
            all_values.extend((low, high))

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9.5, color=INK)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=10)
    recede(ax, axis="x")
    if not all_values:
        # Every run is missing this measure. An empty panel labelled as such is
        # more honest than the autoscaled 0-1 axis matplotlib would draw.
        ax.text(
            0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color=INK_MUTED,
        )
        return
    lo, hi = min(all_values), max(all_values)
    span = max(hi - lo, 1.0)
    ax.set_xlim(0.0 if force_zero else lo - 0.07 * span, hi + 0.42 * span)
    # One right-hand column of values, left-aligned at a common x just past the
    # data, so neighbouring rows separate vertically instead of overprinting.
    for y, text in labels:
        ax.annotate(
            text, (hi + 0.05 * span, y), va="center", ha="left",
            fontsize=8.5, color=INK_2, annotation_clip=False,
        )


def _describe_replicates(runs: pd.DataFrame) -> str:
    """Name the replicate count behind each row, read off the frame itself.

    The caption used to hardcode "8 seeds for Short Bursts", which silently
    became wrong the first time the sweep changed its seed list. Counting is
    per centroid configuration, since that is what a row's repeated points in
    one colour are. A variant whose count differs across centroids reports the
    range, so a partial sweep shows up in the caption instead of being rounded
    into a single confident number.
    """
    counts = (
        runs[runs["variant"] != BASELINE_LABEL]
        .groupby(["variant", "centroid"])
        .size()
        .groupby("variant")
    )
    described: list[str] = []
    for variant, sizes in counts:
        low, high = int(sizes.min()), int(sizes.max())
        if high < 2:
            continue
        described.append(f"{low if low == high else f'{low}-{high}'} for {variant}")
    if not described:
        return "Each row holds a single run per centroid configuration."
    return (
        "Repeated points within a row are replicates — "
        + ", ".join(described)
        + ", per centroid configuration; the large marker and the printed value "
        "are their mean."
    )


def _describe_baseline(runs: pd.DataFrame) -> str:
    """Caption the relaxed-ReCom row, with its real iteration count."""
    chain = runs[runs["variant"] == BASELINE_LABEL]
    if chain.empty:
        return ""
    sizes = chain.groupby("centroid").size()
    low, high = int(sizes.min()), int(sizes.max())
    scored = f"{low:,}" if low == high else f"{low:,}-{high:,}"
    return (
        f"  The {BASELINE_LABEL} row is one unguided chain per centroid configuration that optimizes "
        "nothing and rejects any recombination breaking a balance constraint, so its zonings are ones "
        f"the strategies could have returned too; {scored} feasible iterations are scored, its line "
        "spans them, its bar is the middle half, its marker the mean."
    )


def plot_strategy_comparison(
    runs: pd.DataFrame, output_dir: Path, title_suffix: str = ""
) -> Path:
    """Rank strategies on the common welfare metric, with compactness alongside."""
    if "budget_pct_of_best" not in runs.columns or runs["mid_welfare"].isna().all():
        raise SystemExit(
            "No budget welfare to plot. Budget welfare is the only welfare this "
            "figure reports, so it cannot be drawn without it: run\n"
            "  python -m analysis.plots.recompute_mid_welfare <benchmark-dir> --level base\n"
            "and pass the resulting CSV via --stage-welfare. A CSV replayed from a "
            "different benchmark directory will not join either -- the key is the "
            "run path relative to that directory."
        )
    centroids = sorted(runs["centroid"].unique())
    channel = centroid_channel(centroids)
    # Highest at the top: y=0 is the bottom of the axis, so ascending order
    # puts the best row last. Ranked on ``budget_pct_of_best``, the per-centroid
    # normalized mean, rather than the raw mean: a row that is missing a
    # centroid configuration would otherwise be ranked on a different set of
    # problems than the rows it is being compared against.
    order = (
        runs.groupby("variant")["budget_pct_of_best"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # Budget welfare is the yardstick; the MNL logsum is a surrogate internal to
    # one strategy and is deliberately not shown or ranked on.
    panels: list[tuple[str, str, str]] = [
        ("mid_welfare", "Budget welfare of the best zoning",
         "Discrete budget-set welfare  (higher is better)"),
        ("cut_edges", "Compactness of the same zoning",
         "Cut edges  (lower is more compact)"),
    ]

    # Wrap the captions before laying out the axes: the wrap width scales with
    # the panel count, so a narrower figure spills the same text into more lines
    # and a hardcoded axes top would put the panel titles underneath it.
    lead = "\n".join(
        textwrap.wrap(
            "Each run contributes the one zoning, out of every zoning it saved, with the highest budget "
            "welfare — not the one its own surrogate returned — and both panels describe that same "
            "zoning. Rows are sorted by mean budget welfare. MIP-backed and errored runs are excluded.",
            125 * len(panels) // 2,
        )
    )
    notes = "\n".join(
        textwrap.wrap(
            _describe_replicates(runs)
            + "  Budget welfare is replayed through finite_grid_oracle at base-block "
              "resolution, so every strategy is scored on one budget market."
            + _describe_baseline(runs),
            140 * len(panels) // 2,
        )
    )
    # Lay the figure out in inches and convert to fractions at the end, rather
    # than positioning in fractions of a fixed height. Both the caption block
    # and the panels grow with the sweep -- more centroid configurations means
    # more sub-rows per strategy and a longer replicate sentence -- and a
    # fraction that was right for two centroids silently overlaps at four.
    #
    # Each sub-row carries its own value label, so a strategy row has to be at
    # least as tall as its labels stacked; below that they overprint, which is
    # what a fixed height did once the sweep reached four centroids.
    # A row's sub-rows are SUBROW_SPAN apart in row units, so the gap between
    # two of them in inches is row_height * SUBROW_SPAN / (n - 1); that has to
    # clear one line of label text.
    LABEL_LINE = 0.145  # inches per 8.5pt value label, with leading
    row_height = max(
        0.62, LABEL_LINE * max(1, len(centroids) - 1) / SUBROW_SPAN
    )
    axes_height = len(order) * row_height

    lead_line, notes_line = 0.175, 0.165
    top_text = (
        0.40                                    # suptitle
        + (lead.count("\n") + 1) * lead_line
        + 0.12                                  # gap between lead and notes
        + (notes.count("\n") + 1) * notes_line
        + 0.34                                  # room for the panel titles
    )
    # The centroid legend sits along the bottom, as it does in every other
    # figure here. Top-right put it beside the lead paragraph, which a sweep
    # with more than two centroid configurations is wide enough to collide with.
    bottom_text = 0.52 + 0.52                   # x label, then legend
    height = top_text + axes_height + bottom_text

    axes_top = 1.0 - top_text / height
    lead_y = 1.0 - 0.34 / height
    notes_y = lead_y - ((lead.count("\n") + 1) * lead_line + 0.12) / height

    fig = plt.figure(figsize=(6.4 * len(panels), height), dpi=300)
    grid = fig.add_gridspec(
        1, len(panels), wspace=0.09,
        left=0.205 * 2 / len(panels), right=0.99, top=axes_top,
        bottom=bottom_text / height,
    )
    axes = []
    for index, (column, title, xlabel) in enumerate(panels):
        ax = fig.add_subplot(
            grid[0, index], sharey=axes[0] if axes else None
        )
        _measure_panel(
            ax, runs.dropna(subset=[column]), column, order, centroids, channel,
            ",.0f", f"{chr(ord('A') + index)}.  {title}", xlabel,
        )
        if index:
            plt.setp(ax.get_yticklabels(), visible=False)
        axes.append(ax)

    colors, markers = channel
    handles = [
        Line2D([], [], marker=markers[centroid], ls="none", ms=7.5,
               mfc=colors[centroid], mec=SURFACE, mew=1.2, label=centroid)
        for centroid in centroids
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.10 / height),
        # One legend row per six entries, so a wide sweep wraps instead of
        # running off the figure.
        ncol=min(len(centroids), 6), title="Centroid configuration",
        title_fontsize=9, alignment="left", columnspacing=1.5,
    )

    fig.suptitle(
        "Choice-benchmark strategy comparison"
        + (f" — {title_suffix}" if title_suffix else ""),
        x=0.010, y=1.0 - 0.13 / height, ha="left", fontsize=15, fontweight="bold",
        color=INK,
    )
    fig.text(0.010, lead_y, lead, ha="left", va="top", fontsize=9.5, color=INK_2)
    fig.text(0.010, notes_y, notes, ha="left", va="top", fontsize=9, color=INK_MUTED)
    return save(fig, output_dir / "choice_strategy_comparison.png")


# --------------------------------------------------------------------------- #
# Figures 2-3: model vs actual utility per iteration
# --------------------------------------------------------------------------- #


def plot_model_vs_actual(
    iterations: pd.DataFrame,
    strategy: str,
    output_dir: Path,
    filename: str,
    show_upper_bound: bool,
    subtitle: str,
    ylabel: str = "Choice welfare",
    actual_label: str = "Actual utility (exact evaluation)",
) -> Path | None:
    """Per-iteration model utility against the exact evaluation of the same zoning.

    One line per centroid, so ``iterations`` must already hold a single run per
    centroid; pass a replicated sweep through :func:`single_trace_per_centroid`
    first. ``actual_label`` names what the exact evaluation is, which differs by
    strategy and is the one thing a reader can get wrong here. Returns ``None``
    when the frame holds no iteration of this strategy -- a partially complete
    sweep has strategies with nothing to trace yet.
    """
    data = iterations[iterations["strategy"] == strategy]
    centroids = sorted(data["centroid"].unique())
    if not centroids:
        print(f"  note: no {strategy} iterations to trace; skipping {filename}")
        return None

    # A subtitle longer than two lines has to push the panels down, or it lands
    # on top of the per-panel titles.
    caption = textwrap.wrap(subtitle, 150)
    height = 5.0 + 0.24 * max(0, len(caption) - 2)
    fig, axes = plt.subplots(
        1, len(centroids), figsize=(6.2 * len(centroids), height), dpi=300, squeeze=False
    )
    fig.subplots_adjust(
        left=0.072,
        right=0.985,
        top=1.0 - 1.20 / height,
        bottom=0.975 / height,
        wspace=0.16,
    )

    for c, centroid in enumerate(centroids):
        ax = axes[0][c]
        trace = data[data["centroid"] == centroid]
        live = trace[~trace["is_replay"]]
        if live.empty:
            ax.set_visible(False)
            continue
        x = live["iteration"].to_numpy()
        model = live["model"].to_numpy()
        actual = live["actual"].to_numpy()

        ax.fill_between(
            x, actual, model, color=INK_MUTED, alpha=0.14, lw=0, zorder=1,
            label="Optimism gap",
        )
        if show_upper_bound and live["upper_bound"].notna().all():
            bound = live["upper_bound"].to_numpy()
            ax.plot(
                x, bound, color=SERIES[2], lw=2.0, ls=(0, (1, 1.6)), zorder=2,
                label="Global upper bound",
            )
            ax.annotate(
                "global upper bound", (x[-1], bound[-1]), xytext=(-3, 8),
                textcoords="offset points", ha="right", fontsize=8.0,
                color=SERIES[2], fontweight="bold",
            )
        ax.plot(
            x, model, color=SERIES[0], lw=2.0, marker="o", ms=4.4, mec=SURFACE,
            mew=0.9, zorder=4, label="Model utility (master estimate)",
        )
        ax.plot(
            x, actual, color=SERIES[1], lw=2.0, ls=(0, (5, 2)), marker="s",
            ms=4.4, mec=SURFACE, mew=0.9, zorder=4, label=actual_label,
        )

        selected = trace[trace["is_selected"] & ~trace["is_replay"]]
        if selected.empty:
            flagged = trace[trace["is_selected"]]
            if not flagged.empty:
                # Zero-time replay stage: mark the iteration it selected.
                match = live[np.isclose(live["actual"], flagged["actual"].iloc[0])]
                selected = match.tail(1)
        if not selected.empty:
            ax.plot(
                selected["iteration"], selected["actual"], marker="*", ms=18,
                mfc=SERIES[1], mec=SURFACE, mew=1.3, ls="none", zorder=5,
                label="Selected solution",
            )

        ax.set_title(centroid, loc="left", pad=8)
        ax.set_title(
            f"{len(live)} iterations   |   {fmt_gap(model[-1] - actual[-1], actual[-1])}",
            loc="right", pad=8, fontsize=8.8, color=INK_2, fontweight="normal",
        )
        ax.set_xticks(x[:: max(1, len(x) // 8)])
        ax.margins(x=0.04, y=0.13)
        recede(ax)
        ax.set_xlabel("Iteration")
        if c == 0:
            ax.set_ylabel(ylabel)

    # From the first panel that actually drew: the leading centroid can be a
    # replay-only trace whose axes were hidden, and a hidden panel has no
    # handles, which would silently drop the legend from the whole figure.
    handles, labels = next(
        (
            (h, l)
            for h, l in (ax.get_legend_handles_labels() for ax in axes[0])
            if h
        ),
        ([], []),
    )
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.060 / height),
        ncol=len(labels), columnspacing=1.8, handlelength=2.4, fontsize=9.5,
    )
    fig.suptitle(
        f"{STRATEGY_LABELS.get(strategy, strategy)}: model utility vs. actual utility per iteration",
        x=0.010, y=1.0 - 0.125 / height, ha="left", fontsize=15, fontweight="bold",
        color=INK,
    )
    fig.text(
        0.010, 1.0 - 0.440 / height, "\n".join(caption),
        ha="left", va="top", fontsize=9.5, color=INK_2,
    )
    return save(fig, output_dir / filename)


def _saa_sample_count(iterations: pd.DataFrame) -> str:
    """How many tie-breaking draws the plotted SAA runs averaged over.

    Reported from the run configs rather than written into the caption, which
    is how the figure came to claim 15 samples for a sweep that drew 10. A
    sweep mixing sample counts says so instead of picking one.
    """
    counts = iterations.loc[iterations["strategy"] == "saa", "saa_num_seeds"].dropna()
    if counts.empty:
        return "the configured number of"
    distinct = sorted({int(count) for count in counts})
    return " and ".join(str(count) for count in distinct)


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot choice-benchmark strategy comparison and model-vs-actual utility traces."
    )
    parser.add_argument(
        "benchmark_dir",
        nargs="?",
        default=DEFAULT_BENCHMARK_DIR,
        help=f"Benchmark directory containing per-run result.json files (default: {DEFAULT_BENCHMARK_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for the generated PNGs (defaults to analysis/plots/output)",
    )
    parser.add_argument(
        "--title", default="", help="Optional title suffix for the comparison figure"
    )
    parser.add_argument(
        "--stage-welfare",
        default=None,
        help=(
            "CSV of per-stage replayed welfare from analysis.plots.recompute_mid_welfare, "
            "which decides each run's best zoning "
            "(default: <output-dir>/mid_welfare_base.csv when present)"
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "CSV of the relaxed-ReCom baseline chain from "
            "analysis.plots.relaxed_recom_baseline (default: "
            "<output-dir>/relaxed_recom_baseline.csv when present)"
        ),
    )
    parser.add_argument(
        "--price-scale",
        type=float,
        default=1.0,
        help=(
            "Congestion price multiplier of the priced_access arm to trace in "
            "priced_access_model_vs_actual.png (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help=(
            "Include runs the benchmark marked ERROR. They are excluded by default: "
            "a run that raised part-way can still have saved a zoning, which would "
            "otherwise count as an ordinary replicate of its strategy"
        ),
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Keep only runs whose config sets KEY to VALUE; repeatable. Use this "
            "for a swept arm the strategy axis does not split on, which would "
            "otherwise average into one row (e.g. --filter saa_num_seeds=10 to "
            "hold direct_samples to its 10-scenario arm)"
        ),
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write the tidy run-level and iteration-level tables next to the plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parent / "output"
    )
    style()

    config_filters = []
    for term in args.filters:
        key, sep, value = term.partition("=")
        if not sep or not key:
            raise SystemExit(f"--filter expects KEY=VALUE, got {term!r}")
        config_filters.append((key, value))

    runs = load_runs(
        benchmark_dir, keep_failed=args.keep_failed, config_filters=config_filters
    )
    runs = attach_best_stage(
        runs,
        Path(args.stage_welfare)
        if args.stage_welfare
        else output_dir / "mid_welfare_base.csv",
    )
    iterations = load_iterations(
        benchmark_dir, keep_failed=args.keep_failed, config_filters=config_filters
    )
    compared = append_baseline(
        runs,
        Path(args.baseline)
        if args.baseline
        else output_dir / "relaxed_recom_baseline.csv",
    )

    outputs = [plot_strategy_comparison(compared, output_dir, args.title)]
    if not iterations.empty:
        outputs.append(
            plot_model_vs_actual(
                single_trace_per_centroid(iterations, "iterative_choice"),
                strategy="iterative_choice",
                output_dir=output_dir,
                filename="choice_model_vs_actual.png",
                show_upper_bound=False,
                subtitle=(
                    "The master problem maximizes a cutting-plane approximation of the logsum; each candidate "
                    "zoning it proposes is then scored exactly. The shaded band is how optimistic the model was "
                    "about its own candidate.  One run per panel: lowest seed present."
                ),
            )
        )
        outputs.append(
            plot_model_vs_actual(
                single_trace_per_centroid(iterations, "saa"),
                strategy="saa",
                output_dir=output_dir,
                filename="saa_model_vs_actual.png",
                show_upper_bound=True,
                subtitle=(
                    "The master problem maximizes an outer approximation of sample-average expected welfare; the "
                    "recourse step then evaluates realized welfare for the same zoning over "
                    f"{_saa_sample_count(iterations)} tie-breaking samples. This is expected stable-matching "
                    "welfare, not the MNL logsum of figure 2 — the two are different quantities on different "
                    "scales.  One run per panel: lowest seed present."
                ),
                ylabel="Expected stable-matching welfare",
                actual_label="Realized welfare (sample average)",
            )
        )
        outputs.append(
            plot_model_vs_actual(
                single_trace_per_centroid(
                    iterations, "priced_access", price_scale=args.price_scale
                ),
                strategy="priced_access",
                output_dir=output_dir,
                filename="priced_access_model_vs_actual.png",
                show_upper_bound=True,
                subtitle=(
                    "The master problem maximizes an outer approximation of the priced-access welfare bound; "
                    "separation then evaluates that bound exactly, in closed form, for the same zoning. Orange "
                    "is therefore the surrogate's own value, not realized welfare, so the gap closing to zero "
                    "certifies an optimum of the bound and not of welfare. Both series omit the constant "
                    f"capacity term. One run per panel: price x{args.price_scale:g}, lowest seed present."
                ),
                ylabel="Priced access value",
                actual_label="Exact bound at the same zoning",
            )
        )

    # A partial sweep can have strategies with no trace yet.
    outputs = [figure for figure in outputs if figure is not None]

    if args.csv:
        runs.to_csv(output_dir / "choice_benchmark_runs.csv", index=False)
        iterations.to_csv(output_dir / "choice_benchmark_iterations.csv", index=False)
        print(f"wrote {output_dir / 'choice_benchmark_runs.csv'}")
        print(f"wrote {output_dir / 'choice_benchmark_iterations.csv'}")

    print(f"\nGenerated {len(outputs)} figure(s) in {output_dir}")


if __name__ == "__main__":
    main()

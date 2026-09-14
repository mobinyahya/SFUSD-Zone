"""Plot feasible cut-edge objectives from benchmark logs.

Usage: uv run python -m benchmark.plot_edges benchmark/configs/benchmark_edges.yaml

Like the success heatmap, figures group runs by final graph level, zone count,
and time budget. Each line is one run, never an average across changing sets of
feasible runs in individual mode. The default summarizes a fixed set of feasible
trajectories with medians and interquartile bands. Weighted recursive runs join
their stage logs on cumulative solve time; unweighted recursive runs are excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    LogLocator,
    MaxNLocator,
    NullFormatter,
    NullLocator,
    StrMethodFormatter,
)
import numpy as np
import pandas as pd
import yaml

from benchmark.results import discover_run_dirs
from benchmark.solver_logs import parse_run_dir


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "benchmark_edges.yaml"
# Same solver ordering and names as the success-rate heatmap: these labels and
# their order must match ``SOLVER_DISPLAY_ORDER`` in
# analysis/plots/plot_benchmark_success_heatmap.py, which
# test_plot_benchmark_success_heatmap.py asserts against this dict. The sweep
# config names the strategy ``recursive`` and the method keys keep that name,
# but both figures label it "Multi-Level" for the reader.
# Medians run within a few percent of each other late in a solve, so every
# method carries its own hue AND marker, and the recursive strategies are
# dashed: no two curves are told apart by colour alone.
SOLVERS = {
    "recom": ("Recom", "#8c564b"),
    "relaxed_recom": ("Relaxed Recom", "#7570b3"),
    "short_bursts": ("Short Bursts", "#e69f00"),
    "adaptive_short_bursts": ("Adaptive Short Bursts", "#009e73"),
    "mip": ("MIP", "#cc79a7"),
    "cp_bool": ("CP", "#0072b2"),
    "cp_int": ("CP (Int)", "#56b4e9"),
}
METHODS = {
    **SOLVERS,
    "recursive_mip": ("MIP (Multi-Level)", "#a11d33"),
    "recursive_cp_bool": ("CP (Multi-Level)", "#173f66"),
    "recursive_cp_int": ("CP (Int, Multi-Level)", "#4d4d4d"),
}
MARKERS = dict(zip(METHODS, ["x", "+", "v", "P", "s", "o", "*", "D", "^", "h"]))
# Stroke-only markers have no face to outline, and matplotlib warns if asked.
UNFILLED_MARKERS = {"x", "+"}
DASHES = {
    method: ((0, (5.5, 1.8)) if method.startswith("recursive_") else "solid")
    for method in METHODS
}
# How the vertical axis is scaled, default first: logarithmic over the full
# range, linear and framed on the medians, or the plain zero-baseline linear
# range. ``log`` is the default because it crops nothing, which matters most
# where one method lands several times worse than the rest.
Y_FRAMES = ("log", "zoom", "zero")
# Budgets and solver logs are both in seconds, but a 600-1800 second axis reads
# better in minutes. Only the drawn axis is converted: events, aggregates and
# run CSVs keep ``elapsed_seconds`` in its source unit, as the objective
# columns keep metres.
SECONDS_PER_MINUTE = 60.0


def minutes(seconds):
    """Seconds to minutes, for axis coordinates only. Accepts scalars/arrays."""
    return np.asarray(seconds, dtype=float) / SECONDS_PER_MINUTE


Y_FRAME_NOTES = {
    "log": "Vertical axis is logarithmic over the full range of every drawn value.",
    "zoom": "Vertical axis is scaled to the medians and does not start at zero; first-feasible values and parts of the bands fall outside the panel.",
    "zero": "Vertical axis is linear from zero over the full range of every drawn value.",
}


def marker_phases(methods: list[str]) -> dict[str, float]:
    """Spread markers within one sampling step so overlaid curves stay legible.

    Medians converge to within a line width of each other, so identical sample
    times would stack every marker into one unreadable clump per instant.
    """
    count = max(len(methods), 1)
    return {
        method: (index - (count - 1) / 2) / count * 0.85
        for index, method in enumerate(sorted(methods))
    }


def band_alpha(methods, *, zoomed: bool) -> float:
    """Keep stacked IQR bands as context, not as a wash over the medians.

    A zoomed panel crops the bands to their widest part, so each one covers
    most of the panel and several together hide the curves they belong to.
    """
    alpha = 0.26 / max(len(methods), 1) if zoomed else 0.12
    return float(np.clip(alpha, 0.045, 0.12))


def draw_summary(
    ax, summary: pd.DataFrame, method: str, *, phase: float = 0.0, alpha: float = 0.12
) -> None:
    """Render the unchanged IQR and median, with sparse identity markers."""
    color = METHODS[method][1]
    ax.fill_between(
        minutes(summary["elapsed_seconds"]),
        summary["q25"],
        summary["q75"],
        step="post",
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=1,
    )
    ax.step(
        minutes(summary["elapsed_seconds"]),
        summary["incumbent"],
        where="post",
        color=color,
        linewidth=2.1,
        linestyle=DASHES[method],
        solid_capstyle="butt",
        zorder=3,
    )
    finite = summary.dropna(subset=["incumbent"])
    if not finite.empty:
        # Select actual step values near evenly spaced times, rather than
        # spacing by event count (which clusters markers early in CP logs).
        # Sample strictly inside the span so no method marks the shared
        # endpoint, and offset by ``phase`` to separate coincident curves.
        first = finite["elapsed_seconds"].iloc[0]
        last = finite["elapsed_seconds"].iloc[-1]
        step = (last - first) / 6
        targets = first + step * (np.arange(6) + 0.5 + phase)
        indices = np.searchsorted(finite["elapsed_seconds"], targets, side="right") - 1
        points = finite.iloc[np.unique(np.clip(indices, 0, len(finite) - 1))]
        halo = (
            {}
            if MARKERS[method] in UNFILLED_MARKERS
            else {"edgecolors": "white", "linewidths": 0.5}
        )
        ax.scatter(
            minutes(points["elapsed_seconds"]),
            points["incumbent"],
            marker=MARKERS[method],
            color=color,
            s=34,
            zorder=4,
            **halo,
        )


def frame_ylim(ax, frames: list[pd.DataFrame], *, mode: str) -> None:
    """Scale the vertical axis so the medians, not the early spikes, fill it.

    ``zero`` keeps the zero-baseline linear range over every drawn value, so a
    first feasible solution an order of magnitude above the rest flattens the
    medians into the bottom of the panel. ``zoom`` instead frames the range the
    medians actually occupy and lets earlier values and the outer parts of the
    bands clip. ``log``, the default, keeps everything visible and compresses
    the top.
    """
    if mode not in Y_FRAMES:
        raise ValueError(f"y_frame must be one of {Y_FRAMES}")
    medians = [frame["incumbent"].dropna() for frame in frames]
    medians = [values for values in medians if not values.empty]
    if mode == "zoom" and medians:
        # The worst *final* median has to stay in frame: it is a real result,
        # unlike the transient first-feasible values that precede it.
        best = min(values.min() for values in medians)
        worst = max(values.iloc[-1] for values in medians)
        span = worst - best or max(abs(worst), 1.0) * 0.05
        ax.set_ylim(max(0.0, best - 0.15 * span), worst + 0.20 * span)
        return
    if mode == "log":
        columns = [
            frame[column]
            for frame in frames
            for column in ("incumbent", "q25", "q75")
            if column in frame
        ]
        values = pd.concat(columns) if columns else pd.Series(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        ax.set_yscale("log")
        if not values.empty:
            ax.set_ylim(values.min() / 1.2, values.max() * 1.2)
        # Powers of ten alone leave a 40-160 km panel with one tick.
        ax.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0), numticks=14)
        )
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        return
    ax.margins(y=0.12)
    ax.autoscale_view()
    ax.set_ylim(0, ax.get_ylim()[1])


def late_window(summary: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    """Clip a step series while preserving its value at the window start."""
    earlier = summary[summary["elapsed_seconds"] <= start].tail(1).copy()
    earlier["elapsed_seconds"] = start
    return pd.concat(
        [
            earlier,
            summary[summary["elapsed_seconds"].between(start, end, inclusive="right")],
        ]
    )


def spread_positions(anchors: list[float], gap: float) -> list[float]:
    """Least-movement separation of sorted anchors by ``gap``, kept in [0, 1].

    One upward pass separates the stack, then a downward pass pulls it back
    inside the axes; labels denser than the axes can hold are spaced evenly.
    """
    if not anchors:
        return []
    if gap * (len(anchors) - 1) >= 1.0:
        return list(np.linspace(0.0, 1.0, len(anchors)))
    positions = list(anchors)
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + gap)
    overflow = positions[-1] - 1.0
    if overflow > 0:
        positions = [position - overflow for position in positions]
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - gap)
    shortfall = -positions[0]
    if shortfall > 0:
        positions = [position + shortfall for position in positions]
    return positions


def label_endpoints(ax, series: dict[str, pd.DataFrame], unit: str) -> None:
    """Name each median at the right edge, spreading labels to avoid overlap."""
    entries = []
    low, high = ax.get_ylim()
    if ax.get_yscale() == "log":
        low, high = math.log10(low), math.log10(high)
    for method, summary in series.items():
        finite = summary.dropna(subset=["incumbent"])
        if not finite.empty:
            row = finite.iloc[-1]
            value = row["incumbent"]
            scaled = math.log10(value) if ax.get_yscale() == "log" else value
            entries.append(((scaled - low) / (high - low), method, row))
    entries.sort(key=lambda item: (item[0], item[1]))
    positions = spread_positions([fraction for fraction, _, _ in entries], 0.072)
    for (_, method, row), position in zip(entries, positions):
        label, color = METHODS[method]
        ax.annotate(
            f"{label}  {row['incumbent']:.1f}{unit}",
            xy=(minutes(row["elapsed_seconds"]), row["incumbent"]),
            xycoords="data",
            xytext=(1.035, position),
            textcoords="axes fraction",
            color=color,
            fontsize=9.5,
            va="center",
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.8,
                "shrinkB": 2,
            },
        )


def legend_handles(methods) -> list[Line2D]:
    """One entry per drawn method, in the canonical ordering."""
    return [
        Line2D(
            [],
            [],
            color=color,
            label=label,
            linewidth=2.2,
            linestyle=DASHES[method],
            marker=MARKERS[method],
            markersize=6,
        )
        for method, (label, color) in METHODS.items()
        if method in methods
    ]


def render_zone_detail(
    series: dict[str, pd.DataFrame],
    *,
    level: str,
    zones: int,
    budget: float,
    output_dir: Path,
    weighted: bool,
    length_unit: str,
    skip_initial: int,
    dpi: int,
    y_frame: str = "log",
) -> Path:
    """Full range beside a second-half zoom, one zone count per figure."""
    end = max(budget, max(frame["elapsed_seconds"].max() for frame in series.values()))
    late = {
        method: late_window(frame, budget / 2, end) for method, frame in series.items()
    }
    phases = marker_phases(list(series))
    fig, axes = plt.subplots(
        1, 2, figsize=(15.5, 6.0), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    zoom_title = (
        "Second half · logarithmic scale"
        if y_frame == "log"
        else "Second half · expanded vertical scale"
    )
    alphas = (
        band_alpha(series, zoomed=False),
        band_alpha(series, zoomed=y_frame == "zoom"),
    )
    for ax, frames, title, alpha in zip(
        axes, (series, late), ("Full run", zoom_title), alphas
    ):
        for method, summary in frames.items():
            draw_summary(ax, summary, method, phase=phases[method], alpha=alpha)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=15)
        ax.grid(axis="y", color="#e6eaf0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd5e1")
        ax.tick_params(labelsize=10, colors="#475569", length=3)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.set_xlabel("Cumulative solve time (minutes)", fontsize=11, labelpad=10)
    axes[0].set_xlim(0, minutes(end * 1.01))
    # The left panel always shows the full range, so nothing the right panel
    # crops out of frame goes unreported.
    frame_ylim(
        axes[0], list(series.values()), mode="log" if y_frame == "log" else "zero"
    )
    axes[0].set_ylabel(
        f"Cut length ({length_unit})" if weighted else "Cut edges", fontsize=12
    )
    axes[1].set_xlim(minutes(budget / 2), minutes(end * 1.01))
    frame_ylim(axes[1], list(late.values()), mode=y_frame)
    label_endpoints(axes[1], late, f" {length_unit}" if weighted else "")
    handles = legend_handles(series)
    fig.text(
        0.065,
        0.97,
        f"{level} · {zones} zones · {minutes(budget):g}-minute budget",
        fontsize=19,
        fontweight="bold",
        va="top",
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.058, 0.90),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.5,
    )
    fig.text(
        0.065,
        0.025,
        f"Band: middle 50% of available runs. Median starts once all retained runs are ready. First {skip_initial} improvement omitted.\n"
        + (
            "Both panels use a logarithmic vertical scale; multi-level curves use stage feasibility."
            if y_frame == "log"
            else "The right panel is scaled to the medians, so early values and parts of the bands fall outside it."
        ),
        fontsize=9,
        color="#64748b",
    )
    fig.subplots_adjust(left=0.065, right=0.79, top=0.73, bottom=0.17, wspace=0.30)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "weighted_boundary" if weighted else "cut_edges"
    suffix = "" if y_frame == "log" else f"_{y_frame}"
    path = output_dir / f"{prefix}_{level}_{zones}zones_tl_{budget:g}s{suffix}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def recursive_events(rows: list[dict], stages: list[dict]) -> list[dict]:
    """Join stage-local logs using recorded durations, never budget allocations.

    Stage bounds are retained only as audit fields: they are not global bounds.
    Missing non-skipped logs or invalid timing makes the run unplottable.
    """
    if not stages:
        raise ValueError("missing recursive stage metadata")
    joined = []
    offset = 0.0
    for stage in sorted(stages, key=lambda item: item["index"]):
        log_path = stage.get("metadata", {}).get("solver_log_path")
        stage_rows = [
            row
            for row in rows
            if (
                row.get("log_file") == Path(log_path).name
                if log_path
                else row.get("level") == stage["level"]
                and row.get("log_index") == stage["index"]
            )
        ]
        if not stage_rows and stage.get("status") != "SKIPPED":
            raise ValueError("missing recursive stage log")
        duration = stage.get("wall_time")
        if (
            duration is None
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise ValueError("missing or invalid recursive stage duration")
        duration = float(duration)
        for row in stage_rows:
            elapsed = row.get("elapsed_seconds")
            if (
                elapsed is None
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0
            ):
                continue
            # Ignore root-relaxation phase durations when measuring elapsed time.
            if row.get("record") == "root_relaxation":
                continue
            elapsed = float(elapsed)
            duration = max(duration, elapsed)
            joined.append(
                {
                    **row,
                    "stage_level": stage["level"],
                    "stage_index": stage["index"],
                    "stage_elapsed_seconds": elapsed,
                    "stage_start_seconds": offset,
                    "elapsed_seconds": offset + elapsed,
                    "stage_bound": row.get("bound"),
                    "bound": None,
                }
            )
        joined.append(
            {
                "record": "stage_end",
                "stage_level": stage["level"],
                "stage_index": stage["index"],
                "stage_start_seconds": offset,
                "stage_elapsed_seconds": duration,
                "elapsed_seconds": offset + duration,
            }
        )
        offset += duration
    return joined


def benchmark_root(source: str | Path) -> Path:
    """Resolve a directory, summary.csv, or sweep YAML without loading data."""
    path = Path(source).expanduser().resolve()
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open() as handle:
            config = yaml.safe_load(handle)
        # The benchmark runner resolves execution.output_dir against the cwd.
        path = Path(config["execution"]["output_dir"]).expanduser().resolve()
    elif path.suffix.lower() == ".csv":
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"Benchmark output directory does not exist: {path}")
    return path


def load_trajectories(
    root: Path,
    *,
    centroids: list[str] | None = None,
    seeds: list[int] | None = None,
    levels: list[str] | None = None,
    time_limits: list[float] | None = None,
    solvers: list[str] | None = None,
    weighted: bool = False,
    single_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    """Load historical run configs and raw events, including unsuccessful runs.

    Manifests, rather than regenerated task hashes, identify the saved results.
    This permits plotting copied results without the original district data.
    """
    events, runs = [], []
    skipped = Counter()
    for directory in discover_run_dirs(str(root)):
        run_dir = Path(directory)
        try:
            manifest = json.loads((run_dir / "benchmark_manifest.json").read_text())
            config = manifest.get("config")
            if not config:
                config = json.loads((run_dir / "result.json").read_text())["config"]
        except (OSError, ValueError, KeyError) as exc:
            skipped["unreadable config"] += 1
            print(f"SKIP {run_dir}: {exc}")
            continue
        strategy = config.get("strategy", "single")
        if strategy != "single" and not (
            strategy == "recursive" and weighted and not single_only
        ):
            skipped[f"strategy={strategy}"] += 1
            continue
        solver = config.get("solver")
        if solver not in SOLVERS:
            skipped["unsupported solver"] += 1
            continue
        if strategy == "recursive" and solver not in {"cp_bool", "cp_int", "mip"}:
            skipped["unsupported recursive solver"] += 1
            continue
        if strategy == "recursive" and float(config.get("looseness", 1.0)) != 1.0:
            skipped["recursive constraints are relaxed"] += 1
            continue
        if bool(config.get("weight_edges", False)) != weighted or config.get(
            "secondary_objective", False
        ):
            skipped[
                "objective is not weighted boundary length"
                if weighted
                else "objective is not plain cut edges"
            ] += 1
            continue
        target = config["levels"][-1]
        centroid = str(config["centroids_type"])
        seed = int(config["seed"])
        limits = config["solve_time_limits"]
        budget = (
            sum(
                float(limits[min(i, len(limits) - 1)])
                for i in range(len(config["levels"]))
            )
            if strategy == "recursive"
            else float(limits[-1])
        )
        if any(
            (
                centroids is not None and centroid not in centroids,
                seeds is not None and seed not in seeds,
                levels is not None and target not in levels,
                time_limits is not None and budget not in time_limits,
                solvers is not None and solver not in solvers,
            )
        ):
            skipped["filtered"] += 1
            continue
        zone_match = re.match(r"(\d+)-zone", centroid)
        if zone_match is None:
            skipped["unknown zone count"] += 1
            continue
        identity = {
            "run_id": str(run_dir.relative_to(root)),
            "task_id": manifest.get("task_id"),
            "solver": solver,
            "strategy": strategy,
            "method": f"recursive_{solver}" if strategy == "recursive" else solver,
            "level": target,
            "centroids_type": centroid,
            "seed": seed,
            "num_zones": int(zone_match[1]),
            "time_limit": budget,
            "objective_unit": "meter" if weighted else "cut_edges",
        }
        rows = [
            row for row in parse_run_dir(str(run_dir)) if row.get("solver") == solver
        ]
        if strategy == "recursive":
            try:
                rows = recursive_events(rows, manifest.get("stages", []))
            except ValueError as exc:
                skipped[str(exc)] += 1
                continue
        else:
            rows = [row for row in rows if row.get("level") == target]
        # Never silently join restarts / enumerated solves on a resetting clock.
        if strategy == "single" and len({row["log_file"] for row in rows}) > 1:
            skipped["multiple logs at target level"] += 1
            continue
        trajectory = prepare_trajectory(pd.DataFrame(rows))
        runs.append(
            {
                **identity,
                "status": manifest.get("status"),
                "has_log_events": bool(rows),
                "has_feasible_objective": trajectory["incumbent"].notna().any(),
                "has_bound": trajectory["bound"].notna().any(),
                "last_log_seconds": (
                    trajectory["elapsed_seconds"].max()
                    if not trajectory.empty
                    else np.nan
                ),
            }
        )
        for row in rows:
            events.append({**row, **identity})
    return pd.DataFrame(events), pd.DataFrame(runs), skipped


def prepare_trajectory(events: pd.DataFrame, *, skip_initial: int = 0) -> pd.DataFrame:
    """Step-function knots: no infeasible objectives or future backfilling.

    Keep the last observation at tied (rounded) timestamps, carry values only
    forward, and extend to the last logged time, not an invented budget endpoint.
    Gurobi's root-relaxation duration is local to that phase, not solve elapsed
    time, so it cannot be placed on the common time axis. ``skip_initial`` hides
    that many distinct feasible improvements, not repeated node/bound/summary
    records. Count improvements before collapsing rounded timestamps.
    """
    if skip_initial < 0:
        raise ValueError("skip_initial must be nonnegative")
    columns = ["elapsed_seconds", "incumbent", "bound"]
    if events.empty:
        return pd.DataFrame(columns=columns, dtype=float)
    frame = events.copy()
    if "record" in frame:
        frame = frame[frame["record"] != "root_relaxation"].copy()
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[columns] = frame[columns].replace([np.inf, -np.inf], np.nan)
    if "feasible" in frame:
        frame.loc[frame["feasible"] == False, "incumbent"] = np.nan  # noqa: E712
    frame = frame.loc[frame["elapsed_seconds"].ge(0), columns]
    frame = frame.sort_values("elapsed_seconds", kind="stable")
    # Logs expose the best incumbent / bound, but printed rounding and tied
    # records can repeat them. Retain monotonic best-so-far series per run.
    frame["incumbent"] = frame["incumbent"].cummin().ffill()
    frame["bound"] = frame["bound"].cummax().ffill()
    improvements = frame["incumbent"].notna() & frame["incumbent"].ne(
        frame["incumbent"].shift()
    )
    frame.loc[improvements.cumsum() <= skip_initial, "incumbent"] = np.nan
    return frame.drop_duplicates("elapsed_seconds", keep="last").reset_index(drop=True)


def _level_key(level: str) -> tuple[str, int]:
    unit, _, depth = level.rpartition("_")
    return unit, -int(depth)


def visible_trajectories(
    events: pd.DataFrame, runs: pd.DataFrame, *, skip_initial: int = 1
) -> dict[str, pd.DataFrame]:
    """Exclude unsuccessful runs entirely, including their bound trajectories."""
    if events.empty or runs.empty:
        return {}
    feasible_ids = set(runs.loc[runs["has_feasible_objective"], "run_id"])
    traces = {}
    for run_id, group in events[events["run_id"].isin(feasible_ids)].groupby(
        "run_id", sort=False
    ):
        trace = prepare_trajectory(group, skip_initial=skip_initial)
        # A short successful run can have all its improvements trimmed. Its
        # bounds alone should not reintroduce it into a feasible-objective plot.
        if trace["incumbent"].notna().any():
            traces[run_id] = trace
    return traces


def summarize_traces(traces: list[pd.DataFrame], *, horizon: float = 0) -> pd.DataFrame:
    """Pointwise median/IQR of best logged values, never backfill incumbents.

    Show quartiles of available runs immediately, but start the median only
    when every run has a retained feasible value. Its cohort is then fixed,
    preventing upward median jumps as new runs join.
    Hold final values through the common horizon: a completed run's incumbent
    remains valid, and different stop times should not remove runs.
    """
    horizon = max(horizon, max(trace["elapsed_seconds"].iloc[-1] for trace in traces))
    times = np.unique(
        np.concatenate([trace["elapsed_seconds"] for trace in traces] + [[horizon]])
    )
    start = max(
        trace.loc[trace["incumbent"].notna(), "elapsed_seconds"].iloc[0]
        for trace in traces
    )
    objectives = []
    for trace in traces:
        elapsed = trace["elapsed_seconds"].to_numpy()
        indices = np.searchsorted(elapsed, times, side="right") - 1
        active = indices >= 0
        objective = np.where(
            active, trace["incumbent"].to_numpy()[np.maximum(indices, 0)], np.nan
        )
        objectives.append(objective)
    objectives = pd.DataFrame(np.array(objectives).T)
    result = pd.DataFrame({"elapsed_seconds": times, "n": objectives.count(axis=1)})
    result["incumbent"] = objectives.median(axis=1)
    result.loc[times < start, "incumbent"] = np.nan
    result["q25"] = objectives.quantile(0.25, axis=1)
    result["q75"] = objectives.quantile(0.75, axis=1)
    return result


def aggregate_trajectories(
    traces: dict[str, pd.DataFrame], runs: pd.DataFrame
) -> pd.DataFrame:
    """Tidy aggregate export with contributor counts at every event time."""
    frames = []
    keys = ["time_limit", "level", "num_zones", "method"]
    for identity, group in runs[runs["run_id"].isin(traces)].groupby(keys, sort=True):
        frame = summarize_traces(
            [traces[run_id] for run_id in group["run_id"]], horizon=identity[0]
        )
        frames.append(
            frame.assign(
                **dict(zip(keys, identity)),
                solver=group["solver"].iloc[0],
                strategy=group["strategy"].iloc[0],
                objective_unit=group["objective_unit"].iloc[0],
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def render_figures(
    events: pd.DataFrame,
    runs: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int = 180,
    skip_initial: int = 1,
    aggregate: str = "median",
    weighted: bool = False,
    length_unit: str = "km",
    summaries: pd.DataFrame | None = None,
    y_frame: str = "log",
) -> list[Path]:
    """Separate figures by budget AND level, with compact zone-count panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if length_unit not in {"m", "km"}:
        raise ValueError("length_unit must be m or km")
    if y_frame not in Y_FRAMES:
        raise ValueError(f"y_frame must be one of {Y_FRAMES}")
    divisor = 1000.0 if weighted and length_unit == "km" else 1.0
    traces = {}
    if summaries is None or aggregate == "individual":
        traces = visible_trajectories(events, runs, skip_initial=skip_initial)
        visible_runs = runs[runs["run_id"].isin(traces)]
        if aggregate == "median":
            summaries = aggregate_trajectories(traces, runs)
    else:
        visible_runs = runs[runs["plotted"]]
    if visible_runs.empty:
        return []
    lookup = {}
    if aggregate == "median":
        summaries = summaries.copy()
        summaries[["incumbent", "q25", "q75"]] /= divisor
        lookup = dict(
            tuple(
                summaries.groupby(
                    ["time_limit", "level", "num_zones", "method"], sort=False
                )
            )
        )
    else:
        for trace in traces.values():
            trace["incumbent"] /= divisor
    paths = []
    groups = visible_runs.groupby(["time_limit", "level"], sort=False)
    for budget, level in sorted(
        groups.groups, key=lambda key: (key[0], _level_key(key[1]))
    ):
        page_runs = groups.get_group((budget, level))
        zone_counts = sorted(page_runs["num_zones"].unique())
        ncols = min(2, len(zone_counts))
        nrows = (len(zone_counts) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            squeeze=False,
            figsize=(max(9.0, 6.0 * ncols), 3.8 * nrows + 2.0),
            facecolor="white",
        )
        for ax, zones in zip(axes.flat, zone_counts):
            panel = page_runs[page_runs["num_zones"] == zones]
            ax.set_facecolor("white")
            ax.grid(axis="y", color="#e8ecf1", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_title(
                f"{zones} zones",
                loc="left",
                fontsize=13,
                fontweight="semibold",
                color="#182536",
                pad=13,
            )
            ax.text(
                1,
                1.035,
                f"{len(panel)} runs",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#788392",
            )
            panel_frames = []
            methods = list(panel["method"].unique())
            phases = marker_phases(methods)
            fill_alpha = band_alpha(methods, zoomed=y_frame == "zoom")
            for method, solver_runs in panel.groupby("method", sort=True):
                color = METHODS[method][1]
                alpha = max(0.24, min(0.9, 1.5 / np.sqrt(len(solver_runs))))
                if aggregate == "median":
                    summary = lookup[(budget, level, zones, method)]
                    draw_summary(
                        ax, summary, method, phase=phases[method], alpha=fill_alpha
                    )
                    panel_frames.append(summary)
                    continue
                for run in solver_runs.itertuples():
                    values = traces[run.run_id].dropna(subset=["incumbent"])
                    # Only changes and the endpoint are needed for a step curve.
                    keep = values["incumbent"].ne(values["incumbent"].shift())
                    keep.iloc[-1] = True
                    values = values[keep]
                    panel_frames.append(values)
                    ax.step(
                        minutes(values["elapsed_seconds"]),
                        values["incumbent"],
                        where="post",
                        color=color,
                        linewidth=1.65,
                        linestyle=DASHES[method],
                        alpha=alpha,
                        zorder=3,
                    )
                    endpoints = values.iloc[[0, -1]].drop_duplicates()
                    ax.scatter(
                        minutes(endpoints["elapsed_seconds"]),
                        endpoints["incumbent"],
                        color=color,
                        s=12,
                        alpha=alpha,
                        zorder=4,
                        edgecolors="none",
                    )
            endpoint = panel["last_log_seconds"].max()
            ax.set_xlim(0, minutes(max(float(budget), endpoint) * 1.015))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=6, integer=not weighted or length_unit == "m")
            )
            ax.yaxis.set_major_formatter(
                StrMethodFormatter(
                    "{x:g}" if weighted and length_unit == "km" else "{x:,.0f}"
                )
            )
            # Applied last: the log frame installs its own ticker.
            frame_ylim(ax, panel_frames, mode=y_frame)
            ax.tick_params(axis="both", labelsize=10, colors="#5a6778", length=0, pad=7)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#d6dde6")
        handles = legend_handles(set(page_runs["method"]))
        spare = list(axes.flat)[len(zone_counts) :]
        for ax in spare:
            ax.set_visible(False)
        # An odd panel count leaves a cell free: a legend there is larger and
        # closer to the curves than one stranded above the figure.
        inset_legend = spare[0] if spare else None
        if inset_legend is not None:
            inset_legend.set_visible(True)
            inset_legend.axis("off")
            inset_legend.legend(
                handles=handles,
                loc="upper left",
                frameon=False,
                fontsize=11.5,
                handlelength=2.8,
                labelspacing=1.05,
                borderpad=0.0,
            )
        fig.text(
            0.065,
            0.965,
            f"{'Cut length' if weighted else 'Cut edges'} over time  /  {level}",
            fontsize=20,
            fontweight="semibold",
            color="#182536",
            va="top",
        )
        fig.text(
            0.065,
            0.915,
            f"{minutes(budget):g}-minute budget  ·  Successful runs only  ·  First {skip_initial} feasible improvements omitted",
            fontsize=10,
            color="#647286",
            va="top",
        )
        if inset_legend is None:
            fig.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(0.057, 0.875),
                ncol=min(3, len(handles)),
                frameon=False,
                fontsize=10,
                handlelength=2.5,
                columnspacing=1.9,
            )
        fig.supxlabel(
            "Cumulative solve time (minutes)", fontsize=11, color="#364559", y=0.06
        )
        fig.supylabel(
            f"Cut length ({length_unit})" if weighted else "Cut edges",
            fontsize=11,
            color="#364559",
            x=0.012,
        )
        fig.text(
            0.065,
            0.017,
            "\n".join(
                [
                    (
                        "Band: middle 50% of available runs. Solid median starts only once all runs have retained feasible values."
                        if aggregate == "median"
                        else "Each line is one seed / centroid run; multi-level curves show best stage-feasible cut length so far."
                    ),
                    Y_FRAME_NOTES[y_frame],
                ]
                + (
                    [
                        "Multi-level curves (dashed) use stage feasibility and cumulative recorded solve durations."
                    ]
                    if page_runs["strategy"].eq("recursive").any()
                    else []
                )
            ),
            fontsize=8,
            color="#788392",
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            bottom=0.15 if nrows == 1 else 0.12,
            top=(0.855 if inset_legend is not None else 0.77) if nrows > 1 else 0.71,
            hspace=0.40,
            wspace=0.25,
        )
        prefix = "weighted_boundary" if weighted else "cut_edges"
        suffix = "" if y_frame == "log" else f"_{y_frame}"
        path = output_dir / f"{prefix}_{level}_tl_{budget:g}s_{aggregate}{suffix}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        if aggregate == "median":
            for zones in zone_counts:
                series = {
                    method: lookup[(budget, level, zones, method)]
                    for method in METHODS
                    if (budget, level, zones, method) in lookup
                }
                paths.append(
                    render_zone_detail(
                        series,
                        level=level,
                        zones=zones,
                        budget=budget,
                        output_dir=output_dir / "by_zone",
                        weighted=weighted,
                        length_unit=length_unit,
                        skip_initial=skip_initial,
                        dpi=dpi,
                        y_frame=y_frame,
                    )
                )

    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Sweep YAML, output directory, or summary.csv.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, help="Default: <benchmark root>/plots/edges."
    )
    parser.add_argument(
        "--centroids", nargs="+", help="Exact centroid names, e.g. 6-zone-3."
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--levels", nargs="+", help="Final levels, e.g. Block_2 Block_1."
    )
    parser.add_argument(
        "--time-limits", nargs="+", type=float, help="Budgets in seconds."
    )
    parser.add_argument("--solvers", nargs="+", choices=list(SOLVERS))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Plot weight_edges=true runs, including recursive methods, in cut-length units.",
    )
    parser.add_argument(
        "--length-unit",
        choices=["km", "m"],
        default="km",
        help="Display unit for weighted cut length (default: km); source CSVs remain in metres.",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Exclude recursive methods from weighted plots.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["median", "individual"],
        default="median",
        help="Median with middle-50%% band (default), or individual runs.",
    )
    parser.add_argument(
        "--y-frame",
        choices=list(Y_FRAMES),
        default="log",
        help=(
            "Vertical scale: log keeps the full range on a log axis (default), "
            "zoom frames a linear axis on the median curves, zero is linear from zero. "
            "Non-default choices add a filename suffix, so variants can coexist."
        ),
    )
    parser.add_argument(
        "--skip-initial",
        type=int,
        default=1,
        metavar="N",
        help="Omit the first N distinct feasible improvements per run (default: 1; 0 shows all).",
    )
    args = parser.parse_args(argv)
    if args.skip_initial < 0:
        parser.error("--skip-initial must be nonnegative")
    root = benchmark_root(args.source)
    events, runs, skipped = load_trajectories(
        root,
        **{
            key: getattr(args, key)
            for key in (
                "centroids",
                "seeds",
                "levels",
                "time_limits",
                "solvers",
                "weighted",
                "single_only",
            )
        },
    )
    print(f"Read {len(runs)} selected runs; skipped: {dict(skipped)}")
    if runs.empty:
        print("No matching runs found for the selected objective units.")
        return 1
    output_dir = (args.output_dir or root / "plots" / "edges").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "weighted_boundary" if args.weighted else "cut_edges"
    events.to_csv(output_dir / f"{prefix}_events.csv", index=False)
    retained = visible_trajectories(events, runs, skip_initial=args.skip_initial)
    runs["plotted"] = runs["run_id"].isin(retained)
    runs["skip_initial"] = args.skip_initial
    runs.to_csv(output_dir / f"{prefix}_runs.csv", index=False)
    summaries = aggregate_trajectories(retained, runs)
    summaries.to_csv(output_dir / f"{prefix}_aggregates.csv", index=False)
    for path in render_figures(
        events,
        runs,
        output_dir,
        dpi=args.dpi,
        skip_initial=args.skip_initial,
        aggregate=args.aggregate,
        weighted=args.weighted,
        length_unit=args.length_unit,
        summaries=summaries,
        y_frame=args.y_frame,
    ):
        print(f"Saved {path}")
    print(
        f"Plotted {len(retained)} runs after trimming; "
        f"excluded {int((~runs['has_feasible_objective']).sum())} never-feasible runs "
        f"and {int(runs['has_feasible_objective'].sum()) - len(retained)} with no retained improvements. "
        f"CSVs: {output_dir}"
    )
    return 0 if retained else 1


if __name__ == "__main__":
    raise SystemExit(main())

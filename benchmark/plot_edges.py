"""Plot feasible cut-edge objectives from benchmark logs.

Usage: uv run python -m benchmark.plot_edges benchmark/configs/benchmark_edges.yaml

Like the success heatmap, figures group runs by final graph level, zone count,
and time budget. Each line is one run, never an average across changing sets of
feasible runs in individual mode. The default summarizes a fixed set of feasible
trajectories with medians and interquartile bands. Recursive methods are excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd
import yaml

from benchmark.results import discover_run_dirs
from benchmark.solver_logs import parse_run_dir


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "benchmark_edges.yaml"
# Same solver ordering and names as the success-rate heatmap.
SOLVERS = {
    "recom": ("Recom", "#0072b2"),
    "relaxed_recom": ("Relaxed Recom", "#009e73"),
    "short_bursts": ("Short Bursts", "#e69f00"),
    "adaptive_short_bursts": ("Adaptive Short Bursts", "#cc79a7"),
    "mip": ("MIP", "#d55e00"),
    "cp_bool": ("CP", "#5955ad"),
    "cp_int": ("CP (Int)", "#333333"),
}


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
        if strategy != "single":
            skipped[f"strategy={strategy}"] += 1
            continue
        solver = config.get("solver")
        if solver not in SOLVERS:
            skipped["unsupported solver"] += 1
            continue
        if config.get("weight_edges", False) or config.get(
            "secondary_objective", False
        ):
            skipped["objective is not plain cut edges"] += 1
            continue
        target = config["levels"][-1]
        centroid = str(config["centroids_type"])
        seed = int(config["seed"])
        budget = float(config["solve_time_limits"][-1])
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
            "level": target,
            "centroids_type": centroid,
            "seed": seed,
            "num_zones": int(zone_match[1]),
            "time_limit": budget,
        }
        rows = [
            row
            for row in parse_run_dir(str(run_dir))
            if row.get("level") == target and row.get("solver") == solver
        ]
        # Never silently join restarts / enumerated solves on a resetting clock.
        if len({row["log_file"] for row in rows}) > 1:
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
    keys = ["time_limit", "level", "num_zones", "solver"]
    for identity, group in runs[runs["run_id"].isin(traces)].groupby(keys, sort=True):
        frame = summarize_traces(
            [traces[run_id] for run_id in group["run_id"]], horizon=identity[0]
        )
        frames.append(frame.assign(**dict(zip(keys, identity))))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def render_figures(
    events: pd.DataFrame,
    runs: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int = 180,
    skip_initial: int = 1,
    aggregate: str = "median",
) -> list[Path]:
    """Separate figures by budget AND level, with compact zone-count panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = visible_trajectories(events, runs, skip_initial=skip_initial)
    if not traces:
        return []
    visible_runs = runs[runs["run_id"].isin(traces)]
    paths = []
    groups = visible_runs.groupby(["time_limit", "level"], sort=False)
    for budget, level in sorted(
        groups.groups, key=lambda key: (key[0], _level_key(key[1]))
    ):
        page_runs = groups.get_group((budget, level))
        zone_counts = sorted(page_runs["num_zones"].unique())
        ncols = min(3, len(zone_counts))
        nrows = (len(zone_counts) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            squeeze=False,
            figsize=(max(8.5, 4.8 * ncols), 3.35 * nrows + 2.0),
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
            for solver, solver_runs in panel.groupby("solver", sort=True):
                color = SOLVERS[solver][1]
                alpha = max(0.24, min(0.9, 1.5 / np.sqrt(len(solver_runs))))
                if aggregate == "median":
                    summary = summarize_traces(
                        [traces[run_id] for run_id in solver_runs["run_id"]],
                        horizon=budget,
                    )
                    ax.fill_between(
                        summary["elapsed_seconds"],
                        summary["q25"],
                        summary["q75"],
                        step="post",
                        color=color,
                        alpha=0.13,
                        linewidth=0,
                        zorder=1,
                    )
                    ax.step(
                        summary["elapsed_seconds"],
                        summary["incumbent"],
                        where="post",
                        color=color,
                        linewidth=2.0,
                        zorder=3,
                    )
                    finite = summary.dropna(subset=["incumbent"])
                    if len(finite) == 1:
                        ax.scatter(
                            finite["elapsed_seconds"],
                            finite["incumbent"],
                            color=color,
                            s=12,
                            zorder=4,
                        )
                    continue
                for run in solver_runs.itertuples():
                    values = traces[run.run_id].dropna(subset=["incumbent"])
                    # Only changes and the endpoint are needed for a step curve.
                    keep = values["incumbent"].ne(values["incumbent"].shift())
                    keep.iloc[-1] = True
                    values = values[keep]
                    ax.step(
                        values["elapsed_seconds"],
                        values["incumbent"],
                        where="post",
                        color=color,
                        linewidth=1.65,
                        alpha=alpha,
                        zorder=3,
                    )
                    endpoints = values.iloc[[0, -1]].drop_duplicates()
                    ax.scatter(
                        endpoints["elapsed_seconds"],
                        endpoints["incumbent"],
                        color=color,
                        s=12,
                        alpha=alpha,
                        zorder=4,
                        edgecolors="none",
                    )
            endpoint = panel["last_log_seconds"].max()
            ax.set_xlim(0, max(float(budget), endpoint) * 1.015)
            ax.set_ylim(bottom=0)
            ax.margins(y=0.12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.tick_params(axis="both", labelsize=9, colors="#5a6778", length=0, pad=7)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#d6dde6")
        for ax in list(axes.flat)[len(zone_counts) :]:
            ax.set_visible(False)
        present = set(page_runs["solver"])
        handles = [
            Line2D([], [], color=color, label=label, linewidth=2.2)
            for solver, (label, color) in SOLVERS.items()
            if solver in present
        ]
        fig.text(
            0.065,
            0.965,
            f"Cut edges over time  /  {level}",
            fontsize=20,
            fontweight="semibold",
            color="#182536",
            va="top",
        )
        fig.text(
            0.065,
            0.915,
            f"{budget:,.0f}-second budget  ·  Successful runs only  ·  First {skip_initial} feasible improvements omitted",
            fontsize=10,
            color="#647286",
            va="top",
        )
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.057, 0.875),
            ncol=min(4, len(handles)),
            frameon=False,
            fontsize=9,
            handlelength=2.5,
            columnspacing=1.9,
        )
        fig.supxlabel(
            "Elapsed solve time (seconds)", fontsize=11, color="#364559", y=0.06
        )
        fig.supylabel("Cut edges", fontsize=11, color="#364559", x=0.012)
        fig.text(
            0.065,
            0.017,
            (
                "Band: middle 50% of available runs. Solid median starts only once all runs have retained feasible values."
                if aggregate == "median"
                else "Each line is one seed / centroid run. Recursive methods excluded."
            ),
            fontsize=8,
            color="#788392",
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            bottom=0.15 if nrows == 1 else 0.12,
            top=0.71 if nrows == 1 else 0.77,
            hspace=0.40,
            wspace=0.25,
        )
        path = output_dir / f"cut_edges_{level}_tl_{budget:g}s_{aggregate}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
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
        "--aggregate",
        choices=["median", "individual"],
        default="median",
        help="Median with middle-50%% band (default), or individual runs.",
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
            for key in ("centroids", "seeds", "levels", "time_limits", "solvers")
        },
    )
    print(f"Read {len(runs)} selected runs; skipped: {dict(skipped)}")
    if runs.empty:
        print("No matching single-solve, unweighted cut-edge runs found.")
        return 1
    output_dir = (args.output_dir or root / "plots" / "edges").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "cut_edges_events.csv", index=False)
    retained = visible_trajectories(events, runs, skip_initial=args.skip_initial)
    runs["plotted"] = runs["run_id"].isin(retained)
    runs["skip_initial"] = args.skip_initial
    runs.to_csv(output_dir / "cut_edges_runs.csv", index=False)
    aggregate_trajectories(retained, runs).to_csv(
        output_dir / "cut_edges_aggregates.csv", index=False
    )
    for path in render_figures(
        events,
        runs,
        output_dir,
        dpi=args.dpi,
        skip_initial=args.skip_initial,
        aggregate=args.aggregate,
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

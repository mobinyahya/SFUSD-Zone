#!/usr/bin/env python3
"""Plot feasible-MCMC penalties and objective progress over time.

This script reads the benchmark outputs produced by
``Zone_Generation/benchmark/configs/sweep.feasible-mcmc.yaml``. ReCom solver
logs provide every sampled penalty trajectory; solver-progress logs provide only
real feasible incumbents, which is what we use for the objective trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Zone_Generation.benchmark.config import BenchmarkTask, SimulationSweep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_CONFIG = (
    PROJECT_ROOT / "Zone_Generation/benchmark/configs/sweep.feasible-mcmc.yaml"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_OUTPUT_NAME = "feasible_mcmc_penalties_objective_over_time.png"

SOLVER_LABELS = {
    "relaxed_recom": "Relaxed ReCom",
    "short_bursts_recom": "Short Bursts ReCom",
}
SOLVER_ORDER = ["Relaxed ReCom", "Short Bursts ReCom"]
COMPONENT_LABELS = {
    "assignment": "Assignment",
    "candidate": "Candidate zones",
    "contiguity": "Contiguity",
    "frl": "FRL",
    "overage": "Overage",
    "schools": "Schools",
    "shortage": "Shortage",
}


@dataclass(frozen=True)
class MCMCTables:
    """Tidy tables for plotting feasible-MCMC trajectories."""

    penalties: pd.DataFrame
    components: pd.DataFrame
    objectives: pd.DataFrame
    runs: pd.DataFrame


def main() -> None:
    args = parse_args()
    sweep = SimulationSweep.from_yaml(str(args.config))
    results_dir = (args.results_dir or Path(sweep.execution.output_dir)).expanduser()

    tables = build_mcmc_tables(
        sweep,
        results_dir=results_dir,
        max_log_points_per_run=args.max_log_points_per_run,
        max_components=args.max_components,
    )
    if tables.penalties.empty:
        raise ValueError(
            "No solver penalty rows were found. Confirm save_solver_logs was true "
            f"and the sweep output exists under {results_dir}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    if args.separate_zones:
        output_paths = []
        for centroids_type in centroids_order(tables.penalties):
            output_paths.append(
                plot_mcmc_tables(
                    tables,
                    centroids_type_output_path(output_path, centroids_type),
                    centroids_type=centroids_type,
                    time_bin_seconds=args.time_bin_seconds,
                )
            )
    else:
        output_paths = [
            plot_mcmc_tables(
                tables,
                output_path,
                time_bin_seconds=args.time_bin_seconds,
            )
        ]

    csv_paths = write_csv_tables(tables, args.csv_output or output_path)

    feasible_runs = int(tables.runs["has_real_feasible_solution"].sum())
    print(f"Loaded {len(tables.runs)} generated run(s).")
    print(f"Runs with real feasible incumbents: {feasible_runs}/{len(tables.runs)}")
    print(f"Penalty rows plotted: {len(tables.penalties)}")
    print(f"Objective incumbent rows plotted: {len(tables.objectives)}")
    for csv_path in csv_paths:
        print(f"Wrote {csv_path}")
    for path in output_paths:
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot feasible-MCMC penalty trajectories and feasible objective "
            "improvements from saved benchmark solver logs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SWEEP_CONFIG,
        help=f"Sweep YAML. Default: {DEFAULT_SWEEP_CONFIG}",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Benchmark output directory. Default: execution.output_dir from --config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plots and CSVs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Plot filename. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help=(
            "Base CSV path. Default uses --output-dir/--output-name with "
            "_penalties.csv, _components.csv, _objectives.csv, and _runs.csv."
        ),
    )
    parser.add_argument(
        "--separate-zones",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write one plot per centroids_type. Enabled by default; use "
            "--no-separate-zones to write one combined plot."
        ),
    )
    parser.add_argument(
        "--max-log-points-per-run",
        type=int,
        default=600,
        help=(
            "Maximum solver-log rows to plot per run, sampled evenly while "
            "preserving feasibility and incumbent-change events. Default: 600."
        ),
    )
    parser.add_argument(
        "--time-bin-seconds",
        type=float,
        default=30.0,
        help="Bin width for bold median trajectories. Default: 30 seconds.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=6,
        help="Maximum penalty components to show, ranked by total contribution.",
    )
    return parser.parse_args()


def build_mcmc_tables(
    sweep: SimulationSweep,
    *,
    results_dir: Path,
    max_log_points_per_run: int | None,
    max_components: int,
) -> MCMCTables:
    """Load solver logs and feasible incumbent progress for a sweep."""

    if max_log_points_per_run is not None and max_log_points_per_run < 2:
        raise ValueError("max_log_points_per_run must be at least 2 when provided.")
    if max_components < 1:
        raise ValueError("max_components must be at least 1.")

    penalty_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    sweep_root = Path(sweep.execution.output_dir).expanduser()
    tasks = sweep.generate_tasks()
    for run_number, task in enumerate(tasks, start=1):
        run_dir = task_run_dir(task, sweep_root=sweep_root, results_dir=results_dir)
        manifest_path = run_dir / "benchmark_manifest.json"
        if not manifest_path.exists():
            continue

        manifest = load_json(manifest_path)
        config = dict(task.config)
        run_context = run_context_for(
            run_number=run_number,
            task=task,
            run_dir=run_dir,
            manifest=manifest,
            config=config,
        )
        stage_log_first_feasible_times: list[float] = []
        stage_progress_first_feasible_times: list[float] = []
        stage_objectives: list[float] = []

        for stage in manifest.get("stages", []):
            if not isinstance(stage, Mapping):
                continue
            stage_context = stage_context_for(run_dir, stage)
            context = {**run_context, **stage_context}
            metadata = dict(stage.get("metadata") or {})

            log_path = resolve_metadata_path(run_dir, metadata.get("solver_log_path"))
            if log_path is not None and log_path.exists():
                log_rows = load_jsonl(log_path)
                selected_rows = selected_log_rows(log_rows, max_log_points_per_run)
                for log_index, log_row in selected_rows:
                    penalty_row = penalty_record(context, log_path, log_index, log_row)
                    if penalty_row is not None:
                        penalty_rows.append(penalty_row)
                first_log_feasible = first_feasible_time(log_rows)
                if first_log_feasible is not None:
                    stage_log_first_feasible_times.append(first_log_feasible)

            progress_path = stage_progress_path(run_dir, stage, metadata)
            if progress_path is not None and progress_path.exists():
                progress_rows = load_jsonl(progress_path)
                for progress_index, progress_row in enumerate(progress_rows):
                    objective_row = objective_record(
                        context,
                        progress_path,
                        progress_index,
                        progress_row,
                    )
                    if objective_row is not None:
                        objective_rows.append(objective_row)
                        stage_progress_first_feasible_times.append(
                            float(objective_row["elapsed_seconds"])
                        )
                        stage_objectives.append(float(objective_row["objective"]))

        first_real_feasible_time = min(stage_progress_first_feasible_times, default=None)
        if first_real_feasible_time is None:
            first_real_feasible_time = min(stage_log_first_feasible_times, default=None)
        run_rows.append(
            {
                **run_context,
                "status": manifest.get("status"),
                "has_real_feasible_solution": first_real_feasible_time is not None,
                "first_real_feasible_seconds": first_real_feasible_time,
                "best_objective": min(stage_objectives, default=None),
                "num_objective_incumbents": len(stage_objectives),
            }
        )

    penalty_df = pd.DataFrame(penalty_rows)
    objective_df = pd.DataFrame(objective_rows)
    run_df = pd.DataFrame(run_rows)
    component_df = penalty_components_dataframe(penalty_df, max_components)
    return MCMCTables(
        penalties=penalty_df,
        components=component_df,
        objectives=objective_df,
        runs=run_df,
    )


def run_context_for(
    *,
    run_number: int,
    task: BenchmarkTask,
    run_dir: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    solver = str(config.get("solver") or "unknown")
    centroids_type = str(config.get("centroids_type") or "unknown")
    seed = config.get("seed")
    shortage = config.get("shortage")
    run_id = str(manifest.get("task_id") or task.task_id or task.config_hash[:12])
    return {
        "run_number": run_number,
        "run_id": run_id,
        "task_id": manifest.get("task_id", task.task_id),
        "config_hash": manifest.get("config_hash", task.config_hash),
        "run_dir": str(run_dir),
        "centroids_type": centroids_type,
        "zone_count": zone_count(centroids_type),
        "solver": solver,
        "solver_label": solver_label(solver),
        "seed": seed,
        "shortage": shortage,
        "overage": config.get("overage"),
        "frl_dev": config.get("frl_dev"),
        "racial_dev": config.get("racial_dev"),
        "run_label": run_label(centroids_type, solver, seed, shortage, run_id),
    }


def stage_context_for(run_dir: Path, stage: Mapping[str, Any]) -> dict[str, Any]:
    stage_path = stage.get("path")
    return {
        "stage_name": stage.get("name"),
        "stage_index": stage.get("index"),
        "stage_level": stage.get("level"),
        "stage_status": stage.get("status"),
        "stage_objective": stage.get("objective"),
        "stage_wall_time": stage.get("wall_time"),
        "stage_dir": str(run_dir / str(stage_path or "")),
    }


def penalty_record(
    context: Mapping[str, Any],
    log_path: Path,
    log_index: int,
    row: Mapping[str, Any],
) -> dict[str, Any] | None:
    penalty = finite_float(row.get("penalty"))
    components = numeric_components(row.get("penalty_components"))
    if penalty is None:
        penalty = sum(components.values()) if components else None
    if penalty is None:
        return None
    elapsed_seconds = finite_float(row.get("elapsed_seconds"))
    if elapsed_seconds is None:
        return None

    best_cut_edges = finite_float(row.get("best_cut_edges"))
    return {
        **context,
        "source": "solver_log",
        "log_path": str(log_path),
        "log_index": log_index,
        "event": row.get("event"),
        "iteration": row.get("iteration"),
        "elapsed_seconds": elapsed_seconds,
        "penalty": float(penalty),
        "cut_edges": finite_float(row.get("cut_edges")),
        "feasible": bool(row.get("feasible", False)),
        "best_feasible": bool(row.get("best_feasible", False)),
        "best_cut_edges": best_cut_edges,
        "penalty_components": components,
        "penalty_components_json": json.dumps(
            components, sort_keys=True, separators=(",", ":")
        ),
    }


def objective_record(
    context: Mapping[str, Any],
    progress_path: Path,
    progress_index: int,
    row: Mapping[str, Any],
) -> dict[str, Any] | None:
    objective = finite_float(row.get("objective"))
    elapsed_seconds = finite_float(row.get("elapsed_seconds"))
    if objective is None or elapsed_seconds is None:
        return None
    return {
        **context,
        "source": "solver_progress",
        "progress_path": str(progress_path),
        "progress_index": progress_index,
        "solution_index": row.get("solution_index", progress_index),
        "iteration": row.get("iteration"),
        "elapsed_seconds": elapsed_seconds,
        "objective": float(objective),
    }


def penalty_components_dataframe(
    penalty_df: pd.DataFrame,
    max_components: int,
) -> pd.DataFrame:
    if penalty_df.empty:
        return pd.DataFrame()

    component_totals: dict[str, float] = {}
    for components in penalty_df["penalty_components"]:
        if not isinstance(components, Mapping):
            continue
        for component, value in components.items():
            number = finite_float(value)
            if number is None:
                continue
            component_totals[str(component)] = component_totals.get(str(component), 0.0) + number
    components = [
        component
        for component, _total in sorted(
            component_totals.items(), key=lambda item: (-item[1], item[0])
        )[:max_components]
    ]
    if not components:
        return pd.DataFrame()

    id_columns = [
        "run_number",
        "run_id",
        "task_id",
        "centroids_type",
        "zone_count",
        "solver",
        "solver_label",
        "seed",
        "shortage",
        "stage_name",
        "stage_index",
        "stage_level",
        "elapsed_seconds",
        "iteration",
        "log_index",
    ]
    rows: list[dict[str, Any]] = []
    for row in penalty_df.to_dict("records"):
        values = row.get("penalty_components") or {}
        base = {column: row.get(column) for column in id_columns}
        for component in components:
            rows.append(
                {
                    **base,
                    "component": component,
                    "component_label": component_label(component),
                    "component_value": float(values.get(component, 0.0) or 0.0),
                }
            )
    return pd.DataFrame(rows)


def plot_mcmc_tables(
    tables: MCMCTables,
    output_path: str | Path,
    *,
    centroids_type: str | None = None,
    time_bin_seconds: float = 30.0,
) -> Path:
    penalty_df = filter_centroids(tables.penalties, centroids_type)
    component_df = filter_centroids(tables.components, centroids_type)
    objective_df = filter_centroids(tables.objectives, centroids_type)
    run_df = filter_centroids(tables.runs, centroids_type)

    if penalty_df.empty:
        raise ValueError(f"No penalty rows found for centroids_type={centroids_type!r}.")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 12),
        sharex=True,
        constrained_layout=True,
        height_ratios=[1.1, 1.1, 1.0],
    )
    colors = solver_colors(penalty_df, objective_df)

    plot_total_penalty(
        axes[0], penalty_df, colors, time_bin_seconds=time_bin_seconds
    )
    plot_penalty_components(
        axes[1], component_df, colors, time_bin_seconds=time_bin_seconds
    )
    plot_objectives(axes[2], objective_df, run_df, colors)

    title_suffix = centroids_type or "all centroid configurations"
    fig.suptitle(
        f"Feasible-MCMC Penalties and Real Feasible Objective Progress: {title_suffix}",
        fontsize=15,
        fontweight="bold",
    )

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_total_penalty(
    ax: plt.Axes,
    df: pd.DataFrame,
    colors: Mapping[str, tuple[float, float, float]],
    *,
    time_bin_seconds: float,
) -> None:
    for (_run_id, solver), run_df in df.groupby(["run_id", "solver_label"], sort=False):
        run_df = run_df.sort_values("elapsed_seconds")
        ax.plot(
            run_df["elapsed_seconds"],
            run_df["penalty"],
            color=colors.get(str(solver)),
            alpha=0.14,
            linewidth=0.8,
        )

    median_df = binned_median(
        df,
        "penalty",
        group_columns=["solver_label"],
        time_bin_seconds=time_bin_seconds,
    )
    for solver, solver_df in median_df.groupby("solver_label", sort=False):
        ax.plot(
            solver_df["time_seconds"],
            solver_df["penalty"],
            color=colors.get(str(solver)),
            linewidth=2.8,
            label=f"{solver} median",
        )

    maybe_symlog(ax, df["penalty"])
    ax.set_title("Total constraint penalty over sampled MCMC time")
    ax.set_ylabel("Penalty")
    ax.set_xlim(left=0)
    ax.legend(loc="upper right", frameon=True)


def plot_penalty_components(
    ax: plt.Axes,
    df: pd.DataFrame,
    colors: Mapping[str, tuple[float, float, float]],
    *,
    time_bin_seconds: float,
) -> None:
    if df.empty:
        empty_axis(ax, "No penalty component rows found")
        return

    median_df = binned_median(
        df,
        "component_value",
        group_columns=["solver_label", "component_label"],
        time_bin_seconds=time_bin_seconds,
    )
    styles = {solver: style for solver, style in zip(SOLVER_ORDER, ["-", "--"], strict=False)}
    component_labels = sorted(median_df["component_label"].dropna().unique())
    component_palette = dict(
        zip(component_labels, sns.color_palette("tab10", len(component_labels)), strict=False)
    )

    for (solver, component), group_df in median_df.groupby(
        ["solver_label", "component_label"], sort=False
    ):
        ax.plot(
            group_df["time_seconds"],
            group_df["component_value"],
            color=component_palette.get(component),
            linestyle=styles.get(str(solver), "-"),
            linewidth=2.0,
            label=f"{component} ({solver})",
        )

    maybe_symlog(ax, df["component_value"])
    ax.set_title("Median penalty components over sampled MCMC time")
    ax.set_ylabel("Component penalty")
    ax.legend(loc="upper right", ncols=2, fontsize=8, frameon=True)


def plot_objectives(
    ax: plt.Axes,
    objective_df: pd.DataFrame,
    run_df: pd.DataFrame,
    colors: Mapping[str, tuple[float, float, float]],
) -> None:
    if objective_df.empty:
        total_runs = len(run_df)
        empty_axis(ax, f"No real feasible incumbents saved for {total_runs} run(s)")
        ax.set_xlabel("Elapsed solver time (seconds)")
        return

    labeled: set[str] = set()
    for (run_id, solver), run_objectives in objective_df.groupby(
        ["run_id", "solver_label"], sort=False
    ):
        run_objectives = run_objectives.sort_values("elapsed_seconds")
        solver = str(solver)
        label = solver if solver not in labeled else None
        ax.step(
            run_objectives["elapsed_seconds"],
            run_objectives["objective"],
            where="post",
            color=colors.get(solver),
            alpha=0.45,
            linewidth=1.4,
            label=label,
        )
        ax.scatter(
            run_objectives["elapsed_seconds"],
            run_objectives["objective"],
            color=colors.get(solver),
            edgecolor="white",
            linewidth=0.5,
            s=28,
            alpha=0.85,
            zorder=3,
        )
        first = run_objectives.iloc[0]
        ax.scatter(
            [first["elapsed_seconds"]],
            [first["objective"]],
            color=colors.get(solver),
            edgecolor="black",
            marker="*",
            linewidth=0.7,
            s=105,
            zorder=4,
        )
        labeled.add(solver)

    feasible_runs = int(run_df["has_real_feasible_solution"].sum()) if not run_df.empty else 0
    total_runs = len(run_df)
    ax.text(
        0.01,
        0.96,
        f"Star = first real feasible incumbent\nFeasible runs: {feasible_runs}/{total_runs}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.set_title("Objective after a real feasible solution is found")
    ax.set_xlabel("Elapsed solver time (seconds)")
    ax.set_ylabel("Objective (cut edges)")
    ax.set_xlim(left=0)
    ax.legend(loc="upper right", frameon=True)


def binned_median(
    df: pd.DataFrame,
    value_column: str,
    *,
    group_columns: Sequence[str],
    time_bin_seconds: float,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    data = df[[*group_columns, "run_id", "elapsed_seconds", value_column]].copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data["elapsed_seconds"] = pd.to_numeric(data["elapsed_seconds"], errors="coerce")
    data = data.dropna(subset=["elapsed_seconds", value_column])
    if data.empty:
        return pd.DataFrame()

    bin_seconds = max(float(time_bin_seconds), 1e-9)
    data["time_bin"] = (data["elapsed_seconds"] / bin_seconds).round().astype(int)
    per_run = (
        data.groupby([*group_columns, "run_id", "time_bin"], as_index=False)[value_column]
        .mean()
        .sort_values([*group_columns, "time_bin"], kind="stable")
    )
    median = per_run.groupby([*group_columns, "time_bin"], as_index=False)[
        value_column
    ].median()
    median["time_seconds"] = median["time_bin"] * bin_seconds
    return median.sort_values([*group_columns, "time_seconds"], kind="stable")


def write_csv_tables(tables: MCMCTables, output_base: str | Path) -> list[Path]:
    base = Path(output_base).expanduser()
    if base.suffix.lower() != ".csv":
        base = base.with_suffix(".csv")
    base.parent.mkdir(parents=True, exist_ok=True)

    penalty_path = suffixed_csv(base, "penalties")
    component_path = suffixed_csv(base, "components")
    objective_path = suffixed_csv(base, "objectives")
    run_path = suffixed_csv(base, "runs")

    penalty_df = tables.penalties.drop(columns=["penalty_components"], errors="ignore")
    penalty_df.to_csv(penalty_path, index=False)
    tables.components.to_csv(component_path, index=False)
    tables.objectives.to_csv(objective_path, index=False)
    tables.runs.to_csv(run_path, index=False)
    return [penalty_path, component_path, objective_path, run_path]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def selected_log_rows(
    rows: Sequence[dict[str, Any]],
    max_rows: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    if not rows:
        return []
    if max_rows is None or len(rows) <= max_rows:
        return list(enumerate(rows))

    last_idx = len(rows) - 1
    indices = {
        round(i * last_idx / (max_rows - 1))
        for i in range(max_rows)
    }
    previous_best = object()
    for idx, row in enumerate(rows):
        if row.get("feasible"):
            indices.add(idx)
        best = row.get("best_cut_edges")
        if best is not None and best != previous_best:
            indices.add(idx)
            previous_best = best
    indices.add(0)
    indices.add(last_idx)
    return [(idx, rows[idx]) for idx in sorted(indices)]


def first_feasible_time(rows: Sequence[Mapping[str, Any]]) -> float | None:
    times = [
        elapsed
        for row in rows
        if row.get("feasible")
        for elapsed in [finite_float(row.get("elapsed_seconds"))]
        if elapsed is not None
    ]
    return min(times, default=None)


def task_run_dir(
    task: BenchmarkTask,
    *,
    sweep_root: Path,
    results_dir: Path,
) -> Path:
    task_dir = Path(task.output_dir).expanduser()
    try:
        rel_path = task_dir.relative_to(sweep_root)
    except ValueError:
        return task_dir if results_dir == sweep_root else results_dir / task_dir.name
    return results_dir / rel_path


def stage_progress_path(
    run_dir: Path,
    stage: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path | None:
    raw_path = metadata.get("solver_progress_path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    stage_dir = run_dir / str(stage.get("path") or "")
    stage_candidate = stage_dir / path
    if stage_candidate.exists():
        return stage_candidate
    run_candidate = run_dir / path
    if run_candidate.exists():
        return run_candidate
    return stage_candidate


def resolve_metadata_path(base_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base_dir / path


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def numeric_components(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out = {}
    for key, raw_number in value.items():
        number = finite_float(raw_number)
        if number is not None:
            out[str(key)] = number
    return out


def solver_label(solver: str) -> str:
    return SOLVER_LABELS.get(str(solver), str(solver).replace("_", " ").title())


def component_label(component: str) -> str:
    if component.startswith("race:"):
        return component.replace("race:", "Race: ").replace("_", " ")
    return COMPONENT_LABELS.get(component, component.replace("_", " ").title())


def run_label(
    centroids_type: str,
    solver: str,
    seed: Any,
    shortage: Any,
    run_id: str,
) -> str:
    return (
        f"{centroids_type} seed={seed} shortage={shortage} "
        f"{solver_label(solver)} ({run_id})"
    )


def zone_count(centroids_type: str) -> int | None:
    match = re.search(r"(\d+)-zone", str(centroids_type))
    return int(match.group(1)) if match else None


def solver_colors(*dfs: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    values: list[str] = []
    for df in dfs:
        if df.empty or "solver_label" not in df:
            continue
        values.extend(str(value) for value in df["solver_label"].dropna().unique())
    ordered = [solver for solver in SOLVER_ORDER if solver in values]
    ordered.extend(sorted(set(values) - set(ordered)))
    palette = sns.color_palette("colorblind", max(1, len(ordered)))
    return dict(zip(ordered, palette, strict=False))


def maybe_symlog(ax: plt.Axes, values: pd.Series) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    positive = numeric[numeric > 0]
    if positive.empty:
        return
    if float(positive.max()) / max(float(positive.median()), 1e-9) > 50:
        ax.set_yscale("symlog", linthresh=0.01)


def empty_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
    ax.set_yticks([])
    ax.grid(False)


def centroids_order(df: pd.DataFrame) -> list[str]:
    values = [str(value) for value in df["centroids_type"].dropna().unique()]
    return sorted(values, key=lambda value: (zone_count(value) or 999, value))


def filter_centroids(df: pd.DataFrame, centroids_type: str | None) -> pd.DataFrame:
    if centroids_type is None or df.empty or "centroids_type" not in df:
        return df.copy()
    return df[df["centroids_type"].astype(str) == str(centroids_type)].copy()


def centroids_type_output_path(output_path: str | Path, centroids_type: str) -> Path:
    output = Path(output_path).expanduser()
    suffix = output.suffix or ".png"
    return output.with_name(
        f"{output.stem}_{safe_filename(str(centroids_type))}{suffix}"
    )


def suffixed_csv(base: Path, suffix: str) -> Path:
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "plot"


if __name__ == "__main__":
    main()

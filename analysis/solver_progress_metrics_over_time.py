#!/usr/bin/env python3
"""Compute and plot metrics over saved solver-progress incumbents.

This reads benchmark sweep output from the sweep config's ``execution.output_dir``.
For every generated task, each saved stage's ``progress.jsonl`` is loaded, each
intermediate assignment is reconstructed as a ``ZoneSolution``, and the normal
optimization-native metrics are recomputed for that incumbent.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is available in normal uv envs.

    def tqdm(iterable, **_kwargs):
        return iterable

from Zone_Generation.Config.metrics_config import (
    METRIC_BY_COLUMN,
    METRIC_BY_NAME,
    resolve_metric_identifiers,
)
from Zone_Generation.benchmark.config import SimulationSweep
from Zone_Generation.benchmark.runner import load_solutions
from Zone_Generation.metrics import (
    choice,
    distance,
    diversity,
    programs,
    quality,
    run_metrics,
    structure,
    MetricsCalculator,
)
from Zone_Generation.optimization.solution import ZoneSolution


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
BASE_PROGRESS_COLUMNS = {
    "assignment_path",
    "config_hash",
    "elapsed_seconds",
    "explicit_task_index",
    "explicit_task_number",
    "iteration",
    "level",
    "objective",
    "progress_path",
    "progress_step",
    "run_dir",
    "solution_index",
    "stage_index",
    "stage_name",
    "stage_path",
    "task_id",
    "task_number",
    "time_seconds",
}
CATEGORY_MODULES = {
    "choice": choice.compute,
    "diversity": diversity.compute,
    "programs": programs.compute,
    "proximity": distance.compute,
    "quality": quality.compute,
    "run": run_metrics.compute,
    "structure": structure.compute,
}
OUTLIER_MAD_MULTIPLIER = 4.0
OUTLIER_IQR_MULTIPLIER = 2.0


@dataclass(frozen=True)
class ProgressTaskJob:
    """One generated benchmark run assigned to a worker."""

    task_number: int
    explicit_task_number: int
    output_dir: str
    task_id: str
    config_hash: str
    config: dict[str, Any]
    strict_metrics: bool
    metrics: tuple[str, ...] | None
    sample_every: int
    max_steps: int | None


def solver_progress_metrics_dataframe(
    config_path: str | Path,
    *,
    strict: bool | None = None,
    show_progress: bool = True,
    metrics: Sequence[str] | None = None,
    sample_every: int = 1,
    max_steps: int | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """Return one row per saved solver-progress incumbent in a sweep.

    The returned frame includes task/stage/progress metadata plus every flat
    metric emitted by ``MetricsCalculator`` for the intermediate solution.
    ``time_seconds`` is cumulative within each task, so recursive stages form one
    continuous timeline; ``elapsed_seconds`` remains the stage-local solver time.
    """

    sweep = SimulationSweep.from_yaml(str(config_path))
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1.")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be >= 1 when provided.")
    if workers < 1:
        raise ValueError("workers must be >= 1.")
    strict_metrics = sweep.metrics.strict if strict is None else strict
    tasks = sweep.generate_tasks()
    explicit_task_count = _explicit_task_count(sweep)
    metric_names = tuple(metrics) if metrics is not None else None
    jobs = [
        ProgressTaskJob(
            task_number=task_number,
            explicit_task_number=((task_number - 1) % explicit_task_count) + 1,
            output_dir=task.output_dir,
            task_id=task.task_id,
            config_hash=task.config_hash,
            config=dict(task.config),
            strict_metrics=strict_metrics,
            metrics=metric_names,
            sample_every=sample_every,
            max_steps=max_steps,
        )
        for task_number, task in enumerate(tasks, start=1)
    ]

    rows = _run_progress_jobs(jobs, workers=workers, show_progress=show_progress)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["task_number", "stage_index", "time_seconds", "solution_index"],
        kind="stable",
    ).reset_index(drop=True)
    df["progress_step"] = df.groupby("task_number").cumcount()
    return df


def plot_progress_metric(
    df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    separate: bool = False,
    ignore_outliers: bool = True,
) -> Path:
    """Plot one metric over cumulative solver time.

    By default, generated runs are aggregated by the explicit YAML task index.
    With ``separate=True``, each generated run is plotted separately.
    """

    if df.empty:
        raise ValueError("No solver-progress metric rows were found.")
    if metric not in df.columns:
        available = ", ".join(_numeric_columns(df)[:30])
        raise ValueError(
            f"Metric {metric!r} is not present in the progress DataFrame. "
            f"Available numeric columns include: {available}"
        )

    if separate:
        return _plot_separate_progress_metric(
            df,
            metric,
            output_path,
            labels=labels,
            ignore_outliers=ignore_outliers,
        )
    return _plot_aggregated_progress_metric(
        df,
        metric,
        output_path,
        labels=labels,
        ignore_outliers=ignore_outliers,
    )


def plot_progress_metric_by_centroids_type(
    df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    separate: bool = False,
    ignore_outliers: bool = True,
    centroids_types: Sequence[str] | None = None,
) -> list[Path]:
    """Write one progress plot per ``config_centroids_type`` value."""

    if "config_centroids_type" not in df.columns:
        raise ValueError(
            "Progress DataFrame is missing config_centroids_type; cannot write "
            "separate zone files. Pass --no-separate-zones to write one file."
        )

    output_paths: list[Path] = []
    zone_keys = _ordered_centroids_types(df, centroids_types)
    centroids_column = df["config_centroids_type"].map(_centroids_type_key)
    for centroids_type in zone_keys:
        zone_df = df[centroids_column == centroids_type].copy()
        if zone_df.empty:
            continue
        if labels is not None:
            group_column = "task_number" if separate else "explicit_task_number"
            label_df = zone_df[[group_column, metric]].copy()
            label_df[metric] = pd.to_numeric(label_df[metric], errors="coerce")
            group_count = label_df.dropna(subset=[metric])[group_column].nunique()
            if group_count < len(labels):
                print(
                    f"Skipping {centroids_type}: expected {len(labels)} label group(s), "
                    f"found {group_count}."
                )
                continue
        output_paths.append(
            plot_progress_metric(
                zone_df,
                metric,
                _centroids_type_output_path(output_path, centroids_type),
                labels=labels,
                separate=separate,
                ignore_outliers=ignore_outliers,
            )
        )

    if not output_paths:
        raise ValueError(
            "No solver-progress rows matched any centroids_type value with enough "
            "groups to plot."
        )
    return output_paths


def _plot_separate_progress_metric(
    df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    ignore_outliers: bool = True,
) -> Path:
    """Plot one line per generated task/run."""

    task_numbers = list(dict.fromkeys(df["task_number"].tolist()))
    if labels is not None and len(labels) != len(task_numbers):
        raise ValueError(
            f"Expected {len(task_numbers)} label(s), got {len(labels)}."
        )
    label_by_task = {
        task_number: labels[idx] if labels is not None else f"Task {task_number}"
        for idx, task_number in enumerate(task_numbers)
    }

    plot_df = df[["task_number", "time_seconds", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if ignore_outliers:
        plot_df = _drop_upper_outliers(plot_df, metric, group_columns=["task_number"])
    if plot_df.empty:
        raise ValueError(f"Metric {metric!r} has no numeric values to plot.")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for task_number, task_df in plot_df.groupby("task_number", sort=False):
        task_df = task_df.sort_values("time_seconds")
        ax.plot(
            task_df["time_seconds"],
            task_df[metric],
            marker="o",
            linewidth=2.0,
            label=label_by_task[int(task_number)],
        )

    title = _metric_title(metric)
    ax.set_title(f"{title} Over Solver Progress")
    ax.set_xlabel("Solver time (seconds)")
    ax.set_ylabel(title)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _plot_aggregated_progress_metric(
    df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    ignore_outliers: bool = True,
) -> Path:
    """Plot mean trajectory and min/max run bands per explicit YAML task."""

    required = {"explicit_task_number", "task_number", "progress_step"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Progress DataFrame is missing columns: {sorted(missing)}")

    plot_df = df[
        ["explicit_task_number", "task_number", "progress_step", "time_seconds", metric]
    ].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if ignore_outliers:
        plot_df = _drop_upper_outliers(
            plot_df,
            metric,
            group_columns=["explicit_task_number"],
        )
    if plot_df.empty:
        raise ValueError(f"Metric {metric!r} has no numeric values to plot.")

    groups = list(dict.fromkeys(plot_df["explicit_task_number"].tolist()))
    if labels is not None and len(labels) != len(groups):
        raise ValueError(f"Expected {len(groups)} label(s), got {len(labels)}.")
    label_by_group = {
        group: labels[idx] if labels is not None else f"Task {group}"
        for idx, group in enumerate(groups)
    }

    agg = (
        plot_df.groupby(["explicit_task_number", "progress_step"], as_index=False)
        .agg(
            time_seconds=("time_seconds", "mean"),
            metric_mean=(metric, "mean"),
            metric_min=(metric, "min"),
            metric_max=(metric, "max"),
            run_count=("task_number", "nunique"),
        )
        .sort_values(["explicit_task_number", "progress_step"])
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for group, group_df in agg.groupby("explicit_task_number", sort=False):
        group_df = group_df.sort_values("time_seconds")
        x = group_df["time_seconds"].to_numpy(dtype=float)
        mean = group_df["metric_mean"].to_numpy(dtype=float)
        lower = group_df["metric_min"].to_numpy(dtype=float)
        upper = group_df["metric_max"].to_numpy(dtype=float)
        line = ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.3,
            label=label_by_group[int(group)],
        )[0]
        ax.fill_between(
            x,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.18,
            linewidth=0,
        )

    title = _metric_title(metric)
    ax.set_title(f"Mean {title} Over Solver Progress")
    ax.set_xlabel("Solver time (seconds)")
    ax.set_ylabel(title)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a selected metric over saved solver-progress incumbents for "
            "every task in a benchmark sweep."
        )
    )
    parser.add_argument("config", type=Path, help="Path to simulation sweep YAML.")
    parser.add_argument(
        "metric",
        help=(
            "Metric column or display name to plot, such as frl_mad, cut_edges, "
            "or 'Socioeconomic Diversity'."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Default: analysis/plots/solver_progress_<metric>.png. "
            "With --separate-zones, the centroids_type is appended to the filename."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Optional labels for plotted lines. Default mode expects one label per "
            "explicit YAML task; --seperate expects one label per generated run."
        ),
    )
    parser.add_argument(
        "--seperate",
        "--separate",
        dest="separate",
        action="store_true",
        help=(
            "Plot each generated run separately instead of aggregating by explicit "
            "YAML task. The misspelled --seperate spelling is supported as requested."
        ),
    )
    parser.add_argument(
        "--separate-zones",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write one plot per centroids_type, aggregating only across the other "
            "generated task fields. Enabled by default; use --no-separate-zones "
            "to write one combined plot."
        ),
    )
    parser.add_argument(
        "--ignore-outliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ignore high-side metric outliers when plotting, which keeps early "
            "very-high objective values from dominating the y-axis. Enabled by "
            "default; use --no-ignore-outliers to plot every point."
        ),
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Continue when an individual metric module fails.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help=(
            "Compute every metric for the DataFrame. By default the CLI computes "
            "only the selected plot metric when it can infer the required module."
        ),
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Compute metrics for every Nth progress row, always keeping the last row.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum progress rows per stage to evaluate, sampled evenly.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for recomputing metrics across generated "
            "runs. Default: 1."
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Load a previously saved progress metrics CSV instead of recomputing.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="CSV output path. Default: analysis/plots/solver_progress_<metric>.csv",
    )
    args = parser.parse_args(argv)

    if args.input_csv is not None:
        df = pd.read_csv(Path(args.input_csv).expanduser())
    else:
        metric_for_compute = _resolve_metric_identifier(args.metric)
        df = solver_progress_metrics_dataframe(
            args.config,
            strict=False if args.non_strict else None,
            metrics=None if args.all_metrics else [metric_for_compute],
            sample_every=args.sample_every,
            max_steps=args.max_steps,
            workers=args.workers,
        )
    if df.empty:
        raise ValueError(
            "No solver-progress rows found. Confirm save_solver_progress was true "
            "and the sweep output exists under execution.output_dir."
        )

    metric = resolve_metric(args.metric, df.columns)
    output = args.output or (
        DEFAULT_OUTPUT_DIR / f"solver_progress_{_safe_filename(metric)}.png"
    )
    csv_output = args.csv_output or (
        DEFAULT_OUTPUT_DIR / f"solver_progress_{_safe_filename(metric)}.csv"
    )
    csv_output = Path(csv_output).expanduser()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_output, index=False)

    if args.separate_zones:
        output_paths = plot_progress_metric_by_centroids_type(
            df,
            metric,
            output,
            labels=args.labels,
            separate=args.separate,
            ignore_outliers=args.ignore_outliers,
            centroids_types=_config_centroids_types(args.config),
        )
    else:
        output_paths = [
            plot_progress_metric(
                df,
                metric,
                output,
                labels=args.labels,
                separate=args.separate,
                ignore_outliers=args.ignore_outliers,
            )
        ]
    print(f"Loaded {df['task_number'].nunique()} generated run(s).")
    print(f"Found {df['explicit_task_number'].nunique()} explicit YAML task group(s).")
    print(f"Computed {len(df)} solver-progress metric row(s).")
    print(f"Wrote {csv_output}")
    for output_path in output_paths:
        print(f"Wrote {output_path}")


def resolve_metric(metric: str, columns: Sequence[str]) -> str:
    """Resolve display names through the metric registry, then fall back to columns."""

    if metric in columns:
        return metric
    if metric in METRIC_BY_NAME:
        column = METRIC_BY_NAME[metric].column
        if column in columns:
            return column
    if metric in METRIC_BY_COLUMN and metric in columns:
        return metric

    resolved = resolve_metric_identifiers([metric])
    if len(resolved) == 1 and resolved[0] in columns:
        return resolved[0]

    lower_lookup = {str(column).lower(): str(column) for column in columns}
    if metric.lower() in lower_lookup:
        return lower_lookup[metric.lower()]

    available = ", ".join(_numeric_columns(pd.DataFrame(columns=columns))[:30])
    raise ValueError(
        f"Could not resolve metric {metric!r}. Available columns include: {available}"
    )


def _run_progress_jobs(
    jobs: Sequence[ProgressTaskJob],
    *,
    workers: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if workers == 1:
        rows: list[dict[str, Any]] = []
        for job in tqdm(
            jobs,
            desc="Building progress metrics",
            unit="run",
            disable=not show_progress,
        ):
            rows.extend(_progress_rows_for_task(job))
        return rows

    max_workers = min(int(workers), len(jobs))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_progress_rows_for_task, job) for job in jobs]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Building progress metrics ({max_workers} workers)",
            unit="run",
            disable=not show_progress,
        ):
            rows.extend(future.result())
    return rows


def _progress_rows_for_task(job: ProgressTaskJob) -> list[dict[str, Any]]:
    metric_modules, compute_metrics = _metric_modules_for(job.metrics)
    run_dir = Path(job.output_dir).expanduser()
    try:
        solutions, config, manifest = load_solutions(str(run_dir))
    except FileNotFoundError:
        return []

    rows: list[dict[str, Any]] = []
    config_columns = _config_columns(job.config)
    stage_offset_seconds = 0.0
    for stage, stage_solution in zip(manifest.get("stages", []), solutions):
        stage_dir = run_dir / str(stage.get("path", ""))
        progress_path = _stage_progress_path(stage_dir, stage)
        if progress_path is None or not progress_path.exists():
            stage_offset_seconds += _stage_wall_time(stage, stage_solution)
            continue

        progress_rows = _load_progress_rows(progress_path)
        max_elapsed_seconds = max(
            (float(row.get("elapsed_seconds") or 0.0) for row in progress_rows),
            default=0.0,
        )
        for fallback_index, progress_row in _selected_progress_rows(
            progress_rows,
            sample_every=job.sample_every,
            max_steps=job.max_steps,
        ):
            elapsed_seconds = float(progress_row.get("elapsed_seconds") or 0.0)
            assignment_path = _resolve_progress_path(
                progress_path.parent,
                progress_row.get("assignment_path"),
            )
            if assignment_path is None or not assignment_path.exists():
                continue

            assignment = _load_assignment(assignment_path)
            solution = ZoneSolution(
                problem=stage_solution.problem,
                assignment=assignment,
                status="FEASIBLE",
                objective=progress_row.get("objective"),
                wall_time=elapsed_seconds,
                metadata={
                    **_progress_solution_metadata(stage_solution.metadata),
                    "solver_progress_path": str(progress_path),
                    "solver_progress_solution_index": progress_row.get(
                        "solution_index", fallback_index
                    ),
                },
            )
            metric_values = {}
            if compute_metrics:
                kwargs = {"strict": job.strict_metrics}
                if metric_modules is not None:
                    kwargs["modules"] = metric_modules
                metric_values = MetricsCalculator(
                    solution,
                    config=config,
                    **kwargs,
                ).compute().metrics

            row = {
                "task_number": job.task_number,
                "explicit_task_number": job.explicit_task_number,
                "explicit_task_index": job.explicit_task_number - 1,
                "task_id": manifest.get("task_id", job.task_id),
                "config_hash": manifest.get("config_hash", job.config_hash),
                "run_dir": str(run_dir),
                "stage_name": stage.get("name"),
                "stage_index": stage.get("index"),
                "stage_path": stage.get("path"),
                "level": stage.get("level", stage_solution.level.name),
                "solution_index": progress_row.get("solution_index", fallback_index),
                "iteration": progress_row.get("iteration"),
                "objective": progress_row.get("objective"),
                "elapsed_seconds": elapsed_seconds,
                "time_seconds": stage_offset_seconds + elapsed_seconds,
                "progress_path": str(progress_path),
                "assignment_path": str(assignment_path),
            }
            row.update(config_columns)
            row.update(metric_values)
            rows.append(row)

        stage_offset_seconds += _stage_wall_time(
            stage,
            stage_solution,
            fallback=max_elapsed_seconds,
        )
    return rows


def _resolve_metric_identifier(metric: str) -> str:
    if metric in METRIC_BY_NAME:
        return METRIC_BY_NAME[metric].column
    if metric in METRIC_BY_COLUMN:
        return metric
    resolved = resolve_metric_identifiers([metric])
    if len(resolved) == 1:
        return resolved[0]
    return metric


def _metric_modules_for(
    metrics: Sequence[str] | None,
) -> tuple[tuple[Any, ...] | None, bool]:
    """Return modules needed for selected metrics.

    ``None`` modules means compute all default modules. ``False`` means no metric
    computation is needed because all requested fields are base progress columns.
    """

    if metrics is None:
        return None, True

    modules = []
    for raw_metric in metrics:
        metric = _resolve_metric_identifier(str(raw_metric))
        if metric in BASE_PROGRESS_COLUMNS or metric.startswith("config_"):
            continue
        spec = METRIC_BY_COLUMN.get(metric)
        if spec is not None:
            module = CATEGORY_MODULES.get(spec.category)
        elif _is_dynamic_program_metric(metric):
            module = programs.compute
        else:
            return None, True
        if module is None:
            return None, True
        if module not in modules:
            modules.append(module)
    return tuple(modules), bool(modules)


def _is_dynamic_program_metric(metric: str) -> bool:
    return metric.startswith("avg_") and metric.endswith("_per_zone")


def _drop_upper_outliers(
    df: pd.DataFrame,
    metric: str,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return df

    parts = []
    for _key, group_df in df.groupby(list(group_columns), sort=False, dropna=False):
        threshold = _upper_outlier_threshold(group_df[metric])
        if threshold is None:
            parts.append(group_df)
            continue

        filtered = group_df[group_df[metric] <= threshold]
        parts.append(filtered if not filtered.empty else group_df)

    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, axis=0).sort_index(kind="stable")


def _upper_outlier_threshold(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 3:
        return None

    median = float(numeric.median())
    mad = float((numeric - median).abs().median())
    if mad > 0:
        return median + OUTLIER_MAD_MULTIPLIER * 1.4826 * mad

    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    if iqr > 0:
        return q3 + OUTLIER_IQR_MULTIPLIER * iqr
    return None


def _stage_progress_path(stage_dir: Path, stage: Mapping[str, Any]) -> Path | None:
    metadata = stage.get("metadata") or {}
    return _resolve_progress_path(stage_dir, metadata.get("solver_progress_path"))


def _explicit_task_count(sweep: SimulationSweep) -> int:
    return len(sweep.tasks) if sweep.tasks else 1


def _config_centroids_types(config_path: str | Path) -> list[str]:
    sweep = SimulationSweep.from_yaml(str(config_path))
    centroids_types: list[str] = []
    seen: set[str] = set()
    for task in sweep.generate_tasks():
        key = _centroids_type_key(task.config.get("centroids_type"))
        if key and key not in seen:
            centroids_types.append(key)
            seen.add(key)
    return centroids_types


def _ordered_centroids_types(
    df: pd.DataFrame,
    centroids_types: Sequence[str] | None,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        key = _centroids_type_key(value)
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)

    for value in centroids_types or []:
        add(value)
    if "config_centroids_type" in df.columns:
        for value in df["config_centroids_type"].dropna().tolist():
            add(value)
    return ordered


def _centroids_type_key(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _centroids_type_output_path(output_path: str | Path, centroids_type: str) -> Path:
    output = Path(output_path).expanduser()
    suffix = output.suffix or ".png"
    return output.with_name(
        f"{output.stem}_{_safe_filename(str(centroids_type))}{suffix}"
    )


def _resolve_progress_path(base_dir: Path, path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    return path if path.is_absolute() else base_dir / path


def _load_progress_rows(progress_path: Path) -> list[dict[str, Any]]:
    rows = []
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _selected_progress_rows(
    rows: Sequence[dict[str, Any]],
    *,
    sample_every: int = 1,
    max_steps: int | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    if not rows:
        return []
    last_idx = len(rows) - 1
    indices = [idx for idx in range(len(rows)) if idx % sample_every == 0]
    if last_idx not in indices:
        indices.append(last_idx)
    indices = sorted(set(indices))

    if max_steps is not None and len(indices) > max_steps:
        if max_steps == 1:
            indices = [last_idx]
        else:
            positions = [
                round(i * (len(indices) - 1) / (max_steps - 1))
                for i in range(max_steps)
            ]
            indices = [indices[position] for position in sorted(set(positions))]
            if last_idx not in indices:
                indices[-1] = last_idx
    return [(idx, rows[idx]) for idx in indices]


def _load_assignment(path: Path) -> dict[int, int]:
    with path.open("r", encoding="utf-8") as f:
        return {int(k): int(v) for k, v in json.load(f).items()}


def _progress_solution_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "choice_utility",
        "choice_model_utility",
        "choice_model_utility_change",
        "choice_cuts_added",
        "choice_cuts_total",
    }
    return {str(k): v for k, v in dict(metadata).items() if k not in excluded}


def _config_columns(config: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in config.items():
        column = f"config_{key}"
        if isinstance(value, (dict, list, tuple, set)):
            payload = sorted(value) if isinstance(value, set) else value
            out[column] = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        else:
            out[column] = value
    return out


def _stage_wall_time(
    stage: Mapping[str, Any],
    solution: ZoneSolution,
    *,
    fallback: float = 0.0,
) -> float:
    value = stage.get("wall_time", solution.wall_time)
    if value is None:
        return float(fallback or 0.0)
    return max(float(value), float(fallback or 0.0))


def _metric_title(metric: str) -> str:
    spec = METRIC_BY_COLUMN.get(metric)
    return spec.display_name if spec else metric.replace("_", " ").title()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "metric"


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return [str(column) for column in df.columns]
    columns = []
    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if not numeric.isna().all():
            columns.append(str(column))
    return columns


if __name__ == "__main__":
    main()

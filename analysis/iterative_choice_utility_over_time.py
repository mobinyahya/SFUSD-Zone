#!/usr/bin/env python3
"""Plot iterative-choice utility trajectories from benchmark outputs.

The first iterative-choice solve has no choice cuts, so pass ``--skip-iteration 0``
if you want to omit it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_CONFIG = PROJECT_ROOT / "benchmark/configs/sweep.iterative_choice.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_OUTPUT_NAME = "iterative_choice_utility_over_time.png"

PRE_CHOICE_COLUMN = "choice_total_preassignment_utility"
LEGACY_POST_CHOICE_COLUMN = "choice_total_mnl_utility"
POST_CHOICE_RE = re.compile(r"^choice_(?P<matching_config>.+)_total_mnl_utility$")

UTILITY_KIND_ORDER = ["model", "pre", "post"]
UTILITY_KIND_LABELS = {
    "model": "Model objective",
    "pre": "Pre-assignment utility",
    "post": "Post-match utility",
}
BASE_COLORS = {
    "model": "#4C78A8",
    "pre": "#F58518",
}
POST_COLORS = {
    "sd": "#E45756",
    "no_reserves": "#72B7B2",
    "soft_reserves": "#54A24B",
    "hard_reserves": "#B279A2",
    "default": "#59A14F",
}
LINESTYLES = {
    "model": "-",
    "pre": "--",
    "post": "-.",
}


def main() -> None:
    args = parse_args()
    sweep_config = load_sweep_config(args.config) if args.config else {}
    results_dir = (args.results_dir or sweep_results_dir(sweep_config)).expanduser()

    result_paths = discover_result_jsons(results_dir)
    if not result_paths:
        raise FileNotFoundError(f"No result.json files found under {results_dir}")

    utility_rows = build_trajectory_table(
        result_paths,
        root=result_root(results_dir),
        skip_iteration=args.skip_iteration,
    )
    trajectories = iterative_trajectory_rows(utility_rows)
    if trajectories.empty:
        raise ValueError(
            "No iterative-choice stages with model, pre-choice, or post-choice "
            "utility data were found after filtering."
        )
    baselines = single_utility_baselines(utility_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = plot_trajectories_by_group(
        trajectories,
        baselines,
        args.output_dir,
        args.output_name,
    )

    print(f"Loaded {trajectories['run_label'].nunique()} iterative-choice run(s).")
    baseline_count = 0 if baselines.empty else len(baselines)
    print(f"Loaded {baseline_count} single-run utility baseline(s).")
    print(f"Plotted {len(trajectories)} utility row(s).")
    print(
        "Centroid types: "
        f"{', '.join(str(value) for value in centroid_order(trajectories))}"
    )
    for output_path in output_paths:
        print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot model, pre-choice, and post-choice utilities over iterative "
            "choice iterations."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Benchmark output directory containing one or more result.json files. "
            "Default: execution.output_dir from --config."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SWEEP_CONFIG,
        help=f"Sweep YAML used to derive defaults. Default: {DEFAULT_SWEEP_CONFIG}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plot. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Plot filename. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--skip-iteration",
        type=int,
        default=-1,
        help="Iteration index to skip. Use -1 to skip nothing. Default: -1",
    )
    return parser.parse_args()


def load_sweep_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find sweep config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sweep_results_dir(sweep_config: dict[str, Any]) -> Path:
    execution = sweep_config.get("execution") or {}
    output_dir = execution.get("output_dir")
    if not output_dir:
        raise ValueError(
            "Sweep config is missing execution.output_dir; pass results_dir instead."
        )
    return Path(str(output_dir))


def discover_result_jsons(results_dir: Path) -> list[Path]:
    if results_dir.is_file():
        if results_dir.name != "result.json":
            raise ValueError(f"Expected a result.json file, got {results_dir}")
        return [results_dir]
    return sorted(results_dir.rglob("result.json"))


def result_root(results_dir: Path) -> Path:
    return results_dir.parent if results_dir.is_file() else results_dir


def build_trajectory_table(
    result_paths: list[Path],
    *,
    root: Path,
    skip_iteration: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        result = load_json(result_path)
        if not is_utility_source_result(result):
            continue

        run_label = run_label_for(result_path, root)
        config = result.get("config") or {}
        benchmark = result.get("benchmark") or {}
        run_id = str(
            benchmark.get("config_hash") or benchmark.get("task_id") or run_label
        )
        strategy = str(config.get("strategy", ""))
        stages = (result.get("run") or {}).get("stages") or []
        for fallback_index, stage in enumerate(stages):
            iteration = stage_iteration(stage, fallback_index)
            if (
                strategy.lower() == "iterative_choice"
                and skip_iteration >= 0
                and iteration == skip_iteration
            ):
                continue

            base_row = {
                "trajectory_run_id": run_id,
                "run_label": run_label,
                "result_path": str(result_path),
                "task_id": benchmark.get("task_id"),
                "config_hash": benchmark.get("config_hash"),
                "centroids_type": config.get("centroids_type"),
                "seed": config.get("seed"),
                "solver": config.get("solver"),
                "choice_model_method": config.get("choice_model_method"),
                "strategy": strategy,
                "iteration": iteration,
                "stage_name": stage.get("name"),
                "status": stage.get("status"),
            }

            add_utility_row(
                rows,
                base_row,
                utility_kind="model",
                matching_config=None,
                value=stage_model_utility(stage),
            )
            add_utility_row(
                rows,
                base_row,
                utility_kind="pre",
                matching_config=None,
                value=stage_pre_choice_utility(stage, result),
            )
            for matching_config, value in stage_post_choice_utilities(stage).items():
                add_utility_row(
                    rows,
                    base_row,
                    utility_kind="post",
                    matching_config=matching_config,
                    value=value,
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(
        [
            "centroids_type",
            "run_label",
            "utility_kind_order",
            "matching_config_label",
            "iteration",
        ],
        inplace=True,
    )
    return df.reset_index(drop=True)


def iterative_trajectory_rows(utility_rows: pd.DataFrame) -> pd.DataFrame:
    if utility_rows.empty:
        return utility_rows
    strategy = utility_rows["strategy"].fillna("").astype(str).str.lower()
    return utility_rows[strategy == "iterative_choice"].copy()


def single_utility_baselines(utility_rows: pd.DataFrame) -> pd.DataFrame:
    if utility_rows.empty:
        return pd.DataFrame()

    strategy = utility_rows["strategy"].fillna("").astype(str).str.lower()
    candidates = utility_rows[
        (strategy == "single")
        & utility_rows["utility_kind"].isin(["pre", "post"])
        & utility_rows["utility"].notna()
        & utility_rows["centroids_type"].notna()
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    post_is_sd = (
        candidates["matching_config"].fillna("").astype(str).str.lower() == "sd"
    )
    candidates = candidates[(candidates["utility_kind"] != "post") | post_is_sd].copy()
    if candidates.empty:
        return pd.DataFrame()

    group_cols = [
        "centroids_type",
        "solver",
        "choice_model_method",
        "utility_kind",
        "utility_kind_order",
        "matching_config",
        "matching_config_label",
    ]
    baselines = (
        candidates.groupby(group_cols, dropna=False)["utility"]
        .agg(
            baseline_max="max",
            baseline_median="median",
            baseline_min="min",
        )
        .reset_index()
    )
    baselines.sort_values(
        [
            "centroids_type",
            "solver",
            "choice_model_method",
            "utility_kind_order",
            "matching_config_label",
        ],
        inplace=True,
    )
    return baselines.reset_index(drop=True)


def add_utility_row(
    rows: list[dict[str, Any]],
    base_row: dict[str, Any],
    *,
    utility_kind: str,
    matching_config: str | None,
    value: Any,
) -> None:
    numeric = to_number(value)
    if numeric is None:
        return

    matching_config_label = format_matching_config(matching_config)
    series_label = UTILITY_KIND_LABELS[utility_kind]
    if utility_kind == "post":
        series_label = f"Post: {matching_config_label}"

    rows.append(
        {
            **base_row,
            "utility_kind": utility_kind,
            "utility_kind_label": UTILITY_KIND_LABELS[utility_kind],
            "utility_kind_order": UTILITY_KIND_ORDER.index(utility_kind),
            "matching_config": matching_config,
            "matching_config_label": matching_config_label,
            "series_label": series_label,
            "utility": numeric,
            "trajectory_id": trajectory_id(
                base_row["trajectory_run_id"], utility_kind, matching_config
            ),
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_utility_source_result(result: dict[str, Any]) -> bool:
    config = result.get("config") or {}
    strategy = str(config.get("strategy", "")).lower()
    if strategy in {"iterative_choice", "single"}:
        return True
    stages = (result.get("run") or {}).get("stages") or []
    return any(
        (stage.get("metadata") or {}).get("choice_iteration") is not None
        for stage in stages
    )


def run_label_for(result_path: Path, root: Path) -> str:
    try:
        return str(result_path.parent.relative_to(root))
    except ValueError:
        return str(result_path.parent)


def stage_iteration(stage: dict[str, Any], fallback_index: int) -> int:
    metadata = stage.get("metadata") or {}
    iteration = metadata.get("choice_iteration")
    if iteration is not None:
        return int(iteration)

    name = str(stage.get("name") or "")
    match = re.search(r"(?:iteration|stage)_(\d+)", name)
    if match:
        return int(match.group(1))

    index = stage.get("index")
    if index is not None:
        return int(index)
    return fallback_index


def stage_model_utility(stage: dict[str, Any]) -> Any:
    return stage.get("objective")


def stage_pre_choice_utility(stage: dict[str, Any], result: dict[str, Any]) -> Any:
    if stage.get(PRE_CHOICE_COLUMN) is not None:
        return stage.get(PRE_CHOICE_COLUMN)
    if stage.get("choice_utility") is not None:
        return stage.get("choice_utility")

    metadata = stage.get("metadata") or {}
    # Iterative stages store the same preassignment metric as choice_utility.
    if metadata.get(PRE_CHOICE_COLUMN) is not None:
        return metadata.get(PRE_CHOICE_COLUMN)
    if metadata.get("choice_utility") is not None:
        return metadata.get("choice_utility")

    if stage.get("name") == (result.get("run") or {}).get("final_stage"):
        return (result.get("metrics") or {}).get(PRE_CHOICE_COLUMN)
    return None


def stage_post_choice_utilities(stage: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metrics in stage_choice_metric_dicts(stage):
        for key, value in metrics.items():
            key = str(key)
            match = POST_CHOICE_RE.match(key)
            if match:
                matching_config = match.group("matching_config")
            elif key == LEGACY_POST_CHOICE_COLUMN:
                matching_config = "default"
            else:
                continue
            numeric = to_number(value)
            if numeric is not None:
                out[matching_config] = numeric
    return out


def stage_choice_metric_dicts(stage: dict[str, Any]) -> list[dict[str, Any]]:
    metric_dicts: list[dict[str, Any]] = []
    metrics = stage.get("choice_metrics_metrics")
    if isinstance(metrics, dict):
        metric_dicts.append(metrics)

    choice_payload = stage.get("choice_metrics") or {}
    metrics = choice_payload.get("metrics") or {}
    if isinstance(metrics, dict):
        metric_dicts.append(metrics)

    # Combined multi-matching outputs also retain each matching run with the
    # legacy unprefixed metric name inside choice_metrics.run.runs.
    runs = (choice_payload.get("run") or {}).get("runs") or {}
    if isinstance(runs, dict):
        for name, run_payload in runs.items():
            run_metrics = (run_payload or {}).get("metrics") or {}
            # if LEGACY_POST_CHOICE_COLUMN in run_metrics:
            #     metric_dicts.append(
            #         {
            #             f"choice_{name}_total_mnl_utility": run_metrics[
            #                 LEGACY_POST_CHOICE_COLUMN
            #             ]
            #         }
            #     )

    return metric_dicts


def to_number(value: Any) -> float | None:
    if value is None:
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def plot_trajectories_by_group(
    trajectories: pd.DataFrame,
    baselines: pd.DataFrame,
    output_dir: Path,
    output_name: str,
) -> list[Path]:
    groups = plot_group_order(trajectories)
    multiple = len(groups) > 1
    output_paths: list[Path] = []
    for centroids_type, solver, choice_model_method in groups:
        plot_df = trajectories[
            (trajectories["centroids_type"] == centroids_type)
            & (trajectories["solver"].fillna("").astype(str) == solver)
            & (
                trajectories["choice_model_method"].fillna("").astype(str)
                == choice_model_method
            )
        ]
        plot_baselines = baselines_for_group(
            baselines,
            centroids_type,
            solver,
            choice_model_method,
        )
        output_path = centroid_output_path(
            output_dir,
            output_name,
            str(centroids_type),
            solver,
            choice_model_method,
            multiple=multiple,
        )
        plot_trajectories(
            plot_df,
            output_path,
            str(centroids_type),
            solver,
            choice_model_method,
            plot_baselines,
        )
        output_paths.append(output_path)
    return output_paths


def plot_trajectories(
    trajectories: pd.DataFrame,
    output_path: Path,
    centroids_type: str,
    solver: str,
    choice_model_method: str,
    baselines: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)

    plot_df = trajectories.dropna(subset=["utility"]).copy()
    if plot_df.empty:
        raise ValueError(f"No utility values to plot for {centroids_type}.")

    run_count = plot_df["run_label"].nunique()
    plotted_any = False
    legend_labels: set[str] = set()
    for _, series_df in plot_df.groupby("trajectory_id", sort=False):
        series_df = series_df.sort_values("iteration")
        first = series_df.iloc[0]
        label = str(first["series_label"])
        legend_label = label if label not in legend_labels else "_nolegend_"
        legend_labels.add(label)
        ax.plot(
            series_df["iteration"],
            series_df["utility"],
            color=series_color(first),
            linestyle=LINESTYLES[str(first["utility_kind"])],
            marker="o",
            markersize=2.5,
            linewidth=1.15,
            alpha=0.52,
            label=legend_label,
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError(
            f"No utility series had numeric values to plot for {centroids_type}."
        )

    for _, baseline in baselines.iterrows():
        color = baseline_color(baseline)
        ax.axhline(
            float(baseline["baseline_median"]),
            color=color,
            linestyle="--",
            linewidth=2.0,
            label="_nolegend_",
            zorder=1,
        )
        ax.axhspan(
            float(baseline["baseline_min"]),
            float(baseline["baseline_max"]),
            color=color,
            alpha=0.14,
            label=baseline_label(baseline),
            zorder=0,
        )

    ax.set_title(
        "Iterative Choice Utilities Over Time: "
        f"{centroids_type}, {format_solver(solver)}, {format_choice_method(choice_model_method)}"
    )
    ax.set_xlabel("Choice iteration")
    ax.set_ylabel("Utility")
    ax.grid(True, alpha=0.35)
    ax.legend(
        title=f"Utility series ({run_count} runs)",
        loc="best",
        frameon=True,
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def centroid_order(trajectories: pd.DataFrame) -> list[Any]:
    values = [value for value in trajectories["centroids_type"].dropna().unique()]
    return sorted(values, key=centroid_sort_key)


def plot_group_order(trajectories: pd.DataFrame) -> list[tuple[Any, str, str]]:
    groups = []
    for row in (
        trajectories[["centroids_type", "solver", "choice_model_method"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        groups.append(
            (
                row.centroids_type,
                "" if pd.isna(row.solver) else str(row.solver),
                ""
                if pd.isna(row.choice_model_method)
                else str(row.choice_model_method),
            )
        )
    return sorted(
        groups, key=lambda item: (centroid_sort_key(item[0]), item[1], item[2])
    )


def baselines_for_group(
    baselines: pd.DataFrame,
    centroids_type: Any,
    solver: str,
    choice_model_method: str,
) -> pd.DataFrame:
    if baselines.empty:
        return baselines
    matches = baselines[
        (baselines["centroids_type"] == centroids_type)
        & (baselines["solver"].fillna("").astype(str) == solver)
        & (
            baselines["choice_model_method"].fillna("").astype(str)
            == choice_model_method
        )
    ]
    return matches.sort_values(["utility_kind_order", "matching_config_label"])


def centroid_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    match = re.match(r"(\d+)-zone", text)
    if match:
        return int(match.group(1)), text
    return 10**9, text


def centroid_output_path(
    output_dir: Path,
    output_name: str,
    centroids_type: str,
    solver: str,
    choice_model_method: str,
    *,
    multiple: bool,
) -> Path:
    output = Path(output_name)
    if not multiple:
        return output_dir / output.name
    suffix = "_".join(
        safe_filename(value)
        for value in [
            centroids_type,
            solver or "solver",
            choice_model_method or "method",
        ]
    )
    return output_dir / f"{output.stem}_{suffix}{output.suffix}"


def series_color(row: pd.Series) -> str:
    utility_kind = str(row["utility_kind"])
    if utility_kind in BASE_COLORS:
        return BASE_COLORS[utility_kind]
    matching_config = str(row.get("matching_config") or "default")
    return POST_COLORS.get(matching_config, "#9D755D")


def baseline_color(row: pd.Series) -> str:
    utility_kind = str(row.get("utility_kind") or "")
    if utility_kind == "pre":
        return BASE_COLORS["pre"]
    matching_config = str(row.get("matching_config") or "default")
    return POST_COLORS.get(matching_config, "#222222")


def format_matching_config(value: str | None) -> str:
    if not value:
        return "Default"
    return str(value).replace("_", " ").title()


def format_solver(value: str | None) -> str:
    if not value:
        return "Unknown solver"
    return (
        str(value).replace("_", " ").upper()
        if value == "mip"
        else str(value).replace("_", " ").title()
    )


def format_choice_method(value: str | None) -> str:
    if not value:
        return "Unknown choice method"
    return f"{str(value).replace('_', ' ').title()} choice model"


def baseline_label(baseline: pd.Series) -> str:
    utility_kind = str(baseline.get("utility_kind") or "post")
    utility_label = "pre-choice" if utility_kind == "pre" else "post-choice"
    matching_config = baseline.get("matching_config")
    if utility_kind == "post" and not pd.isna(matching_config) and matching_config:
        matching_label = (
            "SD"
            if str(matching_config).lower() == "sd"
            else format_matching_config(str(matching_config))
        )
        utility_label = f"{matching_label} {utility_label}"
    return f"Single {utility_label} baseline range"


def trajectory_id(
    run_id: str,
    utility_kind: str,
    matching_config: str | None,
) -> str:
    return f"{run_id}:{utility_kind}:{matching_config or 'none'}"


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "centroids"


if __name__ == "__main__":
    main()

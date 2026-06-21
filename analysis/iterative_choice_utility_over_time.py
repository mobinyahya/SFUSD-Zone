#!/usr/bin/env python3
"""Plot iterative-choice utility trajectories from benchmark result.json files.

The first iterative-choice solve has no choice cuts, so its utility objective is
not comparable to later iterations. By default this script drops iteration 0.
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


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_OUTPUT_NAME = "iterative_choice_utility_over_time.png"

POST_CHOICE_COLUMN = "choice_total_mnl_utility"
UTILITY_SERIES = {
    "model_utility": {
        "label": "Model Utility Objective",
        "description": "Solver objective value",
        "color": "#4C78A8",
    },
    "real_prechoice_utility": {
        "label": "Real Pre-Choice Utility",
        "description": "Choice model evaluation before matching",
        "color": "#F58518",
    },
    "real_postchoice_utility": {
        "label": "Real Post-Choice Utility",
        "description": "Matched assignment utility",
        "color": "#54A24B",
    },
}


def main() -> None:
    args = parse_args()
    result_paths = discover_result_jsons(args.results_dir)
    if not result_paths:
        raise FileNotFoundError(f"No result.json files found under {args.results_dir}")

    trajectories = build_trajectory_table(
        result_paths,
        root=args.results_dir,
        skip_iteration=args.skip_iteration,
    )
    if trajectories.empty:
        raise ValueError(
            "No iterative-choice stages with utility data were found after filtering."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    plot_trajectories(trajectories, output_path)

    print(f"Loaded {trajectories['run_label'].nunique()} run(s).")
    print(f"Plotted {len(trajectories)} iteration row(s).")
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
        type=Path,
        help="Benchmark output directory containing one or more result.json files.",
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
        default=0,
        help="Iteration index to skip. Use -1 to skip nothing. Default: 0",
    )
    return parser.parse_args()


def discover_result_jsons(results_dir: Path) -> list[Path]:
    if results_dir.is_file():
        if results_dir.name != "result.json":
            raise ValueError(f"Expected a result.json file, got {results_dir}")
        return [results_dir]
    return sorted(results_dir.rglob("result.json"))


def build_trajectory_table(
    result_paths: list[Path],
    *,
    root: Path,
    skip_iteration: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        result = load_json(result_path)
        if not is_iterative_choice_result(result):
            continue

        run_label = run_label_for(result_path, root)
        config = result.get("config") or {}
        benchmark = result.get("benchmark") or {}
        stages = (result.get("run") or {}).get("stages") or []
        for fallback_index, stage in enumerate(stages):
            iteration = stage_iteration(stage, fallback_index)
            if skip_iteration >= 0 and iteration == skip_iteration:
                continue

            row = {
                "run_label": run_label,
                "result_path": str(result_path),
                "task_id": benchmark.get("task_id"),
                "config_hash": benchmark.get("config_hash"),
                "centroids_type": config.get("centroids_type"),
                "seed": config.get("seed"),
                "iteration": iteration,
                "stage_name": stage.get("name"),
                "status": stage.get("status"),
                "model_utility": to_number(stage.get("objective")),
                "real_prechoice_utility": to_number(stage_choice_utility(stage)),
                "real_postchoice_utility": to_number(stage_post_choice_utility(stage)),
            }
            if any(pd.notna(row[column]) for column in UTILITY_SERIES):
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["run_label", "iteration"], inplace=True)
    return df.reset_index(drop=True)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_iterative_choice_result(result: dict[str, Any]) -> bool:
    config = result.get("config") or {}
    strategy = str(config.get("strategy", "")).lower()
    if strategy == "iterative_choice":
        return True
    stages = (result.get("run") or {}).get("stages") or []
    return any(stage_choice_utility(stage) is not None for stage in stages)


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


def stage_choice_utility(stage: dict[str, Any]) -> Any:
    if stage.get("choice_utility") is not None:
        return stage.get("choice_utility")
    metadata = stage.get("metadata") or {}
    return metadata.get("choice_utility")


def stage_post_choice_utility(stage: dict[str, Any]) -> Any:
    metrics = stage.get("choice_metrics_metrics") or {}
    if POST_CHOICE_COLUMN in metrics:
        return metrics[POST_CHOICE_COLUMN]

    choice_payload = stage.get("choice_metrics") or {}
    metrics = choice_payload.get("metrics") or {}
    return metrics.get(POST_CHOICE_COLUMN)


def to_number(value: Any) -> float | None:
    if value is None:
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def plot_trajectories(trajectories: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    run_count = trajectories["run_label"].nunique()
    plotted_any = False
    for column, meta in UTILITY_SERIES.items():
        series_df = trajectories.dropna(subset=[column]).copy()
        if series_df.empty:
            continue

        mean_df = (
            series_df.groupby("iteration", as_index=False)[column]
            .mean()
            .sort_values("iteration")
        )
        ax.plot(
            mean_df["iteration"],
            mean_df[column],
            color=meta["color"],
            marker="o",
            linewidth=2.6,
            label=meta["label"],
        )
        plotted_any = True

    if not plotted_any:
        raise ValueError("No utility series had numeric values to plot.")

    title_suffix = " (mean across runs)" if run_count > 1 else ""
    ax.set_title(f"Iterative Choice Utility Over Time{title_suffix}")
    ax.set_xlabel("Choice iteration")
    ax.set_ylabel("Utility")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

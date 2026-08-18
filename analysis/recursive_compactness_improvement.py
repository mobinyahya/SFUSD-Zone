#!/usr/bin/env python3
"""Plot direct compactness values from recursive benchmark runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_DIR = Path("/soalnas/share/data/school_choice/local_runs/full_recursive_sweep")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

SINGLE_APPROACH_LABEL = "Single Block_0"

METRICS = {
    "avg_polsby_popper_score": {
        "label": "Polsby-Popper",
        "direction": "higher is better",
    },
    "avg_reock_score": {
        "label": "Reock",
        "direction": "higher is better",
    },
    "fractional_cut_edges": {
        "label": "Fractional cut edges",
        "direction": "lower is better",
    },
}


def main() -> None:
    args = parse_args()
    summary_path = args.results_dir / "summary.csv"
    output_dir = args.output_dir

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_path)
    compactness_values = build_compactness_table(summary_df)
    if compactness_values.empty:
        raise ValueError(
            "No completed final Block_0 runs with usable compactness metrics were found."
        )

    zone_order = sorted(compactness_values["zone_count"].unique())
    overall_path = output_dir / "recursive_compactness_values_overall.png"

    plot_overall(compactness_values, overall_path)
    zone_paths = plot_zone_count_files(compactness_values, output_dir, zone_order)

    print(f"Loaded solution rows: {compactness_values['solution_id'].nunique()}")
    print(f"Plotted zone counts: {', '.join(str(zone) for zone in zone_order)}")
    print(f"Wrote {overall_path}")
    for zone_path in zone_paths:
        print(f"Wrote {zone_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot direct compactness values for recursive benchmark runs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Benchmark result directory containing summary.csv. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def build_compactness_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "config_strategy",
        "levels",
        "config_solve_time_limits",
        "config_centroids_type",
        "final_stage",
        "final_stage_index",
        "num_stages",
        "status",
        *METRICS.keys(),
    }
    missing_columns = sorted(required_columns - set(summary_df.columns))
    if missing_columns:
        raise ValueError(f"summary.csv is missing required columns: {missing_columns}")

    df = summary_df.copy()
    metric_columns = list(METRICS)
    for column in metric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["zone_count"] = df.apply(extract_zone_count, axis=1)
    df["levels"] = df["levels"].fillna("")
    df["time_limit_key"] = df["config_solve_time_limits"].map(time_limit_key)
    df["approach"] = df.apply(approach_label, axis=1)

    # The unsuffixed metric columns in summary.csv are final-stage metrics.
    # Stage-specific columns are intentionally ignored here.
    usable_metric_rows = df[metric_columns].notna().all(axis=1) & (
        df[metric_columns] >= 0
    ).all(axis=1)
    completed_final_rows = final_block0_rows(df)
    df = df[
        usable_metric_rows
        & completed_final_rows
        & df["zone_count"].notna()
        & df["approach"].notna()
    ].copy()

    rows: list[dict[str, object]] = []
    for solution_id, row in df.reset_index(drop=True).iterrows():
        for metric, meta in METRICS.items():
            rows.append(
                {
                    "solution_id": int(solution_id),
                    "approach": row["approach"],
                    "zone_count": int(row["zone_count"]),
                    "metric": metric,
                    "metric_label": metric_label(meta),
                    "value": float(row[metric]),
                }
            )

    return pd.DataFrame(rows)


def final_block0_rows(df: pd.DataFrame) -> pd.Series:
    final_stage_index = pd.to_numeric(df["final_stage_index"], errors="coerce")
    num_stages = pd.to_numeric(df["num_stages"], errors="coerce")
    return (
        df["final_stage"].fillna("").astype(str).str.endswith("Block_0")
        & final_stage_index.notna()
        & num_stages.notna()
        & (final_stage_index == num_stages - 1)
    )


def extract_zone_count(row: pd.Series) -> int | None:
    centroids_type = str(row.get("config_centroids_type", ""))
    match = re.match(r"(\d+)-zone", centroids_type)
    if match:
        return int(match.group(1))

    num_zones = pd.to_numeric(row.get("num_zones"), errors="coerce")
    if pd.notna(num_zones) and num_zones > 0:
        return int(num_zones)
    return None


def time_limit_key(value: object) -> str:
    values = re.findall(r"\d+", str(value))
    return "-".join(values)


def recursive_approach_label(row: pd.Series) -> str | None:
    levels = str(row.get("levels", "")).strip()
    if not levels:
        return None

    level_label = levels.replace("-", " -> ")
    time_limit_key = str(row.get("time_limit_key", "")).strip()
    if not time_limit_key:
        return level_label
    return f"{level_label} ({time_limit_key.replace('-', '/')})"


def approach_label(row: pd.Series) -> str | None:
    strategy = str(row.get("config_strategy", "")).lower()
    levels = str(row.get("levels", ""))
    if strategy == "single" and levels == "Block_0":
        return SINGLE_APPROACH_LABEL
    if strategy == "recursive":
        return recursive_approach_label(row)
    return None


def metric_label(meta: dict[str, str]) -> str:
    return f"{meta['label']}\n({meta['direction']})"


def plot_overall(compactness_values: pd.DataFrame, output_path: Path) -> None:
    plot_metric_bars(
        compactness_values,
        output_path,
        "Compactness Metrics Across All Solutions",
    )


def plot_zone_count_files(
    compactness_values: pd.DataFrame, output_dir: Path, zone_order: list[int]
) -> list[Path]:
    output_paths: list[Path] = []
    for zone_count in zone_order:
        zone_values = compactness_values[compactness_values["zone_count"] == zone_count]
        if zone_values.empty:
            continue

        output_path = (
            output_dir / f"recursive_compactness_values_{zone_count}_zones.png"
        )
        plot_metric_bars(
            zone_values,
            output_path,
            f"Compactness Metrics for {zone_count} Zones",
        )
        output_paths.append(output_path)
    return output_paths


def plot_metric_bars(
    compactness_values: pd.DataFrame, output_path: Path, title: str
) -> None:
    metric_order = [metric_label(meta) for meta in METRICS.values()]
    approach_order = present_approach_order(compactness_values)
    plot_df = compactness_values.groupby(["approach", "metric_label"], as_index=False)[
        "value"
    ].mean()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="metric_label",
        y="value",
        hue="approach",
        order=metric_order,
        hue_order=approach_order,
        ax=ax,
    )
    format_axis(ax, title)
    ax.legend(title="Approach")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def present_approach_order(compactness_values: pd.DataFrame) -> list[str]:
    present = set(compactness_values["approach"].dropna())
    recursive = sorted(
        (approach for approach in present if approach != SINGLE_APPROACH_LABEL),
        key=approach_sort_key,
    )
    if SINGLE_APPROACH_LABEL in present:
        return [SINGLE_APPROACH_LABEL, *recursive]
    return recursive


def approach_sort_key(approach: str) -> tuple[int, list[int], list[int], str]:
    levels = [int(value) for value in re.findall(r"Block_(\d+)", approach)]
    time_match = re.search(r"\(([^)]*)\)", approach)
    time_limits = (
        [int(value) for value in re.findall(r"\d+", time_match.group(1))]
        if time_match
        else []
    )
    return (len(levels), levels, time_limits, approach)


def format_axis(ax: plt.Axes, title: str) -> None:
    if title:
        ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Mean direct metric value")
    ax.tick_params(axis="x", rotation=20)


if __name__ == "__main__":
    main()

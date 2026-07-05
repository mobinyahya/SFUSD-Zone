#!/usr/bin/env python3
"""Plot solver comparison metrics from benchmark summary.csv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_DIR = Path("/share/data/school_choice/local_runs/solver_comparison")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

SUCCESS_STATUSES = {"FEASIBLE", "OPTIMAL"}
SOLVER_LABELS = {
    "cp_bool": "CP Bool",
    "cp_int": "CP Int",
    "mip": "MIP",
    "recom": "ReCom",
}
SOLVER_ORDER = ["CP Bool", "CP Int", "MIP", "ReCom"]

METRICS = {
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
    solver_values = build_solver_metric_table(summary_df)
    if solver_values.empty:
        raise ValueError(
            "No completed feasible final Block_0 runs with usable solver metrics were found."
        )

    zone_order = sorted(solver_values["zone_count"].unique())
    overall_path = output_dir / "solver_comparison_metrics_overall.png"

    plot_overall(solver_values, overall_path)
    zone_paths = plot_zone_count_files(solver_values, output_dir, zone_order)

    print(f"Loaded feasible solution rows: {solver_values['solution_id'].nunique()}")
    print(f"Plotted solvers: {', '.join(present_solver_order(solver_values))}")
    print(f"Plotted zone counts: {', '.join(str(zone) for zone in zone_order)}")
    print_status_counts(summary_df)
    print(f"Wrote {overall_path}")
    for zone_path in zone_paths:
        print(f"Wrote {zone_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot solver comparison metrics for benchmark runs."
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


def build_solver_metric_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "config_solver",
        "config_centroids_type",
        "final_stage",
        "final_stage_index",
        "num_stages",
        "status",
        "fractional_cut_edges",
    }
    missing_columns = sorted(required_columns - set(summary_df.columns))
    if missing_columns:
        raise ValueError(f"summary.csv is missing required columns: {missing_columns}")

    df = summary_df.copy()
    df["fractional_cut_edges"] = pd.to_numeric(
        df["fractional_cut_edges"], errors="coerce"
    )

    df["zone_count"] = df.apply(extract_zone_count, axis=1)
    df["solver"] = df["config_solver"].map(solver_label)

    completed_final_rows = final_block0_rows(df)
    successful_rows = (
        df["status"].fillna("").astype(str).str.upper().isin(SUCCESS_STATUSES)
    )
    df = df[
        completed_final_rows
        & successful_rows
        & df["zone_count"].notna()
        & df["solver"].notna()
    ].copy()

    rows: list[dict[str, object]] = []
    for solution_id, row in df.reset_index(drop=True).iterrows():
        for metric, meta in METRICS.items():
            value = row[metric]
            if pd.isna(value) or value < 0:
                continue
            rows.append(
                {
                    "solution_id": int(solution_id),
                    "centroids_type": row["config_centroids_type"],
                    "solver": row["solver"],
                    "zone_count": int(row["zone_count"]),
                    "metric": metric,
                    "metric_label": metric_label(meta),
                    "value": float(value),
                }
            )

    return pd.DataFrame(rows)


def final_block0_rows(df: pd.DataFrame) -> pd.Series:
    final_stage_index = pd.to_numeric(df["final_stage_index"], errors="coerce")
    num_stages = pd.to_numeric(df["num_stages"], errors="coerce")
    final_stage = df["final_stage"].fillna("").astype(str)
    return (
        (final_stage.str.endswith("Block_0") | final_stage.str.endswith("BlockGroup_0"))
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


def solver_label(value: object) -> str | None:
    if pd.isna(value):
        return None

    key = str(value).strip().lower()
    if not key:
        return None
    return SOLVER_LABELS.get(key, key.replace("_", " ").title())


def metric_label(meta: dict[str, str]) -> str:
    return f"{meta['label']}\n({meta['direction']})"


def plot_overall(solver_values: pd.DataFrame, output_path: Path) -> None:
    plot_metric_bars(
        solver_values,
        output_path,
        "Solver Comparison Across All Feasible Solutions",
    )


def plot_zone_count_files(
    solver_values: pd.DataFrame, output_dir: Path, zone_order: list[int]
) -> list[Path]:
    output_paths: list[Path] = []
    for zone_count in zone_order:
        zone_values = solver_values[solver_values["zone_count"] == zone_count]
        if zone_values.empty:
            continue

        output_path = output_dir / f"solver_comparison_metrics_{zone_count}_zones.png"
        plot_metric_bars(
            zone_values,
            output_path,
            f"Solver Comparison for {zone_count} Zones",
        )
        output_paths.append(output_path)
    return output_paths


def plot_metric_bars(
    solver_values: pd.DataFrame, output_path: Path, title: str
) -> None:
    metric_order = [metric_label(meta) for meta in METRICS.values()]
    solver_order = present_solver_order(solver_values)
    plot_df = (
        solver_values.groupby(["solver", "metric_label"], as_index=False)["value"]
        .mean()
        .sort_values(["metric_label", "solver"])
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="metric_label",
        y="value",
        hue="solver",
        order=metric_order,
        hue_order=solver_order,
        ax=ax,
    )
    format_axis(ax, title)
    ax.legend(title="Solver")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def present_solver_order(solver_values: pd.DataFrame) -> list[str]:
    present = set(solver_values["solver"].dropna())
    ordered = [solver for solver in SOLVER_ORDER if solver in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def format_axis(ax: plt.Axes, title: str) -> None:
    if title:
        ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Mean metric value")
    ax.tick_params(axis="x", rotation=15)


def print_status_counts(summary_df: pd.DataFrame) -> None:
    if "status" not in summary_df:
        return

    counts = (
        summary_df["status"].fillna("MISSING").astype(str).value_counts().sort_index()
    )
    formatted_counts = ", ".join(
        f"{status}={count}" for status, count in counts.items()
    )
    print(f"Summary status counts: {formatted_counts}")


if __name__ == "__main__":
    main()

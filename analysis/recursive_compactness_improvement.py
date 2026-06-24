#!/usr/bin/env python3
"""Plot compactness improvements from recursive benchmark runs.

The script compares each recursive run against the matching single Block_0 run
with the same non-strategy benchmark configuration. Improvements are reported as
percent changes, sign-adjusted so that positive values always mean better
compactness.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_DIR = Path("/share/data/school_choice/local_runs/full_recursive_sweep")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

APPROACH_LABELS = {
    ("Block_1-Block_0", "300-300"): "Block_1 -> Block_0 (300/300)",
    (
        "Block_2-Block_1-Block_0",
        "200-200-200",
    ): "Block_2 -> Block_1 -> Block_0 (200/200/200)",
    ("Block_1-Block_0", "200-400"): "Block_1 -> Block_0 (200/400)",
    (
        "Block_2-Block_1-Block_0",
        "100-100-400",
    ): "Block_2 -> Block_1 -> Block_0 (100/100/400)",
}

METRICS = {
    "avg_polsby_popper_score": {
        "label": "Polsby-Popper",
        "direction": "higher",
    },
    "cut_edges": {
        "label": "Cut edges",
        "direction": "lower",
    },
    "normalized_cut_edges": {
        "label": "Normalized cut edges",
        "direction": "lower",
    },
    "avg_reock_score": {
        "label": "Reock",
        "direction": "higher",
    },
}

EXCLUDED_MATCH_COLUMNS = {
    "config_hash",
    "config_strategy",
    "config_levels",
    "config_solve_time_limits",
}


def main() -> None:
    args = parse_args()
    summary_path = args.results_dir / "summary.csv"
    output_dir = args.output_dir

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {summary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_path)
    zone_order = extract_zone_order(summary_df)
    improvements = build_improvement_table(summary_df)
    if improvements.empty:
        raise ValueError("No matched recursive/single runs with usable metrics were found.")

    overall_path = output_dir / "recursive_compactness_improvement_overall.png"
    by_zones_path = output_dir / "recursive_compactness_improvement_by_zones.png"

    plot_overall(improvements, overall_path)
    plot_by_zones(improvements, by_zones_path, zone_order)

    print(f"Matched single-baseline configs: {improvements['pair_id'].nunique()}")
    print(
        "Matched recursive comparisons: "
        f"{improvements[['pair_id', 'recursive_approach']].drop_duplicates().shape[0]}"
    )
    print(f"Wrote {overall_path}")
    print(f"Wrote {by_zones_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot compactness improvements for recursive benchmark runs."
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


def build_improvement_table(summary_df: pd.DataFrame) -> pd.DataFrame:
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
    df["recursive_approach"] = df.apply(recursive_approach_label, axis=1)

    # The unsuffixed metric columns in summary.csv are final-stage metrics.
    # Stage-specific columns are intentionally ignored here.
    usable_metric_rows = df[metric_columns].notna().all(axis=1) & (df[metric_columns] > 0).all(axis=1)
    completed_final_rows = final_block0_rows(df)
    df = df[
        usable_metric_rows & completed_final_rows & df["zone_count"].notna()
    ].copy()

    key_columns = match_columns(df)
    single = df[(df["config_strategy"] == "single") & (df["levels"] == "Block_0")]
    recursive = df[
        (df["config_strategy"] == "recursive") & df["recursive_approach"].notna()
    ]

    single = (
        single.groupby(key_columns, dropna=False)[metric_columns]
        .mean()
        .reset_index()
    )
    recursive = (
        recursive.groupby(key_columns + ["recursive_approach", "zone_count"], dropna=False)[
            metric_columns
        ]
        .mean()
        .reset_index()
    )

    paired = recursive.merge(
        single,
        on=key_columns,
        suffixes=("_recursive", "_single"),
        how="inner",
        validate="many_to_one",
    )
    paired["pair_id"] = paired.groupby(key_columns, dropna=False).ngroup()

    rows: list[dict[str, object]] = []
    for _, row in paired.reset_index(drop=True).iterrows():
        for metric, meta in METRICS.items():
            single_value = row[f"{metric}_single"]
            recursive_value = row[f"{metric}_recursive"]
            if pd.isna(single_value) or single_value <= 0 or pd.isna(recursive_value):
                continue

            if meta["direction"] == "lower":
                improvement_pct = (single_value - recursive_value) / single_value * 100
                raw_change = single_value - recursive_value
            else:
                improvement_pct = (recursive_value - single_value) / single_value * 100
                raw_change = recursive_value - single_value

            rows.append(
                {
                    "pair_id": int(row["pair_id"]),
                    "recursive_approach": row["recursive_approach"],
                    "zone_count": int(row["zone_count"]),
                    "metric": metric,
                    "metric_label": meta["label"],
                    "single_value": single_value,
                    "recursive_value": recursive_value,
                    "raw_change": raw_change,
                    "improvement_pct": improvement_pct,
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


def match_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in df.columns
        if column.startswith("config_") and column not in EXCLUDED_MATCH_COLUMNS
    ]
    if "config_centroids_type" not in columns:
        raise ValueError("config_centroids_type is required to pair runs.")
    return columns


def extract_zone_count(row: pd.Series) -> int | None:
    centroids_type = str(row.get("config_centroids_type", ""))
    match = re.match(r"(\d+)-zone", centroids_type)
    if match:
        return int(match.group(1))

    num_zones = pd.to_numeric(row.get("num_zones"), errors="coerce")
    if pd.notna(num_zones) and num_zones > 0:
        return int(num_zones)
    return None


def extract_zone_order(summary_df: pd.DataFrame) -> list[int]:
    zone_counts = summary_df.apply(extract_zone_count, axis=1).dropna().astype(int)
    return [int(zone_count) for zone_count in sorted(zone_counts.unique())]


def time_limit_key(value: object) -> str:
    values = re.findall(r"\d+", str(value))
    return "-".join(values)


def recursive_approach_label(row: pd.Series) -> str | None:
    return APPROACH_LABELS.get((row["levels"], row["time_limit_key"]))


def plot_overall(improvements: pd.DataFrame, output_path: Path) -> None:
    metric_order = [meta["label"] for meta in METRICS.values()]
    approach_order = list(APPROACH_LABELS.values())
    overall = (
        improvements.groupby(["recursive_approach", "metric_label"], as_index=False)[
            "improvement_pct"
        ]
        .mean()
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=overall,
        x="metric_label",
        y="improvement_pct",
        hue="recursive_approach",
        order=metric_order,
        hue_order=approach_order,
        ax=ax,
    )
    format_axis(ax, "Average Compactness Improvement From Recursion")
    ax.legend(title="Recursive approach")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_by_zones(
    improvements: pd.DataFrame, output_path: Path, zone_order: list[int]
) -> None:
    metric_order = [meta["label"] for meta in METRICS.values()]
    approach_order = list(APPROACH_LABELS.values())
    by_zones = (
        improvements.groupby(
            ["zone_count", "recursive_approach", "metric_label"], as_index=False
        )["improvement_pct"]
        .mean()
    )

    sns.set_theme(style="whitegrid")
    grid = sns.catplot(
        data=by_zones,
        kind="bar",
        x="metric_label",
        y="improvement_pct",
        hue="recursive_approach",
        col="zone_count",
        order=metric_order,
        hue_order=approach_order,
        col_order=zone_order,
        height=4.5,
        aspect=1.05,
        sharey=True,
    )
    grid.set_axis_labels("", "Mean improvement vs single run (%)")
    grid.set_titles("{col_name} zones")
    grid.fig.suptitle("Compactness Improvement From Recursion by Number of Zones", y=1.08)

    for ax in grid.axes.flat:
        format_axis(ax, None)
        ax.tick_params(axis="x", rotation=20)
        if not ax.patches:
            ax.text(
                0.5,
                0.5,
                "No matched\nfinal Block_0 baseline",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.35",
            )

    if grid.legend is not None:
        grid.legend.set_title("Recursive approach")

    grid.fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(grid.fig)


def format_axis(ax: plt.Axes, title: str | None) -> None:
    if title:
        ax.set_title(title)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Mean improvement vs single run (%)")
    ax.tick_params(axis="x", rotation=20)


if __name__ == "__main__":
    main()

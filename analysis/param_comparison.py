#!/usr/bin/env python3
"""Plot CP-SAT parameter comparison metrics from benchmark summary.csv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_CONFIG = (
    PROJECT_ROOT / "Zone_Generation/benchmark/configs/sweep.test_params.yaml"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plots"

SUCCESS_STATUSES = {"FEASIBLE", "OPTIMAL"}
METRIC = "normalized_cut_edges"
REFERENCE_SOLVER = "CP Bool"
REFERENCE_VARIANT = "Baseline"
REFERENCE_SERIES = f"{REFERENCE_SOLVER} {REFERENCE_VARIANT}"
METRIC_LABEL = "% change in normalized cut edges\n(vs CP Bool Baseline)"
MATCH_COLUMNS = [
    "config_centroids_type",
    "config_frl_dev",
    "config_racial_dev",
    "config_seed",
    "config_overage",
    "config_shortage",
    "config_capacity_scenario",
    "config_max_distance",
    "config_solve_time_limits",
    "config_levels",
]

VARIANT_LABELS = {
    "baseline": "Baseline",
    "probing": "Probing",
    "search_strategy": "Search Strategy",
    "linearization": "Linearization",
}
VARIANT_ORDER = [
    VARIANT_LABELS["baseline"],
    VARIANT_LABELS["probing"],
    VARIANT_LABELS["search_strategy"],
    VARIANT_LABELS["linearization"],
]
SOLVER_LABELS = {
    "cp_int": "CP Int",
    "cp_bool": "CP Bool",
}
SOLVER_ORDER = ["CP Int", "CP Bool"]
SERIES_ORDER = [
    f"{solver} {variant}"
    for solver in SOLVER_ORDER
    for variant in VARIANT_ORDER
    if f"{solver} {variant}" != REFERENCE_SERIES
]


def main() -> None:
    args = parse_args()
    sweep_config = load_sweep_config(args.config)
    results_dir = args.results_dir or sweep_results_dir(sweep_config)
    summary_path = results_dir / args.summary_csv

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {summary_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_path)
    zone_order = expected_zone_order(sweep_config, summary_df)
    comparison_values = build_param_comparison_table(summary_df)
    if comparison_values.empty:
        raise ValueError(
            "No completed feasible final Block_0/BlockGroup_0 runs with usable "
            f"{METRIC} values and matching {REFERENCE_SERIES} rows were found."
        )

    overall_path = (
        args.output_dir / "param_comparison_normalized_cut_edges_pct_change_overall.png"
    )
    by_zone_path = (
        args.output_dir / "param_comparison_normalized_cut_edges_pct_change_by_zone.png"
    )

    plot_overall(comparison_values, overall_path)
    plot_by_zone(comparison_values, by_zone_path, zone_order)

    print(
        f"Loaded matched comparison rows: {comparison_values['solution_id'].nunique()}"
    )
    print(f"Plotted run types: {', '.join(present_series_order(comparison_values))}")
    print(f"Expected zone counts: {', '.join(str(zone) for zone in zone_order)}")
    print(
        "Present zone counts: "
        f"{', '.join(str(zone) for zone in sorted(comparison_values['zone_count'].unique()))}"
    )
    print_status_counts(summary_df)
    print(f"Wrote {overall_path}")
    print(f"Wrote {by_zone_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot normalized cut-edge comparisons for CP-SAT parameter runs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SWEEP_CONFIG,
        help=f"Sweep YAML used to derive defaults. Default: {DEFAULT_SWEEP_CONFIG}",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Benchmark result directory containing summary.csv. Default: execution.output_dir from --config",
    )
    parser.add_argument(
        "--summary-csv",
        default="summary.csv",
        help="Summary CSV filename under --results-dir. Default: summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plots. Default: {DEFAULT_OUTPUT_DIR}",
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
            "Sweep config is missing execution.output_dir; pass --results-dir instead."
        )
    return Path(str(output_dir)).expanduser()


def expected_zone_order(
    sweep_config: dict[str, Any], summary_df: pd.DataFrame
) -> list[int]:
    sweep = sweep_config.get("sweep") or {}
    centroids_types = normalize_list(sweep.get("centroids_type"))
    expected = [
        zone for value in centroids_types if (zone := zone_count_from_value(value))
    ]

    present: list[int] = []
    if "config_centroids_type" in summary_df:
        for value in summary_df["config_centroids_type"].dropna().unique():
            zone = zone_count_from_value(value)
            if zone is not None:
                present.append(zone)
    if "num_zones" in summary_df:
        for value in pd.to_numeric(summary_df["num_zones"], errors="coerce").dropna():
            if value > 0:
                present.append(int(value))

    ordered = dedupe_preserving_order(expected)
    for zone in sorted(set(present) - set(ordered)):
        ordered.append(zone)
    return ordered


def build_param_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "config_solver",
        "config_centroids_type",
        "config_cp_model_probing_level",
        "config_cp_sat_search_strategy",
        "config_linearization_level",
        "final_stage",
        "final_stage_index",
        "num_stages",
        "status",
        METRIC,
    }
    missing_columns = sorted(required_columns - set(summary_df.columns))
    if missing_columns:
        raise ValueError(f"summary.csv is missing required columns: {missing_columns}")

    df = summary_df.copy()
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    df["zone_count"] = df.apply(extract_zone_count, axis=1)
    df["solver"] = df["config_solver"].map(solver_label)
    df["variant"] = df.apply(variant_label, axis=1)

    completed_final_rows = final_block_rows(df)
    successful_rows = (
        df["status"].fillna("").astype(str).str.upper().isin(SUCCESS_STATUSES)
    )
    df = df[
        completed_final_rows
        & successful_rows
        & df[METRIC].notna()
        & (df[METRIC] >= 0)
        & df["zone_count"].notna()
        & df["solver"].notna()
        & df["variant"].notna()
    ].copy()

    df = add_percent_change_from_reference(df)
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for solution_id, row in df.reset_index(drop=True).iterrows():
        rows.append(
            {
                "solution_id": int(solution_id),
                "centroids_type": row["config_centroids_type"],
                "zone_count": int(row["zone_count"]),
                "solver": row["solver"],
                "variant": row["variant"],
                "series": series_label(row["solver"], row["variant"]),
                "metric_label": METRIC_LABEL,
                "value": float(row["percent_change"]),
                "raw_value": float(row[METRIC]),
                "reference_value": float(row["reference_value"]),
            }
        )

    return pd.DataFrame(rows)


def add_percent_change_from_reference(df: pd.DataFrame) -> pd.DataFrame:
    key_columns = [column for column in MATCH_COLUMNS if column in df.columns]
    if not key_columns:
        raise ValueError("No matching columns are available to identify baseline runs.")

    df = df.copy()
    df["series"] = df.apply(
        lambda row: series_label(row["solver"], row["variant"]), axis=1
    )
    reference_rows = df[df["series"] == REFERENCE_SERIES]
    if reference_rows.empty:
        return pd.DataFrame()

    reference_values = (
        reference_rows.groupby(key_columns, as_index=False, dropna=False)[METRIC]
        .mean()
        .rename(columns={METRIC: "reference_value"})
    )

    comparison_rows = df[df["series"] != REFERENCE_SERIES].copy()
    comparison_rows = comparison_rows.merge(
        reference_values, on=key_columns, how="left"
    )
    comparison_rows = comparison_rows[
        comparison_rows["reference_value"].notna()
        & (comparison_rows["reference_value"] != 0)
    ].copy()
    comparison_rows["percent_change"] = (
        (comparison_rows[METRIC] - comparison_rows["reference_value"])
        / comparison_rows["reference_value"]
        * 100
    )
    return comparison_rows


def final_block_rows(df: pd.DataFrame) -> pd.Series:
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
    zone = zone_count_from_value(row.get("config_centroids_type"))
    if zone is not None:
        return zone

    num_zones = pd.to_numeric(row.get("num_zones"), errors="coerce")
    if pd.notna(num_zones) and num_zones > 0:
        return int(num_zones)
    return None


def zone_count_from_value(value: object) -> int | None:
    match = re.match(r"(\d+)-zone", str(value or ""))
    if match:
        return int(match.group(1))
    return None


def solver_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    return SOLVER_LABELS.get(key)


def variant_label(row: pd.Series) -> str | None:
    active_variants: list[str] = []
    if has_numeric_value(row.get("config_cp_model_probing_level")):
        active_variants.append(VARIANT_LABELS["probing"])
    if has_text_value(row.get("config_cp_sat_search_strategy")):
        active_variants.append(VARIANT_LABELS["search_strategy"])
    if has_numeric_value(row.get("config_linearization_level")):
        active_variants.append(VARIANT_LABELS["linearization"])

    if not active_variants:
        return VARIANT_LABELS["baseline"]
    if len(active_variants) == 1:
        return active_variants[0]
    return None


def has_numeric_value(value: object) -> bool:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return pd.notna(numeric)


def has_text_value(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def series_label(solver: object, variant: object) -> str:
    return f"{solver} {variant}"


def plot_overall(comparison_values: pd.DataFrame, output_path: Path) -> None:
    plot_df = (
        comparison_values.groupby(["series", "metric_label"], as_index=False)["value"]
        .mean()
        .sort_values(["metric_label", "series"])
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="metric_label",
        y="value",
        hue="series",
        order=[METRIC_LABEL],
        hue_order=SERIES_ORDER,
        palette="tab10",
        ax=ax,
    )
    ax.set_title("CP-SAT Parameter Comparison Across All Runs")
    ax.set_xlabel("")
    ax.set_ylabel("Mean percent change")
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Solver and run type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_by_zone(
    comparison_values: pd.DataFrame, output_path: Path, zone_order: list[int]
) -> None:
    if not zone_order:
        zone_order = sorted(comparison_values["zone_count"].unique())
    zone_labels = [str(zone) for zone in zone_order]

    plot_df = (
        comparison_values.groupby(["zone_count", "series"], as_index=False)["value"]
        .mean()
        .sort_values(["zone_count", "series"])
    )
    plot_df["zone_count_label"] = plot_df["zone_count"].astype(int).astype(str)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(
        data=plot_df,
        x="zone_count_label",
        y="value",
        hue="series",
        order=zone_labels,
        hue_order=SERIES_ORDER,
        palette="tab10",
        ax=ax,
    )
    ax.set_title("CP-SAT Parameter Comparison by Zone Count")
    ax.set_xlabel("Zone count")
    ax.set_ylabel("Mean percent change")
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.legend(title="Solver and run type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def present_series_order(comparison_values: pd.DataFrame) -> list[str]:
    present = set(comparison_values["series"].dropna())
    ordered = [series for series in SERIES_ORDER if series in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


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


def normalize_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def dedupe_preserving_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


if __name__ == "__main__":
    main()

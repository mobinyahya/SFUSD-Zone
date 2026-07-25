"""Analyze trends of Status Quo simulations across multiple runs/years.

All (run × CSV) evaluation tasks are flattened into a single
ProcessPoolExecutor — both outer (run) and inner (CSV) loops run in
parallel.

The evaluator needs a schools lat/lon CSV (config key ``schools_data``)
and optionally an equity-block ``.npy`` (config key ``new_ctip_path``);
both can be set globally or overridden per run entry.
"""

import argparse
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add project root to path (must be before student_assignment imports)
sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tqdm import tqdm

from student_assignment.evaluation.match_evaluator import MatchEvaluator
from student_assignment.utils.plotting import (
    apply_plot_style,
    get_color_palette,
    save_figure,
)

logger = logging.getLogger(__name__)

# Task args: (year, program_file, assignment_path, student_data_path,
#             schools_data_path, new_ctip_path)
Task = tuple[int, str, str, str, str | None, str | None]


def _evaluate_csv_worker(args: Task) -> pd.Series | None:
    """Worker entry point: evaluate one CSV file.

    Args:
        args: Tuple of (year, program_file, assignment_path,
            student_data_path, schools_data_path, new_ctip_path).

    Returns:
        A Series of metrics, or None on failure.
    """
    (
        year,
        program_file,
        assignment_path,
        student_data_path,
        schools_data_path,
        new_ctip_path,
    ) = args
    try:
        student_data = pd.read_csv(student_data_path)
        file_assignment = pd.read_csv(assignment_path)
        me_year = int(f"{year}{year + 1}")
        match_eval = MatchEvaluator(
            student_data,
            file_assignment,
            first_round=True,
            dropout=False,
            low_income=95292,
            medium_income=95292,
            high_income=110850,
            grade=None,
            year=me_year,
            no_special_program=True,
            program_file=program_file,
            schools_latlon_path=schools_data_path,
            new_ctip_path=new_ctip_path,
        )
        return match_eval.eval_assignment_full().fillna(0)
    except FileNotFoundError as exc:
        logger.error("Student data not found: %s", exc)
        return None
    except Exception as exc:
        logger.error("Error evaluating %s: %s", assignment_path, exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_config(path: str) -> dict:
    """Load a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config as a dict.
    """
    with open(path) as fh:
        return yaml.safe_load(fh)


def _collect_csv_files(run: dict) -> tuple[list[str], bool]:
    """Discover CSV files for a run config entry.

    Args:
        run: Single run dict (must have 'run_csv' or 'folder').

    Returns:
        Tuple of (list_of_csv_paths, is_single_file).
    """
    if "run_csv" in run:
        return [run["run_csv"]], True

    csv_files = []
    for path in sorted(Path(run["folder"]).rglob("*.csv")):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            logger.warning("Could not inspect CSV %s: %s", path, exc)
            continue
        required = {"studentno", "programno", "rank"}
        if required <= columns and "programcodes" in columns:
            csv_files.append(str(path))
    return csv_files, False


def _aggregate_metrics(
    metrics_list: list[pd.Series],
    is_single_file: bool,
) -> dict[str, pd.Series]:
    """Aggregate a list of per-CSV metric Series into mean and std.

    Args:
        metrics_list: Non-empty list of metric Series.
        is_single_file: If True, std is zeroed out.

    Returns:
        Dict with keys 'mean' and 'std', each a pd.Series.
    """
    if is_single_file:
        mean = metrics_list[0]
        std = pd.Series(0.0, index=mean.index)
    else:
        df = pd.concat(metrics_list, axis=1).T
        mean = df.mean()
        std = df.std()
    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------------
# Export & Plotting
# ---------------------------------------------------------------------------


def export_to_excel(
    all_metrics_data: dict[str, dict[str, pd.Series]],
    labels: list[str],
    output_dir: str,
    row_order: list[str] | None = None,
) -> str:
    """Export all metrics to Excel for comprehensive comparison.

    Args:
        all_metrics_data: Dict mapping label -> {"mean": Series, "std": Series}.
        labels: List of run labels in order.
        output_dir: Directory to save Excel file.
        row_order: Optional list of metric names for row ordering. Listed
            metrics come first; remaining are appended alphabetically.

    Returns:
        Path to the exported Excel (or CSV fallback) file.
    """
    df_mean = pd.DataFrame({lbl: all_metrics_data[lbl]["mean"] for lbl in labels})
    df_std = pd.DataFrame({lbl: all_metrics_data[lbl]["std"] for lbl in labels})

    if row_order:
        existing_ordered = [r for r in row_order if r in df_mean.index]
        remaining = sorted(r for r in df_mean.index if r not in set(existing_ordered))
        final_order = existing_ordered + remaining
        df_mean = df_mean.reindex(final_order)
        df_std = df_std.reindex(final_order)
    else:
        df_mean = df_mean.sort_index()
        df_std = df_std.sort_index()

    excel_path = os.path.join(output_dir, "metrics_comparison.xlsx")

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_mean.to_excel(writer, sheet_name="Mean Values")
            df_std.to_excel(writer, sheet_name="Std Values")

            df_combined = df_mean.copy()
            for col in df_combined.columns:
                df_combined[col] = (
                    df_mean[col].apply(lambda x: f"{x:.4f}")
                    + " ± "
                    + df_std[col].apply(lambda x: f"{x:.4f}")
                )
            df_combined.to_excel(writer, sheet_name="Mean ± Std")

        logger.info("Excel exported to: %s", excel_path)
    except (ImportError, ModuleNotFoundError):
        csv_mean = os.path.join(output_dir, "metrics_mean.csv")
        csv_std = os.path.join(output_dir, "metrics_std.csv")
        df_mean.to_csv(csv_mean)
        df_std.to_csv(csv_std)
        logger.warning("openpyxl unavailable. Exported CSVs: %s, %s", csv_mean, csv_std)
        excel_path = csv_mean

    logger.info("  Total metrics: %d | Total runs: %d", len(df_mean), len(labels))
    return excel_path


def plot_diagnostic_trends(
    all_metrics_data: dict[str, dict[str, pd.Series]],
    labels: list[str],
    output_dir: str,
) -> None:
    """Generate diagnostic trend plots from collected metrics.

    Args:
        all_metrics_data: Dict mapping label -> {"mean": Series, "std": Series}.
        labels: List of run labels in order.
        output_dir: Directory to save plots.
    """
    import seaborn as sns

    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    records = [
        {"label": lbl, **all_metrics_data[lbl]["mean"].to_dict()} for lbl in labels
    ]
    metrics_df = pd.DataFrame(records)

    if metrics_df.empty:
        logger.warning("No diagnostic metrics to plot.")
        return

    def _plot_metric_group(prefix: str, title: str, ylabel: str, filename: str) -> None:
        cols = [c for c in metrics_df.columns if c.startswith(prefix)]
        if not cols:
            return

        plt.figure(figsize=(12, 6))
        melted = metrics_df.melt(
            id_vars=["label"],
            value_vars=cols,
            var_name="Metric",
            value_name="Value",
        )
        melted["Metric"] = melted["Metric"].str.replace(prefix, "", regex=False)

        sns.lineplot(data=melted, x="label", y="Value", hue="Metric", marker="o")
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        save_figure(os.path.join(diag_dir, filename))

    _plot_metric_group(
        "utilization_",
        "Capacity Utilization Trends by Program Type",
        "Utilization Rate",
        "trend_utilization_by_type.png",
    )
    _plot_metric_group(
        "count_students_",
        "Total Student Count by Ethnicity",
        "Number of Students",
        "trend_student_counts_by_ethnicity.png",
    )
    _plot_metric_group(
        "enrollment_count_",
        "Assigned Students by Ethnicity",
        "Number of Assigned Students",
        "trend_enrollment_counts_by_ethnicity.png",
    )
    _plot_metric_group(
        "enrollment_rate_",
        "Enrollment Rate by Ethnicity",
        "Percent Assigned",
        "trend_enrollment_rate_by_ethnicity.png",
    )

    logger.info("Diagnostic plots saved to: %s", diag_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args, process all runs in parallel, export results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Analyze trends of Status Quo simulations."
    )
    parser.add_argument(
        "--config",
        default="configs/analysis_config.yaml",
        help="Path to analysis config",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel worker processes (default: os.cpu_count())",
    )
    args = parser.parse_args()

    config = get_config(args.config)

    # Evaluator inputs — global defaults, overridable per run entry.
    default_schools_data = config.get("schools_data")
    default_new_ctip_path = config.get("new_ctip_path")

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    runs: list[dict] = config.get("runs", [])
    if not runs:
        logger.error("No runs found in config.")
        return

    logger.info("Found runs: %s", [r["label"] for r in runs])

    # -----------------------------------------------------------------------
    # Build flat task list: all (run, csv_path) pairs across every run
    # -----------------------------------------------------------------------
    # task key: (label, csv_path) → task args (see Task type alias)
    task_map: dict[tuple[str, str], Task] = {}
    run_meta: dict[str, tuple[int, bool]] = {}  # label -> (n_csvs, is_single)

    for run in runs:
        label = run["label"]
        if "student_data" not in run or "program_data" not in run:
            logger.warning(
                "Run '%s' missing student_data or program_data — skipping.",
                label,
            )
            continue

        csv_files, is_single_file = _collect_csv_files(run)
        if not csv_files:
            logger.warning("No CSVs found for run '%s'.", label)
            continue

        run_meta[label] = (len(csv_files), is_single_file)
        for csv_path in csv_files:
            task_map[(label, csv_path)] = (
                run["year"],
                run["program_data"],
                csv_path,
                run["student_data"],
                run.get("schools_data", default_schools_data),
                run.get("new_ctip_path", default_new_ctip_path),
            )

    if not task_map:
        logger.error("No tasks to process.")
        return

    # -----------------------------------------------------------------------
    # Execute all tasks in a single pool (runs × CSVs fully parallel)
    # -----------------------------------------------------------------------
    raw_results: dict[str, list[pd.Series]] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_key = {
            pool.submit(_evaluate_csv_worker, task): key
            for key, task in task_map.items()
        }
        for future in tqdm(
            as_completed(future_to_key),
            total=len(future_to_key),
            desc="Evaluating",
        ):
            label, csv_path = future_to_key[future]
            result = future.result()
            if result is not None:
                raw_results.setdefault(label, []).append(result)

    # -----------------------------------------------------------------------
    # Aggregate per run (preserving config order)
    # -----------------------------------------------------------------------
    all_metrics_data: dict[str, dict[str, pd.Series]] = {}
    for run in runs:
        label = run["label"]
        if label not in raw_results or label not in run_meta:
            continue
        _, is_single_file = run_meta[label]
        all_metrics_data[label] = _aggregate_metrics(raw_results[label], is_single_file)

    labels = [r["label"] for r in runs if r["label"] in all_metrics_data]

    if not labels:
        logger.error("No data to plot.")
        return

    # -----------------------------------------------------------------------
    # Export to Excel
    # -----------------------------------------------------------------------
    row_order = config.get("row_order", None)
    export_to_excel(all_metrics_data, labels, output_dir, row_order=row_order)

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    apply_plot_style()

    def get_series(metric_name: str, stat: str = "mean") -> np.ndarray:
        """Return metric values across labels as a float array (NaN for missing)."""
        values = [
            all_metrics_data[lbl][stat].get(metric_name, np.nan) for lbl in labels
        ]
        return np.array(values, dtype=float)

    # Single metrics
    for metric in config.get("single_metrics", []):
        means = get_series(metric, "mean")
        stds = get_series(metric, "std")

        if np.all(np.isnan(means)):
            continue

        plt.figure(figsize=(10, 6))
        x_indices = np.arange(len(labels))
        plt.errorbar(x_indices, means, yerr=stds, marker="o", capsize=5)
        plt.title(f"{metric} over Runs")
        plt.xlabel("Run")
        plt.ylabel(metric)
        plt.xticks(x_indices, labels, rotation=45, ha="right")
        plt.tight_layout()

        filename = (
            metric.replace(" ", "_").replace("/", "_").replace("%", "Pct") + ".png"
        )
        save_figure(os.path.join(output_dir, filename))

    # Group metrics
    sample_keys = all_metrics_data[labels[0]]["mean"].index.tolist()

    for pattern in config.get("group_metrics", []):
        regex_pattern = "^" + re.escape(pattern).replace("\\{group\\}", "(.*)") + "$"

        matches = [
            (k, re.match(regex_pattern, k).group(1))  # type: ignore[union-attr]
            for k in sample_keys
            if re.match(regex_pattern, k)
        ]

        if not matches:
            continue

        plt.figure(figsize=(12, 8))
        palette = get_color_palette(len(matches))
        x_indices = np.arange(len(labels))

        for i, (metric_key, group_name) in enumerate(matches):
            plt.errorbar(
                x_indices,
                get_series(metric_key, "mean"),
                yerr=get_series(metric_key, "std"),
                marker="o",
                capsize=5,
                label=group_name,
                color=palette[i],
            )

        base_name = pattern.replace("({group})", "").strip()
        plt.title(f"{base_name} by Group over Runs")
        plt.xlabel("Run")
        plt.ylabel(base_name)
        plt.xticks(x_indices, labels, rotation=45, ha="right")
        plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        filename = (
            "Group_"
            + base_name.replace(" ", "_").replace("/", "_").replace("%", "Pct")
            + ".png"
        )
        save_figure(os.path.join(output_dir, filename))

    # Diagnostic plots
    plot_diagnostic_trends(all_metrics_data, labels, output_dir)

    logger.info("Analysis complete. Plots saved to %s", output_dir)


if __name__ == "__main__":
    main()

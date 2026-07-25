#!/usr/bin/env python3
"""Plot the soft-reserve FRL dissimilarity-distance Pareto frontier."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.config import OptimizationConfig  # noqa: E402
from optimization.levels import LevelSpec  # noqa: E402
from optimization.problem import ZoneProblem  # noqa: E402
from optimization.solution import ZoneSolution  # noqa: E402
from optimization.visualization import (  # noqa: E402
    VisualizationArtifactStore,
    render_solution_map,
)


DEFAULT_INPUT = Path("~/Downloads/clean_summary.csv").expanduser()
DEFAULT_PLOTS_DIR = Path(__file__).resolve().parent / "plots"
DEFAULT_OUTPUT = DEFAULT_PLOTS_DIR / "choice_reserves_pareto_frontier.png"
SOFT_RESERVES_RESULTS_PATTERN = "soft_reserves_06frl_25_eval_assignment_full_*.csv"
STATUS_QUO_RESULTS_DIR = (
    Path(__file__).resolve().parent / "matches/status_quo_policies_25"
)
STATUS_QUO_RESULTS_PATTERN = "status_quo_policies_25_eval_assignment_full_*.csv"
SPECIAL_ZONE_RESULTS_DIR = (
    Path(__file__).resolve().parent / "matches/zones_soft_reserves_06frl_25"
)
SPECIAL_ZONE_RESULTS_PATTERN = "zones_soft_reserves_06frl_25_eval_assignment_full_*.csv"

PARETO_METRICS = {
    "Dissimilarity (High FRL)": "frl_dissimilarity",
    "Distance Av (All Assigned)": "avg_student_distance",
}
SOFT_RESERVES_METRICS = {
    **PARETO_METRICS,
    "FRL Max Dev": "frl_max_dev",
}
STATUS_QUO_POLICIES = {
    "status_quo": {
        "label": "Status Quo",
        "marker": "*",
        "color": "#dc2626",
        "edgecolor": "#7f1d1d",
        "size": 340,
        "label_offset": (10, 8),
        "label_alignment": "left",
    },
    # "status_quo+soft_reserves_06frl": {
    #     "label": "Status Quo + Soft Reserves",
    #     "marker": "P",
    #     "color": "#f97316",
    #     "edgecolor": "#9a3412",
    #     "size": 220,
    #     "label_offset": (-10, 8),
    #     "label_alignment": "right",
    # },
}
SPECIAL_ZONE_POLICIES = {
    "small_zones_1": {
        "label": "Small Zones 1",
        "marker": "X",
        "color": "#2563eb",
        "edgecolor": "#1e3a8a",
        "size": 190,
        "label_offset": (10, -16),
        "label_alignment": "left",
    },
    "small_zones_2": {
        "label": "Small Zones 2",
        "marker": "D",
        "color": "#7c3aed",
        "edgecolor": "#4c1d95",
        "size": 150,
        "label_offset": (10, 8),
        "label_alignment": "left",
    },
    "medium_zones": {
        "label": "Medium Zones",
        "marker": "s",
        "color": "#059669",
        "edgecolor": "#064e3b",
        "size": 165,
        "label_offset": (-10, 8),
        "label_alignment": "right",
    },
}
RUN_ID_PATTERN = re.compile(r"(?:^|_)([0-9a-fA-F]{8,64})$")
FEASIBLE_STATUSES = {"FEASIBLE", "OPTIMAL"}
POLICIES = {
    # "no": {"label": "No reserves", "marker": "o"},
    "soft": {"label": "Soft reserves", "marker": "^"},
    # "hard": {"label": "Hard reserves", "marker": "s"},
}

FRONTIER_CSV_COLUMNS = [
    "frontier_number",
    "task_id",
    "reserve_policy",
    "num_zones",
    "frl_dissimilarity",
    "avg_student_distance",
    "frl_max_dev",
    "source_row",
    "path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot choice-assignment FRL dissimilarity against average student "
            "distance, with reserve policies as separate observations."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input summary CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output plot path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--soft-reserves-input",
        type=Path,
        help=(
            "CSV written by evaluate_soft_reserve_matches.py. Default: newest "
            f"{SOFT_RESERVES_RESULTS_PATTERN} in {DEFAULT_PLOTS_DIR}."
        ),
    )
    parser.add_argument(
        "--status-quo-input",
        type=Path,
        help=(
            "CSV written by evaluate_status_quo_matches.py. Default: newest "
            f"{STATUS_QUO_RESULTS_PATTERN} in {STATUS_QUO_RESULTS_DIR}."
        ),
    )
    parser.add_argument(
        "--special-zones-input",
        type=Path,
        help=(
            "CSV containing the Small Zones 1, Small Zones 2, and Medium Zones "
            "evaluations. Default: newest "
            f"{SPECIAL_ZONE_RESULTS_PATTERN} in {SPECIAL_ZONE_RESULTS_DIR}."
        ),
    )
    parser.add_argument(
        "--max-frl-max-dev",
        type=float,
        help="Keep solutions with FRL Max Dev strictly below this value.",
    )
    parser.add_argument(
        "--pareto-solutions-dir",
        type=Path,
        help=(
            "Directory for numbered frontier visualizations. Default: "
            "<output stem>_viz beside the plot."
        ),
    )
    return parser.parse_args()


def reshape_choice_metrics(
    summary: pd.DataFrame,
    soft_reserve_results: pd.DataFrame,
) -> pd.DataFrame:
    """Join evaluated soft-reserve matching metrics to benchmark metadata."""
    required = {"num_zones", "path", "status"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    if "metric" not in soft_reserve_results.columns:
        raise ValueError("Soft-reserve results CSV is missing the 'metric' column.")
    duplicate_metrics = soft_reserve_results["metric"].duplicated(keep=False)
    duplicated_required = sorted(
        set(soft_reserve_results.loc[duplicate_metrics, "metric"])
        & set(SOFT_RESERVES_METRICS)
    )
    if duplicated_required:
        raise ValueError(
            "Soft-reserve results CSV contains duplicate required metrics: "
            f"{duplicated_required}"
        )

    indexed_results = soft_reserve_results.set_index("metric")
    missing_metrics = sorted(set(SOFT_RESERVES_METRICS) - set(indexed_results.index))
    if missing_metrics:
        raise ValueError(
            f"Soft-reserve results CSV is missing required metrics: {missing_metrics}"
        )

    matching_points = (
        indexed_results.loc[list(SOFT_RESERVES_METRICS)]
        .T.rename(columns=SOFT_RESERVES_METRICS)
        .rename_axis("matching_path")
        .reset_index()
    )
    matching_points["run_id"] = matching_points["matching_path"].map(extract_run_id)
    invalid_matching_paths = matching_points.loc[
        matching_points["run_id"].isna(), "matching_path"
    ]
    if not invalid_matching_paths.empty:
        examples = invalid_matching_paths.astype(str).head(3).tolist()
        raise ValueError(
            "Could not extract run IDs from soft-reserve result columns, including: "
            f"{examples}"
        )
    if matching_points["run_id"].duplicated().any():
        duplicates = sorted(
            matching_points.loc[
                matching_points["run_id"].duplicated(keep=False), "run_id"
            ].unique()
        )
        raise ValueError(
            f"Soft-reserve results contain duplicate run IDs: {duplicates}"
        )

    summary_points = pd.DataFrame(
        {
            "source_row": summary.index,
            "path": summary["path"],
            "run_id": summary["path"].map(extract_run_id),
            "summary_task_id": (
                summary["task_id"]
                if "task_id" in summary.columns
                else pd.Series(pd.NA, index=summary.index, dtype="string")
            ),
            "num_zones": pd.to_numeric(summary["num_zones"], errors="coerce"),
            "status": summary["status"].astype("string").str.upper(),
        }
    )
    summary_points = summary_points[
        summary_points["status"].isin(FEASIBLE_STATUSES)
    ].copy()
    summary_run_ids = summary_points["run_id"].dropna()
    if summary_run_ids.duplicated().any():
        duplicates = sorted(summary_run_ids[summary_run_ids.duplicated()].unique())
        raise ValueError(f"Input summary contains duplicate run IDs: {duplicates}")

    points = summary_points.merge(
        matching_points,
        on="run_id",
        how="inner",
        validate="one_to_one",
    )
    points["reserve_policy"] = POLICIES["soft"]["label"]
    for column in SOFT_RESERVES_METRICS.values():
        points[column] = pd.to_numeric(points[column], errors="coerce")
    points = points.dropna(subset=["num_zones", *SOFT_RESERVES_METRICS.values()]).copy()
    points = points[
        (points["num_zones"] > 0)
        & (points["frl_dissimilarity"] >= 0)
        & (points["avg_student_distance"] >= 0)
        & (points["frl_max_dev"] >= 0)
    ].copy()
    if points.empty:
        raise ValueError("No rows contain valid zone counts and choice metrics.")

    rounded_zones = points["num_zones"].round()
    if not np.allclose(points["num_zones"], rounded_zones):
        raise ValueError("num_zones contains non-integer values.")
    points["num_zones"] = rounded_zones.astype(int)
    return points


def filter_by_frl_max_dev(points: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Keep solutions strictly below the requested maximum FRL deviation."""
    if threshold < 0:
        raise ValueError("FRL Max Dev threshold must be non-negative.")
    filtered = points.loc[points["frl_max_dev"] < threshold].copy()
    if filtered.empty:
        raise ValueError(f"No solutions have FRL Max Dev below {threshold}.")
    return filtered


def find_latest_soft_reserves_results(search_dir: Path) -> Path:
    """Find the newest timestamped output from the soft-reserve evaluator."""
    candidates = sorted(search_dir.glob(SOFT_RESERVES_RESULTS_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            "Could not find soft-reserve evaluation results matching "
            f"{search_dir / SOFT_RESERVES_RESULTS_PATTERN}"
        )
    return candidates[-1]


def find_latest_status_quo_results(search_dir: Path) -> Path:
    """Find the newest timestamped status-quo policy evaluation."""
    candidates = sorted(search_dir.glob(STATUS_QUO_RESULTS_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            "Could not find status-quo evaluation results matching "
            f"{search_dir / STATUS_QUO_RESULTS_PATTERN}"
        )
    return candidates[-1]


def find_latest_special_zone_results(search_dir: Path) -> Path:
    """Find the newest timestamped special-zone evaluation."""
    candidates = sorted(search_dir.glob(SPECIAL_ZONE_RESULTS_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            "Could not find special-zone evaluation results matching "
            f"{search_dir / SPECIAL_ZONE_RESULTS_PATTERN}"
        )
    return candidates[-1]


def extract_status_quo_points(results: pd.DataFrame) -> pd.DataFrame:
    """Extract the configured status-quo reference points."""
    return _extract_reference_points(results, STATUS_QUO_POLICIES, "Status-quo")


def extract_special_zone_points(results: pd.DataFrame) -> pd.DataFrame:
    """Extract the three named base-zone reference points."""
    return _extract_reference_points(results, SPECIAL_ZONE_POLICIES, "Special-zone")


def _extract_reference_points(
    results: pd.DataFrame,
    policies: Mapping[str, Mapping[str, object]],
    description: str,
) -> pd.DataFrame:
    if "metric" not in results.columns:
        raise ValueError(f"{description} results CSV is missing the 'metric' column.")

    duplicate_metrics = results["metric"].duplicated(keep=False)
    duplicated_required = sorted(
        set(results.loc[duplicate_metrics, "metric"]) & set(PARETO_METRICS)
    )
    if duplicated_required:
        raise ValueError(
            f"{description} results CSV contains duplicate required metrics: "
            f"{duplicated_required}"
        )

    indexed_results = results.set_index("metric")
    missing_metrics = sorted(set(PARETO_METRICS) - set(indexed_results.index))
    if missing_metrics:
        raise ValueError(
            f"{description} results CSV is missing required metrics: {missing_metrics}"
        )
    missing_policies = sorted(set(policies) - set(results.columns))
    if missing_policies:
        raise ValueError(
            f"{description} results CSV is missing policies: {missing_policies}"
        )

    points = []
    for policy, style in policies.items():
        values = pd.to_numeric(
            indexed_results.loc[list(PARETO_METRICS), policy],
            errors="coerce",
        )
        point = {
            "policy": policy,
            "label": style["label"],
            **{
                output_column: values.loc[metric]
                for metric, output_column in PARETO_METRICS.items()
            },
        }
        points.append(point)

    frame = pd.DataFrame(points)
    metric_columns = list(PARETO_METRICS.values())
    if frame[metric_columns].isna().any().any():
        raise ValueError(f"{description} results contain non-numeric required metrics.")
    if (frame[metric_columns] < 0).any().any():
        raise ValueError(f"{description} results contain negative required metrics.")
    return frame


def extract_run_id(value: object) -> str | None:
    """Extract the short trailing run ID used in benchmark directory names."""
    if pd.isna(value):
        return None
    match = RUN_ID_PATTERN.search(Path(str(value)).name)
    return match.group(1) if match else None


def export_frontier_solutions(
    frontier: pd.DataFrame,
    destination_dir: Path,
) -> pd.DataFrame:
    """Add authoritative task IDs and render final Block_0 visualizations."""
    frontier = frontier.copy()
    task_ids: list[str] = []
    destination_dir.mkdir(parents=True, exist_ok=False)
    artifact_store = VisualizationArtifactStore()

    for row in frontier.itertuples():
        run_path = Path(str(row.path)).expanduser()
        manifest_path = run_path / "benchmark_manifest.json"
        try:
            with manifest_path.open(encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read benchmark manifest: {manifest_path}"
            ) from exc

        task_id = str(manifest.get("task_id") or "")
        config_hash = str(manifest.get("config_hash") or "")
        if not task_id:
            raise ValueError(f"Benchmark manifest has no task_id: {manifest_path}")
        if config_hash and task_id != config_hash[:12]:
            raise ValueError(
                f"Manifest task_id {task_id!r} does not match config_hash in "
                f"{manifest_path}"
            )
        if not pd.isna(row.summary_task_id) and str(row.summary_task_id) != task_id:
            raise ValueError(
                f"Summary task_id {row.summary_task_id!r} does not match manifest "
                f"task_id {task_id!r} for {run_path}"
            )

        status = str(manifest.get("status") or "").upper()
        if status not in FEASIBLE_STATUSES:
            raise ValueError(
                f"Frontier run {run_path} has non-feasible manifest status {status!r}"
            )

        solution, stage = load_final_block0_solution(run_path, manifest)
        render_solution_to_path(
            solution,
            stage,
            destination_dir / f"{row.frontier_number}.png",
            artifact_store,
        )
        task_ids.append(task_id)

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Pareto frontier contains duplicate manifest task IDs.")

    frontier["task_id"] = task_ids
    return frontier


def load_final_block0_solution(
    run_path: Path,
    manifest: Mapping[str, object],
) -> tuple[ZoneSolution, str]:
    """Reconstruct the final solution without loading any coarser graph stages."""
    final_stage = str(manifest.get("final_stage") or "")
    stage = next(
        (
            item
            for item in manifest.get("stages", [])
            if isinstance(item, Mapping) and item.get("name") == final_stage
        ),
        None,
    )
    if stage is None:
        raise ValueError(f"Manifest has no final stage record for {final_stage!r}")

    level = LevelSpec.parse(str(stage.get("level") or ""))
    if level.name != "Block_0":
        raise ValueError(
            f"Frontier run {run_path} ends at {level.name}, expected Block_0"
        )

    saved_config = dict(manifest.get("config") or {})
    saved_graphs_dir_value = str(saved_config.get("graphs_dir") or "")
    saved_graphs_dir = Path(saved_graphs_dir_value).expanduser()
    graphs_dir = (
        str(saved_graphs_dir)
        if saved_graphs_dir_value and saved_graphs_dir.is_dir()
        else ""
    )
    config = OptimizationConfig(
        centroids_type=str(saved_config.get("centroids_type") or "5-zone-AF"),
        levels=[level.name],
        years=[
            int(year)
            for year in saved_config.get("years") or [14, 15, 16, 17, 18, 21, 22]
        ],
        population_type=str(saved_config.get("population_type") or "GE"),
        drop_optout=bool(saved_config.get("drop_optout", True)),
        capacity_scenario=str(saved_config.get("capacity_scenario") or "A"),
        new_schools=bool(saved_config.get("new_schools", True)),
        include_k8=bool(saved_config.get("include_k8", False)),
        graphs_dir=graphs_dir,
    )
    graph = config.make_dataset().graph_for(level)

    stage_path = run_path / str(stage.get("path") or "")
    solution_path = stage_path / f"solution_{level.name}.json"
    area_assignment_path = stage_path / f"zone_dict_area_{level.name}.json"
    solution_data = _load_json(solution_path)
    raw_area_assignment = _load_json(area_assignment_path)
    area_assignment = {
        int(area_id): int(zone) for area_id, zone in raw_area_assignment.items()
    }

    assignment: dict[int, int] = {}
    for node, attrs in graph.nodes(data=True):
        area_ids = (
            [attrs["area_id"]] if "area_id" in attrs else attrs.get("block_ids", [])
        )
        zones = {
            area_assignment[int(area_id)]
            for area_id in area_ids
            if int(area_id) in area_assignment
        }
        if len(zones) > 1:
            raise ValueError(
                f"Saved Block_0 assignment maps graph node {node} to multiple zones"
            )
        if zones:
            assignment[int(node)] = zones.pop()
    if not assignment:
        raise ValueError(
            f"No Block_0 assignments could be reconstructed for {run_path}"
        )

    centroids = [int(node) for node in solution_data.get("centroids") or []]
    if not centroids:
        raise ValueError(f"Saved solution has no centroids: {solution_path}")
    metadata = dict(solution_data.get("metadata") or stage.get("metadata") or {})
    centroid_school_ids = metadata.get("centroid_school_ids") or centroids
    problem = ZoneProblem(
        G=graph,
        level=level,
        centroids=centroids,
        centroid_school_ids=[int(school_id) for school_id in centroid_school_ids],
    )
    solution = ZoneSolution(
        problem=problem,
        assignment=assignment,
        status=str(solution_data.get("status") or stage.get("status") or "UNKNOWN"),
        objective=solution_data.get("objective", stage.get("objective")),
        wall_time=solution_data.get("wall_time", stage.get("wall_time")),
        metadata=metadata,
    )
    if not solution.feasible:
        raise ValueError(
            f"Final Block_0 solution for {run_path} has status {solution.status!r}"
        )
    return solution, final_stage


def render_solution_to_path(
    solution: ZoneSolution,
    stage: str,
    destination: Path,
    artifact_store: VisualizationArtifactStore,
) -> None:
    """Render a zoning solution directly to a caller-selected PNG path."""
    if destination.exists():
        raise FileExistsError(destination)
    geometry, _ = artifact_store.geometry_for(solution.level, solution.problem.G)
    figure = render_solution_map(solution, geometry, stage)
    try:
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    finally:
        plt.close(figure)


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def pareto_mask(points: pd.DataFrame) -> np.ndarray:
    """Return points not dominated when both plotted metrics are minimized."""
    x = points["frl_dissimilarity"].to_numpy(dtype=float)
    y = points["avg_student_distance"].to_numpy(dtype=float)

    not_worse = (x[:, None] <= x[None, :]) & (y[:, None] <= y[None, :])
    strictly_better = (x[:, None] < x[None, :]) | (y[:, None] < y[None, :])
    dominated = (not_worse & strictly_better).any(axis=0)
    return ~dominated


def plot_frontier(
    points: pd.DataFrame,
    status_quo_points: pd.DataFrame,
    special_zone_points: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Choice Assignment Pareto Frontier",
) -> pd.DataFrame:
    """Draw all policy points and emphasize the global Pareto frontier."""
    points = points.copy()
    points["pareto_optimal"] = pareto_mask(points)
    frontier = (
        points[points["pareto_optimal"]]
        .sort_values(["frl_dissimilarity", "avg_student_distance"])
        .copy()
    )
    frontier.insert(0, "frontier_number", range(1, len(frontier) + 1))

    zone_counts = sorted(points["num_zones"].unique())
    colors = colormaps["viridis"].resampled(len(zone_counts))(
        np.arange(len(zone_counts))
    )
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(zone_counts) + 1) - 0.5, cmap.N)
    zone_positions = {zone: position for position, zone in enumerate(zone_counts)}
    points["zone_color_position"] = points["num_zones"].map(zone_positions)

    sns.set_theme(context="talk", style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    for policy, style in POLICIES.items():
        label = style["label"]
        policy_points = points[points["reserve_policy"] == label]
        dominated = policy_points[~policy_points["pareto_optimal"]]
        optimal = policy_points[policy_points["pareto_optimal"]]

        ax.scatter(
            dominated["avg_student_distance"],
            dominated["frl_dissimilarity"],
            c=dominated["zone_color_position"],
            cmap=cmap,
            norm=norm,
            marker=style["marker"],
            s=58,
            alpha=0.48,
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            optimal["avg_student_distance"],
            optimal["frl_dissimilarity"],
            c=optimal["zone_color_position"],
            cmap=cmap,
            norm=norm,
            marker=style["marker"],
            s=145,
            alpha=1,
            edgecolors="#111827",
            linewidths=1.35,
            zorder=4,
        )

    if len(frontier) > 1:
        ax.plot(
            frontier["avg_student_distance"],
            frontier["frl_dissimilarity"],
            color="#111827",
            linewidth=1.6,
            linestyle="--",
            alpha=0.8,
            zorder=3,
        )

    labels = [
        ax.text(
            row.avg_student_distance,
            row.frl_dissimilarity,
            str(row.frontier_number),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#111827",
            bbox={
                "boxstyle": "circle,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#111827",
                "linewidth": 0.7,
                "alpha": 0.92,
            },
            zorder=5,
        )
        for row in frontier.itertuples()
    ]
    if labels:
        adjust_text(
            labels,
            ax=ax,
            expand=(1.4, 1.6),
            arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.7},
        )

    reference_styles = {**STATUS_QUO_POLICIES, **SPECIAL_ZONE_POLICIES}
    reference_points = pd.concat(
        [status_quo_points, special_zone_points], ignore_index=True
    )
    for row in reference_points.itertuples(index=False):
        style = reference_styles[row.policy]
        ax.scatter(
            row.avg_student_distance,
            row.frl_dissimilarity,
            color=style["color"],
            marker=style["marker"],
            s=style["size"],
            edgecolors=style["edgecolor"],
            linewidths=1,
            zorder=6,
        )
        ax.annotate(
            row.label,
            (row.avg_student_distance, row.frl_dissimilarity),
            xytext=style["label_offset"],
            textcoords="offset points",
            ha=style["label_alignment"],
            color=style["edgecolor"],
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
            zorder=7,
        )

    shape_handles = [
        Line2D(
            [],
            [],
            color="none",
            marker=style["marker"],
            markerfacecolor="#64748b",
            markeredgecolor="none",
            markersize=9,
            label=style["label"],
        )
        for style in POLICIES.values()
    ]
    shape_handles.append(
        Line2D(
            [],
            [],
            color="#111827",
            linestyle="--",
            linewidth=1.6,
            label="Pareto frontier",
        )
    )
    shape_handles.extend(
        Line2D(
            [],
            [],
            color="none",
            marker=style["marker"],
            markerfacecolor=style["color"],
            markeredgecolor=style["edgecolor"],
            markersize=11,
            label=style["label"],
        )
        for style in reference_styles.values()
    )
    ax.legend(handles=shape_handles, title="Reserve policy", frameon=True)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        ticks=np.arange(len(zone_counts)),
        pad=0.02,
        fraction=0.05,
    )
    colorbar.ax.set_yticklabels([str(zone) for zone in zone_counts])
    colorbar.set_label("Number of zones")

    ax.set_title(title, fontweight="bold", pad=14)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("Average student distance (miles)")
    ax.set_ylabel("FRL dissimilarity (%)")
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return frontier


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    frontier_csv_path = output_path.with_suffix(".csv")
    pareto_solutions_dir = (
        args.pareto_solutions_dir.expanduser().resolve()
        if args.pareto_solutions_dir
        else output_path.parent / f"{output_path.stem}_viz"
    )
    for destination in (output_path, frontier_csv_path, pareto_solutions_dir):
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {destination}"
            )
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {input_path}")
    soft_reserves_input = (
        args.soft_reserves_input.expanduser().resolve()
        if args.soft_reserves_input
        else find_latest_soft_reserves_results(DEFAULT_PLOTS_DIR)
    )
    if not soft_reserves_input.exists():
        raise FileNotFoundError(
            f"Could not find soft-reserve results CSV: {soft_reserves_input}"
        )
    status_quo_input = (
        args.status_quo_input.expanduser().resolve()
        if args.status_quo_input
        else find_latest_status_quo_results(STATUS_QUO_RESULTS_DIR)
    )
    if not status_quo_input.exists():
        raise FileNotFoundError(
            f"Could not find status-quo results CSV: {status_quo_input}"
        )
    special_zones_input = (
        args.special_zones_input.expanduser().resolve()
        if args.special_zones_input
        else find_latest_special_zone_results(SPECIAL_ZONE_RESULTS_DIR)
    )
    if not special_zones_input.exists():
        raise FileNotFoundError(
            f"Could not find special-zone results CSV: {special_zones_input}"
        )

    summary = pd.read_csv(input_path)
    soft_reserve_results = pd.read_csv(soft_reserves_input)
    status_quo_results = pd.read_csv(status_quo_input)
    special_zone_results = pd.read_csv(special_zones_input)
    points = reshape_choice_metrics(summary, soft_reserve_results)
    unfiltered_point_count = len(points)
    title = "Choice Assignment Pareto Frontier"
    if args.max_frl_max_dev is not None:
        points = filter_by_frl_max_dev(points, args.max_frl_max_dev)
        title += f"\nFRL Max Dev < {args.max_frl_max_dev:.0%}"
    status_quo_points = extract_status_quo_points(status_quo_results)
    special_zone_points = extract_special_zone_points(special_zone_results)
    frontier = plot_frontier(
        points,
        status_quo_points,
        special_zone_points,
        output_path,
        title=title,
    )
    frontier = export_frontier_solutions(frontier, pareto_solutions_dir)
    with frontier_csv_path.open("x", encoding="utf-8", newline="") as output_file:
        frontier[FRONTIER_CSV_COLUMNS].to_csv(output_file, index=False)

    print(
        f"Plotted {len(points)} of {unfiltered_point_count} matched points from "
        f"{len(summary)} result rows "
        f"using {soft_reserves_input}."
    )
    print(f"Loaded status-quo reference points from {status_quo_input}.")
    print(f"Loaded special-zone reference points from {special_zones_input}.")
    print(f"Pareto frontier contains {len(frontier)} points.")
    print(f"Wrote {output_path}")
    print(f"Wrote {frontier_csv_path}")
    print(f"Rendered frontier visualizations to {pareto_solutions_dir}")


if __name__ == "__main__":
    main()

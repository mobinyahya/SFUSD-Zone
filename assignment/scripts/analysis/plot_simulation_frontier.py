"""Plot a Pareto frontier from a folder of simulation assignment CSVs.

This is the student-assignment analogue of RA_SFUSD's
``runners/plot_frontier.py``. Instead of RA_SFUSD's metric stack, it reuses
this project's evaluation pipeline: every simulation CSV under
``data_folder`` is scored with the same ``MatchEvaluator`` /
``eval_assignment_full`` path as ``analyze_trends.py``. The two
frontier axes are therefore real "paper metrics" (by default average travel
distance vs. the high-FRL dissimilarity index), and the script extracts the
non-dominated (Pareto-optimal) simulations and plots them.

It does NOT solve any optimization: it reads simulations that already exist
and traces their frontier.

Usage:
    uv run python scripts/analysis/plot_simulation_frontier.py \
        --config configs/custom_configs/simulation_frontier.yaml

Config keys (YAML):
    data_folder: folder searched recursively for simulation ``*.csv`` files.
    year: 2-digit school year (e.g. 23 for 2023-24).
    program_data / student_data / schools_data: evaluator input CSVs.
    new_ctip_path: optional ``.npy`` equity-block file.
    x_metric / y_metric: metric names for the two axes (must be produced by
        ``eval_assignment_full``; run with ``--list-metrics`` to see
        the available names).
    x_minimize / y_minimize: whether lower is better on each axis (default
        true for both).
    group_iterations: if true (default), average all iterations of a policy
        (grouped by the top-level subfolder under ``data_folder``) into one
        point, so the frontier is over policies, not noisy single draws.
    annotate: if true (default), label the frontier points (de-overlapped
        with adjustText).
    current_policies_csv: optional CSV of baseline points to overlay as red
        X markers; needs a ``scenario`` column plus the x/y metric columns.
    output_dir: where ``results.csv``, ``frontier.csv`` and
        ``frontier_plot.png`` are written.
    label_parts: trailing path components kept as a point label (default 2).
    skip_substrings: filename substrings to ignore (default
        ``[utility_matrix, precomputed]``).
    recompute: if false and ``results.csv`` already exists, reuse it instead
        of re-evaluating every simulation (default true).
    title: plot title.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports: reuse the per-CSV evaluation from analyze_trends and the
# shared plotting helpers.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from analyze_trends import (  # noqa: E402
    _collect_csv_files,
    _evaluate_csv_worker,
    get_config,
)

from student_assignment.utils.plotting import (  # noqa: E402
    apply_plot_style,
    save_figure,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_X_METRIC = "Distance Av (All Assigned)"
DEFAULT_Y_METRIC = "Dissimilarity (High FRL)"
DEFAULT_SKIP_SUBSTRINGS = ["utility_matrix", "precomputed"]

# Number of zones implied by a policy/scenario name, used to colour points
# (matches RA_SFUSD's frontier plots).
ZONE_NAME_MAP = {"small_zones": 18.0, "medium_zones": 6.0}


def n_zones_from_key(group_key: str) -> float:
    """Infer the number of zones from a policy/scenario name.

    Args:
        group_key: Scenario/policy label (e.g. "small_zones+reserves",
            "6zone-1_...", "status_quo").

    Returns:
        The implied zone count: 18 for small zones, 6 for medium zones, the
        leading number of an ``<n>zone`` token, or 1.0 (citywide) otherwise.
    """
    for name, count in ZONE_NAME_MAP.items():
        if name in group_key:
            return count
    match = re.search(r"(\d+)[Zz]one|[Zz]ones?[_\-](\d+)", group_key)
    if match:
        return float(match.group(1) or match.group(2))
    return 1.0


def compute_pareto_frontier(
    points: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_minimize: bool = True,
    y_minimize: bool = True,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Extract the non-dominated (Pareto) points from a scatter.

    A point dominates another when it is at least as good on both axes and
    strictly better on one. The computation is fully vectorized (no row
    iteration): points are sorted by the x objective, and a point is kept
    when its y objective beats the best y seen among all strictly-better-x
    points.

    Args:
        points: DataFrame of candidate points (one row per simulation).
        x_col: Column name for the x-axis metric.
        y_col: Column name for the y-axis metric.
        x_minimize: If True, lower x is better; else higher is better.
        y_minimize: If True, lower y is better; else higher is better.
        tol: Numeric tolerance when comparing objectives.

    Returns:
        DataFrame of non-dominated points, ordered by the x objective.
    """
    working = points.dropna(subset=[x_col, y_col])
    if working.empty:
        return working

    # Map both axes to "lower is better" so one rule covers all cases.
    x_obj = working[x_col].to_numpy(dtype=float)
    y_obj = working[y_col].to_numpy(dtype=float)
    if not x_minimize:
        x_obj = -x_obj
    if not y_minimize:
        y_obj = -y_obj

    # Sort by x then y; the running min of y over earlier points (smaller x)
    # tells us whether the current point is dominated.
    order = np.lexsort((y_obj, x_obj))
    y_sorted = y_obj[order]
    best_before = np.concatenate(([np.inf], np.minimum.accumulate(y_sorted)[:-1]))
    keep = y_sorted <= best_before + tol

    frontier_positions = order[keep]
    frontier = working.iloc[frontier_positions].copy()
    return frontier.sort_values(x_col, ascending=x_minimize)


def _point_label(csv_path: str, label_parts: int) -> str:
    """Build a short point label from the trailing path components.

    Args:
        csv_path: Path to the simulation CSV.
        label_parts: Number of trailing path components to keep.

    Returns:
        A "/"-joined label with the ``.csv`` suffix stripped.
    """
    parts = Path(csv_path).with_suffix("").parts
    return "/".join(parts[-label_parts:]) if label_parts > 0 else Path(csv_path).stem


def _group_key(csv_path: str, data_folder: str) -> str:
    """Derive the policy/scenario label a simulation CSV belongs to.

    Simulations are typically laid out as
    ``<data_folder>/<scenario>/<variant>/..._iterationN.csv``; grouping by
    the top-level subfolder under ``data_folder`` collapses all iterations of
    a scenario into one averaged point (matching RA_SFUSD's frontier plots).

    Args:
        csv_path: Path to the simulation CSV.
        data_folder: Root folder the search started from.

    Returns:
        The first path component below ``data_folder``, or the file stem if
        the CSV sits directly in ``data_folder``.
    """
    try:
        relative = Path(csv_path).resolve().relative_to(Path(data_folder).resolve())
    except ValueError:
        return Path(csv_path).stem
    return relative.parts[0] if len(relative.parts) > 1 else Path(csv_path).stem


def evaluate_simulations(config: dict) -> pd.DataFrame:
    """Score every simulation CSV under ``data_folder`` with paper metrics.

    Args:
        config: Parsed config dict (see module docstring for keys).

    Returns:
        DataFrame with one row per simulation: a ``label`` column plus one
        column per paper metric.

    Raises:
        FileNotFoundError: If ``data_folder`` does not exist.
        ValueError: If no usable simulation CSV is found.
    """
    data_folder = config["data_folder"]
    if not Path(data_folder).exists():
        raise FileNotFoundError(f"data_folder does not exist: {data_folder}")

    skip_substrings = config.get("skip_substrings", DEFAULT_SKIP_SUBSTRINGS)
    label_parts = int(config.get("label_parts", 2))

    csv_files, _ = _collect_csv_files({"folder": data_folder})
    csv_files = [
        path
        for path in csv_files
        if not any(token in path for token in skip_substrings)
    ]
    if not csv_files:
        raise ValueError(f"No simulation CSV files found under {data_folder}")

    logger.info("Evaluating %d simulation files...", len(csv_files))
    year = int(config["year"])
    program_data = config["program_data"]
    student_data = config["student_data"]
    schools_data = config.get("schools_data")
    new_ctip_path = config.get("new_ctip_path")

    records: list[dict[str, object]] = []
    for index, csv_path in enumerate(csv_files, start=1):
        metrics = _evaluate_csv_worker(
            (
                year,
                program_data,
                csv_path,
                student_data,
                schools_data,
                new_ctip_path,
            )
        )
        if metrics is None:
            logger.warning("Skipping (evaluation failed): %s", csv_path)
            continue
        record: dict[str, object] = {
            "label": _point_label(csv_path, label_parts),
            "group_key": _group_key(csv_path, data_folder),
        }
        record.update({str(name): value for name, value in metrics.items()})
        records.append(record)
        if index % 25 == 0:
            logger.info("  ... %d / %d evaluated", index, len(csv_files))

    if not records:
        raise ValueError(f"No valid simulation results under {data_folder}")
    return pd.DataFrame.from_records(records)


def aggregate_by_policy(results: pd.DataFrame) -> pd.DataFrame:
    """Average per-iteration metrics into one point per policy.

    Args:
        results: Per-simulation metrics with a ``group_key`` column.

    Returns:
        One row per ``group_key`` with the mean of every numeric metric and
        ``label`` set to the group key.
    """
    numeric_cols = results.select_dtypes("number").columns.tolist()
    aggregated = results.groupby("group_key", as_index=False)[numeric_cols].mean()
    aggregated["label"] = aggregated["group_key"]
    return aggregated


def plot_frontier(
    points: pd.DataFrame,
    frontier: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    annotate: bool = True,
    current_policies: pd.DataFrame | None = None,
) -> None:
    """Plot policy points coloured by zone count (RA_SFUSD style).

    Every policy is a point coloured by its number of zones (viridis,
    shared with a "Number of Zones" colorbar). Dominated policies are faded;
    Pareto-optimal policies are bold with a black edge, joined by a dotted
    frontier line. Frontier labels are de-overlapped with ``adjustText``.

    Args:
        points: One row per policy (already aggregated).
        frontier: The non-dominated subset of ``points``.
        x_col: Metric name for the x-axis.
        y_col: Metric name for the y-axis.
        output_path: Where to save the PNG.
        title: Plot title.
        annotate: When True, label the frontier points (every point for
            <= 7 zones, one per distinct zone count above that).
        current_policies: Optional DataFrame of baseline points to overlay as
            red X markers; must have a ``scenario`` column plus ``x_col`` and
            ``y_col`` (e.g. the real/current assignment).
    """
    apply_plot_style()
    line_color = "#4CB8E0"
    text_color = "#2f2f2f"

    points = points.copy()
    points["n_zones"] = points["group_key"].map(n_zones_from_key)
    frontier = frontier.copy()
    frontier["n_zones"] = frontier["group_key"].map(n_zones_from_key)

    frontier_keys = set(frontier["group_key"])
    background = points[~points["group_key"].isin(frontier_keys)]

    # Colormap anchored to the zone-count range, shared by both scatters.
    norm = plt.Normalize(
        vmin=float(points["n_zones"].min()),
        vmax=float(points["n_zones"].max()),
    )
    cmap = plt.cm.viridis

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.margins(x=0.04, y=0.06)

    scatter_bg = axis.scatter(
        background[x_col],
        background[y_col],
        c=background["n_zones"],
        cmap=cmap,
        norm=norm,
        alpha=0.25,
        s=60,
        linewidths=0,
        zorder=2,
    )
    scatter_fg = axis.scatter(
        frontier[x_col],
        frontier[y_col],
        c=frontier["n_zones"],
        cmap=cmap,
        norm=norm,
        alpha=0.9,
        s=180,
        edgecolors="k",
        linewidths=1.5,
        zorder=4,
    )

    frontier_sorted = frontier.sort_values(x_col)
    if len(frontier_sorted) >= 2:
        axis.plot(
            frontier_sorted[x_col],
            frontier_sorted[y_col],
            color=line_color,
            linewidth=1.5,
            linestyle=":",
            zorder=3,
            label="Pareto frontier",
        )

    if annotate:
        # <= 7 zones: label every point. Above that: one label per distinct
        # zone count (otherwise the many-zone points overlap).
        seen_large_zones: set = set()
        texts = []
        for group_key, n_zones, x_val, y_val in zip(
            frontier_sorted["group_key"],
            frontier_sorted["n_zones"],
            frontier_sorted[x_col],
            frontier_sorted[y_col],
        ):
            if n_zones > 7:
                if n_zones in seen_large_zones:
                    continue
                seen_large_zones.add(n_zones)
            texts.append(
                axis.text(
                    x_val,
                    y_val,
                    str(group_key).split("+")[0],
                    fontsize=8,
                    fontweight="bold",
                    color=text_color,
                    zorder=5,
                )
            )
        if texts:
            adjust_text(
                texts,
                ax=axis,
                expand=(1.5, 2.0),
                add_objects=[scatter_bg, scatter_fg],
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

    # Current-policy overlay: real/baseline points as red X with a boxed label.
    if current_policies is not None and not current_policies.empty:
        axis.scatter(
            current_policies[x_col],
            current_policies[y_col],
            color="red",
            s=100,
            marker="X",
            label="Current assignments",
            zorder=6,
        )
        for scenario, x_val, y_val in zip(
            current_policies["scenario"],
            current_policies[x_col],
            current_policies[y_col],
        ):
            axis.annotate(
                str(scenario),
                (x_val, y_val),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

    # Colorbar at the bottom, keeping the right side free for the legend.
    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable, ax=axis, location="bottom", pad=0.12, shrink=0.6
    )
    colorbar.set_label("Number of Zones", fontsize=11)

    axis.set_xlabel(x_col)
    axis.set_ylabel(y_col)
    axis.set_title(title)
    axis.legend(loc="best")
    save_figure(output_path, fig=fig)
    logger.info("Saved frontier plot to %s", output_path)


def run(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate simulations, compute the frontier, and write all outputs.

    Args:
        config: Parsed config dict.

    Returns:
        Tuple of (policy points DataFrame, frontier DataFrame).
    """
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"

    x_col = config.get("x_metric", DEFAULT_X_METRIC)
    y_col = config.get("y_metric", DEFAULT_Y_METRIC)

    if not bool(config.get("recompute", True)) and results_path.exists():
        logger.info("Reusing precomputed metrics: %s", results_path)
        results = pd.read_csv(results_path)
    else:
        results = evaluate_simulations(config)
        results.to_csv(results_path, index=False)
        logger.info("Wrote per-simulation metrics to %s", results_path)

    for metric_name in (x_col, y_col):
        if metric_name not in results.columns:
            raise KeyError(
                f"Metric '{metric_name}' not found. Available metrics: "
                f"{sorted(c for c in results.columns if c not in ('label', 'group_key'))}"
            )

    # Average iterations into one point per policy (RA_SFUSD convention), so
    # the frontier is over policies rather than noisy individual draws.
    group_iterations = bool(config.get("group_iterations", True))
    if group_iterations and "group_key" in results.columns:
        points = aggregate_by_policy(results)
        unit = "policies"
    else:
        points = results.copy()
        if "group_key" not in points.columns:
            points["group_key"] = points["label"]
        unit = "simulations"

    frontier = compute_pareto_frontier(
        points,
        x_col=x_col,
        y_col=y_col,
        x_minimize=bool(config.get("x_minimize", True)),
        y_minimize=bool(config.get("y_minimize", True)),
    )
    frontier_path = output_dir / "frontier.csv"
    frontier.to_csv(frontier_path, index=False)
    logger.info(
        "Frontier: %d of %d %s are non-dominated -> %s",
        len(frontier),
        len(points),
        unit,
        frontier_path,
    )

    # Optional baseline/current-assignment overlay (CSV with a "scenario"
    # column plus the x/y metric columns).
    current_policies = None
    current_policies_csv = config.get("current_policies_csv")
    if current_policies_csv:
        current_policies = pd.read_csv(current_policies_csv)
        logger.info(
            "Overlaying %d current-policy point(s) from %s",
            len(current_policies),
            current_policies_csv,
        )

    plot_frontier(
        points,
        frontier,
        x_col=x_col,
        y_col=y_col,
        output_path=output_dir / "frontier_plot.png",
        title=config.get("title", "Pareto frontier from simulations"),
        annotate=bool(config.get("annotate", True)),
        current_policies=current_policies,
    )
    return points, frontier


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot a Pareto frontier from simulation assignment CSVs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML frontier config.",
    )
    args = parser.parse_args()

    config = get_config(args.config)
    run(config)


if __name__ == "__main__":
    main()

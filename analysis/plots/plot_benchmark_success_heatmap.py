#!/usr/bin/env python3
"""Plot benchmark success rate heatmaps across levels, solvers, and zone counts.

Generates publication-quality heatmap visualizations where:
- Each row is a solver (CP, Recom, Relaxed Recom, Short Bursts, Adaptive Short Bursts, CP (Multi-Level)).
- Columns are grouped by level (using the final level in multi-level lists).
- Under each level, columns represent centroids aggregated by number of zones.
- Results are aggregated across seeds and centroid variants.
- Separate heatmap PNGs are generated for each total solve time limit (summing recursive stage times).
- Each square is labeled with '<successes> / <total runs>'.
- A success is defined as FEASIBLE or OPTIMAL (failure is anything else).
- Cell colors follow a color gradient from lowest (0%) to highest (100%) success rate.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Success statuses according to specification
SUCCESS_STATUSES = {"FEASIBLE", "OPTIMAL"}

# Canonical hierarchy order for levels (coarse to fine)
LEVEL_HIERARCHY_ORDER = [
    "Block_4",
    "Block_3",
    "Block_2",
    "Block_1",
    "Block_0",
    "BlockGroup_2",
    "BlockGroup_1",
    "BlockGroup_0",
    "Tract_0",
]

# Canonical display order for solvers. Must stay identical to the labels and
# ordering of ``benchmark.plot_edges.METHODS``, so the same method carries the
# same name in the success heatmap and the cut-length figures. The sweep config
# calls the strategy ``recursive``; both figures display it as "Multi-Level".
SOLVER_DISPLAY_ORDER = [
    "Recom",
    "Relaxed Recom",
    "Short Bursts",
    "Adaptive Short Bursts",
    "MIP",
    "CP",
    "CP (Int)",
    "MIP (Multi-Level)",
    "CP (Multi-Level)",
    "CP (Int, Multi-Level)",
]


def parse_zone_sort_key(zone_label: str) -> tuple[int, str]:
    """Sort key for zone count strings like '6 Zones', '10 Zones', etc."""
    m = re.search(r"(\d+)", str(zone_label))
    if m:
        return (int(m.group(1)), str(zone_label))
    return (999, str(zone_label))


def parse_level_sort_key(level: str) -> tuple[int, str]:
    """Sort key for levels based on standard hierarchy or alphabetical fallback."""
    clean = str(level).strip()
    if clean in LEVEL_HIERARCHY_ORDER:
        return (LEVEL_HIERARCHY_ORDER.index(clean), clean)
    return (len(LEVEL_HIERARCHY_ORDER), clean)


def parse_solver_sort_key(solver: str) -> tuple[int, str]:
    """Sort key for solvers based on standard display order."""
    clean = str(solver).strip()
    if clean in SOLVER_DISPLAY_ORDER:
        return (SOLVER_DISPLAY_ORDER.index(clean), clean)
    return (len(SOLVER_DISPLAY_ORDER), clean)


def parse_time_limit_sort_key(tl: str) -> tuple[float, str]:
    """Sort key for time limits based on numerical seconds."""
    m = re.search(r"(\d+(?:\.\d+)?)", str(tl))
    if m:
        return (float(m.group(1)), str(tl))
    return (999999.0, str(tl))


def clean_level_value(val: Any) -> str:
    """Extract and normalize the last level in a level sequence or string."""
    if val is None:
        return "Unknown"
    if isinstance(val, (list, tuple)):
        return str(val[-1]) if len(val) > 0 else "Unknown"
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and len(parsed) > 0:
                return str(parsed[-1])
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    return str(parsed[-1])
            except Exception:
                pass
    parts = [p.strip() for p in s.split("->")]
    if parts:
        return parts[-1]
    return s


def clean_total_time_limit_value(val: Any) -> str:
    """Calculate total solve time limit sum and return formatted label (e.g., '600s')."""
    if val is None:
        return "Unknown"
    if isinstance(val, (list, tuple)):
        total = sum(float(x) for x in val)
        return f"{int(total)}s" if total.is_integer() else f"{total}s"
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and len(parsed) > 0:
                total = sum(float(x) for x in parsed)
                return f"{int(total)}s" if total.is_integer() else f"{total}s"
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    total = sum(float(x) for x in parsed)
                    return f"{int(total)}s" if total.is_integer() else f"{total}s"
            except Exception:
                pass
    try:
        num = float(s.rstrip("s"))
        return f"{int(num)}s" if num.is_integer() else f"{num}s"
    except ValueError:
        return s


def extract_zone_count_label(row: pd.Series | dict[str, Any]) -> str:
    """Extract number of zones and format as 'X Zones'."""
    if "num_zones" in row and not pd.isna(row["num_zones"]):
        try:
            val = int(float(row["num_zones"]))
            return f"{val} Zones"
        except Exception:
            pass

    cent = str(row.get("config_centroids_type") or row.get("centroids_type") or "").strip()
    m = re.match(r"^(\d+)-zone", cent)
    if m:
        return f"{m.group(1)} Zones"
    if cent:
        return cent
    return "Unknown"


def map_solver_display_name(row: pd.Series | dict[str, Any]) -> str:
    """Map solver and strategy to canonical display labels."""
    solver = str(row.get("config_solver") or row.get("solver") or "").strip().lower()
    strategy = str(row.get("config_strategy") or row.get("strategy") or "").strip().lower()

    if solver in {"cp_bool", "cp"}:
        return "CP (Multi-Level)" if strategy == "recursive" else "CP"
    elif solver in {"cp_int"}:
        return "CP (Int, Multi-Level)" if strategy == "recursive" else "CP (Int)"
    elif solver in {"mip"}:
        return "MIP (Multi-Level)" if strategy == "recursive" else "MIP"
    elif solver in {"recom"}:
        return "Recom"
    elif solver in {"relaxed_recom"}:
        return "Relaxed Recom"
    elif solver in {"short_bursts"}:
        return "Short Bursts"
    elif solver in {"adaptive_short_bursts"}:
        return "Adaptive Short Bursts"
    else:
        name = solver.replace("_", " ").title()
        if strategy == "recursive":
            name += " (Multi-Level)"
        return name


def format_time_limit_minutes(time_limit_str: Any) -> str:
    """Render a '600s' budget as '10 min', matching the cut-length figures.

    Only the displayed title changes: the filename slug and the CSV columns
    keep seconds, which is the unit the sweep config states budgets in.
    """
    m = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(time_limit_str).strip())
    if not m:
        return str(time_limit_str)
    return f"{float(m[1]) / 60.0:g} min"


def is_success_status(status_val: Any) -> bool:
    """Check whether a status indicates success (FEASIBLE or OPTIMAL)."""
    if pd.isna(status_val) or status_val is None:
        return False
    return str(status_val).strip().upper() in SUCCESS_STATUSES


def load_benchmark_data(benchmark_path: str | Path) -> pd.DataFrame:
    """Load benchmark data directly from summary.csv or a specified CSV file."""
    path = Path(benchmark_path).expanduser().resolve()

    if path.is_file() and path.suffix.lower() == ".csv":
        csv_file = path
    elif (path / "summary.csv").is_file():
        csv_file = path / "summary.csv"
    else:
        raise FileNotFoundError(
            f"No summary.csv found at '{benchmark_path}'."
        )

    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"No benchmark data rows loaded from '{csv_file}'.")

    return df


def prepare_benchmark_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and clean columns for benchmark heatmap analysis."""
    df = df.copy()

    # Solver display mapping
    solver_labels = df.apply(map_solver_display_name, axis=1)

    # Number of zones mapping
    zone_groups = df.apply(extract_zone_count_label, axis=1)

    # Level mapping (last level in list)
    if "config_levels" in df.columns:
        level_clean = df["config_levels"].apply(clean_level_value)
    elif "levels" in df.columns:
        level_clean = df["levels"].apply(clean_level_value)
    else:
        level_clean = pd.Series(["Default"] * len(df), index=df.index)

    # Total solve time limit mapping
    if "config_solve_time_limits" in df.columns:
        time_limit_clean = df["config_solve_time_limits"].apply(clean_total_time_limit_value)
    elif "solve_time_limits" in df.columns:
        time_limit_clean = df["solve_time_limits"].apply(clean_total_time_limit_value)
    else:
        time_limit_clean = pd.Series(["All"] * len(df), index=df.index)

    # Status & Success evaluation
    status_series = df["status"] if "status" in df.columns else pd.Series(index=df.index, dtype=object)
    final_status_series = df["final_status"] if "final_status" in df.columns else pd.Series(index=df.index, dtype=object)

    is_success = status_series.apply(is_success_status) | final_status_series.apply(is_success_status)

    df_cleaned = df.assign(
        solver_clean=solver_labels,
        zone_group=zone_groups,
        level_clean=level_clean,
        time_limit_clean=time_limit_clean,
        is_success=is_success,
    )

    return df_cleaned


def create_success_heatmap_matrix(
    df_subset: pd.DataFrame,
    ordered_levels: Sequence[str] | None = None,
    ordered_zones: Sequence[str] | None = None,
    ordered_solvers: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[list[str]], list[str], list[tuple[str, str]], list[str]]:
    """Construct the numerical success rate matrix and label grid.

    Returns:
        matrix: 2D np.ndarray with success rates (0.0 to 1.0, or np.nan for empty cells)
        labels: 2D list of formatted '<successes> / <total>' strings
        solvers: list of solver row names
        columns: list of (level, zone_group) column tuples
        levels: list of unique level groups in column order
    """
    if ordered_levels is None:
        raw_levels = df_subset["level_clean"].unique().tolist()
        ordered_levels = sorted(raw_levels, key=parse_level_sort_key)

    if ordered_zones is None:
        raw_zones = df_subset["zone_group"].unique().tolist()
        ordered_zones = sorted(raw_zones, key=parse_zone_sort_key)

    if ordered_solvers is None:
        raw_solvers = df_subset["solver_clean"].unique().tolist()
        ordered_solvers = sorted(raw_solvers, key=parse_solver_sort_key)

    # Build columns grouped by level: every zone count for level 1, then every zone count for level 2, etc.
    columns: list[tuple[str, str]] = []
    active_levels: list[str] = []

    for lvl in ordered_levels:
        lvl_df = df_subset[df_subset["level_clean"] == lvl]
        if lvl_df.empty:
            continue
        active_levels.append(lvl)
        lvl_zones = [z for z in ordered_zones if z in lvl_df["zone_group"].values]
        if not lvl_zones:
            lvl_zones = ordered_zones
        for z in lvl_zones:
            columns.append((lvl, z))

    n_rows = len(ordered_solvers)
    n_cols = len(columns)

    matrix = np.full((n_rows, n_cols), np.nan, dtype=float)
    labels = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    # Aggregate by (solver, level, zone_group) across all seeds and centroid variants
    agg = (
        df_subset.groupby(["solver_clean", "level_clean", "zone_group"])["is_success"]
        .agg(total="count", successes="sum")
        .reset_index()
    )
    agg_map = {
        (row["solver_clean"], row["level_clean"], row["zone_group"]): (
            int(row["successes"]),
            int(row["total"]),
        )
        for _, row in agg.iterrows()
    }

    for r_idx, solver in enumerate(ordered_solvers):
        for c_idx, (lvl, z) in enumerate(columns):
            key = (solver, lvl, z)
            if key in agg_map:
                successes, total = agg_map[key]
                rate = successes / total if total > 0 else np.nan
                matrix[r_idx, c_idx] = rate
                labels[r_idx][c_idx] = f"{successes} / {total}"
            else:
                matrix[r_idx, c_idx] = np.nan
                labels[r_idx][c_idx] = "-"

    return matrix, labels, list(ordered_solvers), columns, active_levels


def render_heatmap_figure(
    matrix: np.ndarray,
    labels: list[list[str]],
    solvers: list[str],
    columns: list[tuple[str, str]],
    active_levels: list[str],
    time_limit_str: str,
    output_path: Path,
    title_suffix: str = "",
    cmap_name: str = "RdYlGn",
) -> None:
    """Render and save a high-quality heatmap figure."""
    n_rows, n_cols = matrix.shape

    # Figure dimensions tailored to grid size
    cell_width = 1.25
    cell_height = 0.95
    fig_width = max(8.0, n_cols * cell_width + 4.5)
    fig_height = max(5.0, n_rows * cell_height + 3.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    # Style: elegant light theme with crisp grid
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    # Colormap with bad/masked color for missing cells
    base_cmap = mpl.colormaps.get_cmap(cmap_name).copy()
    if hasattr(base_cmap, "with_extremes"):
        base_cmap = base_cmap.with_extremes(bad="#e2e8f0")
    else:
        base_cmap.set_bad(color="#e2e8f0")

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(
        matrix,
        cmap=base_cmap,
        norm=norm,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )

    # Add cell dividers
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", size=0)

    # Row labels (Solvers)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(solvers, fontsize=12, fontweight="bold", color="#1e293b")
    ax.set_ylabel("Solver", fontsize=13, fontweight="bold", color="#0f172a", labelpad=12)

    # Column labels (Zone counts)
    zone_labels = [z for (_, z) in columns]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        zone_labels,
        fontsize=11,
        fontweight="normal",
        color="#334155",
        rotation=30,
        ha="right",
    )
    ax.set_xlabel("Number of Zones", fontsize=13, fontweight="bold", color="#0f172a", labelpad=10)

    # Annotate cells with '<successes> / <total>'
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            label = labels[i][j]
            if np.isnan(val):
                text_color = "#94a3b8"
                font_weight = "normal"
            else:
                rgba = base_cmap(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "#ffffff" if luminance < 0.48 else "#0f172a"
                font_weight = "bold"

            text = ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight=font_weight,
            )
            if text_color == "#ffffff":
                text.set_path_effects([
                    pe.withStroke(linewidth=1.5, foreground="#1e293b", alpha=0.7)
                ])

    # Group columns by level: draw prominent separators and top level-group banners
    level_col_indices: dict[str, list[int]] = {}
    for idx, (lvl, _) in enumerate(columns):
        level_col_indices.setdefault(lvl, []).append(idx)

    # Draw vertical separator lines between distinct level blocks
    for lvl, indices in list(level_col_indices.items())[:-1]:
        boundary = indices[-1] + 0.5
        ax.axvline(x=boundary, color="#0f172a", linewidth=2.5, zorder=5)

    # Draw top banners for Level groups
    y_top_pos = -0.75
    for lvl, indices in level_col_indices.items():
        min_c, max_c = min(indices), max(indices)
        mid_c = (min_c + max_c) / 2.0

        ax.text(
            mid_c,
            y_top_pos,
            f"Level: {lvl}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(
                boxstyle="round,pad=0.35,rounding_size=0.2",
                facecolor="#f1f5f9",
                edgecolor="#94a3b8",
                linewidth=1.2,
            ),
        )

    # Adjust top limit to make room for level banners
    ax.set_ylim(n_rows - 0.5, -1.35)

    # Spines styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#94a3b8")
        spine.set_linewidth(1.5)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.78, aspect=20)
    cbar.set_label("Success Rate (Feasible / Optimal)", fontsize=11, fontweight="bold", color="#1e293b", labelpad=10)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.ax.tick_params(labelsize=10, colors="#334155")
    cbar.outline.set_edgecolor("#94a3b8")

    # Title & Subtitle
    title = (
        "Optimization Benchmark Success Rate "
        f"(Time Limit: {format_time_limit_minutes(time_limit_str)})"
    )
    if title_suffix:
        title += f" - {title_suffix}"
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#0f172a", y=0.98)
    ax.set_title(
        "Aggregated across seeds & centroid variants  •  Cell labels: <successes> / <total runs>",
        fontsize=10.5,
        color="#64748b",
        pad=28,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to: {output_path}")


def generate_benchmark_heatmaps(
    benchmark_dir: str | Path,
    output_dir: str | Path | None = None,
    cmap: str = "RdYlGn",
    title_suffix: str = "",
) -> list[Path]:
    """Generate separate success rate heatmap PNGs for each time limit in the benchmark."""
    benchmark_path = Path(benchmark_dir).expanduser().resolve()
    df_raw = load_benchmark_data(benchmark_path)
    df = prepare_benchmark_dataframe(df_raw)

    if output_dir is None:
        base_dir = benchmark_path if benchmark_path.is_dir() else benchmark_path.parent
        out_dir = base_dir / "plots"
    else:
        out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify all distinct total time limits, sorted numerically
    time_limits = sorted(df["time_limit_clean"].unique(), key=parse_time_limit_sort_key)
    print(f"Found {len(df)} total runs across {len(time_limits)} total time limit budget(s): {time_limits}")

    output_files: list[Path] = []

    for tl in time_limits:
        df_tl = df[df["time_limit_clean"] == tl]
        print(f"\nProcessing Total Time Limit: {tl} ({len(df_tl)} runs)")

        matrix, labels, solvers, columns, active_levels = create_success_heatmap_matrix(df_tl)

        # Print terminal summary table
        col_headers = [f"{lvl}:{z}" for lvl, z in columns]
        summary_df = pd.DataFrame(labels, index=solvers, columns=col_headers)
        print(summary_df.to_string())

        tl_slug = re.sub(r"[^\w]+", "_", str(tl)).strip("_")
        out_file = out_dir / f"success_rate_heatmap_tl_{tl_slug}.png"

        render_heatmap_figure(
            matrix=matrix,
            labels=labels,
            solvers=solvers,
            columns=columns,
            active_levels=active_levels,
            time_limit_str=tl,
            output_path=out_file,
            title_suffix=title_suffix,
            cmap_name=cmap,
        )
        output_files.append(out_file)

    return output_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create heatmaps of benchmark success rates across solvers, levels, and zone counts."
    )
    parser.add_argument(
        "benchmark_dir",
        nargs="?",
        default="/soalnas/share/data/school_choice/local_runs/benchmark_v1_5",
        help="Path to benchmark directory containing summary.csv or direct path to summary.csv (default: /soalnas/share/data/school_choice/local_runs/benchmark_v1_5)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to save generated heatmap PNGs (defaults to <benchmark_dir>/plots)",
    )
    parser.add_argument(
        "--cmap",
        default="RdYlGn",
        help="Matplotlib colormap for success rate gradient (default: RdYlGn, e.g. YlGn, viridis, Blues)",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional title suffix for the plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_files = generate_benchmark_heatmaps(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        cmap=args.cmap,
        title_suffix=args.title,
    )
    print(f"\nSuccessfully generated {len(output_files)} heatmap plot(s).")


if __name__ == "__main__":
    main()

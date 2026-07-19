"""Generate an analyze_trends config and produce the sensitivity Excel report.

After the DA has been run for each utility_gap (via run_sensitivity_experiment.sh),
this script:
  1. Discovers all run folders under {output_dir}/runs/.
  2. Generates a ``sensitivity_trends_config.yaml`` in analyze_trends format.
  3. Delegates to ``scripts/analyze_trends.py`` to compute metrics and export
     the standard ``metrics_comparison.xlsx`` (Mean Values / Std Values /
     Mean ± Std sheets).
  4. Adds a normalised comparison plot as a bonus (sensitivity_plot.png).

Usage:
    python scripts/compare_sensitivity_results.py \\
        --output-dir metrics/sensitivity_real_vs_estimates/ \\
        --year 2223
"""

import logging
import pathlib
import subprocess
import sys

import click
import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from student_assignment.utils.plotting import (
    apply_plot_style,
    get_categorical_palette,
    save_figure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Metrics to highlight in the bonus normalised plot
PLOT_METRICS: list[str] = [
    "Dissimilarity (High FRL)",
    "AALPI in school with +10% FRL",
    "AALPI in school with +15% FRL",
    "Prop students in schools above +15% district FRL (All Assigned)",
    "Distance Av (All Assigned)",
    "Prop Top 1 choice (All Assigned)",
    "Prop Top 3 choice (All Assigned)",
    "Unassigned",
]


# ---------------------------------------------------------------------------
# Run-folder discovery
# ---------------------------------------------------------------------------


def _find_subconfig_folder(run_folder: pathlib.Path) -> pathlib.Path | None:
    """Return the first non-precomputed subfolder that contains CSV files.

    The DA simulator saves assignments under a subconfig subfolder
    (e.g. ``status_quo`` or ``status_quo_real``). This function locates it.

    Args:
        run_folder: Root of the DA run (e.g. runs/gap_1p0/).

    Returns:
        Path to the subfolder, or None if none found.
    """
    if not run_folder.is_dir():
        return None
    for subfolder in sorted(run_folder.iterdir()):
        if not subfolder.is_dir():
            continue
        if subfolder.name == "precomputed":
            continue
        if any(subfolder.rglob("*.csv")):
            return subfolder
    return None


def discover_runs(
    runs_dir: pathlib.Path,
) -> list[tuple[str, pathlib.Path]]:
    """Discover sensitivity runs sorted as: real first, then gaps ascending.

    Args:
        runs_dir: Directory containing ``real/`` and ``gap_*/`` sub-folders.

    Returns:
        List of (label, subconfig_folder) pairs in display order.
    """
    result: list[tuple[str, pathlib.Path]] = []

    real_dir = runs_dir / "real"
    if real_dir.exists():
        sub = _find_subconfig_folder(real_dir)
        if sub is not None:
            result.append(("real", sub))
        else:
            log.warning(
                "No assignment CSVs found under %s — skipping.", real_dir
            )

    def _gap_sort_key(folder: pathlib.Path) -> float:
        label = folder.name  # e.g. gap_1p0
        return float(label[4:].replace("p", "."))

    gap_folders = sorted(
        [
            f
            for f in runs_dir.iterdir()
            if f.is_dir() and f.name.startswith("gap_")
        ],
        key=_gap_sort_key,
        reverse=True,  # largest gap (closest to real ordering) first
    )
    for gf in gap_folders:
        sub = _find_subconfig_folder(gf)
        if sub is not None:
            result.append((gf.name, sub))
        else:
            log.warning("No assignment CSVs found under %s — skipping.", gf)

    return result


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def build_analyze_trends_config(
    runs: list[tuple[str, pathlib.Path]],
    output_dir: pathlib.Path,
    year: str,
) -> dict:
    """Build an analyze_trends-compatible config dict.

    Args:
        runs: List of (label, subconfig_folder) pairs.
        output_dir: Root output directory for the Excel file.
        year: 4-character year string (e.g. '2223').

    Returns:
        Config dict ready to serialise as YAML.
    """
    year_int = int(year[:2])  # 2-digit year used by analyze_trends
    student_data = str(
        PROJECT_ROOT
        / "local-data"
        / "student_filter"
        / f"student_{year}_filtered.csv"
    )
    program_data = str(
        PROJECT_ROOT
        / "local-data"
        / "program_filter"
        / f"programs_without_specialprogs_{year}.csv"
    )

    run_entries = [
        {
            "label": label,
            "folder": str(subconfig_folder),
            "year": year_int,
            "program_data": program_data,
            "student_data": student_data,
        }
        for label, subconfig_folder in runs
    ]

    return {
        "output_dir": str(output_dir),
        "runs": run_entries,
        "row_order": [
            "Distance Av (All Assigned)",
            "Distance < 0.5 (All Assigned)",
            "Distance > 3 (All Assigned)",
            "#Schools above 10% district FRL",
            "#Schools above 10% district FRL (Non-Designated)",
            "#Schools above 15% district FRL",
            "#Schools above 15% district FRL (Non-Designated)",
            "AALPI in school with +10% FRL",
            "AALPI in school with +15% FRL",
            "#Students in schools above 10% district FRL",
            "#Students in schools above 15% district FRL",
            "Prop students in schools above +15% district FRL (All Assigned)",
            "Dissimilarity (High FRL)",
            "Black/White exposure to poverty",
            "#Schools with -10% High Income (95292)",
            "#Schools with -15% High Income (95292)",
            "Dissimilarity (Income below 95292)",
            "#GE programs that have 1-4 African American or Pacific Islander students",
            "Unassigned",
            "Designated",
            "Designated or Unassigned",
            "Prop Top 1 choice (All Assigned)",
            "Prop Top 3 choice (All Assigned)",
            "Top 1 in-zone choice (All Assigned)",
            "Top 3 in-zone choice (All Assigned)",
            "Prop Distance > 3 and Rank>=5 (All Assigned)",
            "Variance of rank (All Assigned)",
            "Variance of in-zone rank (All Assigned)",
            "Variance of distance (All Assigned)",
            "Top 3 in-zone non-desig choice All Assigned (non-CTIP)",
            "Number of assigned students (non-CTIP)",
            "Number of designated students (non-CTIP)",
            "Number of unassigned students (non-CTIP)",
            "Prop designated or unassigned students (non-CTIP)",
            "Prop designated students (non-CTIP)",
            "Prop designated students All Assigned (non-CTIP)",
            "Distance Av All Assigned (Black)",
            "Distance < 0.5 All Assigned (Black)",
            "Distance > 3 All Assigned (Black)",
            "Prop students in schools above +15% district FRL (Black)",
            "Prop Top 1 non-desig choice All Assigned (Black)",
            "Prop Top 3 non-desig choice All Assigned (Black)",
            "Prop Distance > 3 and (Rank>=5 or designated) (Black)",
            "Distance Av All Assigned (Asian)",
            "Distance < 0.5 All Assigned (Asian)",
            "Distance > 3 All Assigned (Asian)",
            "Prop students in schools above +15% district FRL (Asian)",
            "Prop Top 1 non-desig choice All Assigned (Asian)",
            "Prop Top 3 non-desig choice All Assigned (Asian)",
            "Prop Distance > 3 and (Rank>=5 or designated) (Asian)",
            "Distance Av All Assigned (Hispanic)",
            "Distance < 0.5 All Assigned (Hispanic)",
            "Distance > 3 All Assigned (Hispanic)",
            "Prop students in schools above +15% district FRL (Hispanic)",
            "Prop Top 1 non-desig choice All Assigned (Hispanic)",
            "Prop Top 3 non-desig choice All Assigned (Hispanic)",
            "Prop Distance > 3 and (Rank>=5 or designated) (Hispanic)",
            "Distance Av All Assigned (White)",
            "Distance < 0.5 All Assigned (White)",
            "Distance > 3 All Assigned (White)",
            "Prop students in schools above +15% district FRL (White)",
            "Prop Top 1 non-desig choice All Assigned (White)",
            "Prop Top 3 non-desig choice All Assigned (White)",
            "Prop Distance > 3 and (Rank>=5 or designated) (White)",
            "Distance Av All Assigned (High FRL)",
            "Distance < 0.5 All Assigned (High FRL)",
            "Distance > 3 All Assigned (High FRL)",
            "Prop students in schools above +15% district FRL (High FRL)",
            "Prop Top 1 non-desig choice All Assigned (High FRL)",
            "Prop Top 3 non-desig choice All Assigned (High FRL)",
            "Prop Distance > 3 and (in-zone Rank>=5 or designated) (High FRL)",
            "Distance Av All Assigned (Low FRL)",
            "Distance < 0.5 All Assigned (Low FRL)",
            "Distance > 3 All Assigned (Low FRL)",
            "Prop students in schools above +15% district FRL (Low FRL)",
            "Prop Top 1 non-desig choice All Assigned (Low FRL)",
            "Prop Top 3 non-desig choice All Assigned (Low FRL)",
            "Prop Distance > 3 and (Rank>=5 or designated) (Low FRL)",
            "Distance Av All Assigned (CTIP)",
            "Distance < 0.5 All Assigned (CTIP)",
            "Distance > 3 All Assigned (CTIP)",
            "Prop students in schools above +15% district FRL (CTIP)",
            "Prop Top 1 non-desig choice All Assigned (CTIP)",
            "Prop Top 3 non-desig choice All Assigned (CTIP)",
            "Prop Distance > 3 and (Rank>=5 or designated) (CTIP)",
            "Distance Av All Assigned (non-CTIP)",
            "Distance < 0.5 All Assigned (non-CTIP)",
            "Distance > 3 All Assigned (non-CTIP)",
            "Prop students in schools above +15% district FRL (non-CTIP)",
            "Prop Top 1 non-desig choice All Assigned (non-CTIP)",
            "Prop Top 3 non-desig choice All Assigned (non-CTIP)",
            "Prop Distance > 3 and (Rank>=5 or designated) (non-CTIP)",
        ],
    }


# ---------------------------------------------------------------------------
# Bonus normalised plot
# ---------------------------------------------------------------------------


def _gap_label_to_float(label: str) -> float | None:
    """Convert a run label like 'gap_1p0' to a float (1.0).

    Args:
        label: Run label string.

    Returns:
        Float value, or None for non-gap labels (e.g. 'real').
    """
    if not label.startswith("gap_"):
        return None
    return float(label[4:].replace("p", "."))


def plot_normalised_sensitivity(
    excel_path: pathlib.Path,
    output_path: pathlib.Path,
    metrics: list[str],
) -> None:
    """Plot metrics normalised to the real baseline vs utility gap.

    Reads the ``Mean Values`` sheet from the Excel produced by analyze_trends
    and draws one line per metric on a log-scaled X axis.

    Args:
        excel_path: Path to ``metrics_comparison.xlsx``.
        output_path: Destination PNG path.
        metrics: Metric names to include in the plot.
    """
    if not excel_path.exists():
        log.warning("Excel not found at %s — skipping plot.", excel_path)
        return

    df = pd.read_excel(excel_path, sheet_name="Mean Values", index_col=0)

    gap_cols = [c for c in df.columns if str(c).startswith("gap_")]
    if not gap_cols:
        log.warning("No gap columns found in Excel — skipping plot.")
        return

    gap_cols_sorted = sorted(
        gap_cols,
        key=lambda c: _gap_label_to_float(str(c)) or 0,
    )
    gap_floats = [_gap_label_to_float(str(c)) for c in gap_cols_sorted]

    real_col = "real" if "real" in df.columns else None

    apply_plot_style()
    available = [m for m in metrics if m in df.index]
    palette = get_categorical_palette(len(available))

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, metric in enumerate(available):
        raw_real = df.loc[metric, real_col] if real_col else 1.0
        real_val: float = float(raw_real)  # type: ignore[arg-type]
        norm = real_val if (not pd.isna(real_val) and real_val != 0) else 1.0

        y_vals: list[float] = [
            float(df.loc[metric, col]) / norm  # type: ignore[arg-type]
            if not pd.isna(df.loc[metric, col])
            else float("nan")
            for col in gap_cols_sorted
        ]
        ax.plot(
            [g for g in gap_floats if g is not None],
            y_vals,
            marker="o",
            label=metric,
            color=palette[i],
        )

    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="grey", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Utility gap between consecutive ranks (log scale)")
    ax.set_ylabel("Value normalised to real-preferences baseline (= 1.0)")
    ax.set_title(
        "Sensitivity of DA outcomes to utility gap strength\n"
        "(real preferences baseline shown as grey dashed line at 1.0)"
    )
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
    )
    save_figure(output_path, fig=fig)
    log.info("Normalised plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--output-dir",
    required=True,
    help="Root output directory (contains runs/ sub-folder).",
)
@click.option(
    "--year",
    default="2223",
    show_default=True,
    help="SFUSD school year (e.g. 2223).",
)
@click.option(
    "--workers",
    default=None,
    type=int,
    help="Parallel workers for analyze_trends (default: all CPUs).",
)
def main(output_dir: str, year: str, workers: int | None) -> None:
    """Generate Excel metrics report for sensitivity runs.

    Wraps analyze_trends.py to produce the standard
    metrics_comparison.xlsx (Mean Values / Std Values / Mean ± Std).

    Args:
        output_dir: Root directory containing runs/ sub-folder.
        year: SFUSD school year string.
        workers: Parallel workers for analyze_trends.
    """
    log.info("Comparing sensitivity results for year %s", year)
    out_root = pathlib.Path(output_dir)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root = out_root.resolve()

    runs_dir = out_root / "runs"
    if not runs_dir.exists():
        log.error("Runs directory not found: %s", runs_dir)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Discover runs
    # ------------------------------------------------------------------
    runs = discover_runs(runs_dir)
    if not runs:
        log.error("No completed runs found under %s", runs_dir)
        sys.exit(1)

    log.info("Found %d run(s): %s", len(runs), [r[0] for r in runs])

    # ------------------------------------------------------------------
    # Generate analyze_trends config
    # ------------------------------------------------------------------
    trends_cfg = build_analyze_trends_config(runs, out_root, year)
    trends_cfg_path = out_root / "sensitivity_trends_config.yaml"
    with trends_cfg_path.open("w") as fh:
        yaml.dump(trends_cfg, fh, default_flow_style=False, allow_unicode=True)
    log.info("analyze_trends config written to %s", trends_cfg_path)

    # ------------------------------------------------------------------
    # Run analyze_trends → metrics_comparison.xlsx
    # ------------------------------------------------------------------
    analyze_script = str(PROJECT_ROOT / "scripts" / "analyze_trends.py")
    cmd = [sys.executable, analyze_script, "--config", str(trends_cfg_path)]
    if workers is not None:
        cmd += ["--workers", str(workers)]

    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    # ------------------------------------------------------------------
    # Bonus: normalised comparison plot
    # ------------------------------------------------------------------
    excel_path = out_root / "metrics_comparison.xlsx"
    plot_path = out_root / "sensitivity_plot.png"
    plot_normalised_sensitivity(excel_path, plot_path, PLOT_METRICS)

    log.info("Done.")
    log.info("  Excel : %s", excel_path)
    log.info("  Plot  : %s", plot_path)


if __name__ == "__main__":
    main()

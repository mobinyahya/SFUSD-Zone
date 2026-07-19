"""Generate synthetic estimates CSVs and DA configs for sensitivity analysis.

For each utility_gap value, produces:
  - An estimates CSV where programs in the student's real R1 list get
    utility = -(rank-1) * utility_gap, and all other programs are NaN
    (which the DA simulator treats as -inf).
  - A config YAML ready to be fed to run_custom_config.py.

Also produces a baseline config for the real-preferences run (no utility
model, uses actual R1 lists directly).

Usage:
    python scripts/generate_sensitivity_estimates.py \\
        --year 2223 \\
        --gaps 1.0 0.5 0.1 0.01 \\
        --iterations 5 \\
        --output-dir metrics/sensitivity_real_vs_estimates/
"""

import ast
import logging
import pathlib
import sys

import click
import pandas as pd
import yaml

sys.path.append(str(pathlib.Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Template configs used as base for YAML generation
_ESTIMATES_TEMPLATE = (
    PROJECT_ROOT
    / "configs"
    / "custom_configs"
    / "t7_2223_k3_prog_gesplit_2223.yaml"
)
_REAL_TEMPLATE = (
    PROJECT_ROOT / "configs" / "custom_configs" / "status_quo_real_2223.yaml"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_list_col(series: pd.Series) -> pd.Series:
    """Parse a stringified-list column with ast.literal_eval.

    Args:
        series: Pandas Series with string-encoded Python lists.

    Returns:
        Series of Python lists.
    """

    def _safe_parse(val: str) -> list:  # type: ignore[override]
        try:
            result = ast.literal_eval(val)
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []

    return series.apply(_safe_parse)


def _year_prefix(year_str: str) -> str:
    """Return the 2-digit year prefix used in estimates studentno column.

    Args:
        year_str: 4-character year string, e.g. '2223'.

    Returns:
        2-character prefix, e.g. '22'.
    """
    return year_str[:2]


def _build_rankings_long(
    df_students: pd.DataFrame,
    year_str: str,
) -> pd.DataFrame:
    """Build a long-format DataFrame of (studentno_key, program_id, rank).

    Args:
        df_students: Student DataFrame with columns studentno, grade,
            r1_ranked_idschool, r1_programs (both as Python lists).
        year_str: 4-character year string (e.g. '2223').

    Returns:
        DataFrame with columns [studentno_key, program_id, rank].
    """
    prefix = _year_prefix(year_str)
    records = []
    for row in df_students.itertuples(index=False):
        key = f"{prefix}-{row.studentno}"
        schools: list = row.r1_ranked_idschool  # type: ignore[assignment]
        prog_types: list = row.r1_programs  # type: ignore[assignment]
        grade: str = row.grade  # type: ignore[assignment]
        for rank_0, (school, ptype) in enumerate(zip(schools, prog_types)):
            prog_id = f"{int(school)}-{ptype}-{grade}"
            records.append((key, prog_id, rank_0))
    return pd.DataFrame(
        records, columns=["studentno_key", "program_id", "rank_0"]
    )


def build_synthetic_estimates(
    df_students: pd.DataFrame,
    year_str: str,
    utility_gap: float,
) -> pd.DataFrame:
    """Build a wide estimates DataFrame for one utility_gap value.

    Programs in the student's real R1 list receive:
        utility(rank_i) = -rank_0 * utility_gap   (rank_0 is 0-based)
    All other programs remain NaN (→ -inf in the DA simulator).

    Every student from df_students is guaranteed to appear as a row,
    even those with no ranked programs (all-NaN row).  This prevents
    KeyError in UtilityModel when it looks up students that have no
    R1 preferences but are still part of the DA market.

    Args:
        df_students: Student DataFrame with parsed list columns
            (ALL students, including those with no preferences).
        year_str: 4-character year string (e.g. '2223').
        utility_gap: Utility gap between consecutive ranks.

    Returns:
        Wide DataFrame indexed by studentno_key with program IDs as columns.
    """
    prefix = _year_prefix(year_str)
    all_keys = df_students["studentno"].apply(lambda sid: f"{prefix}-{sid}")

    long_df = _build_rankings_long(df_students, year_str)
    long_df["utility"] = -long_df["rank_0"].astype(float) * utility_gap

    wide = long_df.pivot(
        index="studentno_key", columns="program_id", values="utility"
    )
    wide.index.name = "studentno"
    wide.columns.name = None

    # Reindex to include ALL students; those with no preferences get NaN
    # rows (treated as -inf by UtilityModel).
    wide = wide.reindex(all_keys.values)
    return wide


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _load_template(path: pathlib.Path) -> dict:
    """Load a YAML config template.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dict.
    """
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _year_data_paths(year_str: str) -> dict:
    """Return the student, program and school data paths for a given year.

    Args:
        year_str: 4-character year string (e.g. '2324').

    Returns:
        Dict with keys 'student-data', 'program-data', 'school-data'.
    """
    return {
        "student-data": str(
            PROJECT_ROOT
            / "local-data"
            / "student_filter"
            / f"student_{year_str}_filtered.csv"
        ),
        "program-data": str(
            PROJECT_ROOT
            / "local-data"
            / "program_filter"
            / f"programs_without_specialprogs_{year_str}.csv"
        ),
        "school-data": f"Cleaned/schools_rehauled_{year_str}.csv",
    }


def generate_estimates_config(
    template_path: pathlib.Path,
    estimates_csv_path: pathlib.Path,
    assignment_folder: pathlib.Path,
    iterations: int,
    year_str: str,
) -> dict:
    """Build a DA config dict for an estimates-based run.

    Args:
        template_path: Path to the estimates-based YAML template.
        estimates_csv_path: Absolute path to the synthetic estimates CSV.
        assignment_folder: Absolute path where DA should save assignments.
        iterations: Number of DA iterations to run.
        year_str: 4-character year string (e.g. '2223').

    Returns:
        Modified config dict.
    """
    cfg = _load_template(template_path)
    cfg["paths"].update(_year_data_paths(year_str))
    cfg["paths"]["estimate-path"] = str(estimates_csv_path)
    cfg["paths"]["assignment-folder"] = str(assignment_folder) + "/"
    cfg["paths"]["student-save"] = str(assignment_folder / "precomputed") + "/"
    cfg["utility-model"]["enable"] = True
    cfg["iterations"]["end"] = iterations
    cfg["year"] = int(year_str[:2])
    return cfg


def generate_real_config(
    template_path: pathlib.Path,
    assignment_folder: pathlib.Path,
    iterations: int,
    year_str: str,
) -> dict:
    """Build a DA config dict for the real-preferences baseline run.

    Args:
        template_path: Path to the real-preferences YAML template.
        assignment_folder: Absolute path where DA should save assignments.
        iterations: Number of DA iterations to run.
        year_str: 4-character year string (e.g. '2223').

    Returns:
        Modified config dict.
    """
    cfg = _load_template(template_path)
    cfg["paths"].update(_year_data_paths(year_str))
    cfg["paths"]["assignment-folder"] = str(assignment_folder) + "/"
    cfg["paths"]["student-save"] = str(assignment_folder / "precomputed") + "/"
    cfg["iterations"]["end"] = iterations
    cfg["year"] = int(year_str[:2])
    return cfg


def _save_yaml(cfg: dict, path: pathlib.Path) -> None:
    """Write a config dict to a YAML file.

    Args:
        cfg: Config dict to serialise.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--year",
    default="2223",
    show_default=True,
    help="SFUSD school year (4-char string, e.g. 2223).",
)
@click.option(
    "--gaps",
    default="1.0 0.5 0.1 0.01",
    show_default=True,
    help="Space-separated utility gap values.",
)
@click.option(
    "--iterations",
    default=5,
    show_default=True,
    help="Number of DA iterations per run.",
)
@click.option(
    "--output-dir",
    default="metrics/sensitivity_real_vs_estimates",
    show_default=True,
    help="Output directory (relative to project root or absolute).",
)
def main(
    year: str,
    gaps: str,
    iterations: int,
    output_dir: str,
) -> None:
    """Generate synthetic estimates CSVs and DA config YAMLs.

    Args:
        year: SFUSD school year string.
        gaps: Space-separated utility gap values.
        iterations: DA iterations per run.
        output_dir: Root output directory.
    """
    gap_values: list[float] = [float(g) for g in gaps.split()]
    out_root = pathlib.Path(output_dir)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root = out_root.resolve()

    estimates_dir = out_root / "estimates"
    configs_dir = out_root / "configs"
    estimates_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load student data
    # ------------------------------------------------------------------
    student_path = (
        PROJECT_ROOT
        / "local-data"
        / "student_filter"
        / f"student_{year}_filtered.csv"
    )
    log.info("Loading student data from %s", student_path)
    df_raw = pd.read_csv(student_path, low_memory=False)

    needed_cols = ["studentno", "grade", "r1_ranked_idschool", "r1_programs"]
    df_students = df_raw[needed_cols].copy()
    df_students["r1_ranked_idschool"] = _parse_list_col(
        df_students["r1_ranked_idschool"]
    )
    df_students["r1_programs"] = _parse_list_col(df_students["r1_programs"])

    n_with_prefs = df_students["r1_ranked_idschool"].apply(len).gt(0).sum()
    log.info(
        "%d / %d students have at least one ranked program"
        " (%d without preferences — included as all-NaN rows in estimates)",
        n_with_prefs,
        len(df_students),
        len(df_students) - n_with_prefs,
    )

    # ------------------------------------------------------------------
    # Generate estimates CSVs + configs for each gap value
    # ------------------------------------------------------------------
    generated = []

    for gap in gap_values:
        gap_label = str(gap).replace(".", "p")
        log.info("Building estimates for utility_gap=%.4f ...", gap)

        estimates_df = build_synthetic_estimates(df_students, year, gap)
        estimates_path = estimates_dir / f"estimates_gap_{gap_label}.csv"
        estimates_df.to_csv(estimates_path)
        log.info(
            "  Saved %d students × %d programs → %s",
            len(estimates_df),
            len(estimates_df.columns),
            estimates_path,
        )

        assignment_folder = out_root / "runs" / f"gap_{gap_label}"
        cfg = generate_estimates_config(
            template_path=_ESTIMATES_TEMPLATE,
            estimates_csv_path=estimates_path,
            assignment_folder=assignment_folder,
            iterations=iterations,
            year_str=year,
        )
        config_path = configs_dir / f"config_gap_{gap_label}.yaml"
        _save_yaml(cfg, config_path)
        log.info("  Config → %s", config_path)
        generated.append(("gap_" + gap_label, config_path))

    # ------------------------------------------------------------------
    # Real-preferences baseline config
    # ------------------------------------------------------------------
    log.info("Generating real-preferences baseline config ...")
    real_assignment_folder = out_root / "runs" / "real"
    real_cfg = generate_real_config(
        template_path=_REAL_TEMPLATE,
        assignment_folder=real_assignment_folder,
        iterations=iterations,
        year_str=year,
    )
    real_config_path = configs_dir / "config_real.yaml"
    _save_yaml(real_cfg, real_config_path)
    log.info("  Config → %s", real_config_path)
    generated.insert(0, ("real", real_config_path))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("\n--- Generated configs ---")
    for label, path in generated:
        log.info("  %-20s  %s", label, path)
    log.info("Done. Run with:")
    log.info(
        "  bash scripts/run_sensitivity_experiment.sh "
        "--output-dir %s --year %s --gaps '%s' --iterations %d",
        out_root,
        year,
        gaps,
        iterations,
    )


if __name__ == "__main__":
    main()

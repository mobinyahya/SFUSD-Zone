"""Generate per-year and per-experiment analysis configs for FS estimates.

Scans local-data/local-runs/fs_estimates_runs/ to discover experiments,
years, and policies. Produces:
  - 1 config per (experiment, year) → 16-policy plots
  - 1 config per experiment          → aggregated Excel

Usage:
    python scripts/generate_fs_analysis_configs.py
"""

from pathlib import Path
from typing import Any

import yaml

# ── Constants ────────────────────────────────────────────────────────

DATA_ROOT = Path("local-data/local-runs/fs_estimates_runs")
STUDENT_FILTER = Path("local-data/student_filter")
PROGRAM_FILTER = Path("local-data/program_filter")
CONFIG_OUT = Path("configs/fs_estimates")

SINGLE_METRICS: list[str] = [
    "Unassigned",
    "#Unassigned",
    "Avg. Distance",
    "Median Distance",
    "% assigned within 0.5mi",
    "% assigned beyond 3mi",
    "Assigned to 1st choice",
    "Assigned top-3 choice",
    "Capacity",
    "Assigned students",
    "Students",
    "Not assigned",
    "Designated",
    "Empty Seats",
    "# GE Empty Seats",
    "# high-poverty schools",
    "# students in high-poverty schools",
    "# AALPI students in high-poverty schools",
    "Total AALPI students",
    "Total AALPI assigned students",
    "Dissimilarity (High FRL)",
    "Black/White exposure to AALPI",
    "Black/White exposure to poverty",
    "Hispanic/White exposure to AALPI",
    "Hispanic/White exposure to poverty",
    "AALPI exposure to high FRL",
    "AALPI exposure to low FRL",
]

GROUP_METRICS: list[str] = [
    "Prop Top 1 choice ({group})",
    "Prop Top 2 choice ({group})",
    "Prop Top 3 choice ({group})",
    "Dissimilarity ({group})",
    "Number of assigned students ({group})",
    "Distance Median ({group})",
    "Distance Av ({group})",
    "Distance < 0.5 ({group})",
    "Distance > 3 ({group})",
]


# ── Helpers ──────────────────────────────────────────────────────────


def year_str_to_int(year_str: str) -> int:
    """Convert '1415' → 14, '2324' → 23.

    Args:
        year_str: Four-digit year string (e.g. ``'2223'``).

    Returns:
        Integer representing the first two digits.
    """
    return int(year_str[:2])


def discover_structure(
    root: Path,
) -> dict[str, dict[str, list[str]]]:
    """Walk the data tree and return {exp: {year: [policies]}}.

    Args:
        root: Path to fs_estimates_runs/.

    Returns:
        Nested dict mapping experiment → year_str → sorted
        list of policy folder names.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    structure: dict[str, dict[str, list[str]]] = {}

    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        structure[exp_name] = {}

        for year_dir in sorted(exp_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            year_str = year_dir.name
            policies = sorted(p.name for p in year_dir.iterdir() if p.is_dir())
            if policies:
                structure[exp_name][year_str] = policies

    return structure


def make_run_entry(
    exp: str,
    year_str: str,
    policy: str,
    *,
    label_prefix: str = "",
) -> dict[str, Any]:
    """Build a single run dict for the YAML config.

    Args:
        exp: Experiment name (e.g. ``'fwd_sel_2223'``).
        year_str: Year string (e.g. ``'2223'``).
        policy: Policy folder name (e.g.
            ``'status_quo+reserves'``).
        label_prefix: Optional prefix prepended to the label.

    Returns:
        Dict with keys ``label``, ``folder``, ``year``,
        ``program_data``, ``student_data``.
    """
    year_int = year_str_to_int(year_str)
    folder = f"./{DATA_ROOT}/{exp}/{year_str}/{policy}"
    student = str(STUDENT_FILTER / f"student_{year_str}_filtered.csv")
    program = str(
        PROGRAM_FILTER / f"programs_without_specialprogs_{year_str}.csv"
    )
    label = f"{label_prefix}{policy}" if label_prefix else policy
    return {
        "label": label,
        "folder": folder,
        "year": year_int,
        "program_data": program,
        "student_data": student,
    }


def write_config(
    path: Path,
    output_dir: str,
    runs: list[dict[str, Any]],
) -> None:
    """Write a YAML analysis config file.

    Args:
        path: Destination file path.
        output_dir: Value for the ``output_dir`` key.
        runs: List of run entry dicts.
    """
    config: dict[str, Any] = {
        "output_dir": output_dir,
        "runs": runs,
        "single_metrics": SINGLE_METRICS,
        "group_metrics": GROUP_METRICS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(
            config,
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"  ✓ {path}  ({len(runs)} runs)")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Discover data and generate all FS analysis configs."""
    structure = discover_structure(DATA_ROOT)

    total_per_year = 0
    total_per_exp = 0

    for exp, years in structure.items():
        exp_runs: list[dict[str, Any]] = []

        for year_str, policies in years.items():
            # ── Per-year config (plots) ──────────────────────
            year_runs = [make_run_entry(exp, year_str, pol) for pol in policies]
            cfg_path = CONFIG_OUT / f"fs_{exp}_{year_str}.yaml"
            out_dir = f"metrics/fs_estimates/{exp}/{year_str}"
            write_config(cfg_path, out_dir, year_runs)
            total_per_year += 1

            # ── Accumulate for per-experiment config ─────────
            for pol in policies:
                exp_runs.append(
                    make_run_entry(
                        exp,
                        year_str,
                        pol,
                        label_prefix=f"{year_str} | ",
                    )
                )

        # ── Per-experiment config (Excel) ────────────────────
        cfg_path = CONFIG_OUT / f"fs_{exp}_all.yaml"
        out_dir = f"metrics/fs_estimates/{exp}"
        write_config(cfg_path, out_dir, exp_runs)
        total_per_exp += 1

    print(
        f"\nDone: {total_per_year} per-year configs"
        f" + {total_per_exp} per-experiment configs"
        f" → {CONFIG_OUT}/"
    )


if __name__ == "__main__":
    main()

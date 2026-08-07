#!/usr/bin/env python3
"""Export configured student demographics by elementary attendance area.

The ``new_frl`` and ``old_frl`` columns are independent expected student counts.
``new_frl`` follows the updated-metrics definition: map student coordinates to a
2020 Census block, use its updated rate when available, and use the legacy rate
as a fallback. ``old_frl`` uses only the original student-file free-plus-reduced
probabilities. The columns are not added together.

Usage:
    uv run python analysis/attendance_area_student_demographics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.recalculate_updated_frl_metrics import (  # noqa: E402
    DEFAULT_2020_BLOCKS,
    DEFAULT_UPDATED_FRL,
    load_frl_lookup,
    load_2020_block_geometry,
    load_yaml,
    parse_ranked_list,
    resolve_data_path,
    student_frl_data,
)
from assignment.student_assignment.definitions.constants import (  # noqa: E402
    SPECIAL_PROGRAMS,
)

DEFAULT_CONFIG = PROJECT_ROOT / "assignment/configs/kumar.config.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis/attendance_area_student_demographics.csv"
DEFAULT_FALLBACK_OUTPUT = PROJECT_ROOT / "analysis/frl_fallback_blocks.csv"
RACE_GROUPS = (
    "Asian",
    "Black",
    "Decline to State",
    "Hispanic",
    "Other",
    "Pacific Islander",
    "Two or More Races",
    "White",
)


def diagnostic_race(value: object) -> str:
    """Use the same broad race groups as the assignment evaluator."""
    if value in {
        "Asian",
        "Asian Indian",
        "Chinese",
        "Vietnamese",
        "Filipino",
        "Japanese",
        "Korean",
        "Hmong",
        "Other Asian",
        "Cambodian",
        "Laotian",
    }:
        return "Asian"
    if value in {"Hispanic/Latino", "Hispanic/Latinx", "Hispanic"}:
        return "Hispanic"
    if value in {"White", "Middle Eastern/Arabic"}:
        return "White"
    if value in {"Black or African American", "Black/African American"}:
        return "Black"
    if value in {
        "Other Pacific Islander",
        "Pacific Islander",
        "Samoan",
        "Hawaiian Native",
    }:
        return "Pacific Islander"
    if value in {"Two or More Races", "Multi-Racial", "Two or More"}:
        return "Two or More Races"
    if value in {"Decline to State", "Decline To State"}:
        return "Decline to State"
    return "Other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--updated-frl", type=Path, default=DEFAULT_UPDATED_FRL)
    parser.add_argument("--block-geometry", type=Path, default=DEFAULT_2020_BLOCKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fallback-output", type=Path, default=DEFAULT_FALLBACK_OUTPUT)
    return parser.parse_args()


def configured_students(
    config: dict[str, Any],
    lookup: pd.Series,
    block_geometry: gpd.GeoDataFrame,
) -> pd.DataFrame:
    students = pd.read_csv(resolve_data_path(config, "student-data"), low_memory=False)
    grade = str(config.get("grade", "KG"))
    students = students.loc[students["grade"].astype("string") == grade].copy()

    if config.get("remove-special-lps", False):
        if "r1_programs" not in students:
            raise ValueError("student data has no r1_programs column")
        special_programs = set(SPECIAL_PROGRAMS)
        has_special_program = (
            students["r1_programs"]
            .map(parse_ranked_list)
            .map(lambda programs: bool(set(programs) & special_programs))
        )
        students = students.loc[~has_special_program].copy()

    required = {
        "studentno",
        "idschoolattendance",
        "resolved_ethnicity",
        "latitude",
        "longitude",
        "freelunch_prob",
        "reducedlunch_prob",
    }
    missing = required - set(students.columns)
    if missing:
        raise ValueError(f"student data is missing columns {sorted(missing)}")
    if students["studentno"].duplicated().any():
        raise ValueError("studentno must be unique after applying config filters")

    return add_frl_datasets(students, lookup, block_geometry)


def add_frl_datasets(
    students: pd.DataFrame,
    lookup: pd.Series,
    block_geometry: gpd.GeoDataFrame,
) -> pd.DataFrame:
    result = students.copy()
    frl_data = student_frl_data(result, lookup, block_geometry)
    result["old_frl"] = frl_data["legacy_frl"]
    result["new_frl"] = frl_data["effective_frl"]
    result["census_block_2020"] = frl_data["census_block_2020"]
    result["frl_fallback_reason"] = frl_data["frl_fallback_reason"]
    return result


def fallback_block_report(students: pd.DataFrame) -> pd.DataFrame:
    required = {
        "studentno",
        "idschoolattendance",
        "census_block_2020",
        "frl_fallback_reason",
        "old_frl",
    }
    missing = required - set(students.columns)
    if missing:
        raise ValueError(f"student data is missing fallback columns {sorted(missing)}")

    fallback = students.loc[students["frl_fallback_reason"].notna()].copy()
    fallback["attendance_area"] = pd.to_numeric(
        fallback["idschoolattendance"], errors="raise"
    ).astype(int)

    def joined_attendance_areas(values: pd.Series) -> str:
        return "|".join(str(value) for value in sorted(values.unique()))

    report = (
        fallback.groupby(
            ["frl_fallback_reason", "census_block_2020"],
            dropna=False,
            sort=True,
        )
        .agg(
            student_count=("studentno", "size"),
            attendance_areas=("attendance_area", joined_attendance_areas),
            old_frl=("old_frl", "sum"),
        )
        .reset_index()
    )
    report["old_frl"] = report["old_frl"].round(6)
    return report[
        [
            "census_block_2020",
            "frl_fallback_reason",
            "student_count",
            "attendance_areas",
            "old_frl",
        ]
    ]


def attendance_schools(config: dict[str, Any]) -> pd.DataFrame:
    schools = pd.read_csv(resolve_data_path(config, "school-data"))
    required = {"school_id", "school_name", "category"}
    missing = required - set(schools.columns)
    if missing:
        raise ValueError(f"school data is missing columns {sorted(missing)}")

    schools = schools.loc[
        schools["category"].astype("string").str.casefold() == "attendance",
        ["school_id", "school_name"],
    ].copy()
    schools["school_id"] = pd.to_numeric(schools["school_id"], errors="raise").astype(
        int
    )
    if schools["school_id"].duplicated().any():
        raise ValueError("attendance school IDs must be unique")
    return schools.sort_values("school_id").reset_index(drop=True)


def build_summary(students: pd.DataFrame, schools: pd.DataFrame) -> pd.DataFrame:
    students = students.copy()
    missing_frl = {"new_frl", "old_frl"} - set(students.columns)
    if missing_frl:
        raise ValueError(f"student data is missing FRL columns {sorted(missing_frl)}")
    if students["idschoolattendance"].isna().any():
        raise ValueError("students with missing attendance areas cannot be summarized")
    students["attendance_area"] = pd.to_numeric(
        students["idschoolattendance"], errors="raise"
    ).astype(int)

    valid_areas = set(schools["school_id"])
    unknown_areas = sorted(set(students["attendance_area"]) - valid_areas)
    if unknown_areas:
        raise ValueError(
            f"student attendance areas are absent from school data: {unknown_areas[:3]}"
        )

    students["race"] = students["resolved_ethnicity"].map(diagnostic_race)
    students["new_frl"] = pd.to_numeric(students["new_frl"], errors="coerce")
    students["old_frl"] = pd.to_numeric(students["old_frl"], errors="raise")

    grouped = students.groupby("attendance_area", sort=True)
    population = grouped.agg(
        total_students=("studentno", "size"),
        new_frl=("new_frl", "sum"),
        old_frl=("old_frl", "sum"),
    )

    race_counts = pd.crosstab(students["attendance_area"], students["race"])
    race_columns = [f"race_{race}" for race in RACE_GROUPS]
    race_counts = race_counts.reindex(columns=RACE_GROUPS, fill_value=0).rename(
        columns=lambda race: f"race_{race}"
    )

    summary = schools.rename(
        columns={"school_id": "attendance_area", "school_name": "attendance_area_name"}
    ).merge(population, left_on="attendance_area", right_index=True, how="left")
    summary = summary.merge(
        race_counts, left_on="attendance_area", right_index=True, how="left"
    )
    count_columns = ["total_students", *race_columns]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    summary[["new_frl", "old_frl"]] = (
        summary[["new_frl", "old_frl"]].fillna(0.0).round(6)
    )
    return summary[
        [
            "attendance_area",
            "attendance_area_name",
            "total_students",
            "new_frl",
            "old_frl",
            *race_columns,
        ]
    ]


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.expanduser().resolve())
    lookup = load_frl_lookup(args.updated_frl.expanduser().resolve())
    block_geometry = load_2020_block_geometry(
        args.block_geometry.expanduser().resolve()
    )
    students = configured_students(config, lookup, block_geometry)
    schools = attendance_schools(config)
    summary = build_summary(students, schools)
    fallback_report = fallback_block_report(students)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    fallback_output_path = args.fallback_output.expanduser().resolve()
    fallback_output_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_report.to_csv(fallback_output_path, index=False)
    print(f"Wrote {len(summary)} attendance areas to {output_path}")
    print(f"Wrote {len(fallback_report)} fallback block rows to {fallback_output_path}")
    print(f"Students: {summary['total_students'].sum()}")
    print(f"New FRL: {summary['new_frl'].sum():.6f}")
    print(f"Old FRL: {summary['old_frl'].sum():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

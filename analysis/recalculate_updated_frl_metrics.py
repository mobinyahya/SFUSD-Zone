#!/usr/bin/env python3
"""Recalculate saved assignment reports with updated block-level FRL rates.

The script reuses the 25 raw assignments behind two existing metric reports. It
does not rerun matching. Every student's FRL probability is replaced by the
``FRL Rate`` for their census block when that rate is available; otherwise the
evaluator's legacy free-plus-reduced-lunch calculation is retained.

A CSV lists the KG census blocks for which that legacy fallback is used. It
distinguishes blocks absent from the updated lookup, blocks with blank updated
rates, and students whose census block is missing.

Zone metrics are appended for geographic zone configurations, including:

* FRL proportion by zone among all 2023-24 KG applicants
* signed FRL deviation from the district proportion and maximum absolute deviation
* fractional GE students, applicants, GE seats, and GE-student-to-seat ratios
* empty seats by school, in ascending numeric school ID order
* unassigned matching students by residential zone
* empty GE seats by zone, counting attendance schools only

Distance and status-quo configurations receive ``[]`` for all four rows. Counts
that vary by assignment are averaged elementwise over the 25 iterations.
Students without the census geography used by a zone plan are omitted from its
zone-level lists because they cannot be assigned to a geographic zone.

Usage:
    uv run python analysis/recalculate_updated_frl_metrics.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment.student_assignment.evaluation.match_evaluator import (  # noqa: E402
    MatchEvaluator,
)

LOGGER = logging.getLogger(__name__)

ITERATION_COUNT = 25
ITERATION_PATTERN = re.compile(r"iteration(\d+)\.csv$")
TIMESTAMP_PATTERN = re.compile(r"_\d{8}T\d+Z$")

DEFAULT_SOFT_METRICS = (
    PROJECT_ROOT
    / "analysis/plots/soft_reserves_06frl_25_eval_assignment_full_20260723T200131970561Z.csv"
)
DEFAULT_SOFT_MATCHES_ROOT = (
    PROJECT_ROOT / "analysis/matches/zones+soft_reserves_05frl_25"
)
DEFAULT_ZONE_METRICS = (
    PROJECT_ROOT
    / "analysis/matches/zone_subconfigs_choice_model_25_soft_reserves_updated"
    / "zone_subconfigs_25_eval_assignment_full_20260723T192723268919Z.csv"
)
DEFAULT_ZONE_MATCHES_ROOT = (
    PROJECT_ROOT
    / "analysis/matches/zone_subconfigs_choice_model_25_soft_reserves_updated"
)
DEFAULT_UPDATED_FRL = PROJECT_ROOT / "analysis/updated_frl_block.csv"
DEFAULT_FALLBACK_BLOCKS = (
    PROJECT_ROOT / "analysis/recalculate_updated_frl_fallback_blocks.csv"
)
DEFAULT_ALL_STUDENTS = Path("/share/data/school_choice/Data/Cleaned/student_2324.csv")
DEFAULT_NEW_CTIP = Path(
    "/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/ETB_2024.npy"
)

FRL_MAX_DEV_METRIC = "FRL Max Dev"
LIST_METRICS = (
    "FRL proportion by zone",
    "FRL Devs by Zone",
    "GE Students by Zone",
    "Applicants per Zone",
    "GE Seat Disparity by Zone",
    "GE Seats by Zone",
    "Empty seats by school (ascending school_id)",
    "Unassigned students by zone",
    "Empty GE seats by zone (attendance schools)",
)
NEW_METRICS = (
    LIST_METRICS[0],
    LIST_METRICS[1],
    FRL_MAX_DEV_METRIC,
    *LIST_METRICS[2:],
)


@dataclass(frozen=True)
class ZoneDefinition:
    """A row-ordered zone plan and its residential geography unit."""

    unit: str
    area_to_zone: dict[str, int]
    zone_count: int


@dataclass(frozen=True)
class ConfigurationTask:
    """All saved inputs needed to recalculate one output column."""

    label: str
    config_path: str
    assignment_paths: tuple[str, ...]
    updated_frl_path: str
    all_students_path: str
    new_ctip_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft-metrics", type=Path, default=DEFAULT_SOFT_METRICS)
    parser.add_argument(
        "--soft-matches-root", type=Path, default=DEFAULT_SOFT_MATCHES_ROOT
    )
    parser.add_argument("--zone-metrics", type=Path, default=DEFAULT_ZONE_METRICS)
    parser.add_argument(
        "--zone-matches-root", type=Path, default=DEFAULT_ZONE_MATCHES_ROOT
    )
    parser.add_argument("--updated-frl", type=Path, default=DEFAULT_UPDATED_FRL)
    parser.add_argument(
        "--fallback-blocks-output",
        type=Path,
        default=DEFAULT_FALLBACK_BLOCKS,
        help="CSV listing KG blocks that use legacy FRL data.",
    )
    parser.add_argument("--all-students", type=Path, default=DEFAULT_ALL_STUDENTS)
    parser.add_argument(
        "--new-ctip-path",
        type=Path,
        default=DEFAULT_NEW_CTIP,
        help="Optional equity-block NPY used by MatchEvaluator.",
    )
    parser.add_argument(
        "--dataset",
        choices=("both", "soft", "zone"),
        default="both",
        help="Select which saved report to recalculate.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Independent output columns evaluated in parallel.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return data


def normalize_geoid(value: object) -> str | None:
    """Return a stable integer-like GEOID string without a trailing ``.0``."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite() or number != number.to_integral_value():
        return text
    return str(int(number))


def normalized_geoids(values: pd.Series) -> pd.Series:
    return values.map(normalize_geoid)


def load_frl_lookup(path: Path) -> pd.Series:
    frame = pd.read_csv(path, dtype={"BlockID": "string"})
    required = {"BlockID", "FRL Rate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"FRL lookup is missing columns {sorted(missing)}: {path}")

    block_ids = normalized_geoids(frame["BlockID"])
    rates = pd.to_numeric(frame["FRL Rate"], errors="coerce")
    present = block_ids.notna()
    if block_ids[present].duplicated().any():
        duplicates = block_ids[present][block_ids[present].duplicated()].unique()
        raise ValueError(
            f"FRL lookup has duplicate BlockIDs, including {duplicates[:3]}"
        )
    return pd.Series(rates[present].to_numpy(), index=block_ids[present], dtype=float)


def enrich_student_frl(students: pd.DataFrame, lookup: pd.Series) -> pd.DataFrame:
    """Override legacy student FRL with non-null updated block rates."""
    if "census_block" not in students:
        raise ValueError("student data has no census_block column")

    result = students.copy()
    free = pd.to_numeric(
        result.get("freelunch_prob", pd.Series(0, index=result.index)),
        errors="coerce",
    ).fillna(0)
    reduced = pd.to_numeric(
        result.get("reducedlunch_prob", pd.Series(0, index=result.index)),
        errors="coerce",
    ).fillna(0)
    legacy_frl = free + reduced
    updated_frl = normalized_geoids(result["census_block"]).map(lookup)
    effective_frl = updated_frl.fillna(legacy_frl)

    # MatchEvaluator derives `frl` by adding these two source columns.
    result["freelunch_prob"] = effective_frl.astype(float)
    result["reducedlunch_prob"] = 0.0
    return result


def fallback_block_report(
    students: pd.DataFrame,
    lookup: pd.Series,
    grade: object = "KG",
) -> pd.DataFrame:
    """List blocks whose students retain legacy FRL values."""
    required = {
        "grade",
        "census_block",
        "freelunch_prob",
        "reducedlunch_prob",
    }
    missing = required - set(students.columns)
    if missing:
        raise ValueError(f"student data is missing columns {sorted(missing)}")

    grade_students = students.loc[students["grade"].astype("string") == str(grade)]
    block_ids = normalized_geoids(grade_students["census_block"])
    updated_frl = block_ids.map(lookup)
    legacy_frl = pd.to_numeric(
        grade_students["freelunch_prob"], errors="coerce"
    ).fillna(0) + pd.to_numeric(
        grade_students["reducedlunch_prob"], errors="coerce"
    ).fillna(0)

    reason = pd.Series(pd.NA, index=grade_students.index, dtype="string")
    reason.loc[block_ids.isna()] = "missing student census block"
    reason.loc[block_ids.notna() & ~block_ids.isin(lookup.index)] = (
        "absent from updated lookup"
    )
    blank_rate = block_ids.notna() & block_ids.isin(lookup.index) & updated_frl.isna()
    reason.loc[blank_rate] = "blank updated FRL rate"

    fallback = pd.DataFrame(
        {
            "census_block": block_ids,
            "frl_fallback_reason": reason,
            "legacy_frl": legacy_frl,
        }
    ).loc[updated_frl.isna()]
    report = (
        fallback.groupby(
            ["census_block", "frl_fallback_reason"],
            dropna=False,
            sort=False,
        )
        .agg(
            student_count=("legacy_frl", "size"),
            legacy_frl=("legacy_frl", "sum"),
        )
        .reset_index()
        .sort_values("census_block", na_position="last")
        .reset_index(drop=True)
    )
    report["legacy_frl"] = report["legacy_frl"].round(6)
    return report


def resolve_config_value(config: Mapping[str, Any], value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    paths = config.get("paths")
    if not isinstance(paths, Mapping) or not paths.get("sfusd"):
        raise ValueError(f"relative config path has no paths.sfusd root: {value}")
    return (Path(str(paths["sfusd"])).expanduser() / path).resolve()


def resolve_data_path(config: Mapping[str, Any], key: str) -> Path:
    paths = config.get("paths")
    if not isinstance(paths, Mapping) or not paths.get(key):
        raise ValueError(f"configuration has no paths.{key}")
    return resolve_config_value(config, paths[key])


def resolve_zone_definition(config: Mapping[str, Any]) -> ZoneDefinition | None:
    unit = str(config.get("zone-building-blocks") or "").strip().lower()
    if unit not in {"block", "block_group"}:
        return None

    policies = config.get("policies")
    paths = config.get("paths")
    if not isinstance(policies, list) or len(policies) != 1:
        raise ValueError("zone configuration must name exactly one policy")
    if not isinstance(paths, Mapping) or not isinstance(
        paths.get("zone-files"), Mapping
    ):
        raise ValueError("zone configuration has no paths.zone-files mapping")
    zone_files = paths["zone-files"]
    policy = str(policies[0])
    if not zone_files.get(policy):
        raise ValueError(f"zone configuration has no file for policy {policy!r}")
    zone_path = resolve_config_value(config, zone_files[policy])
    return load_zone_definition(zone_path, unit)


def load_zone_definition(path: Path, unit: str) -> ZoneDefinition:
    area_to_zone: dict[str, int] = {}
    zone_count = 0
    with path.open(newline="", encoding="utf-8") as zone_file:
        for zone_id, row in enumerate(csv.reader(zone_file)):
            areas = [normalize_geoid(value) for value in row if value.strip()]
            areas = [area for area in areas if area is not None]
            if not areas:
                raise ValueError(f"zone {zone_id} is empty: {path}")
            duplicates = sorted(set(areas) & set(area_to_zone))
            if duplicates:
                raise ValueError(
                    f"zone file assigns areas more than once, including {duplicates[:3]}"
                )
            area_to_zone.update({area: zone_id for area in areas})
            zone_count = zone_id + 1
    if zone_count == 0:
        raise ValueError(f"zone file is empty: {path}")
    return ZoneDefinition(unit=unit, area_to_zone=area_to_zone, zone_count=zone_count)


def zone_column(unit: str, *, school: bool = False) -> str:
    if unit == "block":
        return "Block" if school else "census_block"
    if unit == "block_group":
        return "BlockGroup" if school else "census_blockgroup"
    raise ValueError(f"unsupported zone unit: {unit}")


def map_areas_to_zones(values: pd.Series, zone: ZoneDefinition) -> pd.Series:
    areas = normalized_geoids(values)
    zones = areas.map(zone.area_to_zone)
    outside = areas.notna() & zones.isna()
    if outside.any():
        examples = areas[outside].drop_duplicates().head(3).tolist()
        raise ValueError(
            f"geographic IDs are outside the zone plan, including {examples}"
        )
    return zones


def parse_ranked_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, np.ndarray)):
        return list(value)
    if value is None or pd.isna(value):
        return []
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


def ordered_zone_totals(
    values: pd.Series,
    zone: ZoneDefinition,
    *,
    integer: bool = False,
) -> list[float] | list[int]:
    """Reindex zone values into strict ascending zone-number order."""
    ordered = values.reindex(range(zone.zone_count), fill_value=0)
    if integer:
        return ordered.astype(int).tolist()
    return ordered.astype(float).tolist()


def zone_population_metrics(
    all_students: pd.DataFrame,
    grade: object,
    zone: ZoneDefinition,
) -> dict[str, float | list[float] | list[int]]:
    """Calculate static applicant metrics in ascending zone-number order."""
    required = {
        "grade",
        "freelunch_prob",
        "r1_programs",
        "r1_ranked_idschool",
        zone_column(zone.unit),
    }
    missing = required - set(all_students.columns)
    if missing:
        raise ValueError(f"all-student data is missing columns {sorted(missing)}")

    grade_students = all_students.loc[
        all_students["grade"].astype("string") == str(grade)
    ].copy()
    zones = map_areas_to_zones(grade_students[zone_column(zone.unit)], zone)
    mapped = zones.notna()
    district_frl = float(grade_students.loc[mapped, "freelunch_prob"].mean())

    program_lists = grade_students["r1_programs"].map(parse_ranked_list)
    ranked_school_lists = grade_students["r1_ranked_idschool"].map(parse_ranked_list)
    mismatched_lists = [
        index
        for index, (programs, schools) in enumerate(
            zip(program_lists, ranked_school_lists, strict=True)
        )
        if len(programs) != len(schools)
    ]
    if mismatched_lists:
        raise ValueError(
            "ranked school and program list lengths differ, including rows "
            f"{mismatched_lists[:3]}"
        )
    ge_weights = pd.Series(
        (
            sum(str(program).upper() == "GE" for program in programs) / len(schools)
            if schools
            else 0.0
        )
        for programs, schools in zip(program_lists, ranked_school_lists, strict=True)
    )
    ge_weights.index = grade_students.index
    applicants = ranked_school_lists.map(bool)

    frl_by_zone: list[float] = []
    for zone_id in range(zone.zone_count):
        values = grade_students.loc[zones == zone_id, "freelunch_prob"]
        frl_by_zone.append(float(values.mean()) if len(values) else 0.0)
    frl_devs = [value - district_frl for value in frl_by_zone]
    ge_students = ordered_zone_totals(ge_weights.groupby(zones).sum(), zone)
    applicants_by_zone = ordered_zone_totals(
        applicants.astype(int).groupby(zones).sum(), zone, integer=True
    )

    return {
        "district_frl": district_frl,
        "frl_by_zone": frl_by_zone,
        "frl_devs": frl_devs,
        "frl_max_dev": max((abs(value) for value in frl_devs), default=0.0),
        "ge_students": ge_students,
        "applicants": applicants_by_zone,
    }


def zone_frl_proportions(
    all_students: pd.DataFrame,
    grade: object,
    zone: ZoneDefinition,
) -> list[float]:
    metrics = zone_population_metrics(all_students, grade, zone)
    return list(metrics["frl_by_zone"])  # type: ignore[arg-type]


def normalized_program_data(programs: pd.DataFrame) -> pd.DataFrame:
    required = {"program_id", "school_id", "program_type", "capacity"}
    missing = required - set(programs.columns)
    if missing:
        raise ValueError(f"program data is missing columns {sorted(missing)}")

    program_data = programs.loc[:, list(required)].copy()
    program_data["program_id"] = program_data["program_id"].astype(str)
    if program_data["program_id"].duplicated().any():
        raise ValueError("program_id must be unique in program data")
    program_data["capacity"] = pd.to_numeric(program_data["capacity"], errors="raise")
    program_data["school_id"] = pd.to_numeric(
        program_data["school_id"], errors="raise"
    ).astype(int)
    return program_data


def program_vacancies(
    assignment: pd.DataFrame,
    programs: pd.DataFrame,
) -> pd.DataFrame:
    program_data = normalized_program_data(programs)

    assigned = assignment.loc[
        pd.to_numeric(assignment["programno"], errors="coerce").fillna(0) > 0,
        "programcodes",
    ].dropna()
    assigned_counts = assigned.astype(str).value_counts()
    unknown = sorted(set(assigned_counts.index) - set(program_data["program_id"]))
    if unknown:
        raise ValueError(
            f"assignments contain unknown programs, including {unknown[:3]}"
        )
    program_data["assigned"] = (
        program_data["program_id"].map(assigned_counts).fillna(0).astype(int)
    )
    program_data["empty_seats"] = (
        program_data["capacity"] - program_data["assigned"]
    ).clip(lower=0)
    return program_data


def attendance_school_zones(
    schools: pd.DataFrame,
    zone: ZoneDefinition,
) -> pd.DataFrame:
    school_column = zone_column(zone.unit, school=True)
    required = {"school_id", "category", school_column}
    missing = required - set(schools.columns)
    if missing:
        raise ValueError(f"school data is missing columns {sorted(missing)}")
    school_data = schools.loc[:, list(required)].copy()
    school_data["school_id"] = pd.to_numeric(
        school_data["school_id"], errors="raise"
    ).astype(int)
    if school_data["school_id"].duplicated().any():
        raise ValueError("school_id must be unique in school data")
    attendance = school_data["category"].astype("string").str.casefold() == "attendance"
    attendance_schools = school_data.loc[attendance].copy()
    attendance_schools["zone"] = map_areas_to_zones(
        attendance_schools[school_column], zone
    )
    if attendance_schools["zone"].isna().any():
        missing_ids = attendance_schools.loc[
            attendance_schools["zone"].isna(), "school_id"
        ].tolist()
        raise ValueError(
            f"attendance schools have no zone, including {missing_ids[:3]}"
        )
    attendance_schools["zone"] = attendance_schools["zone"].astype(int)
    return attendance_schools[["school_id", "zone"]]


def ge_seats_by_zone(
    programs: pd.DataFrame,
    schools: pd.DataFrame,
    zone: ZoneDefinition,
) -> list[float]:
    program_data = normalized_program_data(programs)
    attendance_zones = attendance_school_zones(schools, zone)
    ge_programs = program_data.loc[
        program_data["program_type"].astype("string").str.upper() == "GE"
    ].merge(
        attendance_zones,
        on="school_id",
        how="inner",
        validate="many_to_one",
    )
    return ordered_zone_totals(ge_programs.groupby("zone")["capacity"].sum(), zone)


def ge_seat_disparity_by_zone(
    ge_students: list[float],
    ge_seats: list[float],
    zone: ZoneDefinition,
) -> list[float | None]:
    if len(ge_students) != zone.zone_count or len(ge_seats) != zone.zone_count:
        raise ValueError("GE student and seat lists must contain every zone")
    return [
        students / seats if seats > 0 else None
        for students, seats in zip(ge_students, ge_seats, strict=True)
    ]


def assignment_list_metrics(
    assignment: pd.DataFrame,
    students: pd.DataFrame,
    programs: pd.DataFrame,
    schools: pd.DataFrame,
    zone: ZoneDefinition,
) -> tuple[list[float], list[float], list[float]]:
    """Return school vacancies, zone unassigned counts, and zone GE vacancies."""
    vacancies = program_vacancies(assignment, programs)

    school_ids = sorted(vacancies["school_id"].unique().tolist())
    empty_by_school = (
        vacancies.groupby("school_id")["empty_seats"]
        .sum()
        .reindex(school_ids, fill_value=0)
        .astype(float)
        .tolist()
    )

    student_areas = students[["studentno", zone_column(zone.unit)]].copy()
    if student_areas["studentno"].duplicated().any():
        raise ValueError("studentno must be unique in student data")
    assignment_zones = assignment[["studentno", "programno"]].merge(
        student_areas,
        on="studentno",
        how="left",
        validate="one_to_one",
    )
    residential_zones = map_areas_to_zones(
        assignment_zones[zone_column(zone.unit)], zone
    )
    unassigned = (
        pd.to_numeric(assignment_zones["programno"], errors="coerce").fillna(0).le(0)
    )
    unassigned_by_zone = ordered_zone_totals(
        residential_zones[unassigned].value_counts(), zone
    )

    attendance_schools = attendance_school_zones(schools, zone)

    ge_vacancies = vacancies.loc[
        vacancies["program_type"].astype("string").str.upper() == "GE"
    ].merge(
        attendance_schools[["school_id", "zone"]],
        on="school_id",
        how="inner",
        validate="many_to_one",
    )
    empty_ge_by_zone = ordered_zone_totals(
        ge_vacancies.groupby("zone")["empty_seats"].sum(), zone
    )
    return empty_by_school, unassigned_by_zone, empty_ge_by_zone


def mean_lists(values: list[list[float]]) -> list[float]:
    if not values:
        raise ValueError("cannot average an empty list collection")
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError(f"list metric lengths differ across iterations: {lengths}")
    return np.asarray(values, dtype=float).mean(axis=0).tolist()


def json_list(value: list[float | int | None]) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def zone_json_list(
    value: list[float | int | None],
    zone: ZoneDefinition,
) -> str:
    if len(value) != zone.zone_count:
        raise ValueError(
            f"zone list has {len(value)} values, expected {zone.zone_count}"
        )
    return json_list(value)


def evaluate_configuration(task: ConfigurationTask) -> tuple[str, pd.Series]:
    config_path = Path(task.config_path)
    config = load_yaml(config_path)
    student_path = resolve_data_path(config, "student-data")
    program_path = resolve_data_path(config, "program-data")
    school_path = resolve_data_path(config, "school-data")
    for path in (student_path, program_path, school_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    lookup = load_frl_lookup(Path(task.updated_frl_path))
    students = enrich_student_frl(
        pd.read_csv(student_path, low_memory=False),
        lookup,
    )
    programs = pd.read_csv(program_path)
    schools = pd.read_csv(school_path)
    zone = resolve_zone_definition(config)

    evaluator_year = int(
        f"{int(config.get('year', 23)):02d}{(int(config.get('year', 23)) + 1) % 100:02d}"
    )
    base_metrics: list[pd.Series] = []
    empty_school_iterations: list[list[float]] = []
    unassigned_iterations: list[list[float]] = []
    empty_ge_iterations: list[list[float]] = []

    for assignment_path in task.assignment_paths:
        assignment = pd.read_csv(assignment_path)
        evaluator = MatchEvaluator(
            students,
            assignment,
            first_round=True,
            dropout=False,
            low_income=95292,
            medium_income=95292,
            high_income=110850,
            grade=None,
            year=evaluator_year,
            no_special_program=True,
            program_file=str(program_path),
            schools_latlon_path=str(school_path),
            new_ctip_path=task.new_ctip_path,
        )
        base_metrics.append(evaluator.eval_assignment_full())
        if zone is not None:
            empty_school, unassigned, empty_ge = assignment_list_metrics(
                assignment,
                students,
                programs,
                schools,
                zone,
            )
            empty_school_iterations.append(empty_school)
            unassigned_iterations.append(unassigned)
            empty_ge_iterations.append(empty_ge)

    if len(base_metrics) != ITERATION_COUNT:
        raise ValueError(
            f"{task.label} evaluated {len(base_metrics)} iterations, "
            f"expected {ITERATION_COUNT}"
        )
    metrics = pd.concat(base_metrics, axis=1).mean(axis=1, skipna=True)

    if zone is None:
        for name in LIST_METRICS:
            metrics[name] = "[]"
        metrics[FRL_MAX_DEV_METRIC] = float("nan")
        return task.label, metrics

    all_students = enrich_student_frl(
        pd.read_csv(task.all_students_path, low_memory=False),
        lookup,
    )
    population = zone_population_metrics(
        all_students,
        config.get("grade", "KG"),
        zone,
    )
    ge_seats = ge_seats_by_zone(programs, schools, zone)
    ge_students = list(population["ge_students"])  # type: ignore[arg-type]
    ge_disparity = ge_seat_disparity_by_zone(ge_students, ge_seats, zone)

    metrics["FRL proportion by zone"] = zone_json_list(
        list(population["frl_by_zone"]),
        zone,  # type: ignore[arg-type]
    )
    metrics["FRL Devs by Zone"] = zone_json_list(
        list(population["frl_devs"]),
        zone,  # type: ignore[arg-type]
    )
    metrics[FRL_MAX_DEV_METRIC] = float(population["frl_max_dev"])
    metrics["GE Students by Zone"] = zone_json_list(ge_students, zone)
    metrics["Applicants per Zone"] = zone_json_list(
        list(population["applicants"]),
        zone,  # type: ignore[arg-type]
    )
    metrics["GE Seat Disparity by Zone"] = zone_json_list(ge_disparity, zone)
    metrics["GE Seats by Zone"] = zone_json_list(ge_seats, zone)
    metrics["Empty seats by school (ascending school_id)"] = json_list(
        mean_lists(empty_school_iterations)
    )
    metrics["Unassigned students by zone"] = zone_json_list(
        mean_lists(unassigned_iterations), zone
    )
    metrics["Empty GE seats by zone (attendance schools)"] = zone_json_list(
        mean_lists(empty_ge_iterations), zone
    )
    return task.label, metrics


def discover_assignments(root: Path) -> tuple[str, ...]:
    paths_by_iteration: dict[int, Path] = {}
    for path in root.rglob("*.csv"):
        match = ITERATION_PATTERN.search(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration in paths_by_iteration:
            raise ValueError(
                f"multiple assignments found for iteration {iteration}: {root}"
            )
        paths_by_iteration[iteration] = path
    expected = set(range(ITERATION_COUNT))
    if set(paths_by_iteration) != expected:
        missing = sorted(expected - set(paths_by_iteration))
        extra = sorted(set(paths_by_iteration) - expected)
        raise ValueError(
            f"incomplete assignments at {root}: missing={missing}, extra={extra}"
        )
    return tuple(str(paths_by_iteration[index]) for index in range(ITERATION_COUNT))


def build_tasks(
    labels: list[str],
    matches_root: Path,
    artifact_type: str,
    updated_frl_path: Path,
    all_students_path: Path,
    new_ctip_path: Path | None,
) -> list[ConfigurationTask]:
    tasks = []
    for label in labels:
        output_root = matches_root / label
        if artifact_type == "soft":
            config_path = output_root / "matching/config.generated.yaml"
            assignments_root = output_root / "matching/assignments_raw"
        elif artifact_type == "zone":
            config_path = output_root / "policy_config.generated.yaml"
            assignments_root = output_root
        else:
            raise ValueError(f"unknown artifact type: {artifact_type}")
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        tasks.append(
            ConfigurationTask(
                label=label,
                config_path=str(config_path),
                assignment_paths=discover_assignments(assignments_root),
                updated_frl_path=str(updated_frl_path),
                all_students_path=str(all_students_path),
                new_ctip_path=str(new_ctip_path) if new_ctip_path else None,
            )
        )
    return tasks


def align_metrics(metrics: pd.Series, expected_base_metrics: list[str]) -> pd.Series:
    base = metrics.drop(labels=list(NEW_METRICS), errors="ignore")
    missing = sorted(set(expected_base_metrics) - set(base.index))
    extra = sorted(set(base.index) - set(expected_base_metrics))
    if missing or extra:
        raise ValueError(
            f"recalculated metric schema differs from source: missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    return pd.concat(
        [base.reindex(expected_base_metrics), metrics.reindex(list(NEW_METRICS))]
    )


def run_tasks(
    tasks: list[ConfigurationTask],
    expected_base_metrics: list[str],
    workers: int,
) -> dict[str, pd.Series]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    results: dict[str, pd.Series] = {}
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            label, metrics = evaluate_configuration(task)
            results[label] = align_metrics(metrics, expected_base_metrics)
            LOGGER.info("Evaluated %d/%d: %s", index, len(tasks), label)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = {
                executor.submit(evaluate_configuration, task): task for task in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                label, metrics = future.result()
                results[label] = align_metrics(metrics, expected_base_metrics)
                LOGGER.info("Evaluated %d/%d: %s", index, len(tasks), label)
    return {task.label: results[task.label] for task in tasks}


def output_path_for(source: Path, timestamp: str) -> Path:
    prefix = TIMESTAMP_PATTERN.sub("", source.stem)
    return source.parent / f"{prefix}_updated_frl_{timestamp}.csv"


def recalculate_report(
    source_metrics: Path,
    matches_root: Path,
    artifact_type: str,
    updated_frl_path: Path,
    all_students_path: Path,
    new_ctip_path: Path | None,
    workers: int,
    timestamp: str,
) -> Path:
    source = pd.read_csv(source_metrics, index_col="metric")
    labels = source.columns.tolist()
    tasks = build_tasks(
        labels,
        matches_root,
        artifact_type,
        updated_frl_path,
        all_students_path,
        new_ctip_path,
    )
    metrics = run_tasks(tasks, source.index.tolist(), workers)
    frame = pd.DataFrame(metrics)
    frame.index.name = "metric"
    output_path = output_path_for(source_metrics, timestamp)
    with output_path.open("x", encoding="utf-8", newline="") as output_file:
        frame.to_csv(output_file)
    return output_path


def checked_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    updated_frl_path = checked_file(args.updated_frl)
    all_students_path = checked_file(args.all_students)
    fallback_report = fallback_block_report(
        pd.read_csv(all_students_path, low_memory=False),
        load_frl_lookup(updated_frl_path),
    )
    fallback_output = args.fallback_blocks_output.expanduser().resolve()
    fallback_output.parent.mkdir(parents=True, exist_ok=True)
    fallback_report.to_csv(fallback_output, index=False)
    LOGGER.info(
        "Wrote %d FRL fallback block rows to %s",
        len(fallback_report),
        fallback_output,
    )
    new_ctip_path = args.new_ctip_path.expanduser().resolve()
    if not new_ctip_path.is_file():
        LOGGER.warning(
            "Equity-block file not found; ET metrics will use no blocks: %s",
            new_ctip_path,
        )
        new_ctip_path = None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    outputs = []
    if args.dataset in {"both", "soft"}:
        outputs.append(
            recalculate_report(
                checked_file(args.soft_metrics),
                args.soft_matches_root.expanduser().resolve(),
                "soft",
                updated_frl_path,
                all_students_path,
                new_ctip_path,
                args.workers,
                timestamp,
            )
        )
    if args.dataset in {"both", "zone"}:
        outputs.append(
            recalculate_report(
                checked_file(args.zone_metrics),
                args.zone_matches_root.expanduser().resolve(),
                "zone",
                updated_frl_path,
                all_students_path,
                new_ctip_path,
                args.workers,
                timestamp,
            )
        )

    for output in outputs:
        LOGGER.info("Wrote recalculated metrics to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Recalculate saved assignment reports with updated block-level FRL rates.

The script reuses the 25 raw assignments behind two existing metric reports. It
does not rerun matching. Student latitude and longitude are mapped to a 2020
Census block, and each student's FRL probability is replaced by that block's
``FRL Rate`` when available. Otherwise, the evaluator's legacy
free-plus-reduced-lunch calculation is retained.

Passing ``--zone-real-matches-root`` evaluates each zone subconfiguration with
both the source choice-model assignments and observed-preference assignments.
The resulting CSV has adjacent ``__choice_model`` and ``__real_preferences``
columns for every source subconfiguration.

A CSV lists the KG 2020 Census blocks for which that legacy fallback is used.
It distinguishes missing or invalid coordinates, coordinates outside the block
geometry, blocks absent from the updated lookup, and blank updated rates.

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
import copy
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
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment.student_assignment.evaluation.match_evaluator import (  # noqa: E402
    MatchEvaluator,
)
from loaders import (  # noqa: E402
    DataScenario,
    ResolvedSource,
    load_program_records,
    load_scenario,
    load_school_records,
    load_student_records,
    normalize_student_records,
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
DEFAULT_2020_BLOCKS = Path(
    "/soalnas/share/data/school_choice/Census/2020/blocks/tl_2020_06075_tabblock20.shp"
)
DEFAULT_FALLBACK_BLOCKS = (
    PROJECT_ROOT / "analysis/recalculate_updated_frl_fallback_blocks.csv"
)
DEFAULT_ALL_STUDENTS = Path("/soalnas/share/data/school_choice/Data/Cleaned/student_2324.csv")
DEFAULT_NEW_CTIP = Path(
    "/soalnas/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/ETB_2024.npy"
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
    block_geometry_path: str
    all_students_path: str
    new_ctip_path: str | None
    first_round: bool = True


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
    parser.add_argument(
        "--zone-real-matches-root",
        type=Path,
        help="Optional observed-preference assignments for a paired zone report.",
    )
    parser.add_argument("--updated-frl", type=Path, default=DEFAULT_UPDATED_FRL)
    parser.add_argument(
        "--block-geometry",
        type=Path,
        default=DEFAULT_2020_BLOCKS,
        help="2020 Census tabulation-block geometry containing GEOID20.",
    )
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
    parser.add_argument(
        "--all-rounds",
        action="store_true",
        help="Evaluate every matched student instead of round-one applicants only.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return data


def evaluation_first_round(config: Mapping[str, Any]) -> bool:
    population = config.get("evaluation-population", "first_round")
    if population == "first_round":
        return True
    if population == "all_rounds":
        return False
    raise ValueError(f"unknown evaluation population: {population!r}")


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


@lru_cache(maxsize=None)
def load_2020_block_geometry(path: Path) -> gpd.GeoDataFrame:
    """Load 2020 Census blocks and normalize GEOID20 like the FRL lookup."""
    blocks = gpd.read_file(path, columns=["GEOID20"])
    if "GEOID20" not in blocks:
        raise ValueError(f"2020 block geometry has no GEOID20 column: {path}")
    if blocks.crs is None:
        raise ValueError(
            f"2020 block geometry has no coordinate reference system: {path}"
        )
    if blocks.geometry.isna().any() or blocks.geometry.is_empty.any():
        raise ValueError(f"2020 block geometry contains empty geometries: {path}")
    if (~blocks.geometry.is_valid).any():
        raise ValueError(f"2020 block geometry contains invalid geometries: {path}")

    result = blocks.rename(columns={"GEOID20": "census_block_2020"}).copy()
    result["census_block_2020"] = normalized_geoids(result["census_block_2020"])
    if result["census_block_2020"].isna().any():
        raise ValueError(f"2020 block geometry contains blank GEOID20 values: {path}")
    if result["census_block_2020"].duplicated().any():
        raise ValueError(
            f"2020 block geometry contains duplicate GEOID20 values: {path}"
        )
    return result[["census_block_2020", "geometry"]]


def student_2020_block_ids(
    students: pd.DataFrame,
    block_geometry: gpd.GeoDataFrame,
) -> pd.Series:
    """Map student WGS84 coordinates to normalized 2020 Census block IDs."""
    required = {"latitude", "longitude"}
    missing = required - set(students.columns)
    if missing:
        raise ValueError(f"student data is missing columns {sorted(missing)}")
    if "census_block_2020" not in block_geometry:
        raise ValueError("2020 block geometry has no census_block_2020 column")
    if block_geometry.crs is None:
        raise ValueError("2020 block geometry has no coordinate reference system")

    latitude = pd.to_numeric(students["latitude"], errors="coerce")
    longitude = pd.to_numeric(students["longitude"], errors="coerce")
    valid = latitude.between(-90, 90, inclusive="both") & longitude.between(
        -180, 180, inclusive="both"
    )
    result = pd.Series(pd.NA, index=students.index, dtype="string")
    if not valid.any():
        return result

    positions = np.flatnonzero(valid.to_numpy())
    points = gpd.GeoDataFrame(
        {"student_position": positions},
        geometry=gpd.points_from_xy(
            longitude.loc[valid].to_numpy(),
            latitude.loc[valid].to_numpy(),
        ),
        crs="EPSG:4326",
    ).to_crs(block_geometry.crs)
    matches = gpd.sjoin(
        points,
        block_geometry[["census_block_2020", "geometry"]],
        how="left",
        predicate="intersects",
    ).dropna(subset=["census_block_2020"])

    duplicate_positions = matches.loc[
        matches["student_position"].duplicated(keep=False), "student_position"
    ].unique()
    if len(duplicate_positions):
        examples = duplicate_positions[:3].tolist()
        raise ValueError(
            "student coordinates intersect multiple 2020 Census blocks, "
            f"including row positions {examples}"
        )

    result.iloc[matches["student_position"].astype(int).to_numpy()] = (
        matches["census_block_2020"].astype("string").to_numpy()
    )
    return result


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


def student_frl_data(
    students: pd.DataFrame,
    lookup: pd.Series,
    block_geometry: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Return legacy, updated, and effective FRL data keyed by 2020 block."""
    free = pd.to_numeric(
        students.get("freelunch_prob", pd.Series(0, index=students.index)),
        errors="coerce",
    ).fillna(0)
    reduced = pd.to_numeric(
        students.get("reducedlunch_prob", pd.Series(0, index=students.index)),
        errors="coerce",
    ).fillna(0)
    legacy_frl = free + reduced
    block_ids = student_2020_block_ids(students, block_geometry)
    updated_frl = block_ids.map(lookup)
    effective_frl = updated_frl.fillna(legacy_frl)

    latitude = pd.to_numeric(students["latitude"], errors="coerce")
    longitude = pd.to_numeric(students["longitude"], errors="coerce")
    coordinates_present = latitude.notna() & longitude.notna()
    coordinates_valid = (
        coordinates_present
        & latitude.between(-90, 90, inclusive="both")
        & longitude.between(-180, 180, inclusive="both")
    )
    fallback_reason = pd.Series(pd.NA, index=students.index, dtype="string")
    fallback_reason.loc[~coordinates_present] = "missing student coordinates"
    fallback_reason.loc[coordinates_present & ~coordinates_valid] = (
        "invalid student coordinates"
    )
    fallback_reason.loc[coordinates_valid & block_ids.isna()] = (
        "outside 2020 census block geometry"
    )
    fallback_reason.loc[block_ids.notna() & ~block_ids.isin(lookup.index)] = (
        "absent from updated lookup"
    )
    blank_rate = block_ids.notna() & block_ids.isin(lookup.index) & updated_frl.isna()
    fallback_reason.loc[blank_rate] = "blank updated FRL rate"

    return pd.DataFrame(
        {
            "census_block_2020": block_ids,
            "legacy_frl": legacy_frl,
            "updated_frl": updated_frl,
            "effective_frl": effective_frl,
            "frl_fallback_reason": fallback_reason,
        },
        index=students.index,
    )


def enrich_student_frl(
    students: pd.DataFrame,
    lookup: pd.Series,
    block_geometry: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Override legacy student FRL with non-null updated 2020-block rates."""
    result = students.copy()
    frl_data = student_frl_data(result, lookup, block_geometry)

    # MatchEvaluator derives `frl` by adding these two source columns.
    result["freelunch_prob"] = frl_data["effective_frl"].astype(float)
    result["reducedlunch_prob"] = 0.0
    result["census_block_2020"] = frl_data["census_block_2020"]
    return result


def fallback_block_report(
    students: pd.DataFrame,
    lookup: pd.Series,
    block_geometry: gpd.GeoDataFrame,
    grade: object = "KG",
) -> pd.DataFrame:
    """List 2020 blocks whose students retain legacy FRL values."""
    required = {
        "grade",
        "latitude",
        "longitude",
        "freelunch_prob",
        "reducedlunch_prob",
    }
    missing = required - set(students.columns)
    if missing:
        raise ValueError(f"student data is missing columns {sorted(missing)}")

    grade_students = students.loc[students["grade"].astype("string") == str(grade)]
    frl_data = student_frl_data(grade_students, lookup, block_geometry)

    fallback = pd.DataFrame(
        {
            "census_block_2020": frl_data["census_block_2020"],
            "frl_fallback_reason": frl_data["frl_fallback_reason"],
            "legacy_frl": frl_data["legacy_frl"],
        }
    ).loc[frl_data["updated_frl"].isna()]
    report = (
        fallback.groupby(
            ["census_block_2020", "frl_fallback_reason"],
            dropna=False,
            sort=False,
        )
        .agg(
            student_count=("legacy_frl", "size"),
            legacy_frl=("legacy_frl", "sum"),
        )
        .reset_index()
        .sort_values("census_block_2020", na_position="last")
        .reset_index(drop=True)
    )
    report["legacy_frl"] = report["legacy_frl"].round(6)
    return report


def config_data_scenario(config: Mapping[str, Any]) -> DataScenario:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("configuration has no data scenario")
    return load_scenario(data)


def resolve_data_path(config: Mapping[str, Any], role: str) -> Path:
    return config_data_scenario(config).source(role).path


def resolve_zone_definition(config: Mapping[str, Any]) -> ZoneDefinition | None:
    unit = str(config.get("zone-building-blocks") or "").strip().lower()
    if unit not in {"block", "block_group", "tract"}:
        return None

    policies = config.get("policies")
    if not isinstance(policies, list) or len(policies) != 1:
        raise ValueError("zone configuration must name exactly one policy")
    zone_files = config_data_scenario(config).source_map("assignment.zones")
    policy = str(policies[0])
    zone_source = zone_files.get(policy)
    if not isinstance(zone_source, ResolvedSource):
        raise ValueError(f"zone configuration has no file for policy {policy!r}")
    return load_zone_definition(zone_source.path, unit)


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
    if unit == "tract":
        return "Tract" if school else "census_tract"
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
    zone: ZoneDefinition,
) -> dict[str, float | list[float] | list[int]]:
    """Calculate static applicant metrics in ascending zone-number order."""
    required = {
        "freelunch_prob",
        "selected_programs",
        "selected_ranked_idschool",
        zone_column(zone.unit),
    }
    missing = required - set(all_students.columns)
    if missing:
        raise ValueError(f"all-student data is missing columns {sorted(missing)}")

    grade_students = all_students.copy()
    zones = map_areas_to_zones(grade_students[zone_column(zone.unit)], zone)
    mapped = zones.notna()
    district_frl = float(grade_students.loc[mapped, "freelunch_prob"].mean())

    program_lists = grade_students["selected_programs"].map(parse_ranked_list)
    ranked_school_lists = grade_students["selected_ranked_idschool"].map(
        parse_ranked_list
    )
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
    zone: ZoneDefinition,
) -> list[float]:
    metrics = zone_population_metrics(all_students, zone)
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
    data = copy.deepcopy(dict(config["data"]))
    data.setdefault("overrides", {}).setdefault("filters", {}).setdefault(
        "assignment", {}
    )["rounds"] = [1] if task.first_round else "all"
    scenario = load_scenario(data)

    lookup = load_frl_lookup(Path(task.updated_frl_path))
    block_geometry = load_2020_block_geometry(Path(task.block_geometry_path))
    students = enrich_student_frl(
        load_student_records(
            scenario,
            "assignment.students",
            filter_group="assignment",
            low_memory=False,
        ),
        lookup,
        block_geometry,
    )
    programs = load_program_records(
        scenario, "assignment.programs", filter_group="assignment"
    )
    schools = load_school_records(
        scenario, "assignment.schools", filter_group="assignment"
    )
    zone = resolve_zone_definition(config)

    evaluator_year = int(scenario.filter("assignment", "year")[:2])
    evaluator_grade = scenario.filter("assignment", "grades")[0]
    base_metrics: list[pd.Series] = []
    empty_school_iterations: list[list[float]] = []
    unassigned_iterations: list[list[float]] = []
    empty_ge_iterations: list[list[float]] = []

    for assignment_path in task.assignment_paths:
        assignment = pd.read_csv(assignment_path)
        evaluator = MatchEvaluator(
            students,
            assignment,
            first_round=task.first_round,
            dropout=False,
            low_income=95292,
            medium_income=95292,
            high_income=110850,
            grade=evaluator_grade,
            year=evaluator_year,
            no_special_program=False,
            program_data=programs,
            schools_data=schools,
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

    all_students = normalize_student_records(
        pd.read_csv(task.all_students_path, low_memory=False), scenario, "assignment"
    )
    all_students = enrich_student_frl(
        all_students,
        lookup,
        block_geometry,
    )
    population = zone_population_metrics(all_students, zone)
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
    block_geometry_path: Path,
    all_students_path: Path,
    new_ctip_path: Path | None,
    label_suffix: str = "",
    first_round: bool | None = None,
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
        task_first_round = (
            evaluation_first_round(load_yaml(config_path))
            if first_round is None
            else first_round
        )
        tasks.append(
            ConfigurationTask(
                label=f"{label}{label_suffix}",
                config_path=str(config_path),
                assignment_paths=discover_assignments(assignments_root),
                updated_frl_path=str(updated_frl_path),
                block_geometry_path=str(block_geometry_path),
                all_students_path=str(all_students_path),
                new_ctip_path=str(new_ctip_path) if new_ctip_path else None,
                first_round=task_first_round,
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
    block_geometry_path: Path,
    all_students_path: Path,
    new_ctip_path: Path | None,
    workers: int,
    timestamp: str,
    comparison_matches_root: Path | None = None,
    first_round: bool | None = None,
) -> Path:
    source = pd.read_csv(source_metrics, index_col="metric")
    labels = source.columns.tolist()
    tasks = build_tasks(
        labels,
        matches_root,
        artifact_type,
        updated_frl_path,
        block_geometry_path,
        all_students_path,
        new_ctip_path,
        "__choice_model" if comparison_matches_root is not None else "",
        first_round,
    )
    if comparison_matches_root is not None:
        comparison_tasks = build_tasks(
            labels,
            comparison_matches_root,
            artifact_type,
            updated_frl_path,
            block_geometry_path,
            all_students_path,
            new_ctip_path,
            "__real_preferences",
            first_round,
        )
        tasks = [
            task
            for pair in zip(tasks, comparison_tasks, strict=True)
            for task in pair
        ]
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
    block_geometry_path = checked_file(args.block_geometry)
    block_geometry = load_2020_block_geometry(block_geometry_path)
    all_students_path = checked_file(args.all_students)
    fallback_report = fallback_block_report(
        pd.read_csv(all_students_path, low_memory=False),
        load_frl_lookup(updated_frl_path),
        block_geometry,
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
                block_geometry_path,
                all_students_path,
                new_ctip_path,
                args.workers,
                timestamp,
                first_round=False if args.all_rounds else None,
            )
        )
    if args.dataset in {"both", "zone"}:
        outputs.append(
            recalculate_report(
                checked_file(args.zone_metrics),
                args.zone_matches_root.expanduser().resolve(),
                "zone",
                updated_frl_path,
                block_geometry_path,
                all_students_path,
                new_ctip_path,
                args.workers,
                timestamp,
                args.zone_real_matches_root.expanduser().resolve()
                if args.zone_real_matches_root
                else None,
                first_round=False if args.all_rounds else None,
            )
        )

    for output in outputs:
        LOGGER.info("Wrote recalculated metrics to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

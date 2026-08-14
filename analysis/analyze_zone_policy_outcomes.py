#!/usr/bin/env python3
"""Analyze saved zone-policy assignments without rerunning matching.

The analysis treats choice-model and observed-preference assignments as equal
comparison modes. It validates and reads the 25 saved assignments for every
zone subconfiguration, reconstructs raw preferences before policy filtering,
and writes aggregate policy-outcome CSVs with no student identifiers.

Usage:
    uv run python analysis/analyze_zone_policy_outcomes.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.evaluate_zone_subconfig_matches import SUBCONFIGS  # noqa: E402
from analysis.recalculate_updated_frl_metrics import (  # noqa: E402
    DEFAULT_2020_BLOCKS,
    DEFAULT_UPDATED_FRL,
    discover_assignments,
    load_2020_block_geometry,
    load_frl_lookup,
    load_yaml,
    parse_ranked_list,
    resolve_data_path,
    student_frl_data,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (  # noqa: E402
    MarketGenerator,
)

LOGGER = logging.getLogger(__name__)

ITERATION_COUNT = 25
BASELINE_POLICY = "status_quo"
MODE_ORDER = ("choice_model", "real_preferences")
SES_SOURCES = ("legacy", "updated")
FAKE_TOP_POLICIES = (
    "status_quo_4",
    "distance_05_1_2+reserves_05frl_#4",
)

DEFAULT_CHOICE_ROOT = (
    PROJECT_ROOT
    / "analysis/matches/zone_subconfigs_rerun_20260811T043406Z_choice_model_25"
)
DEFAULT_REAL_ROOT = (
    PROJECT_ROOT
    / "analysis/matches/zone_subconfigs_rerun_20260811T043406Z_real_preferences_no_special_25"
)

CHOICE_ASSIGNMENT_COLUMNS = (
    "studentno",
    "programno",
    "programcodes",
    "rank",
    "designation",
    "assigned_utility",
    "In-Zone Rank",
)
REAL_ASSIGNMENT_COLUMNS = tuple(
    column for column in CHOICE_ASSIGNMENT_COLUMNS if column != "assigned_utility"
)
OUTPUT_FILENAMES = (
    "fake_top_choice_by_iteration.csv",
    "fake_top_choice_summary.csv",
    "school_enrollment.csv",
    "school_ses_transitions.csv",
    "travel_by_iteration.csv",
    "travel_summary.csv",
    "winners_losers.csv",
)


@dataclass(slots=True)
class AssignmentData:
    """Compact validated assignment arrays in canonical student order."""

    programno: np.ndarray
    rank: np.ndarray
    designation: np.ndarray
    assigned_utility: np.ndarray | None


@dataclass(slots=True)
class RawPreferenceData:
    """Raw ranks for loaded programs plus unfiltered top-choice metadata."""

    ranks: np.ndarray
    applicant_mask: np.ndarray
    top_school_ids: np.ndarray
    unloaded_program_entries: int


@dataclass(slots=True)
class Stratum:
    """One non-crossed winner/loser reporting stratum."""

    stratum_type: str
    stratum_value: str
    indices: np.ndarray
    attendance_school_id: int | None = None
    attendance_school_name: str | None = None


@dataclass(slots=True)
class ModeData:
    """Validated inputs and reusable arrays for one preference mode."""

    mode: str
    mode_order: int
    root: Path
    config_path: Path
    config: dict[str, Any]
    students: pd.DataFrame
    programs: pd.DataFrame
    schools: pd.DataFrame
    loaded_schools: pd.DataFrame
    assignments: dict[str, tuple[AssignmentData, ...]]
    raw_ranks: tuple[np.ndarray, ...]
    raw_top_disallowed: tuple[np.ndarray, ...]
    applicant_masks: tuple[np.ndarray, ...]
    frl: dict[str, np.ndarray]
    district_frl: dict[str, float]
    family_frl_tiers: dict[str, np.ndarray]
    distance_by_program: np.ndarray
    program_school_positions: np.ndarray
    program_school_ids: np.ndarray
    school_id_to_position: dict[int, int]
    fallback_count: int
    raw_unloaded_program_entries: int = 0
    utility_validation_count: int = 0
    utility_max_abs_error: float = 0.0
    school_frl: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class WinnerStats:
    """Running iteration-level moments for one winner/loser output row."""

    stratum_student_count: int
    iterations: int = 0
    count_sums: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    rate_counts: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=int))
    rate_sums: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    rate_squares: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    delta_count: int = 0
    delta_sum: float = 0.0
    delta_square: float = 0.0

    def update(self, result: Mapping[str, float | int]) -> None:
        self.iterations += 1
        counts = np.array(
            [
                result["eligible_count"],
                result["win_count"],
                result["tie_count"],
                result["loss_count"],
            ],
            dtype=float,
        )
        self.count_sums += counts
        rates = np.array(
            [
                result["eligible_rate"],
                result["win_rate"],
                result["tie_rate"],
                result["loss_rate"],
            ],
            dtype=float,
        )
        finite = np.isfinite(rates)
        self.rate_counts[finite] += 1
        self.rate_sums[finite] += rates[finite]
        self.rate_squares[finite] += rates[finite] ** 2
        delta = float(result["mean_delta"])
        if math.isfinite(delta):
            self.delta_count += 1
            self.delta_sum += delta
            self.delta_square += delta * delta


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choice-root", type=Path, default=DEFAULT_CHOICE_ROOT)
    parser.add_argument("--real-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "New output directory. By default, a timestamped analysis directory "
            "is created directly under --choice-root."
        ),
    )
    parser.add_argument("--updated-frl", type=Path, default=DEFAULT_UPDATED_FRL)
    parser.add_argument("--block-geometry", type=Path, default=DEFAULT_2020_BLOCKS)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def sample_sd_from_moments(count: int, total: float, square: float) -> float:
    if count < 2:
        return float("nan")
    variance = (square - total * total / count) / (count - 1)
    return math.sqrt(max(0.0, variance))


def finite_mean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


def finite_sd(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")


def finite_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.min()) if len(finite) else float("nan")


def finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max()) if len(finite) else float("nan")


def frl_tier(value: float, district_mean: float, *, missing: str) -> str:
    """Classify a probability using strict district mean +/- 0.15 cutoffs."""
    if not math.isfinite(float(value)):
        return missing
    if value > district_mean + 0.15:
        return "high"
    if value < district_mean - 0.15:
        return "low"
    return "medium"


def family_frl_tiers(values: np.ndarray, district_mean: float) -> np.ndarray:
    return np.asarray(
        [frl_tier(float(value), district_mean, missing="missing") for value in values],
        dtype=object,
    )


def inverse_preference_ranks(preferences: np.ndarray, num_programs: int) -> np.ndarray:
    """Convert row-ordered 1-based program preferences to inverse ranks."""
    preferences = np.asarray(preferences)
    if preferences.ndim != 2 or preferences.shape[1] != num_programs:
        raise ValueError(
            "preference matrix must have one column for every loaded program"
        )
    numeric = preferences.astype(int)
    expected = np.arange(1, num_programs + 1)
    if not np.array_equal(
        np.sort(numeric, axis=1), np.tile(expected, (len(numeric), 1))
    ):
        raise ValueError("choice-model preferences must be program permutations")
    result = np.zeros_like(numeric, dtype=np.int16)
    rows = np.arange(len(numeric))[:, np.newaxis]
    result[rows, numeric - 1] = expected.astype(np.int16)
    return result


def _list_value(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, np.ndarray)):
        return list(value)
    return parse_ranked_list(value)


def build_real_raw_preference_ranks(
    student_data: pd.DataFrame,
    first_round: np.ndarray,
    program_indices: Mapping[str, int],
    grade: str,
) -> np.ndarray:
    """Build raw exact-program ranks from each student's first active round."""
    return build_real_raw_preference_data(
        student_data, first_round, program_indices, grade
    ).ranks


def build_real_raw_preference_data(
    student_data: pd.DataFrame,
    first_round: np.ndarray,
    program_indices: Mapping[str, int],
    grade: str,
    valid_school_ids: set[int] | None = None,
) -> RawPreferenceData:
    """Retain raw order even when policy filtering removed a listed program."""
    if len(student_data) != len(first_round):
        raise ValueError("first-round array does not match student population")
    num_programs = len(program_indices)
    ranks = np.zeros((len(student_data), num_programs), dtype=np.int16)
    applicant_mask = np.zeros(len(student_data), dtype=bool)
    top_school_ids = np.full(len(student_data), np.nan, dtype=float)
    unloaded_program_entries = 0
    for position, (_, student) in enumerate(student_data.iterrows()):
        round_number = int(first_round[position]) + 1
        schools_column = f"r{round_number}_ranked_idschool"
        programs_column = f"r{round_number}_programs"
        if schools_column not in student_data or programs_column not in student_data:
            raise ValueError(
                f"first active round {round_number} has no raw preference columns"
            )
        schools = _list_value(student[schools_column])
        program_types = _list_value(student[programs_column])
        if len(schools) != len(program_types):
            raise ValueError(
                "raw ranked-school and program lists differ in length for row "
                f"{position}"
            )
        applicant_mask[position] = bool(schools)
        seen: set[str] = set()
        for rank, (school, program_type) in enumerate(
            zip(schools, program_types, strict=True), start=1
        ):
            numeric_school = pd.to_numeric(pd.Series([school]), errors="coerce").iloc[0]
            if pd.isna(numeric_school) or float(numeric_school) % 1:
                raise ValueError(
                    f"invalid raw school ID at student row {position}: {school}"
                )
            school_id = int(numeric_school)
            if valid_school_ids is not None and school_id not in valid_school_ids:
                raise ValueError(
                    f"raw preference references unknown school {school_id}"
                )
            program_id = f"{school_id}-{str(program_type)}-{grade}"
            if program_id in seen:
                raise ValueError(
                    f"raw preferences repeat exact program {program_id!r} at row {position}"
                )
            seen.add(program_id)
            if rank == 1:
                top_school_ids[position] = school_id
            if program_id in program_indices:
                programno = int(program_indices[program_id])
                ranks[position, programno - 1] = rank
            else:
                unloaded_program_entries += 1
    return RawPreferenceData(
        ranks=ranks,
        applicant_mask=applicant_mask,
        top_school_ids=top_school_ids,
        unloaded_program_entries=unloaded_program_entries,
    )


def assigned_program_values(matrix: np.ndarray, programno: np.ndarray) -> np.ndarray:
    """Look up a student-by-program matrix, returning NaN when unassigned."""
    result = np.full(len(programno), np.nan, dtype=float)
    assigned = programno > 0
    result[assigned] = matrix[
        np.flatnonzero(assigned), programno[assigned].astype(int) - 1
    ]
    return result


def assigned_raw_ranks(raw_ranks: np.ndarray, programno: np.ndarray) -> np.ndarray:
    return assigned_program_values(raw_ranks, programno).astype(float)


def haversine_miles(
    latitude: float,
    longitude: float,
    school_latitude: float,
    school_longitude: float,
) -> float:
    """Match ``MatchEvaluator``'s Haversine formula and 3958.8-mile radius."""
    values = (latitude, longitude, school_latitude, school_longitude)
    if any(pd.isna(value) for value in values):
        return float("nan")
    lat1, lat2, lon1, lon2 = [
        math.radians(float(value))
        for value in (latitude, school_latitude, longitude, school_longitude)
    ]
    angle = 2 * math.asin(
        math.sqrt(
            math.sin((lat2 - lat1) / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
        )
    )
    return 3958.8 * angle


def _integer_array(series: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain finite integers")
    return numeric.astype(np.int64)


def read_validated_assignment(
    path: Path,
    mode: str,
    expected_student_ids: np.ndarray,
    program_ids: np.ndarray,
) -> AssignmentData:
    """Read one assignment and enforce its complete schema and mappings."""
    with path.open(newline="", encoding="utf-8") as input_file:
        header = next(csv.reader(input_file), None)
    expected_columns = (
        CHOICE_ASSIGNMENT_COLUMNS if mode == "choice_model" else REAL_ASSIGNMENT_COLUMNS
    )
    if header is None:
        raise ValueError(f"assignment is empty: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"assignment has duplicate column names: {path}")
    if tuple(header) != expected_columns:
        raise ValueError(
            f"assignment schema differs at {path}: expected={expected_columns}, "
            f"actual={tuple(header)}"
        )

    frame = pd.read_csv(path)
    student_ids = _integer_array(frame["studentno"], "studentno")
    if len(student_ids) != len(expected_student_ids):
        raise ValueError(
            f"assignment row count differs at {path}: {len(student_ids)} != "
            f"{len(expected_student_ids)}"
        )
    if len(np.unique(student_ids)) != len(student_ids):
        raise ValueError(f"assignment contains duplicate student IDs: {path}")
    if not np.array_equal(student_ids, expected_student_ids):
        raise ValueError(f"assignment student set or order differs at {path}")

    programno = _integer_array(frame["programno"], "programno")
    if ((programno < 0) | (programno > len(program_ids))).any():
        raise ValueError(f"assignment contains out-of-range program numbers: {path}")
    codes = frame["programcodes"].astype("string").fillna("").str.strip().to_numpy()
    assigned = programno > 0
    expected_codes = np.full(len(frame), "", dtype=object)
    expected_codes[assigned] = program_ids[programno[assigned] - 1]
    if not np.array_equal(codes.astype(object), expected_codes):
        mismatch = int(np.flatnonzero(codes.astype(object) != expected_codes)[0])
        raise ValueError(
            f"program number/code mapping differs at {path}, row {mismatch}"
        )

    rank = _integer_array(frame["rank"], "rank")
    in_zone_rank = _integer_array(frame["In-Zone Rank"], "In-Zone Rank")
    if (rank < 0).any() or (in_zone_rank < 0).any():
        raise ValueError(f"assignment ranks cannot be negative: {path}")
    designation = _integer_array(frame["designation"], "designation")
    if not np.isin(designation, [0, 1]).all():
        raise ValueError(f"designation must be a 0/1 output flag: {path}")

    utility = None
    if mode == "choice_model":
        utility = pd.to_numeric(frame["assigned_utility"], errors="coerce").to_numpy(
            dtype=float
        )
        if np.isnan(utility[assigned]).any():
            raise ValueError(f"assigned rows have missing assigned_utility: {path}")
        if not np.isnan(utility[~assigned]).all():
            raise ValueError(
                f"unassigned rows unexpectedly have assigned_utility: {path}"
            )

    return AssignmentData(
        programno=programno.astype(np.int16),
        rank=rank.astype(np.int16),
        designation=designation.astype(np.int8),
        assigned_utility=utility,
    )


def _config_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("generated config has no paths mapping")
    utility = config.get("utility-model")
    if not isinstance(utility, Mapping):
        raise ValueError("generated config has no utility-model mapping")
    return {
        "grade": config.get("grade"),
        "year": config.get("year"),
        "iterations": config.get("iterations"),
        "random-seed": config.get("random-seed"),
        "remove-special-lps": config.get("remove-special-lps"),
        "r1-only": config.get("r1-only"),
        "student-data": str(paths.get("student-data")),
        "program-data": str(paths.get("program-data")),
        "school-data": str(paths.get("school-data")),
        "estimate-path": str(paths.get("estimate-path")),
        "utility-enable": utility.get("enable"),
        "utility-list-length": utility.get("list-length"),
        "gumbel-scale": utility.get("gumbel-scale", 1.0),
    }


def validate_generated_configs(root: Path, mode: str) -> tuple[Path, dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    simulation_path = root / "simulation_config.yaml"
    simulation = load_yaml(simulation_path)
    if tuple(simulation.get("subconfigs", [])) != tuple(SUBCONFIGS):
        raise ValueError(f"simulation subconfig order differs at {simulation_path}")

    representative_path = root / BASELINE_POLICY / "policy_config.generated.yaml"
    representative = load_yaml(representative_path)
    signature = _config_signature(representative)
    expected_enable = mode == "choice_model"
    if signature["utility-enable"] is not expected_enable:
        raise ValueError(
            f"utility-model mode does not match {mode}: {representative_path}"
        )
    if signature["iterations"] != {"start": 0, "end": ITERATION_COUNT}:
        raise ValueError("generated config does not specify iterations 0..24")
    if signature["random-seed"] != 2023:
        raise ValueError("generated assignments must use random seed 2023")

    for label in SUBCONFIGS:
        path = root / label / "policy_config.generated.yaml"
        config = load_yaml(path)
        if _config_signature(config) != signature:
            raise ValueError(f"market input configuration differs for policy {label}")
        if config.get("subconfig-name") != label:
            raise ValueError(f"generated config has wrong subconfig-name: {path}")
        if config.get("ties-options") != ["MTB"]:
            raise ValueError(f"analysis expects exactly one MTB tie-breaker: {path}")
        if label in FAKE_TOP_POLICIES and (
            config.get("remove_non_aa_or_citywide") is not True
            or config.get("overscribe_aa") is not True
        ):
            LOGGER.warning(
                "Fake-top results may be less meaningful without _4 filtering and "
                "AA oversubscription: %s",
                path,
            )
    return representative_path, representative


def _canonical_programs(market: MarketGenerator) -> pd.DataFrame:
    programs = market.programs.program_df.copy()
    required = {"program_id", "programno", "school_id", "program_type", "capacity"}
    missing = required - set(programs)
    if missing:
        raise ValueError(f"loaded program data is missing columns {sorted(missing)}")
    programs["program_id"] = programs["program_id"].astype(str)
    programs["programno"] = _integer_array(programs["programno"], "programno")
    programs["school_id"] = _integer_array(programs["school_id"], "school_id")
    programs["capacity"] = pd.to_numeric(programs["capacity"], errors="coerce").fillna(
        0
    )
    programs = programs.sort_values("programno").reset_index(drop=True)
    if (
        programs["program_id"].duplicated().any()
        or programs["programno"].duplicated().any()
    ):
        raise ValueError("loaded program IDs and numbers must be unique")
    if programs["programno"].tolist() != list(range(1, len(programs) + 1)):
        raise ValueError("loaded program numbers must be contiguous and 1-based")
    parsed_school_ids = pd.to_numeric(
        programs["program_id"].str.split("-", n=1).str[0], errors="coerce"
    )
    if not np.array_equal(
        parsed_school_ids.to_numpy(), programs["school_id"].to_numpy()
    ):
        raise ValueError("program IDs do not map to their school_id column")
    return programs


def _canonical_schools(
    config: Mapping[str, Any], programs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schools = pd.read_csv(resolve_data_path(config, "school-data"))
    required = {"school_id", "school_name", "category", "lat", "lon"}
    missing = required - set(schools)
    if missing:
        raise ValueError(f"school data is missing columns {sorted(missing)}")
    schools["school_id"] = _integer_array(schools["school_id"], "school_id")
    if schools["school_id"].duplicated().any():
        raise ValueError("school data contains duplicate school IDs")
    loaded_ids = sorted(programs["school_id"].unique().tolist())
    unknown = sorted(set(loaded_ids) - set(schools["school_id"]))
    if unknown:
        raise ValueError(f"programs map to unknown schools, including {unknown[:3]}")
    loaded = schools.loc[schools["school_id"].isin(loaded_ids)].copy()
    loaded = loaded.set_index("school_id").loc[loaded_ids].reset_index()
    return schools, loaded


def _distance_matrix(
    students: pd.DataFrame,
    programs: pd.DataFrame,
    schools: pd.DataFrame,
) -> np.ndarray:
    locations = schools.set_index("school_id")[["lat", "lon"]].to_dict("index")
    latitude = pd.to_numeric(students["latitude"], errors="coerce").to_numpy(float)
    longitude = pd.to_numeric(students["longitude"], errors="coerce").to_numpy(float)
    school_cache: dict[int, np.ndarray] = {}
    result = np.full((len(students), len(programs)), np.nan, dtype=float)
    for column, school_id in enumerate(programs["school_id"].to_numpy(dtype=int)):
        if school_id not in school_cache:
            location = locations[school_id]
            school_cache[school_id] = np.fromiter(
                (
                    haversine_miles(lat, lon, location["lat"], location["lon"])
                    for lat, lon in zip(latitude, longitude, strict=True)
                ),
                dtype=float,
                count=len(students),
            )
        result[:, column] = school_cache[school_id]
    return result


def _raw_top_disallowed(
    raw_ranks: np.ndarray,
    attendance_areas: np.ndarray,
    program_school_ids: np.ndarray,
    citywide_school_ids: set[int],
) -> np.ndarray:
    top_mask = raw_ranks == 1
    has_top = top_mask.any(axis=1)
    top_program_positions = np.argmax(top_mask, axis=1)
    top_school = program_school_ids[top_program_positions]
    citywide = np.isin(top_school, list(citywide_school_ids))
    return has_top & (top_school != attendance_areas) & ~citywide


def _top_school_disallowed(
    top_school_ids: np.ndarray,
    attendance_areas: np.ndarray,
    citywide_school_ids: set[int],
) -> np.ndarray:
    has_top = np.isfinite(top_school_ids)
    citywide = np.isin(top_school_ids, list(citywide_school_ids))
    return has_top & (top_school_ids != attendance_areas) & ~citywide


def _choice_raw_ranks(
    market: MarketGenerator,
    config: Mapping[str, Any],
    assignments: Mapping[str, tuple[AssignmentData, ...]],
) -> tuple[tuple[np.ndarray, ...], int, float]:
    seed = int(config["random-seed"])
    if seed != 2023:
        raise ValueError("choice utility reconstruction requires seed 2023")
    n = market.students.n
    p = market.programs.num_programs
    gumbel_scale = float(config["utility-model"].get("gumbel-scale", 1.0))
    np.random.seed(seed)
    inverse_ranks: list[np.ndarray] = []
    validation_count = 0
    max_abs_error = 0.0
    for iteration in range(ITERATION_COUNT):
        market.umodel.draw_utility_model_randomness(
            iteration,
            rows_to_keep=market.students.only_keep_rows,
            cols_to_keep=market.programs.only_keep_cols,
            gumbel_scale=gumbel_scale,
        )
        utilities = market.umodel.original_utilities
        preferences = market.umodel.original_preferences
        inverse_ranks.append(inverse_preference_ranks(preferences, p))
        for label in SUBCONFIGS:
            assignment = assignments[label][iteration]
            if assignment.assigned_utility is None:
                raise ValueError(
                    f"choice assignment has no utility: {label}/{iteration}"
                )
            expected = assigned_program_values(utilities, assignment.programno)
            assigned = assignment.programno > 0
            finite = (
                assigned
                & np.isfinite(expected)
                & np.isfinite(assignment.assigned_utility)
            )
            differences = np.abs(expected[finite] - assignment.assigned_utility[finite])
            if len(differences):
                max_abs_error = max(max_abs_error, float(differences.max()))
            if not np.allclose(
                expected,
                assignment.assigned_utility,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    "regenerated choice utility differs from saved assigned_utility "
                    f"for {label}, iteration {iteration}"
                )
            validation_count += int(assigned.sum())

        # Matching consumes one full MTB lottery before the next utility draw.
        np.random.rand(n, p)
    return tuple(inverse_ranks), validation_count, max_abs_error


def load_mode_data(
    mode: str,
    root: Path,
    lookup: pd.Series,
    block_geometry: Any,
) -> ModeData:
    """Instantiate a non-saving market and validate every saved assignment."""
    config_path, config = validate_generated_configs(root, mode)
    initialization_config = copy.deepcopy(config)
    initialization_config["save-assignment"] = False
    market = MarketGenerator(config=initialization_config)

    students = market.students.student_data.copy()
    if students.index.name != "studentno":
        raise ValueError("market student data must be indexed by studentno")
    student_ids = _integer_array(students.index.to_series(), "studentno")
    if len(np.unique(student_ids)) != len(student_ids):
        raise ValueError("market population contains duplicate student IDs")

    programs = _canonical_programs(market)
    schools, loaded_schools = _canonical_schools(config, programs)
    program_ids = programs["program_id"].to_numpy(dtype=object)
    assignments: dict[str, tuple[AssignmentData, ...]] = {}
    for policy_order, label in enumerate(SUBCONFIGS):
        paths = tuple(Path(path) for path in discover_assignments(root / label))
        assignments[label] = tuple(
            read_validated_assignment(path, mode, student_ids, program_ids)
            for path in paths
        )
        LOGGER.info(
            "Validated %s %d/%d: %s",
            mode,
            policy_order + 1,
            len(SUBCONFIGS),
            label,
        )

    real_raw_data = None
    if mode == "choice_model":
        raw_ranks, utility_count, utility_error = _choice_raw_ranks(
            market, config, assignments
        )
    else:
        real_raw_data = build_real_raw_preference_data(
            students,
            market.students.first_round,
            market.programs.indices,
            str(config["grade"]),
            set(schools["school_id"].astype(int)),
        )
        raw_ranks = tuple(real_raw_data.ranks for _ in range(ITERATION_COUNT))
        utility_count = 0
        utility_error = 0.0

    frl_data = student_frl_data(students, lookup, block_geometry)
    frl = {
        "legacy": frl_data["legacy_frl"].to_numpy(dtype=float),
        "updated": frl_data["effective_frl"].to_numpy(dtype=float),
    }
    for source, values in frl.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{mode} {source} FRL values are not all finite")
    district_frl = {source: float(values.mean()) for source, values in frl.items()}
    tiers = {
        source: family_frl_tiers(values, district_frl[source])
        for source, values in frl.items()
    }

    loaded_school_ids = loaded_schools["school_id"].to_numpy(dtype=int)
    school_id_to_position = {
        int(school_id): position for position, school_id in enumerate(loaded_school_ids)
    }
    program_school_ids = programs["school_id"].to_numpy(dtype=int)
    program_school_positions = np.asarray(
        [school_id_to_position[int(school_id)] for school_id in program_school_ids],
        dtype=np.int16,
    )
    attendance_numeric = pd.to_numeric(students["idschoolattendance"], errors="coerce")
    invalid_attendance = (
        students["idschoolattendance"].notna() & attendance_numeric.isna()
    )
    if invalid_attendance.any():
        raise ValueError("attendance area contains nonnumeric nonmissing values")
    attendance_areas = _integer_array(attendance_numeric.fillna(0), "attendance area")
    citywide_ids = set(
        schools.loc[
            schools["category"].astype("string").str.casefold() == "citywide",
            "school_id",
        ].astype(int)
    )
    if real_raw_data is None:
        raw_disallowed = tuple(
            _raw_top_disallowed(
                ranks, attendance_areas, program_school_ids, citywide_ids
            )
            for ranks in raw_ranks
        )
        applicant_masks = tuple((ranks > 0).any(axis=1) for ranks in raw_ranks)
        raw_unloaded_program_entries = 0
    else:
        disallowed = _top_school_disallowed(
            real_raw_data.top_school_ids, attendance_areas, citywide_ids
        )
        raw_disallowed = tuple(disallowed for _ in range(ITERATION_COUNT))
        applicant_masks = tuple(
            real_raw_data.applicant_mask for _ in range(ITERATION_COUNT)
        )
        raw_unloaded_program_entries = real_raw_data.unloaded_program_entries

    return ModeData(
        mode=mode,
        mode_order=MODE_ORDER.index(mode),
        root=root,
        config_path=config_path,
        config=config,
        students=students,
        programs=programs,
        schools=schools,
        loaded_schools=loaded_schools,
        assignments=assignments,
        raw_ranks=raw_ranks,
        raw_top_disallowed=raw_disallowed,
        applicant_masks=applicant_masks,
        frl=frl,
        district_frl=district_frl,
        family_frl_tiers=tiers,
        distance_by_program=_distance_matrix(students, programs, schools),
        program_school_positions=program_school_positions,
        program_school_ids=program_school_ids,
        school_id_to_position=school_id_to_position,
        fallback_count=int(frl_data["updated_frl"].isna().sum()),
        raw_unloaded_program_entries=raw_unloaded_program_entries,
        utility_validation_count=utility_count,
        utility_max_abs_error=utility_error,
    )


def fake_top_choice_metrics(
    assignment: AssignmentData,
    raw_ranks: np.ndarray,
    applicant_mask: np.ndarray,
    raw_top_disallowed: np.ndarray,
) -> dict[str, int | float]:
    assigned = assignment.programno > 0
    assigned_count = int(assigned.sum())
    applicant_count = int(applicant_mask.sum())
    raw_rank = assigned_raw_ranks(raw_ranks, assignment.programno)
    reported_top = assigned & (assignment.rank == 1)
    corrected_top = assigned & (raw_rank == 1)
    fake = reported_top & ~corrected_top
    fake_disallowed = fake & raw_top_disallowed
    assigned_disallowed = assigned & raw_top_disallowed
    return {
        "applicant_count": applicant_count,
        "assigned_count": assigned_count,
        "reported_top1_count": int(reported_top.sum()),
        "reported_top1_rate_assigned": safe_rate(reported_top.sum(), assigned_count),
        "fake_top1_count": int(fake.sum()),
        "fake_share_reported_top1": safe_rate(fake.sum(), reported_top.sum()),
        "corrected_raw_top1_count": int(corrected_top.sum()),
        "corrected_raw_top1_rate_assigned": safe_rate(
            corrected_top.sum(), assigned_count
        ),
        "corrected_raw_top1_rate_applicants": safe_rate(
            corrected_top.sum(), applicant_count
        ),
        "raw_top_disallowed_count_applicants": int(
            (applicant_mask & raw_top_disallowed).sum()
        ),
        "raw_top_disallowed_rate_applicants": safe_rate(
            (applicant_mask & raw_top_disallowed).sum(), applicant_count
        ),
        "raw_top_disallowed_count_assigned": int(assigned_disallowed.sum()),
        "raw_top_disallowed_rate_assigned": safe_rate(
            assigned_disallowed.sum(), assigned_count
        ),
        "fake_raw_top_disallowed_count": int(fake_disallowed.sum()),
        "fake_raw_top_disallowed_share_fake": safe_rate(
            fake_disallowed.sum(), fake.sum()
        ),
    }


def build_fake_top_choice(
    mode_data: Sequence[ModeData],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for data in mode_data:
        for label in FAKE_TOP_POLICIES:
            for iteration, assignment in enumerate(data.assignments[label]):
                rows.append(
                    {
                        "mode": data.mode,
                        "mode_order": data.mode_order,
                        "policy": label,
                        "policy_order": SUBCONFIGS.index(label),
                        "iteration": iteration,
                        **fake_top_choice_metrics(
                            assignment,
                            data.raw_ranks[iteration],
                            data.applicant_masks[iteration],
                            data.raw_top_disallowed[iteration],
                        ),
                    }
                )
    by_iteration = pd.DataFrame(rows)
    summary = macro_summary(
        by_iteration,
        ["mode", "mode_order", "policy", "policy_order"],
    )
    return by_iteration, summary


def enrollment_value_statistics(values: np.ndarray, capacity: float) -> dict[str, Any]:
    """Summarize one school's enrollment over iterations."""
    values = np.asarray(values, dtype=float)
    if len(values) != ITERATION_COUNT:
        raise ValueError(f"expected {ITERATION_COUNT} enrollment iterations")
    mean = float(values.mean())
    if capacity > 0:
        utilization = values / capacity
        mean_utilization = float(utilization.mean())
        under_100 = float((utilization < 1).mean())
        under_90 = float((utilization < 0.9).mean())
        over_100 = float((utilization > 1).mean())
        if mean_utilization < 0.9:
            mean_status = "under_90_percent"
        elif mean_utilization < 1:
            mean_status = "under_capacity"
        elif mean_utilization > 1:
            mean_status = "over_capacity"
        else:
            mean_status = "at_capacity"
    else:
        mean_utilization = float("nan")
        under_100 = under_90 = over_100 = float("nan")
        mean_status = "no_capacity"
    gaps = capacity - values
    return {
        "capacity": float(capacity),
        "mean_enrollment": mean,
        "sd_enrollment": float(values.std(ddof=1)),
        "min_enrollment": float(values.min()),
        "max_enrollment": float(values.max()),
        "mean_utilization": mean_utilization,
        "mean_seat_gap": float(gaps.mean()),
        "mean_empty_seats": float(np.clip(gaps, 0, None).mean()),
        "mean_over_seats": float(np.clip(-gaps, 0, None).mean()),
        "share_iterations_under_100pct": under_100,
        "share_iterations_under_90pct": under_90,
        "share_iterations_over_100pct": over_100,
        "mean_status": mean_status,
    }


def _assignment_school_positions(
    data: ModeData, assignment: AssignmentData
) -> np.ndarray:
    positions = np.full(len(assignment.programno), -1, dtype=np.int16)
    assigned = assignment.programno > 0
    positions[assigned] = data.program_school_positions[
        assignment.programno[assigned] - 1
    ]
    return positions


def build_school_enrollment(mode_data: Sequence[ModeData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for data in mode_data:
        school_count = len(data.loaded_schools)
        program_capacity = data.programs["capacity"].to_numpy(dtype=float)
        program_types = data.programs["program_type"].astype(str).str.upper().to_numpy()
        school_categories = (
            data.loaded_schools["category"].astype("string").str.casefold().to_numpy()
        )
        scope_program_masks = {
            "all_loaded_programs": np.ones(len(data.programs), dtype=bool),
            "attendance_school_ge": (program_types == "GE")
            & (school_categories[data.program_school_positions] == "attendance"),
        }
        for policy_order, label in enumerate(SUBCONFIGS):
            for scope_order, (scope, program_mask) in enumerate(
                scope_program_masks.items()
            ):
                capacities = np.bincount(
                    data.program_school_positions[program_mask],
                    weights=program_capacity[program_mask],
                    minlength=school_count,
                )
                included_schools = np.unique(
                    data.program_school_positions[program_mask]
                )
                enrollments = np.zeros((ITERATION_COUNT, school_count), dtype=float)
                for iteration, assignment in enumerate(data.assignments[label]):
                    assigned = assignment.programno > 0
                    program_positions = assignment.programno[assigned] - 1
                    in_scope = program_mask[program_positions]
                    school_positions = data.program_school_positions[
                        program_positions[in_scope]
                    ]
                    enrollments[iteration] = np.bincount(
                        school_positions, minlength=school_count
                    )
                for school_position in included_schools:
                    school = data.loaded_schools.iloc[int(school_position)]
                    rows.append(
                        {
                            "mode": data.mode,
                            "mode_order": data.mode_order,
                            "policy": label,
                            "policy_order": policy_order,
                            "scope": scope,
                            "scope_order": scope_order,
                            "school_id": int(school["school_id"]),
                            "school_name": school["school_name"],
                            "school_category": school["category"],
                            **enrollment_value_statistics(
                                enrollments[:, school_position],
                                float(capacities[school_position]),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _school_frl_means(
    data: ModeData,
    assignment: AssignmentData,
    values: np.ndarray,
) -> np.ndarray:
    positions = _assignment_school_positions(data, assignment)
    assigned = positions >= 0
    count = np.bincount(positions[assigned], minlength=len(data.loaded_schools)).astype(
        float
    )
    total = np.bincount(
        positions[assigned],
        weights=values[assigned],
        minlength=len(data.loaded_schools),
    )
    result = np.full(len(data.loaded_schools), np.nan, dtype=float)
    np.divide(total, count, out=result, where=count > 0)
    return result


def build_school_frl_cache(data: ModeData) -> None:
    for source in SES_SOURCES:
        for label in SUBCONFIGS:
            data.school_frl[(source, label)] = np.stack(
                [
                    _school_frl_means(data, assignment, data.frl[source])
                    for assignment in data.assignments[label]
                ]
            )


def school_ses_statistics(
    values: np.ndarray,
    baseline_values: np.ndarray,
    district_mean: float,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    if len(values) != ITERATION_COUNT or len(baseline_values) != ITERATION_COUNT:
        raise ValueError(f"school SES summaries require {ITERATION_COUNT} iterations")
    categories = np.asarray(
        [frl_tier(value, district_mean, missing="no_enrollment") for value in values],
        dtype=object,
    )
    mean_frl = finite_mean(values)
    baseline_mean = finite_mean(baseline_values)
    category = frl_tier(mean_frl, district_mean, missing="no_enrollment")
    baseline_category = frl_tier(baseline_mean, district_mean, missing="no_enrollment")
    paired = np.isfinite(values) & np.isfinite(baseline_values)
    mean_delta = finite_mean(values[paired] - baseline_values[paired])
    return {
        "district_frl_mean": float(district_mean),
        "mean_school_frl": mean_frl,
        "sd_school_frl": finite_sd(values),
        "min_school_frl": finite_min(values),
        "max_school_frl": finite_max(values),
        "school_frl_category": category,
        "share_iterations_high": float((categories == "high").mean()),
        "share_iterations_medium": float((categories == "medium").mean()),
        "share_iterations_low": float((categories == "low").mean()),
        "share_iterations_no_enrollment": float((categories == "no_enrollment").mean()),
        "percent_iterations_high": float((categories == "high").mean() * 100),
        "percent_iterations_medium": float((categories == "medium").mean() * 100),
        "percent_iterations_low": float((categories == "low").mean() * 100),
        "percent_iterations_no_enrollment": float(
            (categories == "no_enrollment").mean() * 100
        ),
        "baseline_mean_school_frl": baseline_mean,
        "baseline_category": baseline_category,
        "transition": f"{baseline_category}_to_{category}",
        "mean_policy_minus_baseline_frl": mean_delta,
        "paired_delta_iterations": int(paired.sum()),
    }


def build_school_ses_transitions(mode_data: Sequence[ModeData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for data in mode_data:
        if not data.school_frl:
            build_school_frl_cache(data)
        for source_order, source in enumerate(SES_SOURCES):
            baseline = data.school_frl[(source, BASELINE_POLICY)]
            for policy_order, label in enumerate(SUBCONFIGS):
                policy_values = data.school_frl[(source, label)]
                for school_position, school in data.loaded_schools.iterrows():
                    rows.append(
                        {
                            "mode": data.mode,
                            "mode_order": data.mode_order,
                            "ses_source": source,
                            "ses_source_order": source_order,
                            "policy": label,
                            "policy_order": policy_order,
                            "school_id": int(school["school_id"]),
                            "school_name": school["school_name"],
                            "school_category": school["category"],
                            **school_ses_statistics(
                                policy_values[:, school_position],
                                baseline[:, school_position],
                                data.district_frl[source],
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def travel_iteration_metrics(
    assignment: AssignmentData,
    raw_rank: np.ndarray,
    assigned_distance: np.ndarray,
) -> dict[str, int | float]:
    assigned = assignment.programno > 0
    valid = assigned & np.isfinite(assigned_distance)
    designated = assigned & (assignment.designation != 0)
    non_designated = assigned & ~designated
    valid_designated = valid & designated
    valid_non_designated = valid & non_designated
    long = valid & (assigned_distance > 3)
    long_designated = long & designated
    long_non_designated = long & non_designated
    saved_ge4 = long & (assignment.rank >= 4)
    raw_ge4 = long & (raw_rank >= 4)
    raw_missing = long & ((raw_rank <= 0) | ~np.isfinite(raw_rank))
    return {
        "assigned_count": int(assigned.sum()),
        "assigned_with_distance_count": int(valid.sum()),
        "designated_assigned_count": int(designated.sum()),
        "designated_with_distance_count": int(valid_designated.sum()),
        "non_designated_assigned_count": int(non_designated.sum()),
        "non_designated_with_distance_count": int(valid_non_designated.sum()),
        "mean_distance_all_assigned_miles": finite_mean(assigned_distance[valid]),
        "mean_distance_designated_miles": finite_mean(
            assigned_distance[valid_designated]
        ),
        "mean_distance_non_designated_miles": finite_mean(
            assigned_distance[valid_non_designated]
        ),
        "long_all_count": int(long.sum()),
        "long_all_rate": safe_rate(long.sum(), valid.sum()),
        "long_designated_count": int(long_designated.sum()),
        "long_designated_rate": safe_rate(
            long_designated.sum(), valid_designated.sum()
        ),
        "long_non_designated_count": int(long_non_designated.sum()),
        "long_non_designated_rate": safe_rate(
            long_non_designated.sum(), valid_non_designated.sum()
        ),
        "designated_share_long_travelers": safe_rate(long_designated.sum(), long.sum()),
        "non_designated_share_long_travelers": safe_rate(
            long_non_designated.sum(), long.sum()
        ),
        "long_saved_rank_ge4_count": int(saved_ge4.sum()),
        "long_saved_rank_ge4_rate": safe_rate(saved_ge4.sum(), valid.sum()),
        "long_saved_rank_ge4_non_designated_count": int(
            (saved_ge4 & non_designated).sum()
        ),
        "long_saved_rank_ge4_non_designated_rate": safe_rate(
            (saved_ge4 & non_designated).sum(), valid_non_designated.sum()
        ),
        "long_raw_rank_ge4_count": int(raw_ge4.sum()),
        "long_raw_rank_ge4_rate": safe_rate(raw_ge4.sum(), valid.sum()),
        "long_raw_rank_ge4_non_designated_count": int((raw_ge4 & non_designated).sum()),
        "long_raw_rank_ge4_non_designated_rate": safe_rate(
            (raw_ge4 & non_designated).sum(), valid_non_designated.sum()
        ),
        "long_raw_rank_missing_count": int(raw_missing.sum()),
        "long_raw_rank_missing_rate": safe_rate(raw_missing.sum(), valid.sum()),
        "long_raw_rank_missing_non_designated_count": int(
            (raw_missing & non_designated).sum()
        ),
        "long_raw_rank_missing_non_designated_rate": safe_rate(
            (raw_missing & non_designated).sum(), valid_non_designated.sum()
        ),
    }


def build_travel(mode_data: Sequence[ModeData]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for data in mode_data:
        for policy_order, label in enumerate(SUBCONFIGS):
            for iteration, assignment in enumerate(data.assignments[label]):
                raw_rank = assigned_raw_ranks(
                    data.raw_ranks[iteration], assignment.programno
                )
                assigned_distance = assigned_program_values(
                    data.distance_by_program, assignment.programno
                )
                rows.append(
                    {
                        "mode": data.mode,
                        "mode_order": data.mode_order,
                        "policy": label,
                        "policy_order": policy_order,
                        "iteration": iteration,
                        "applicant_count": int(data.applicant_masks[iteration].sum()),
                        **travel_iteration_metrics(
                            assignment, raw_rank, assigned_distance
                        ),
                    }
                )
    by_iteration = pd.DataFrame(rows)
    summary = macro_summary(
        by_iteration,
        ["mode", "mode_order", "policy", "policy_order"],
    )
    return by_iteration, summary


def macro_summary(frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    """Macro-average every numeric iteration metric and retain its iteration SD."""
    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {*key_columns, "iteration"}
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(key_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(key_columns, keys, strict=True))
        row["iterations"] = int(group["iteration"].nunique())
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            row[f"mean_{column}"] = finite_mean(values)
            row[f"sd_{column}"] = finite_sd(values)
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_change_counts(
    policy_values: np.ndarray,
    baseline_values: np.ndarray,
    eligible: np.ndarray,
    direction: str,
) -> dict[str, int | float]:
    """Classify exact per-student changes with the requested outcome direction."""
    policy_values = np.asarray(policy_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    eligible = (
        np.asarray(eligible, dtype=bool)
        & np.isfinite(policy_values)
        & np.isfinite(baseline_values)
    )
    if direction == "higher_is_better":
        wins = eligible & (policy_values > baseline_values)
        losses = eligible & (policy_values < baseline_values)
    elif direction == "lower_is_better":
        wins = eligible & (policy_values < baseline_values)
        losses = eligible & (policy_values > baseline_values)
    else:
        raise ValueError(f"unknown outcome direction: {direction}")
    ties = eligible & (policy_values == baseline_values)
    eligible_count = int(eligible.sum())
    stratum_count = len(eligible)
    return {
        "eligible_count": eligible_count,
        "eligible_rate": safe_rate(eligible_count, stratum_count),
        "win_count": int(wins.sum()),
        "win_rate": safe_rate(wins.sum(), eligible_count),
        "tie_count": int(ties.sum()),
        "tie_rate": safe_rate(ties.sum(), eligible_count),
        "loss_count": int(losses.sum()),
        "loss_rate": safe_rate(losses.sum(), eligible_count),
        "mean_delta": finite_mean(policy_values[eligible] - baseline_values[eligible]),
    }


def _strata(data: ModeData) -> list[Stratum]:
    student_count = len(data.students)
    strata = [Stratum("overall", "overall", np.arange(student_count))]
    school_names = data.schools.set_index("school_id")["school_name"].to_dict()
    attendance = pd.to_numeric(
        data.students["idschoolattendance"], errors="coerce"
    ).to_numpy(float)
    for school_id in sorted(set(attendance[np.isfinite(attendance)].astype(int))):
        if school_id not in school_names:
            raise ValueError(
                f"residential attendance area maps to unknown school {school_id}"
            )
        strata.append(
            Stratum(
                "residential_attendance_area",
                str(school_id),
                np.flatnonzero(attendance == school_id),
                attendance_school_id=school_id,
                attendance_school_name=str(school_names[school_id]),
            )
        )
    if (~np.isfinite(attendance)).any():
        strata.append(
            Stratum(
                "residential_attendance_area",
                "missing",
                np.flatnonzero(~np.isfinite(attendance)),
            )
        )

    ctip = pd.to_numeric(data.students["ctip1"], errors="coerce").fillna(0).to_numpy()
    strata.extend(
        [
            Stratum("ctip", "ctip", np.flatnonzero(ctip == 1)),
            Stratum("ctip", "non_ctip", np.flatnonzero(ctip != 1)),
        ]
    )
    for source in SES_SOURCES:
        tier_values = data.family_frl_tiers[source]
        for tier in ("low", "medium", "high", "missing"):
            indices = np.flatnonzero(tier_values == tier)
            if len(indices):
                strata.append(Stratum(f"{source}_family_frl_tier", tier, indices))
    return strata


def _student_school_values(
    data: ModeData,
    assignment: AssignmentData,
    school_values: np.ndarray,
) -> np.ndarray:
    positions = _assignment_school_positions(data, assignment)
    result = np.full(len(positions), np.nan, dtype=float)
    assigned = positions >= 0
    result[assigned] = school_values[positions[assigned]]
    return result


def _school_tier_values(school_frl: np.ndarray, district_mean: float) -> np.ndarray:
    mapping = {"low": 0.0, "medium": 1.0, "high": 2.0}
    return np.asarray(
        [
            mapping.get(
                frl_tier(float(value), district_mean, missing="no_enrollment"),
                np.nan,
            )
            for value in school_frl
        ],
        dtype=float,
    )


def _winner_outcomes(
    data: ModeData,
    label: str,
    iteration: int,
) -> list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]]:
    assignment = data.assignments[label][iteration]
    baseline = data.assignments[BASELINE_POLICY][iteration]
    raw_ranks = data.raw_ranks[iteration]
    applicant = data.applicant_masks[iteration]
    current_assigned = assignment.programno > 0
    baseline_assigned = baseline.programno > 0
    both_assigned = current_assigned & baseline_assigned
    current_raw_rank = assigned_raw_ranks(raw_ranks, assignment.programno)
    baseline_raw_rank = assigned_raw_ranks(raw_ranks, baseline.programno)
    current_distance = assigned_program_values(
        data.distance_by_program, assignment.programno
    )
    baseline_distance = assigned_program_values(
        data.distance_by_program, baseline.programno
    )
    outcomes = [
        (
            "assignment",
            "",
            "higher_is_better",
            current_assigned.astype(float),
            baseline_assigned.astype(float),
            np.ones(len(assignment.programno), dtype=bool),
        ),
        (
            "raw_top1",
            "",
            "higher_is_better",
            (current_raw_rank == 1).astype(float),
            (baseline_raw_rank == 1).astype(float),
            applicant,
        ),
        (
            "raw_top3",
            "",
            "higher_is_better",
            ((current_raw_rank >= 1) & (current_raw_rank <= 3)).astype(float),
            ((baseline_raw_rank >= 1) & (baseline_raw_rank <= 3)).astype(float),
            applicant,
        ),
        (
            "raw_choice_rank",
            "",
            "lower_is_better",
            current_raw_rank,
            baseline_raw_rank,
            both_assigned & (current_raw_rank > 0) & (baseline_raw_rank > 0),
        ),
        (
            "designation",
            "",
            "lower_is_better",
            assignment.designation.astype(float),
            baseline.designation.astype(float),
            both_assigned,
        ),
        (
            "distance",
            "",
            "lower_is_better",
            current_distance,
            baseline_distance,
            both_assigned,
        ),
        (
            "long_travel",
            "",
            "lower_is_better",
            (current_distance > 3).astype(float),
            (baseline_distance > 3).astype(float),
            both_assigned
            & np.isfinite(current_distance)
            & np.isfinite(baseline_distance),
        ),
    ]
    for source in SES_SOURCES:
        current_school_frl = data.school_frl[(source, label)][iteration]
        baseline_school_frl = data.school_frl[(source, BASELINE_POLICY)][iteration]
        current_exposure = _student_school_values(data, assignment, current_school_frl)
        baseline_exposure = _student_school_values(data, baseline, baseline_school_frl)
        outcomes.append(
            (
                "assigned_school_frl",
                source,
                "lower_is_better",
                current_exposure,
                baseline_exposure,
                both_assigned,
            )
        )
        current_tier = _student_school_values(
            data,
            assignment,
            _school_tier_values(current_school_frl, data.district_frl[source]),
        )
        baseline_tier = _student_school_values(
            data,
            baseline,
            _school_tier_values(baseline_school_frl, data.district_frl[source]),
        )
        outcomes.append(
            (
                "assigned_school_poverty_tier",
                source,
                "lower_is_better",
                current_tier,
                baseline_tier,
                both_assigned,
            )
        )
    return outcomes


def build_winners_losers(mode_data: Sequence[ModeData]) -> pd.DataFrame:
    outcome_order = {
        "assignment": 0,
        "raw_top1": 1,
        "raw_top3": 2,
        "raw_choice_rank": 3,
        "designation": 4,
        "distance": 5,
        "long_travel": 6,
        "assigned_school_frl": 7,
        "assigned_school_poverty_tier": 8,
    }
    accumulators: dict[tuple[Any, ...], WinnerStats] = {}
    for data in mode_data:
        if not data.school_frl:
            build_school_frl_cache(data)
        strata = _strata(data)
        for policy_order, label in enumerate(SUBCONFIGS):
            for iteration in range(ITERATION_COUNT):
                for (
                    outcome,
                    source,
                    direction,
                    policy_values,
                    baseline_values,
                    eligible,
                ) in _winner_outcomes(data, label, iteration):
                    for stratum in strata:
                        indices = stratum.indices
                        result = outcome_change_counts(
                            policy_values[indices],
                            baseline_values[indices],
                            eligible[indices],
                            direction,
                        )
                        key = (
                            data.mode,
                            data.mode_order,
                            label,
                            policy_order,
                            outcome,
                            outcome_order[outcome],
                            source,
                            direction,
                            stratum.stratum_type,
                            stratum.stratum_value,
                            stratum.attendance_school_id,
                            stratum.attendance_school_name,
                        )
                        accumulator = accumulators.setdefault(
                            key, WinnerStats(stratum_student_count=len(indices))
                        )
                        accumulator.update(result)

    key_columns = (
        "mode",
        "mode_order",
        "policy",
        "policy_order",
        "outcome",
        "outcome_order",
        "ses_source",
        "direction",
        "stratum_type",
        "stratum_value",
        "attendance_school_id",
        "attendance_school_name",
    )
    count_names = ("eligible", "win", "tie", "loss")
    rows: list[dict[str, Any]] = []
    for key, stats in accumulators.items():
        row = dict(zip(key_columns, key, strict=True))
        row["stratum_student_count"] = stats.stratum_student_count
        row["iterations"] = stats.iterations
        for index, name in enumerate(count_names):
            row[f"mean_{name}_count"] = stats.count_sums[index] / stats.iterations
            count = int(stats.rate_counts[index])
            row[f"mean_{name}_rate"] = (
                stats.rate_sums[index] / count if count else float("nan")
            )
            row[f"sd_{name}_rate"] = sample_sd_from_moments(
                count,
                float(stats.rate_sums[index]),
                float(stats.rate_squares[index]),
            )
        row["mean_policy_minus_baseline_delta"] = (
            stats.delta_sum / stats.delta_count if stats.delta_count else float("nan")
        )
        row["sd_policy_minus_baseline_delta"] = sample_sd_from_moments(
            stats.delta_count, stats.delta_sum, stats.delta_square
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["attendance_school_id"] = result["attendance_school_id"].astype("Int64")
    return result


def validate_output_frame(
    frame: pd.DataFrame,
    filename: str,
    key_columns: Sequence[str],
) -> None:
    if frame.empty:
        raise ValueError(f"output would be empty: {filename}")
    if not frame.columns.is_unique:
        raise ValueError(f"output has duplicate column names: {filename}")
    missing = set(key_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"output {filename} is missing key columns {sorted(missing)}")
    if frame.duplicated(list(key_columns)).any():
        raise ValueError(f"output has duplicate key rows: {filename}")


def default_output_dir(choice_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return choice_root / f"policy_outcomes_analysis_{timestamp}"


def write_outputs(
    output_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
) -> None:
    """Create a new output directory and write every artifact exclusively."""
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, frame in frames.items():
        path = output_dir / filename
        with path.open("x", encoding="utf-8", newline="") as output_file:
            frame.to_csv(output_file, index=False)
    metadata_path = output_dir / "methodology.json"
    with metadata_path.open("x", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def methodology_metadata(
    mode_data: Sequence[ModeData],
    updated_frl_path: Path,
    block_geometry_path: Path,
    output_dir: Path,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for data in mode_data:
        modes[data.mode] = {
            "matches_root": str(data.root),
            "generated_config": str(data.config_path),
            "student_data": str(resolve_data_path(data.config, "student-data")),
            "program_data": str(resolve_data_path(data.config, "program-data")),
            "school_data": str(resolve_data_path(data.config, "school-data")),
            "matching_population": len(data.students),
            "raw_applicants": int(data.applicant_masks[0].sum()),
            "assignments_per_policy": ITERATION_COUNT,
            "district_frl_means": {
                source: float(data.district_frl[source]) for source in SES_SOURCES
            },
            "updated_frl_fallback_students": data.fallback_count,
            "raw_unloaded_program_entries": data.raw_unloaded_program_entries,
            "choice_assigned_utilities_validated": data.utility_validation_count,
            "choice_utility_max_absolute_error": data.utility_max_abs_error,
        }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output_dir),
        "output_rows": {filename: len(frame) for filename, frame in frames.items()},
        "paths": {
            "updated_frl": str(updated_frl_path),
            "block_geometry_2020": str(block_geometry_path),
        },
        "policy_labels_in_order": list(SUBCONFIGS),
        "preference_modes_in_equal_order": list(MODE_ORDER),
        "modes": modes,
        "definitions": {
            "matching": "Saved assignments only; matching was not rerun.",
            "legacy_ses": "freelunch_prob + reducedlunch_prob",
            "updated_ses": (
                "2020 Census-block FRL Rate, with legacy SES fallback exactly as in "
                "analysis.recalculate_updated_frl_metrics"
            ),
            "raw_real_preferences": (
                "Exact school-program order from the first active application round, "
                "before eligibility, truncation, addition, filtering, or designation; "
                "programs removed from the matching market retain their original rank "
                "and top-school attribution."
            ),
            "raw_choice_preferences": (
                "Full noisy utility order per iteration, regenerated with seed 2023; "
                "one intervening np.random.rand(n,p) MTB draw was consumed each iteration."
            ),
            "fake_top_choice": (
                "Saved rank == 1 while the assigned exact school-program differs from "
                "raw rank 1; applies only to the two named _4 policies."
            ),
            "remove_non_aa_or_citywide_disallowed": (
                "Raw-top school is neither the student's attendance-area school nor "
                "a citywide school."
            ),
            "school_poverty_tiers": (
                "high when school mean FRL > mode/source district mean + 0.15; "
                "low when < mean - 0.15; medium otherwise; no_enrollment separately; "
                "share columns are 0-1 and percent columns are 0-100."
            ),
            "travel": (
                "Haversine miles with Earth radius 3958.8; long travel is strictly >3 "
                "miles; saved and raw rank thresholds are >=4; rates use assigned "
                "students with nonmissing distance in the named designation group."
            ),
            "winner_loser_directions": {
                "higher_is_better": ["assignment", "raw_top1", "raw_top3"],
                "lower_is_better": [
                    "raw_choice_rank",
                    "designation",
                    "distance",
                    "long_travel",
                    "assigned_school_frl",
                    "assigned_school_poverty_tier",
                ],
            },
            "winner_loser_eligibility": (
                "Assignment uses the full matching cohort; raw top outcomes use raw "
                "applicants; rank requires both assigned programs to be raw-listed; "
                "designation, distance, and school exposure require assignment in both "
                "policy and baseline, plus nonmissing values where applicable."
            ),
            "winner_loser_delta": "Policy value minus status_quo value.",
            "standard_deviation": "Sample SD across 25 iteration-level values (ddof=1).",
            "privacy": "No student-level records or student identifiers are exported.",
        },
    }


def analyze(
    choice_root: Path,
    real_root: Path,
    updated_frl_path: Path,
    block_geometry_path: Path,
    output_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    lookup = load_frl_lookup(updated_frl_path)
    block_geometry = load_2020_block_geometry(block_geometry_path)
    data = [
        load_mode_data("choice_model", choice_root, lookup, block_geometry),
        load_mode_data("real_preferences", real_root, lookup, block_geometry),
    ]
    for mode in data:
        build_school_frl_cache(mode)

    fake_iteration, fake_summary = build_fake_top_choice(data)
    travel_iteration, travel_summary = build_travel(data)
    frames = {
        "fake_top_choice_by_iteration.csv": fake_iteration,
        "fake_top_choice_summary.csv": fake_summary,
        "school_enrollment.csv": build_school_enrollment(data),
        "school_ses_transitions.csv": build_school_ses_transitions(data),
        "travel_by_iteration.csv": travel_iteration,
        "travel_summary.csv": travel_summary,
        "winners_losers.csv": build_winners_losers(data),
    }
    key_columns = {
        "fake_top_choice_by_iteration.csv": (
            "mode",
            "policy",
            "iteration",
        ),
        "fake_top_choice_summary.csv": ("mode", "policy"),
        "school_enrollment.csv": ("mode", "policy", "scope", "school_id"),
        "school_ses_transitions.csv": (
            "mode",
            "ses_source",
            "policy",
            "school_id",
        ),
        "travel_by_iteration.csv": ("mode", "policy", "iteration"),
        "travel_summary.csv": ("mode", "policy"),
        "winners_losers.csv": (
            "mode",
            "policy",
            "outcome",
            "ses_source",
            "stratum_type",
            "stratum_value",
        ),
    }
    for filename, frame in frames.items():
        validate_output_frame(frame, filename, key_columns[filename])
    metadata = methodology_metadata(
        data,
        updated_frl_path,
        block_geometry_path,
        output_dir,
        frames,
    )
    return frames, metadata


def _checked_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    choice_root = args.choice_root.expanduser().resolve()
    real_root = args.real_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(choice_root)
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)

    updated_frl_path = _checked_file(args.updated_frl)
    block_geometry_path = _checked_file(args.block_geometry)
    frames, metadata = analyze(
        choice_root,
        real_root,
        updated_frl_path,
        block_geometry_path,
        output_dir,
    )
    write_outputs(output_dir, frames, metadata)
    for filename in (*OUTPUT_FILENAMES, "methodology.json"):
        LOGGER.info("Wrote %s", output_dir / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

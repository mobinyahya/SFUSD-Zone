"""Shared CSV loading and normalization for optimization and assignment."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from loaders.config import DataScenario, ResolvedSource

_SCHOOL_ROUND = re.compile(r"r(\d+)_ranked_idschool")
_PROGRAM_ROUND = re.compile(r"r(\d+)_programs")
_SCALAR_SCHOOL_COLUMNS = {
    "school_id",
    "idschoolattendance",
    "attendance_area",
    "enrolled_idschool",
    "final_school",
    "msf",
}
_SCHOOL_LIST_COLUMNS = ("sibling", "aaprek", "prek")
_ROUND_ALIGNED_LIST_SUFFIXES = ("listed_ranks", "cohortstring", "randomnumber")
_ROUND_PREFERENCE_COLUMN = re.compile(
    r"r(\d+)_(ranked_idschool|programs|listed_ranks|cohortstring|randomnumber|"
    r"designation_randomnumber)"
)
_MISSION_BAY_SCHOOL_IDS = {909, 999}
SPECIAL_PROGRAMS = frozenset({"AF", "DA", "DT", "ED", "MM", "MS", "SA", "TC", "AO"})
_FRL_COUNT_COLUMNS = ("Not FRL", "FRLunch", "Students")


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool | np.bool_) else False


def normalize_grade(value: Any) -> str:
    """Normalize grade labels while retaining standard KG/TK/PK labels."""
    if _missing(value):
        return ""
    text = str(value).strip().upper()
    if text in {"K", "KG", "KINDERGARTEN"}:
        return "KG"
    if text in {"PRE-K", "PREK", "PK"}:
        return "PK"
    if text in {"TRANSITIONAL KINDERGARTEN", "TK"}:
        return "TK"
    try:
        number = float(text)
    except ValueError:
        return text
    if np.isfinite(number) and number.is_integer():
        return str(int(number)).zfill(2)
    return text


def _literal_list(value: Any, label: str) -> list[Any]:
    if isinstance(value, list | tuple | np.ndarray):
        return list(value)
    if _missing(value) or (isinstance(value, str) and not value.strip()):
        return []
    if not isinstance(value, str):
        raise ValueError(f"Expected {label} to contain a list, got {value!r}.")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not safely parse {label} list {value!r}.") from exc
    if not isinstance(parsed, list | tuple):
        raise ValueError(f"Expected {label} to contain a list, got {value!r}.")
    return list(parsed)


def _school_identity(value: Any) -> int:
    if isinstance(value, bool) or _missing(value):
        raise ValueError(f"Invalid school ID {value!r}.")
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid school ID {value!r}.") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid school ID {value!r}.")
    return int(number)


def parse_ranked_schools(value: Any) -> list[int]:
    """Safely parse one ranked-school cell into integer school IDs."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if not (text.startswith("[") and text.endswith("]")):
            raise ValueError(f"Could not safely parse ranked-school list {value!r}.")
        items: list[Any] = text[1:-1].split(",")
    else:
        items = _literal_list(value, "ranked-school")

    schools = []
    for item in items:
        if isinstance(item, str):
            item = item.strip().strip("'\"")
            if not item:
                continue
        schools.append(_school_identity(item))
    return schools


def parse_ranked_programs(value: Any) -> list[str]:
    """Safely parse one ranked-program cell into non-empty program codes."""
    programs = []
    for item in _literal_list(value, "ranked-program"):
        if _missing(item):
            raise ValueError("Ranked-program lists cannot contain null values.")
        program = str(item).strip()
        if program:
            programs.append(program)
    return programs


def read_csv_source(
    source: ResolvedSource,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read one resolved CSV source."""
    if not isinstance(source, ResolvedSource):
        raise TypeError("source must be a ResolvedSource.")
    return pd.read_csv(source.path, **read_csv_kwargs)


def read_csv(
    scenario: DataScenario,
    role: str,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read and concatenate every CSV configured for a source role."""
    frames = [
        read_csv_source(source, **read_csv_kwargs) for source in scenario.sources(role)
    ]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True, sort=False)


def _filter_group(role: str, filter_group: str | None) -> str:
    if filter_group is not None:
        if filter_group not in {"optimization", "assignment"}:
            raise ValueError(f"Unknown filter group {filter_group!r}.")
        return filter_group
    prefix = role.split(".", 1)[0]
    if prefix not in {"optimization", "assignment"}:
        raise ValueError(
            f"Cannot infer filters from role {role!r}; pass filter_group explicitly."
        )
    return prefix


def school_id_aliases(scenario: DataScenario, group: str) -> dict[int, int]:
    """Return centrally derived school-ID aliases for one filter group."""
    return {909: 999} if scenario.filter(group, "include_mission_bay") else {}


def _map_scalar_school(value: Any, aliases: Mapping[int, int]) -> Any:
    if _missing(value) or (isinstance(value, str) and not value.strip()):
        return value
    try:
        key = _school_identity(value)
    except ValueError:
        return value
    return aliases.get(key, value)


def _is_scalar_school_column(column: str) -> bool:
    return column in _SCALAR_SCHOOL_COLUMNS or bool(
        re.fullmatch(r"r\d+_idschool", column)
    )


def _selected_rounds(configured: Any, discovered: set[int]) -> list[int]:
    if configured == "all":
        return sorted(discovered)
    requested = sorted(configured)
    missing = sorted(set(requested) - discovered)
    if missing:
        raise ValueError(f"Configured preference rounds are absent: {missing}.")
    return requested


def _set_source_attrs(
    frame: pd.DataFrame, source_rows: np.ndarray, source_row_count: int
) -> None:
    frame.attrs["source_rows"] = source_rows.tolist()
    frame.attrs["source_row_count"] = source_row_count


def _filter_student_rows(
    frame: pd.DataFrame,
    source_rows: np.ndarray,
    keep: pd.Series | np.ndarray,
    source_row_count: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = np.asarray(keep, dtype=bool)
    source_rows = source_rows[mask]
    frame = frame.loc[mask].copy()
    _set_source_attrs(frame, source_rows, source_row_count)
    return frame, source_rows


def filter_outside_district_students(
    frame: pd.DataFrame,
    scenario: DataScenario,
    group: str,
) -> pd.DataFrame:
    """Apply the configured policy to students without a district Census Block."""
    if scenario.filter(group, "outside_district_students") == "include":
        return frame
    if "census_block" not in frame.columns:
        return frame

    source_row_count = int(frame.attrs.get("source_row_count", len(frame)))
    source_rows = np.asarray(
        frame.attrs.get("source_rows", np.arange(len(frame))), dtype=int
    )
    if len(source_rows) != len(frame):
        raise ValueError("Student source_rows metadata must align with the table rows.")
    filtered, source_rows = _filter_student_rows(
        frame,
        source_rows,
        frame["census_block"].notna(),
        source_row_count,
    )
    filtered = filtered.reset_index(drop=True)
    _set_source_attrs(filtered, source_rows, source_row_count)
    return filtered


def _validate_student_identities(frame: pd.DataFrame) -> None:
    if "studentno" not in frame.columns:
        raise ValueError("Student data is missing required column 'studentno'.")
    values = frame["studentno"]
    missing = values.isna() | values.astype("string").str.strip().eq("")
    if missing.fillna(True).any():
        raise ValueError("Student data contains a missing studentno identity.")
    duplicates = values[values.duplicated(keep=False)]
    if not duplicates.empty:
        duplicate_values = duplicates.astype(str).unique().tolist()
        raise ValueError(
            "Student data contains duplicate studentno identities: "
            f"{duplicate_values[:10]}."
        )


def load_student_records(
    scenario: DataScenario,
    role: str = "optimization.students",
    *,
    filter_group: str | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load students and normalize selectors, preferences, and school IDs."""
    group = _filter_group(role, filter_group)
    frame = read_csv(scenario, role, **read_csv_kwargs)
    frame = normalize_student_records(frame, scenario, group)
    from loaders.geography import normalize_census_geography

    frame = normalize_census_geography(
        frame,
        scenario,
        group,
        source_vintage=_role_geography_vintage(scenario, role),
        style="student",
    )
    frame = apply_student_frl_estimate(frame, scenario, group)
    return filter_outside_district_students(frame, scenario, group)


def _frl_count_rates(source: ResolvedSource) -> pd.Series:
    frame = read_csv_source(source, dtype={"BlockID": "string"})
    required = {"BlockID", *_FRL_COUNT_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Student FRL estimate is missing columns {sorted(missing)}: {source.path}."
        )

    block_ids = frame["BlockID"].astype("string").str.strip()
    block_ids = block_ids.str.replace(r"\.0$", "", regex=True)
    invalid_ids = block_ids.isna() | ~block_ids.str.fullmatch(r"\d+").fillna(False)
    if invalid_ids.any():
        examples = frame.loc[invalid_ids, "BlockID"].head(5).tolist()
        raise ValueError(f"Student FRL estimate contains invalid BlockIDs: {examples}.")
    numeric_ids = pd.to_numeric(block_ids, errors="raise").astype("int64")
    if numeric_ids.duplicated().any():
        duplicates = numeric_ids[numeric_ids.duplicated(False)].unique()[:5].tolist()
        raise ValueError(
            f"Student FRL estimate contains duplicate BlockIDs: {duplicates}."
        )

    counts = frame.loc[:, list(_FRL_COUNT_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_counts = counts.isna() | ~np.isfinite(counts) | counts.lt(0)
    fractional_counts = counts.mod(1).ne(0)
    if invalid_counts.any().any() or fractional_counts.any().any():
        invalid_rows = (invalid_counts | fractional_counts).any(axis=1)
        examples = frame.loc[invalid_rows, ["BlockID", *_FRL_COUNT_COLUMNS]].head(5)
        raise ValueError(
            "Student FRL estimate counts must be non-negative integers; invalid "
            f"rows include {examples.to_dict('records')}."
        )
    inconsistent = counts["Not FRL"] + counts["FRLunch"] != counts["Students"]
    if inconsistent.any():
        examples = frame.loc[inconsistent, ["BlockID", *_FRL_COUNT_COLUMNS]].head(5)
        raise ValueError(
            "Student FRL estimate Students must equal Not FRL + FRLunch; rows "
            f"include {examples.to_dict('records')}."
        )

    rates = counts["FRLunch"].div(counts["Students"].replace(0, np.nan))
    return pd.Series(rates.to_numpy(), index=numeric_ids.to_numpy(), dtype=float)


def apply_student_frl_estimate(
    frame: pd.DataFrame, scenario: DataScenario, group: str
) -> pd.DataFrame:
    """Apply the selected block-count FRL estimate with source-data fallback."""
    estimate = scenario.filter(group, "frl_estimate")
    if estimate is None:
        return frame
    if "census_block" not in frame.columns:
        raise ValueError("Student data is missing required column 'census_block'.")

    role = f"{group}.frl_estimate"
    rates = _frl_count_rates(scenario.source(role))
    block_ids = pd.to_numeric(frame["census_block"], errors="coerce").astype("Int64")
    estimated = block_ids.map(rates)

    result = frame.copy()
    if group == "optimization":
        if "FRL Score" not in result.columns:
            raise ValueError("Student data is missing required column 'FRL Score'.")
        legacy = pd.to_numeric(result["FRL Score"], errors="coerce")
        result["FRL Score"] = estimated.fillna(legacy)
    elif group == "assignment":
        required = {"freelunch_prob", "reducedlunch_prob"}
        missing = required - set(result.columns)
        if missing:
            raise ValueError(f"Student data is missing columns {sorted(missing)}.")
        legacy = pd.to_numeric(result["freelunch_prob"], errors="coerce").fillna(
            0
        ) + pd.to_numeric(result["reducedlunch_prob"], errors="coerce").fillna(0)
        effective = estimated.fillna(legacy)
        result["freelunch_prob"] = effective
        result["reducedlunch_prob"] = 0.0
        if "FRL Score" in result.columns:
            legacy_score = pd.to_numeric(result["FRL Score"], errors="coerce")
            result["FRL Score"] = estimated.fillna(legacy_score)
        else:
            result["FRL Score"] = effective
    else:  # pragma: no cover - validated by DataScenario.filter
        raise ValueError(f"Unknown filter group {group!r}.")
    return result


def normalize_student_records(
    frame: pd.DataFrame,
    scenario: DataScenario,
    group: str,
) -> pd.DataFrame:
    """Normalize an already-loaded student table using scenario filters."""
    if group not in {"optimization", "assignment"}:
        raise ValueError(f"Unknown filter group {group!r}.")

    source_row_count = int(frame.attrs.get("source_row_count", len(frame)))
    source_rows = np.asarray(
        frame.attrs.get("source_rows", np.arange(len(frame))), dtype=int
    )
    if len(source_rows) != len(frame):
        raise ValueError("Student source_rows metadata must align with the table rows.")
    if "grade" not in frame.columns:
        raise ValueError("Student data is missing required column 'grade'.")
    grades = tuple(scenario.filter(group, "grades"))
    normalized_grades = frame["grade"].map(normalize_grade)
    grade_mask = normalized_grades.isin(grades)
    frame, source_rows = _filter_student_rows(
        frame, source_rows, grade_mask, source_row_count
    )
    frame["grade"] = normalized_grades.loc[frame.index]
    _validate_student_identities(frame)

    school_rounds = {
        int(match.group(1))
        for column in frame.columns
        if (match := _SCHOOL_ROUND.fullmatch(str(column)))
    }
    program_rounds = {
        int(match.group(1))
        for column in frame.columns
        if (match := _PROGRAM_ROUND.fullmatch(str(column)))
    }
    if school_rounds != program_rounds:
        raise ValueError(
            "Ranked school/program columns must occur in pairs; "
            f"missing school rounds={sorted(program_rounds - school_rounds)}, "
            f"missing program rounds={sorted(school_rounds - program_rounds)}."
        )

    rounds = _selected_rounds(scenario.filter(group, "rounds"), school_rounds)
    if not rounds:
        raise ValueError("Student data contains no selected preference rounds.")
    unselected = school_rounds - set(rounds)
    frame.drop(
        columns=[
            column
            for column in frame.columns
            if (match := _ROUND_PREFERENCE_COLUMN.fullmatch(str(column)))
            and int(match.group(1)) in unselected
        ],
        inplace=True,
    )

    aliases = school_id_aliases(scenario, group)
    include_mission_bay = scenario.filter(group, "include_mission_bay")
    special_mode = scenario.filter(group, "special_programs")
    any_special = pd.Series(False, index=frame.index)
    for round_number in rounds:
        school_column = f"r{round_number}_ranked_idschool"
        program_column = f"r{round_number}_programs"
        parsed_schools: list[list[int]] = []
        parsed_programs: list[list[str]] = []
        aligned_columns = [
            f"r{round_number}_{suffix}"
            for suffix in _ROUND_ALIGNED_LIST_SUFFIXES
            if f"r{round_number}_{suffix}" in frame.columns
        ]
        parsed_aligned: dict[str, list[list[Any]]] = {
            column: [] for column in aligned_columns
        }
        for index, row in frame.iterrows():
            identity = row.get("studentno", index)
            try:
                schools = parse_ranked_schools(row[school_column])
                programs = parse_ranked_programs(row[program_column])
                aligned = {
                    column: _literal_list(row[column], column)
                    for column in aligned_columns
                }
            except ValueError as exc:
                raise ValueError(
                    f"Invalid round {round_number} preferences for student "
                    f"{identity}: {exc}"
                ) from exc
            if len(schools) != len(programs):
                raise ValueError(
                    f"Student {identity} round {round_number} has "
                    f"{len(schools)} ranked schools but {len(programs)} "
                    "ranked programs."
                )
            for column, values in aligned.items():
                if values and len(values) != len(schools):
                    raise ValueError(
                        f"Student {identity} round {round_number} has "
                        f"{len(schools)} ranked schools but {len(values)} "
                        f"values in {column}."
                    )
            special_positions = {
                position
                for position, program in enumerate(programs)
                if program in SPECIAL_PROGRAMS
            }
            if special_positions:
                any_special.at[index] = True
            keep_positions = list(range(len(schools)))
            if not include_mission_bay:
                keep_positions = [
                    position
                    for position in keep_positions
                    if schools[position] not in _MISSION_BAY_SCHOOL_IDS
                ]
            if special_mode == "exclude_only_special":
                keep_positions = [
                    position
                    for position in keep_positions
                    if position not in special_positions
                ]
            if len(keep_positions) != len(schools):
                schools = [schools[position] for position in keep_positions]
                programs = [programs[position] for position in keep_positions]
                aligned = {
                    column: (
                        [values[position] for position in keep_positions]
                        if values
                        else []
                    )
                    for column, values in aligned.items()
                }
            if include_mission_bay:
                schools = [aliases.get(school, school) for school in schools]
            parsed_schools.append(schools)
            parsed_programs.append(programs)
            for column, values in aligned.items():
                parsed_aligned[column].append(values)
        frame[school_column] = pd.Series(
            parsed_schools, index=frame.index, dtype=object
        )
        frame[program_column] = pd.Series(
            parsed_programs, index=frame.index, dtype=object
        )
        for column, values in parsed_aligned.items():
            frame[column] = pd.Series(values, index=frame.index, dtype=object)

    if special_mode == "exclude_any_special":
        frame, source_rows = _filter_student_rows(
            frame, source_rows, ~any_special, source_row_count
        )

    for column in _SCHOOL_LIST_COLUMNS:
        if column not in frame.columns:
            continue
        parsed_values: list[list[int]] = []
        for index, value in frame[column].items():
            identity = frame.at[index, "studentno"]
            try:
                schools = parse_ranked_schools(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid {column} schools for student {identity}: {exc}"
                ) from exc
            if not include_mission_bay:
                schools = [
                    school
                    for school in schools
                    if school not in _MISSION_BAY_SCHOOL_IDS
                ]
            parsed_values.append(
                [aliases.get(school, school) for school in schools]
                if include_mission_bay
                else schools
            )
        frame[column] = pd.Series(parsed_values, index=frame.index, dtype=object)

    if "currentlpsibling" in frame.columns:
        parsed_program_values: list[list[str]] = []
        for index, value in frame["currentlpsibling"].items():
            identity = frame.at[index, "studentno"]
            try:
                programs = parse_ranked_programs(value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid currentlpsibling programs for student {identity}: {exc}"
                ) from exc
            if not include_mission_bay:
                programs = [
                    program
                    for program in programs
                    if _program_school_identity(program) not in _MISSION_BAY_SCHOOL_IDS
                ]
            parsed_program_values.append(
                [
                    str(_program_id_alias(program, aliases))
                    if include_mission_bay
                    else program
                    for program in programs
                ]
            )
        frame["currentlpsibling"] = pd.Series(
            parsed_program_values, index=frame.index, dtype=object
        )

    for column in frame.columns:
        if _is_scalar_school_column(str(column)):
            if not include_mission_bay:
                frame[column] = frame[column].map(
                    lambda value: (
                        pd.NA
                        if not _missing(value)
                        and _safe_school_identity(value) in _MISSION_BAY_SCHOOL_IDS
                        else value
                    )
                )
            if include_mission_bay:
                frame[column] = frame[column].map(
                    lambda value: _map_scalar_school(value, aliases)
                )

    participating = pd.Series(False, index=frame.index)
    for round_number in rounds:
        participating |= frame[f"r{round_number}_ranked_idschool"].map(bool)
    frame, source_rows = _filter_student_rows(
        frame, source_rows, participating, source_row_count
    )

    first_rounds: list[int] = []
    first_ordinals: list[int] = []
    for _, row in frame.iterrows():
        ordinal = next(
            index
            for index, round_number in enumerate(rounds)
            if row[f"r{round_number}_ranked_idschool"]
        )
        first_rounds.append(rounds[ordinal])
        first_ordinals.append(ordinal)
    frame["first_participating_round"] = first_rounds
    frame["first_participating_round_ordinal"] = first_ordinals

    selected_suffixes = {
        "ranked_idschool": "selected_ranked_idschool",
        "programs": "selected_programs",
        "listed_ranks": "selected_listed_ranks",
        "randomnumber": "selected_randomnumber",
        "cohortstring": "selected_cohortstring",
    }
    for suffix, selected_column in selected_suffixes.items():
        if suffix not in {"ranked_idschool", "programs"} and not any(
            f"r{round_number}_{suffix}" in frame.columns for round_number in rounds
        ):
            continue
        frame[selected_column] = pd.Series(
            [
                row.get(f"r{round_number}_{suffix}", [])
                for (_, row), round_number in zip(frame.iterrows(), first_rounds)
            ],
            index=frame.index,
            dtype=object,
        )
    if any(
        f"r{round_number}_designation_randomnumber" in frame.columns
        for round_number in rounds
    ):
        frame["selected_designation_randomnumber"] = [
            row.get(f"r{round_number}_designation_randomnumber", pd.NA)
            for (_, row), round_number in zip(frame.iterrows(), first_rounds)
        ]

    frame = frame.reset_index(drop=True)
    _set_source_attrs(frame, source_rows, source_row_count)
    return frame


def _safe_school_identity(value: Any) -> int | None:
    try:
        return _school_identity(value)
    except ValueError:
        return None


def _program_id_alias(value: Any, aliases: Mapping[int, int]) -> Any:
    if _missing(value):
        return value
    text = str(value)
    prefix, separator, suffix = text.partition("-")
    if not separator:
        return value
    try:
        school = _school_identity(prefix)
    except ValueError:
        return value
    mapped = aliases.get(school, school)
    return f"{mapped}-{suffix}"


def _program_school_identity(value: Any) -> int | None:
    if _missing(value):
        return None
    prefix, separator, _ = str(value).partition("-")
    if not separator:
        return None
    return _safe_school_identity(prefix)


def _drop_alias_duplicates(
    frame: pd.DataFrame,
    original_ids: pd.Series,
) -> pd.DataFrame:
    """Prefer canonical source IDs, then drop exact alias collisions."""
    if "school_id" not in frame.columns or not frame["school_id"].duplicated().any():
        return frame
    drop: list[Any] = []
    for _, positions in frame.groupby(
        "school_id", sort=False, dropna=False
    ).groups.items():
        group_positions = list(positions)
        if len(group_positions) < 2:
            continue
        mapped_id = _safe_school_identity(frame.loc[group_positions[0], "school_id"])
        canonical_positions = [
            position
            for position in group_positions
            if _safe_school_identity(original_ids.loc[position]) == mapped_id
        ]
        if canonical_positions and len(canonical_positions) < len(group_positions):
            drop.extend(
                position
                for position in group_positions
                if position not in canonical_positions
            )
            group_positions = canonical_positions
        if len(group_positions) < 2:
            continue
        if original_ids.loc[group_positions].nunique(dropna=False) < 2:
            continue
        group = frame.loc[group_positions]
        comparison_columns = [
            column for column in group.columns if not str(column).startswith("Unnamed:")
        ]
        duplicates = group.duplicated(subset=comparison_columns, keep="first")
        drop.extend(group.index[duplicates])
    return frame.drop(index=drop)


def _load_school_keyed_records(
    scenario: DataScenario,
    role: str,
    group: str,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    frame = read_csv(scenario, role, **read_csv_kwargs)
    return normalize_school_records(frame, scenario, group)


def _role_geography_vintage(scenario: DataScenario, role: str) -> str | None:
    vintages = {
        source.geography_vintage
        for source in scenario.sources(role)
        if source.geography_vintage is not None
    }
    if len(vintages) > 1:
        raise ValueError(
            f"Source role {role!r} mixes Census geography vintages: "
            f"{sorted(vintages)}."
        )
    return next(iter(vintages), None)


def normalize_school_records(
    frame: pd.DataFrame,
    scenario: DataScenario,
    group: str,
) -> pd.DataFrame:
    """Normalize school-keyed rows already loaded by a shared table reader."""
    if group not in {"optimization", "assignment"}:
        raise ValueError(f"Unknown filter group {group!r}.")
    frame = frame.copy()
    if "school_id" not in frame.columns:
        raise ValueError("School-keyed data is missing required column 'school_id'.")
    original_ids = frame["school_id"].copy()
    include_mission_bay = scenario.filter(group, "include_mission_bay")
    if not include_mission_bay:
        keep = ~frame["school_id"].map(_safe_school_identity).isin(
            _MISSION_BAY_SCHOOL_IDS
        )
        frame = frame.loc[keep].copy()
        original_ids = original_ids.loc[frame.index]

    aliases = school_id_aliases(scenario, group)
    if include_mission_bay:
        frame["school_id"] = frame["school_id"].map(
            lambda value: _map_scalar_school(value, aliases)
        )
        for column in frame.columns:
            if column != "school_id" and _is_scalar_school_column(str(column)):
                frame[column] = frame[column].map(
                    lambda value: _map_scalar_school(value, aliases)
                )
        if "program_id" in frame.columns:
            frame["program_id"] = frame["program_id"].map(
                lambda value: _program_id_alias(value, aliases)
            )
    frame = _drop_alias_duplicates(frame, original_ids)
    return frame.reset_index(drop=True)


def _program_grades(frame: pd.DataFrame, label: str) -> pd.Series:
    if "grade" in frame.columns:
        return frame["grade"].map(normalize_grade)
    elif "program_id" in frame.columns:
        return frame["program_id"].map(
            lambda value: (
                normalize_grade(str(value).rsplit("-", 1)[-1])
                if not _missing(value) and "-" in str(value)
                else ""
            )
        )
    else:
        raise ValueError(
            f"{label} must contain 'grade' or grade-suffixed 'program_id' values."
        )


def _filter_program_records(
    frame: pd.DataFrame, scenario: DataScenario, group: str, label: str
) -> pd.DataFrame:
    grades = tuple(scenario.filter(group, "grades"))
    program_grades = _program_grades(frame, label)
    available = set(program_grades) - {""}
    missing = sorted(set(grades) - available)
    if missing:
        raise ValueError(
            f"{label} does not contain requested grades {missing}; available grades "
            f"are {sorted(available)}."
        )
    frame = frame.loc[program_grades.isin(grades)].copy()
    if "grade" in frame.columns:
        frame["grade"] = program_grades.loc[frame.index]

    if scenario.filter(group, "special_programs") != "include":
        if "program_type" not in frame.columns:
            raise ValueError(
                f"{label} is missing required column 'program_type' for special-"
                "program exclusion."
            )
        frame = frame.loc[~frame["program_type"].astype(str).isin(SPECIAL_PROGRAMS)]
    if frame.empty:
        raise ValueError(f"{label} contains no programs after selector filtering.")
    return frame.reset_index(drop=True)


def apply_capacity_scenario(
    frame: pd.DataFrame,
    scenario: DataScenario,
    group: str,
) -> pd.DataFrame:
    """Overlay an explicit capacity scenario on selected program records."""
    scenario_name = scenario.filter(group, "capacity_scenario")
    result = frame.copy()
    if scenario_name == "programs":
        return result

    required_program_columns = {"school_id", "program_type", "capacity"}
    missing = required_program_columns - set(result.columns)
    if missing:
        raise ValueError(
            f"Selected program data is missing capacity columns: {sorted(missing)}."
        )

    capacity_column = f"Scenario_{scenario_name}_Capacity"
    overrides = read_csv(scenario, f"{group}.capacity", low_memory=False)
    overrides = overrides.rename(
        columns={
            "SchNum": "school_id",
            "PathwayCode": "program_type",
            capacity_column: "capacity",
        }
    )
    required_override_columns = {"school_id", "program_type", "capacity"}
    missing = required_override_columns - set(overrides.columns)
    if missing:
        raise ValueError(
            f"Capacity scenario {scenario_name!r} is missing columns: "
            f"{sorted(missing)}."
        )

    keys = ["school_id", "program_type"]
    if "Grade" in overrides.columns:
        overrides["_capacity_grade"] = overrides["Grade"].map(normalize_grade)
        selected_grades = set(scenario.filter(group, "grades"))
        overrides = overrides.loc[
            overrides["_capacity_grade"].isin(selected_grades)
        ].copy()
        if overrides.empty:
            raise ValueError(
                f"Capacity scenario {scenario_name!r} has no rows for selected "
                f"grades {sorted(selected_grades)}."
            )
        result["_capacity_grade"] = _program_grades(
            result, "Selected program data"
        )
        keys.append("_capacity_grade")

    overrides["_source_school_id"] = pd.to_numeric(
        overrides["school_id"], errors="coerce"
    )
    overrides = normalize_school_records(overrides, scenario, group)
    overrides["program_type"] = overrides["program_type"].astype(str)
    result["program_type"] = result["program_type"].astype(str)
    overrides["capacity"] = pd.to_numeric(overrides["capacity"], errors="coerce")
    invalid_capacity = (
        overrides["capacity"].isna()
        | ~np.isfinite(overrides["capacity"])
        | (overrides["capacity"] < 0)
    )
    if invalid_capacity.any():
        examples = overrides.loc[invalid_capacity, keys].head(10).to_dict("records")
        raise ValueError(
            f"Capacity scenario {scenario_name!r} has invalid capacities for "
            f"{examples}."
        )
    if overrides[keys].isna().any().any():
        raise ValueError(
            f"Capacity scenario {scenario_name!r} contains blank matching keys."
        )

    resolved_rows = []
    for _, candidates in overrides.groupby(keys, sort=False, dropna=False):
        if len(candidates) == 1:
            resolved_rows.append(candidates.iloc[0])
            continue
        canonical = candidates.loc[
            pd.to_numeric(candidates["school_id"], errors="coerce").eq(
                candidates["_source_school_id"]
            )
        ]
        if len(canonical) == 1:
            resolved_rows.append(canonical.iloc[0])
        elif candidates["capacity"].nunique() == 1:
            resolved_rows.append(candidates.iloc[0])
        else:
            duplicate_keys = candidates[keys].iloc[0].to_dict()
            raise ValueError(
                f"Capacity scenario {scenario_name!r} has conflicting rows for "
                f"{duplicate_keys}."
            )
    overrides = pd.DataFrame(resolved_rows)
    capacity_by_key = {
        tuple(row[key] for key in keys): float(row["capacity"])
        for _, row in overrides.iterrows()
    }
    matched_capacity = pd.Series(
        [
            capacity_by_key.get(tuple(row[key] for key in keys), np.nan)
            for _, row in result.iterrows()
        ],
        index=result.index,
        dtype=float,
    )
    if not result.empty and matched_capacity.notna().sum() == 0:
        raise ValueError(
            f"Capacity scenario {scenario_name!r} has no rows matching the "
            "selected programs."
        )
    result["capacity"] = matched_capacity.combine_first(
        pd.to_numeric(result["capacity"], errors="coerce")
    )
    return result.drop(columns=["_capacity_grade"], errors="ignore")


def load_program_records(
    scenario: DataScenario,
    role: str = "assignment.programs",
    *,
    filter_group: str | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load programs selected by grade, special mode, and Mission Bay policy."""
    group = _filter_group(role, filter_group)
    frame = _load_school_keyed_records(scenario, role, group, **read_csv_kwargs)
    frame = _filter_program_records(frame, scenario, group, "Program capacity data")
    catalog_role = f"{role}.catalog"
    try:
        scenario.resolved(catalog_role)
    except KeyError:
        selected = frame
    else:
        catalog = _load_school_keyed_records(
            scenario, catalog_role, group, **read_csv_kwargs
        )
        catalog = _filter_program_records(
            frame=catalog,
            scenario=scenario,
            group=group,
            label="Program catalog",
        )
        missing = {"program_id", "capacity"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"Program capacity data is missing columns: {sorted(missing)}."
            )
        if frame["program_id"].duplicated().any():
            duplicates = frame.loc[
                frame["program_id"].duplicated(False), "program_id"
            ].tolist()
            raise ValueError(
                f"Program capacity data has duplicate program IDs: {duplicates[:10]}."
            )

        updates = pd.to_numeric(frame["capacity"], errors="coerce")
        updates.index = frame["program_id"].astype(str)
        mapped = catalog["program_id"].astype(str).map(updates)
        if "capacity" in catalog.columns:
            catalog["capacity"] = mapped.combine_first(catalog["capacity"])
        else:
            catalog["capacity"] = mapped
        selected = catalog
    selected = apply_capacity_scenario(selected, scenario, group)
    return _attach_program_geography(selected, scenario, group)


def _attach_program_geography(
    frame: pd.DataFrame, scenario: DataScenario, group: str
) -> pd.DataFrame:
    """Attach each program's selected-vintage school location when available."""
    if "school_id" not in frame.columns:
        return frame
    role = f"{group}.schools"
    try:
        schools = load_school_records(scenario, role, filter_group=group)
    except KeyError:
        return frame
    location_columns = [
        column
        for column in ("Block", "BlockGroup", "Tract", "lat", "lon")
        if column in schools.columns
    ]
    if not location_columns:
        return frame
    locations = schools[["school_id", *location_columns]].drop_duplicates()
    if locations["school_id"].duplicated().any():
        duplicates = locations.loc[
            locations["school_id"].duplicated(False), "school_id"
        ].tolist()
        raise ValueError(
            f"School locations contain duplicate school IDs: {duplicates[:10]}."
        )
    result = frame.drop(columns=location_columns, errors="ignore").merge(
        locations, how="left", on="school_id", validate="m:1"
    )
    return result


def load_school_records(
    scenario: DataScenario,
    role: str = "assignment.schools",
    *,
    filter_group: str | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Load schools with central aliases and deterministic collision handling."""
    group = _filter_group(role, filter_group)
    frame = _load_school_keyed_records(scenario, role, group, **read_csv_kwargs)
    from loaders.geography import normalize_census_geography

    return normalize_census_geography(
        frame,
        scenario,
        group,
        source_vintage=_role_geography_vintage(scenario, role),
        style="school",
    )


__all__ = [
    "SPECIAL_PROGRAMS",
    "load_program_records",
    "load_school_records",
    "load_student_records",
    "normalize_school_records",
    "normalize_student_records",
    "normalize_grade",
    "parse_ranked_programs",
    "parse_ranked_schools",
    "read_csv",
    "read_csv_source",
    "school_id_aliases",
]

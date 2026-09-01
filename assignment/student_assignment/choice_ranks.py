"""Canonical preference ranks and cumulative choice outcomes."""

import ast
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from loaders import normalize_grade


ASSIGNMENT_SCHEMA_VERSION = 2
LISTED_RANK_BASIS = "listed"
UTILITY_RANK_BASIS = "utility"
RANK_BASES = {LISTED_RANK_BASIS, UTILITY_RANK_BASIS}
CANONICAL_RANK_COLUMNS = {
    "assignment_schema_version",
    "rank_basis",
    "submitted_rank",
    "utility_rank",
    "mechanism_rank",
    "rank",
    "In-Zone Rank",
}


@dataclass(frozen=True)
class ChoiceRate:
    numerator: int
    denominator: int

    @property
    def value(self) -> float:
        if self.denominator == 0:
            return float("nan")
        return self.numerator / self.denominator


def cumulative_choice_rates(
    students: pd.DataFrame,
    rank_column: str,
    thresholds: Iterable[int],
) -> dict[int, ChoiceRate]:
    """Return cumulative top-k outcomes for the supplied denominator cohort."""
    required = {"programno", "designation", rank_column}
    missing = required - set(students.columns)
    if missing:
        raise ValueError(f"Choice evaluation is missing columns: {sorted(missing)}")

    thresholds = tuple(dict.fromkeys(int(value) for value in thresholds))
    if any(value <= 0 for value in thresholds):
        raise ValueError("Choice thresholds must be positive integers.")

    programnos = pd.to_numeric(students["programno"], errors="coerce")
    designation = pd.to_numeric(students["designation"], errors="coerce")
    ranks = pd.to_numeric(students[rank_column], errors="coerce")
    assigned = programnos.gt(0).fillna(False)
    eligible_assignment = assigned & designation.eq(0).fillna(False)
    denominator = len(students)
    return {
        threshold: ChoiceRate(
            numerator=int((eligible_assignment & ranks.le(threshold)).sum()),
            denominator=denominator,
        )
        for threshold in thresholds
    }


def normalize_assignment_ranks(
    assignments: pd.DataFrame,
    *,
    listed_ranks: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Validate canonical ranks or migrate an unambiguous listed assignment."""
    result = assignments.copy()
    source_ranks = _aligned_source_ranks(listed_ranks, result.index)
    missing = CANONICAL_RANK_COLUMNS - set(result.columns)
    new_columns = CANONICAL_RANK_COLUMNS - {"rank", "In-Zone Rank"}

    if missing:
        partial = new_columns & set(result.columns)
        if partial:
            raise ValueError(
                "Assignments have a partial canonical rank schema; missing columns: "
                f"{sorted(missing)}"
            )
        if "assigned_utility" in result.columns:
            raise ValueError(
                "Legacy utility assignments require an explicit utility_rank."
            )
        if source_ranks is None:
            raise ValueError(
                "Legacy listed assignments require source exact-program preferences "
                "to reconstruct submitted_rank."
            )
        if "In-Zone Rank" not in result:
            raise ValueError(
                "Legacy listed assignments require In-Zone Rank to reconstruct "
                "mechanism_rank."
            )

        assigned = pd.to_numeric(result["programno"], errors="coerce").gt(0)
        result["assignment_schema_version"] = ASSIGNMENT_SCHEMA_VERSION
        result["rank_basis"] = LISTED_RANK_BASIS
        result["submitted_rank"] = source_ranks.where(assigned)
        result["utility_rank"] = np.nan
        result["rank"] = result["submitted_rank"]
        result["mechanism_rank"] = pd.to_numeric(
            result["In-Zone Rank"], errors="coerce"
        ).where(assigned)
        result["In-Zone Rank"] = result["mechanism_rank"]

    _validate_canonical_rank_columns(result, source_ranks)
    return result


def listed_preference_rank_matrix(
    student_data: pd.DataFrame,
    program_indices: Mapping[str, int],
) -> np.ndarray:
    """Build exact-program source ranks, preserving listed-rank gaps."""
    school_column, program_column, rank_column = _preference_columns(student_data)
    program_by_key, program_by_school_type = _program_by_choice_key(program_indices)
    matrix = np.full((len(student_data), len(program_indices)), np.nan, dtype=float)

    for row_position, (_, student) in enumerate(student_data.iterrows()):
        schools = _list_value(student[school_column], school_column)
        program_types = _list_value(student[program_column], program_column)
        if len(schools) != len(program_types):
            raise ValueError(
                f"{school_column} and {program_column} differ in length at row "
                f"{row_position}."
            )

        if rank_column is None:
            ranks = list(range(1, len(schools) + 1))
        else:
            ranks = _list_value(student[rank_column], rank_column)
            if not ranks and schools:
                ranks = list(range(1, len(schools) + 1))
            if len(ranks) != len(schools):
                raise ValueError(
                    f"{rank_column} does not align with preferences at row "
                    f"{row_position}."
                )

        numeric_ranks = _validated_ranks(ranks, rank_column or "preference position")
        grade = normalize_grade(student["grade"]) if "grade" in student_data else None
        if grade == "":
            grade = None
        seen_choices = set()
        for school, program_type, rank in zip(
            schools, program_types, numeric_ranks, strict=True
        ):
            school_type = (_school_key(school), str(program_type).strip().upper())
            key = (*school_type, grade)
            if key in seen_choices:
                raise ValueError(
                    f"Preference row {row_position} repeats school/program {key}."
                )
            seen_choices.add(key)
            if grade is None:
                if school_type not in program_by_school_type:
                    programno = None
                else:
                    programno = program_by_school_type[school_type]
                if programno is None and school_type in program_by_school_type:
                    raise ValueError(
                        "Student preferences require a grade to distinguish "
                        f"school/program {school_type}."
                    )
            else:
                programno = program_by_key.get(key)
            if programno is not None:
                matrix[row_position, programno - 1] = rank
    return matrix


def listed_ranks_for_program_ids(
    student_data: pd.DataFrame,
    program_ids: pd.Series,
) -> pd.Series:
    """Return source listed ranks for assigned exact program IDs."""
    if not isinstance(program_ids, pd.Series):
        raise TypeError("program_ids must be a Series indexed like student_data.")
    if not program_ids.index.equals(student_data.index):
        raise ValueError("Assigned program IDs do not align with student data.")

    normalized = program_ids.astype("string").str.strip()
    assigned_ids = normalized[normalized.notna() & normalized.ne("")]
    unique_ids = list(dict.fromkeys(assigned_ids.astype(str)))
    if not unique_ids:
        return pd.Series(np.nan, index=student_data.index, dtype=float)

    program_indices = {
        program_id: position for position, program_id in enumerate(unique_ids, start=1)
    }
    matches = normalized.map(program_indices).fillna(0).to_numpy(dtype=int)
    ranks = ranks_for_matches(
        listed_preference_rank_matrix(student_data, program_indices),
        matches,
    )
    return pd.Series(ranks, index=student_data.index, dtype=float)


def ranks_for_matches(rank_matrix: np.ndarray, matches: np.ndarray) -> np.ndarray:
    """Read assigned-program ranks from a student-by-program rank matrix."""
    matrix = np.asarray(rank_matrix, dtype=float)
    matches = np.asarray(matches)
    if matrix.ndim != 2 or matches.shape != (matrix.shape[0],):
        raise ValueError("Rank matrix and matches do not align by student.")
    if not np.issubdtype(matches.dtype, np.number):
        raise ValueError("Matches must contain numeric program numbers.")
    if np.any(~np.isfinite(matches)) or np.any(matches % 1 != 0):
        raise ValueError("Matches must contain finite integer program numbers.")
    matches = matches.astype(int)
    if np.any(matches < 0) or np.any(matches > matrix.shape[1]):
        raise ValueError("Matches contain an unknown program number.")

    ranks = np.full(len(matches), np.nan, dtype=float)
    assigned_rows = np.flatnonzero(matches > 0)
    ranks[assigned_rows] = matrix[assigned_rows, matches[assigned_rows] - 1]
    return ranks


def ranks_from_preference_order(
    preferences: np.ndarray, matches: np.ndarray
) -> np.ndarray:
    """Return each assigned program's position in an exact preference order."""
    preferences = np.asarray(preferences)
    matches = np.asarray(matches)
    if preferences.ndim != 2 or matches.shape != (preferences.shape[0],):
        raise ValueError("Preference order and matches do not align by student.")

    ranks = np.full(len(matches), np.nan, dtype=float)
    for student in np.flatnonzero(matches > 0):
        positions = np.flatnonzero(preferences[student] == matches[student])
        if len(positions) != 1:
            raise ValueError(
                f"Assigned program {matches[student]} occurs {len(positions)} times "
                f"in preference row {student}."
            )
        ranks[student] = int(positions[0]) + 1
    return ranks


def _validate_canonical_rank_columns(
    assignments: pd.DataFrame,
    source_ranks: pd.Series | None,
) -> None:
    schema_version = pd.to_numeric(
        assignments["assignment_schema_version"], errors="coerce"
    )
    if not schema_version.eq(ASSIGNMENT_SCHEMA_VERSION).all():
        raise ValueError(
            f"Assignments require rank schema version {ASSIGNMENT_SCHEMA_VERSION}."
        )
    assignments["assignment_schema_version"] = ASSIGNMENT_SCHEMA_VERSION

    rank_basis = assignments["rank_basis"].astype("string").str.strip().str.lower()
    if rank_basis.isna().any() or not rank_basis.isin(RANK_BASES).all():
        raise ValueError("Assignments contain an invalid rank_basis.")
    assignments["rank_basis"] = rank_basis.astype(str)

    for column in (
        "rank",
        "submitted_rank",
        "utility_rank",
        "mechanism_rank",
        "In-Zone Rank",
    ):
        raw = assignments[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        supplied = raw.notna() & raw.astype("string").str.strip().ne("")
        invalid = supplied & (
            numeric.isna()
            | ~np.isfinite(numeric)
            | numeric.le(0)
            | numeric.mod(1).ne(0)
        )
        if invalid.any():
            raise ValueError(f"Assignments contain invalid {column} values.")
        assignments[column] = numeric

    assigned = pd.to_numeric(assignments["programno"], errors="coerce").gt(0)
    expected_rank = assignments["submitted_rank"].where(
        rank_basis.eq(LISTED_RANK_BASIS),
        assignments["utility_rank"],
    )
    inconsistent = ~_nullable_equal(assignments["rank"], expected_rank)
    inconsistent |= ~_nullable_equal(
        assignments["mechanism_rank"], assignments["In-Zone Rank"]
    )
    inconsistent |= (
        rank_basis.eq(UTILITY_RANK_BASIS)
        & assigned
        & assignments["utility_rank"].isna()
    )
    rank_columns = [
        "rank",
        "submitted_rank",
        "utility_rank",
        "mechanism_rank",
        "In-Zone Rank",
    ]
    inconsistent |= (~assigned) & assignments[rank_columns].notna().any(axis=1)
    if source_ranks is not None:
        inconsistent |= ~_nullable_equal(assignments["submitted_rank"], source_ranks)
    if inconsistent.any():
        raise ValueError("Assignments contain inconsistent canonical rank values.")


def _aligned_source_ranks(
    listed_ranks: Iterable[float] | None,
    index: pd.Index,
) -> pd.Series | None:
    if listed_ranks is None:
        return None
    if isinstance(listed_ranks, pd.Series):
        if not listed_ranks.index.equals(index):
            raise ValueError("Source submitted ranks do not align with assignments.")
        return pd.to_numeric(listed_ranks, errors="coerce")
    values = np.asarray(listed_ranks, dtype=float)
    if values.shape != (len(index),):
        raise ValueError("Source submitted ranks do not align with assignments.")
    return pd.Series(values, index=index, dtype=float)


def _nullable_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.eq(right) | (left.isna() & right.isna())


def _preference_columns(student_data: pd.DataFrame) -> tuple[str, str, str | None]:
    selected = {"selected_ranked_idschool", "selected_programs"}
    if selected <= set(student_data.columns):
        rank_column = (
            "selected_listed_ranks"
            if "selected_listed_ranks" in student_data.columns
            else None
        )
        return "selected_ranked_idschool", "selected_programs", rank_column

    first_round = {"r1_ranked_idschool", "r1_programs"}
    if first_round <= set(student_data.columns):
        rank_column = "r1_listed_ranks" if "r1_listed_ranks" in student_data else None
        return "r1_ranked_idschool", "r1_programs", rank_column
    raise ValueError(
        "Student data has no selected or round-one exact-program preferences."
    )


def _program_by_choice_key(
    program_indices: Mapping[str, int],
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str], int | None]]:
    expected = set(range(1, len(program_indices) + 1))
    actual = {int(value) for value in program_indices.values()}
    if actual != expected:
        raise ValueError("Program indices must be contiguous and one-based.")

    result = {}
    result_by_school_type = {}
    for program_id, programno in program_indices.items():
        parts = str(program_id).split("-")
        if len(parts) < 3:
            raise ValueError(f"Invalid exact program ID: {program_id!r}")
        school_type = (_school_key(parts[0]), "-".join(parts[1:-1]).upper())
        key = (*school_type, normalize_grade(parts[-1]))
        if key in result:
            raise ValueError(
                f"Program catalog repeats exact school/program/grade {key}."
            )
        result[key] = int(programno)
        if school_type in result_by_school_type:
            result_by_school_type[school_type] = None
        else:
            result_by_school_type[school_type] = int(programno)
    return result, result_by_school_type


def _school_key(value) -> str:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid school identity: {value!r}") from exc
    if not np.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"Invalid school identity: {value!r}")
    return str(int(number))


def _list_value(value, column: str) -> list:
    if isinstance(value, list):
        return value.copy()
    if isinstance(value, (tuple, np.ndarray)):
        return list(value)
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return []
    if not isinstance(value, str):
        raise ValueError(f"{column} must contain a list, got {value!r}.")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{column} contains an invalid list: {value!r}.") from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{column} must contain a list, got {value!r}.")
    return list(parsed)


def _validated_ranks(values: list, column: str) -> list[int]:
    numeric = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce")
    invalid = (
        numeric.isna() | ~np.isfinite(numeric) | (numeric <= 0) | (numeric % 1 != 0)
    )
    if invalid.any():
        raise ValueError(f"{column} contains invalid preference ranks.")
    ranks = numeric.astype(int).tolist()
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"{column} contains duplicate preference ranks.")
    return ranks

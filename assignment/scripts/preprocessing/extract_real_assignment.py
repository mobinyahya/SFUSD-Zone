"""Extract the real assignment from the student dataset.

This script reads the student CSV file and extracts the Round 1 assignment
columns to produce an assignment CSV file formatted for downstream analysis.

Usage:
    python scripts/extract_real_assignment.py \
        student_csv_file=/path/to/students.csv \
        assignment_csv_file=/path/to/output/assignment.csv \
        programs_csv_file=/path/to/programs.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from assignment.student_assignment.choice_ranks import (
    ASSIGNMENT_SCHEMA_VERSION,
    LISTED_RANK_BASIS,
    listed_preference_rank_matrix,
    normalize_assignment_ranks,
    ranks_for_matches,
)


def _normalize_grade(value) -> str:
    """Normalize numeric grades to the two-character program suffix."""
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    try:
        number = float(text)
    except ValueError:
        return text
    if np.isfinite(number) and number.is_integer():
        return str(int(number)).zfill(2)
    return text


def extract_real_assignment(
    df_students: pd.DataFrame,
    df_programs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract the real assignment from the student dataframe.

    This function extracts the Round 1 assignment columns from the student
    dataframe and formats them according to the canonical assignment schema.

    Args:
        df_students: The student dataframe containing Round 1 assignment
            columns (r1_idschool, r1_programcode, grade,
            r1_ranked_idschool, r1_programs, and optionally r1_rank and
            r1_isdesignation).
        df_programs: Programs dataframe used for an exact programno lookup.

    Returns:
        A dataframe with canonical submitted and mechanism rank columns.
    """
    required_columns = [
        "studentno",
        "r1_idschool",
        "r1_programcode",
        "r1_ranked_idschool",
        "r1_programs",
        "grade",
    ]
    missing_columns = [
        col for col in required_columns if col not in df_students.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in student dataframe: {missing_columns}"
        )
    if df_programs is None:
        raise ValueError("df_programs is required")

    student_ids = df_students["studentno"]
    missing_student_ids = student_ids.isna() | student_ids.astype(
        "string"
    ).str.strip().eq("")
    if missing_student_ids.fillna(True).any():
        raise ValueError("Student dataframe contains a missing studentno")
    duplicate_students = student_ids[student_ids.duplicated(keep=False)]
    if not duplicate_students.empty:
        raise ValueError(
            "Student dataframe contains duplicate studentno values: "
            f"{duplicate_students.unique().tolist()}"
        )

    required_program_columns = {"program_id", "programno"}
    missing_program_columns = required_program_columns - set(df_programs.columns)
    if missing_program_columns:
        raise ValueError(
            "Missing required columns in programs dataframe: "
            f"{sorted(missing_program_columns)}"
        )

    raw_school_ids = df_students["r1_idschool"]
    school_ids = pd.to_numeric(raw_school_ids, errors="coerce")
    supplied_school = raw_school_ids.notna() & raw_school_ids.astype(
        "string"
    ).str.strip().ne("")
    invalid_school = supplied_school & (
        school_ids.isna() | (school_ids < 0) | (school_ids % 1 != 0)
    )
    if invalid_school.any():
        students = df_students.loc[invalid_school, "studentno"].tolist()
        raise ValueError(f"Invalid r1_idschool for students: {students}")
    assigned = school_ids.fillna(0) > 0

    program_types = df_students["r1_programcode"].astype("string").str.strip()
    grades = df_students["grade"].map(_normalize_grade)
    missing_assigned_parts = assigned & (
        program_types.isna()
        | program_types.eq("").fillna(False)
        | grades.isna()
        | grades.eq("").fillna(False)
    )
    if missing_assigned_parts.any():
        students = df_students.loc[missing_assigned_parts, "studentno"].tolist()
        raise ValueError(
            f"Assigned students require r1_programcode and grade values: {students}"
        )

    assignment = pd.DataFrame(index=df_students.index)
    assignment["studentno"] = df_students["studentno"]
    assignment["programno"] = 0
    assignment["programcodes"] = ""

    school_id_strings = school_ids.loc[assigned].astype(int).astype(str).str.zfill(3)
    assignment.loc[assigned, "programcodes"] = (
        school_id_strings
        + "-"
        + program_types.loc[assigned].astype(str)
        + "-"
        + grades.loc[assigned].astype(str)
    )

    programs = df_programs.copy()
    program_ids = programs["program_id"].astype("string").str.strip()
    if (program_ids.isna() | program_ids.eq("").fillna(False)).any():
        raise ValueError("Programs dataframe contains a missing program_id")
    if program_ids.duplicated().any():
        duplicates = program_ids[program_ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            f"Programs dataframe contains duplicate program_id values: {duplicates}"
        )
    program_numbers = pd.to_numeric(programs["programno"], errors="coerce")
    invalid_program_numbers = (
        program_numbers.isna()
        | ~np.isfinite(program_numbers)
        | (program_numbers <= 0)
        | (program_numbers % 1 != 0)
    )
    if invalid_program_numbers.any():
        raise ValueError("Programs dataframe contains invalid programno values")
    if program_numbers.duplicated().any():
        duplicates = program_numbers[program_numbers.duplicated(keep=False)].unique()
        raise ValueError(
            "Programs dataframe contains duplicate programno values: "
            f"{duplicates.tolist()}"
        )
    order = np.argsort(program_numbers.to_numpy(), kind="stable")
    normalized_program_numbers = np.empty(len(programs), dtype=int)
    normalized_program_numbers[order] = np.arange(1, len(programs) + 1)
    program_lookup = dict(zip(program_ids.astype(str), normalized_program_numbers))

    mapped_programs = assignment.loc[assigned, "programcodes"].map(program_lookup)
    if mapped_programs.isna().any():
        unknown = (
            assignment.loc[
                mapped_programs.index[mapped_programs.isna()], "programcodes"
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Assigned programcodes have no exact program mapping: {unknown}"
        )
    assignment.loc[assigned, "programno"] = mapped_programs.astype(int)
    assignment["programno"] = assignment["programno"].astype(int)

    if "r1_rank" in df_students:
        raw_ranks = df_students["r1_rank"]
        ranks = pd.to_numeric(raw_ranks, errors="coerce")
        supplied_ranks = raw_ranks.notna() & raw_ranks.astype("string").str.strip().ne(
            ""
        )
        invalid_ranks = (
            assigned
            & supplied_ranks
            & (ranks.isna() | ~np.isfinite(ranks) | (ranks <= 0) | (ranks % 1 != 0))
        )
        if invalid_ranks.any():
            students = df_students.loc[invalid_ranks, "studentno"].tolist()
            raise ValueError(f"Invalid r1_rank for students: {students}")
    else:
        ranks = pd.Series(float("nan"), index=df_students.index, dtype=float)

    if "r1_isdesignation" in df_students:
        raw_designation = df_students["r1_isdesignation"]
        designation = pd.to_numeric(raw_designation, errors="coerce")
        supplied_designation = raw_designation.notna() & raw_designation.astype(
            "string"
        ).str.strip().ne("")
        invalid_designation = supplied_designation & (
            designation.isna() | ~np.isfinite(designation) | ~designation.isin([0, 1])
        )
        if invalid_designation.any():
            students = df_students.loc[invalid_designation, "studentno"].tolist()
            raise ValueError(f"Invalid r1_isdesignation for students: {students}")
        designation = designation.fillna(0)
    else:
        designation = pd.Series(0, index=df_students.index, dtype=int)
    unassigned_designated = ~assigned & designation.eq(1)
    if unassigned_designated.any():
        students = df_students.loc[unassigned_designated, "studentno"].tolist()
        raise ValueError(f"Unassigned students cannot be designated: {students}")
    assignment["designation"] = designation.astype(int)

    preference_columns = ["r1_ranked_idschool", "r1_programs", "grade"]
    if "r1_listed_ranks" in df_students:
        preference_columns.append("r1_listed_ranks")
    submitted_rank = ranks_for_matches(
        listed_preference_rank_matrix(
            df_students[preference_columns],
            program_lookup,
        ),
        assignment["programno"].to_numpy(),
    )
    assignment["assignment_schema_version"] = ASSIGNMENT_SCHEMA_VERSION
    assignment["rank_basis"] = LISTED_RANK_BASIS
    assignment["submitted_rank"] = submitted_rank
    assignment["utility_rank"] = np.nan
    assignment["rank"] = submitted_rank
    assignment["mechanism_rank"] = ranks.mask(~assigned)
    assignment["In-Zone Rank"] = assignment["mechanism_rank"]
    assignment = normalize_assignment_ranks(
        assignment,
        listed_ranks=pd.Series(submitted_rank, index=assignment.index),
    )

    # Select and order output columns
    output_columns = [
        "assignment_schema_version",
        "studentno",
        "programno",
        "programcodes",
        "rank_basis",
        "submitted_rank",
        "utility_rank",
        "rank",
        "mechanism_rank",
        "designation",
        "In-Zone Rank",
    ]
    return assignment[output_columns].copy()


def main(args) -> None:
    """Main entry point for extracting real assignment.

    Args:
        cfg: Hydra configuration containing:
            - student_csv_file: Path to the input student CSV file.
            - assignment_csv_file: Path to the output assignment CSV file.
            - programs_csv_file: Path to the programs CSV file for programno
              lookup.
    """
    # Validate required config keys
    if not getattr(args, "student_csv_file", None):
        raise ValueError("Config must contain 'student_csv_file'.")
    if not getattr(args, "assignment_csv_file", None):
        raise ValueError("Config must contain 'assignment_csv_file'.")
    if not getattr(args, "programs_csv_file", None):
        raise ValueError("Config must contain 'programs_csv_file'.")

    student_path = Path(args.student_csv_file)
    assignment_path = Path(args.assignment_csv_file)

    if not student_path.exists():
        raise FileNotFoundError(f"Student CSV file not found: {student_path}")

    # Load student data
    df_students = pd.read_csv(student_path)

    programs_path = Path(args.programs_csv_file)
    if not programs_path.exists():
        raise FileNotFoundError(f"Programs CSV file not found: {programs_path}")
    df_programs = pd.read_csv(programs_path)

    # Extract assignment
    df_assignment = extract_real_assignment(df_students, df_programs)

    # Ensure output directory exists
    assignment_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    df_assignment.to_csv(assignment_path, index=False)

    print(f"Extracted {len(df_assignment)} assignments to {assignment_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the real assignment from the student dataset."
    )
    parser.add_argument(
        "--student_csv_file",
        required=True,
        help="Path to the input student CSV file.",
    )
    parser.add_argument(
        "--assignment_csv_file",
        required=True,
        help="Path to the output assignment CSV file.",
    )
    parser.add_argument(
        "--programs_csv_file",
        required=True,
        help="Path to the programs CSV file for programno lookup.",
    )
    args = parser.parse_args()
    main(args)

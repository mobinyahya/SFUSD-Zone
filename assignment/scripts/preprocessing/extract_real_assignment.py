"""Extract the real assignment from the student dataset.

This script reads the student CSV file and extracts the Round 1 assignment
columns to produce an assignment CSV file formatted for downstream analysis.

Usage:
    python scripts/extract_real_assignment.py \
        student_csv_file=/path/to/students.csv \
        assignment_csv_file=/path/to/output/assignment.csv \
        [programs_csv_file=/path/to/programs.csv]
"""

import argparse
from pathlib import Path

import pandas as pd


def extract_real_assignment(
    df_students: pd.DataFrame,
    df_programs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract the real assignment from the student dataframe.

    This function extracts the Round 1 assignment columns from the student
    dataframe and formats them according to the assignment schema.

    Args:
        df_students: The student dataframe containing Round 1 assignment
            columns (r1_idschool, r1_programcode, grade, r1_rank,
            r1_isdesignation).
        df_programs: Optional programs dataframe for looking up programno.
            If not provided, programno will be set based on row index.

    Returns:
        A dataframe with columns: studentno, programno, programcodes, rank,
        designation, In-Zone Rank.
    """
    # Filter to students with a valid Round 1 assignment
    required_columns = ["studentno", "r1_idschool", "r1_programcode", "grade"]
    missing_columns = [
        col for col in required_columns if col not in df_students.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in student dataframe: {missing_columns}"
        )

    df_assigned = df_students.dropna(subset=["r1_idschool"]).copy()

    # Build programcodes: {school_id}-{program_type}-{grade}
    # Ensure school_id is formatted as 3 digits with leading zeros
    df_assigned["school_id_str"] = (
        df_assigned["r1_idschool"].astype(int).astype(str).str.zfill(3)
    )
    df_assigned["programcodes"] = (
        df_assigned["school_id_str"]
        + "-"
        + df_assigned["r1_programcode"].astype(str)
        + "-"
        + df_assigned["grade"].astype(str)
    )

    # Look up programno from programs table if provided
    if df_programs is not None and "programno" in df_programs.columns:
        program_lookup = df_programs.set_index("program_id")["programno"]
        df_assigned["programno"] = df_assigned["programcodes"].map(
            program_lookup
        )
    else:
        # Fallback: use a sequential index
        df_assigned["programno"] = range(len(df_assigned))

    # Extract rank and designation columns
    # Use 0 as default for missing rank values
    df_assigned["rank"] = df_assigned.get("r1_rank", 0).fillna(0).astype(int)
    df_assigned["designation"] = (
        df_assigned.get("r1_isdesignation", 0).fillna(0).astype(int)
    )

    # In-Zone Rank is the same as rank for real assignments
    df_assigned["In-Zone Rank"] = df_assigned["rank"]

    # Select and order output columns
    output_columns = [
        "studentno",
        "programno",
        "programcodes",
        "rank",
        "designation",
        "In-Zone Rank",
    ]
    df_assignment = df_assigned[output_columns].copy()

    return df_assignment


def main(args) -> None:
    """Main entry point for extracting real assignment.

    Args:
        cfg: Hydra configuration containing:
            - student_csv_file: Path to the input student CSV file.
            - assignment_csv_file: Path to the output assignment CSV file.
            - programs_csv_file (optional): Path to the programs CSV file
              for programno lookup.
    """
    # Validate required config keys
    if "student_csv_file" not in args:
        raise ValueError("Config must contain 'student_csv_file'.")
    if "assignment_csv_file" not in args:
        raise ValueError("Config must contain 'assignment_csv_file'.")

    student_path = Path(args.student_csv_file)
    assignment_path = Path(args.assignment_csv_file)

    if not student_path.exists():
        raise FileNotFoundError(f"Student CSV file not found: {student_path}")

    # Load student data
    df_students = pd.read_csv(student_path)

    # Optionally load programs data for programno lookup
    df_programs = None
    if "programs_csv_file" in args and args.programs_csv_file:
        programs_path = Path(args.programs_csv_file)
        if programs_path.exists():
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
        help="Path to the programs CSV file for programno lookup.",
    )
    args = parser.parse_args()
    main(args)

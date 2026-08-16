"""Prepare paired KG round-one student and program inputs.

The full outputs retain special-program students and alternatives so the
scenario's ``assignment.special_programs`` mode can select them. Explicit
no-special outputs are also written for standalone consumers.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from loaders import SPECIAL_PROGRAMS  # noqa: E402


DEFAULT_CLEANED_DIR = Path("/share/data/school_choice/Data/Cleaned")


def _parse_list(value: Any, column: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Could not parse {column} value {value!r} as a list."
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Expected {column} value {value!r} to be a list.")
    return parsed


def prepare_student_inputs(
    students: pd.DataFrame, grade: str = "KG"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return round-one students with and without special-program applicants."""
    required = {"studentno", "grade", "r1_ranked_idschool", "r1_programs"}
    missing = required - set(students.columns)
    if missing:
        raise ValueError(f"Student input is missing columns: {sorted(missing)}")
    if students["studentno"].isna().any() or students["studentno"].duplicated().any():
        raise ValueError(
            "Student input must contain unique, non-null studentno values."
        )

    grade_students = students.loc[
        students["grade"].astype(str).str.upper() == grade.upper()
    ].copy()
    rankings = grade_students["r1_ranked_idschool"].map(
        lambda value: _parse_list(value, "r1_ranked_idschool")
    )
    programs = grade_students["r1_programs"].map(
        lambda value: _parse_list(value, "r1_programs")
    )

    round_one = grade_students.loc[rankings.map(bool)].copy()
    rankings = rankings.loc[round_one.index]
    programs = programs.loc[round_one.index]
    mismatched = rankings.map(len) != programs.map(len)
    if mismatched.any():
        studentnos = round_one.loc[mismatched, "studentno"].tolist()
        raise ValueError(
            "Round-one school and program lists have different lengths for "
            f"studentno values: {studentnos[:10]}"
        )

    has_special = programs.map(
        lambda values: bool(set(map(str, values)) & SPECIAL_PROGRAMS)
    )
    without_special = round_one.loc[~has_special].copy()
    return round_one.reset_index(drop=True), without_special.reset_index(drop=True)


def prepare_program_inputs(
    programs: pd.DataFrame, grade: str = "KG"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return grade-specific programs with and without special alternatives."""
    required = {"program_id", "program_type"}
    missing = required - set(programs.columns)
    if missing:
        raise ValueError(f"Program input is missing columns: {sorted(missing)}")

    programs = programs.loc[
        :, ~programs.columns.astype(str).str.startswith("Unnamed:")
    ].copy()
    program_grades = programs["program_id"].astype(str).str.rsplit("-", n=1).str[-1]
    grade_programs = programs.loc[program_grades.str.upper() == grade.upper()].copy()
    if grade_programs["program_id"].duplicated().any():
        raise ValueError("Program input contains duplicate program_id values.")

    grade_programs.reset_index(drop=True, inplace=True)
    without_special = grade_programs.loc[
        ~grade_programs["program_type"].isin(SPECIAL_PROGRAMS)
    ].copy()
    without_special.reset_index(drop=True, inplace=True)
    if "programno" in grade_programs.columns:
        grade_programs["programno"] = grade_programs.index + 1
        without_special["programno"] = without_special.index + 1
    return grade_programs, without_special


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def generate_inputs(
    students_path: Path,
    programs_path: Path,
    output_dir: Path,
    *,
    year: str = "2324",
    grade: str = "KG",
) -> dict[str, Path]:
    """Generate full and no-special student/program CSV pairs."""
    students = pd.read_csv(students_path, low_memory=False)
    programs = pd.read_csv(programs_path, low_memory=False)
    full_students, no_special_students = prepare_student_inputs(students, grade)
    full_programs, no_special_programs = prepare_program_inputs(programs, grade)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{year}_{grade.lower()}_r1"
    paths = {
        "students": output_dir / f"student_{prefix}.csv",
        "students_no_special": output_dir / f"student_{prefix}_no_special.csv",
        "programs": output_dir / f"programs_{prefix}.csv",
        "programs_no_special": output_dir / f"programs_{prefix}_no_special.csv",
    }
    _write_csv(full_students, paths["students"])
    _write_csv(no_special_students, paths["students_no_special"])
    _write_csv(full_programs, paths["programs"])
    _write_csv(no_special_programs, paths["programs_no_special"])

    print(f"Students with special programs: {len(full_students)}")
    print(f"Students without special programs: {len(no_special_students)}")
    print(f"Programs with special programs: {len(full_programs)}")
    print(f"Programs without special programs: {len(no_special_programs)}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare unmodified KG round-one inputs with optional specials."
    )
    parser.add_argument(
        "--students",
        type=Path,
        default=DEFAULT_CLEANED_DIR / "student_2324.csv",
    )
    parser.add_argument(
        "--programs",
        type=Path,
        default=DEFAULT_CLEANED_DIR / "programs_2324.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local-data/choice_inputs/2324"),
    )
    parser.add_argument("--year", default="2324")
    parser.add_argument("--grade", default="KG")
    args = parser.parse_args()
    generate_inputs(
        args.students,
        args.programs,
        args.output_dir,
        year=args.year,
        grade=args.grade,
    )


if __name__ == "__main__":
    main()

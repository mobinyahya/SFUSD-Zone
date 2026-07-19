"""
Compare two assignment files and report the number of students with the same
assignment.

This script takes two assignment CSV files as input and compares how many
students received the same programcode in both assignments.
"""

import argparse
from pathlib import Path

import pandas as pd


def load_assignment(file_path: Path) -> pd.DataFrame:
    """
    Load an assignment CSV file.

    Args:
        file_path: Path to the assignment CSV file.

    Returns:
        DataFrame containing the assignment data.
    """
    df = pd.read_csv(file_path)

    # Ensure required columns exist
    if "studentno" not in df.columns or "programcodes" not in df.columns:
        raise ValueError(
            f"Assignment file must contain 'studentno' and 'programcodes' "
            f"columns. Found: {df.columns.tolist()}"
        )

    return df


def compare_assignments(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> tuple[int, int, int, pd.DataFrame]:
    """
    Compare two assignment DataFrames.

    Args:
        df1: First assignment DataFrame.
        df2: Second assignment DataFrame.

    Returns:
        Tuple containing:
            - Number of students with same assignment
            - Total number of students in common
            - Total unique students across both assignments
            - DataFrame with comparison details
    """
    # Get the intersection of students
    students_in_both = set(df1["studentno"]) & set(df2["studentno"])
    all_students = set(df1["studentno"]) | set(df2["studentno"])

    # Filter to common students
    df1_common = df1[df1["studentno"].isin(students_in_both)].copy()
    df2_common = df2[df2["studentno"].isin(students_in_both)].copy()

    # Merge on studentno
    merged = df1_common.merge(
        df2_common, on="studentno", suffixes=("_file1", "_file2"), how="inner"
    )

    # Count matches
    merged["same_assignment"] = (
        merged["programcodes_file1"] == merged["programcodes_file2"]
    )
    num_same = merged["same_assignment"].sum()
    num_common = len(students_in_both)
    num_total = len(all_students)

    return num_same, num_common, num_total, merged


def main() -> None:
    """Main function to compare two assignment files."""
    parser = argparse.ArgumentParser(
        description="Compare two assignment files and count students with "
        "the same assignment."
    )
    parser.add_argument(
        "file1", type=str, help="Path to the first assignment CSV file"
    )
    parser.add_argument(
        "file2", type=str, help="Path to the second assignment CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Path to save detailed comparison CSV",
    )

    args = parser.parse_args()

    # Load assignments
    print(f"Loading first assignment from: {args.file1}")
    df1 = load_assignment(Path(args.file1))
    print(f"  - Found {len(df1)} students")

    print(f"\nLoading second assignment from: {args.file2}")
    df2 = load_assignment(Path(args.file2))
    print(f"  - Found {len(df2)} students")

    # Compare
    print("\nComparing assignments...")
    num_same, num_common, num_total, merged = compare_assignments(df1, df2)

    # Report results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Total unique students across both files: {num_total}")
    print(f"Students in both files: {num_common}")
    print(f"Students only in file 1: {len(df1) - num_common}")
    print(f"Students only in file 2: {len(df2) - num_common}")
    print(f"\nStudents with SAME assignment: {num_same}")
    print(f"Students with DIFFERENT assignment: {num_common - num_same}")
    print(
        f"\nPercentage with same assignment: {100 * num_same / num_common:.2f}%"
    )
    print("=" * 60)

    # Save detailed comparison if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"\nDetailed comparison saved to: {output_path}")


if __name__ == "__main__":
    main()

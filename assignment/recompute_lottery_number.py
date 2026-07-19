"""Recompute student lottery numbers based on distance to ranked schools.

This script takes a student dataframe and a school dataframe as input,
and returns a new student dataframe where the lottery numbers
(in column "r1_randomnumber") have been recomputed using the formula:

    new_lottery_number = 1 - (distance / max_distance)

Where:
    - distance: haversine distance from student to the corresponding ranked school
    - max_distance: maximum distance between any student and any school in the data
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def haversine_vectorized(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """Compute great-circle distance between coordinate pairs in miles (vectorized).

    Uses the haversine formula on a sphere with radius 3958.8 miles.

    Args:
        lat1: Latitude(s) of point(s) 1 in degrees.
        lon1: Longitude(s) of point(s) 1 in degrees.
        lat2: Latitude(s) of point(s) 2 in degrees.
        lon2: Longitude(s) of point(s) 2 in degrees.

    Returns:
        Distance(s) in miles (non-negative).
    """
    earth_radius_miles = 3958.8

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    haversine_a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    haversine_c = 2 * np.arcsin(np.sqrt(haversine_a))

    return earth_radius_miles * haversine_c


def parse_string_list(value: str) -> list:
    """Safely parse a string-encoded list.

    Args:
        value: A string representation of a list (e.g., "[1, 2, 3]").

    Returns:
        The parsed list, or an empty list if parsing fails.
    """
    if pd.isna(value) or value == "" or value == "[]":
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def compute_max_distance(
    df_students: pd.DataFrame,
    df_schools: pd.DataFrame,
    student_lat_col: str = "latitude",
    student_lon_col: str = "longitude",
    school_lat_col: str = "lat",
    school_lon_col: str = "lon",
) -> float:
    """Compute maximum possible distance between any student and any school.

    Uses the bounding box corners to efficiently find the maximum distance
    without computing all pairwise distances.

    Args:
        df_students: Student dataframe with latitude/longitude columns.
        df_schools: School dataframe with lat/lon columns.
        student_lat_col: Name of student latitude column.
        student_lon_col: Name of student longitude column.
        school_lat_col: Name of school latitude column.
        school_lon_col: Name of school longitude column.

    Returns:
        Maximum distance in miles between any student-school pair.
    """
    # Get valid coordinates only
    valid_students = df_students[
        df_students[student_lat_col].notna()
        & df_students[student_lon_col].notna()
    ]
    valid_schools = df_schools[
        df_schools[school_lat_col].notna() & df_schools[school_lon_col].notna()
    ]

    if valid_students.empty or valid_schools.empty:
        raise ValueError(
            "No valid coordinates found in student or school data."
        )

    # Get bounding box corners for students and schools
    student_lat_min = valid_students[student_lat_col].min()
    student_lat_max = valid_students[student_lat_col].max()
    student_lon_min = valid_students[student_lon_col].min()
    student_lon_max = valid_students[student_lon_col].max()

    school_lat_min = valid_schools[school_lat_col].min()
    school_lat_max = valid_schools[school_lat_col].max()
    school_lon_min = valid_schools[school_lon_col].min()
    school_lon_max = valid_schools[school_lon_col].max()

    # Compute distances between all corner combinations
    student_corners = [
        (student_lat_min, student_lon_min),
        (student_lat_min, student_lon_max),
        (student_lat_max, student_lon_min),
        (student_lat_max, student_lon_max),
    ]
    school_corners = [
        (school_lat_min, school_lon_min),
        (school_lat_min, school_lon_max),
        (school_lat_max, school_lon_min),
        (school_lat_max, school_lon_max),
    ]

    max_dist = 0.0
    for student_lat, student_lon in student_corners:
        for school_lat, school_lon in school_corners:
            dist = haversine_vectorized(
                np.array([student_lat]),
                np.array([student_lon]),
                np.array([school_lat]),
                np.array([school_lon]),
            )[0]
            max_dist = max(max_dist, dist)

    return max_dist


def build_school_coordinate_lookup(
    df_schools: pd.DataFrame,
    school_id_col: str = "school_id",
    school_lat_col: str = "lat",
    school_lon_col: str = "lon",
) -> tuple[pd.Series, pd.Series]:
    """Build lookup series for school coordinates indexed by school_id.

    Args:
        df_schools: School dataframe.
        school_id_col: Name of school ID column.
        school_lat_col: Name of school latitude column.
        school_lon_col: Name of school longitude column.

    Returns:
        Tuple of (lat_lookup, lon_lookup) Series indexed by school_id.
    """
    school_coords = df_schools.drop_duplicates(
        subset=[school_id_col]
    ).set_index(school_id_col)
    lat_lookup = school_coords[school_lat_col]
    lon_lookup = school_coords[school_lon_col]
    return lat_lookup, lon_lookup


def compute_lottery_numbers_for_row(
    ranked_schools: list[int],
    student_lat: float,
    student_lon: float,
    school_lat_lookup: pd.Series,
    school_lon_lookup: pd.Series,
    max_distance: float,
) -> list[float]:
    """Compute new lottery numbers for a single student's ranked schools.

    Args:
        ranked_schools: List of school IDs the student ranked.
        student_lat: Student's home latitude.
        student_lon: Student's home longitude.
        school_lat_lookup: Series mapping school_id -> latitude.
        school_lon_lookup: Series mapping school_id -> longitude.
        max_distance: Maximum distance for normalization.

    Returns:
        List of new lottery numbers (one per ranked school).
    """
    if not ranked_schools:
        return []

    # If student coordinates are missing, return neutral values (1.0) for all schools
    if pd.isna(student_lat) or pd.isna(student_lon):
        return [1.0] * len(ranked_schools)

    new_lottery_numbers: list[float] = []
    for school_id in ranked_schools:
        try:
            school_lat = school_lat_lookup.get(school_id, np.nan)
            school_lon = school_lon_lookup.get(school_id, np.nan)

            if pd.isna(school_lat) or pd.isna(school_lon):
                # If school coordinates missing, assign neutral value
                new_lottery_numbers.append(0.5)
                continue

            distance = haversine_vectorized(
                np.array([student_lat]),
                np.array([student_lon]),
                np.array([school_lat]),
                np.array([school_lon]),
            )[0]

            # Formula: 1 - distance/max_distance
            # Closer schools get higher lottery numbers (closer to 1)
            lottery_number = (
                1.0 - (distance / max_distance) if max_distance > 0 else 0.5
            )
            # Convert to native Python float to avoid np.float64 in string output
            new_lottery_numbers.append(float(lottery_number))

        except (KeyError, TypeError):
            new_lottery_numbers.append(0.5)
    return new_lottery_numbers


def recompute_lottery_numbers(
    df_students: pd.DataFrame,
    df_schools: pd.DataFrame,
    student_lat_col: str = "latitude",
    student_lon_col: str = "longitude",
    school_id_col: str = "school_id",
    school_lat_col: str = "lat",
    school_lon_col: str = "lon",
    ranked_schools_col: str = "r1_ranked_idschool",
    lottery_col: str = "r1_randomnumber",
) -> pd.DataFrame:
    """Recompute lottery numbers based on student-school distances.

    The new lottery number for each ranked school is computed as:
        lottery_number = 1 - (distance / max_distance)

    Where distance is the haversine distance from the student to the ranked
    school, and max_distance is the maximum distance between any student
    and any school in the dataset.

    Args:
        df_students: Student dataframe with coordinates and ranked schools.
        df_schools: School dataframe with coordinates.
        student_lat_col: Name of student latitude column.
        student_lon_col: Name of student longitude column.
        school_id_col: Name of school ID column in schools dataframe.
        school_lat_col: Name of school latitude column.
        school_lon_col: Name of school longitude column.
        ranked_schools_col: Name of column containing ranked school IDs
            (string-encoded list).
        lottery_col: Name of column to store new lottery numbers.

    Returns:
        New student dataframe with recomputed lottery numbers.

    Raises:
        ValueError: If required columns are missing or no valid coordinates exist.
    """
    # Validate required columns
    required_student_cols = {
        student_lat_col,
        student_lon_col,
        ranked_schools_col,
    }
    missing_student_cols = required_student_cols - set(df_students.columns)
    if missing_student_cols:
        raise ValueError(
            f"Missing required student columns: {', '.join(sorted(missing_student_cols))}"
        )

    required_school_cols = {school_id_col, school_lat_col, school_lon_col}
    missing_school_cols = required_school_cols - set(df_schools.columns)
    if missing_school_cols:
        raise ValueError(
            f"Missing required school columns: {', '.join(sorted(missing_school_cols))}"
        )

    # Work on a copy to avoid modifying original
    df_result = df_students.copy()

    # Compute maximum distance
    max_distance = compute_max_distance(
        df_students=df_students,
        df_schools=df_schools,
        student_lat_col=student_lat_col,
        student_lon_col=student_lon_col,
        school_lat_col=school_lat_col,
        school_lon_col=school_lon_col,
    )

    print(f"Maximum student-school distance: {max_distance:.2f} miles")

    # Build school coordinate lookups
    school_lat_lookup, school_lon_lookup = build_school_coordinate_lookup(
        df_schools=df_schools,
        school_id_col=school_id_col,
        school_lat_col=school_lat_col,
        school_lon_col=school_lon_col,
    )
    # Parse ranked schools column if it's string-encoded
    parsed_schools = df_result[ranked_schools_col].apply(
        lambda x: (
            parse_string_list(x) if isinstance(x, str) else (x if x else [])
        )
    )
    # Compute new lottery numbers using vectorized apply
    df_result[lottery_col] = [
        compute_lottery_numbers_for_row(
            ranked_schools=schools,
            student_lat=lat,
            student_lon=lon,
            school_lat_lookup=school_lat_lookup,
            school_lon_lookup=school_lon_lookup,
            max_distance=max_distance,
        )
        for schools, lat, lon in zip(
            parsed_schools,
            df_result[student_lat_col],
            df_result[student_lon_col],
        )
    ]

    # Convert lists to string representation to match original format
    df_result[lottery_col] = df_result[lottery_col].apply(str)

    return df_result


def main(
    student_file: Path,
    school_file: Path,
    output_file: Path | None = None,
) -> pd.DataFrame:
    """Main entry point for recomputing lottery numbers.

    Args:
        student_file: Path to student CSV file.
        school_file: Path to school CSV file.
        output_file: Optional path to save output CSV. If None, only returns df.

    Returns:
        Student dataframe with recomputed lottery numbers.
    """
    print(f"Loading student data from: {student_file}")
    df_students = pd.read_csv(student_file)
    df_students = df_students[df_students["grade"] == "KG"].reset_index(
        drop=True
    )

    print(f"Loading school data from: {school_file}")
    df_schools = pd.read_csv(school_file)

    print(f"Students: {len(df_students)}, Schools: {len(df_schools)}")

    df_result = recompute_lottery_numbers(
        df_students=df_students,
        df_schools=df_schools,
    )

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(output_file, index=False)
        print(f"Output saved to: {output_file}")

    return df_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recompute lottery numbers based on student-school distances."
    )
    parser.add_argument(
        "--students",
        type=Path,
        help="Path to student CSV file.",
        default="/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/r1_filter_student_without_specialprogs_2324.csv",
    )
    parser.add_argument(
        "--schools",
        type=Path,
        help="Path to school CSV file.",
        default="/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/schools_rehauled_withMissionBay_2324.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="./local-data/new_lottery/recomputed_lottery_numbers.csv",
        help="Path to save output CSV (relative to the repo root by default).",
    )

    args = parser.parse_args()

    main(
        student_file=args.students,
        school_file=args.schools,
        output_file=args.output,
    )

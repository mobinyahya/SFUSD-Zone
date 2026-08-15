"""Script to filter student school/program choices by distance or by top-N closest schools.

- Reads students and schools CSVs from config (YAML).
- For each student, removes from r1/r2/r3_* lists any school/program not within X miles (if config.distance=X),
  or not in the top Y closest schools (if config.number=Y).
- Saves the modified students DataFrame to config.output_csv.

Strictly follows project coding standards (see INSTRUCTIONS.md).
"""

import ast
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig


def haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Calculates the haversine distance between two points or arrays of points in miles.

    Args:
        lat1: Latitude(s) of point 1.
        lon1: Longitude(s) of point 1.
        lat2: Latitude(s) of point 2.
        lon2: Longitude(s) of point 2.

    Returns:
        np.ndarray: Distance(s) in miles.
    """
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def load_config(config_path: Path) -> dict[str, Any]:
    """Loads YAML config file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def _parse_choice_list(value: Any, row_index: Any, column: str) -> list:
    """Parse one choice-list cell and report malformed values precisely."""
    if isinstance(value, list):
        return value
    if value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    ):
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Row {row_index}, column {column} is not a valid list: {value!r}"
            ) from exc
        if isinstance(parsed, list):
            return parsed
    raise ValueError(
        f"Row {row_index}, column {column} must contain a list, got {value!r}"
    )


def _school_id_key(value: Any) -> Any:
    """Normalize numeric school IDs for matching without changing output lists."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value.strip()
    return value


def filter_student_choices(
    df_students: pd.DataFrame,
    df_schools: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    """Filters student school/program choices by distance or top-N closest schools.

    Args:
        df_students: Students DataFrame.
        df_schools: Schools DataFrame.
        config: Configuration dictionary or DictConfig.

    Returns:
        pd.DataFrame: Modified students DataFrame.
    """
    # Precompute all school distances for all students
    student_lats = df_students["latitude"].astype(float)
    student_lons = df_students["longitude"].astype(float)
    school_ids = df_schools["school_id"].values
    school_keys = [_school_id_key(school_id) for school_id in school_ids]
    school_lats = df_schools["lat"].values
    school_lons = df_schools["lon"].values
    n_students = len(df_students)
    n_schools = len(school_ids)
    # shape: (n_students, n_schools)
    dists = haversine(
        np.repeat(student_lats.values[:, None], n_schools, axis=1),
        np.repeat(student_lons.values[:, None], n_schools, axis=1),
        np.tile(school_lats, (n_students, 1)),
        np.tile(school_lons, (n_students, 1)),
    )

    # For each student, get mapping: normalized school_id -> distance
    student_school_dist = [
        {school_key: dists[i, j] for j, school_key in enumerate(school_keys)}
        for i in range(n_students)
    ]

    # List of all r*_ columns with string-encoded lists to filter in sync
    round_suffixes = [
        "ranked_idschool",
        "listed_ranks",
        "programs",
        "randomnumber",
        "cohortstring",
    ]
    distance_limit = (
        config.get("distance")
        if isinstance(config, dict)
        else getattr(config, "distance", None)
    )
    number_limit = (
        config.get("number")
        if isinstance(config, dict)
        else getattr(config, "number", None)
    )

    # For each round, filter all relevant columns in sync
    for r in [1, 2, 3]:
        # Build list of columns to filter for this round
        col_names = [f"r{r}_{suffix}" for suffix in round_suffixes]
        present_cols = [col for col in col_names if col in df_students.columns]
        if not present_cols:
            continue

        school_col = f"r{r}_ranked_idschool"
        program_col = f"r{r}_programs"
        if school_col not in present_cols or program_col not in present_cols:
            raise ValueError(
                f"Round {r} must include both {school_col} and {program_col}."
            )

        def filter_row(row: pd.Series, student_pos: int) -> dict[str, str]:
            lists = {
                col: _parse_choice_list(row[col], row.name, col)
                for col in present_cols
            }
            school_list = lists[school_col]
            program_list = lists[program_col]
            choice_count = len(school_list)

            if len(program_list) != choice_count:
                raise ValueError(
                    f"Row {row.name}, round {r} has {choice_count} schools but "
                    f"{len(program_list)} programs."
                )
            for col, values in lists.items():
                if col in {school_col, program_col} or not values:
                    continue
                if len(values) != choice_count:
                    raise ValueError(
                        f"Row {row.name}, column {col} has {len(values)} values; "
                        f"expected {choice_count} to match {school_col}."
                    )

            distances = student_school_dist[student_pos]
            if distance_limit is not None:
                max_dist = float(distance_limit)
                keep = [
                    i
                    for i, sid in enumerate(school_list)
                    if program_list[i] != "GE"
                    or (
                        _school_id_key(sid) in distances
                        and distances[_school_id_key(sid)] <= max_dist
                    )
                ]

            elif number_limit is not None:
                top_n = int(number_limit)
                sorted_schools = sorted(distances.items(), key=lambda item: item[1])
                top_school_ids = {sid for sid, _ in sorted_schools[:top_n]}
                keep = [
                    i
                    for i, sid in enumerate(school_list)
                    if _school_id_key(sid) in top_school_ids
                    or program_list[i] != "GE"
                ]

            else:
                keep = list(range(choice_count))

            # Empty ancillary lists mean the source has no metadata for that
            # round. Keep them empty; populated lists remain index-aligned.
            return {
                col: str([values[i] for i in keep] if values else [])
                for col, values in lists.items()
            }

        filtered_df = pd.DataFrame(
            [
                filter_row(row, student_pos)
                for student_pos, (_, row) in enumerate(df_students.iterrows())
            ],
            index=df_students.index,
            columns=present_cols,
        )
        for col in present_cols:
            df_students[col] = filtered_df[col]

    return df_students


def load_estimates(
    file_path: str, number_of_programs: int = 10
) -> pd.DataFrame:
    """Load estimate data from a CSV file into a DataFrame.

    Args:
        file_path: CSV path to the estimates table.
        number_of_programs: Number of top programs to keep per student.

    Returns:
        DataFrame with ranked program lists and derived columns.
    """
    df = pd.read_csv(file_path)

    # if the first column has no name, rename it to studentno
    if df.columns[0].strip() == "":
        df = df.rename(columns={df.columns[0]: "studentno"})

    # Check if unnamed first column was read as strict "Unnamed: 0" or similar
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "studentno"})

    # extract student number from studentno column, if format is <year>-<studentno>
    if (
        not df.empty
        and isinstance(df["studentno"].iloc[0], str)
        and "-" in df["studentno"].iloc[0]
    ):
        df["studentno"] = df["studentno"].str.split("-").str[1]

    # to numpy.int64
    df["studentno"] = df["studentno"].astype("int64")

    # Exclude studentno from columns to consider for ranking
    cols_to_consider = [c for c in df.columns if c != "studentno"]

    # Select top N programs based on values (assumed to be utilities/probabilities)
    df["selected_programs"] = df[cols_to_consider].apply(
        lambda row: row.nlargest(number_of_programs).index.tolist(), axis=1
    )

    # school numbers are the first part of the column names (e.g., "801-GE-KG")
    df["r1_ranked_idschool"] = df["selected_programs"].apply(
        lambda programs: str(
            [int(str(program).split("-", 1)[0]) for program in programs]
        )
    )
    df["r1_programs"] = df["selected_programs"].apply(
        lambda x: str([i.split("-")[1] for i in x])
    )
    df["grade"] = df["selected_programs"].apply(
        lambda x: [i.split("-")[2] for i in x]
    )
    # df grade is the first element of the list since all choices are for the same grade
    df["grade"] = df["grade"].apply(lambda x: str(x[0]) if len(x) > 0 else "")

    return df


def merge_students_with_estimates(
    df_students: pd.DataFrame, df_estimates: pd.DataFrame
) -> pd.DataFrame:
    """Merge students with estimates on `studentno`.

    Args:
        df_students: Base students DataFrame.
        df_estimates: Estimates DataFrame containing ranked programs.

    Returns:
        Students DataFrame with estimate-derived columns merged in.
    """
    # remove the columns in df_estimates that are not needed
    cols_to_keep = ["studentno", "r1_ranked_idschool", "r1_programs", "grade"]
    df_estimates = df_estimates[cols_to_keep]

    # remove the columns in df_students that are in df_estimates except studentno
    # This ensures we overwrite the student choices with the estimates
    common_columns = set(df_students.columns).intersection(
        set(df_estimates.columns)
    ) - {"studentno"}
    df_students = df_students.drop(columns=common_columns, errors="ignore")

    # Estimate rankings have no corresponding historical rank, lottery, or
    # cohort metadata. Retaining those old per-choice lists would associate
    # values with the wrong schools, so preserve the columns as explicitly
    # empty where downstream readers expect them.
    for col in ["r1_listed_ranks", "r1_randomnumber", "r1_cohortstring"]:
        if col in df_students.columns:
            df_students[col] = "[]"

    # merge on studentno
    df_merged = pd.merge(df_students, df_estimates, on="studentno", how="left")

    # Fill NaN for students who didn't have estimates? Or perhaps we should keep inner join?
    # The original loader used left join. Let's stick to that.

    return df_merged


def main(cfg: DictConfig) -> None:
    """Main function to filter student choices and return output using Hydra config.

    Args:
        cfg: Hydra DictConfig object.

    Returns:
        pd.DataFrame: Filtered students DataFrame.
    """
    students_csv = Path(cfg.data.students_csv)
    schools_csv = Path(cfg.data.schools_csv)

    df_students = pd.read_csv(students_csv)
    # Ensure KG only as per original logic often seen
    if "grade" in df_students.columns:
        df_students = df_students[df_students["grade"] == "KG"].reset_index(
            drop=True
        )

    df_schools = pd.read_csv(schools_csv)

    # --- Load and Merge Estimates if provided ---
    if "estimates_csv" in cfg.data and cfg.data.estimates_csv:
        print(f"Loading estimates from: {cfg.data.estimates_csv}")
        df_estimates = load_estimates(cfg.data.estimates_csv)
        df_students = merge_students_with_estimates(df_students, df_estimates)

    df_students = filter_student_choices(df_students, df_schools, cfg)

    output_path = Path(cfg.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_students.to_csv(output_path, index=False)


@hydra.main(
    version_base=None,
    config_path="configs/custom_configs",
    config_name="distance_filter",
)
def run_hydra(cfg: DictConfig) -> None:
    """Hydra entry point."""
    main(cfg)


if __name__ == "__main__":
    run_hydra()

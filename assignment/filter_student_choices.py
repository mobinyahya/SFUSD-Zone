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

    # For each student, get mapping: school_id -> distance
    school_id_to_idx = {sid: i for i, sid in enumerate(school_ids)}
    student_school_dist = [
        {sid: dists[i, j] for j, sid in enumerate(school_ids)}
        for i in range(n_students)
    ]

    # List of all r*_ columns with string-encoded lists to filter in sync
    round_cols = [
        ("ranked_idschool", True),
        ("listed_ranks", True),
        ("programs", True),
        ("randomnumber", True),
        ("cohortstring", True),
    ]
    # For each round, filter all relevant columns in sync
    for r in [1, 2, 3]:
        # Build list of columns to filter for this round
        col_names = [f"r{r}_{suffix}" for suffix, _ in round_cols]
        present_cols = [col for col in col_names if col in df_students.columns]
        if not present_cols:
            continue

        def filter_row(row, student_idx):
            # Parse all columns as lists (if missing, fill with empty list)
            lists = []
            for col in col_names:
                val = row[col] if col in row else "[]"
                try:
                    parsed = ast.literal_eval(val)
                except Exception:
                    parsed = []
                lists.append(parsed)

            # Identify which list corresponds to schools (index 0) and programs (index 2)
            # These indices match the order in col_names which follows round_cols
            # round_cols order: ranked_idschool, listed_ranks, programs, randomnumber, cohortstring
            school_list = lists[0]
            program_list = lists[2]

            n = len(school_list)
            lists = [
                lst if isinstance(lst, list) and len(lst) == n else [None] * n
                for lst in lists
            ]

            if config.distance:
                max_dist = float(config.distance)
                keep = [
                    i
                    for i, sid in enumerate(school_list)
                    if sid in school_id_to_idx
                    and student_school_dist[student_idx][sid] <= max_dist
                    or program_list[i] != "GE"
                ]

            elif config.number:
                top_n = int(config.number)
                sorted_schools = sorted(
                    student_school_dist[student_idx].items(), key=lambda x: x[1]
                )
                top_school_ids = set([sid for sid, _ in sorted_schools[:top_n]])
                keep = [
                    i
                    for i, sid in enumerate(school_list)
                    if sid in top_school_ids or program_list[i] != "GE"
                ]

            else:
                keep = list(range(n))
            filtered = [[lst[i] for i in keep] for lst in lists]

            return {col: str(filtered[j]) for j, col in enumerate(col_names)}

        filtered_df = df_students.apply(
            lambda row: filter_row(row, row.name), axis=1, result_type="expand"
        )
        for col in filtered_df.columns:
            df_students[col] = filtered_df[col]
        # Print how many students have non-empty lists after filtering

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
        lambda x: str([i.split("-")[0] for i in x])
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

    # Also drop r1_listed_ranks if present, as estimates don't provide it typically
    if "r1_listed_ranks" in df_students.columns:
        df_students = df_students.drop(columns=["r1_listed_ranks"])

    # merge on studentno
    df_merged = pd.merge(df_students, df_estimates, on="studentno", how="left")

    # Fill NaN for students who didn't have estimates? Or perhaps we should keep inner join?
    # The original loader used left join. Let's stick to that.

    print(df_merged.head())

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

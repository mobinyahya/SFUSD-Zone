"""Export demographic summary to Excel.

This script reads student and program data across multiple years from a config
file and generates an Excel file with two sheets: student demographics (ethnicity
counts and totals) and program seats (capacity by program type and totals).

Usage:
    python scripts/export_demographics_excel.py \
        --config configs/demographics_export_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# Add project root to path (must be before student_assignment imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


from student_assignment.utils.plotting import (
    apply_plot_style,
    get_color_palette,
    save_figure,
)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration values.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_student_data(student_path: str) -> pd.DataFrame:
    """Load student data from CSV file.

    Args:
        student_path: Path to the student CSV file.

    Returns:
        DataFrame with student data.
    """
    return pd.read_csv(student_path)


def load_program_data(program_path: str) -> pd.DataFrame:
    """Load program data from CSV file.

    Args:
        program_path: Path to the program CSV file.

    Returns:
        DataFrame with program data.
    """
    return pd.read_csv(program_path)


def compute_student_demographics(df_students: pd.DataFrame) -> pd.Series:
    """Compute ethnicity counts and total students.

    Args:
        df_students: DataFrame containing student data with 'resolved_ethnicity'.

    Returns:
        Series with ethnicity counts, including 'Total Students'.
    """
    # Filter for enrolled students only
    df_enrolled = df_students[df_students["enrolled_idschool"].notna()]

    ethnicity_counts = (
        df_enrolled["resolved_ethnicity"].fillna("Unknown").value_counts().sort_index()
    )
    total_students = ethnicity_counts.sum()
    ethnicity_counts["Total Students"] = total_students
    return ethnicity_counts


def compute_program_seats(df_programs: pd.DataFrame) -> pd.Series:
    """Compute seats by program type and total seats.

    Args:
        df_programs: DataFrame containing program data with 'program_type' and
            'capacity'.

    Returns:
        Series with capacity sums by program type, including 'Total Seats'.
    """
    cols = ["capacity", "r2_capacity"]
    for col in cols:
        if col not in df_programs.columns:
            df_programs[col] = 0

    grouped = df_programs.groupby("program_type")[cols].sum().sort_index()

    # Calculate totals
    total_row = grouped.sum()
    total_row.name = "Total Seats"
    grouped = pd.concat([grouped, total_row.to_frame().T])

    # Stack to create a MultiIndex Series (program_type, metric)
    return grouped.stack()


def plot_dataframe_trends(
    df: pd.DataFrame, title: str, output_path: Path, ylabel: str = "Count"
) -> None:
    """Generate and save a trend plot from a DataFrame.

    Args:
        df: DataFrame with labels (years) as index and categories as columns.
        title: Title for the plot.
        output_path: Path to save the PNG file.
        ylabel: Label for the Y-axis.
    """
    if df.empty:
        return

    # Apply the centralized style
    apply_plot_style()

    # Create figure
    plt.figure(figsize=(12, 7))

    # Exclude 'Total' columns for the main trend plot to avoid scale issues
    cols_to_plot = [
        c for c in df.columns if "Total" not in str(c) and "total" not in str(c)
    ]
    df_plot = df[cols_to_plot]

    # Use line plot with markers for trends
    palette = get_color_palette(len(cols_to_plot))
    sns.lineplot(data=df_plot, markers=True, dashes=False, palette=palette)

    plt.title(title)
    plt.xlabel("Year/Run")
    plt.ylabel(ylabel)
    plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)

    # Save the figure
    save_figure(output_path)
    print(f"Plot saved to {output_path}")


def plot_totals(
    df: pd.DataFrame, title: str, output_path: Path, ylabel: str = "Count"
) -> None:
    """Plot the 'Total' columns from a DataFrame.

    Args:
        df: DataFrame containing 'Total' columns.
        title: Title for the plot.
        output_path: Path to save the PNG file.
        ylabel: Label for the Y-axis.
    """
    total_cols = [c for c in df.columns if "Total" in str(c) or "total" in str(c)]
    if not total_cols:
        return

    apply_plot_style()
    plt.figure(figsize=(10, 6))

    palette = get_color_palette(len(total_cols))
    sns.lineplot(data=df[total_cols], markers=True, dashes=False, palette=palette)

    plt.title(title)
    plt.xlabel("Year/Run")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    save_figure(output_path)
    print(f"Total plot saved to {output_path}")


def plot_capacity_difference(
    df_r1: pd.DataFrame, df_r2: pd.DataFrame, output_path: Path
) -> None:
    """Plot the difference (capacity - r2_capacity) between R1 and R2.

    Args:
        df_r1: DataFrame with R1 capacities.
        df_r2: DataFrame with R2 capacities.
        output_path: Path to save the PNG file.
    """
    # Calculate difference for all common categories, excluding Totals for the first plot
    # Actually, the user might just want the Total difference.
    # Let's plot both categorical differences and Total difference if they exist.

    common_cols = [c for c in df_r1.columns if c in df_r2.columns]
    if not common_cols:
        return

    apply_plot_style()
    plt.figure(figsize=(12, 7))

    df_diff = df_r1[common_cols] - df_r2[common_cols]

    # Categorical difference plot (exclude Totals)
    categorical_cols = [
        c for c in common_cols if "Total" not in str(c) and "total" not in str(c)
    ]
    if categorical_cols:
        palette = get_color_palette(len(categorical_cols))
        sns.lineplot(
            data=df_diff[categorical_cols],
            markers=True,
            dashes=False,
            palette=palette,
        )

        plt.title("Capacity Difference Trends (R1 - R2) by Category")
        plt.xlabel("Year/Run")
        plt.ylabel("Difference (Seats)")
        plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45)
        save_figure(output_path)
        print(f"Capacity difference plot saved to {output_path}")

    # Total difference plot
    if "Total Seats" in common_cols:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df_diff["Total Seats"],
            markers=True,
            dashes=False,
            color="crimson",
        )
        plt.title("Total Capacity Difference Trend (R1 - R2)")
        plt.xlabel("Year/Run")
        plt.ylabel("Difference (Total Seats)")
        plt.xticks(rotation=45)
        total_output_path = output_path.with_name(
            output_path.stem + "_total" + output_path.suffix
        )
        save_figure(total_output_path)
        print(f"Total capacity difference plot saved to {total_output_path}")


def plot_r1_r2_comparison(
    df_r1: pd.DataFrame, df_r2: pd.DataFrame, output_path: Path
) -> None:
    """Compare Total Capacity between R1 and R2.

    Args:
        df_r1: DataFrame with R1 capacities.
        df_r2: DataFrame with R2 capacities.
        output_path: Path to save the PNG file.
    """
    if "Total Seats" not in df_r1.columns or "Total Seats" not in df_r2.columns:
        return

    apply_plot_style()
    plt.figure(figsize=(10, 6))

    df_comp = pd.DataFrame(
        {
            "Total Capacity (R1)": df_r1["Total Seats"],
            "Total Capacity (R2)": df_r2["Total Seats"],
        }
    )

    palette = get_color_palette(2)
    sns.lineplot(data=df_comp, markers=True, dashes=False, palette=palette)

    plt.title("Total Capacity Comparison: R1 vs R2")
    plt.xlabel("Year/Run")
    plt.ylabel("Total Seats")
    plt.xticks(rotation=45)
    save_figure(output_path)
    print(f"Capacity comparison plot saved to {output_path}")


def main() -> None:
    """Main entry point for demographics Excel export."""
    parser = argparse.ArgumentParser(description="Export demographic summary to Excel")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/demographics_export_config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)

    output_dir = Path(config["output_dir"])
    runs = config["runs"]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data for each year
    student_demos = []
    program_seats = []

    for run in runs:
        label = run["label"]
        student_path = run["student_data"]
        program_path = run["program_data"]

        # Load data
        df_students = load_student_data(student_path)
        df_programs = load_program_data(program_path)

        # Compute demographics
        demo_series = compute_student_demographics(df_students)
        demo_series.name = label
        student_demos.append(demo_series)

        # Compute seats
        seats_series = compute_program_seats(df_programs)
        seats_series.name = label
        program_seats.append(seats_series)

    # Combine into DataFrames
    df_student_demos = pd.DataFrame(student_demos).fillna(0).astype(int)
    # Create DataFrame from MultiIndex Series list
    df_program_seats_raw = pd.DataFrame(program_seats).fillna(0).astype(int)

    # Split into R1 and R2
    df_r1 = df_program_seats_raw.xs("capacity", level=1, axis=1).sort_index(axis=1)
    df_r2 = df_program_seats_raw.xs("r2_capacity", level=1, axis=1).sort_index(axis=1)

    # Write to Excel or CSV
    excel_path = output_dir / "demographics_summary.xlsx"
    csv_students_path = output_dir / "student_demographics.csv"
    csv_seats_r1_path = output_dir / "program_seats_r1.csv"
    csv_seats_r2_path = output_dir / "program_seats_r2.csv"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_student_demos.to_excel(writer, sheet_name="Student Demographics")
            df_r1.to_excel(writer, sheet_name="Program Seats (R1)")
            df_r2.to_excel(writer, sheet_name="Program Seats (R2)")
        print(f"Demographics summary exported to {excel_path}")
    except ImportError:
        # Fallback to separate CSVs if openpyxl not available
        df_student_demos.to_csv(csv_students_path)
        df_r1.to_csv(csv_seats_r1_path)
        df_r2.to_csv(csv_seats_r2_path)
        print(
            f"openpyxl not available. Student demographics exported to {csv_students_path}"
        )
        print(f"Program seats (R1) exported to {csv_seats_r1_path}")
        print(f"Program seats (R2) exported to {csv_seats_r2_path}")

    # Generate plots
    plot_dataframe_trends(
        df_student_demos,
        "Student Demographics Trends",
        output_dir / "student_demographics_trends.png",
        ylabel="Number of Students",
    )
    plot_dataframe_trends(
        df_r1,
        "Program Seats Trends (R1)",
        output_dir / "program_seats_r1_trends.png",
        ylabel="Seat Capacity",
    )
    plot_dataframe_trends(
        df_r2,
        "Program Seats Trends (R2)",
        output_dir / "program_seats_r2_trends.png",
        ylabel="Seat Capacity",
    )

    # Plot Totals
    plot_totals(
        df_student_demos,
        "Total Enrolled Students Trend",
        output_dir / "total_students_trend.png",
        ylabel="Total Students",
    )
    plot_totals(
        df_r1,
        "Total Program Seats Trend (R1)",
        output_dir / "total_seats_r1_trend.png",
        ylabel="Total Seats",
    )

    # Plot R1 vs R2 comparison
    plot_r1_r2_comparison(df_r1, df_r2, output_dir / "capacity_comparison_r1_r2.png")

    # Plot R1 - R2 difference
    plot_capacity_difference(df_r1, df_r2, output_dir / "capacity_difference_r1_r2.png")


if __name__ == "__main__":
    main()

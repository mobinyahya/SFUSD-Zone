"""
Script to filter program files removing special programs.

Usage:
    python scripts/preprocessing/filter_programs.py \
        [--data-dir /share/data/school_choice/Data] \
        [--output-dir local-data/program_filter]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path to import constants
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

try:
    from student_assignment.definitions.constants import SPECIAL_PROGRAMS
except ImportError:
    SPECIAL_PROGRAMS = {"AF", "DA", "DT", "ED", "MM", "MS", "SA", "TC", "AO"}
    print(
        "Warning: Could not import SPECIAL_PROGRAMS from constants. Using fallback set."
    )

# Years to process (13-23)
# File format: programs_{YY}{YY+1}.csv (e.g. programs_1516.csv)
# For 1314 and 1415 missing files we fall back to 1516.


def main():
    parser = argparse.ArgumentParser(
        description="Remove special programs from yearly program CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/share/data/school_choice/Data"),
        help="SFUSD data root containing Cleaned/programs_<year>.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local-data/program_filter"),
        help="Directory for the filtered program CSVs",
    )
    args = parser.parse_args()

    cleaned_dir = args.data_dir / "Cleaned"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    years = range(13, 24)  # 13 to 23 (2324)

    for y in years:
        y1 = y
        y2 = y + 1
        filename = f"programs_{y1:02d}{y2:02d}.csv"
        input_path = cleaned_dir / filename

        # Fallback mechanism
        if not input_path.exists():
            print(f"File not found: {input_path}")
            if y < 15:  # Fallback for 1314, 1415
                fallback_filename = "programs_1516.csv"
                print(f"Falling back to {fallback_filename} for year {y1}-{y2}")
                input_path = cleaned_dir / fallback_filename
            else:
                print(f"Skipping year {y1}-{y2}")
                continue

        output_filename = f"programs_without_specialprogs_{y1:02d}{y2:02d}.csv"
        output_path = output_dir / output_filename

        print(f"Processing {input_path} -> {output_path}")

        df = pd.read_csv(input_path)

        # Filter logic
        if "program_type" in df.columns:
            initial_count = len(df)
            df_filtered = df[~df["program_type"].isin(SPECIAL_PROGRAMS)]
            final_count = len(df_filtered)
            print(f"  Removed {initial_count - final_count} special programs.")
            df_filtered.to_csv(output_path, index=False)
        else:
            print(
                f"  Warning: 'program_type' column missing in {input_path}. Copying as is."
            )
            df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

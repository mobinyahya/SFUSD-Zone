"""
Script to generate status quo config files for multiple years.

Usage:
    python scripts/generate_configs.py
"""

from pathlib import Path

import yaml

# Paths
BASE_CONFIG_PATH = Path("configs/erabasse.config.yaml")
OUTPUT_CONFIG_DIR = Path("configs/custom_configs")
OUTPUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Data Base Paths (relative to project root or absolute)
STUDENT_DATA_BASE = Path("local-data/student_filter")
PROGRAM_DATA_BASE = Path("local-data/program_filter")
SCHOOL_DATA_BASE = Path(
    "Cleaned"
)  # Within share/data/... relative to sfusd path in config


def main():
    # Load base config
    with open(BASE_CONFIG_PATH) as f:
        base_config = yaml.safe_load(f)

    years = range(13, 24)  # 13 to 22 (2223)

    for y in years:
        y1 = y
        y2 = y + 1
        year_str = f"{y1:02d}{y2:02d}"

        # Create a deep copy of config if needed, but simple dict copy might suffice for top level
        # For nested dicts, be careful.
        config = yaml.safe_load(yaml.dump(base_config))  # Deep copy/clean slate

        # Update fields
        config["year"] = y1

        # Helper to get absolute path
        cwd = Path.cwd()

        # Student Data: local-data/student_filter/student_{YY}{YY+1}_filtered.csv
        student_file = STUDENT_DATA_BASE / f"student_{year_str}_filtered.csv"
        config["paths"]["student-data"] = str(cwd / student_file)

        # Program Data: local-data/program_filter/programs_without_specialprogs_{YY}{YY+1}.csv
        program_file = (
            PROGRAM_DATA_BASE / f"programs_without_specialprogs_{year_str}.csv"
        )
        config["paths"]["program-data"] = str(cwd / program_file)

        # School Data: Cleaned/schools_rehauled_{YY}{YY+1}.csv
        # Fallback for 13, 14 -> 1516
        if y < 15:
            school_year_str = "1516"
        else:
            school_year_str = year_str

        # School data is likely in the sfusd path, so we keep it relative or check if we need absolute
        # (Since Base Config uses Cleaned/... relative to sfusd, we can keep using that if we point to sfusd correctly)
        # But we want to be safe. If Cleaned is INSIDE sfusd path, we can output "Cleaned/..."
        # The script fallback said "Cleaned/schools..."

        school_file_rel = f"Cleaned/schools_rehauled_{school_year_str}.csv"
        # We assume this is relative to sfusd path set in config.
        # But wait, config["paths"]["sfusd"] is /share/data/school_choice/Data/
        # And we want /share/data/school_choice/Data/Cleaned/schools_rehauled...
        # So "Cleaned/..." is correct IF we don't mess with it.
        config["paths"]["school-data"] = school_file_rel

        # Output Folder
        output_folder = (
            f"./local-data/local-runs/status_quo_runs/run_{year_str}/"
        )
        config["paths"]["assignment-folder"] = output_folder

        # Subconfigs - Only Status Quo
        config["subconfigs"] = ["status_quo"]

        # Save
        output_filename = f"status_quo_generated_{year_str}.yaml"
        output_path = OUTPUT_CONFIG_DIR / output_filename

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Generated {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert zone pickle files to CSV format for student-assignment simulation.

This script takes zone dictionaries from pickle files (mapping zone_id -> list of geounits)
and converts them to the CSV format expected by the student-assignment simulation.

Usage:
    python generate_zone_from_pickle.py --input path/to/zone.pkl --output path/to/zone.csv
    python generate_zone_from_pickle.py --input-dir Generated_Zones/Zones_6/FRL_Dev_0.20 --output-dir zones/

Example:
    python generate_zone_from_pickle.py \
        --input ~/Generated_Zones/Zones_6/FRL_Dev_0.20/cost_123.456.pkl \
        --output /share/data/school_choice/Data/assignment/zones/6zone-custom.csv
"""

import argparse
import csv
import pickle
from pathlib import Path


def load_zone_dict_from_pickle(pickle_path: Path) -> dict[int, list[int]]:
    """Load zone dictionary from a pickle file.

    Handles two formats:
    - Standard format: {zone_id: [geounit1, geounit2, ...]}
    - Inverted format: {geounit: zone_id} (auto-detected and converted)

    Args:
        pickle_path: Path to the pickle file containing the zone dictionary.

    Returns:
        Dictionary mapping zone_id to list of geounit IDs (block groups, blocks,
        or attendance areas).

    Raises:
        FileNotFoundError: If pickle file does not exist.
        ValueError: If pickle content is not a valid zone dictionary.
    """
    pickle_path = Path(pickle_path).expanduser().resolve()

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    with open(pickle_path, "rb") as file:
        raw_dict = pickle.load(file)

    if not isinstance(raw_dict, dict):
        raise ValueError(
            f"Expected dict, got {type(raw_dict).__name__}. "
            "Pickle should contain a zone mapping."
        )

    if not raw_dict:
        raise ValueError("Empty dictionary found in pickle file.")

    # Detect format by checking the first value
    first_value = next(iter(raw_dict.values()))

    if isinstance(first_value, (list, set, tuple)):
        # Standard format: {zone_id: [geounits]}
        zone_dict = {k: list(v) for k, v in raw_dict.items()}
    elif isinstance(first_value, (int, float)):
        # Inverted format: {geounit: zone_id} -> convert to {zone_id: [geounits]}
        zone_dict: dict[int, list[int]] = {}
        for geounit, zone_id in raw_dict.items():
            zone_id_int = int(zone_id)
            if zone_id_int not in zone_dict:
                zone_dict[zone_id_int] = []
            zone_dict[zone_id_int].append(geounit)
        print("  Converted from inverted format (geounit -> zone_id)")
    else:
        raise ValueError(
            f"Unexpected value type in dict: {type(first_value).__name__}. "
            "Expected list (zone->geounits) or int (geounit->zone)."
        )

    return zone_dict


def zone_dict_to_csv(
    zone_dict: dict[int, list[int]],
    output_path: Path,
) -> None:
    """Convert zone dictionary to CSV format expected by student-assignment.

    The CSV format has one row per zone, with comma-separated geounit IDs.

    Args:
        zone_dict: Dictionary mapping zone_id to list of geounit IDs.
        output_path: Path where the CSV file will be saved.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort zones by zone_id to ensure consistent ordering
    sorted_zone_ids = sorted(zone_dict.keys())

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for zone_id in sorted_zone_ids:
            geounits = zone_dict[zone_id]
            # Convert all geounits to strings for CSV writing
            writer.writerow([str(g) for g in geounits])

    print(f"Saved zone CSV to: {output_path}")
    print(f"  - Number of zones: {len(zone_dict)}")
    total_geounits = sum(len(v) for v in zone_dict.values())
    print(f"  - Total geounits: {total_geounits}")


def generate_policy_config_snippet(
    zone_name: str,
    zone_csv_path: Path,
    num_zones: int,
    building_blocks: str = "block_group",
) -> str:
    """Generate a YAML snippet for the policy config.

    Args:
        zone_name: Name identifier for the zone (e.g., "6zone-custom").
        zone_csv_path: Path to the zone CSV file.
        num_zones: Number of zones in the configuration.
        building_blocks: Type of building blocks ("block_group", "block", or
            "attendance_area").

    Returns:
        YAML configuration snippet as a string.
    """
    return f"""
# Add this to your policy config YAML (e.g., configs/policy_configs/custom_zones.yaml)
# Add this direct source to the executable run config:
# data:
#   scenario: legacy
#   overrides:
#     sources:
#       assignment.zones:
#         {zone_name}: {{path: {zone_csv_path}, classification: public}}

# Then create a policy config like:
# ---
# assignment-algorithm: DA
# ctip-options:
# - 1
#
# restrict-zone: true
# guard-rails: 1
# reserve-settings:
#   column: median_hh_income
#   thresholds: [95292]
#   lower_disadvantaged: true
#   citywide_only: false
#   reserve_fraction: [0.57, 0.43]
#
# policies:
# - {zone_name}
#
# priority-weights:
#   ctip: 8
#   sibling: 16
#   zone: 4
#
# sibling-access: true
# zone-building-blocks: {building_blocks}
# designate: true
# ties-options:
# - MTB
"""


def process_single_pickle(
    input_path: Path,
    output_path: Path | None = None,
    building_blocks: str = "block_group",
    verbose: bool = True,
) -> Path:
    """Process a single pickle file and generate zone CSV.

    Args:
        input_path: Path to the input pickle file.
        output_path: Optional output path. If None, generates path from input.
        building_blocks: Type of zone building blocks.
        verbose: Whether to print config snippet (default: True).

    Returns:
        Path to the generated CSV file.
    """
    input_path = Path(input_path).expanduser().resolve()

    if output_path is None:
        output_path = input_path.with_suffix(".csv")
    else:
        output_path = Path(output_path).expanduser().resolve()

    zone_dict = load_zone_dict_from_pickle(input_path)
    zone_dict_to_csv(zone_dict, output_path)

    # Generate config snippet only if verbose
    if verbose:
        zone_name = output_path.stem
        num_zones = len(zone_dict)
        config_snippet = generate_policy_config_snippet(
            zone_name, output_path, num_zones, building_blocks
        )
        print(config_snippet)

    return output_path


def process_directory(
    input_dir: Path,
    output_dir: Path,
    building_blocks: str = "block_group",
    pattern: str = "*",
) -> list[Path]:
    """Process all pickle files in a directory recursively.

    Args:
        input_dir: Directory containing pickle files.
        output_dir: Directory where CSV files will be saved.
        building_blocks: Type of zone building blocks.
        pattern: Glob pattern for finding files (default: "*" for all files).

    Returns:
        List of paths to generated CSV files.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all files matching the pattern (default: all files, not just .pkl)
    all_files = sorted(input_dir.rglob(pattern))
    # Filter to only regular files (not directories)
    candidate_files = [f for f in all_files if f.is_file()]

    if not candidate_files:
        print(f"No files found in {input_dir} matching pattern '{pattern}'")
        return []

    print(f"Found {len(candidate_files)} files to process")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for file_path in candidate_files:
        # Preserve directory structure in output
        relative_path = file_path.relative_to(input_dir)
        relative_path.with_suffix(".csv").name
        # Flatten the structure or preserve it based on preference
        # Here we flatten by using just the filename with parent dirs as prefix
        flat_name = "_".join(relative_path.parent.parts + (relative_path.name,))
        csv_path = output_dir / f"{flat_name}.csv"

        try:
            print(f"Processing: {file_path}")
            process_single_pickle(
                file_path, csv_path, building_blocks, verbose=False
            )
            generated_files.append(csv_path)
        except Exception as exc:
            print(f"  Skipping {file_path}: {exc}")

    print(f"\nSuccessfully generated {len(generated_files)} CSV files")
    return generated_files


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert zone pickle files to CSV format for simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a single pickle file to convert.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the CSV file (optional, defaults to input path with .csv).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing pickle files to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for CSV files (required if --input-dir is used).",
    )
    parser.add_argument(
        "--building-blocks",
        type=str,
        choices=["block_group", "block", "attendance_area"],
        default="block_group",
        help="Type of zone building blocks (default: block_group).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern for finding files (default: '*' for all files).",
    )

    args = parser.parse_args()

    if args.input:
        process_single_pickle(args.input, args.output, args.building_blocks)
    elif args.input_dir:
        if not args.output_dir:
            parser.error("--output-dir is required when using --input-dir")
        process_directory(
            args.input_dir, args.output_dir, args.building_blocks, args.pattern
        )
    else:
        parser.error("Either --input or --input-dir must be specified")


if __name__ == "__main__":
    main()

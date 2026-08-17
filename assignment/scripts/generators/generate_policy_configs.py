#!/usr/bin/env python3
"""Generate policy config YAML files for each zone in the zones folder.

This script creates individual policy config files for each zone CSV file,
allowing batch simulation runs across all zone configurations.

Usage:
    python scripts/generate_policy_configs.py
    python scripts/generate_policy_configs.py \
        --zones-dir /share/data/school_choice/Data/assignment/zones
    python scripts/generate_policy_configs.py --template small_zones+no_reserves

Example:
    python scripts/generate_policy_configs.py \
        --zones-dir /share/data/school_choice/Data/assignment/zones \
        --output-dir ./configs/policy_configs/generated \
        --template small_zones+no_reserves
"""

import argparse
from pathlib import Path

import yaml


def get_zone_names_from_directory(zones_dir: Path) -> list[str]:
    """Get list of zone names from CSV files in directory.

    Args:
        zones_dir: Path to directory containing zone CSV files.

    Returns:
        List of zone names (without .csv extension).
    """
    zones_dir = Path(zones_dir).expanduser().resolve()

    if not zones_dir.exists():
        raise FileNotFoundError(f"Zones directory not found: {zones_dir}")

    zone_files = sorted(zones_dir.glob("*.csv"))
    zone_names = [zf.stem for zf in zone_files]

    return zone_names


def generate_policy_config(
    zone_name: str,
    template: str = "small_zones+no_reserves",
    guard_rails: int = -1,
    with_reserves: bool = False,
    reserve_frl: float | None = None,
) -> dict:
    """Generate a policy config dictionary for a given zone.

    Args:
        zone_name: Name of the zone (must match key in zone-files).
        template: Template name for naming convention.
        guard_rails: Guard rails setting (-1 for no reserves, 1 for reserves).
        with_reserves: Whether to include reserve settings.
        reserve_frl: Optional FRL threshold for reserves (0.5, 0.6, etc.).

    Returns:
        Dictionary containing the policy configuration.
    """
    config = {
        "assignment-algorithm": "DA",
        "ctip-options": [1],
        "restrict-zone": True,
        "guard-rails": guard_rails,
        "non_designation_boost": 128,
        "policies": [zone_name],
        "priority-weights": {
            "ctip": 8,
            "sibling": 16,
            "zone": 4,
            "language-programs": {
                "lp-sibling": 16,
                "lp": 8,
                "sibling": 4,
                "ctip": 2,
            },
        },
        "sibling-access": True,
        "zone-building-blocks": "block_group",
        "designate": True,
        "ties-options": ["MTB"],
    }

    if with_reserves:
        config["guard-rails"] = 1
        config["soft_reserve_boost"] = 64

        if reserve_frl is not None:
            reserve_settings = {
                "column": "freelunch_prob",
                "thresholds": f"percentile:{int(reserve_frl * 100)}",
                "lower_disadvantaged": True,
                "reserve_fraction": [reserve_frl, 1.0 - reserve_frl],
            }
        else:
            reserve_settings = {
                "column": "median_hh_income",
                "thresholds": [95292],
                "lower_disadvantaged": True,
                "citywide_only": False,
                "reserve_fraction": [0.57, 0.43],
            }

        config["reserve-settings"] = reserve_settings

    return config


def write_policy_config(
    config: dict,
    output_path: Path,
) -> None:
    """Write policy config to YAML file.

    Args:
        config: Policy configuration dictionary.
        output_path: Path where YAML file will be saved.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)


def generate_all_policy_configs(
    zones_dir: Path,
    output_dir: Path,
    variants: list[str] | None = None,
) -> list[Path]:
    """Generate policy config files for all zones.

    Args:
        zones_dir: Directory containing zone CSV files.
        output_dir: Directory where policy config YAMLs will be saved.
        variants: List of variants to generate. Options:
            - "no_reserves": No reserves (guard-rails: -1)
            - "reserves": With reserves
            - "reserves_05frl": With reserves, 50% FRL target
            - "reserves_06frl": With reserves, 60% FRL target

    Returns:
        List of paths to generated config files.
    """
    if variants is None:
        variants = ["no_reserves", "reserves"]

    zone_names = get_zone_names_from_directory(zones_dir)
    print(f"Found {len(zone_names)} zones")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for zone_name in zone_names:
        for variant in variants:
            # Determine config parameters based on variant
            if variant == "no_reserves":
                config = generate_policy_config(
                    zone_name=zone_name,
                    guard_rails=-1,
                    with_reserves=False,
                )
                suffix = "no_reserves"
            elif variant == "reserves":
                config = generate_policy_config(
                    zone_name=zone_name,
                    guard_rails=1,
                    with_reserves=True,
                )
                suffix = "reserves"
            elif variant == "reserves_05frl":
                config = generate_policy_config(
                    zone_name=zone_name,
                    guard_rails=1,
                    with_reserves=True,
                    reserve_frl=0.5,
                )
                suffix = "reserves_05frl"
            elif variant == "reserves_06frl":
                config = generate_policy_config(
                    zone_name=zone_name,
                    guard_rails=1,
                    with_reserves=True,
                    reserve_frl=0.6,
                )
                suffix = "reserves_06frl"
            else:
                print(f"Unknown variant: {variant}, skipping")
                continue

            # Generate output filename
            output_filename = f"{zone_name}+{suffix}.yaml"
            output_path = output_dir / output_filename

            write_policy_config(config, output_path)
            generated_files.append(output_path)

    print(
        f"Generated {len(generated_files)} policy config files in {output_dir}"
    )
    return generated_files


def update_base_config_subconfigs(
    base_config_path: Path,
    generated_configs: list[Path],
    output_path: Path | None = None,
) -> None:
    """Update base config with list of generated subconfigs.

    Args:
        base_config_path: Path to base config YAML.
        generated_configs: List of paths to generated config files.
        output_path: Optional output path. If None, prints to stdout.
    """
    # Extract subconfig names (filename without .yaml extension), preserving
    # the "generated/" prefix for configs living in that subdirectory.
    subconfig_names = [
        f"generated/{gc.stem}" if gc.parent.name == "generated" else gc.stem
        for gc in generated_configs
    ]

    subconfigs_yaml = "subconfigs:\n"
    for name in sorted(subconfig_names):
        subconfigs_yaml += f"- {name}\n"

    if output_path:
        # Insert new subconfigs after their matching +reserves counterpart,
        # preserving the existing ordering in the file.
        lines = base_config_path.read_text().splitlines(keepends=True)
        new_set = set(subconfig_names)
        out = []
        for line in lines:
            out.append(line)
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            base = stripped[2:]  # strip leading "- "
            # For each +reserves line, check if a _05frl / _06frl variant exists
            for suffix in ("reserves_05frl", "reserves_06frl"):
                candidate = base.replace("+reserves", f"+{suffix}")
                if candidate in new_set:
                    indent = " " * (len(line) - len(line.lstrip()))
                    out.append(f"{indent}- {candidate}\n")
                    new_set.discard(candidate)
        # Append any remaining new subconfigs at end of subconfigs block
        for name in sorted(new_set):
            out.append(f"- {name}\n")
        output_path.write_text("".join(out))
        print(f"Updated base config saved to: {output_path}")
    else:
        print("\n# Add these to your base config's subconfigs section:")
        print(subconfigs_yaml)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate policy config files for each zone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--zones-dir",
        type=Path,
        default=Path("/share/data/school_choice/Data/assignment/zones"),
        help="Directory containing zone CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./configs/policy_configs/generated"),
        help="Output directory for policy configs (default: ./configs/policy_configs/generated).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["no_reserves", "reserves", "reserves_05frl", "reserves_06frl"],
        default=["no_reserves", "reserves"],
        help="Policy variants to generate (default: no_reserves reserves).",
    )
    parser.add_argument(
        "--update-base-config",
        type=Path,
        help="Path to base config to update with new subconfigs.",
    )
    parser.add_argument(
        "--list-subconfigs",
        action="store_true",
        help="Print list of subconfig names for base config.",
    )

    args = parser.parse_args()

    # Generate policy configs
    generated_files = generate_all_policy_configs(
        zones_dir=args.zones_dir,
        output_dir=args.output_dir,
        variants=args.variants,
    )

    # Optionally update base config or list subconfigs
    if args.list_subconfigs or args.update_base_config:
        update_base_config_subconfigs(
            base_config_path=args.update_base_config
            or Path("configs/all_zones.yaml"),
            generated_configs=generated_files,
            output_path=args.update_base_config,
        )


if __name__ == "__main__":
    main()

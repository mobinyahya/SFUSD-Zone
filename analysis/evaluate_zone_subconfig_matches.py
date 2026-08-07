#!/usr/bin/env python3
"""Run and evaluate 25-iteration matches for selected zone subconfigs.

The small-zone policies use the selected 13-zone plan, the medium-zone
policies use the selected 6-zone plan, and distance/status-quo policies keep
the attendance-area ``Con1`` zones from the base assignment config. Complete
assignment outputs are reused, and each CSV column is the mean of
``eval_assignment_full()`` over one subconfig's 25 iterations.

Usage:
    uv run python analysis/evaluate_zone_subconfig_matches.py
    uv run python analysis/evaluate_zone_subconfig_matches.py --real-preferences
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.evaluate_status_quo_matches import (  # noqa: E402
    DEFAULT_BASE_CONFIG,
    DEFAULT_NEW_CTIP_PATH,
    build_simulation_config as build_policy_simulation_config,
    evaluate_policy,
    load_yaml,
    run_policy,
    save_simulation_config,
)

LOGGER = logging.getLogger(__name__)

SUBCONFIGS = (
    "small_zones+no_reserves",
    "small_zones+reserves",
    "small_zones+reserves_05frl",
    "small_zones+reserves_06frl",
    "medium_zones+no_reserves",
    "medium_zones+reserves",
    "medium_zones+reserves_05frl",
    "medium_zones+reserves_06frl",
    "distance_05_1_2+reserves",
    "distance_05_1_2+reserves_05frl",
    "distance_05_1_2+reserves_06frl",
    "status_quo",
    "status_quo_3",
    "status_quo_4",
    "status_quo+reserves",
    "status_quo+reserves_05frl",
    "status_quo+reserves_06frl",
    "distance_05_1_2+reserves_05frl_#3",
    "distance_05_1_2+reserves_05frl_#4" 
)
DEFAULT_POLICY_DIR = PROJECT_ROOT / "assignment/configs/policy_configs"
DEFAULT_ZONE_ROOT = Path("~/sfusd-local-data/zones/zone_backend")
DEFAULT_SMALL_ZONES = DEFAULT_ZONE_ROOT / "Zones_13-FRL_Dev_0.25-Objective_2500_BG.csv"
DEFAULT_MEDIUM_ZONES = DEFAULT_ZONE_ROOT / "Zones_6-FRL_Dev_0.10-Objective_1430_BG.csv"
DEFAULT_MATCHES_ROOT = PROJECT_ROOT / "analysis/matches/zone_subconfigs_25"
DEFAULT_REAL_MATCHES_ROOT = (
    PROJECT_ROOT / "analysis/matches/zone_subconfigs_real_preferences_25"
)
DEFAULT_REAL_STUDENTS = Path(
    "/share/data/school_choice/Data/Cleaned/student_2324.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--small-zones", type=Path, default=DEFAULT_SMALL_ZONES)
    parser.add_argument("--medium-zones", type=Path, default=DEFAULT_MEDIUM_ZONES)
    parser.add_argument(
        "--matches-root",
        type=Path,
        help="Output root; defaults to a separate root for each preference mode.",
    )
    parser.add_argument(
        "--real-preferences",
        action="store_true",
        help="Disable the utility model and use observed student preferences.",
    )
    parser.add_argument(
        "--real-student-data",
        type=Path,
        default=DEFAULT_REAL_STUDENTS,
        help="Student CSV used with --real-preferences.",
    )
    parser.add_argument(
        "--new-ctip-path",
        type=Path,
        default=DEFAULT_NEW_CTIP_PATH,
        help="Optional equity-block NPY used by the full evaluator.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the requested runs without matching.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_simulation_config(
    base_config: Mapping[str, Any],
    matches_root: Path,
    small_zones: Path,
    medium_zones: Path,
    *,
    real_student_data: Path | None = None,
) -> dict[str, Any]:
    """Set the requested subconfigs, outputs, and selected zone plans."""
    config = build_policy_simulation_config(base_config, list(SUBCONFIGS), matches_root)
    paths = copy.deepcopy(config.get("paths") or {})
    zone_files = copy.deepcopy(paths.get("zone-files") or {})
    zone_files["18zone_2"] = str(small_zones)
    zone_files["6zone-1"] = str(medium_zones)
    if real_student_data is not None:
        paths["student-data"] = str(real_student_data)
        paths["student-save"] = str(matches_root / "precomputed")
    paths["zone-files"] = zone_files
    config["paths"] = paths
    if real_student_data is not None:
        config["utility-model"] = {
            "designate-lp-for-all": False,
            "enable": False,
            "list-length": "0.8*round(real_length)",
        }
        config["random-seed"] = 2023
        config["r1-only"] = True
        config["remove-special-lps"] = True
        config["rounds-merged-options"] = [0]
    return config


def load_policies(policy_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the requested policy overlays in output-column order."""
    return {label: load_yaml(policy_dir / f"{label}.yaml") for label in SUBCONFIGS}


def write_metrics_csv(
    metrics: Mapping[str, pd.Series],
    matches_root: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = (
        matches_root / f"zone_subconfigs_25_eval_assignment_full_{timestamp}.csv"
    )
    frame = pd.DataFrame(metrics)
    frame.index.name = "metric"
    with output_path.open("x", encoding="utf-8", newline="") as output_file:
        frame.to_csv(output_file)
    return output_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    base_config_path = args.base_config.expanduser().resolve()
    policy_dir = args.policy_dir.expanduser().resolve()
    small_zones = args.small_zones.expanduser().resolve()
    medium_zones = args.medium_zones.expanduser().resolve()
    matches_root_arg = args.matches_root or (
        DEFAULT_REAL_MATCHES_ROOT if args.real_preferences else DEFAULT_MATCHES_ROOT
    )
    matches_root = matches_root_arg.expanduser().resolve()

    real_student_data = None
    if args.real_preferences:
        real_student_data = args.real_student_data.expanduser().resolve()
        if not real_student_data.is_file():
            raise FileNotFoundError(real_student_data)

    for zone_path in (small_zones, medium_zones):
        if not zone_path.is_file():
            raise FileNotFoundError(zone_path)

    policies = load_policies(policy_dir)
    config = build_simulation_config(
        load_yaml(base_config_path),
        matches_root,
        small_zones,
        medium_zones,
        real_student_data=real_student_data,
    )

    if args.dry_run:
        LOGGER.info("Small-zone plan: %s", small_zones)
        LOGGER.info("Medium-zone plan: %s", medium_zones)
        LOGGER.info(
            "Preferences: %s",
            "real" if args.real_preferences else "choice model",
        )
        for label in SUBCONFIGS:
            LOGGER.info("Would run %s", label)
        return 0

    matches_root.mkdir(parents=True, exist_ok=True)
    save_simulation_config(config, matches_root)

    new_ctip_path = args.new_ctip_path.expanduser().resolve()
    if not new_ctip_path.is_file():
        LOGGER.warning(
            "Equity-block file not found; ET metrics will use no blocks: %s",
            new_ctip_path,
        )
        new_ctip_path = None

    assignments_by_policy = {
        label: run_policy(config, label, policies[label], matches_root)
        for label in SUBCONFIGS
    }
    metrics = {}
    for label in SUBCONFIGS:
        metrics[label] = evaluate_policy(
            assignments_by_policy[label], config, new_ctip_path
        )
        LOGGER.info("Evaluated %s", label)

    output_path = write_metrics_csv(metrics, matches_root)
    LOGGER.info(
        "Wrote %d metrics for %d subconfigs to %s",
        len(pd.DataFrame(metrics)),
        len(metrics),
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

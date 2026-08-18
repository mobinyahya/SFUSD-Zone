#!/usr/bin/env python3
"""Run and evaluate 25-iteration matches for two status-quo policies.

By default this compares ``status_quo+reserves_06frl`` with ``status_quo``
using the assignment inputs from ``assignment/configs/kumar.config.yaml``.
Raw assignments and the metric CSV are written under
``analysis/matches/status_quo_policies_25``.

Complete assignment outputs are reused. An existing incomplete policy output
is never reset or overwritten. Each metric CSV column is the mean of
``eval_assignment_full()`` over that policy's 25 assignment iterations.

Usage:
    uv run python analysis/evaluate_status_quo_matches.py
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
import shutil
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loaders import load_scenario  # noqa: E402
from analysis.evaluate_soft_reserve_matches import (  # noqa: E402
    EvaluationTask,
    evaluate_assignment,
)
from assignment.student_assignment.market_generator.school_choice_market_generator import (  # noqa: E402
    MarketGenerator,
)

LOGGER = logging.getLogger(__name__)

ITERATION_START = 0
ITERATION_COUNT = 25
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "assignment/configs/kumar.config.yaml"
DEFAULT_POLICIES = (
    PROJECT_ROOT / "assignment/configs/policy_configs/status_quo+soft_reserves_06frl.yaml",
    PROJECT_ROOT / "assignment/configs/policy_configs/status_quo.yaml",
)
DEFAULT_MATCHES_ROOT = PROJECT_ROOT / "analysis/matches/status_quo_policies_25"
DEFAULT_NEW_CTIP_PATH = Path(
    "/soalnas/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/ETB_2024.npy"
)
ITERATION_PATTERN = re.compile(r"_iteration(\d+)\.csv$")


class PolicyConfigLoader:
    """Supply one policy overlay to ``MarketGenerator.simulate``."""

    def __init__(
        self,
        base_config: Mapping[str, Any],
        label: str,
        policy: Mapping[str, Any],
    ) -> None:
        self._original_config = copy.deepcopy(dict(base_config))
        self._config = copy.deepcopy(dict(base_config))
        self._pending = [(label, copy.deepcopy(dict(policy)))]

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def load_next_subconfig(self) -> bool:
        if not self._pending:
            return False
        label, policy = self._pending.pop(0)
        self._config = {**copy.deepcopy(self._original_config), **policy}
        self._config["subconfig-name"] = label
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--policies",
        type=Path,
        nargs="+",
        default=list(DEFAULT_POLICIES),
        help="Policy YAMLs to simulate and compare.",
    )
    parser.add_argument("--matches-root", type=Path, default=DEFAULT_MATCHES_ROOT)
    parser.add_argument(
        "--new-ctip-path",
        type=Path,
        default=DEFAULT_NEW_CTIP_PATH,
        help="Optional equity-block NPY used by the full evaluator.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_simulation_config(
    base_config: Mapping[str, Any],
    policy_labels: list[str],
    matches_root: Path,
) -> dict[str, Any]:
    """Apply this script's output and iteration settings to a base config."""
    config = copy.deepcopy(dict(base_config))
    config.pop("output_dir", None)
    config["iterations"] = {
        "start": ITERATION_START,
        "end": ITERATION_START + ITERATION_COUNT,
    }
    config["save-assignment"] = True
    config["subconfigs"] = policy_labels
    paths = copy.deepcopy(config.get("paths") or {})
    paths["assignment-folder"] = str(matches_root)
    config["paths"] = paths
    return config


def validate_assignment_output(
    matches_root: Path,
    label: str,
) -> list[Path]:
    """Return naturally ordered assignments or raise for incomplete output."""
    output_root = matches_root / label
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)

    paths_by_iteration: dict[int, Path] = {}
    unexpected: list[Path] = []
    for path in output_root.rglob("*.csv"):
        match = ITERATION_PATTERN.search(path.name)
        if match is None:
            unexpected.append(path)
            continue
        iteration = int(match.group(1))
        if iteration in paths_by_iteration:
            raise ValueError(
                f"{label} has multiple assignments for iteration {iteration}"
            )
        paths_by_iteration[iteration] = path

    expected = set(range(ITERATION_START, ITERATION_START + ITERATION_COUNT))
    actual = set(paths_by_iteration)
    if actual != expected or unexpected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} output is incomplete: found {len(paths_by_iteration)} "
            f"assignments, missing={missing}, extra={extra}, "
            f"unexpected_csvs={len(unexpected)}"
        )
    return [paths_by_iteration[index] for index in sorted(paths_by_iteration)]


def run_policy(
    base_config: Mapping[str, Any],
    label: str,
    policy: Mapping[str, Any],
    matches_root: Path,
) -> list[Path]:
    """Generate one policy's assignments without loading a personal config."""
    destination = matches_root / label
    if destination.exists():
        assignments = validate_assignment_output(matches_root, label)
        LOGGER.info("Reusing %d assignments for %s", len(assignments), label)
        return assignments

    run_config = copy.deepcopy(dict(base_config))
    run_config["subconfigs"] = [label]
    loader = PolicyConfigLoader(run_config, label, policy)

    market = MarketGenerator(
        assignment_path=str(matches_root),
        configurator=loader,
        write_config=False,
    )

    try:
        market.simulate()
        assignments = validate_assignment_output(matches_root, label)
        effective_config = {**run_config, **dict(policy), "subconfig-name": label}
        with (destination / "policy_config.generated.yaml").open(
            "x", encoding="utf-8"
        ) as output_file:
            yaml.safe_dump(effective_config, output_file, sort_keys=False)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    LOGGER.info("Generated %d assignments for %s", len(assignments), label)
    return assignments


def evaluation_tasks(
    assignments: list[Path],
    config: Mapping[str, Any],
    new_ctip_path: Path | None,
    *,
    first_round: bool = True,
) -> list[EvaluationTask]:
    data = copy.deepcopy(dict(config["data"]))
    assignment_filters = data.setdefault("overrides", {}).setdefault(
        "filters", {}
    ).setdefault("assignment", {})
    assignment_filters["rounds"] = [1] if first_round else "all"
    scenario = load_scenario(data)
    student_path = scenario.source("assignment.students").path
    program_path = scenario.source("assignment.programs").path
    school_path = scenario.source("assignment.school_coordinates").path
    for path in (student_path, program_path, school_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    return [
        EvaluationTask(
            assignment_path=str(path),
            data=copy.deepcopy(data),
            new_ctip_path=str(new_ctip_path) if new_ctip_path else None,
        )
        for path in assignments
    ]


def evaluate_policy(
    assignments: list[Path],
    config: Mapping[str, Any],
    new_ctip_path: Path | None,
    *,
    first_round: bool = True,
) -> pd.Series:
    metrics = [
        evaluate_assignment(task)
        for task in evaluation_tasks(
            assignments, config, new_ctip_path, first_round=first_round
        )
    ]
    if len(metrics) != ITERATION_COUNT:
        raise ValueError(
            f"evaluated {len(metrics)} iterations, expected {ITERATION_COUNT}"
        )
    return pd.concat(metrics, axis=1).mean(axis=1, skipna=True)


def write_metrics_csv(
    metrics: Mapping[str, pd.Series],
    matches_root: Path,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = (
        matches_root / f"status_quo_policies_25_eval_assignment_full_{timestamp}.csv"
    )
    frame = pd.DataFrame(metrics)
    frame.index.name = "metric"
    with output_path.open("x", encoding="utf-8", newline="") as output_file:
        frame.to_csv(output_file)
    return output_path


def save_simulation_config(config: Mapping[str, Any], matches_root: Path) -> Path:
    output_path = matches_root / "simulation_config.yaml"
    if output_path.exists():
        return output_path
    with output_path.open("x", encoding="utf-8") as output_file:
        yaml.safe_dump(dict(config), output_file, sort_keys=False)
    return output_path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file)
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    base_config_path = args.base_config.expanduser().resolve()
    policy_paths = [path.expanduser().resolve() for path in args.policies]
    matches_root = args.matches_root.expanduser().resolve()
    matches_root.mkdir(parents=True, exist_ok=True)

    labels = [path.stem for path in policy_paths]
    if len(set(labels)) != len(labels):
        raise ValueError(f"policy filenames must have unique stems: {labels}")
    policies = {path.stem: load_yaml(path) for path in policy_paths}
    config = build_simulation_config(load_yaml(base_config_path), labels, matches_root)
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
        for label in labels
    }
    metrics = {}
    for label in labels:
        metrics[label] = evaluate_policy(
            assignments_by_policy[label], config, new_ctip_path
        )
        LOGGER.info("Evaluated %s", label)

    output_path = write_metrics_csv(metrics, matches_root)
    LOGGER.info(
        "Wrote %d metrics for %d policies to %s",
        len(pd.DataFrame(metrics)),
        len(metrics),
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run and evaluate 25-iteration soft-reserve matches for saved zone runs.

Matching artifacts are written outside the source benchmark tree. Existing
destinations are reused only when they already contain all 25 assignments;
they are never reset or overwritten. Each output CSV column is the mean of
``eval_assignment_full()`` over one run's 25 assignment iterations.

Independent benchmark runs execute in parallel. The 25 matching iterations
within a run remain serial so their seeded NumPy random stream is unchanged.

Usage:
    uv run python analysis/evaluate_soft_reserve_matches.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment.student_assignment.evaluation.match_evaluator import (  # noqa: E402
    MatchEvaluator,
)
from benchmark.config import MatchingRunConfig  # noqa: E402
from benchmark.matching import run_matching_for_solution  # noqa: E402
from benchmark.results import discover_run_dirs  # noqa: E402
from optimization.levels import LevelSpec  # noqa: E402

LOGGER = logging.getLogger(__name__)

ITERATION_COUNT = 25
DEFAULT_RUNS_ROOT = Path("/share/data/school_choice/local_runs/sfusd_zone_test_3")
DEFAULT_POLICY = PROJECT_ROOT / "benchmark/matching/zones+soft_reserves_05frl.yaml"
DEFAULT_MATCHES_ROOT = PROJECT_ROOT / "analysis/matches/zones+soft_reserves_05frl_25"
DEFAULT_BASE_ZONE_MATCHES_ROOT = (
    PROJECT_ROOT / "analysis/matches/zones_soft_reserves_05frl_25"
)
DEFAULT_PLOTS_ROOT = PROJECT_ROOT / "analysis/plots"
DEFAULT_NEW_CTIP_PATH = Path(
    "/share/data/school_choice/Data/2025_cleaned_data/Cleaned_new/ETB_2024.npy"
)


@dataclass(frozen=True)
class SavedAreaSolution:
    """The saved solution fields required by the matching integration."""

    level: LevelSpec
    assignment: dict[int, int]
    status: str

    @property
    def feasible(self) -> bool:
        return self.status in {"FEASIBLE", "OPTIMAL"}

    def area_assignment(self) -> dict[int, int]:
        return dict(self.assignment)


@dataclass(frozen=True)
class EvaluationTask:
    assignment_path: str
    student_path: str
    program_path: str
    school_path: str
    new_ctip_path: str | None
    year: int
    no_special_program: bool = True


@dataclass(frozen=True)
class MatchingRunTask:
    label: str
    source_run: str
    destination: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--matches-root", type=Path, default=DEFAULT_MATCHES_ROOT)
    parser.add_argument("--plots-root", type=Path, default=DEFAULT_PLOTS_ROOT)
    parser.add_argument(
        "--base-zone",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help=(
            "Evaluate a standalone row-per-zone BlockGroup CSV. Repeat for "
            "multiple labeled zone plans."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Processes used to generate independent matching runs.",
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Processes used to evaluate independent matching runs.",
    )
    parser.add_argument(
        "--new-ctip-path",
        type=Path,
        default=DEFAULT_NEW_CTIP_PATH,
        help="Optional equity-block NPY used by the full evaluator.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N discovered runs (useful for validation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover runs and planned destinations without writing anything.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_saved_area_solution(run_dir: Path) -> tuple[SavedAreaSolution, dict[str, Any]]:
    """Load the root-level saved area assignment without rebuilding its graph."""
    manifest = _load_json(run_dir / "benchmark_manifest.json")
    final_stage_name = manifest.get("final_stage")
    final_stage = next(
        (
            stage
            for stage in manifest.get("stages", [])
            if stage.get("name") == final_stage_name
        ),
        None,
    )
    if final_stage is None:
        raise ValueError(f"manifest has no final stage record for {final_stage_name!r}")

    level_name = str(final_stage["level"])
    solution_data = _load_json(run_dir / f"solution_{level_name}.json")
    status = str(solution_data.get("status") or final_stage.get("status") or "UNKNOWN")
    if status not in {"FEASIBLE", "OPTIMAL"}:
        raise ValueError(f"root solution is not feasible (status={status})")
    if (solution_data.get("metadata") or {}).get("partial_assignment"):
        raise ValueError("root solution contains only a partial assignment")

    raw_assignment = _load_json(run_dir / f"zone_dict_area_{level_name}.json")
    assignment = {
        int(area_id): int(zone_id) for area_id, zone_id in raw_assignment.items()
    }
    if not assignment:
        raise ValueError("root area assignment is empty")

    return SavedAreaSolution(LevelSpec.parse(level_name), assignment, status), manifest


def load_base_zone_solution(zone_path: Path) -> SavedAreaSolution:
    """Load a row-per-zone BlockGroup CSV as a matching-ready solution."""
    assignment: dict[int, int] = {}
    with zone_path.open(newline="", encoding="utf-8") as zone_file:
        for zone_id, row in enumerate(csv.reader(zone_file)):
            area_ids = [int(value) for value in row if value.strip()]
            if not area_ids:
                raise ValueError(f"base zone {zone_id} is empty: {zone_path}")
            duplicates = sorted(set(area_ids) & set(assignment))
            if duplicates:
                raise ValueError(
                    f"base zone CSV assigns areas more than once, including {duplicates[:3]}"
                )
            assignment.update({area_id: zone_id for area_id in area_ids})

    if not assignment:
        raise ValueError(f"base zone CSV is empty: {zone_path}")
    return SavedAreaSolution(LevelSpec.parse("BlockGroup_0"), assignment, "FEASIBLE")


def parse_base_zones(values: list[str]) -> list[tuple[str, Path]]:
    """Parse and validate repeated ``LABEL=CSV`` command-line values."""
    parsed: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        label = label.strip()
        if not separator or not label or not raw_path.strip():
            raise ValueError(f"invalid --base-zone {value!r}; expected LABEL=CSV")
        if label in seen_labels:
            raise ValueError(f"duplicate --base-zone label: {label}")
        path = Path(raw_path.strip()).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parsed.append((label, path))
        seen_labels.add(label)
    return parsed


def ensure_matching_output(
    *,
    source_run: Path,
    destination: Path,
    solution: SavedAreaSolution,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    workers: int,
) -> Path:
    """Create one matching output, refusing to alter an existing destination."""
    if destination.exists():
        validate_matching_output(destination)
        LOGGER.info("Reusing existing match: %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        effective_policy_path = destination / "zones+soft_reserves_06frl_25.yaml"
        effective_policy = dict(policy)
        effective_policy["iterations"] = {"start": 0, "end": ITERATION_COUNT}
        with effective_policy_path.open("x", encoding="utf-8") as policy_file:
            yaml.safe_dump(effective_policy, policy_file, sort_keys=False)

        result = run_matching_for_solution(
            solution,  # type: ignore[arg-type]
            str(destination),
            MatchingRunConfig(enabled=True, config=str(effective_policy_path)),
            workers=workers,
        )
        if result is None or result.status != "OK":
            status = None if result is None else result.status
            raise RuntimeError(
                f"matching did not finish successfully (status={status})"
            )
        validate_matching_output(destination)

        source_payload = {
            "source_run": str(source_run),
            "task_id": manifest.get("task_id"),
            "config_hash": manifest.get("config_hash"),
            "iterations": ITERATION_COUNT,
        }
        with (destination / "source_run.json").open("x", encoding="utf-8") as file:
            json.dump(source_payload, file, indent=2, sort_keys=True)
            file.write("\n")
    except BaseException:
        shutil.rmtree(destination)
        raise

    return destination


def run_matching_task(
    task: MatchingRunTask,
    policy: Mapping[str, Any],
) -> str:
    """Generate or validate all matching iterations for one benchmark run."""
    source_run = Path(task.source_run)
    destination = Path(task.destination)
    if source_run.is_file():
        solution = load_base_zone_solution(source_run)
        manifest: dict[str, Any] = {}
    else:
        solution, manifest = load_saved_area_solution(source_run)
    output = ensure_matching_output(
        source_run=source_run,
        destination=destination,
        solution=solution,
        manifest=manifest,
        policy=policy,
        workers=1,
    )
    return str(output)


def run_matching_tasks(
    tasks: list[MatchingRunTask],
    policy: Mapping[str, Any],
    workers: int,
    *,
    verbose: bool = False,
) -> tuple[dict[str, Path], int]:
    """Execute independent matching runs with bounded process parallelism."""
    outputs_by_label: dict[str, Path] = {}
    failures = 0
    if not tasks:
        return {}, 0

    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            try:
                outputs_by_label[task.label] = Path(run_matching_task(task, policy))
                LOGGER.info("Matched %d/%d: %s", index, len(tasks), task.label)
            except Exception as exc:
                failures += 1
                LOGGER.warning("Skipping matching run %s: %s", task.label, exc)
                if verbose:
                    LOGGER.exception("Matching failure details")
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = {
                executor.submit(run_matching_task, task, policy): task for task in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    outputs_by_label[task.label] = Path(future.result())
                    LOGGER.info("Matched %d/%d: %s", index, len(tasks), task.label)
                except Exception as exc:
                    failures += 1
                    LOGGER.warning("Skipping matching run %s: %s", task.label, exc)
                    if verbose:
                        LOGGER.exception("Matching failure details")

    ordered_outputs = {
        task.label: outputs_by_label[task.label]
        for task in tasks
        if task.label in outputs_by_label
    }
    return ordered_outputs, failures


def validate_matching_output(output_root: Path) -> list[Path]:
    """Return the 25 raw assignments or raise for an incomplete output."""
    matching_dir = output_root / "matching"
    config = _load_yaml(matching_dir / "config.generated.yaml")
    iterations = config.get("iterations") or {}
    count = int(iterations.get("end", 0)) - int(iterations.get("start", 0))
    if count != ITERATION_COUNT:
        raise ValueError(
            f"existing output config has {count} iterations, expected {ITERATION_COUNT}"
        )

    assignments_dir = matching_dir / "assignments_raw"
    assignment_paths = sorted(assignments_dir.rglob("*.csv"))
    if len(assignment_paths) != ITERATION_COUNT:
        raise ValueError(
            f"found {len(assignment_paths)} raw assignments, expected {ITERATION_COUNT}"
        )
    return assignment_paths


def evaluation_tasks(
    output_root: Path,
    new_ctip_path: Path | None,
) -> list[EvaluationTask]:
    config = _load_yaml(output_root / "matching/config.generated.yaml")
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("generated matching config has no paths mapping")

    student_path = _resolve_config_path(paths, "student-data")
    program_path = _resolve_config_path(paths, "program-data")
    school_path = _resolve_config_path(paths, "school-data")
    for path in (student_path, program_path, school_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    year = int(config.get("year", 23))
    evaluator_year = int(f"{year:02d}{(year + 1) % 100:02d}")
    assignments = validate_matching_output(output_root)
    return [
        EvaluationTask(
            assignment_path=str(assignment),
            student_path=str(student_path),
            program_path=str(program_path),
            school_path=str(school_path),
            new_ctip_path=str(new_ctip_path) if new_ctip_path else None,
            year=evaluator_year,
        )
        for assignment in assignments
    ]


@lru_cache(maxsize=4)
def _load_students(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluate_assignment(task: EvaluationTask) -> pd.Series:
    assignment = pd.read_csv(task.assignment_path)
    evaluator = MatchEvaluator(
        _load_students(task.student_path),
        assignment,
        first_round=True,
        dropout=False,
        low_income=95292,
        medium_income=95292,
        high_income=110850,
        grade=None,
        year=task.year,
        no_special_program=task.no_special_program,
        program_file=task.program_path,
        schools_latlon_path=task.school_path,
        new_ctip_path=task.new_ctip_path,
    )
    return evaluator.eval_assignment_full()


def evaluate_run(
    output_root: Path,
    new_ctip_path: Path | None,
) -> pd.Series:
    tasks = evaluation_tasks(output_root, new_ctip_path)
    iteration_metrics = [evaluate_assignment(task) for task in tasks]

    if len(iteration_metrics) != ITERATION_COUNT:
        raise ValueError(
            f"evaluated {len(iteration_metrics)} iterations, expected {ITERATION_COUNT}"
        )
    return pd.concat(iteration_metrics, axis=1).mean(axis=1, skipna=True)


def run_evaluation_tasks(
    matching_outputs: Mapping[str, Path],
    new_ctip_path: Path | None,
    workers: int,
    *,
    verbose: bool = False,
) -> tuple[dict[str, pd.Series], int]:
    """Evaluate complete matching runs in parallel, one run per process task."""
    output_items = list(matching_outputs.items())
    metrics_by_label: dict[str, pd.Series] = {}
    failures = 0
    if not output_items:
        return {}, 0

    if workers == 1:
        for index, (label, output_root) in enumerate(output_items, start=1):
            try:
                metrics_by_label[label] = evaluate_run(output_root, new_ctip_path)
                LOGGER.info("Evaluated %d/%d: %s", index, len(output_items), label)
            except Exception as exc:
                failures += 1
                LOGGER.warning("Skipping evaluation run %s: %s", label, exc)
                if verbose:
                    LOGGER.exception("Evaluation failure details")
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(output_items))
        ) as executor:
            futures = {
                executor.submit(evaluate_run, output_root, new_ctip_path): label
                for label, output_root in output_items
            }
            for index, future in enumerate(as_completed(futures), start=1):
                label = futures[future]
                try:
                    metrics_by_label[label] = future.result()
                    LOGGER.info("Evaluated %d/%d: %s", index, len(output_items), label)
                except Exception as exc:
                    failures += 1
                    LOGGER.warning("Skipping evaluation run %s: %s", label, exc)
                    if verbose:
                        LOGGER.exception("Evaluation failure details")

    ordered_metrics = {
        label: metrics_by_label[label]
        for label, _ in output_items
        if label in metrics_by_label
    }
    return ordered_metrics, failures


def write_metrics_csv(
    metrics: Mapping[str, pd.Series],
    output_root: Path,
    prefix: str = "soft_reserves_06frl_25",
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_root / f"{prefix}_eval_assignment_full_{timestamp}.csv"
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
    if args.workers < 1 or args.evaluation_workers < 1:
        raise ValueError("worker counts must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    base_zones = parse_base_zones(args.base_zone)
    runs_root = args.runs_root.expanduser().resolve()
    matches_root = args.matches_root.expanduser().resolve()
    if base_zones and args.matches_root == DEFAULT_MATCHES_ROOT:
        matches_root = DEFAULT_BASE_ZONE_MATCHES_ROOT.resolve()
    plots_root = args.plots_root.expanduser().resolve()
    policy_path = args.policy.expanduser().resolve()
    policy = _load_yaml(policy_path)
    if base_zones:
        if args.limit is not None:
            base_zones = base_zones[: args.limit]
        matching_tasks = [
            MatchingRunTask(
                label=label,
                source_run=str(zone_path),
                destination=str(matches_root / label),
            )
            for label, zone_path in base_zones
        ]
        LOGGER.info("Loaded %d standalone base-zone plans", len(matching_tasks))
    else:
        run_dirs = [Path(path).resolve() for path in discover_run_dirs(str(runs_root))]
        if args.limit is not None:
            run_dirs = run_dirs[: args.limit]
        matching_tasks = [
            MatchingRunTask(
                label=run_dir.relative_to(runs_root).as_posix(),
                source_run=str(run_dir),
                destination=str(matches_root / run_dir.relative_to(runs_root)),
            )
            for run_dir in run_dirs
        ]
        LOGGER.info("Discovered %d benchmark runs under %s", len(run_dirs), runs_root)

    if args.dry_run:
        for task in matching_tasks:
            LOGGER.info("%s -> %s", task.source_run, task.destination)
        return 0

    new_ctip_path = args.new_ctip_path.expanduser().resolve()
    if not new_ctip_path.is_file():
        LOGGER.warning(
            "Equity-block file not found; ET metrics will use no blocks: %s",
            new_ctip_path,
        )
        new_ctip_path = None

    matching_outputs, matching_failures = run_matching_tasks(
        matching_tasks,
        policy,
        args.workers,
        verbose=args.verbose,
    )

    metrics_by_run, evaluation_failures = run_evaluation_tasks(
        matching_outputs,
        new_ctip_path,
        args.evaluation_workers,
        verbose=args.verbose,
    )

    if not metrics_by_run:
        LOGGER.error("No runs completed matching and full evaluation; no CSV written.")
        return 1

    if base_zones:
        output_path = write_metrics_csv(
            metrics_by_run,
            matches_root,
            prefix="zones_soft_reserves_06frl_25",
        )
    else:
        output_path = write_metrics_csv(metrics_by_run, plots_root)
    LOGGER.info(
        "Wrote %d metrics for %d runs to %s",
        len(pd.DataFrame(metrics_by_run)),
        len(metrics_by_run),
        output_path,
    )
    LOGGER.info(
        "Skipped %d matching runs and %d evaluation runs",
        matching_failures,
        evaluation_failures,
    )
    return 0


def _resolve_config_path(paths: Mapping[str, Any], key: str) -> Path:
    value = paths.get(key)
    if not value:
        raise ValueError(f"generated matching config has no {key!r} path")
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        return path.resolve()
    root = paths.get("sfusd")
    if not root:
        raise ValueError(f"relative {key!r} path has no 'sfusd' root")
    return (Path(os.path.expanduser(str(root))) / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())

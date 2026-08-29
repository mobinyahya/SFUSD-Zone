"""Thin generated-zone adapter between benchmark and assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assignment.generated_zones import (
    GENERATED_ZONE_FILENAME,
    SKIP_MARKER_FILENAME,
    run_generated_zone_assignments,
    write_generated_zones,
)
from benchmark.config import MatchingRunConfig, optimization_config_from_dict
from benchmark.results import discover_run_dirs
from benchmark.runner import load_solutions
from metrics.base import MetricsContext
from optimization.config import OptimizationConfig
from optimization.solution import ZoneSolution


@dataclass
class AssignmentBatchResult:
    total: int = 0
    successful: int = 0
    skipped: int = 0
    failed: int = 0


def process_solution_assignments(
    solutions: list[ZoneSolution],
    final_solution: ZoneSolution,
    stage_records: list[dict],
    output_dir: str,
    config: OptimizationConfig,
    matching: MatchingRunConfig,
    *,
    target_prefix: str | None = None,
) -> list[dict]:
    """Prepare generated zone targets for one benchmark run."""
    if not matching.config:
        raise ValueError("Assignment execution requires matching.config.")
    root = Path(output_dir).expanduser().resolve()
    prefix = target_prefix or root.name
    solution_targets = [(final_solution, root, f"{prefix}-root")]
    if matching.compute_stage_assignments:
        solution_targets.extend(
            (solution, root / stage["path"], f"{prefix}-stage-{stage['index']}")
            for solution, stage in zip(solutions, stage_records)
        )

    targets = []
    for solution, target, target_id in solution_targets:
        skip_marker = target / SKIP_MARKER_FILENAME
        if not solution.feasible or solution.metadata.get("partial_assignment"):
            target.mkdir(parents=True, exist_ok=True)
            skip_marker.write_text(
                "ineligible optimization solution\n", encoding="utf-8"
            )
            continue
        skip_marker.unlink(missing_ok=True)
        zone_file = target / GENERATED_ZONE_FILENAME
        write_generated_zones(solution.area_assignment(), zone_file)
        targets.append(
            {
                "id": target_id,
                "zone_file": str(zone_file),
                "skip_marker": str(skip_marker),
                "zone_building_blocks": zone_building_blocks(solution.level.unit),
                "geography_vintage": config.data_scenario.filter(
                    "optimization", "geography_vintage"
                ),
            }
        )

    if matching.compute_stage_assignments and config.strategy == "saa":
        level = config.levels[-1]
        for index in range(len(solutions), config.max_iterations + 1):
            target = root / "stages" / f"iteration_{index:02d}_{level}"
            target.mkdir(parents=True, exist_ok=True)
            (target / SKIP_MARKER_FILENAME).write_text(
                "optimization stage not produced\n", encoding="utf-8"
            )

    return targets


def run_assignments_for_existing_runs(
    root_folder: str,
    matching: MatchingRunConfig,
    *,
    fail_fast: bool = False,
    dataset_factory=None,
) -> AssignmentBatchResult:
    """Execute one root-level assignment batch for every saved benchmark run."""
    result = AssignmentBatchResult()
    run_dirs = discover_run_dirs(root_folder)
    if not matching.enabled:
        result.total = len(run_dirs)
        result.skipped = len(run_dirs)
        return result
    targets = []
    prepared_runs = 0
    workers = 1
    for run_dir in run_dirs:
        result.total += 1
        try:
            dataset = None
            if dataset_factory is not None:
                from benchmark.runner import load_manifest

                manifest = load_manifest(run_dir)
                config = optimization_config_from_dict(manifest["config"])
                dataset = dataset_factory(config, manifest)
            solutions, config, manifest = load_solutions(run_dir, dataset=dataset)
            if not solutions:
                result.skipped += 1
                continue
            final_solution = MetricsContext(solutions, config=config).solution
            targets.extend(
                process_solution_assignments(
                    solutions,
                    final_solution,
                    manifest.get("stages", []),
                    run_dir,
                    config,
                    matching,
                    target_prefix=str(manifest["task_id"]),
                )
            )
            workers = max(workers, int(config.workers or 1))
            prepared_runs += 1
        except Exception:
            result.failed += 1
            if fail_fast:
                raise

    if not targets:
        result.successful += prepared_runs
        return result
    try:
        run_generated_zone_assignments(
            matching.config,
            targets,
            assignment_folder=Path(root_folder) / "assignments",
            workers=workers,
        )
        result.successful += prepared_runs
    except Exception:
        result.failed += prepared_runs
        if fail_fast:
            raise
    return result


def zone_building_blocks(unit: str) -> str:
    try:
        return {
            "BlockGroup": "block_group",
            "Block": "block",
            "Tract": "tract",
            "attendance_area": "attendance_area",
        }[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported assignment unit: {unit}") from exc

"""Thin generated-zone adapter between benchmark and assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assignment.generated_zones import (
    GENERATED_ZONE_FILENAME,
    SKIP_MARKER_FILENAME,
    run_generated_zone_assignment,
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
    execute: bool = True,
) -> None:
    """Prepare generated zones and optionally execute assignment for one run."""
    if not matching.config:
        raise ValueError("Assignment execution requires matching.config.")
    root = Path(output_dir).expanduser().resolve()
    targets = [(final_solution, root)]
    if matching.compute_stage_assignments:
        targets.extend(
            (solution, root / stage["path"])
            for solution, stage in zip(solutions, stage_records)
        )

    for solution, target in targets:
        skip_marker = target / SKIP_MARKER_FILENAME
        if not solution.feasible or solution.metadata.get("partial_assignment"):
            target.mkdir(parents=True, exist_ok=True)
            skip_marker.write_text(
                "ineligible optimization solution\n", encoding="utf-8"
            )
            continue
        skip_marker.unlink(missing_ok=True)
        if execute:
            run_generated_zone_assignment(
                matching.config,
                solution.area_assignment(),
                assignment_folder=target,
                zone_building_blocks=zone_building_blocks(solution.level.unit),
                geography_vintage=config.data_scenario.filter(
                    "optimization", "geography_vintage"
                ),
                workers=max(1, int(config.workers or 1)),
            )
        else:
            write_generated_zones(
                solution.area_assignment(), target / GENERATED_ZONE_FILENAME
            )


def run_assignments_for_existing_runs(
    root_folder: str,
    matching: MatchingRunConfig,
    *,
    fail_fast: bool = False,
    dataset_factory=None,
) -> AssignmentBatchResult:
    """Execute assignment for every saved benchmark run."""
    result = AssignmentBatchResult()
    run_dirs = discover_run_dirs(root_folder)
    if not matching.enabled:
        result.total = len(run_dirs)
        result.skipped = len(run_dirs)
        return result
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
            process_solution_assignments(
                solutions,
                final_solution,
                manifest.get("stages", []),
                run_dir,
                config,
                matching,
            )
            result.successful += 1
        except Exception:
            result.failed += 1
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

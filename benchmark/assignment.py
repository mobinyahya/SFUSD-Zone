"""Thin generated-zone adapter between benchmark and assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from assignment.generated_zones import (
    GENERATED_ZONE_FILENAME,
    SKIP_MARKER_FILENAME,
    run_generated_zone_assignments,
    write_generated_zones,
)
from benchmark.config import (
    BenchmarkTask,
    MatchingRunConfig,
    optimization_config_from_dict,
    optimization_config_to_dict,
    stable_hash,
)
from benchmark.results import discover_run_dirs
from benchmark.runner import MANIFEST_FILENAME, load_manifest, load_solutions
from metrics.base import MetricsContext, is_iterative_strategy, stage_names
from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
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
        zone_file = target / GENERATED_ZONE_FILENAME
        if not solution.feasible or solution.metadata.get("partial_assignment"):
            target.mkdir(parents=True, exist_ok=True)
            zone_file.unlink(missing_ok=True)
            skip_marker.write_text(
                "ineligible optimization solution\n", encoding="utf-8"
            )
            continue
        skip_marker.unlink(missing_ok=True)
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

    if matching.compute_stage_assignments:
        # Planned assignment jobs wait on every stage the strategy could have
        # saved; mark the ones this run did not produce.
        planned = planned_stage_names(optimization_config_to_dict(config)) or []
        for name in planned[len(solutions) :]:
            target = root / "stages" / name
            target.mkdir(parents=True, exist_ok=True)
            (target / SKIP_MARKER_FILENAME).write_text(
                "optimization stage not produced\n", encoding="utf-8"
            )

    return targets


def planned_stage_names(config: Mapping[str, Any]) -> list[str] | None:
    """Every stage a task can save, or ``None`` when its solve decides the count.

    ``recursive`` saves one stage per level, plain ``single`` one stage at the
    finest level, and ``saa`` up to ``max_iterations + 1`` iterations there.
    """
    strategy = str(config["strategy"])
    levels = list(config["levels"])
    if strategy == "recursive":
        planned = levels
    elif strategy == "single" and int(config.get("enumerated_solutions", -1)) <= 0:
        planned = levels[-1:]
    elif strategy == "saa":
        planned = levels[-1:] * (int(config["max_iterations"]) + 1)
    else:
        return None
    return stage_names(planned, iterative=is_iterative_strategy(strategy))


def plan_assignment_targets(
    tasks: list[BenchmarkTask],
    pending: list[BenchmarkTask],
    matching: MatchingRunConfig,
) -> list[dict[str, str]]:
    """Ready and list the assignment targets of every task in a sweep.

    Slurm turns the list into assignment jobs behind the optimization jobs, so
    every stage target is planned before the run.
    """
    prepare_assignment_targets(tasks, pending, matching)
    return assignment_targets(tasks, matching)


def prepare_assignment_targets(
    tasks: list[BenchmarkTask],
    pending: list[BenchmarkTask],
    matching: MatchingRunConfig,
) -> None:
    """Ready the assignment targets of every task skipped as already complete.

    Pending tasks write their own targets when evaluated, and clear what a
    previous run left when their optimization starts
    (:func:`clear_assignment_targets`), so planning never deletes anything. A
    skipped task may predate matching, so its root target is rebuilt from the
    saved solution here; its stage targets cannot be, so a skipped task asked
    for them must already have them.
    """
    pending_ids = {task.task_id for task in pending}
    for task in tasks:
        if task.task_id in pending_ids:
            continue
        root = Path(task.output_dir)
        manifest_path = root / MANIFEST_FILENAME
        if manifest_path.is_file() and load_manifest(root).get("status") == "ERROR":
            # Kept as complete under rerun_failed: false; nothing to match.
            continue
        if not _has_target(root):
            if matching.compute_stage_assignments:
                raise ValueError(
                    f"Task {task.task_id} finished without matching; stage "
                    "assignments cannot be added afterwards, only root ones."
                )
            prepare_completed_run_target(root, str(task.config["levels"][-1]))
        elif matching.compute_stage_assignments:
            missing = [
                stage.get("name", stage["path"])
                for stage in load_manifest(root).get("stages", [])
                if not _has_target(root / stage["path"])
            ]
            if missing:
                raise ValueError(
                    f"Task {task.task_id} finished without stage assignments "
                    f"for {missing}; they cannot be added afterwards. Rerun "
                    "the task, or turn compute_stage_assignments off."
                )


def clear_assignment_targets(output_dir: str | Path) -> None:
    """Remove the zone files and skip markers a previous run left.

    Called as a task's optimization starts, so a rerun that fails is never
    matched on the old zoning and a stale skip marker never hides a new one.
    """
    root = Path(output_dir)
    for folder in (root, *root.glob("stages/*")):
        (folder / GENERATED_ZONE_FILENAME).unlink(missing_ok=True)
        (folder / SKIP_MARKER_FILENAME).unlink(missing_ok=True)


def _has_target(folder: Path) -> bool:
    return (folder / GENERATED_ZONE_FILENAME).is_file() or (
        folder / SKIP_MARKER_FILENAME
    ).is_file()


def assignment_targets(
    tasks: list[BenchmarkTask],
    matching: MatchingRunConfig,
    *,
    planned_stages: bool = True,
) -> list[dict[str, str]]:
    """List each task's root target, and its stage targets if asked.

    Stage targets are every stage the task can save when ``planned_stages``,
    as Slurm needs them before the run, and otherwise the stages its saved
    manifest records.
    """
    from loaders import load_scenario

    targets = []
    scenarios = {}
    for task in tasks:
        data_key = stable_hash(task.config["data"])
        scenario = scenarios.get(data_key)
        if scenario is None:
            scenario = load_scenario(task.config["data"])
            scenarios[data_key] = scenario
        vintage = scenario.filter("optimization", "geography_vintage")
        unit = zone_building_blocks(LevelSpec.parse(task.config["levels"][0]).unit)
        root = Path(task.output_dir)
        folders = [("root", root)]
        if matching.compute_stage_assignments and planned_stages:
            planned = planned_stage_names(task.config)
            if planned is None:
                raise ValueError(
                    f"Task {task.task_id}: compute_stage_assignments needs a "
                    "stage count known before the run, which strategy "
                    f"{task.config['strategy']!r} does not have."
                )
            folders.extend(
                (f"stage-{index}", root / "stages" / name)
                for index, name in enumerate(planned)
            )
        elif (
            matching.compute_stage_assignments and (root / MANIFEST_FILENAME).is_file()
        ):
            folders.extend(
                (f"stage-{stage['index']}", root / stage["path"])
                for stage in load_manifest(root).get("stages", [])
            )
        for suffix, folder in folders:
            targets.append(
                {
                    "id": f"{task.task_id}-{suffix}",
                    "zone_file": str((folder / GENERATED_ZONE_FILENAME).resolve()),
                    "skip_marker": str((folder / SKIP_MARKER_FILENAME).resolve()),
                    "zone_building_blocks": unit,
                    "geography_vintage": vintage,
                }
            )
    return targets


def run_ready_assignments(
    targets: list[dict[str, str]],
    matching: MatchingRunConfig,
    output_root: str | Path,
    *,
    workers: int,
) -> int:
    """Run the targets whose zone file was written as one local batch.

    Targets marked skipped, or left without a zone file by a failed run (see
    :func:`prepare_assignment_targets`), are left out. Returns the number of targets run.
    """
    ready = [
        target
        for target in targets
        if Path(target["zone_file"]).is_file()
        and not Path(target["skip_marker"]).exists()
    ]
    if ready:
        run_generated_zone_assignments(
            matching.config,
            ready,
            assignment_folder=Path(output_root) / matching.assignment_dir,
            workers=workers,
        )
    return len(ready)


def prepare_completed_run_target(output_dir: str | Path, final_level: str) -> bool:
    """Write the root assignment target of a run that finished without matching.

    The optimization worker only writes ``assignment_zones.csv`` (or the skip
    marker) when matching was enabled for that run, so turning matching on
    after a sweep has finished leaves nothing for the assignment jobs to read.
    This rebuilds the root target from the saved files alone -- the final
    level's ``solution_<level>.json`` for status and the portable
    ``zone_dict_area_<level>.json`` for the zoning -- with the same
    eligibility rule as :func:`process_solution_assignments`, without
    rebuilding any graph. Returns whether the run is eligible.
    """
    import json

    root = Path(output_dir).expanduser().resolve()
    zone_file = root / GENERATED_ZONE_FILENAME
    skip_marker = root / SKIP_MARKER_FILENAME
    with (root / f"solution_{final_level}.json").open(encoding="utf-8") as stream:
        saved = json.load(stream)
    eligible = saved.get("status") in ("OPTIMAL", "FEASIBLE") and not (
        saved.get("metadata") or {}
    ).get("partial_assignment")
    if not eligible:
        zone_file.unlink(missing_ok=True)
        skip_marker.write_text("ineligible optimization solution\n", encoding="utf-8")
        return False
    with (root / f"zone_dict_area_{final_level}.json").open(encoding="utf-8") as stream:
        area_assignment = {int(k): int(v) for k, v in json.load(stream).items()}
    skip_marker.unlink(missing_ok=True)
    write_generated_zones(area_assignment, zone_file)
    return True


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
            assignment_folder=Path(root_folder) / matching.assignment_dir,
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

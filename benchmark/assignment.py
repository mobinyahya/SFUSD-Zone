"""Thin generated-zone adapter between benchmark and assignment."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
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
from benchmark.runner import (
    MANIFEST_FILENAME,
    load_manifest,
    load_solutions,
    write_json,
)
from metrics.base import MetricsContext, is_iterative_strategy, stage_names
from optimization.config import OptimizationConfig
from optimization.levels import LevelSpec
from optimization.solution import ZoneSolution


#: Folder of a run that holds the run's own matching: the raw assignments, the
#: reports averaged over its lotteries (``aggregate_metrics/``), and heatmaps.
MATCHING_DIRNAME = "matching"
#: Written last by :func:`run_task_matching`; a run without it is unmatched.
MATCHING_COMPLETE_FILENAME = "matching_complete.json"
MATCHING_SCHEMA_VERSION = 1
AGGREGATE_METRICS_DIRNAME = "aggregate_metrics"

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
                "data": copy.deepcopy(config.data),
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

    A pipeline (every ``single`` and ``recursive`` run is one) saves one stage
    per step, and ``saa`` up to
    ``max_iterations + 1`` iterations at the finest level.
    """
    levels = planned_stage_levels(config)
    if levels is None:
        return None
    return stage_names(levels, iterative=is_iterative_strategy(str(config["strategy"])))


def planned_stage_levels(config: Mapping[str, Any]) -> list[str] | None:
    """Planned levels in execution order, retaining repeated pipeline solves."""
    strategy = str(config["strategy"])
    levels = list(config["levels"])
    if strategy == "pipeline":
        planned = [
            step["level"] if isinstance(step, Mapping) else step.level
            for step in config["steps"]
        ]
    elif strategy == "saa":
        planned = levels[-1:] * (int(config["max_iterations"]) + 1)
    else:
        return None
    return planned


def _final_level(config: Mapping[str, Any]) -> str:
    if config.get("strategy") == "pipeline":
        step = config["steps"][-1]
        return step["level"] if isinstance(step, Mapping) else step.level
    return str(config["levels"][-1])


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
            prepare_completed_run_target(root, _final_level(task.config))
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
    # The run's own matching belongs to the zoning being replaced.
    clear_run_matching(root)


def mark_missing_targets_skipped(
    output_dir: str | Path,
    config: OptimizationConfig,
    matching: MatchingRunConfig,
    reason: str,
) -> None:
    """Mark each target a failed task left without a zone file as skipped.

    Assignment jobs are planned before optimization runs, so an errored task
    must still answer for every target, or its job fails on the missing zone
    file. A target that already has a zone file is still matched.
    """
    root = Path(output_dir)
    folders = [root]
    if matching.compute_stage_assignments:
        planned = planned_stage_names(optimization_config_to_dict(config)) or []
        folders.extend(root / "stages" / name for name in planned)
    for folder in folders:
        if not _has_target(folder):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / SKIP_MARKER_FILENAME).write_text(f"{reason}\n", encoding="utf-8")


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
        root = Path(task.output_dir)
        folders = [("root", root, _final_level(task.config))]
        if matching.compute_stage_assignments and planned_stages:
            planned = planned_stage_names(task.config)
            if planned is None:
                raise ValueError(
                    f"Task {task.task_id}: compute_stage_assignments needs a "
                    "stage count known before the run, which strategy "
                    f"{task.config['strategy']!r} does not have."
                )
            folders.extend(
                (f"stage-{index}", root / "stages" / name, level)
                for index, (name, level) in enumerate(
                    zip(planned, planned_stage_levels(task.config))
                )
            )
        elif (
            matching.compute_stage_assignments and (root / MANIFEST_FILENAME).is_file()
        ):
            folders.extend(
                (
                    f"stage-{stage['index']}",
                    root / stage["path"],
                    stage.get("level", _final_level(task.config)),
                )
                for stage in load_manifest(root).get("stages", [])
            )
        # A run optimized for one matching policy is matched under that policy
        # alone; one without is matched under every policy of the config.
        policy = task.config.get("matching_policy")
        for suffix, folder, level in folders:
            targets.append(
                {
                    "id": f"{task.task_id}-{suffix}",
                    **({"policies": [policy]} if policy is not None else {}),
                    "zone_file": str((folder / GENERATED_ZONE_FILENAME).resolve()),
                    "skip_marker": str((folder / SKIP_MARKER_FILENAME).resolve()),
                    "zone_building_blocks": zone_building_blocks(
                        LevelSpec.parse(level).unit
                    ),
                    "geography_vintage": vintage,
                    "data": copy.deepcopy(task.config["data"]),
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
    This rebuilds the root target from the saved zonings alone -- the final
    stage's status and its portable area assignment -- with the same
    eligibility rule as :func:`process_solution_assignments`, without
    rebuilding any graph. A legacy run is read from its root alias at
    ``final_level``. Returns whether the run is eligible.
    """
    from optimization import run_archive

    root = Path(output_dir).expanduser().resolve()
    zone_file = root / GENERATED_ZONE_FILENAME
    skip_marker = root / SKIP_MARKER_FILENAME
    archive = run_archive.load_run_zonings(
        root,
        None
        if (root / run_archive.ZONINGS_FILENAME).is_file()
        else [{"name": "final", "level": final_level, "path": "."}],
    )
    saved = run_archive.get_stage(archive)
    eligible = saved.get("status") in ("OPTIMAL", "FEASIBLE") and not (
        saved.get("metadata") or {}
    ).get("partial_assignment")
    if not eligible:
        zone_file.unlink(missing_ok=True)
        skip_marker.write_text("ineligible optimization solution\n", encoding="utf-8")
        return False
    area_assignment = run_archive.area_assignment(archive, saved)
    if area_assignment is None:
        raise ValueError(f"Run {root} saved no area assignment for its final stage.")
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


# --- Per-run matching ------------------------------------------------------
#
# On Slurm a run is matched inside its optimization allocation, right after
# its evaluation, into its own folder. A run is matched under its own targets
# only, and every averaged report groups by ``config_name``
# (``<target id>:<policy>/<variant>``), so a run's reports are complete on
# their own and the sweep's reports are its runs' reports, concatenated.


def matching_dir(output_dir: str | Path) -> Path:
    """The folder a run's own matching is written to."""
    return Path(output_dir) / MATCHING_DIRNAME


def clear_run_matching(output_dir: str | Path) -> None:
    """Remove a run's own matching, its completion marker first."""
    folder = matching_dir(output_dir)
    (folder / MATCHING_COMPLETE_FILENAME).unlink(missing_ok=True)
    if folder.is_dir():
        shutil.rmtree(folder)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matching_identity(task: BenchmarkTask, matching: MatchingRunConfig) -> dict:
    """What a run's matching was computed from; any change means rematching."""
    config_path = Path(str(matching.config)).expanduser().resolve()
    return {
        "schema_version": MATCHING_SCHEMA_VERSION,
        "task_id": task.task_id,
        "config_hash": task.config_hash,
        "matching_config": str(config_path),
        "matching_config_sha256": _file_sha256(config_path),
        "compute_stage_assignments": bool(matching.compute_stage_assignments),
    }


def _read_matching_marker(output_dir: str | Path) -> dict | None:
    path = matching_dir(output_dir) / MATCHING_COMPLETE_FILENAME
    try:
        with path.open(encoding="utf-8") as stream:
            marker = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return marker if isinstance(marker, dict) else None


def matching_complete(task: BenchmarkTask, matching: MatchingRunConfig) -> bool:
    """Whether the run's own matching is done, for its zoning and this config."""
    marker = _read_matching_marker(task.output_dir)
    if marker is None or marker.get("status") not in {"complete", "skipped"}:
        return False
    try:
        identity = _matching_identity(task, matching)
    except OSError:
        return False
    return all(marker.get(key) == value for key, value in identity.items())


def run_task_matching(task: BenchmarkTask, matching: MatchingRunConfig) -> str:
    """Match one evaluated run into its own folder; return the marker status.

    The run's earlier matching is removed first, so the folder holds exactly
    this matching. A run evaluated without matching gets its root target
    rebuilt from its saved zoning (:func:`prepare_assignment_targets`); a run
    with no eligible target (an error, an infeasible stage) is marked
    ``skipped``. The marker is written last, so a run interrupted mid-match
    is matched again.
    """
    if not matching.config:
        raise ValueError("Assignment execution requires matching.config.")
    root = Path(task.output_dir)
    clear_run_matching(root)
    prepare_assignment_targets([task], [], matching)
    targets = assignment_targets([task], matching, planned_stages=False)
    ready = [
        target
        for target in targets
        if Path(target["zone_file"]).is_file()
        and not Path(target["skip_marker"]).exists()
    ]
    folder = matching_dir(root)
    if ready:
        run_generated_zone_assignments(
            matching.config,
            ready,
            assignment_folder=folder,
            workers=1,
        )
    status = "complete" if ready else "skipped"
    write_json(
        str(folder / MATCHING_COMPLETE_FILENAME),
        {
            **_matching_identity(task, matching),
            "status": status,
            "targets": [target["id"] for target in ready],
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return status


def aggregate_matching_metrics(
    root_folder: str | Path, assignment_dir: str
) -> dict[str, int] | None:
    """Rebuild the sweep's ``metrics_*.csv`` from its runs' own reports.

    Writes ``<root>/<assignment_dir>/aggregate_metrics/`` (where the old
    downstream assignment jobs published them) from every run whose matching
    is complete. Each report is the runs' reports concatenated in
    ``config_name`` order: the rows and text the downstream finalizer wrote,
    which sorted by the grouping columns, ``config_name`` first, with each
    run's file already sorted that way. Rows are streamed a run at a time, so
    memory stays flat however many runs there are, and the files are rebuilt
    from scratch each time, so a rerun or resume is idempotent. Returns each
    report's row count, or ``None`` (writing nothing) while no run has
    reports.
    """
    from assignment.student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    files = MarketGenerator.AGGREGATE_METRIC_FILES
    root = Path(root_folder).expanduser().resolve()
    sources: dict[str, list[Path]] = {name: [] for name in files}
    matched_runs = 0
    for run_dir in discover_run_dirs(str(root)):
        marker = _read_matching_marker(run_dir)
        if marker is None or marker.get("status") != "complete":
            continue
        metrics_dir = matching_dir(run_dir) / AGGREGATE_METRICS_DIRNAME
        found = False
        for name, filename in files.items():
            if (metrics_dir / filename).is_file():
                sources[name].append(metrics_dir / filename)
                found = True
        matched_runs += found
    if not matched_runs:
        return None

    output_root = root / assignment_dir
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / AGGREGATE_METRICS_DIRNAME
    staging_dir = Path(
        tempfile.mkdtemp(prefix="aggregate_metrics.tmp.", dir=output_root)
    )
    try:
        row_counts = {
            name: _concatenate_metric_csvs(paths, staging_dir / files[name])
            for name, paths in sources.items()
            if paths
        }
        with (staging_dir / MarketGenerator.AGGREGATE_MANIFEST_FILENAME).open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "source": f"<run>/{MATCHING_DIRNAME}/{AGGREGATE_METRICS_DIRNAME}",
                    "runs": matched_runs,
                    "row_counts": row_counts,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        backup_dir = None
        if output_dir.exists() or output_dir.is_symlink():
            backup_dir = Path(
                tempfile.mkdtemp(prefix="aggregate_metrics.previous.", dir=output_root)
            )
            backup_dir.rmdir()
            output_dir.replace(backup_dir)
        try:
            staging_dir.replace(output_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists():
                backup_dir.replace(output_dir)
            raise
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return row_counts


def _concatenate_metric_csvs(paths: list[Path], output: Path) -> int:
    """Concatenate per-run metric CSVs in ``config_name`` order, row by row.

    Columns are the union of the runs' headers in first-seen order (the
    finalizer's rule); a run missing a column gets an empty cell, which is how
    pandas writes a missing value. Fields go back out through the csv module
    pandas writes with, so every value keeps its text. Returns the row count.
    """
    sources = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            first = next(reader, None)
        if header is None:
            continue
        if "config_name" not in header:
            raise ValueError(f"{path} has no config_name column.")
        key = "" if first is None else first[header.index("config_name")]
        sources.append((key, str(path), header))
    sources.sort()
    columns = list(
        dict.fromkeys(column for _key, _path, header in sources for column in header)
    )
    rows = 0
    previous = None
    with output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(columns)
        for _key, path, header in sources:
            config_index = header.index("config_name")
            positions = (
                None
                if header == columns
                else [
                    header.index(column) if column in header else None
                    for column in columns
                ]
            )
            with open(path, newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                next(reader)
                for row in reader:
                    name = row[config_index]
                    if previous is not None and name < previous:
                        raise ValueError(
                            f"Run metric reports interleave config names ({name!r} "
                            f"after {previous!r} in {path}), so they cannot be "
                            "concatenated in order."
                        )
                    previous = name
                    writer.writerow(
                        row
                        if positions is None
                        else ["" if index is None else row[index] for index in positions]
                    )
                    rows += 1
        out.flush()
        os.fsync(out.fileno())
    return rows

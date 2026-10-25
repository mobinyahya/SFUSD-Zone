"""Optimization-native benchmark task runner."""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from optimization.config import OptimizationConfig
from optimization.run_archive import (
    area_assignment,
    legacy_files,
    load_run_zonings,
    node_assignment,
    save_zonings,
    set_final_stage,
    summary_metadata,
)
from optimization.solution import ZoneSolution, graph_fingerprint, save_zoning_config
from benchmark.config import (
    BenchmarkTask,
    ExecutionConfig,
    MatchingRunConfig,
    VisualizationRunConfig,
    config_snapshot,
    json_ready,
    optimization_config_from_dict,
)
from metrics import MetricsCalculator
from metrics.base import is_iterative_strategy, stage_names


SCHEMA_VERSION = 1
MANIFEST_FILENAME = "benchmark_manifest.json"
RESULT_FILENAME = "result.json"


def _saved_manifest(task: BenchmarkTask) -> dict[str, Any] | None:
    """Return the task's saved manifest when its result belongs to this task."""
    output_dir = os.path.expanduser(task.output_dir)
    if not os.path.exists(os.path.join(output_dir, RESULT_FILENAME)):
        return None
    try:
        manifest = load_manifest(output_dir)
    except Exception:
        return None
    if (
        manifest.get("config_hash") != task.config_hash
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        return None
    return manifest


def valid_existing_result(task: BenchmarkTask, execution: ExecutionConfig) -> bool:
    """Check if task output contains a valid evaluated manifest and result."""
    manifest = _saved_manifest(task)
    if manifest is None or manifest.get("phase") == "optimization":
        return False
    return not (manifest.get("status") == "ERROR" and execution.rerun_failed)


def valid_optimization_result(task: BenchmarkTask, execution: ExecutionConfig) -> bool:
    """Check if task output already contains a valid completed optimization phase."""
    manifest = _saved_manifest(task)
    if manifest is None:
        return False
    if manifest.get("status") == "ERROR" and execution.rerun_failed:
        return False
    return manifest.get("phase") in {"optimization", "complete"}


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    output_dir: str
    status: str
    total_wall_time: float = 0.0
    error_message: str | None = None
    skipped: bool = False


def run_optimization_task(
    task: BenchmarkTask,
    *,
    strict_metrics: bool = True,
    compute_stage_metrics: bool = False,
    matching: MatchingRunConfig | None = None,
    visualization: VisualizationRunConfig | None = None,
) -> TaskResult:
    """Run optimization and evaluate its in-memory stages."""

    phase_result, solutions, manifest = _execute_optimization_phase(task)
    if solutions is None:
        return phase_result
    return evaluate_optimization_task(
        task,
        strict_metrics=strict_metrics,
        compute_stage_metrics=compute_stage_metrics,
        matching=matching,
        visualization=visualization,
        manifest=manifest,
        solutions=solutions,
    )


def run_optimization_phase(task: BenchmarkTask) -> TaskResult:
    """Run and persist optimization stages without calculating benchmark metrics."""

    result, _, _ = _execute_optimization_phase(task)
    return result


def _execute_optimization_phase(
    task: BenchmarkTask,
) -> tuple[TaskResult, list[ZoneSolution] | None, dict[str, Any] | None]:
    from benchmark.assignment import clear_assignment_targets

    output_dir = os.path.expanduser(task.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    clear_assignment_targets(output_dir)
    started_at = _now()
    config = task.optimization_config()
    solutions: list[ZoneSolution] = []
    stage_records: list[dict[str, Any]] = []

    try:
        dataset = config.make_dataset()
        solver = config.make_solver(output_dir=output_dir)
        strategy = config.make_strategy()
        solutions = strategy.run(dataset, solver)
        if not solutions:
            raise ValueError("Optimization strategy returned no solutions.")
        stage_records = save_stage_artifacts(
            solutions, output_dir, stage_names_for(solutions, config), config
        )
        status = str(solutions[-1].status or "UNKNOWN")
        manifest = manifest_for(
            task=task,
            config=config,
            status=status,
            started_at=started_at,
            completed_at=_now(),
            stages=stage_records,
            error_message=None,
            phase="optimization",
        )
        payload = _result_payload(
            {"metrics": {}, "zone_data": {}, "run": {"phase": "optimization"}},
            task=task,
            config=config,
            levels=[solution.level.name for solution in solutions],
            status=status,
            total_wall_time=manifest["total_wall_time"],
        )
        write_json(os.path.join(output_dir, RESULT_FILENAME), payload)
        write_json(os.path.join(output_dir, MANIFEST_FILENAME), manifest)
        result = TaskResult(
            task_id=task.task_id,
            output_dir=output_dir,
            status=status,
            total_wall_time=manifest["total_wall_time"],
        )
        return result, solutions, manifest
    except Exception as exc:
        if solutions and not stage_records:
            try:
                stage_records = save_stage_artifacts(
                    solutions,
                    output_dir,
                    stage_names_for(solutions, config),
                    config,
                )
            except Exception:
                pass
        result = _save_error_result(
            task,
            config,
            output_dir,
            started_at,
            stage_records,
            exc,
            phase="optimization_error",
        )
        return result, None, None


def evaluate_optimization_task(
    task: BenchmarkTask,
    *,
    strict_metrics: bool = True,
    compute_stage_metrics: bool = False,
    matching: MatchingRunConfig | None = None,
    visualization: VisualizationRunConfig | None = None,
    manifest: dict[str, Any] | None = None,
    solutions: list[ZoneSolution] | None = None,
    dataset=None,
    regenerating: bool = False,
) -> TaskResult:
    """Evaluate one persisted optimization task: metrics, matching, and maps.

    ``manifest`` and ``solutions`` default to what is saved under the task's
    output directory, and ``dataset`` to the one its config builds. A task
    whose optimization failed is reported, not evaluated. ``regenerating``
    re-evaluates a finished run: a failure is raised rather than saved over
    its result, and its ``completed_at`` is kept.
    """

    output_dir = os.path.expanduser(task.output_dir)
    started_at = _now()
    config = task.optimization_config()
    failure_phase = "metrics_error"

    try:
        if manifest is None:
            manifest = load_manifest(output_dir)
        if manifest.get("config_hash") != task.config_hash:
            raise ValueError(
                f"Manifest config hash does not match task {task.task_id}."
            )
        if manifest.get("phase") == "optimization_error":
            _skip_missing_targets(output_dir, config, matching, "optimization error")
            return TaskResult(
                task_id=task.task_id,
                output_dir=output_dir,
                status="ERROR",
                total_wall_time=float(manifest.get("total_wall_time") or 0.0),
                error_message=manifest.get("error_message"),
            )
        if solutions is None:
            solutions, config, manifest = load_solutions(
                output_dir, dataset=dataset, manifest=manifest
            )
        if not solutions:
            raise ValueError(
                "No saved optimization stages are available for evaluation."
            )

        calculator = MetricsCalculator(
            solutions,
            config=config,
            strict=strict_metrics,
            compute_stage_metrics=compute_stage_metrics,
        )
        metrics = calculator.compute()
        final_solution = calculator.context.solution
        # A legacy run keeps its root alias; only an archive records the pick.
        set_final_stage(output_dir, calculator.context.final_stage_name)

        if matching and matching.enabled:
            from benchmark.assignment import process_solution_assignments

            process_solution_assignments(
                solutions,
                final_solution,
                manifest.get("stages", []),
                output_dir,
                config,
                matching,
            )

        if visualization and visualization.enabled:
            from benchmark.visualize import (
                render_task_visualizations,
                visualization_is_current,
            )

            failure_phase = "visualization_error"
            if not visualization_is_current(manifest, output_dir, visualization):
                render_task_visualizations(
                    solutions,
                    config,
                    output_dir,
                    visualization,
                    manifest,
                )
            failure_phase = "metrics_error"

        result_payload = result_payload_for(
            metrics=metrics,
            config=config,
            solutions=solutions,
            task=task,
        )
        write_json(os.path.join(output_dir, RESULT_FILENAME), result_payload)
        manifest.update(
            {
                "status": result_payload.get("status") or "UNKNOWN",
                "phase": "complete",
                "error_message": None,
                "total_wall_time": result_payload.get("total_wall_time", 0.0),
                "final_stage": metrics.run.get("final_stage"),
                "metrics_evaluated_at": _now(),
            }
        )
        if regenerating:
            manifest["metrics_regenerated_at"] = _now()
        else:
            manifest["completed_at"] = _now()
        manifest.pop("traceback", None)
        _merge_stage_contiguity(manifest, result_payload)
        write_json(os.path.join(output_dir, MANIFEST_FILENAME), manifest)
        return TaskResult(
            task_id=task.task_id,
            output_dir=output_dir,
            status=str(result_payload.get("status") or "UNKNOWN"),
            total_wall_time=float(result_payload.get("total_wall_time") or 0.0),
        )
    except Exception as exc:
        if regenerating:
            raise
        manifest = manifest or {}
        _skip_missing_targets(output_dir, config, matching, failure_phase)
        return _save_error_result(
            task,
            config,
            output_dir,
            str(manifest.get("started_at") or started_at),
            list(manifest.get("stages") or []),
            exc,
            phase=failure_phase,
            manifest=manifest,
        )


def _skip_missing_targets(
    output_dir: str,
    config: OptimizationConfig,
    matching: MatchingRunConfig | None,
    reason: str,
) -> None:
    if matching and matching.enabled:
        from benchmark.assignment import mark_missing_targets_skipped

        mark_missing_targets_skipped(output_dir, config, matching, reason)


def save_stage_artifacts(
    solutions: Sequence[ZoneSolution],
    output_dir: str,
    stage_names: Sequence[str],
    config: OptimizationConfig | None = None,
) -> list[dict[str, Any]]:
    """Save every stage in the run's one ``zonings.json.xz``.

    Each record's ``path`` names the stage's folder, which only assignment
    jobs create, to hold that stage's target.
    """
    stage_names = list(stage_names)[: len(solutions)]
    solutions = list(solutions)[: len(stage_names)]
    snapshot = config_snapshot(config) if config is not None else None
    save_zonings(output_dir, solutions, stage_names, config=snapshot)
    # A rerun over a run saved in the legacy layout must not leave its old
    # per-stage files beside the archive, where they would read as current.
    for path in legacy_files(output_dir):
        path.unlink()
    if snapshot is not None:
        save_zoning_config(snapshot, output_dir)
    stages_dir = Path(output_dir) / "stages"
    return [
        stage_record(solution, idx, stage_name, stages_dir / stage_name, output_dir)
        for idx, (solution, stage_name) in enumerate(zip(solutions, stage_names))
    ]


def stage_record(
    solution: ZoneSolution,
    index: int,
    stage_name: str,
    stage_dir: Path,
    output_dir: str,
) -> dict[str, Any]:
    return {
        "name": stage_name,
        "index": index,
        "level": solution.level.name,
        "path": os.path.relpath(stage_dir, output_dir),
        "status": solution.status,
        "objective": solution.objective,
        "wall_time": solution.wall_time,
        "num_zones": solution.problem.Z,
        # Filled in from the evaluated stage metrics, when they are computed.
        "contiguous": None,
        # Read for every run on resume and aggregation; the full copy is in
        # the run's archive.
        "metadata": summary_metadata(solution.metadata),
    }


def result_payload_for(
    *,
    metrics,
    config: OptimizationConfig,
    solutions: Sequence[ZoneSolution],
    task: BenchmarkTask,
) -> dict[str, Any]:
    """Build the evaluated ``result.json`` payload from computed metrics."""
    body = metrics.to_full_dict()
    run = body.get("run", {})
    for stage in run.get("stages", []):
        if "metadata" in stage:
            stage["metadata"] = summary_metadata(stage["metadata"])
    return _result_payload(
        body,
        task=task,
        config=config,
        levels=[solution.level.name for solution in solutions],
        status=run.get("final_status"),
        total_wall_time=run.get("total_wall_time", 0.0),
    )


def _result_payload(
    body: dict[str, Any],
    *,
    task: BenchmarkTask,
    config: OptimizationConfig,
    levels: Sequence[str],
    status: str | None,
    total_wall_time: float,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Wrap a metrics/zone_data/run body in the shared ``result.json`` fields."""
    return {
        **body,
        "status": status,
        "error_message": error_message,
        "total_wall_time": total_wall_time,
        "levels": list(levels),
        "config": config_snapshot(config),
        "benchmark": {
            "schema_version": SCHEMA_VERSION,
            "task_id": task.task_id,
            "config_hash": task.config_hash,
        },
    }


def manifest_for(
    *,
    task: BenchmarkTask,
    config: OptimizationConfig,
    status: str,
    started_at: str,
    completed_at: str,
    stages: Sequence[dict[str, Any]],
    error_message: str | None,
    phase: str = "complete",
) -> dict[str, Any]:
    total_wall_time = sum(float(stage.get("wall_time") or 0.0) for stage in stages)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "config_hash": task.config_hash,
        "status": status,
        "phase": phase,
        "error_message": error_message,
        "output_dir": os.path.expanduser(task.output_dir),
        "started_at": started_at,
        "completed_at": completed_at,
        "total_wall_time": total_wall_time,
        "capacity_slots": task.capacity_slots,
        "config": config_snapshot(config),
        "stages": list(stages),
        # Set by evaluation, which selects the final stage.
        "final_stage": None,
        "result_path": RESULT_FILENAME,
    }


def stage_names_for(
    solutions: Sequence[ZoneSolution], config: OptimizationConfig | dict[str, Any]
) -> list[str]:
    strategy = (
        config.get("strategy", "")
        if isinstance(config, dict)
        else getattr(config, "strategy", "")
    )
    return stage_names(
        [solution.level.name for solution in solutions],
        iterative=is_iterative_strategy(strategy),
    )


def load_solutions(
    output_dir: str,
    *,
    dataset=None,
    manifest: dict[str, Any] | None = None,
) -> tuple[list[ZoneSolution], OptimizationConfig, dict[str, Any]]:
    """Reconstruct saved stage solutions for metric regeneration."""

    output_dir = os.path.expanduser(output_dir)
    if manifest is None:
        manifest = load_manifest(output_dir)
    config = optimization_config_from_dict(manifest["config"])
    dataset = dataset or config.make_dataset()
    pipeline = config.make_strategy() if config.strategy == "pipeline" else None
    records = list(manifest.get("stages", []))
    archive = load_run_zonings(output_dir, records)
    saved_by_name = {saved["name"]: saved for saved in archive["stages"]}
    solutions: list[ZoneSolution] = []
    for stage in records:
        level = stage["level"]
        saved = saved_by_name.get(stage["name"])
        if saved is None:
            raise ValueError(
                f"Saved stage {stage['name']} is missing from {output_dir}."
            )
        assignment = node_assignment(saved)
        metadata = dict(saved.get("metadata") or stage.get("metadata") or {})
        centroid_school_ids = metadata.get("centroid_school_ids")
        if (
            centroid_school_ids is None
            and metadata.get("centroid_school_id") is not None
        ):
            centroid_school_ids = [metadata["centroid_school_id"]]
        if pipeline is not None:
            problem = pipeline.problem_for_stage(dataset, metadata)
        else:
            problem = dataset.problem_for(
                level,
                centroid_school_ids=centroid_school_ids,
            )
        if saved.get("graph_fingerprint") != graph_fingerprint(problem.G):
            area_assignment_saved = area_assignment(archive, saved)
            if area_assignment_saved is None:
                raise ValueError(
                    f"Saved stage {stage['name']} uses a different graph and has no "
                    "area assignment for safe reconstruction."
                )
            assignment = _node_assignment_from_areas(
                problem.G,
                area_assignment_saved,
                stage["name"],
            )
        status = saved.get("status")
        solutions.append(
            ZoneSolution(
                problem=problem,
                assignment=assignment,
                status=str(status or stage.get("status") or "UNKNOWN"),
                objective=(
                    saved["objective"]
                    if saved.get("objective") is not None or status is not None
                    else stage.get("objective")
                ),
                wall_time=(
                    saved["wall_time"]
                    if saved.get("wall_time") is not None or status is not None
                    else stage.get("wall_time")
                ),
                metadata=metadata,
            )
        )
    return solutions, config, manifest


def _node_assignment_from_areas(
    G,
    area_assignment: dict[int, int],
    stage_name: str,
) -> dict[int, int]:
    """Reconstruct nodes only when every covered area has the same saved zone."""
    assignment = {}
    for node, attrs in G.nodes(data=True):
        area_ids = (
            [attrs["area_id"]] if "area_id" in attrs else attrs.get("block_ids", [])
        )
        assigned = [
            area_assignment[area_id]
            for area_id in area_ids
            if area_id in area_assignment
        ]
        if not assigned:
            continue
        if len(assigned) != len(area_ids) or len(set(assigned)) != 1:
            raise ValueError(
                f"Saved stage {stage_name} cannot be represented on the current "
                f"graph at node {node}."
            )
        assignment[node] = assigned[0]
    return assignment


def load_manifest(output_dir: str) -> dict[str, Any]:
    path = os.path.join(os.path.expanduser(output_dir), MANIFEST_FILENAME)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    """Atomically replace one JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(json_ready(data), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _save_error_result(
    task: BenchmarkTask,
    config: OptimizationConfig,
    output_dir: str,
    started_at: str,
    stage_records: Sequence[dict[str, Any]],
    exc: Exception,
    *,
    phase: str,
    manifest: dict[str, Any] | None = None,
) -> TaskResult:
    error_message = str(exc) or exc.__class__.__name__
    error_manifest = dict(manifest or {})
    error_manifest.update(
        manifest_for(
            task=task,
            config=config,
            status="ERROR",
            started_at=started_at,
            completed_at=_now(),
            stages=stage_records,
            error_message=error_message,
            phase=phase,
        )
    )
    error_manifest["traceback"] = traceback.format_exc()
    total_wall_time = error_manifest["total_wall_time"]
    write_json(os.path.join(output_dir, MANIFEST_FILENAME), error_manifest)
    write_json(
        os.path.join(output_dir, RESULT_FILENAME),
        _result_payload(
            {"metrics": {}, "zone_data": {}, "run": {"phase": phase}},
            task=task,
            config=config,
            levels=[stage["level"] for stage in stage_records],
            status="ERROR",
            total_wall_time=total_wall_time,
            error_message=error_message,
        ),
    )
    return TaskResult(
        task_id=task.task_id,
        output_dir=output_dir,
        status="ERROR",
        total_wall_time=total_wall_time,
        error_message=error_message,
    )


def _merge_stage_contiguity(
    manifest: dict[str, Any], result_payload: dict[str, Any]
) -> None:
    result_stages = {
        stage.get("name"): stage
        for stage in (result_payload.get("run") or {}).get("stages", [])
    }
    for stage in manifest.get("stages", []):
        evaluated = result_stages.get(stage.get("name"), {})
        if "contiguous" in evaluated:
            stage["contiguous"] = evaluated["contiguous"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

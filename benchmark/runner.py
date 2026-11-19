"""Optimization-native benchmark task runner."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import resource
import socket
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
#: Per-phase wall/CPU seconds and peak RSS of the run's latest attempt.
PHASE_TIMINGS_FILENAME = "phase_timings.json"
PHASE_TIMINGS_SCHEMA_VERSION = 1


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

    phase_result, solutions, manifest, dataset = _execute_optimization_phase(task)
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
        dataset=dataset,
    )


def run_optimization_phase(task: BenchmarkTask) -> TaskResult:
    """Run and persist optimization stages without calculating benchmark metrics."""

    result, _, _, _ = _execute_optimization_phase(task)
    return result


def run_optimize_and_evaluate(
    task: BenchmarkTask,
    *,
    strict_metrics: bool = True,
    compute_stage_metrics: bool = False,
    matching: MatchingRunConfig | None = None,
    visualization: VisualizationRunConfig | None = None,
) -> TaskResult:
    """Optimize, then evaluate in the same process, each phase timed.

    The evaluation is handed the optimization's in-memory dataset, stages,
    and manifest instead of reloading them, as a standalone evaluate
    (:func:`evaluate_optimization_task` given none of them) must. The solver
    and strategy are released first, so the process peaks near the
    optimization's own peak. Each phase persists exactly what it persists
    when run apart, so a crash between them leaves a run that a standalone
    evaluate resumes.
    """

    with timed_phase(task, "optimize") as timing:
        result, solutions, manifest, dataset = _execute_optimization_phase(task)
        timing["status"] = result.status
    if solutions is None:
        return result
    release_memory()
    with timed_phase(task, "evaluate", fused_with="optimize") as timing:
        result = evaluate_optimization_task(
            task,
            strict_metrics=strict_metrics,
            compute_stage_metrics=compute_stage_metrics,
            matching=matching,
            visualization=visualization,
            manifest=manifest,
            solutions=solutions,
            dataset=dataset,
        )
        timing["status"] = result.status
    return result


def release_memory() -> None:
    """Collect garbage and hand freed heap back to the OS where glibc allows."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _peak_rss_mb(who: int) -> float:
    # Linux reports ru_maxrss in KiB, macOS in bytes.
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(who).ru_maxrss / scale


def _current_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cpu_seconds() -> float:
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        usage = resource.getrusage(who)
        total += usage.ru_utime + usage.ru_stime
    return total


@contextlib.contextmanager
def timed_phase(task: BenchmarkTask, phase: str, *, fused_with: str | None = None):
    """Time one task phase; record it and print one line to the log.

    Yields a dict the caller sets ``status`` in. The record (start, end, wall
    and CPU seconds, peak RSS) is merged into the run's
    ``phase_timings.json`` (:func:`record_phase_timing`). ``peak_rss_mb`` is
    the process high-water mark, so a phase fused into the optimization's
    process (``fused_with``) reports the larger of the two; its
    ``rss_start_mb`` shows what the earlier phase still held.
    """
    timing: dict[str, Any] = {"status": None}
    started_at = _now()
    wall_start = time.perf_counter()
    cpu_start = _cpu_seconds()
    rss_start = _current_rss_mb()
    try:
        yield timing
    except BaseException:
        timing["status"] = "CRASHED"
        raise
    finally:
        record = {
            "phase": phase,
            "status": timing.get("status"),
            "started_at": started_at,
            "completed_at": _now(),
            "wall_seconds": round(time.perf_counter() - wall_start, 3),
            "cpu_seconds": round(_cpu_seconds() - cpu_start, 3),
            "peak_rss_mb": round(
                max(
                    _peak_rss_mb(resource.RUSAGE_SELF),
                    _peak_rss_mb(resource.RUSAGE_CHILDREN),
                ),
                1,
            ),
            "rss_start_mb": None if rss_start is None else round(rss_start, 1),
            "fused_with": fused_with,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        try:
            record_phase_timing(task.output_dir, record)
        except Exception as exc:  # A timing never fails its phase.
            print(f"Could not record {phase} timing: {exc}", file=sys.stderr)
        print(format_phase_timing(task.task_id, record), flush=True)


def format_phase_timing(task_id: str, record: Mapping[str, Any]) -> str:
    """The compact one-line allocation-log form of a phase record."""
    fused = f" fused_with={record['fused_with']}" if record.get("fused_with") else ""
    return (
        f"phase task={task_id} phase={record['phase']} "
        f"status={record.get('status')} wall={record['wall_seconds']:.1f}s "
        f"cpu={record['cpu_seconds']:.1f}s peak_rss={record['peak_rss_mb']:.0f}MB"
        f"{fused} host={record.get('host')} pid={record.get('pid')}"
    )


def record_phase_timing(output_dir: str, record: Mapping[str, Any]) -> None:
    """Merge one phase record into ``<run>/phase_timings.json``.

    Only the latest attempt is kept: an optimize record starts a new one,
    dropping the evaluate and match records of the zoning it replaced. One
    process works on a run at a time, so the read-modify-write is safe.
    """
    path = Path(os.path.expanduser(output_dir)) / PHASE_TIMINGS_FILENAME
    phases: dict[str, Any] = {}
    if record["phase"] != "optimize":
        phases = dict(load_phase_timings(output_dir).get("phases") or {})
    phases[record["phase"]] = dict(record)
    write_json(
        str(path),
        {"schema_version": PHASE_TIMINGS_SCHEMA_VERSION, "phases": phases},
    )


def load_phase_timings(output_dir: str) -> dict[str, Any]:
    """The run's ``phase_timings.json``, or an empty record."""
    path = Path(os.path.expanduser(output_dir)) / PHASE_TIMINGS_FILENAME
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _execute_optimization_phase(
    task: BenchmarkTask,
) -> tuple[TaskResult, list[ZoneSolution] | None, dict[str, Any] | None, Any]:
    """Optimize and persist ``task``; return its stages, manifest, and dataset.

    The solver and strategy stay local, so they are released on return; the
    dataset and stages are what an in-process evaluation needs.
    """
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
        del solver, strategy
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
        return result, solutions, manifest, dataset
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
        return result, None, None, None


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
    pipeline = None
    if config.strategy == "pipeline":
        from optimization.strategies.pipeline import PipelineStrategy

        # Only its saved-stage reconstruction is used; nothing is solved.
        pipeline = PipelineStrategy(config=config)
    records = list(manifest.get("stages", []))
    archive = load_run_zonings(output_dir, records)
    saved_by_name = {saved["name"]: saved for saved in archive["stages"]}
    solutions: list[ZoneSolution] = []
    for stage_index, stage in enumerate(records):
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
            problem = pipeline.problem_for_stage(
                dataset, metadata, centroid_school_ids, stage_index
            )
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

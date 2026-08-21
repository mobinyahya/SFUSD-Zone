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
from optimization.solution import ZoneSolution, graph_fingerprint
from benchmark.config import (
    BenchmarkTask,
    MatchingRunConfig,
    VisualizationRunConfig,
    config_snapshot,
    json_ready,
    optimization_config_from_dict,
)
from metrics import MetricsCalculator


SCHEMA_VERSION = 1
MANIFEST_FILENAME = "benchmark_manifest.json"
RESULT_FILENAME = "result.json"


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
    execute_assignments: bool = True,
) -> TaskResult:
    """Run optimization and metrics together for local execution."""

    phase_result, loaded = _execute_optimization_phase(task)
    if loaded is None:
        return phase_result
    return _evaluate_optimization_task(
        task,
        strict_metrics=strict_metrics,
        compute_stage_metrics=compute_stage_metrics,
        matching=matching,
        visualization=visualization,
        execute_assignments=execute_assignments,
        loaded=loaded,
    )


def run_optimization_phase(task: BenchmarkTask) -> TaskResult:
    """Run and persist optimization stages without calculating benchmark metrics."""

    result, _ = _execute_optimization_phase(task)
    return result


def evaluate_optimization_task(
    task: BenchmarkTask,
    *,
    strict_metrics: bool = True,
    compute_stage_metrics: bool = False,
    matching: MatchingRunConfig | None = None,
    visualization: VisualizationRunConfig | None = None,
    execute_assignments: bool = True,
    dataset=None,
) -> TaskResult:
    """Reconstruct and evaluate one previously persisted optimization task."""

    return _evaluate_optimization_task(
        task,
        strict_metrics=strict_metrics,
        compute_stage_metrics=compute_stage_metrics,
        matching=matching,
        visualization=visualization,
        execute_assignments=execute_assignments,
        dataset=dataset,
    )


def _execute_optimization_phase(
    task: BenchmarkTask,
) -> tuple[
    TaskResult,
    tuple[list[ZoneSolution], OptimizationConfig, dict[str, Any]] | None,
]:
    output_dir = os.path.expanduser(task.output_dir)
    os.makedirs(output_dir, exist_ok=True)
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
        stage_names = stage_names_for(solutions, config)
        stage_records = save_stage_artifacts(solutions, output_dir, stage_names)
        final_solution = solutions[-1]
        status = str(final_solution.status or "UNKNOWN")
        total_wall_time = sum(
            float(stage.get("wall_time") or 0.0) for stage in stage_records
        )
        payload = optimization_result_payload_for(
            config=config,
            solutions=solutions,
            task=task,
            status=status,
            total_wall_time=total_wall_time,
        )
        manifest = manifest_for(
            task=task,
            config=config,
            status=status,
            started_at=started_at,
            completed_at=_now(),
            stages=stage_records,
            final_stage=None,
            error_message=None,
            phase="optimization",
        )
        write_json(os.path.join(output_dir, RESULT_FILENAME), payload)
        write_json(os.path.join(output_dir, MANIFEST_FILENAME), manifest)
        return (
            TaskResult(
                task_id=task.task_id,
                output_dir=output_dir,
                status=status,
                total_wall_time=total_wall_time,
            ),
            (solutions, config, manifest),
        )
    except Exception as exc:
        if solutions and not stage_records:
            try:
                stage_records = save_stage_artifacts(
                    solutions,
                    output_dir,
                    stage_names_for(solutions, config),
                )
            except Exception:
                pass
        return _save_error_result(
            task,
            config,
            output_dir,
            started_at,
            stage_records,
            exc,
            phase="optimization_error",
        ), None


def _evaluate_optimization_task(
    task: BenchmarkTask,
    *,
    strict_metrics: bool,
    compute_stage_metrics: bool,
    matching: MatchingRunConfig | None,
    visualization: VisualizationRunConfig | None,
    execute_assignments: bool,
    dataset=None,
    loaded: tuple[list[ZoneSolution], OptimizationConfig, dict[str, Any]] | None = None,
) -> TaskResult:
    output_dir = os.path.expanduser(task.output_dir)
    started_at = _now()
    manifest: dict[str, Any] = {}
    config = task.optimization_config()
    failure_phase = "metrics_error"

    try:
        if loaded is None:
            manifest = load_manifest(output_dir)
            if manifest.get("config_hash") != task.config_hash:
                raise ValueError(
                    f"Manifest config hash does not match task {task.task_id}."
                )
            if manifest.get("phase") == "optimization_error":
                return TaskResult(
                    task_id=task.task_id,
                    output_dir=output_dir,
                    status="ERROR",
                    total_wall_time=float(manifest.get("total_wall_time") or 0.0),
                    error_message=manifest.get("error_message"),
                )
            solutions, config, manifest = load_solutions(output_dir, dataset=dataset)
        else:
            solutions, config, manifest = loaded
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
        final_solution.save(output_dir)

        if matching and matching.enabled:
            from benchmark.assignment import process_solution_assignments

            process_solution_assignments(
                solutions,
                final_solution,
                manifest.get("stages", []),
                output_dir,
                config,
                matching,
                execute=execute_assignments,
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
                "completed_at": _now(),
                "total_wall_time": result_payload.get("total_wall_time", 0.0),
                "final_stage": metrics.run.get("final_stage"),
                "metrics_evaluated_at": _now(),
            }
        )
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
        stage_records = list(manifest.get("stages") or [])
        return _save_error_result(
            task,
            config,
            output_dir,
            str(manifest.get("started_at") or started_at),
            stage_records,
            exc,
            phase=failure_phase,
            manifest=manifest,
        )


def save_stage_artifacts(
    solutions: Sequence[ZoneSolution],
    output_dir: str,
    stage_names: Sequence[str],
    *,
    compute_stage_metrics: bool = False,
) -> list[dict[str, Any]]:
    stages_dir = Path(output_dir) / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for idx, (solution, stage_name) in enumerate(zip(solutions, stage_names)):
        stage_dir = stages_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        solution.save(str(stage_dir))
        records.append(
            stage_record(
                solution,
                idx,
                stage_name,
                stage_dir,
                output_dir,
                compute_stage_metrics=compute_stage_metrics,
            )
        )
    return records


def stage_record(
    solution: ZoneSolution,
    index: int,
    stage_name: str,
    stage_dir: Path,
    output_dir: str,
    *,
    compute_stage_metrics: bool = False,
) -> dict[str, Any]:
    contiguous = None
    if compute_stage_metrics and solution.assignment and solution.feasible:
        try:
            contiguous = solution.is_contiguous()
        except Exception:
            contiguous = None
    return {
        "name": stage_name,
        "index": index,
        "level": solution.level.name,
        "path": os.path.relpath(stage_dir, output_dir),
        "status": solution.status,
        "objective": solution.objective,
        "wall_time": solution.wall_time,
        "num_zones": solution.problem.Z,
        "contiguous": contiguous,
        "metadata": dict(solution.metadata),
    }


def result_payload_for(
    *,
    metrics,
    config: OptimizationConfig,
    solutions: Sequence[ZoneSolution],
    task: BenchmarkTask,
) -> dict[str, Any]:
    payload = metrics.to_full_dict()
    run = payload.get("run", {})
    payload.update(
        {
            "status": run.get("final_status"),
            "error_message": None,
            "total_wall_time": run.get("total_wall_time", 0.0),
            "levels": [solution.level.name for solution in solutions],
            "config": config_snapshot(config),
            "benchmark": {
                "schema_version": SCHEMA_VERSION,
                "task_id": task.task_id,
                "config_hash": task.config_hash,
            },
        }
    )
    return payload


def optimization_result_payload_for(
    *,
    config: OptimizationConfig,
    solutions: Sequence[ZoneSolution],
    task: BenchmarkTask,
    status: str,
    total_wall_time: float,
) -> dict[str, Any]:
    """Build the valid, metrics-free result saved by the optimization phase."""

    return {
        "status": status,
        "error_message": None,
        "total_wall_time": total_wall_time,
        "metrics": {},
        "zone_data": {},
        "run": {"phase": "optimization"},
        "levels": [solution.level.name for solution in solutions],
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
    final_stage: str | None,
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
        "final_stage": final_stage,
        "result_path": RESULT_FILENAME,
    }


def stage_names_for(
    solutions: Sequence[ZoneSolution], config: OptimizationConfig | dict[str, Any]
) -> list[str]:
    strategy = _config_value(config, "strategy", "")
    prefix = "iteration" if "iterative" in str(strategy).lower() else "stage"
    return [
        f"{prefix}_{idx:02d}_{solution.level.name}"
        for idx, solution in enumerate(solutions)
    ]


def load_solutions(
    output_dir: str,
    *,
    dataset=None,
) -> tuple[list[ZoneSolution], OptimizationConfig, dict[str, Any]]:
    """Reconstruct saved stage solutions for metric regeneration."""

    output_dir = os.path.expanduser(output_dir)
    manifest = load_manifest(output_dir)
    config = optimization_config_from_dict(manifest["config"])
    dataset = dataset or config.make_dataset()
    solutions: list[ZoneSolution] = []
    for stage in manifest.get("stages", []):
        level = stage["level"]
        stage_dir = os.path.join(output_dir, stage["path"])
        zone_dict_path = os.path.join(stage_dir, f"zone_dict_{level}.json")
        area_dict_path = os.path.join(stage_dir, f"zone_dict_area_{level}.json")
        solution_path = os.path.join(stage_dir, f"solution_{level}.json")
        with open(zone_dict_path, "r", encoding="utf-8") as f:
            assignment = {int(k): int(v) for k, v in json.load(f).items()}
        info: dict[str, Any] = {}
        if os.path.exists(solution_path):
            with open(solution_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        metadata = dict(info.get("metadata") or stage.get("metadata") or {})
        centroid_school_ids = metadata.get("centroid_school_ids")
        if (
            centroid_school_ids is None
            and metadata.get("centroid_school_id") is not None
        ):
            centroid_school_ids = [metadata["centroid_school_id"]]
        problem = dataset.problem_for(
            level,
            centroid_school_ids=centroid_school_ids,
        )
        saved_fingerprint = info.get("graph_fingerprint")
        if saved_fingerprint != graph_fingerprint(problem.G):
            if not os.path.exists(area_dict_path):
                raise ValueError(
                    f"Saved stage {stage['name']} uses a different graph and has no "
                    "area assignment for safe reconstruction."
                )
            with open(area_dict_path, "r", encoding="utf-8") as f:
                area_assignment = {int(k): int(v) for k, v in json.load(f).items()}
            assignment = _node_assignment_from_areas(
                problem.G,
                area_assignment,
                stage["name"],
            )
        solutions.append(
            ZoneSolution(
                problem=problem,
                assignment=assignment,
                status=str(info.get("status") or stage.get("status") or "UNKNOWN"),
                objective=info.get("objective", stage.get("objective")),
                wall_time=info.get("wall_time", stage.get("wall_time")),
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
            final_stage=None,
            error_message=error_message,
            phase=phase,
        )
    )
    error_manifest["traceback"] = traceback.format_exc()
    total_wall_time = float(error_manifest.get("total_wall_time") or 0.0)
    write_json(os.path.join(output_dir, MANIFEST_FILENAME), error_manifest)
    write_json(
        os.path.join(output_dir, RESULT_FILENAME),
        {
            "status": "ERROR",
            "error_message": error_message,
            "total_wall_time": total_wall_time,
            "metrics": {},
            "zone_data": {},
            "run": {"phase": phase},
            "levels": [stage["level"] for stage in stage_records],
            "config": config_snapshot(config),
            "benchmark": {
                "schema_version": SCHEMA_VERSION,
                "task_id": task.task_id,
                "config_hash": task.config_hash,
            },
        },
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


def _config_value(config: OptimizationConfig | dict[str, Any], key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

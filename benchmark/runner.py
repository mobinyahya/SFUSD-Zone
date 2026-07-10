"""Optimization-native benchmark task runner."""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from optimization.config import OptimizationConfig
from optimization.solution import ZoneSolution, graph_fingerprint
from benchmark.config import (
    BenchmarkTask,
    ChoiceMetricsRunConfig,
    MatchingRunConfig,
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
    choice_metrics: ChoiceMetricsRunConfig | None = None,
) -> TaskResult:
    """Run one benchmark task through the new optimization optimization."""

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
        stage_names = stage_names_for(solutions, config)
        stage_records = save_stage_artifacts(
            solutions,
            output_dir,
            stage_names,
            compute_stage_metrics=compute_stage_metrics,
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

        matching_result = None
        stage_matching_result = None
        if matching and matching.enabled:
            from benchmark.matching import (
                StudentAssignmentSession,
                run_matching_for_solution,
                run_matching_for_stages,
            )

            matching_workers = max(1, int(config.workers or 1))
            student_assignment_session = StudentAssignmentSession()
            shared_precomputed_dir = (
                Path(os.path.expanduser(output_dir)).resolve()
                / "matching"
                / "precomputed"
            )
            if final_solution.feasible:
                matching_result = run_matching_for_solution(
                    final_solution,
                    output_dir,
                    matching,
                    student_assignment_session=student_assignment_session,
                    precomputed_dir=shared_precomputed_dir,
                    workers=matching_workers,
                )
            stage_matching_result = run_matching_for_stages(
                solutions,
                stage_records,
                output_dir,
                matching,
                choice_metrics=choice_metrics,
                student_assignment_session=student_assignment_session,
                precomputed_dir=shared_precomputed_dir,
                workers=matching_workers,
            )

        result_payload = result_payload_for(
            metrics=metrics,
            config=config,
            solutions=solutions,
            task=task,
        )
        if matching_result is not None:
            from benchmark.matching import (
                merge_matching_result,
                merge_stage_matching_result,
            )

            merge_matching_result(result_payload, matching_result)
            merge_stage_matching_result(result_payload, stage_matching_result)

        choice_metrics_result = None
        if choice_metrics and choice_metrics.enabled and final_solution.feasible:
            from benchmark.choice_metrics import (
                compute_choice_metrics_for_run,
                merge_choice_metrics_result,
            )

            choice_metrics_result = compute_choice_metrics_for_run(
                output_dir,
                choice_metrics,
            )
            merge_choice_metrics_result(result_payload, choice_metrics_result)
        write_json(os.path.join(output_dir, RESULT_FILENAME), result_payload)

        manifest = manifest_for(
            task=task,
            config=config,
            status=result_payload.get("status") or "UNKNOWN",
            started_at=started_at,
            completed_at=_now(),
            stages=stage_records,
            final_stage=metrics.run.get("final_stage"),
            error_message=None,
        )
        write_json(os.path.join(output_dir, MANIFEST_FILENAME), manifest)
        return TaskResult(
            task_id=task.task_id,
            output_dir=output_dir,
            status=str(result_payload.get("status") or "UNKNOWN"),
            total_wall_time=float(result_payload.get("total_wall_time") or 0.0),
        )
    except Exception as exc:
        if solutions and not stage_records:
            stage_records = save_stage_artifacts(
                solutions,
                output_dir,
                stage_names_for(solutions, config),
                compute_stage_metrics=compute_stage_metrics,
            )
        error_message = str(exc) or exc.__class__.__name__
        manifest = manifest_for(
            task=task,
            config=config,
            status="ERROR",
            started_at=started_at,
            completed_at=_now(),
            stages=stage_records,
            final_stage=None,
            error_message=error_message,
        )
        manifest["traceback"] = traceback.format_exc()
        write_json(os.path.join(output_dir, MANIFEST_FILENAME), manifest)
        write_json(
            os.path.join(output_dir, RESULT_FILENAME),
            {
                "status": "ERROR",
                "error_message": error_message,
                "total_wall_time": 0.0,
                "metrics": {},
                "zone_data": {},
                "run": {},
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
            error_message=error_message,
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
) -> dict[str, Any]:
    total_wall_time = sum(float(stage.get("wall_time") or 0.0) for stage in stages)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "config_hash": task.config_hash,
        "status": status,
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
        problem = dataset.problem_for(level)
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
                metadata=dict(info.get("metadata") or stage.get("metadata") or {}),
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(data), f, indent=2, sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _config_value(config: OptimizationConfig | dict[str, Any], key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

"""Capacity-aware parallel execution for benchmark sweeps."""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field

from benchmark.config import (
    BenchmarkTask,
    ChoiceMetricsRunConfig,
    ExecutionConfig,
    MatchingRunConfig,
    MetricsRunConfig,
    SimulationSweep,
)
from benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    TaskResult,
    load_manifest,
    run_optimization_task,
)


@dataclass
class BatchResult:
    total: int = 0
    feasible: int = 0
    optimal: int = 0
    unknown: int = 0
    infeasible: int = 0
    failed: int = 0
    skipped: int = 0
    total_wall_time: float = 0.0
    results: list[TaskResult] = field(default_factory=list)

    def add(self, result: TaskResult) -> None:
        self.results.append(result)
        status = str(result.status or "UNKNOWN").upper()
        if result.skipped or status == "SKIPPED":
            self.skipped += 1
        elif status == "ERROR":
            self.failed += 1
        elif status == "FEASIBLE":
            self.feasible += 1
        elif status == "OPTIMAL":
            self.optimal += 1
        elif status == "INFEASIBLE":
            self.infeasible += 1
        else:
            self.unknown += 1

    @property
    def successful(self) -> int:
        return self.feasible + self.optimal + self.unknown + self.infeasible

    @property
    def completed(self) -> int:
        return self.successful + self.failed + self.skipped

    def status_count_summary(self, separator: str = " ") -> str:
        return separator.join(
            (
                f"num_feasible={self.feasible}",
                f"num_optimal={self.optimal}",
                f"num_unknown={self.unknown}",
                f"num_infeasible={self.infeasible}",
                f"num_failed={self.failed}",
                f"num_skipped={self.skipped}",
            )
        )


def run_sweep(sweep: SimulationSweep) -> BatchResult:
    return run_tasks(
        sweep.generate_tasks(),
        execution=sweep.execution,
        metrics=sweep.metrics,
        matching=sweep.matching,
        choice_metrics=sweep.choice_metrics,
    )


def run_tasks(
    tasks: list[BenchmarkTask],
    *,
    execution: ExecutionConfig,
    metrics: MetricsRunConfig,
    matching: MatchingRunConfig | None = None,
    choice_metrics: ChoiceMetricsRunConfig | None = None,
) -> BatchResult:
    start = time.time()
    batch = BatchResult(total=len(tasks))
    pending: list[BenchmarkTask] = []

    for task in tasks:
        if execution.skip_existing and _valid_existing_result(task, execution):
            batch.add(
                TaskResult(
                    task_id=task.task_id,
                    output_dir=task.output_dir,
                    status="SKIPPED",
                    skipped=True,
                )
            )
        else:
            pending.append(task)

    if execution.sequential:
        for task in pending:
            result = run_optimization_task(
                task,
                strict_metrics=metrics.strict,
                compute_stage_metrics=metrics.compute_stage_metrics,
                matching=matching,
                choice_metrics=choice_metrics,
            )
            batch.add(result)
            _print_progress(batch)
            if result.status == "ERROR" and execution.fail_fast:
                break
        batch.total_wall_time = time.time() - start
        return batch

    _run_parallel(pending, execution, metrics, matching, choice_metrics, batch)
    batch.total_wall_time = time.time() - start
    return batch


def _run_parallel(
    pending: list[BenchmarkTask],
    execution: ExecutionConfig,
    metrics: MetricsRunConfig,
    matching: MatchingRunConfig | None,
    choice_metrics: ChoiceMetricsRunConfig | None,
    batch: BatchResult,
) -> None:
    if not pending:
        return
    max_workers = _max_workers(execution, len(pending))
    capacity = max(1, int(execution.capacity or max_workers))
    running_slots = 0
    futures = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        max_tasks_per_child=execution.max_tasks_per_worker,
    ) as executor:
        while pending or futures:
            made_progress = True
            while pending and len(futures) < max_workers and made_progress:
                made_progress = False
                idx = _first_task_that_fits(
                    pending, running_slots, capacity, bool(futures)
                )
                if idx is None:
                    break
                task = pending.pop(idx)
                future = executor.submit(
                    _worker_run_task,
                    task,
                    metrics.strict,
                    metrics.compute_stage_metrics,
                    matching,
                    choice_metrics,
                )
                futures[future] = (task, _effective_slots(task, capacity))
                running_slots += _effective_slots(task, capacity)
                made_progress = True

            if not futures:
                continue

            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                task, slots = futures.pop(future)
                running_slots -= slots
                try:
                    result = future.result()
                except Exception as exc:
                    if execution.fail_fast:
                        raise
                    result = TaskResult(
                        task_id=task.task_id,
                        output_dir=task.output_dir,
                        status="ERROR",
                        error_message=str(exc) or exc.__class__.__name__,
                    )
                batch.add(result)
                _print_progress(batch)
                if result.status == "ERROR" and execution.fail_fast:
                    for remaining in futures:
                        remaining.cancel()
                    return


def _worker_run_task(
    task: BenchmarkTask,
    strict_metrics: bool,
    compute_stage_metrics: bool,
    matching: MatchingRunConfig | None,
    choice_metrics: ChoiceMetricsRunConfig | None,
) -> TaskResult:
    return run_optimization_task(
        task,
        strict_metrics=strict_metrics,
        compute_stage_metrics=compute_stage_metrics,
        matching=matching,
        choice_metrics=choice_metrics,
    )


def _valid_existing_result(task: BenchmarkTask, execution: ExecutionConfig) -> bool:
    manifest_path = os.path.join(os.path.expanduser(task.output_dir), MANIFEST_FILENAME)
    result_path = os.path.join(os.path.expanduser(task.output_dir), RESULT_FILENAME)
    if not os.path.exists(manifest_path) or not os.path.exists(result_path):
        return False
    try:
        manifest = load_manifest(task.output_dir)
    except Exception:
        return False
    if manifest.get("config_hash") != task.config_hash:
        return False
    if manifest.get("schema_version") != 1:
        return False
    if manifest.get("status") == "ERROR" and execution.rerun_failed:
        return False
    return True


def _first_task_that_fits(
    tasks: list[BenchmarkTask], running_slots: int, capacity: int, has_running: bool
) -> int | None:
    for idx, task in enumerate(tasks):
        slots = _effective_slots(task, capacity)
        if running_slots + slots <= capacity:
            return idx
    if not has_running and tasks:
        return 0
    return None


def _effective_slots(task: BenchmarkTask, capacity: int) -> int:
    return max(1, min(int(task.capacity_slots), capacity))


def _max_workers(execution: ExecutionConfig, task_count: int) -> int:
    if execution.max_workers is not None:
        return max(1, min(int(execution.max_workers), task_count))
    cpu_count = os.cpu_count() or 1
    capacity = execution.capacity or cpu_count
    return max(1, min(task_count, cpu_count, int(capacity)))


def _print_progress(batch: BatchResult) -> None:
    print(
        f"[{batch.completed}/{batch.total}] {batch.status_count_summary()}",
        flush=True,
    )

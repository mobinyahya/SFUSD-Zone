"""Plan, submit, and execute batched benchmark jobs on Slurm."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import cached_property
import getpass
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from benchmark.config import (
    BenchmarkTask,
    ExecutionConfig,
    MatchingRunConfig,
    MetricsRunConfig,
    SimulationSweep,
    VisualizationRunConfig,
    json_ready,
    optimization_worker_count,
)
from benchmark.assignment import matching_complete, run_task_matching
from benchmark.results import aggregate_results
from benchmark.runner import (
    TaskResult,
    evaluate_optimization_task,
    run_optimization_phase,
    valid_existing_result,
    valid_optimization_result,
    write_json,
)


# 4: execution.queue_multiplier removed.
# 5: matching runs inside the optimization allocations; no assignment plan.
PLAN_SCHEMA_VERSION = 5
MAX_CPUS_PER_NODE = 40
#: The partition's per-user job limit. Every job a plan submits is an
#: optimization allocation: each task is optimized, evaluated, and matched
#: inside its allocation, so there is no downstream job graph.
MAX_SLURM_JOBS = 12
MAX_SUBMITTED_SLURM_JOBS = MAX_SLURM_JOBS
#: Memory per allocated CPU. Every phase of a task (optimize, evaluate,
#: match) runs in its own fresh process on the task's CPUs, so a CPU's
#: memory bounds one phase's peak rather than several stacked.
SBATCH_MEM_PER_CPU = "5G"
#: The phases of one task, in order; each runs in its own process.
TASK_PHASES = ("optimize", "evaluate", "match")
SLURM_DIRNAME = ".slurm"
DEFAULT_PLAN_FILENAME = "benchmark-plan.json"
DEFAULT_SCRIPT_FILENAME = "submit-benchmark.sh"
SBATCH_ACCOUNT = "soal"
SBATCH_PARTITION = "soal"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
)
# Prefer a per-host Gurobi license, then the shared one.
GUROBI_LICENSE_SETUP = (
    'if [ -f "$HOME/gurobi_licenses/gurobi.lic.$(hostname -s)" ]; then '
    'export GRB_LICENSE_FILE="$HOME/gurobi_licenses/gurobi.lic.$(hostname -s)"; '
    'elif [ -n "${SLURMD_NODENAME:-}" ] && [ -f "$HOME/gurobi_licenses/gurobi.lic.${SLURMD_NODENAME}" ]; then '
    'export GRB_LICENSE_FILE="$HOME/gurobi_licenses/gurobi.lic.${SLURMD_NODENAME}"; '
    'else export GRB_LICENSE_FILE="$HOME/gurobi.lic"; fi'
)


@dataclass(frozen=True)
class SlurmPlan:
    """Self-contained task and metric snapshot consumed by compute workers."""

    name: str
    created_at: str
    source_config: str
    project_root: str
    output_root: str
    metrics: MetricsRunConfig
    tasks: list[BenchmarkTask]
    visualization: VisualizationRunConfig = VisualizationRunConfig()
    matching: MatchingRunConfig = MatchingRunConfig()
    execution: ExecutionConfig = ExecutionConfig()
    schema_version: int = PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": self.schema_version,
                "name": self.name,
                "created_at": self.created_at,
                "source_config": self.source_config,
                "project_root": self.project_root,
                "output_root": self.output_root,
                "metrics": asdict(self.metrics),
                "visualization": asdict(self.visualization),
                "matching": asdict(self.matching),
                "execution": asdict(self.execution),
                "tasks": [asdict(task) for task in self.tasks],
            }
        )

    @cached_property
    def allocations(self) -> list["SlurmAllocation"]:
        """The plan's Slurm allocations, packed once per loaded plan."""
        return _plan_allocations(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SlurmPlan":
        schema_version = int(value.get("schema_version") or 0)
        if schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Slurm plan schema {schema_version}; "
                f"expected {PLAN_SCHEMA_VERSION}. Regenerate the plan with "
                "`python -m benchmark.slurm generate`."
            )
        tasks = [BenchmarkTask.from_dict(task) for task in value.get("tasks", [])]
        plan = cls(
            schema_version=schema_version,
            name=str(value["name"]),
            created_at=str(value["created_at"]),
            source_config=str(value["source_config"]),
            project_root=str(value["project_root"]),
            output_root=str(value["output_root"]),
            metrics=MetricsRunConfig.from_dict(value.get("metrics")),
            tasks=tasks,
            visualization=VisualizationRunConfig.from_dict(value.get("visualization")),
            matching=MatchingRunConfig.from_dict(value.get("matching")),
            execution=ExecutionConfig.from_dict(value.get("execution")),
        )
        _validate_plan(plan)
        return plan


@dataclass(frozen=True)
class SlurmAllocation:
    """A homogeneous group of work tasks sharing one Slurm allocation."""

    phase: str
    task_indices: tuple[int, ...]
    cpus: int
    task_cpus: int


@dataclass(frozen=True)
class SubmittedAllocation:
    """Slurm job identifier for one submitted allocation."""

    allocation_index: int
    phase: str
    job_id: str


def create_plan(config_path: str) -> SlurmPlan:
    """Generate and validate a self-contained plan from one sweep YAML."""

    source_config = Path(config_path).expanduser().resolve()
    sweep = SimulationSweep.from_yaml(str(source_config))
    _validate_sweep(sweep)
    output_root = Path(sweep.execution.output_dir).expanduser().resolve()
    all_tasks = [
        replace(task, output_dir=str(Path(task.output_dir).expanduser().resolve()))
        for task in sweep.generate_tasks()
    ]
    if not all_tasks:
        raise ValueError("Slurm benchmark plan contains no tasks.")
    # A task is planned while any of its phases is left: with skip_existing,
    # a run optimized but not yet matched is planned for its matching alone.
    tasks = [
        task
        for task in all_tasks
        if task_phases(task, sweep.execution, sweep.matching)
    ]

    plan = SlurmPlan(
        name=sweep.name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_config=str(source_config),
        project_root=str(Path(__file__).resolve().parent.parent),
        output_root=str(output_root),
        metrics=sweep.metrics,
        tasks=tasks,
        visualization=sweep.visualization,
        matching=sweep.matching,
        execution=sweep.execution,
    )
    _validate_plan(plan)
    return plan


def write_plan(plan: SlurmPlan, path: str | None = None) -> Path:
    """Atomically write a plan JSON beneath its absolute output root."""

    plan_path = _artifact_path(plan, path, DEFAULT_PLAN_FILENAME)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(plan_path), plan.to_dict())
    return plan_path


def load_plan(path: str) -> SlurmPlan:
    """Load a task snapshot without reading or regenerating its source sweep."""

    import json

    plan_path = Path(path).expanduser().resolve()
    with plan_path.open("r", encoding="utf-8") as f:
        return SlurmPlan.from_dict(json.load(f))


def write_submission_script(
    plan: SlurmPlan,
    plan_path: str | Path,
    path: str | None = None,
) -> Path:
    """Write a dry-runnable shell script containing every ``sbatch`` call."""

    script_path = _artifact_path(plan, path, DEFAULT_SCRIPT_FILENAME)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    _log_dir(plan).mkdir(parents=True, exist_ok=True)
    text = submission_script(plan, Path(plan_path).expanduser().resolve())
    _write_text_atomic(script_path, text, mode=0o755)
    return script_path


def submission_script(plan: SlurmPlan, plan_path: Path) -> str:
    """Render the plan's benchmark allocations as ``sbatch`` calls."""

    _validate_plan(plan)
    job_count = _submission_job_count(plan)
    allocations = plan.allocations
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", GUROBI_LICENSE_SETUP, ""]
    if job_count:
        lines.extend(
            [
                'existing_jobs=$(squeue --noheader --array --user "$(id -un)" --format=%i | wc -l)',
                f"if (( existing_jobs + {job_count} > {MAX_SUBMITTED_SLURM_JOBS} )); then",
                '    printf "Cannot submit: %s existing jobs plus '
                f'{job_count} planned jobs exceeds the {MAX_SUBMITTED_SLURM_JOBS}-job limit.\\n" '
                '"$existing_jobs" >&2',
                "    exit 1",
                "fi",
                "",
            ]
        )
    for index in range(len(allocations)):
        command = _sbatch_options(plan, plan_path, index)
        variable = f"benchmark_job_{index}"
        lines.append(f"{variable}=$({_shell_command(command)})")
        lines.append(f'{variable}="${{{variable}%%;*}}"')
        lines.append(f'printf "%s\\n" "allocation {index}=${{{variable}}}"')
        lines.append("")
    if not allocations:
        lines.append(
            'printf "%s\\n" "All benchmark tasks completed; no jobs to submit."'
        )
    return "\n".join(lines)


def submit_plan(
    plan: SlurmPlan,
    plan_path: str | Path,
    *,
    sbatch: str = "sbatch",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[SubmittedAllocation]:
    """Explicitly submit all planned jobs and return their Slurm identifiers."""

    _validate_plan(plan)
    job_count = _submission_job_count(plan)
    if job_count:
        _check_submission_capacity(job_count, run)
    _log_dir(plan).mkdir(parents=True, exist_ok=True)
    plan_path = Path(plan_path).expanduser().resolve()
    allocations = plan.allocations
    submitted: list[SubmittedAllocation] = []
    for index, allocation in enumerate(allocations):
        job_id = _submit(_sbatch_options(plan, plan_path, index, sbatch=sbatch), run)
        submitted.append(
            SubmittedAllocation(
                allocation_index=index,
                phase=allocation.phase,
                job_id=job_id,
            )
        )
    return submitted


def _submission_job_count(plan: SlurmPlan) -> int:
    """Count the jobs a plan submits: its allocations, matching included."""
    count = len(plan.allocations)
    if count > MAX_SUBMITTED_SLURM_JOBS:
        raise ValueError(
            f"Benchmark Slurm plan has {count} jobs, exceeding the "
            f"{MAX_SUBMITTED_SLURM_JOBS}-job submission limit. Regenerate the plan."
        )
    return count


def _check_submission_capacity(
    job_count: int, run: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    """Check all of the user's queued jobs before submitting any new work."""
    result = run(
        [
            "squeue",
            "--noheader",
            "--array",
            "--user",
            getpass.getuser(),
            "--format=%i",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    existing = sum(bool(line.strip()) for line in result.stdout.splitlines())
    if existing + job_count > MAX_SUBMITTED_SLURM_JOBS:
        raise RuntimeError(
            f"Cannot submit: {existing} existing jobs plus {job_count} planned "
            f"jobs exceeds the {MAX_SUBMITTED_SLURM_JOBS}-job limit. "
            "Wait for queue capacity, then submit again. No jobs were submitted."
        )


def task_phases(
    task: BenchmarkTask,
    execution: ExecutionConfig | None,
    matching: MatchingRunConfig | None,
) -> tuple[str, ...]:
    """The phases of :data:`TASK_PHASES` a task still needs, in order.

    Without ``skip_existing`` every phase runs. With it, a saved optimization
    is kept; a saved evaluation is kept unless its optimization reruns; and a
    run's own matching is kept unless its optimization reruns (which clears
    it) or it was made for another config (:func:`matching_complete`).
    """
    skip = bool(execution and execution.skip_existing)
    optimize = not (skip and valid_optimization_result(task, execution))
    evaluate = optimize or not (skip and valid_existing_result(task, execution))
    match = bool(matching and matching.enabled) and (
        optimize or not (skip and matching_complete(task, matching))
    )
    needed = {"optimize": optimize, "evaluate": evaluate, "match": match}
    return tuple(phase for phase in TASK_PHASES if needed[phase])


def _run_optimization_task(task: BenchmarkTask) -> TaskResult:
    return run_optimization_phase(task)


def _run_evaluation_task(
    task: BenchmarkTask,
    strict_metrics: bool,
    compute_stage_metrics: bool,
    matching: MatchingRunConfig,
    visualization: VisualizationRunConfig,
) -> TaskResult:
    return evaluate_optimization_task(
        task,
        strict_metrics=strict_metrics,
        compute_stage_metrics=compute_stage_metrics,
        matching=matching,
        visualization=visualization,
    )


def _run_matching_task(task: BenchmarkTask, matching: MatchingRunConfig) -> TaskResult:
    """Match one evaluated task into its own run folder."""
    skipped = run_task_matching(task, matching) == "skipped"
    return TaskResult(
        task_id=task.task_id,
        output_dir=task.output_dir,
        status="SKIPPED" if skipped else "COMPLETE",
        skipped=skipped,
    )


def _submit_phase(executor, plan: SlurmPlan, index: int, phase: str):
    task = _task_at(plan, index)
    if phase == "optimize":
        return executor.submit(_run_optimization_task, task)
    if phase == "evaluate":
        return executor.submit(
            _run_evaluation_task,
            task,
            plan.metrics.strict,
            plan.metrics.compute_stage_metrics,
            plan.matching,
            plan.visualization,
        )
    if phase == "match":
        return executor.submit(_run_matching_task, task, plan.matching)
    raise ValueError(f"Unknown task phase {phase!r}.")


def run_benchmark_worker(plan_path: str, allocation_index: int) -> int:
    """Optimize, evaluate, and match every task in one Slurm allocation.

    Each concurrent slot (``task_cpus`` CPUs) carries one task through its
    remaining phases (:func:`task_phases`). A task's next phase takes the
    slot its last phase freed, ahead of any new task, so a task is evaluated
    and matched as soon as it is optimized and every CPU stays busy. Every
    phase runs in a fresh process (``max_tasks_per_child=1``), so one
    phase's memory is returned before the next starts. Aggregation, the
    sweep's matching reports included, runs once the allocation is done.
    """

    plan: SlurmPlan | None = None
    failed = False
    work_started = False
    try:
        plan = load_plan(plan_path)
        allocation = _allocation_at(plan, allocation_index)
        _validate_worker_cpus(allocation.cpus)
        work_started = True
        slots = max(1, allocation.cpus // allocation.task_cpus)
        remaining = {
            index: deque(
                task_phases(_task_at(plan, index), plan.execution, plan.matching)
            )
            for index in allocation.task_indices
        }
        queued = deque(index for index in allocation.task_indices if remaining[index])
        continuing: deque[int] = deque()
        with ProcessPoolExecutor(max_workers=slots, max_tasks_per_child=1) as executor:
            running = {}
            while queued or continuing or running:
                while len(running) < slots and (continuing or queued):
                    index = continuing.popleft() if continuing else queued.popleft()
                    phase = remaining[index].popleft()
                    future = _submit_phase(executor, plan, index, phase)
                    running[future] = (index, phase)
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    index, phase = running.pop(future)
                    try:
                        _report_task_error(phase, index, future.result())
                    except Exception as exc:
                        print(
                            f"Task {index} {phase} failed: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        failed = True
                        remaining[index].clear()
                        continue
                    if remaining[index]:
                        continuing.append(index)
    except Exception as exc:
        print(f"Benchmark allocation failed: {exc}", file=sys.stderr, flush=True)
        failed = True
    finally:
        if plan is not None and work_started:
            try:
                _aggregate(plan)
            except Exception as exc:
                print(f"Aggregation failed: {exc}", file=sys.stderr, flush=True)
                failed = True
    return 1 if failed or plan is None else 0


def _aggregate(plan: SlurmPlan) -> None:
    """Rebuild the sweep's CSVs, its matching reports too, from every run."""
    aggregate_results(
        plan.output_root,
        summary_csv=plan.metrics.summary_csv,
        stages_csv=plan.metrics.stages_csv,
        matching_dir=plan.matching.assignment_dir if plan.matching.enabled else None,
    )


def _validate_sweep(sweep: SimulationSweep) -> None:
    if sweep.mode != "run":
        raise ValueError(
            f"Slurm benchmark execution supports only mode 'run', not {sweep.mode!r}."
        )


def _validate_plan(plan: SlurmPlan) -> None:
    output_root = Path(plan.output_root)
    if not output_root.is_absolute():
        raise ValueError("Slurm plan output_root must be absolute.")
    if not Path(plan.project_root).is_absolute():
        raise ValueError("Slurm plan project_root must be absolute.")
    if not plan.tasks and not (plan.execution and plan.execution.skip_existing):
        raise ValueError("Slurm benchmark plan contains no tasks.")
    if (
        plan.visualization.artifact_dir
        and not Path(plan.visualization.artifact_dir).is_absolute()
    ):
        raise ValueError("Visualization artifact_dir must be absolute for Slurm.")
    if plan.matching.enabled and not (
        plan.matching.config and Path(plan.matching.config).is_absolute()
    ):
        raise ValueError(
            "Matching-enabled Slurm plans need an absolute matching.config."
        )

    hashes: dict[str, str] = {}
    output_dirs: dict[str, str] = {}
    for task in plan.tasks:
        if task.config_hash in hashes:
            raise ValueError(
                "Duplicate benchmark task config hash "
                f"{task.config_hash}: {hashes[task.config_hash]} and {task.task_id}."
            )
        hashes[task.config_hash] = task.task_id
        normalized_output = str(Path(task.output_dir).expanduser().resolve())
        if normalized_output in output_dirs:
            raise ValueError(
                "Duplicate benchmark task output directory "
                f"{normalized_output}: {output_dirs[normalized_output]} and "
                f"{task.task_id}."
            )
        output_dirs[normalized_output] = task.task_id
        if not Path(task.output_dir).is_absolute():
            raise ValueError(f"Task {task.task_id} output_dir must be absolute.")
        _optimization_cpus(task)
    plan.allocations  # Packing rejects plans with too many worker counts.


def _optimization_cpus(task: BenchmarkTask) -> int:
    cpus = optimization_worker_count(task.config)
    if cpus < 1:
        raise ValueError(f"Task {task.task_id} workers must be at least one.")
    if cpus > MAX_CPUS_PER_NODE:
        raise ValueError(
            f"Task {task.task_id} workers cannot exceed {MAX_CPUS_PER_NODE}."
        )
    return cpus


def _split_indices(indices: list[int], count: int) -> list[tuple[int, ...]]:
    quotient, remainder = divmod(len(indices), count)
    batches = []
    start = 0
    for index in range(count):
        size = quotient + (index < remainder)
        batches.append(tuple(indices[start : start + size]))
        start += size
    return batches


def _plan_allocations(plan: SlurmPlan) -> list[SlurmAllocation]:
    if not plan.tasks:
        return []
    groups: dict[int, list[int]] = defaultdict(list)
    for index, task in enumerate(plan.tasks):
        groups[_optimization_cpus(task)].append(index)
    optimization_limit = MAX_SLURM_JOBS
    if len(groups) > optimization_limit:
        raise ValueError(
            "Slurm benchmark plans support at most "
            f"{optimization_limit} distinct optimization worker counts."
        )

    allocation_counts = {task_cpus: 1 for task_cpus in groups}
    ideal_counts = {
        task_cpus: min(
            len(indices),
            (len(indices) + MAX_CPUS_PER_NODE // task_cpus - 1)
            // (MAX_CPUS_PER_NODE // task_cpus),
        )
        for task_cpus, indices in groups.items()
    }
    while sum(allocation_counts.values()) < optimization_limit:
        candidates = [
            task_cpus
            for task_cpus in groups
            if allocation_counts[task_cpus] < ideal_counts[task_cpus]
        ]
        if not candidates:
            break
        task_cpus = max(
            candidates,
            key=lambda value: (
                (
                    len(groups[value])
                    + allocation_counts[value] * (MAX_CPUS_PER_NODE // value)
                    - 1
                )
                // (allocation_counts[value] * (MAX_CPUS_PER_NODE // value)),
                len(groups[value]) * value,
                value,
            ),
        )
        allocation_counts[task_cpus] += 1

    allocations = []
    for task_cpus, indices in sorted(groups.items()):
        concurrency = MAX_CPUS_PER_NODE // task_cpus
        for task_indices in _split_indices(indices, allocation_counts[task_cpus]):
            allocations.append(
                SlurmAllocation(
                    phase="benchmark",
                    task_indices=task_indices,
                    cpus=min(concurrency, len(task_indices)) * task_cpus,
                    task_cpus=task_cpus,
                )
            )
    return allocations


def _artifact_path(plan: SlurmPlan, value: str | None, default_name: str) -> Path:
    output_root = Path(plan.output_root).resolve()
    path = Path(value).expanduser() if value else Path(SLURM_DIRNAME) / default_name
    if not path.is_absolute():
        path = output_root / path
    path = path.resolve()
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"Slurm artifact must be under output root {output_root}."
        ) from exc
    return path


def _log_dir(plan: SlurmPlan) -> Path:
    return Path(plan.output_root) / SLURM_DIRNAME / "logs"


def _sbatch_options(
    plan: SlurmPlan,
    plan_path: Path,
    index: int,
    *,
    sbatch: str = "sbatch",
) -> list[str]:
    allocation = _allocation_at(plan, index)
    worker_command = f"{GUROBI_LICENSE_SETUP} && " + shlex.join(
        [
            "uv",
            "run",
            "python",
            "-m",
            "benchmark.slurm",
            "worker-allocation",
            "--plan",
            str(plan_path),
            "--allocation-index",
            str(index),
        ]
    )
    log_prefix = _log_dir(plan) / f"{index:02d}-benchmark-%j"
    return [
        sbatch,
        "--parsable",
        "-A",
        SBATCH_ACCOUNT,
        "-p",
        SBATCH_PARTITION,
        f"--job-name=benchmark-{index}",
        "--ntasks=1",
        f"--cpus-per-task={allocation.cpus}",
        f"--mem-per-cpu={SBATCH_MEM_PER_CPU}",
        f"--chdir={plan.project_root}",
        f"--output={log_prefix}.out",
        f"--error={log_prefix}.err",
        f"--export=ALL,{','.join(THREAD_ENVIRONMENT)}",
        "--wrap",
        worker_command,
    ]


def _submit(
    command: Sequence[str],
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    completed = run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    )
    output = str(completed.stdout or "").strip()
    job_id = output.split(";", 1)[0]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise RuntimeError(f"sbatch returned an invalid job id: {output!r}.")
    return job_id


def _shell_command(values: Sequence[str]) -> str:
    return " ".join(shlex.quote(value) for value in values)


def _task_at(plan: SlurmPlan, task_index: int) -> BenchmarkTask:
    if task_index < 0 or task_index >= len(plan.tasks):
        raise IndexError(
            f"Task index {task_index} is outside plan range 0..{len(plan.tasks) - 1}."
        )
    return plan.tasks[task_index]


def _allocation_at(plan: SlurmPlan, allocation_index: int) -> SlurmAllocation:
    allocations = plan.allocations
    if allocation_index < 0 or allocation_index >= len(allocations):
        raise IndexError(
            "Allocation index "
            f"{allocation_index} is outside plan range 0..{len(allocations) - 1}."
        )
    allocation = allocations[allocation_index]
    return allocation


def _validate_worker_cpus(expected: int) -> None:
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated is None:
        return
    try:
        actual = int(allocated)
    except ValueError as exc:
        raise RuntimeError(f"Invalid SLURM_CPUS_PER_TASK={allocated!r}.") from exc
    if actual != expected:
        raise RuntimeError(
            f"Worker expected {expected} CPU(s), but Slurm allocated {actual}."
        )


def _report_task_error(phase: str, index: int, result: TaskResult) -> None:
    """Log a task's saved ERROR without failing its allocation.

    The ERROR is in the task's result and summary rows, and its assignment
    targets carry skip markers, so its matching is recorded as skipped. Only
    a crash fails the allocation.
    """
    if str(result.status).upper() == "ERROR":
        print(
            f"{phase} task {index} recorded ERROR: {result.error_message}",
            file=sys.stderr,
            flush=True,
        )


def _write_text_atomic(path: Path, text: str, *, mode: int) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or submit two-phase SFUSD benchmark jobs on Slurm."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate a plan and sbatch script without contacting Slurm."
    )
    _add_generation_arguments(generate_parser)

    submit_parser = subparsers.add_parser(
        "submit", help="Generate a plan and explicitly submit all jobs with sbatch."
    )
    _add_generation_arguments(submit_parser)
    submit_parser.add_argument(
        "--sbatch", default="sbatch", help="sbatch executable (default: sbatch)."
    )

    worker_parser = subparsers.add_parser(
        "worker-allocation", help="Run one benchmark allocation from a saved plan."
    )
    _add_worker_arguments(worker_parser)
    return parser


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        help="Path to simulation sweep YAML.",
    )
    parser.add_argument(
        "--plan",
        help=(
            "Plan JSON path beneath the output root "
            f"(default: {SLURM_DIRNAME}/{DEFAULT_PLAN_FILENAME})."
        ),
    )
    parser.add_argument(
        "--script",
        help=(
            "Submission script path beneath the output root "
            f"(default: {SLURM_DIRNAME}/{DEFAULT_SCRIPT_FILENAME})."
        ),
    )


def _add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True, help="Absolute saved plan JSON path.")
    parser.add_argument(
        "--allocation-index",
        required=True,
        type=int,
        help="Zero-based allocation index in the plan.",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by launchers and Slurm compute workers."""

    args = _build_parser().parse_args(argv)
    try:
        if args.command in {"generate", "submit"}:
            plan = create_plan(args.config)
            plan_path = write_plan(plan, args.plan)
            script_path = write_submission_script(plan, plan_path, args.script)
            print(f"Plan: {plan_path}")
            print(f"Submission script: {script_path}")
            print(f"Benchmark tasks: {len(plan.tasks)}")
            if plan.matching.enabled:
                print("Matching: in each task's allocation, after its evaluation")
            if args.command == "submit":
                submitted = submit_plan(plan, plan_path, sbatch=args.sbatch)
                if not submitted:
                    print("All benchmark tasks already completed; no jobs submitted.")
                    _aggregate(plan)
                else:
                    for job in submitted:
                        print(
                            f"allocation {job.allocation_index}: {job.phase}={job.job_id}"
                        )
            return 0
        if args.command == "worker-allocation":
            return run_benchmark_worker(args.plan, args.allocation_index)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"benchmark.slurm: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Plan, submit, and execute batched benchmark jobs on Slurm."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
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
    stable_hash,
)
from benchmark.results import aggregate_results
from benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    TaskResult,
    _valid_existing_result,
    evaluate_optimization_task,
    run_optimization_phase,
    valid_existing_result,
    write_json,
)


PLAN_SCHEMA_VERSION = 3
MAX_CPUS_PER_NODE = 40
MAX_SLURM_JOBS = 12
MAX_DOWNSTREAM_SLURM_JOBS = 8
MAX_DOWNSTREAM_ASSIGNMENT_JOBS = 6
MAX_DOWNSTREAM_METRICS_JOBS = 2
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
    assignment_plan_path: str | None = None
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
                "assignment_plan_path": self.assignment_plan_path,
                "execution": asdict(self.execution),
                "tasks": [asdict(task) for task in self.tasks],
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SlurmPlan":
        schema_version = int(value.get("schema_version") or 0)
        if schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Slurm plan schema {schema_version}; "
                f"expected {PLAN_SCHEMA_VERSION}."
            )
        tasks = [
            BenchmarkTask(
                task_id=str(task["task_id"]),
                config_hash=str(task["config_hash"]),
                config=dict(task["config"]),
                output_dir=str(task["output_dir"]),
                capacity_slots=int(task["capacity_slots"]),
            )
            for task in value.get("tasks", [])
        ]
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
            assignment_plan_path=(
                str(value["assignment_plan_path"])
                if value.get("assignment_plan_path")
                else None
            ),
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
    if sweep.execution.skip_existing:
        tasks = [
            task
            for task in all_tasks
            if not _valid_existing_result(task, sweep.execution)
        ]
    else:
        tasks = all_tasks
    assignment_plan_path = None
    if sweep.matching.enabled:
        from assignment.run_custom_config import load_custom_config
        from assignment.slurm import build_generated_zone_slurm_plan

        assignment_config = load_custom_config(sweep.matching.config)
        metrics_enabled = bool(assignment_config.get("export-aggregate-metrics", False))
        assignment_jobs = (
            MAX_DOWNSTREAM_ASSIGNMENT_JOBS
            if metrics_enabled
            else MAX_DOWNSTREAM_SLURM_JOBS
        )
        metrics_jobs = MAX_DOWNSTREAM_METRICS_JOBS if metrics_enabled else 0
        _assignment_plan, generated_plan_path = build_generated_zone_slurm_plan(
            sweep.matching.config,
            _assignment_targets(all_tasks, sweep.matching),
            assignment_folder=output_root / "assignments",
            plan_dir=output_root / SLURM_DIRNAME / "assignment",
            max_assignment_jobs=assignment_jobs,
            max_metrics_jobs=metrics_jobs,
        )
        assignment_plan_path = str(generated_plan_path)

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
        assignment_plan_path=assignment_plan_path,
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
    """Render benchmark allocations and the dependent assignment graph."""

    _validate_plan(plan)
    allocations = _plan_allocations(plan)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'if [ -f "$HOME/gurobi_licenses/gurobi.lic.$(hostname -s)" ]; then',
        '    export GRB_LICENSE_FILE="$HOME/gurobi_licenses/gurobi.lic.$(hostname -s)"',
        'elif [ -n "${SLURMD_NODENAME:-}" ] && [ -f "$HOME/gurobi_licenses/gurobi.lic.${SLURMD_NODENAME}" ]; then',
        '    export GRB_LICENSE_FILE="$HOME/gurobi_licenses/gurobi.lic.${SLURMD_NODENAME}"',
        'else',
        '    export GRB_LICENSE_FILE="$HOME/gurobi.lic"',
        'fi',
        "",
    ]
    for index, _allocation in enumerate(allocations):
        command = _sbatch_options(plan, plan_path, index)
        variable = f"benchmark_job_{index}"
        lines.append(f"{variable}=$({_shell_command(command)})")
        lines.append(f'{variable}="${{{variable}%%;*}}"')
        lines.append(f'printf "%s\\n" "allocation {index}=${{{variable}}}"')
        lines.append("")
    if plan.assignment_plan_path:
        from assignment.slurm import write_slurm_scripts

        assignment_submit = write_slurm_scripts(plan.assignment_plan_path)
        command = [shlex.quote(str(assignment_submit))]
        for index in range(len(allocations)):
            command.extend(["--upstream-job-id", f'"${{benchmark_job_{index}}}"'])
        lines.append(" ".join(command))
        lines.append("")
    elif not allocations:
        lines.append('printf "%s\\n" "All benchmark tasks completed; no jobs to submit."')
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
    _log_dir(plan).mkdir(parents=True, exist_ok=True)
    plan_path = Path(plan_path).expanduser().resolve()
    allocations = _plan_allocations(plan)
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
    if plan.assignment_plan_path:
        import json

        from assignment.slurm import submit_slurm_plan, write_slurm_scripts

        assignment_submit = write_slurm_scripts(plan.assignment_plan_path)
        submission_path = submit_slurm_plan(
            plan.assignment_plan_path,
            script_dir=assignment_submit.parent,
            runner=run,
            upstream_job_ids=[item.job_id for item in submitted],
            sbatch=sbatch,
        )
        with submission_path.open(encoding="utf-8") as stream:
            assignment_submission = json.load(stream)
        offset = len(submitted)
        from assignment.slurm import load_plan as load_assignment_plan

        assignment_plan, _ = load_assignment_plan(plan.assignment_plan_path)
        jobs_by_id = {job["id"]: job for job in assignment_plan["jobs"]}
        submitted.extend(
            SubmittedAllocation(
                allocation_index=offset + index,
                phase=jobs_by_id[item["job_id"]]["kind"],
                job_id=item["slurm_job_id"],
            )
            for index, item in enumerate(assignment_submission["jobs"])
        )
    return submitted


def _run_optimization_task(
    task: BenchmarkTask,
    execution: ExecutionConfig | None = None,
) -> TaskResult:
    if execution and execution.skip_existing and _valid_existing_result(task, execution):
        return TaskResult(
            task_id=task.task_id,
            output_dir=task.output_dir,
            status="SKIPPED",
            skipped=True,
        )
    return run_optimization_phase(task)


def _run_evaluation_task(
    task: BenchmarkTask,
    strict_metrics: bool,
    compute_stage_metrics: bool,
    matching: MatchingRunConfig,
    visualization: VisualizationRunConfig,
    execution: ExecutionConfig | None = None,
) -> TaskResult:
    if execution and execution.skip_existing and _valid_existing_result(task, execution):
        return TaskResult(
            task_id=task.task_id,
            output_dir=task.output_dir,
            status="SKIPPED",
            skipped=True,
        )
    return evaluate_optimization_task(
        task,
        strict_metrics=strict_metrics,
        compute_stage_metrics=compute_stage_metrics,
        matching=matching,
        visualization=visualization,
    )


def run_benchmark_worker(plan_path: str, allocation_index: int) -> int:
    """Optimize and then evaluate every task in one Slurm allocation."""

    plan: SlurmPlan | None = None
    failed = False
    work_started = False
    try:
        plan = load_plan(plan_path)
        allocation = _allocation_at(plan, allocation_index)
        _validate_worker_cpus(allocation.cpus)
        work_started = True
        with ProcessPoolExecutor(
            max_workers=allocation.cpus // allocation.task_cpus
        ) as executor:
            futures = {
                executor.submit(
                    _run_optimization_task,
                    _task_at(plan, index),
                    plan.execution,
                ): index
                for index in allocation.task_indices
            }
            for future in as_completed(futures):
                try:
                    failed = _result_exit_code(future.result()) != 0 or failed
                except Exception as exc:
                    print(
                        f"Optimization task {futures[future]} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    failed = True
        with ProcessPoolExecutor(max_workers=allocation.cpus) as executor:
            futures = {
                executor.submit(
                    _run_evaluation_task,
                    _task_at(plan, index),
                    plan.metrics.strict,
                    plan.metrics.compute_stage_metrics,
                    plan.matching,
                    plan.visualization,
                    plan.execution,
                ): index
                for index in allocation.task_indices
            }
            for future in as_completed(futures):
                try:
                    failed = _result_exit_code(future.result()) != 0 or failed
                except Exception as exc:
                    print(
                        f"Evaluation task {futures[future]} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    failed = True
    except Exception as exc:
        print(f"Benchmark allocation failed: {exc}", file=sys.stderr, flush=True)
        failed = True
    finally:
        if plan is not None and work_started:
            try:
                aggregate_results(
                    plan.output_root,
                    summary_csv=plan.metrics.summary_csv,
                    stages_csv=plan.metrics.stages_csv,
                )
            except Exception as exc:
                print(f"Aggregation failed: {exc}", file=sys.stderr, flush=True)
                failed = True
    return 1 if failed or plan is None else 0


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
    if (
        not plan.tasks
        and not plan.assignment_plan_path
        and not (plan.execution and plan.execution.skip_existing)
    ):
        raise ValueError("Slurm benchmark plan contains no tasks.")
    if (
        plan.visualization.artifact_dir
        and not Path(plan.visualization.artifact_dir).is_absolute()
    ):
        raise ValueError("Visualization artifact_dir must be absolute for Slurm.")
    if plan.matching.enabled:
        if not plan.assignment_plan_path:
            raise ValueError("Matching-enabled plans require an assignment Slurm plan.")
        if not Path(plan.assignment_plan_path).is_absolute():
            raise ValueError("Assignment Slurm plan path must be absolute.")

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
    _plan_allocations(plan)


def _optimization_cpus(task: BenchmarkTask) -> int:
    cpus = int(task.config.get("workers", 1))
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


def _assignment_targets(
    tasks: list[BenchmarkTask], matching: MatchingRunConfig
) -> list[dict[str, str]]:
    from assignment.generated_zones import (
        GENERATED_ZONE_FILENAME,
        SKIP_MARKER_FILENAME,
    )
    from benchmark.assignment import zone_building_blocks
    from loaders import load_scenario
    from optimization.levels import LevelSpec

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
        folders = [("root", Path(task.output_dir))]
        if matching.compute_stage_assignments:
            strategy = str(task.config["strategy"])
            normalized = strategy.lower()
            prefix = (
                "iteration"
                if "iterative" in normalized or normalized == "saa"
                else "stage"
            )
            levels = task.config["levels"]
            if normalized == "saa":
                levels = [levels[-1]] * (int(task.config["max_iterations"]) + 1)
            folders.extend(
                (
                    f"stage-{index}",
                    Path(task.output_dir) / "stages" / f"{prefix}_{index:02d}_{level}",
                )
                for index, level in enumerate(levels)
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
    license_setup = (
        'if [ -f "$HOME/gurobi_licenses/gurobi.lic.$(hostname -s)" ]; then '
        'export GRB_LICENSE_FILE="$HOME/gurobi_licenses/gurobi.lic.$(hostname -s)"; '
        'elif [ -n "${SLURMD_NODENAME:-}" ] && [ -f "$HOME/gurobi_licenses/gurobi.lic.${SLURMD_NODENAME}" ]; then '
        'export GRB_LICENSE_FILE="$HOME/gurobi_licenses/gurobi.lic.${SLURMD_NODENAME}"; '
        'else export GRB_LICENSE_FILE="$HOME/gurobi.lic"; fi && '
    )
    worker_command = (
        license_setup
        + shlex.join(
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
    allocations = _plan_allocations(plan)
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


def _result_exit_code(result: TaskResult) -> int:
    return 1 if str(result.status).upper() == "ERROR" else 0


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
            if plan.assignment_plan_path:
                from assignment.slurm import load_plan as load_assignment_plan

                assignment_plan, _ = load_assignment_plan(plan.assignment_plan_path)
                print(
                    f"Assignment tasks: {len(assignment_plan['assignment_tasks'])} "
                    f"across {len(assignment_plan['jobs'])} Slurm jobs"
                )
            if args.command == "submit":
                if not plan.tasks and not plan.assignment_plan_path:
                    print("All benchmark tasks already completed; no jobs submitted.")
                    aggregate_results(
                        plan.output_root,
                        summary_csv=plan.metrics.summary_csv,
                        stages_csv=plan.metrics.stages_csv,
                    )
                    return 0
                submitted = submit_plan(plan, plan_path, sbatch=args.sbatch)
                if not submitted:
                    print("All benchmark tasks already completed; no jobs submitted.")
                    aggregate_results(
                        plan.output_root,
                        summary_csv=plan.metrics.summary_csv,
                        stages_csv=plan.metrics.stages_csv,
                    )
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

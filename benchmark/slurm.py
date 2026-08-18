"""Plan, submit, and execute two-phase benchmark jobs on Slurm."""

from __future__ import annotations

import argparse
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
    MetricsRunConfig,
    SimulationSweep,
    json_ready,
)
from benchmark.results import aggregate_results
from benchmark.runner import (
    TaskResult,
    evaluate_optimization_task,
    run_optimization_phase,
    write_json,
)


PLAN_SCHEMA_VERSION = 1
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
        )
        _validate_plan(plan)
        return plan


@dataclass(frozen=True)
class SubmittedTask:
    """Slurm job identifiers for one planned benchmark task."""

    task_id: str
    optimization_job_id: str
    evaluation_job_id: str


def create_plan(config_path: str) -> SlurmPlan:
    """Generate and validate a self-contained plan from one sweep YAML."""

    source_config = Path(config_path).expanduser().resolve()
    sweep = SimulationSweep.from_yaml(str(source_config))
    _validate_sweep(sweep)
    output_root = Path(sweep.execution.output_dir).expanduser().resolve()
    tasks = [
        replace(task, output_dir=str(Path(task.output_dir).expanduser().resolve()))
        for task in sweep.generate_tasks()
    ]
    plan = SlurmPlan(
        name=sweep.name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_config=str(source_config),
        project_root=str(Path(__file__).resolve().parent.parent),
        output_root=str(output_root),
        metrics=sweep.metrics,
        tasks=tasks,
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
    """Render independent optimization/evaluation submissions with dependencies."""

    _validate_plan(plan)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for index, task in enumerate(plan.tasks):
        optimization = _sbatch_options(
            plan,
            plan_path,
            index,
            phase="optimize",
        )
        evaluation = _sbatch_options(
            plan,
            plan_path,
            index,
            phase="evaluate",
        )
        opt_var = f"optimization_job_{index}"
        eval_var = f"evaluation_job_{index}"
        lines.append(f"{opt_var}=$({_shell_command(optimization)})")
        lines.append(f'{opt_var}="${{{opt_var}%%;*}}"')
        dependency = f'--dependency="afterany:${{{opt_var}}}"'
        evaluation_command = _shell_command(evaluation[:6])
        remaining = " ".join(shlex.quote(value) for value in evaluation[6:])
        lines.append(
            f"{eval_var}=$({evaluation_command} {dependency} {remaining})"
        )
        lines.append(f'{eval_var}="${{{eval_var}%%;*}}"')
        lines.append(
            f'printf "%s\\n" "{task.task_id} optimize=${{{opt_var}}} '
            f'evaluate=${{{eval_var}}}"'
        )
        lines.append("")
    return "\n".join(lines)


def submit_plan(
    plan: SlurmPlan,
    plan_path: str | Path,
    *,
    sbatch: str = "sbatch",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[SubmittedTask]:
    """Explicitly submit all planned jobs and return their Slurm identifiers."""

    _validate_plan(plan)
    _log_dir(plan).mkdir(parents=True, exist_ok=True)
    plan_path = Path(plan_path).expanduser().resolve()
    submitted: list[SubmittedTask] = []
    for index, task in enumerate(plan.tasks):
        optimize_command = _sbatch_options(
            plan, plan_path, index, phase="optimize", sbatch=sbatch
        )
        optimize_job_id = _submit(optimize_command, run)
        evaluate_command = _sbatch_options(
            plan, plan_path, index, phase="evaluate", sbatch=sbatch
        )
        evaluate_command.insert(6, f"--dependency=afterany:{optimize_job_id}")
        evaluation_job_id = _submit(evaluate_command, run)
        submitted.append(
            SubmittedTask(
                task_id=task.task_id,
                optimization_job_id=optimize_job_id,
                evaluation_job_id=evaluation_job_id,
            )
        )
    return submitted


def run_optimization_worker(plan_path: str, task_index: int) -> int:
    """Run one planned optimization allocation, returning nonzero on errors."""

    try:
        plan = load_plan(plan_path)
        task = _task_at(plan, task_index)
        expected_cpus = _optimization_cpus(task)
        _validate_worker_cpus(expected_cpus)
        result = run_optimization_phase(task)
        return _result_exit_code(result)
    except Exception as exc:
        print(f"Optimization worker failed: {exc}", file=sys.stderr, flush=True)
        return 1


def run_evaluation_worker(plan_path: str, task_index: int) -> int:
    """Evaluate one planned run and safely rebuild shared aggregate CSVs."""

    plan: SlurmPlan | None = None
    result: TaskResult | None = None
    failed = False
    try:
        plan = load_plan(plan_path)
        task = _task_at(plan, task_index)
        _validate_worker_cpus(1)
        result = evaluate_optimization_task(
            task,
            strict_metrics=plan.metrics.strict,
            compute_stage_metrics=plan.metrics.compute_stage_metrics,
        )
        failed = _result_exit_code(result) != 0
    except Exception as exc:
        print(f"Evaluation worker failed: {exc}", file=sys.stderr, flush=True)
        failed = True
    finally:
        if plan is not None:
            try:
                aggregate_results(
                    plan.output_root,
                    summary_csv=plan.metrics.summary_csv,
                    stages_csv=plan.metrics.stages_csv,
                )
            except Exception as exc:
                print(f"Aggregation failed: {exc}", file=sys.stderr, flush=True)
                failed = True
    return 1 if failed or result is None else 0


def _validate_sweep(sweep: SimulationSweep) -> None:
    if sweep.mode != "run":
        raise ValueError(
            f"Slurm benchmark execution supports only mode 'run', not {sweep.mode!r}."
        )
    if sweep.matching.enabled:
        raise ValueError("Slurm benchmark execution does not support enabled matching.")
    if sweep.choice_metrics.enabled:
        raise ValueError(
            "Slurm benchmark execution does not support enabled assignment-based "
            "choice_metrics."
        )


def _validate_plan(plan: SlurmPlan) -> None:
    output_root = Path(plan.output_root)
    if not output_root.is_absolute():
        raise ValueError("Slurm plan output_root must be absolute.")
    if not Path(plan.project_root).is_absolute():
        raise ValueError("Slurm plan project_root must be absolute.")
    if not plan.tasks:
        raise ValueError("Slurm benchmark plan contains no tasks.")

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


def _optimization_cpus(task: BenchmarkTask) -> int:
    cpus = int(task.config.get("workers", 1))
    if cpus < 1:
        raise ValueError(f"Task {task.task_id} workers must be at least one.")
    return cpus


def _artifact_path(plan: SlurmPlan, value: str | None, default_name: str) -> Path:
    output_root = Path(plan.output_root).resolve()
    path = Path(value).expanduser() if value else Path(SLURM_DIRNAME) / default_name
    if not path.is_absolute():
        path = output_root / path
    path = path.resolve()
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"Slurm artifact must be under output root {output_root}.") from exc
    return path


def _log_dir(plan: SlurmPlan) -> Path:
    return Path(plan.output_root) / SLURM_DIRNAME / "logs"


def _sbatch_options(
    plan: SlurmPlan,
    plan_path: Path,
    index: int,
    *,
    phase: str,
    sbatch: str = "sbatch",
) -> list[str]:
    task = _task_at(plan, index)
    evaluate = phase == "evaluate"
    if not evaluate and phase != "optimize":
        raise ValueError(f"Unknown Slurm phase {phase!r}.")
    cpus = 1 if evaluate else _optimization_cpus(task)
    short_phase = "eval" if evaluate else "opt"
    worker = "worker-evaluate" if evaluate else "worker-optimize"
    worker_command = shlex.join(
        [
            "uv",
            "run",
            "python",
            "-m",
            "benchmark.slurm",
            worker,
            "--plan",
            str(plan_path),
            "--task-index",
            str(index),
        ]
    )
    log_prefix = _log_dir(plan) / f"{index:04d}-{task.task_id}-{short_phase}-%j"
    return [
        sbatch,
        "--parsable",
        "-A",
        SBATCH_ACCOUNT,
        "-p",
        SBATCH_PARTITION,
        f"--job-name=bench-{short_phase}-{index}-{task.task_id}",
        "--ntasks=1",
        f"--cpus-per-task={cpus}",
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

    plan_parser = subparsers.add_parser(
        "plan", help="Generate a plan and sbatch script without contacting Slurm."
    )
    _add_generation_arguments(plan_parser)

    submit_parser = subparsers.add_parser(
        "submit", help="Generate a plan and explicitly submit all jobs with sbatch."
    )
    _add_generation_arguments(submit_parser)
    submit_parser.add_argument(
        "--sbatch", default="sbatch", help="sbatch executable (default: sbatch)."
    )

    optimize_parser = subparsers.add_parser(
        "worker-optimize", help="Run one optimization task from a saved plan."
    )
    _add_worker_arguments(optimize_parser)

    evaluate_parser = subparsers.add_parser(
        "worker-evaluate", help="Evaluate one task and rebuild aggregate CSVs."
    )
    _add_worker_arguments(evaluate_parser)
    return parser


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", help="Path to simulation sweep YAML.")
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
        "--task-index", required=True, type=int, help="Zero-based task index in the plan."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by launchers and Slurm compute workers."""

    args = _build_parser().parse_args(argv)
    try:
        if args.command in {"plan", "submit"}:
            plan = create_plan(args.config)
            plan_path = write_plan(plan, args.plan)
            script_path = write_submission_script(plan, plan_path, args.script)
            print(f"Plan: {plan_path}")
            print(f"Submission script: {script_path}")
            print(f"Tasks: {len(plan.tasks)}")
            if args.command == "submit":
                submitted = submit_plan(plan, plan_path, sbatch=args.sbatch)
                for job in submitted:
                    print(
                        f"{job.task_id}: optimize={job.optimization_job_id} "
                        f"evaluate={job.evaluation_job_id}"
                    )
            return 0
        if args.command == "worker-optimize":
            return run_optimization_worker(args.plan, args.task_index)
        if args.command == "worker-evaluate":
            return run_evaluation_worker(args.plan, args.task_index)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"benchmark.slurm: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

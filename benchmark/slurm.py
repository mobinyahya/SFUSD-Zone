"""Plan, submit, and execute batched benchmark jobs on Slurm."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import cached_property
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

from benchmark.config import (
    BenchmarkTask,
    ExecutionConfig,
    MatchingRunConfig,
    MetricsRunConfig,
    SimulationSweep,
    VisualizationRunConfig,
    json_ready,
    optimization_worker_count,
    stable_hash,
)
from benchmark.assignment import matching_complete, run_task_matching
from benchmark.results import aggregate_results
from benchmark.runner import (
    TaskResult,
    evaluate_optimization_task,
    run_optimization_phase,  # noqa: F401 - tests patch it to check it is not used
    run_optimize_and_evaluate,
    timed_phase,
    valid_existing_result,
    valid_optimization_result,
    write_json,
)
from benchmark.warmup import (
    WarmSpec,
    dataset_key,
    format_warm_summary,
    market_key,
    warm_dataset,
    warm_specs_for_tasks,
)


# 4: execution.queue_multiplier removed.
# 5: matching runs inside the optimization allocations; no assignment plan.
# 6: tasks are in claim order and every allocation of a worker count claims
#    from that count's one shared queue (claim files), not a fixed chunk.
# 7: tasks are claimed one at a time as slots free; a claim records whether
#    its task's process started (``started``, ``claimed_epoch``) and can be
#    ``released`` by its allocation.
PLAN_SCHEMA_VERSION = 7
MAX_CPUS_PER_NODE = 40
#: The partition's per-user job limit. Every job a plan submits is an
#: optimization allocation: each task is optimized, evaluated, and matched
#: inside its allocation, so there is no downstream job graph.
MAX_SLURM_JOBS = 12
MAX_SUBMITTED_SLURM_JOBS = MAX_SLURM_JOBS
#: Memory per allocated CPU. A task's slot runs one process at a time:
#: optimize and evaluate share one (evaluate reuses optimize's dataset after
#: the solver is released, peaking near optimize's ~1.6 GB), then match gets
#: a fresh one (~0.5 GB). The cache warm-up runs one process per dataset in a
#: slot before the fan-out, peaking like an optimize. Nothing stacks phases.
SBATCH_MEM_PER_CPU = "4G"
#: The phases of one task, in order.
TASK_PHASES = ("optimize", "evaluate", "match")
#: How phases map to processes: optimize and evaluate fused in one, when a
#: task needs both; evaluate alone (resuming a run optimized but not
#: evaluated) and match each in their own.
FUSED_OPTIMIZE_EVALUATE = "optimize+evaluate"
SLURM_DIRNAME = ".slurm"
#: ``<output_root>/.slurm/claims/<task_id>.<generation>``: who runs a task.
CLAIMS_DIRNAME = "claims"
#: How often an allocation touches the claims of the tasks it holds.
CLAIM_HEARTBEAT_SECONDS = 60.0
#: A claim untouched this long belongs to a dead allocation.
STALE_CLAIM_SECONDS = 15 * 60.0
#: How often an allocation with an empty queue rechecks others' claims.
CLAIM_RECOVERY_SECONDS = 5 * 60.0
#: A task process that has not reported its start this long after its
#: allocation claimed (or continued) the task has its claim released, so a
#: stalled node does not sit on work a healthy one could run.
PHASE_START_TIMEOUT_SECONDS = 10 * 60.0
#: How often the allocation loop wakes to check starts and heartbeats.
POLL_SECONDS = 15.0
#: How often an idle allocation rechecks claims another allocation may yet
#: release (not started) or whose job is gone, before it exits.
IDLE_POLL_SECONDS = 30.0
#: The longest an idle allocation waits on such claims before exiting. Each
#: one is resolved sooner: released by its owner within
#: PHASE_START_TIMEOUT_SECONDS, stale at the next squeue read after its job
#: died, or stale by heartbeat within STALE_CLAIM_SECONDS.
EXIT_WAIT_LIMIT_SECONDS = 30 * 60.0
#: At most one squeue read per allocation this often (recovery checks).
LIVE_JOBS_REFRESH_SECONDS = 60.0
#: Delay between the first-wave task processes' starts, so 40 processes do
#: not open the same files on NFS in the same instant.
WORKER_START_STAGGER_SECONDS = 1.0
#: Set to ``0`` to skip the allocation's cache warm-up.
WARM_ENVIRONMENT = "BENCHMARK_SLURM_WARM"
#: The phase processes' start method: ``forkserver`` (default) or ``spawn``.
START_METHOD_ENVIRONMENT = "BENCHMARK_SLURM_START_METHOD"
#: What the forkserver imports once per allocation, so a phase process forks
#: with the environment loaded instead of importing it from NFS. Only
#: imports: none of these starts a thread or keeps a file open at import,
#: except pyproj's PROJ database, which each process reopens
#: (:func:`init_worker_process`). The numba kernels of
#: ``optimization.lottery_da`` are imported, not compiled or loaded from
#: their cache: that happens on first call, in the phase process.
FORKSERVER_PRELOAD = (
    "numpy",
    "scipy",
    "pandas",
    "shapely",
    "pyproj",
    "geopandas",
    "networkx",
    "numba",
    "ortools.sat.python.cp_model",
    "loaders.tables",
    "optimization.lottery_da",
    "optimization.data.mid",
    "metrics",
    "matplotlib.pyplot",
    "yamale",
    "optimization.visualization",
    "assignment.student_assignment.evaluation.full_report",
    "assignment.student_assignment.evaluation.heatmaps",
    "benchmark.slurm",
)
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


    @property
    def plan_id(self) -> str:
        """Identifies this plan in claims, so a resubmit reclaims done tasks."""
        return stable_hash(
            {
                "name": self.name,
                "created_at": self.created_at,
                "source_config": self.source_config,
                "output_root": self.output_root,
            }
        )[:16]


@dataclass(frozen=True)
class SlurmAllocation:
    """One Slurm allocation and the shared queue it claims tasks from.

    ``task_indices`` is the queue of every task with this allocation's
    ``task_cpus``, in claim order; every allocation of that worker count
    claims from the same queue. ``cpus`` is sized from the allocation's
    even share of it.
    """

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
    tasks = claim_order(
        [
            task
            for task in all_tasks
            if task_phases(task, sweep.execution, sweep.matching)
        ]
    )

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
    """Render the plan's benchmark allocations as ``sbatch`` calls.

    Each allocation is submitted on its own: one refused by the job limit is
    reported and the rest still go in. Every allocation claims from the same
    queue, so a sweep runs on however many are accepted; the rest can be added
    later with ``python -m benchmark.slurm topup --plan PLAN``.
    """

    _validate_plan(plan)
    _submission_job_count(plan)
    allocations = plan.allocations
    lines = ["#!/usr/bin/env bash", "set -uo pipefail", GUROBI_LICENSE_SETUP, ""]
    lines.append("refused=0")
    for index in range(len(allocations)):
        command = _sbatch_options(plan, plan_path, index)
        lines.append(f"if job=$({_shell_command(command)}); then")
        lines.append(f'    printf "%s\\n" "allocation {index}=${{job%%;*}}"')
        lines.append("else")
        lines.append(f'    printf "%s\\n" "allocation {index} not submitted" >&2')
        lines.append("    refused=$((refused + 1))")
        lines.append("fi")
        lines.append("")
    if not allocations:
        lines.append(
            'printf "%s\\n" "All benchmark tasks completed; no jobs to submit."'
        )
    else:
        lines.extend(
            [
                "if (( refused )); then",
                '    printf "%s\\n" "$refused allocation(s) not submitted; the rest '
                "share the queue. Add more later with: uv run python -m benchmark.slurm "
                f'topup --plan {shlex.quote(str(plan_path))}" >&2',
                "fi",
            ]
        )
    return "\n".join(lines)


_PLAN_COMMENT_PREFIX = "sfusd-plan:"
#: sbatch's message when a job would exceed a user or QOS limit.
_SUBMIT_LIMIT_MARKERS = ("QOSMaxSubmitJobPerUserLimit", "job submit limit", "QOSMax")


def submit_plan(
    plan: SlurmPlan,
    plan_path: str | Path,
    *,
    sbatch: str = "sbatch",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    indices: Sequence[int] | None = None,
) -> list[SubmittedAllocation]:
    """Submit the plan's allocations (or ``indices``) and return their job ids.

    Allocations are submitted one at a time. When Slurm refuses one for a job
    limit, submission stops there and the accepted ones are returned: every
    allocation claims from the plan's one queue, so the sweep runs on those,
    and ``topup`` adds the rest once the limit allows. Any other sbatch error
    is raised.
    """

    _validate_plan(plan)
    _submission_job_count(plan)
    _log_dir(plan).mkdir(parents=True, exist_ok=True)
    plan_path = Path(plan_path).expanduser().resolve()
    allocations = plan.allocations
    wanted = range(len(allocations)) if indices is None else list(indices)
    submitted: list[SubmittedAllocation] = []
    for index in wanted:
        try:
            job_id = _submit(
                _sbatch_options(plan, plan_path, index, sbatch=sbatch), run
            )
        except subprocess.CalledProcessError as exc:
            message = f"{exc.stderr or ''}{exc.stdout or ''}"
            if any(marker in message for marker in _SUBMIT_LIMIT_MARKERS):
                left = len(wanted) - len(submitted)
                print(
                    f"Slurm refused allocation {index} for a job limit; "
                    f"{left} allocation(s) not submitted. The {len(submitted)} "
                    "submitted share the queue; add the rest later with "
                    f"`python -m benchmark.slurm topup --plan {plan_path}`.",
                    file=sys.stderr,
                )
                break
            raise
        submitted.append(
            SubmittedAllocation(
                allocation_index=index,
                phase=allocations[index].phase,
                job_id=job_id,
            )
        )
    return submitted


def live_plan_allocations(
    plan: SlurmPlan,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[int, str]:
    """The plan's pending or running allocations: ``{index: job id}``."""
    result = run(
        [
            "squeue",
            "--noheader",
            "--user",
            getpass.getuser(),
            "--states=PENDING,RUNNING,CONFIGURING,REQUEUED,RESIZING,SUSPENDED",
            "--format=%i|%j|%k",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tag = f"{_PLAN_COMMENT_PREFIX}{plan.plan_id}"
    live: dict[int, str] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|", 2)
        if len(parts) != 3 or parts[2].strip() != tag:
            continue
        job_id, name, _ = parts
        match = re.fullmatch(r"benchmark-(\d+)", name.strip())
        if match:
            live[int(match.group(1))] = job_id.strip()
    return live


def topup_plan(
    plan: SlurmPlan,
    plan_path: str | Path,
    *,
    sbatch: str = "sbatch",
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[int, str], list[SubmittedAllocation]]:
    """Submit the plan's allocations that are not pending or running.

    Use it after a job limit refused part of a submission, or after
    allocations died: the new ones claim whatever the queue still holds,
    including the dead ones' tasks once their claims go stale.
    """
    live = live_plan_allocations(plan, run)
    missing = [index for index in range(len(plan.allocations)) if index not in live]
    return live, submit_plan(plan, plan_path, sbatch=sbatch, run=run, indices=missing)


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


def execution_units(phases: Sequence[str]) -> tuple[str, ...]:
    """The processes a task's remaining phases run in, in order.

    Optimize and evaluate share one process when a task needs both
    (:data:`FUSED_OPTIMIZE_EVALUATE`), so evaluation reuses the dataset and
    stages optimization holds. Match always gets its own fresh process.
    """
    units = list(phases)
    if units[:2] == ["optimize", "evaluate"]:
        units[:2] = [FUSED_OPTIMIZE_EVALUATE]
    elif units[:1] == ["optimize"]:
        raise ValueError("A task that is optimized is always evaluated.")
    return tuple(units)


def expected_task_seconds(config: Mapping[str, Any]) -> float:
    """A task's expected cost: the time limits it will spend, in seconds.

    A pipeline spends each step's limit; ``recursive`` one limit per level;
    every other strategy runs the last entry as one budget.
    """
    steps = config.get("steps") or []
    if steps:
        return float(sum(float(step.get("time_limit") or 0.0) for step in steps))
    limits = [float(value) for value in config.get("solve_time_limits") or []]
    if not limits:
        return 0.0
    if config.get("strategy") == "recursive":
        return float(sum(limits))
    return limits[-1]


def claim_order(tasks: Sequence[BenchmarkTask]) -> list[BenchmarkTask]:
    """Order tasks as allocations will claim them.

    Longest expected first (:func:`expected_task_seconds`), so the long
    tasks are not left to the end; among equally long tasks, those sharing
    a policy market (:func:`benchmark.warmup.market_key`: the data block and
    ``matching_policy``) are adjacent, in the order their markets first
    appear; otherwise the sweep's order is kept.
    """
    first_seen: dict[str, int] = {}
    keyed = []
    for index, task in enumerate(tasks):
        key = market_key(task.config)
        first_seen.setdefault(key, index)
        keyed.append(
            (-expected_task_seconds(task.config), first_seen[key], index, task)
        )
    keyed.sort(key=lambda item: item[:3])
    return [item[3] for item in keyed]


# ---------------------------------------------------------------------------
# Claims: one plan-wide queue shared by every allocation.

CLAIMED = "claimed"
HELD = "held"
DONE = "done"
STALE = "stale"
#: A claim its allocation gave back (:meth:`ClaimStore.release`): stale for
#: everyone, so the next claimant takes the next generation.
RELEASED = "released"


def _job_identity() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return str(job_id)
    return f"local-{socket.gethostname()}-{os.getpid()}"


#: squeue states of a job that can still run its claims. A cancelled job
#: lingers in squeue as COMPLETING for a while; it is not live.
LIVE_JOB_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "SUSPENDED",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
    }
)


def live_slurm_jobs(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> frozenset[str] | None:
    """The user's live (queued or running) job ids, or ``None`` if squeue fails."""
    try:
        result = run(
            [
                "squeue",
                "--noheader",
                "--user",
                getpass.getuser(),
                "--format=%A %i %T",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    jobs = set()
    for line in str(result.stdout or "").splitlines():
        values = line.split()
        if len(values) >= 3:
            if values[-1].upper() not in LIVE_JOB_STATES:
                continue
            values = values[:-1]
        for value in values:
            jobs.add(value.strip())
            jobs.add(value.strip().split("_", 1)[0])
    jobs.discard("")
    return frozenset(jobs)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_claim(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            claim = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}  # Being written: judged by its age alone.
    return claim if isinstance(claim, dict) else {}


class ClaimStore:
    """Atomic task claims in ``<output_root>/.slurm/claims`` on shared NFS.

    A claim is the file ``<task_id>.<generation>``, made with
    ``O_CREAT | O_EXCL`` (atomic on NFSv3+, so no locks), holding JSON: the
    plan id, Slurm job id, host, pid, allocation index, ``claimed_at`` (and
    ``claimed_epoch``, on the file server's clock), ``started`` (``false``
    until the task's first process reports in, which rewrites the claim
    with ``started_at``), and ``status`` (``running``, then ``done``, or
    ``released`` when the allocation gave it back). The highest generation
    is the task's claim. A stale one is taken over by creating the next
    generation, which again only one claimant can do; files are never
    deleted, so a generation, once seen, stays taken.

    A ``running`` claim is stale when its job (a numeric Slurm id) is not
    live in this allocation's latest ``squeue`` read (at start, then
    refreshed by :meth:`refresh_live_jobs` at each recovery check) and its
    heartbeat is older than that read; when it names this job but another process (an
    earlier run of a requeued job); when it is a local run's claim on this
    host whose pid is gone; when it never started and was claimed over
    ``unstarted_seconds`` ago (a safety net: its owner releases it sooner);
    or when its heartbeat (mtime, touched every
    :data:`CLAIM_HEARTBEAT_SECONDS`) is older than
    :data:`STALE_CLAIM_SECONDS`. A ``released`` claim is always stale. A
    ``done`` claim is final for its own plan and stale for any other, so a
    resubmitted (regenerated) plan reclaims every task it planned and
    :func:`task_phases` decides what is left.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        plan_id: str,
        allocation_index: int,
        live_jobs: frozenset[str] | None,
        job_id: str | None = None,
        stale_seconds: float = STALE_CLAIM_SECONDS,
        unstarted_seconds: float | None = None,
        refresh_seconds: float = LIVE_JOBS_REFRESH_SECONDS,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan_id = plan_id
        self.allocation_index = allocation_index
        self.job_id = job_id or _job_identity()
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.stale_seconds = stale_seconds
        self.unstarted_seconds = (
            2 * PHASE_START_TIMEOUT_SECONDS
            if unstarted_seconds is None
            else unstarted_seconds
        )
        self.live_jobs = live_jobs
        self.refresh_seconds = refresh_seconds
        self._run = run
        self._clock = clock
        self._live_jobs_refreshed = clock()
        self.held: dict[str, Path] = {}
        #: Tasks (not the aggregation) this store has marked done.
        self.finished = 0
        #: The record each held claim was created with, kept up to date.
        self.records: dict[str, dict[str, Any]] = {}
        self._clock_offset = 0.0
        self._sync_clock()
        # Claims older than this were written before the squeue read.
        self.live_jobs_read_at = self.now()

    @classmethod
    def for_plan(
        cls,
        plan: SlurmPlan,
        allocation_index: int,
        *,
        live_jobs: frozenset[str] | None = None,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> "ClaimStore":
        if live_jobs is None:
            live_jobs = live_slurm_jobs(run)
        return cls(
            claims_dir(plan),
            plan_id=plan.plan_id,
            allocation_index=allocation_index,
            live_jobs=live_jobs,
            run=run,
        )

    def _sync_clock(self) -> None:
        """Align ``now`` with the file server's clock, which sets mtimes."""
        probe = self.root / f".clock-{self.host}-{self.pid}"
        try:
            probe.touch()
            os.utime(probe)
            self._clock_offset = probe.stat().st_mtime - time.time()
            probe.unlink(missing_ok=True)
        except OSError:
            self._clock_offset = 0.0

    def now(self) -> float:
        return time.time() + self._clock_offset

    def refresh_live_jobs(self, *, force: bool = False) -> bool:
        """Re-read ``squeue``, at most once per ``refresh_seconds``.

        A claim whose numeric job is missing from the fresh read, and whose
        heartbeat predates it, is then stale (:meth:`judge`): a job
        cancelled mid-run frees its claims at the next recovery check, not
        only once their heartbeat ages out. If squeue fails, the previous
        read is kept. Returns whether a read happened.
        """
        now = self._clock()
        if not force and now - self._live_jobs_refreshed < self.refresh_seconds:
            return False
        self._live_jobs_refreshed = now
        read_at = self.now()  # Before the read: later claims are not judged.
        jobs = live_slurm_jobs(self._run)
        if jobs is None:
            return False
        self.live_jobs = jobs
        self.live_jobs_read_at = read_at
        return True

    def _path(self, task_id: str, generation: int) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
        return self.root / f"{safe}.{generation}"

    def current_path(self, task_id: str) -> Path | None:
        """The task's highest existing claim generation, if any."""
        if not os.path.lexists(self._path(task_id, 0)):
            return None
        generation = 0
        while os.path.lexists(self._path(task_id, generation + 1)):
            generation += 1
        return self._path(task_id, generation)

    def _record(self, task_id: str, status: str, **extra: Any) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "plan_id": self.plan_id,
            "job_id": self.job_id,
            "host": self.host,
            "pid": self.pid,
            "allocation_index": self.allocation_index,
            "status": status,
            "claimed_at": _utc_now(),
            **extra,
        }

    def _create(self, path: Path, task_id: str) -> bool:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        record = self._record(
            task_id, "running", started=False, claimed_epoch=round(self.now(), 3)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        self.records[task_id] = record
        return True

    def try_claim(self, task_id: str) -> str:
        """Claim ``task_id``: :data:`CLAIMED`, :data:`HELD`, or :data:`DONE`."""
        if task_id in self.held:
            return HELD
        generation = 0
        while True:
            path = self._path(task_id, generation)
            if self._create(path, task_id):
                self.held[task_id] = path
                return CLAIMED
            if os.path.lexists(self._path(task_id, generation + 1)):
                generation += 1
                continue
            state = self.judge(path)
            if state != STALE:
                return state
            generation += 1

    def claim_again(self, task_id: str) -> str:
        """Claim a task this plan already finished, as its next generation.

        For the aggregation: an allocation that finished tasks after the
        sweep was last aggregated must aggregate again. Returns
        :data:`CLAIMED`, or :data:`HELD` if another allocation got there first.
        """
        state = self.try_claim(task_id)
        if state != DONE:
            return state
        path = self.current_path(task_id)
        generation = int(path.name.rsplit(".", 1)[1]) + 1 if path else 0
        target = self._path(task_id, generation)
        if self._create(target, task_id):
            self.held[task_id] = target
            return CLAIMED
        return HELD

    def available(self, task_id: str) -> bool:
        """Whether a claim attempt could take ``task_id`` now (no claiming)."""
        path = self.current_path(task_id)
        return path is None or self.judge(path) == STALE

    def awaiting(self, task_id: str) -> str | None:
        """Why an idle allocation should not give up on ``task_id`` yet.

        ``"unstarted"``: another allocation holds it but has not started it,
        so may yet release it. ``"dead-job"``: the claim's job is gone from
        the latest ``squeue`` read, but its last heartbeat came after that
        read, so it is not stale until the next read. ``None`` when it is
        held by a live job that started it (or is not held at all).
        """
        path = self.current_path(task_id)
        if path is None or self.judge(path) != HELD:
            return None
        claim = _read_claim(path)
        if claim.get("status") != "running":
            return None
        job = str(claim.get("job_id") or "")
        if (
            job.isdigit()
            and job != self.job_id
            and self.live_jobs is not None
            and job not in self.live_jobs
        ):
            return "dead-job"
        if claim.get("started") is False:
            return "unstarted"
        return None

    def judge(self, path: Path) -> str:
        """Whether an existing claim is :data:`HELD`, :data:`DONE`, or stale."""
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return STALE
        claim = _read_claim(path)
        status = claim.get("status")
        if status == "done":
            return DONE if claim.get("plan_id") == self.plan_id else STALE
        if status == RELEASED:
            return STALE
        job = str(claim.get("job_id") or "")
        pid = claim.get("pid")
        if job and job == self.job_id:
            same = claim.get("host") == self.host and pid == self.pid
            return HELD if same else STALE
        if (
            job.isdigit()
            and self.live_jobs is not None
            and job not in self.live_jobs
            and mtime < self.live_jobs_read_at
        ):
            return STALE
        if (
            job
            and not job.isdigit()
            and claim.get("host") == self.host
            and isinstance(pid, int)
            and not _pid_alive(pid)
        ):
            return STALE
        epoch = claim.get("claimed_epoch")
        if (
            claim.get("started") is False
            and isinstance(epoch, (int, float))
            and self.now() - epoch > self.unstarted_seconds
        ):
            return STALE
        if self.now() - mtime > self.stale_seconds:
            return STALE
        return HELD

    def heartbeat(self) -> None:
        """Touch every claim this allocation still runs."""
        for path in self.held.values():
            try:
                os.utime(path)
            except OSError as exc:
                print(f"Claim heartbeat failed for {path}: {exc}", file=sys.stderr)

    def note_started(self, task_id: str, started_at: str) -> None:
        """Keep a claim's ``started`` fields in the record this store rewrites."""
        record = self.records.get(task_id)
        if record is not None and not record.get("started"):
            record.update(started=True, started_at=started_at)

    def _finish(self, task_id: str, status: str, **extra: Any) -> None:
        path = self.held.pop(task_id, None)
        if path is None:
            return
        if status == "done" and task_id != AGGREGATE_CLAIM:
            self.finished += 1
        record = dict(self.records.pop(task_id, None) or self._record(task_id, status))
        record.update(status=status, finished_at=_utc_now(), **extra)
        write_json(str(path), record)

    def mark_done(self, task_id: str, outcome: str) -> None:
        """Record ``task_id`` finished for this plan; stop heartbeating it."""
        self._finish(task_id, "done", outcome=outcome)

    def release(self, task_id: str, reason: str) -> None:
        """Give ``task_id`` back: its claim is stale for every allocation."""
        self._finish(task_id, RELEASED, reason=reason)


def mark_claim_started(
    path: str | Path, record: Mapping[str, Any], unit: str
) -> None:
    """Rewrite a claim as started, from the process that runs ``unit``.

    Only the process that won its start token (:func:`_take_token`) writes,
    and its allocation rewrites the claim only after that process ends, so
    the two never write at once.
    """
    now = _utc_now()
    write_json(
        str(path),
        {
            **record,
            "started": True,
            "started_at": record.get("started_at") or now,
            "unit": unit,
            "unit_pid": os.getpid(),
            "unit_started_at": now,
        },
    )


def claims_dir(plan: SlurmPlan) -> Path:
    return Path(plan.output_root) / SLURM_DIRNAME / CLAIMS_DIRNAME


class TaskQueue:
    """One allocation's view of its worker count's shared queue.

    Claims follow the plan's order (:func:`claim_order`), a tier of equally
    long tasks at a time. Within a tier the allocation prefers a task whose
    policy market it has already loaded, then one whose dataset it has, then
    the next in order. A task another allocation holds is rechecked for a
    stale claim once the queue runs dry, at most every
    :data:`CLAIM_RECOVERY_SECONDS` while this allocation still has work.

    The allocation claims one task at a time, only when a slot is free to
    start it. :meth:`peek` names the tasks it would claim next without
    claiming them, which is what the cache warm-up uses.
    """

    def __init__(
        self,
        plan: SlurmPlan,
        indices: Sequence[int],
        claims: ClaimStore,
        *,
        recovery_seconds: float = CLAIM_RECOVERY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.plan = plan
        self.claims = claims
        self.recovery_seconds = recovery_seconds
        self.clock = clock
        indices = list(indices)
        self.tiers: list[list[int]] = []
        last = None
        for index in indices:
            seconds = expected_task_seconds(plan.tasks[index].config)
            if not self.tiers or seconds != last:
                self.tiers.append([])
                last = seconds
            self.tiers[-1].append(index)
        self.held_elsewhere: list[int] = []
        self.loaded_markets: set[str] = set()
        self.loaded_datasets: set[str] = set()
        self._last_recovery: float | None = None
        self.keys = {
            index: (
                market_key(plan.tasks[index].config),
                dataset_key(plan.tasks[index].config),
            )
            for index in indices
        }

    def _pick(
        self,
        tier: list[int],
        markets: set[str],
        datasets: set[str],
        exclude: Callable[[int], bool] | None = None,
    ) -> int | None:
        """The position in ``tier`` to claim next, or ``None`` if none may be."""
        allowed = [
            place for place, index in enumerate(tier) if not (exclude and exclude(index))
        ]
        for position, loaded in enumerate((markets, datasets)):
            if loaded:
                for place in allowed:
                    if self.keys[tier[place]][position] in loaded:
                        return place
        return allowed[0] if allowed else None

    def _claimed(self, index: int) -> int:
        market, dataset = self.keys[index]
        self.loaded_markets.add(market)
        self.loaded_datasets.add(dataset)
        return index

    def mark_loaded(self, markets: Iterable[str], datasets: Iterable[str]) -> None:
        """Record markets and datasets this allocation warmed."""
        self.loaded_markets.update(markets)
        self.loaded_datasets.update(datasets)

    def peek(self, count: int) -> list[int]:
        """The next ``count`` tasks this allocation would claim; claims none.

        It follows :meth:`claim_next`'s order and preferences, skipping
        tasks another allocation holds or finished. Others may claim them
        first, so this is a forecast, not a reservation.
        """
        markets, datasets = set(self.loaded_markets), set(self.loaded_datasets)
        picked: list[int] = []
        for tier in self.tiers:
            if len(picked) >= count:
                break
            rest = [
                index
                for index in tier
                if self.claims.available(self.plan.tasks[index].task_id)
            ]
            while rest and len(picked) < count:
                place = self._pick(rest, markets, datasets)
                index = rest.pop(place)
                picked.append(index)
                markets.add(self.keys[index][0])
                datasets.add(self.keys[index][1])
        return picked

    def requeue(self, index: int) -> None:
        """Retry a task this allocation released, with the others' claims.

        It is rechecked like a task another allocation holds: once the queue
        runs dry, so another allocation usually takes it first.
        """
        if index not in self.held_elsewhere:
            self.held_elsewhere.append(index)

    def awaiting_elsewhere(self) -> dict[str, int]:
        """Tasks held elsewhere that an idle allocation should wait for.

        Counts, by reason (:meth:`ClaimStore.awaiting`), the claims that
        another allocation may yet release or whose job is gone.
        """
        reasons: dict[str, int] = {}
        for index in self.held_elsewhere:
            reason = self.claims.awaiting(self.plan.tasks[index].task_id)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def claim_next(
        self,
        *,
        force_recovery: bool = False,
        exclude: Callable[[int], bool] | None = None,
    ) -> int | None:
        """Claim and return the next task index, or ``None`` if none is free.

        ``exclude`` marks tasks not to claim yet (those whose dataset is
        still warming). Tasks are only taken from the first tier with
        unclaimed tasks, so if all its remaining tasks are excluded, nothing
        is claimed: the slot waits rather than start a shorter task early.
        """
        for tier in self.tiers:
            while True:
                place = self._pick(
                    tier, self.loaded_markets, self.loaded_datasets, exclude
                )
                if place is None:
                    break
                index = tier.pop(place)
                state = self.claims.try_claim(self.plan.tasks[index].task_id)
                if state == CLAIMED:
                    return self._claimed(index)
                if state == HELD:
                    self.held_elsewhere.append(index)
            if tier:
                return None  # The rest of this tier waits for its warm-up.
        now = self.clock()
        if not self.held_elsewhere or not (
            force_recovery
            or self._last_recovery is None
            or now - self._last_recovery >= self.recovery_seconds
        ):
            return None
        # A job cancelled since the last read frees its claims now.
        self.claims.refresh_live_jobs()
        for index in list(self.held_elsewhere):
            state = self.claims.try_claim(self.plan.tasks[index].task_id)
            if state == HELD:
                continue
            self.held_elsewhere.remove(index)
            if state == CLAIMED:
                print(
                    f"Recovered stale claim of task {self.plan.tasks[index].task_id}",
                    flush=True,
                )
                # Unthrottled, the next call resumes the pass.
                return self._claimed(index)
        self._last_recovery = now
        return None


# ---------------------------------------------------------------------------
# Worker processes.

#: What a unit's process returns when its allocation abandoned it first.
ABANDONED = "abandoned"


def init_worker_process() -> None:
    """Start-up of every allocation process: each loader warning once.

    A process forked from the forkserver inherits the server's pyproj
    context, whose PROJ database handle (an SQLite connection) was opened
    when pyproj was imported there. SQLite connections must not cross a
    fork, so the process drops it and opens its own on first use.
    """
    warnings.filterwarnings("once", category=UserWarning, module="loaders")
    context = sys.modules.get("pyproj._context")
    local = getattr(context, "_CONTEXT_MANAGER_LOCAL", None)
    if local is not None:
        try:
            local.context_manager = None
        except AttributeError:
            pass


def _pool_context():
    """The start method of the allocation's process pool.

    ``forkserver`` (the default; :data:`START_METHOD_ENVIRONMENT` overrides
    it): one server process imports :data:`FORKSERVER_PRELOAD` once, and
    every phase process forks from it, instead of re-importing the
    environment from NFS as a spawned process does. Each phase still gets a
    fresh process (``max_tasks_per_child=1``).
    """
    import multiprocessing

    method = os.environ.get(START_METHOD_ENVIRONMENT, "").strip() or "forkserver"
    context = multiprocessing.get_context(method)
    if method == "forkserver":
        context.set_forkserver_preload(list(FORKSERVER_PRELOAD))
    return context


def _make_executor(slots: int):
    return ProcessPoolExecutor(
        max_workers=slots,
        max_tasks_per_child=1,
        initializer=init_worker_process,
        mp_context=_pool_context(),
    )


def _local_token_dir() -> Path:
    """A node-local directory for start tokens: RAM if it can, never NFS."""
    shm = Path("/dev/shm")
    base = str(shm) if shm.is_dir() and os.access(shm, os.W_OK) else None
    return Path(tempfile.mkdtemp(prefix="benchmark-slurm-starts-", dir=base))


def _take_token(path: str | Path, payload: Mapping[str, Any]) -> bool:
    """Create the start token ``path``; ``False`` if someone else did."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream)
    return True


def _read_token(path: str | Path) -> dict[str, Any] | None:
    """A start token's payload; ``None`` while nobody has taken it."""
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return {}  # Taken, still being written.
    return value if isinstance(value, dict) else {}


def _unit_entry(
    token: str,
    claim_path: str | None,
    claim_record: Mapping[str, Any] | None,
    unit: str,
    delay: float,
    function: Callable[..., Any],
    *args: Any,
) -> Any:
    """The first thing a unit's process runs: report the start, then work.

    The process and its allocation race to create the node-local start
    token. If the allocation made it first (it gave the claim back because
    this process took too long to start), the process returns
    :data:`ABANDONED` without touching the task. Otherwise it marks the
    claim started, waits out its stagger ``delay``, and runs the unit.
    """
    if not _take_token(
        token, {"pid": os.getpid(), "unit": unit, "started": time.time()}
    ):
        return ABANDONED
    if claim_path is not None and claim_record is not None:
        try:
            mark_claim_started(claim_path, claim_record, unit)
        except OSError as exc:
            print(f"Could not mark claim {claim_path} started: {exc}", file=sys.stderr)
    if delay > 0:
        time.sleep(delay)
    return function(*args)


def _run_fused_task(
    task: BenchmarkTask,
    strict_metrics: bool,
    compute_stage_metrics: bool,
    matching: MatchingRunConfig,
    visualization: VisualizationRunConfig,
) -> TaskResult:
    return run_optimize_and_evaluate(
        task,
        strict_metrics=strict_metrics,
        compute_stage_metrics=compute_stage_metrics,
        matching=matching,
        visualization=visualization,
    )


def _run_evaluation_task(
    task: BenchmarkTask,
    strict_metrics: bool,
    compute_stage_metrics: bool,
    matching: MatchingRunConfig,
    visualization: VisualizationRunConfig,
) -> TaskResult:
    """Evaluate a run optimized in an earlier process (a resume)."""
    with timed_phase(task, "evaluate") as timing:
        result = evaluate_optimization_task(
            task,
            strict_metrics=strict_metrics,
            compute_stage_metrics=compute_stage_metrics,
            matching=matching,
            visualization=visualization,
        )
        timing["status"] = result.status
    return result


def _run_matching_task(task: BenchmarkTask, matching: MatchingRunConfig) -> TaskResult:
    """Match one evaluated task into its own run folder."""
    with timed_phase(task, "match") as timing:
        skipped = run_task_matching(task, matching) == "skipped"
        timing["status"] = "SKIPPED" if skipped else "COMPLETE"
    return TaskResult(
        task_id=task.task_id,
        output_dir=task.output_dir,
        status="SKIPPED" if skipped else "COMPLETE",
        skipped=skipped,
    )


def _warm_spec(spec: WarmSpec) -> dict[str, Any]:
    return warm_dataset(spec)


def _unit_call(plan: SlurmPlan, index: int, unit: str):
    task = _task_at(plan, index)
    evaluation_args = (
        plan.metrics.strict,
        plan.metrics.compute_stage_metrics,
        plan.matching,
        plan.visualization,
    )
    if unit == FUSED_OPTIMIZE_EVALUATE:
        return _run_fused_task, (task, *evaluation_args)
    if unit == "evaluate":
        return _run_evaluation_task, (task, *evaluation_args)
    if unit == "match":
        return _run_matching_task, (task, plan.matching)
    raise ValueError(f"Unknown task unit {unit!r}.")


def _warm_enabled() -> bool:
    return os.environ.get(WARM_ENVIRONMENT, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@dataclass
class _Running:
    """One process the allocation submitted, and what it is for."""

    kind: str  # "warm", "task", or "abandoned" (released, still in its slot)
    token: Path
    submitted_at: float  # time.time(), comparable with the token's stamp
    unit: str
    index: int | None = None
    key: str | None = None
    started: bool = False


@dataclass
class _AllocationState:
    """The bookkeeping of one allocation's scheduling loop."""

    token_dir: Path
    start_timeout: float = 0.0
    remaining: dict[int, deque[str]] = field(default_factory=dict)
    continuing: deque[int] = field(default_factory=deque)
    warming: dict[str, WarmSpec] = field(default_factory=dict)
    running: dict[Any, _Running] = field(default_factory=dict)
    starts: int = 0
    submissions: int = 0
    failed: bool = False

    def next_token(self) -> Path:
        self.submissions += 1
        return self.token_dir / f"{self.submissions}.start"


def run_benchmark_worker(
    plan_path: str,
    allocation_index: int,
    *,
    claims: ClaimStore | None = None,
    make_executor: Callable[[int], Any] | None = None,
    start_timeout: float | None = None,
) -> int:
    """Optimize, evaluate, and match tasks claimed from the plan's queue.

    Every allocation of a worker count claims tasks one at a time from the
    same queue (:class:`TaskQueue`, :class:`ClaimStore`), and only when a
    slot is free to start one, so allocations finish together and none
    holds work it is not running. Each concurrent slot (``task_cpus`` CPUs)
    carries one task through its remaining phases (:func:`task_phases`):
    optimize and evaluate in one process, then match in a fresh one
    (``max_tasks_per_child=1``, forked from a forkserver that preloaded the
    environment), a task's next phase taking its freed slot ahead of any
    new task.

    Before the fan-out, the allocation peeks at the tasks it would claim
    first and warms each distinct dataset among them once, one process per
    dataset (:mod:`benchmark.warmup`); a task of a warming dataset is not
    claimed until it is warm. The first ``slots`` task starts are staggered
    by :data:`WORKER_START_STAGGER_SECONDS`. A process that has not
    reported its start within ``start_timeout`` seconds
    (:data:`PHASE_START_TIMEOUT_SECONDS`) has its claim released for other
    allocations and is abandoned. The allocation exits when the queue is
    empty, no other allocation has an unstarted claim it may release, and
    its last task is done; aggregation, the sweep's matching reports
    included, then runs.
    """

    plan: SlurmPlan | None = None
    failed = False
    work_started = False
    init_worker_process()
    for setting in THREAD_ENVIRONMENT:
        name, value = setting.split("=", 1)
        os.environ.setdefault(name, value)  # The forkserver inherits these.
    # Every process of the allocation shows each loader warning once between
    # them (loaders.table_cache.WARN_ONCE_DIR_ENV), not once each.
    warn_dir = _warn_once_dir()
    try:
        plan = load_plan(plan_path)
        allocation = _allocation_at(plan, allocation_index)
        _validate_worker_cpus(allocation.cpus)
        work_started = True
        slots = max(1, allocation.cpus // allocation.task_cpus)
        if claims is None:
            claims = ClaimStore.for_plan(plan, allocation_index)
        queue = TaskQueue(plan, allocation.task_indices, claims)
        failed = _run_allocation(
            plan,
            queue,
            claims,
            slots,
            make_executor=make_executor or _make_executor,
            start_timeout=(
                PHASE_START_TIMEOUT_SECONDS if start_timeout is None else start_timeout
            ),
        )
    except Exception as exc:
        print(f"Benchmark allocation failed: {exc}", file=sys.stderr, flush=True)
        failed = True
    finally:
        if plan is not None and work_started and claims is not None:
            try:
                _aggregate_if_last(plan, claims)
            except Exception as exc:
                print(f"Aggregation failed: {exc}", file=sys.stderr, flush=True)
                failed = True
        if warn_dir is not None:
            from loaders.table_cache import WARN_ONCE_DIR_ENV

            os.environ.pop(WARN_ONCE_DIR_ENV, None)
            shutil.rmtree(warn_dir, ignore_errors=True)
    return 1 if failed or plan is None else 0


def _warn_once_dir() -> Path | None:
    """Point this allocation's processes at one node-local warn-once dir."""
    from loaders.table_cache import WARN_ONCE_DIR_ENV

    if os.environ.get(WARN_ONCE_DIR_ENV):
        return None
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
    job = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    directory = base / f"sfusd-warn-{getpass.getuser()}-{job}"
    os.environ[WARN_ONCE_DIR_ENV] = str(directory)
    return directory


#: Claim name that makes one allocation the sweep's aggregator.
AGGREGATE_CLAIM = "__aggregate__"


def _aggregate_if_last(plan: SlurmPlan, claims: "ClaimStore") -> bool:
    """Aggregate the sweep if this is its last allocation; return whether it did.

    Every allocation used to rebuild the sweep's CSVs on exit, one after
    another under the aggregation lock, each re-reading every run over NFS.
    Now an allocation aggregates only once no other allocation still holds a
    task, and only the one that wins the :data:`AGGREGATE_CLAIM` claim does;
    an allocation that leaves work running elsewhere leaves the aggregation to
    whichever allocation finishes last. The claim is marked done afterwards, so
    a resubmitted allocation of the same plan with nothing to do does not
    aggregate again; a crashed aggregator's claim goes stale like any other,
    and ``python -m benchmark.slurm aggregate --plan PLAN`` reruns it by hand.
    """
    claims.refresh_live_jobs(force=True)
    held = [
        task.task_id
        for task in plan.tasks
        if (path := claims.current_path(task.task_id)) is not None
        and claims.judge(path) == HELD
    ]
    if held:
        print(
            f"aggregate skipped: {len(held)} task(s) still held by other "
            "allocations; the last allocation aggregates",
            flush=True,
        )
        return False
    state = (
        claims.claim_again(AGGREGATE_CLAIM)
        if claims.finished
        else claims.try_claim(AGGREGATE_CLAIM)
    )
    if state != CLAIMED:
        print(f"aggregate skipped: another allocation's aggregation is {state}", flush=True)
        return False
    started = time.monotonic()
    try:
        _aggregate(plan)
    except BaseException:
        claims.release(AGGREGATE_CLAIM, "aggregation failed")
        raise
    claims.mark_done(AGGREGATE_CLAIM, "complete")
    print(f"aggregate done in {time.monotonic() - started:.1f}s", flush=True)
    return True


def _claim_with_work(
    plan: SlurmPlan,
    queue: TaskQueue,
    claims: ClaimStore,
    state: _AllocationState,
    *,
    force_recovery: bool = False,
    exclude: Callable[[int], bool] | None = None,
) -> int | None:
    """Claim the next task that still has a phase to run."""
    while True:
        index = queue.claim_next(force_recovery=force_recovery, exclude=exclude)
        if index is None:
            return None
        task = _task_at(plan, index)
        try:
            units = execution_units(task_phases(task, plan.execution, plan.matching))
        except Exception as exc:
            print(f"Task {index} could not be planned: {exc}", file=sys.stderr)
            claims.mark_done(task.task_id, "crashed")
            state.failed = True
            continue
        if units:
            state.remaining[index] = deque(units)
            return index
        claims.mark_done(task.task_id, "complete")


def _run_allocation(
    plan: SlurmPlan,
    queue: TaskQueue,
    claims: ClaimStore,
    slots: int,
    *,
    make_executor: Callable[[int], Any] = _make_executor,
    start_timeout: float = PHASE_START_TIMEOUT_SECONDS,
) -> bool:
    state = _AllocationState(token_dir=_local_token_dir(), start_timeout=start_timeout)
    idle_since: float | None = None
    last_awaiting: dict[str, int] | None = None
    try:
        with make_executor(slots) as executor:
            if _warm_enabled():
                peeked = [plan.tasks[index] for index in queue.peek(slots)]
                for spec in warm_specs_for_tasks(peeked):
                    state.warming[spec.key] = spec
                    token = state.next_token()
                    submitted_at = time.time()
                    future = executor.submit(
                        _unit_entry, str(token), None, None, "warm", 0.0,
                        _warm_spec, spec,
                    )
                    state.running[future] = _Running(
                        "warm", token, submitted_at, "warm", key=spec.key
                    )
            last_heartbeat = time.monotonic()

            while True:
                _fill_slots(executor, plan, queue, claims, state, slots)
                if not state.running:
                    index = _claim_with_work(
                        plan, queue, claims, state, force_recovery=True
                    )
                    if index is not None:
                        state.continuing.append(index)  # Recovered: start at once.
                        idle_since = last_awaiting = None
                        continue
                    awaiting = queue.awaiting_elsewhere()
                    if awaiting:
                        now = time.monotonic()
                        if idle_since is None:
                            idle_since = now
                        if now - idle_since < EXIT_WAIT_LIMIT_SECONDS:
                            if awaiting != last_awaiting:
                                print(
                                    "Queue empty; waiting on other allocations' "
                                    f"claims that may be freed: {awaiting}",
                                    flush=True,
                                )
                                last_awaiting = awaiting
                            time.sleep(IDLE_POLL_SECONDS)
                            continue
                        print(
                            f"Gave up waiting after {EXIT_WAIT_LIMIT_SECONDS:.0f}s "
                            f"on claims {awaiting}; a resubmit runs what is left.",
                            file=sys.stderr,
                            flush=True,
                        )
                    break
                done, _ = wait(
                    state.running, timeout=POLL_SECONDS, return_when=FIRST_COMPLETED
                )
                _check_starts(plan, queue, claims, state)
                if time.monotonic() - last_heartbeat >= CLAIM_HEARTBEAT_SECONDS / 2:
                    claims.heartbeat()
                    last_heartbeat = time.monotonic()
                for future in done:
                    _finish(plan, queue, claims, state, future)
    finally:
        shutil.rmtree(state.token_dir, ignore_errors=True)
    return state.failed


def _fill_slots(executor, plan, queue, claims, state, slots) -> None:
    """Start work in every free slot, claiming a task only to start it."""
    while len(state.running) < slots:
        delay = 0.0
        if state.continuing:
            index = state.continuing.popleft()
        else:
            warming = state.warming
            exclude = (
                (lambda i: dataset_key(plan.tasks[i].config) in warming)
                if warming
                else None
            )
            index = _claim_with_work(plan, queue, claims, state, exclude=exclude)
            if index is None:
                return
            # Only the first wave starts together; later tasks start as
            # slots free, already spread out.
            if state.starts < slots:
                delay = state.starts * WORKER_START_STAGGER_SECONDS
            state.starts += 1
        unit = state.remaining[index].popleft()
        task_id = _task_at(plan, index).task_id
        path = claims.held.get(task_id)
        function, args = _unit_call(plan, index, unit)
        token = state.next_token()
        submitted_at = time.time()
        future = executor.submit(
            _unit_entry,
            str(token),
            None if path is None else str(path),
            claims.records.get(task_id),
            unit,
            delay,
            function,
            *args,
        )
        state.running[future] = _Running("task", token, submitted_at, unit, index=index)


def _check_starts(plan, queue, claims, state) -> None:
    """Log each process's start-up; release claims whose process never started.

    A task unit whose process has not taken its start token within
    ``state.start_timeout`` seconds of being submitted is abandoned: the
    allocation takes the token itself, so the process, if it ever starts,
    returns without running; the claim is released for other allocations;
    and the task is requeued here behind every other. The abandoned
    process keeps its slot until it exits, so a stalled node claims
    nothing new.
    """
    now = time.time()
    for entry in state.running.values():
        if entry.started or entry.kind == "abandoned":
            continue
        token = _read_token(entry.token)
        if token is not None:
            _note_start(plan, claims, entry, token)
            continue
        if entry.kind != "task" or now - entry.submitted_at <= state.start_timeout:
            continue
        if not _take_token(entry.token, {"abandoned": True, "at": now}):
            continue  # The process started just now; seen on the next pass.
        task_id = _task_at(plan, entry.index).task_id
        reason = (
            f"its {entry.unit} process did not start within "
            f"{state.start_timeout:.0f}s"
        )
        claims.release(task_id, reason)
        state.remaining.pop(entry.index, None)
        queue.requeue(entry.index)
        entry.kind = "abandoned"
        print(f"Released task {task_id}: {reason}", file=sys.stderr, flush=True)


def _note_start(plan, claims, entry: _Running, token: Mapping[str, Any]) -> None:
    entry.started = True
    stamp = token.get("started")
    startup = (
        f"{stamp - entry.submitted_at:.2f}s"
        if isinstance(stamp, (int, float))
        else "?"
    )
    if entry.kind == "task":
        task_id = _task_at(plan, entry.index).task_id
        claims.note_started(task_id, _utc_now())
    else:
        task_id = (entry.key or "?")[:12]
    print(
        f"start task={task_id} unit={entry.unit} startup={startup} "
        f"pid={token.get('pid')}",
        flush=True,
    )


def _finish(plan, queue, claims, state, future) -> None:
    entry = state.running.pop(future)
    if not entry.started and entry.kind != "abandoned":
        token = _read_token(entry.token)
        if token is not None:
            _note_start(plan, claims, entry, token)
    if entry.kind == "abandoned":
        try:
            outcome = future.result()
        except Exception as exc:
            outcome = f"{exc.__class__.__name__}: {exc}"
        print(
            f"Abandoned {entry.unit} process of task {entry.index} exited ({outcome}).",
            flush=True,
        )
        return
    if entry.kind == "warm":
        spec = state.warming.pop(entry.key, None)
        try:
            print(format_warm_summary(future.result()), flush=True)
        except Exception as exc:
            print(f"Cache warm-up failed: {exc}", file=sys.stderr, flush=True)
        else:
            if spec is not None:
                queue.mark_loaded(
                    (
                        market_key({**spec.config, "matching_policy": policy})
                        for policy in spec.policies
                    ),
                    (spec.key,),
                )
        return
    index, unit = entry.index, entry.unit
    task_id = _task_at(plan, index).task_id
    try:
        _report_task_error(unit, index, future.result())
    except Exception as exc:
        print(f"Task {index} {unit} failed: {exc}", file=sys.stderr, flush=True)
        state.failed = True
        state.remaining.pop(index, None)
        claims.mark_done(task_id, "crashed")
        return
    if state.remaining.get(index):
        state.continuing.append(index)
    else:
        state.remaining.pop(index, None)
        claims.mark_done(task_id, "complete")


def _aggregate(plan: SlurmPlan) -> None:
    """Rebuild the sweep's CSVs, its matching reports too, from every run.

    Every run under the output root counts, including runs a ``skip_existing``
    plan left out; the walk stops at each run folder, so it lists one folder
    per run rather than every lottery file.
    """
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
        # Every allocation of a worker count claims from its one queue; an
        # even share of it only sizes each allocation's CPUs.
        for share in _split_indices(indices, allocation_counts[task_cpus]):
            allocations.append(
                SlurmAllocation(
                    phase="benchmark",
                    task_indices=tuple(indices),
                    cpus=min(concurrency, len(share)) * task_cpus,
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
        # Tags the job with its plan, so ``topup`` can count a plan's live
        # allocations among the user's other jobs.
        f"--comment={_PLAN_COMMENT_PREFIX}{plan.plan_id}",
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

    topup_parser = subparsers.add_parser(
        "topup",
        help="Submit a saved plan's allocations that are not pending or running.",
    )
    topup_parser.add_argument("--plan", required=True, help="Saved plan JSON path.")
    topup_parser.add_argument(
        "--sbatch", default="sbatch", help="sbatch executable (default: sbatch)."
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Rebuild a saved plan's summary and matching CSVs now."
    )
    aggregate_parser.add_argument("--plan", required=True, help="Saved plan JSON path.")
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
            print(
                f"Allocations: {len(plan.allocations)} "
                f"({', '.join(str(a.cpus) for a in plan.allocations)} CPUs, "
                f"--mem-per-cpu={SBATCH_MEM_PER_CPU}), claiming from "
                f"{claims_dir(plan)}"
            )
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
        if args.command == "topup":
            plan = load_plan(args.plan)
            live, submitted = topup_plan(plan, args.plan, sbatch=args.sbatch)
            print(
                f"Live allocations: {len(live)} "
                f"({', '.join(f'{i}={j}' for i, j in sorted(live.items())) or 'none'})"
            )
            for job in submitted:
                print(f"allocation {job.allocation_index}: {job.phase}={job.job_id}")
            if not submitted:
                print("No allocations submitted.")
            return 0
        if args.command == "aggregate":
            plan = load_plan(args.plan)
            _aggregate(plan)
            print(f"Aggregated {plan.output_root}")
            return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"benchmark.slurm: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

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
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
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
    run_optimization_phase,
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
PLAN_SCHEMA_VERSION = 6
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
#: Delay between the first-wave task processes' starts, so 40 processes do
#: not open the same files on NFS in the same instant.
WORKER_START_STAGGER_SECONDS = 1.0
#: Set to ``0`` to skip the allocation's cache warm-up.
WARM_ENVIRONMENT = "BENCHMARK_SLURM_WARM"
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


def _job_identity() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return str(job_id)
    return f"local-{socket.gethostname()}-{os.getpid()}"


def live_slurm_jobs(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> frozenset[str] | None:
    """The user's queued and running job ids, or ``None`` if squeue fails."""
    try:
        result = run(
            [
                "squeue",
                "--noheader",
                "--user",
                getpass.getuser(),
                "--format=%A %i",
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
        for value in line.split():
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


class ClaimStore:
    """Atomic task claims in ``<output_root>/.slurm/claims`` on shared NFS.

    A claim is the file ``<task_id>.<generation>``, made with
    ``O_CREAT | O_EXCL`` (atomic on NFSv3+, so no locks), holding JSON: the
    plan id, Slurm job id, host, pid, allocation index, ``claimed_at``, and
    ``status`` (``running``, then ``done``). The highest generation is the
    task's claim. A stale one is taken over by creating the next generation,
    which again only one claimant can do; files are never deleted, so a
    generation, once seen, stays taken.

    A ``running`` claim is stale when its job (a numeric Slurm id) is missing
    from ``squeue`` as read when this allocation started and the claim is
    older than that read; when it names this job but another process (an
    earlier run of a requeued job); when it is a local run's claim on this
    host whose pid is gone; or when its heartbeat (mtime, touched every
    :data:`CLAIM_HEARTBEAT_SECONDS`) is older than
    :data:`STALE_CLAIM_SECONDS`. A ``done`` claim is final for its own plan
    and stale for any other, so a resubmitted (regenerated) plan reclaims
    every task it planned and :func:`task_phases` decides what is left.
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
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan_id = plan_id
        self.allocation_index = allocation_index
        self.job_id = job_id or _job_identity()
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.stale_seconds = stale_seconds
        self.live_jobs = live_jobs
        self.held: dict[str, Path] = {}
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

    def _path(self, task_id: str, generation: int) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
        return self.root / f"{safe}.{generation}"

    def _record(self, task_id: str, status: str, **extra: Any) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "plan_id": self.plan_id,
            "job_id": self.job_id,
            "host": self.host,
            "pid": self.pid,
            "allocation_index": self.allocation_index,
            "status": status,
            "claimed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        }

    def _create(self, path: Path, task_id: str) -> bool:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(self._record(task_id, "running"), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
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

    def judge(self, path: Path) -> str:
        """Whether an existing claim is :data:`HELD`, :data:`DONE`, or stale."""
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return STALE
        try:
            with path.open(encoding="utf-8") as stream:
                claim = json.load(stream)
            if not isinstance(claim, dict):
                claim = {}
        except (OSError, json.JSONDecodeError):
            claim = {}  # Being written: judged by its age alone.
        if claim.get("status") == "done":
            return DONE if claim.get("plan_id") == self.plan_id else STALE
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

    def mark_done(self, task_id: str, outcome: str) -> None:
        """Record ``task_id`` finished for this plan; stop heartbeating it."""
        path = self.held.pop(task_id, None)
        if path is None:
            return
        write_json(str(path), self._record(task_id, "done", outcome=outcome))


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

    def _pick(self, tier: list[int]) -> int:
        for position, loaded in enumerate((self.loaded_markets, self.loaded_datasets)):
            if loaded:
                for place, index in enumerate(tier):
                    if self.keys[index][position] in loaded:
                        return tier.pop(place)
        return tier.pop(0)

    def _claimed(self, index: int) -> int:
        market, dataset = self.keys[index]
        self.loaded_markets.add(market)
        self.loaded_datasets.add(dataset)
        return index

    def claim_next(self, *, force_recovery: bool = False) -> int | None:
        """Claim and return the next task index, or ``None`` if none is free."""
        for tier in self.tiers:
            while tier:
                index = self._pick(tier)
                state = self.claims.try_claim(self.plan.tasks[index].task_id)
                if state == CLAIMED:
                    return self._claimed(index)
                if state == HELD:
                    self.held_elsewhere.append(index)
        now = self.clock()
        if not self.held_elsewhere or not (
            force_recovery
            or self._last_recovery is None
            or now - self._last_recovery >= self.recovery_seconds
        ):
            return None
        for position, index in enumerate(list(self.held_elsewhere)):
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


def init_worker_process() -> None:
    """Start-up of every allocation process: each loader warning once."""
    warnings.filterwarnings("once", category=UserWarning, module="loaders")


def _staggered(delay: float, function: Callable[..., Any], *args: Any) -> Any:
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


def _submit_unit(executor, plan: SlurmPlan, index: int, unit: str, delay: float):
    task = _task_at(plan, index)
    evaluation_args = (
        plan.metrics.strict,
        plan.metrics.compute_stage_metrics,
        plan.matching,
        plan.visualization,
    )
    if unit == FUSED_OPTIMIZE_EVALUATE:
        return executor.submit(_staggered, delay, _run_fused_task, task, *evaluation_args)
    if unit == "evaluate":
        return executor.submit(
            _staggered, delay, _run_evaluation_task, task, *evaluation_args
        )
    if unit == "match":
        return executor.submit(_staggered, delay, _run_matching_task, task, plan.matching)
    raise ValueError(f"Unknown task unit {unit!r}.")


def _warm_enabled() -> bool:
    return os.environ.get(WARM_ENVIRONMENT, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@dataclass
class _AllocationState:
    """The bookkeeping of one allocation's scheduling loop."""

    remaining: dict[int, deque[str]] = field(default_factory=dict)
    continuing: deque[int] = field(default_factory=deque)
    ready: deque[int] = field(default_factory=deque)
    waiting: dict[str, list[int]] = field(default_factory=dict)
    running: dict[Any, tuple[str, Any]] = field(default_factory=dict)
    first_wave_started: int = 0
    failed: bool = False


def run_benchmark_worker(
    plan_path: str,
    allocation_index: int,
    *,
    claims: ClaimStore | None = None,
) -> int:
    """Optimize, evaluate, and match tasks claimed from the plan's queue.

    Every allocation of a worker count claims tasks one at a time from the
    same queue (:class:`TaskQueue`, :class:`ClaimStore`), so allocations
    finish together instead of each draining a fixed chunk. Each concurrent
    slot (``task_cpus`` CPUs) carries one task through its remaining phases
    (:func:`task_phases`): optimize and evaluate in one process, then match
    in a fresh one (``max_tasks_per_child=1``), a task's next phase taking
    its freed slot ahead of any new task.

    Before the fan-out, the allocation claims its first wave of tasks and
    warms each distinct dataset among them once, one process per dataset
    (:mod:`benchmark.warmup`); a first-wave task starts once its dataset is
    warm, the starts staggered by :data:`WORKER_START_STAGGER_SECONDS`. The
    allocation exits when the queue is empty and its last task is done;
    aggregation, the sweep's matching reports included, then runs.
    """

    plan: SlurmPlan | None = None
    failed = False
    work_started = False
    init_worker_process()
    try:
        plan = load_plan(plan_path)
        allocation = _allocation_at(plan, allocation_index)
        _validate_worker_cpus(allocation.cpus)
        work_started = True
        slots = max(1, allocation.cpus // allocation.task_cpus)
        if claims is None:
            claims = ClaimStore.for_plan(plan, allocation_index)
        queue = TaskQueue(plan, allocation.task_indices, claims)
        failed = _run_allocation(plan, queue, claims, slots)
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


def _claim_with_work(
    plan: SlurmPlan,
    queue: TaskQueue,
    claims: ClaimStore,
    state: _AllocationState,
    *,
    force_recovery: bool = False,
) -> int | None:
    """Claim the next task that still has a phase to run."""
    while True:
        index = queue.claim_next(force_recovery=force_recovery)
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
    plan: SlurmPlan, queue: TaskQueue, claims: ClaimStore, slots: int
) -> bool:
    state = _AllocationState()
    with ProcessPoolExecutor(
        max_workers=slots,
        max_tasks_per_child=1,
        initializer=init_worker_process,
    ) as executor:
        first_wave = []
        while len(first_wave) < slots:
            index = _claim_with_work(plan, queue, claims, state)
            if index is None:
                break
            first_wave.append(index)
        if _warm_enabled():
            for spec in warm_specs_for_tasks([plan.tasks[i] for i in first_wave]):
                state.waiting[spec.key] = [
                    index
                    for index in first_wave
                    if dataset_key(plan.tasks[index].config) == spec.key
                ]
                future = executor.submit(_warm_spec, spec)
                state.running[future] = ("warm", spec.key)
        waiting = {index for indices in state.waiting.values() for index in indices}
        state.ready.extend(index for index in first_wave if index not in waiting)
        last_heartbeat = time.monotonic()

        while True:
            _fill_slots(executor, plan, queue, claims, state, slots)
            if not state.running:
                index = _claim_with_work(
                    plan, queue, claims, state, force_recovery=True
                )
                if index is None:
                    break
                state.continuing.append(index)  # Recovered: start at once.
                continue
            done, _ = wait(
                state.running,
                timeout=CLAIM_HEARTBEAT_SECONDS,
                return_when=FIRST_COMPLETED,
            )
            if time.monotonic() - last_heartbeat >= CLAIM_HEARTBEAT_SECONDS / 2:
                claims.heartbeat()
                last_heartbeat = time.monotonic()
            for future in done:
                _finish(plan, claims, state, future)
    return state.failed


def _fill_slots(executor, plan, queue, claims, state, slots) -> None:
    while len(state.running) < slots:
        delay = 0.0
        if state.continuing:
            index = state.continuing.popleft()
        elif state.ready:
            index = state.ready.popleft()
            # Only the first wave starts together; later tasks start as
            # slots free, already spread out.
            delay = state.first_wave_started * WORKER_START_STAGGER_SECONDS
            state.first_wave_started += 1
        elif state.waiting:
            return  # The first wave waits for its warm-up before claiming more.
        else:
            index = _claim_with_work(plan, queue, claims, state)
            if index is None:
                return
        unit = state.remaining[index].popleft()
        future = _submit_unit(executor, plan, index, unit, delay)
        state.running[future] = ("task", (index, unit))


def _finish(plan, claims, state, future) -> None:
    kind, payload = state.running.pop(future)
    if kind == "warm":
        try:
            print(format_warm_summary(future.result()), flush=True)
        except Exception as exc:
            print(f"Cache warm-up failed: {exc}", file=sys.stderr, flush=True)
        state.ready.extend(state.waiting.pop(payload, []))
        return
    index, unit = payload
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
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"benchmark.slurm: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

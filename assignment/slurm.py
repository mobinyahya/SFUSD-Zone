"""Plan, submit, and execute batched Slurm assignment jobs."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone

import click
from loaders import load_scenario

from .run_custom_config import _write_provenance_config, load_custom_config
from .student_assignment.configerator import Configerator
from .student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN_SCHEMA_VERSION = 4
MAX_CPUS_PER_NODE = 40
MAX_ASSIGNMENT_JOBS = 12
MAX_METRICS_JOBS = 8
MAX_METRICS_FINALIZER_JOBS = 1
MAX_SLURM_JOBS = MAX_ASSIGNMENT_JOBS + MAX_METRICS_JOBS
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
_WORKER_PLAN = None
_WORKER_MARKET_GENERATOR = None
_WORKER_MARKET_KEY = None


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _write_json_atomic(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = pathlib.Path(temporary_file.name)
            json.dump(_jsonable(value), temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_text_atomic(path: pathlib.Path, text: str, *, executable=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = pathlib.Path(temporary_file.name)
            temporary_file.write(text)
        if executable:
            temporary_path.chmod(0o755)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _data_provenance(config: dict) -> dict:
    scenario = load_scenario(config["data"])
    return {
        "scenario": scenario.id,
        "schema_version": scenario.schema_version,
        "roots": scenario.roots,
        "filters": scenario.filters,
        "sources": scenario.source_manifest()["sources"],
        "semantic_fingerprint": scenario.semantic_fingerprint,
        "source_fingerprint": scenario.source_fingerprint,
    }


def _task_batches(task_count: int, batch_count: int) -> list[list[int]]:
    batch_count = min(task_count, batch_count)
    quotient, remainder = divmod(task_count, batch_count)
    batches = []
    start = 0
    for index in range(batch_count):
        size = quotient + (index < remainder)
        batches.append(list(range(start, start + size)))
        start += size
    return batches


def _build_allocations(
    assignment_count: int,
    metrics_count: int,
    *,
    metrics_work_counts: list[int] | None = None,
) -> list[dict]:
    allocations = []
    if assignment_count:
        assignment_job_count = min(MAX_ASSIGNMENT_JOBS, assignment_count)
        for task_indices in _task_batches(assignment_count, assignment_job_count):
            allocations.append(
                {
                    "phase": "assignment",
                    "task_indices": task_indices,
                    "cpus": min(MAX_CPUS_PER_NODE, len(task_indices)),
                }
            )
    if metrics_count:
        work_counts = metrics_work_counts or [1] * metrics_count
        if len(work_counts) != metrics_count or any(count < 1 for count in work_counts):
            raise ValueError("Metrics work counts must cover every metrics task.")
        metrics_job_count = min(MAX_METRICS_JOBS, metrics_count)
        for task_indices in _task_batches(metrics_count, metrics_job_count):
            allocations.append(
                {
                    "phase": "metrics",
                    "task_indices": task_indices,
                    "cpus": min(
                        MAX_CPUS_PER_NODE,
                        sum(work_counts[index] for index in task_indices),
                    ),
                }
            )
        allocations[-1]["phase"] = "metrics-finalize"
    return allocations


def _metrics_iterations(config: dict) -> list[int | None]:
    policies = config.get("policies", [])
    iterations: list[int | None] = []
    if any(policy != "real_match" for policy in policies):
        iterations.extend(
            range(config["iterations"]["start"], config["iterations"]["end"])
        )
    if "real_match" in policies:
        iterations.append(None)
    if not iterations:
        raise ValueError("Metrics subconfigs require at least one policy iteration.")
    return iterations


def build_slurm_plan(
    config_path: str | pathlib.Path,
    *,
    assignment_folder: str | pathlib.Path | None = None,
    plan_dir: str | pathlib.Path | None = None,
    sample: str | None = None,
    frac: str | None = None,
) -> tuple[dict, pathlib.Path]:
    """Resolve one config and write its absolute Slurm plan and provenance."""
    source_path = pathlib.Path(config_path).expanduser().resolve()
    config = load_custom_config(
        source_path,
        sample=sample,
        frac=frac,
        assignment_folder=assignment_folder,
        absolute_assignment_folder=True,
    )
    configurator = Configerator.from_config(config)
    subconfig_names = list(configurator.config.get("subconfigs", []))
    if not subconfig_names:
        raise ValueError("Slurm assignment runs require at least one subconfig.")
    duplicate_subconfigs = sorted(
        {name for name in subconfig_names if subconfig_names.count(name) > 1}
    )
    if duplicate_subconfigs:
        raise ValueError(
            f"Slurm assignment config contains duplicate subconfigs: "
            f"{duplicate_subconfigs}."
        )

    output_path = pathlib.Path(
        configurator.config["paths"]["assignment-folder"]
    ).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    plan_directory = (
        pathlib.Path(plan_dir).expanduser().resolve()
        if plan_dir is not None
        else output_path / "slurm"
    )
    run_id = uuid.uuid4().hex
    run_directory = plan_directory / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    plan_path = (run_directory / "assignment_plan.json").resolve()
    provenance_path = (run_directory / "provenance.json").resolve()
    metrics_fragment_dir = (run_directory / "metric_fragments").resolve()

    resolved_subconfigs = []
    for name in subconfig_names:
        configurator.load_subconfig_by_name(name)
        subconfig = copy.deepcopy(configurator.config)
        subconfig["subconfigs"] = []
        subconfig["paths"]["assignment-folder"] = str(output_path)
        resolved_subconfigs.append({"name": name, "config": subconfig})

    start = config["iterations"]["start"]
    end = config["iterations"]["end"]
    utility_owner = {"subconfig": subconfig_names[0], "iteration": start}
    assignment_tasks = [
        {
            "subconfig": name,
            "iteration": iteration,
            "write_utility_output": name == utility_owner["subconfig"]
            and iteration == utility_owner["iteration"],
        }
        for name in subconfig_names
        for iteration in range(start, end)
    ]
    metrics_tasks = [
        {
            "subconfig": entry["name"],
            "report_names": (
                list(MarketGenerator.AGGREGATE_METRIC_FILES)
                if entry["config"].get("export-local-metrics", False)
                else ["citywide"]
            ),
        }
        for entry in resolved_subconfigs
        if entry["config"].get("export-aggregate-metrics", False)
    ]
    metrics_work_counts = [
        len(_metrics_iterations(entry["config"]))
        for entry in resolved_subconfigs
        if entry["config"].get("export-aggregate-metrics", False)
    ]
    allocations = _build_allocations(
        len(assignment_tasks),
        len(metrics_tasks),
        metrics_work_counts=metrics_work_counts,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    if metrics_tasks:
        metrics_fragment_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "workspace_root": str(WORKSPACE_ROOT),
        "source_config": str(source_path),
        "source_config_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "resolved_config": config,
        "data": _data_provenance(config),
    }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "workspace_root": str(WORKSPACE_ROOT),
        "source_config": str(source_path),
        "assignment_folder": str(output_path),
        "plan_path": str(plan_path),
        "provenance_path": str(provenance_path),
        "metrics_fragment_dir": str(metrics_fragment_dir),
        "utility_output_owner": utility_owner,
        "subconfigs": resolved_subconfigs,
        "assignment_tasks": assignment_tasks,
        "metrics_tasks": metrics_tasks,
        "allocations": allocations,
    }

    # The launcher owns shared run metadata. Workers are explicitly configured
    # not to replace either this file or aggregate metric output.
    _write_provenance_config(config, clear_aggregate_metrics=False)
    _write_json_atomic(provenance_path, provenance)
    _write_json_atomic(plan_path, plan)
    return _jsonable(plan), plan_path


def load_plan(plan_path: str | pathlib.Path) -> tuple[dict, pathlib.Path]:
    """Load and minimally validate a generated assignment plan."""
    path = pathlib.Path(plan_path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported assignment Slurm plan schema: {plan.get('schema_version')!r}."
        )
    if pathlib.Path(plan.get("plan_path", "")) != path:
        raise ValueError(f"Plan path does not match its recorded absolute path: {path}")
    allocations = plan.get("allocations", [])
    if not allocations or len(allocations) > MAX_SLURM_JOBS:
        raise ValueError("Assignment Slurm plan has an invalid allocation count.")
    assignment_jobs = sum(
        allocation.get("phase") == "assignment" for allocation in allocations
    )
    metrics_jobs = sum(
        allocation.get("phase") in {"metrics", "metrics-finalize"}
        for allocation in allocations
    )
    finalizer_jobs = sum(
        allocation.get("phase") == "metrics-finalize" for allocation in allocations
    )
    if (
        assignment_jobs > MAX_ASSIGNMENT_JOBS
        or metrics_jobs > MAX_METRICS_JOBS
        or finalizer_jobs > MAX_METRICS_FINALIZER_JOBS
    ):
        raise ValueError("Assignment Slurm plan exceeds its phase job limits.")
    metrics_tasks = plan.get("metrics_tasks", [])
    if finalizer_jobs != int(bool(metrics_tasks)):
        raise ValueError(
            "Assignment Slurm plan must have exactly one finalizer when metrics "
            "tasks are present."
        )
    if metrics_tasks:
        if not plan.get("run_id"):
            raise ValueError("Assignment Slurm metrics plan is missing run_id.")
        fragment_dir = pathlib.Path(plan.get("metrics_fragment_dir", ""))
        if not fragment_dir.is_absolute() or not fragment_dir.is_dir():
            raise ValueError(
                "Assignment Slurm metrics fragment directory must be an existing "
                "absolute directory."
            )
        metric_subconfigs = [task.get("subconfig") for task in metrics_tasks]
        if len(metric_subconfigs) != len(set(metric_subconfigs)):
            raise ValueError("Assignment Slurm plan has duplicate metrics tasks.")
        for task in metrics_tasks:
            report_names = MarketGenerator._ordered_report_names(
                task.get("report_names", [])
            )
            if "citywide" not in report_names:
                raise ValueError(
                    "Assignment Slurm metrics tasks must include citywide metrics."
                )
    subconfigs = [entry.get("name") for entry in plan.get("subconfigs", [])]
    if len(subconfigs) != len(set(subconfigs)):
        raise ValueError("Assignment Slurm plan has duplicate subconfigs.")
    if any(task.get("subconfig") not in subconfigs for task in metrics_tasks):
        raise ValueError("Assignment Slurm metrics task has an unknown subconfig.")
    assigned_task_indices = []
    metrics_task_indices = []
    for allocation in allocations:
        cpus = int(allocation.get("cpus", 0))
        if cpus < 1 or cpus > MAX_CPUS_PER_NODE:
            raise ValueError(f"Invalid assignment Slurm allocation CPU count: {cpus}.")
        phase = allocation.get("phase")
        task_indices = allocation.get("task_indices")
        if not isinstance(task_indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in task_indices
        ):
            raise ValueError("Assignment Slurm allocation has invalid task indices.")
        if phase == "assignment":
            assigned_task_indices.extend(task_indices)
        elif phase == "metrics":
            metrics_task_indices.extend(task_indices)
        elif phase == "metrics-finalize":
            metrics_task_indices.extend(task_indices)
        else:
            raise ValueError(f"Unknown assignment Slurm phase {phase!r}.")
    if sorted(assigned_task_indices) != list(
        range(len(plan.get("assignment_tasks", [])))
    ):
        raise ValueError(
            "Assignment Slurm allocations do not cover each assignment task once."
        )
    if sorted(metrics_task_indices) != list(range(len(metrics_tasks))):
        raise ValueError("Assignment Slurm allocations do not cover each metrics task once.")
    return plan, path


def _job_script(
    *,
    job_name: str,
    log_path: pathlib.Path,
    command: list[str],
    workspace_root: pathlib.Path,
    cpus: int,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH -A soal",
        "#SBATCH -p soal",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --chdir={workspace_root}",
        f"#SBATCH --output={log_path}",
        "set -euo pipefail",
    ]
    lines.extend(
        f"export {name}={shlex.quote(value)}"
        for name, value in THREAD_ENVIRONMENT.items()
    )
    lines.append(f"exec {shlex.join(command)}")
    return "\n".join(lines) + "\n"


def _metrics_dependency_slots(plan: dict) -> dict[int, list[int]]:
    """Map each metrics allocation to assignment ID array slots it consumes."""
    assignment_slots = {}
    producers = defaultdict(set)
    for allocation_index, allocation in enumerate(plan["allocations"]):
        if allocation["phase"] != "assignment":
            continue
        slot = len(assignment_slots)
        assignment_slots[allocation_index] = slot
        for task_index in allocation["task_indices"]:
            task = plan["assignment_tasks"][task_index]
            producers[task["subconfig"]].add(slot)

    dependencies = {}
    for allocation_index, allocation in enumerate(plan["allocations"]):
        if allocation["phase"] not in {"metrics", "metrics-finalize"}:
            continue
        subconfigs = {
            plan["metrics_tasks"][task_index]["subconfig"]
            for task_index in allocation["task_indices"]
        }
        slots = sorted(
            {slot for subconfig in subconfigs for slot in producers[subconfig]}
        )
        if assignment_slots and not slots:
            raise ValueError(
                f"Metrics allocation {allocation_index} has no assignment producers."
            )
        dependencies[allocation_index] = slots
    return dependencies


def write_slurm_scripts(
    plan_path: str | pathlib.Path,
    *,
    script_dir: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Generate worker scripts and a non-submitting sbatch launcher."""
    plan, absolute_plan_path = load_plan(plan_path)
    workspace_root = pathlib.Path(plan["workspace_root"])
    directory = (
        pathlib.Path(script_dir).expanduser().resolve()
        if script_dir is not None
        else absolute_plan_path.parent / "scripts"
    )
    logs_dir = absolute_plan_path.parent / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    allocation_scripts = {}
    for index, allocation in enumerate(plan["allocations"]):
        phase = allocation["phase"]
        script_path = directory / f"{phase}-allocation-{index}.sh"
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "assignment.slurm",
            "allocation-worker",
            "--plan",
            str(absolute_plan_path),
            "--allocation-index",
            str(index),
        ]
        _write_text_atomic(
            script_path,
            _job_script(
                job_name=f"asg-{phase}-{index}",
                log_path=logs_dir / f"{phase}-allocation-{index}-%j.log",
                command=command,
                workspace_root=workspace_root,
                cpus=allocation["cpus"],
            ),
            executable=True,
        )
        allocation_scripts[index] = script_path.resolve()

    submit_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"submission_marker={shlex.quote(str(absolute_plan_path.parent / '.submitted'))}",
        'if ! mkdir "$submission_marker"; then',
        '  printf "This Slurm plan has already been submitted: %s\\n" '
        '"$submission_marker" >&2',
        "  exit 1",
        "fi",
        "assignment_ids=()",
        "metrics_ids=()",
        'finalizer_id=""',
        "cleanup_submission() {",
        "  local status=$? cancel_status=0",
        "  set +u",
        "  local all_ids=(\"${assignment_ids[@]}\" \"${metrics_ids[@]}\")",
        "  set -u",
        "  local submitted_ids=() job_id",
        '  for job_id in "${all_ids[@]}"; do',
        '    if [[ -n "$job_id" ]]; then',
        '      submitted_ids+=("$job_id")',
        "    fi",
        "  done",
        "  trap - EXIT",
        '  if (( status != 0 )); then',
        '    if [[ -n "$finalizer_id" ]]; then',
        '      submitted_ids+=("$finalizer_id")',
        "    fi",
        '    if (( ${#submitted_ids[@]} > 0 )); then',
        '      scancel "${submitted_ids[@]}" || cancel_status=$?',
        "    fi",
        "    if (( cancel_status == 0 )); then",
        '      rmdir "$submission_marker"',
        "    else",
        '      printf "Partial submission could not be cancelled; plan remains '
        'locked: %s\\n" "$submission_marker" >&2',
        "    fi",
        "  fi",
        '  exit "$status"',
        "}",
        "trap cleanup_submission EXIT",
        "submit_job() {",
        "  local raw_id job_id",
        '  raw_id=$(sbatch --parsable -A soal -p soal "$@")',
        "  job_id=${raw_id%%;*}",
        '  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then',
        "    printf 'Unexpected sbatch --parsable output: %s\\n' \"$raw_id\" >&2",
        "    return 1",
        "  fi",
        "  printf '%s\\n' \"$job_id\"",
        "}",
    ]
    metrics_allocations = []
    finalizer_allocations = []
    metrics_dependencies = _metrics_dependency_slots(plan)
    for index, allocation in enumerate(plan["allocations"]):
        if allocation["phase"] == "metrics":
            metrics_allocations.append((index, allocation))
            continue
        if allocation["phase"] == "metrics-finalize":
            finalizer_allocations.append((index, allocation))
            continue
        if allocation["phase"] != "assignment":
            raise ValueError(
                f"Unknown assignment Slurm phase {allocation['phase']!r}."
            )
        static_args = shlex.join(
            [
                "--ntasks=1",
                f"--cpus-per-task={allocation['cpus']}",
                "--chdir",
                str(workspace_root),
                str(allocation_scripts[index]),
            ]
        )
        submit_lines.append(f'assignment_ids+=("$(submit_job {static_args})")')
    if metrics_allocations:
        for index, allocation in metrics_allocations:
            static_args = shlex.join(
                [
                    "--ntasks=1",
                    f"--cpus-per-task={allocation['cpus']}",
                    "--chdir",
                    str(workspace_root),
                ]
            )
            metric_script = shlex.quote(str(allocation_scripts[index]))
            slots = metrics_dependencies[index]
            dependency = ""
            if slots:
                variable = f"metrics_dependency_{index}"
                references = ":".join(f"${{assignment_ids[{slot}]}}" for slot in slots)
                submit_lines.append(f'{variable}="{references}"')
                dependency = f'--dependency="afterany:${{{variable}}}" '
            submit_lines.append(
                f'metrics_ids+=("$(submit_job {static_args} '
                f'{dependency}{metric_script})")'
            )
    if finalizer_allocations:
        if len(finalizer_allocations) != 1:
            raise ValueError("Metrics tasks require exactly one finalizer allocation.")
        index, allocation = finalizer_allocations[0]
        static_args = shlex.join(
            [
                "--ntasks=1",
                f"--cpus-per-task={allocation['cpus']}",
                "--chdir",
                str(workspace_root),
            ]
        )
        finalizer_script = shlex.quote(str(allocation_scripts[index]))
        references = [
            *(f"${{metrics_ids[{slot}]}}" for slot in range(len(metrics_allocations))),
            *(
                f"${{assignment_ids[{slot}]}}"
                for slot in metrics_dependencies[index]
            ),
        ]
        if not references:
            raise ValueError("Metrics finalizer has no upstream dependencies.")
        references = ":".join(references)
        submit_lines.append(f'finalizer_dependency="{references}"')
        submit_lines.append(
            f'finalizer_id="$(submit_job {static_args} '
            f'--dependency="afterany:${{finalizer_dependency}}" '
            f'{finalizer_script})"'
        )

    submit_lines.append(
        f"printf 'Submitted {len(plan['allocations'])} Slurm allocations.\\n'"
    )
    submit_path = (directory / "submit.sh").resolve()
    _write_text_atomic(submit_path, "\n".join(submit_lines) + "\n", executable=True)
    return submit_path


def _subconfig_entry(plan: dict, subconfig_name: str) -> dict:
    matches = [entry for entry in plan["subconfigs"] if entry["name"] == subconfig_name]
    if len(matches) != 1:
        raise ValueError(f"Subconfig {subconfig_name!r} is not unique in the plan.")
    return matches[0]


def _create_market(plan: dict, subconfig_name: str):
    from .student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    entry = _subconfig_entry(plan, subconfig_name)
    return MarketGenerator(
        config=entry["config"],
        assignment_path=plan["assignment_folder"],
        write_config=False,
        write_aggregate_metrics=False,
    )


def _market_for_subconfig(plan: dict, subconfig_name: str):
    global _WORKER_MARKET_GENERATOR, _WORKER_MARKET_KEY
    market_key = (plan["plan_path"], subconfig_name)
    if _WORKER_MARKET_GENERATOR is None:
        _WORKER_MARKET_GENERATOR = _create_market(plan, subconfig_name)
    elif _WORKER_MARKET_KEY != market_key:
        entry = _subconfig_entry(plan, subconfig_name)
        _WORKER_MARKET_GENERATOR.reconfigure(
            entry["config"],
            plan["assignment_folder"],
            write_config=False,
        )
    _WORKER_MARKET_KEY = market_key
    return _WORKER_MARKET_GENERATOR


def _initialize_worker(plan_path: str | pathlib.Path) -> None:
    global _WORKER_PLAN, _WORKER_MARKET_GENERATOR, _WORKER_MARKET_KEY
    _WORKER_PLAN, _ = load_plan(plan_path)
    _WORKER_MARKET_GENERATOR = None
    _WORKER_MARKET_KEY = None


def _run_cached_assignment_task(task_index: int) -> None:
    if _WORKER_PLAN is None:
        raise RuntimeError("Assignment worker process was not initialized.")
    task = _WORKER_PLAN["assignment_tasks"][task_index]
    market = _market_for_subconfig(_WORKER_PLAN, task["subconfig"])
    market.simulate_target(
        task["subconfig"],
        task["iteration"],
        write_utility_output=task["write_utility_output"],
    )


def _run_cached_metrics_iteration(task_index: int, iteration: int | None) -> dict:
    if _WORKER_PLAN is None:
        raise RuntimeError("Assignment worker process was not initialized.")
    task = _WORKER_PLAN["metrics_tasks"][task_index]
    market = _market_for_subconfig(_WORKER_PLAN, task["subconfig"])
    return market.evaluate_saved_subconfig_iteration(task["subconfig"], iteration)


def _run_cached_metrics_batch(work_items: list[tuple[int, int | None]]) -> list[tuple]:
    results = []
    for task_index, iteration in work_items:
        try:
            payload = _run_cached_metrics_iteration(task_index, iteration)
            results.append((task_index, iteration, payload, None))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            results.append((task_index, iteration, None, error))
    return results


def _run_metrics_allocation(
    plan: dict, plan_path: pathlib.Path, allocation: dict
) -> bool:
    from .student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    work_items = []
    for task_index in allocation["task_indices"]:
        task = plan["metrics_tasks"][task_index]
        config = _subconfig_entry(plan, task["subconfig"])["config"]
        work_items.extend(
            (task_index, iteration) for iteration in _metrics_iterations(config)
        )

    work_batches = [
        [work_items[index] for index in indices]
        for indices in _task_batches(
            len(work_items), min(allocation["cpus"], len(work_items))
        )
    ]
    payloads = defaultdict(dict)
    failed_tasks = set()
    with ProcessPoolExecutor(
        max_workers=allocation["cpus"],
        initializer=_initialize_worker,
        initargs=(plan_path,),
    ) as executor:
        futures = {
            executor.submit(_run_cached_metrics_batch, work_batch): work_batch
            for work_batch in work_batches
        }
        for future in as_completed(futures):
            try:
                results = future.result()
            except Exception as exc:
                work_batch = futures[future]
                print(
                    f"Metrics worker batch {work_batch} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                failed_tasks.update(task_index for task_index, _ in work_batch)
                continue
            for task_index, iteration, payload, error in results:
                if error is None:
                    payloads[task_index][iteration] = payload
                    continue
                print(
                    f"Metrics task {task_index}, iteration {iteration!r} "
                    f"failed: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                failed_tasks.add(task_index)

    failed = bool(failed_tasks)
    for task_index in allocation["task_indices"]:
        if task_index in failed_tasks:
            continue
        task = plan["metrics_tasks"][task_index]
        config = _subconfig_entry(plan, task["subconfig"])["config"]
        try:
            ordered_payloads = [
                payloads[task_index][iteration]
                for iteration in _metrics_iterations(config)
            ]
            reports = MarketGenerator.combine_metric_batch_payloads(
                ordered_payloads,
                include_local_metrics=config.get("export-local-metrics", False),
            )
            expected_config_names = MarketGenerator.metric_payload_config_names(
                ordered_payloads
            )
            MarketGenerator.write_metric_fragment(
                plan["metrics_fragment_dir"],
                run_id=plan["run_id"],
                subconfig_name=task["subconfig"],
                reports=reports,
                expected_report_names=task["report_names"],
                expected_config_names=expected_config_names,
            )
        except Exception as exc:
            print(
                f"Metrics task {task_index} reduction failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            failed = True
    return failed


def run_assignment_worker(
    plan_path: str | pathlib.Path, subconfig_name: str, iteration: int
) -> None:
    """Execute one planned subconfig/iteration assignment task."""
    plan, _ = load_plan(plan_path)
    task_matches = [
        task
        for task in plan["assignment_tasks"]
        if task["subconfig"] == subconfig_name and task["iteration"] == iteration
    ]
    if len(task_matches) != 1:
        raise ValueError(
            f"Assignment task {subconfig_name!r}, iteration {iteration} is not unique "
            "in the plan."
        )
    market = _create_market(plan, subconfig_name)
    market.simulate_target(
        subconfig_name,
        iteration,
        write_utility_output=task_matches[0]["write_utility_output"],
    )


def run_metrics_worker(plan_path: str | pathlib.Path, subconfig_name: str) -> None:
    """Evaluate all saved iterations and write one subconfig fragment."""
    plan, _ = load_plan(plan_path)
    matches = [
        task for task in plan["metrics_tasks"] if task["subconfig"] == subconfig_name
    ]
    if len(matches) != 1:
        raise ValueError(f"No metrics task is planned for {subconfig_name!r}.")
    market = _create_market(plan, subconfig_name)
    reports = market.evaluate_saved_subconfig(subconfig_name)
    expected_config_names = market.expected_saved_metric_config_names()
    market.write_metric_fragment(
        plan["metrics_fragment_dir"],
        run_id=plan["run_id"],
        subconfig_name=subconfig_name,
        reports=reports,
        expected_report_names=matches[0]["report_names"],
        expected_config_names=expected_config_names,
    )


def run_metrics_finalizer(plan_path: str | pathlib.Path) -> None:
    """Validate all planned fragments and publish the aggregate exactly once."""
    from .student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    plan, absolute_plan_path = load_plan(plan_path)
    reports, fragments = MarketGenerator.combine_metric_fragments(
        plan["metrics_fragment_dir"],
        run_id=plan["run_id"],
        expected_fragments=plan["metrics_tasks"],
    )
    MarketGenerator.write_aggregate_metric_reports(
        plan["assignment_folder"],
        reports,
        manifest={
            "schema_version": MarketGenerator.METRIC_FRAGMENT_SCHEMA_VERSION,
            "run_id": plan["run_id"],
            "plan_path": str(absolute_plan_path),
            "subconfigs": [task["subconfig"] for task in plan["metrics_tasks"]],
            "fragments": fragments,
        },
    )


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


def run_allocation_worker(plan_path: str | pathlib.Path, allocation_index: int) -> int:
    """Execute all work tasks assigned to one Slurm allocation."""
    try:
        plan, absolute_plan_path = load_plan(plan_path)
        if allocation_index < 0 or allocation_index >= len(plan["allocations"]):
            raise IndexError(
                f"Allocation index {allocation_index} is outside the plan."
            )
        allocation = plan["allocations"][allocation_index]
        _validate_worker_cpus(allocation["cpus"])
        if allocation["phase"] == "assignment":
            function = _run_cached_assignment_task
        elif allocation["phase"] == "metrics":
            return (
                1
                if _run_metrics_allocation(plan, absolute_plan_path, allocation)
                else 0
            )
        elif allocation["phase"] == "metrics-finalize":
            if _run_metrics_allocation(plan, absolute_plan_path, allocation):
                return 1
            run_metrics_finalizer(absolute_plan_path)
            return 0
        else:
            raise ValueError(f"Unknown assignment Slurm phase {allocation['phase']!r}.")

        failed = False
        with ProcessPoolExecutor(
            max_workers=allocation["cpus"],
            initializer=_initialize_worker,
            initargs=(absolute_plan_path,),
        ) as executor:
            futures = {
                executor.submit(function, task_index): task_index
                for task_index in allocation["task_indices"]
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(
                        f"{allocation['phase'].title()} task "
                        f"{futures[future]} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    failed = True
        return 1 if failed else 0
    except Exception as exc:
        print(f"Assignment allocation failed: {exc}", file=sys.stderr, flush=True)
        return 1


def _plan_options(function):
    options = [
        click.option(
            "--frac", default=None, help="Override the config's frac variable."
        ),
        click.option(
            "--sample", default=None, help="Override the config's sample variable."
        ),
        click.option(
            "--plan-dir",
            type=click.Path(file_okay=False, path_type=pathlib.Path),
            help="Directory for plan, provenance, scripts, and logs.",
        ),
        click.option(
            "--assignment-folder",
            type=click.Path(file_okay=False, path_type=pathlib.Path),
            help="Override paths.assignment-folder for this run.",
        ),
        click.option(
            "--config",
            "config_path",
            required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
            help="Assignment YAML configuration.",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@click.group()
def cli() -> None:
    """Generate or submit independent SFUSD assignment jobs."""


@cli.command("generate")
@_plan_options
def generate_command(config_path, assignment_folder, plan_dir, sample, frac) -> None:
    """Resolve a config and generate scripts without calling Slurm."""
    plan, plan_path = build_slurm_plan(
        config_path,
        assignment_folder=assignment_folder,
        plan_dir=plan_dir,
        sample=sample,
        frac=frac,
    )
    submit_path = write_slurm_scripts(plan_path)
    click.echo(
        f"Planned {len(plan['assignment_tasks'])} assignment tasks, "
        f"{len(plan['metrics_tasks'])} metrics tasks, and "
        f"{len(plan['allocations'])} Slurm allocations."
    )
    click.echo(f"Plan: {plan_path}")
    click.echo(f"Submission script: {submit_path}")


@cli.command("submit")
@_plan_options
def submit_command(config_path, assignment_folder, plan_dir, sample, frac) -> None:
    """Generate a plan and explicitly submit it with sbatch."""
    _plan, plan_path = build_slurm_plan(
        config_path,
        assignment_folder=assignment_folder,
        plan_dir=plan_dir,
        sample=sample,
        frac=frac,
    )
    submit_path = write_slurm_scripts(plan_path)
    subprocess.run(["bash", str(submit_path)], cwd=WORKSPACE_ROOT, check=True)


@cli.command("allocation-worker", hidden=True)
@click.option("--allocation-index", required=True, type=int)
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
def allocation_worker_command(plan_path, allocation_index) -> None:
    """Run one batch of assignment or metrics tasks from a resolved plan."""
    raise SystemExit(run_allocation_worker(plan_path, allocation_index))


if __name__ == "__main__":
    cli()

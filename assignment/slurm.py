"""Plan, submit, and execute batched Slurm assignment jobs."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import pathlib
import re
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
from .slurm_graph import (
    build_job_graph,
    topological_jobs,
    validate_job_graph,
)
from .student_assignment.configerator import Configerator
from .student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN_SCHEMA_VERSION = 5
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


def _planned_assignment_tasks(subconfigs: list[dict]) -> list[dict]:
    tasks = []
    real_match_planned = False
    for entry in subconfigs:
        config = entry["config"]
        policies = config.get("policies", [])
        has_simulation = any(policy != "real_match" for policy in policies)
        has_real_match = "real_match" in policies
        start = config["iterations"]["start"]
        end = config["iterations"]["end"]
        if has_simulation:
            iterations = range(start, end)
        elif has_real_match and not real_match_planned:
            iterations = [start]
        else:
            continue
        for iteration in iterations:
            include_real_match = (
                has_real_match and not real_match_planned and iteration == start
            )
            tasks.append(
                {
                    "subconfig": entry["name"],
                    "iteration": iteration,
                    "include_real_match": include_real_match,
                    "write_utility_output": False,
                }
            )
            real_match_planned |= include_real_match
    if not tasks:
        raise ValueError("Slurm assignment runs require at least one assignment task.")
    simulation_subconfigs = {
        entry["name"]
        for entry in subconfigs
        if any(policy != "real_match" for policy in entry["config"].get("policies", []))
    }
    utility_task = next(
        (task for task in tasks if task["subconfig"] in simulation_subconfigs),
        tasks[0],
    )
    utility_task["write_utility_output"] = True
    return tasks


def _planned_metrics_tasks(subconfigs: list[dict]) -> list[dict]:
    return [
        {
            "subconfig": entry["name"],
            "report_names": (
                list(MarketGenerator.AGGREGATE_METRIC_FILES)
                if entry["config"].get("export-local-metrics", False)
                else ["citywide"]
            ),
        }
        for entry in subconfigs
        if entry["config"].get("export-aggregate-metrics", False)
    ]


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

    assignment_tasks = _planned_assignment_tasks(resolved_subconfigs)
    utility_task = next(
        task for task in assignment_tasks if task["write_utility_output"]
    )
    utility_owner = {
        key: utility_task[key] for key in ("subconfig", "iteration")
    }
    metrics_tasks = _planned_metrics_tasks(resolved_subconfigs)
    metrics_work_counts = [
        len(_metrics_iterations(entry["config"]))
        for entry in resolved_subconfigs
        if entry["config"].get("export-aggregate-metrics", False)
    ]
    jobs = build_job_graph(
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
        "jobs": jobs,
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
    subconfig_entries = plan.get("subconfigs")
    if not isinstance(subconfig_entries, list) or not subconfig_entries:
        raise ValueError("Assignment Slurm plan has no resolved subconfigs.")
    subconfigs = [entry.get("name") for entry in subconfig_entries]
    if any(not isinstance(name, str) or not name for name in subconfigs):
        raise ValueError("Assignment Slurm plan has an invalid subconfig name.")
    if len(subconfigs) != len(set(subconfigs)):
        raise ValueError("Assignment Slurm plan has duplicate subconfigs.")

    assignment_tasks = plan.get("assignment_tasks")
    if assignment_tasks != _planned_assignment_tasks(subconfig_entries):
        raise ValueError("Assignment Slurm tasks do not match the resolved subconfigs.")
    utility_owners = [
        task for task in assignment_tasks if task.get("write_utility_output")
    ]
    expected_owner = {
        key: utility_owners[0][key] for key in ("subconfig", "iteration")
    }
    if len(utility_owners) != 1 or plan.get("utility_output_owner") != expected_owner:
        raise ValueError("Assignment Slurm plan has an invalid utility output owner.")

    metrics_tasks = plan.get("metrics_tasks")
    if metrics_tasks != _planned_metrics_tasks(subconfig_entries):
        raise ValueError("Metrics tasks do not match the resolved subconfigs.")
    if metrics_tasks:
        if not plan.get("run_id"):
            raise ValueError("Assignment Slurm metrics plan is missing run_id.")
        fragment_dir = pathlib.Path(plan.get("metrics_fragment_dir", ""))
        if not fragment_dir.is_absolute() or not fragment_dir.is_dir():
            raise ValueError(
                "Assignment Slurm metrics fragment directory must be an existing "
                "absolute directory."
            )
        for task in metrics_tasks:
            report_names = MarketGenerator._ordered_report_names(
                task.get("report_names", [])
            )
            if "citywide" not in report_names:
                raise ValueError(
                    "Assignment Slurm metrics tasks must include citywide metrics."
                )
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Assignment Slurm plan has no job graph.")
    validate_job_graph(
        jobs,
        assignment_count=len(assignment_tasks),
        metrics_count=len(metrics_tasks),
    )
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

    for job in plan["jobs"]:
        job_id = job["id"]
        script_path = directory / f"{job_id}.sh"
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "assignment.slurm",
            "job-worker",
            "--plan",
            str(absolute_plan_path),
            "--job-id",
            job_id,
        ]
        _write_text_atomic(
            script_path,
            _job_script(
                job_name=f"asg-{job_id}",
                log_path=logs_dir / f"{job_id}-%j.log",
                command=command,
                workspace_root=workspace_root,
                cpus=job["cpus"],
            ),
            executable=True,
        )
    submit_command = [
        "uv",
        "run",
        "python",
        "-m",
        "assignment.slurm",
        "submit-plan",
        "--plan",
        str(absolute_plan_path),
        "--script-dir",
        str(directory),
    ]
    submit_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(workspace_root))}",
    ]
    submit_lines.append(f"exec {shlex.join(submit_command)}")
    submit_path = (directory / "submit.sh").resolve()
    _write_text_atomic(submit_path, "\n".join(submit_lines) + "\n", executable=True)
    return submit_path


def _slurm_job_id(output: str) -> str:
    match = re.fullmatch(r"([0-9]+)(?:;[^\n]*)?\n?", output)
    if match is None:
        raise RuntimeError(f"Unexpected sbatch --parsable output: {output!r}.")
    return match.group(1)


def submit_slurm_plan(
    plan_path: str | pathlib.Path,
    *,
    script_dir: str | pathlib.Path | None = None,
    runner=None,
) -> pathlib.Path:
    """Submit a planned graph and persist every scheduler job ID."""
    plan, absolute_plan_path = load_plan(plan_path)
    directory = (
        pathlib.Path(script_dir).expanduser().resolve()
        if script_dir is not None
        else absolute_plan_path.parent / "scripts"
    )
    scripts = {job["id"]: directory / f"{job['id']}.sh" for job in plan["jobs"]}
    missing_scripts = [str(path) for path in scripts.values() if not path.is_file()]
    if missing_scripts:
        raise FileNotFoundError(f"Missing Slurm worker scripts: {missing_scripts}.")

    submission_path = absolute_plan_path.parent / "submission.json"
    state = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "plan_path": str(absolute_plan_path),
        "status": "submitting",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "jobs": [],
    }
    try:
        with submission_path.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"Slurm plan already has submission state: {submission_path}."
        ) from exc

    run = runner or subprocess.run
    scheduler_ids = {}
    try:
        for job in topological_jobs(plan["jobs"]):
            command = ["sbatch", "--parsable"]
            for condition, upstream_ids in job["dependencies"].items():
                slurm_ids = ":".join(scheduler_ids[job_id] for job_id in upstream_ids)
                command.append(f"--dependency={condition}:{slurm_ids}")
            command.append(str(scripts[job["id"]]))
            result = run(command, capture_output=True, text=True, check=False)
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"sbatch failed for job {job['id']!r} with exit code "
                    f"{result.returncode}: {detail}"
                )
            scheduler_id = _slurm_job_id(result.stdout)
            scheduler_ids[job["id"]] = scheduler_id
            state["jobs"].append(
                {"job_id": job["id"], "slurm_job_id": scheduler_id}
            )
            _write_json_atomic(submission_path, state)
    except Exception as exc:
        state["status"] = "submission-failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        if scheduler_ids:
            try:
                cancel = run(
                    ["scancel", *scheduler_ids.values()],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if cancel.returncode:
                    raise RuntimeError(
                        cancel.stderr.strip()
                        or cancel.stdout.strip()
                        or f"scancel exited with code {cancel.returncode}"
                    )
            except Exception as cancel_exc:
                state["status"] = "cancellation-failed"
                state["cancellation_error"] = (
                    f"{type(cancel_exc).__name__}: {cancel_exc}"
                )
        _write_json_atomic(submission_path, state)
        raise

    state["status"] = "submitted"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(submission_path, state)
    return submission_path


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
        include_real_match=task["include_real_match"],
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


def _run_metrics_job(plan: dict, plan_path: pathlib.Path, job: dict) -> bool:
    from .student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    work_items = []
    for task_index in job["task_indices"]:
        task = plan["metrics_tasks"][task_index]
        config = _subconfig_entry(plan, task["subconfig"])["config"]
        work_items.extend(
            (task_index, iteration) for iteration in _metrics_iterations(config)
        )

    work_batches = [
        [work_items[index] for index in indices]
        for indices in _task_batches(
            len(work_items), min(job["cpus"], len(work_items))
        )
    ]
    payloads = defaultdict(dict)
    failed_tasks = set()
    with ProcessPoolExecutor(
        max_workers=job["cpus"],
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
    for task_index in job["task_indices"]:
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


def run_job_worker(plan_path: str | pathlib.Path, job_id: str) -> int:
    """Execute all work tasks assigned to one planned Slurm job."""
    try:
        plan, absolute_plan_path = load_plan(plan_path)
        matches = [job for job in plan["jobs"] if job["id"] == job_id]
        if len(matches) != 1:
            raise ValueError(f"Slurm job {job_id!r} is not unique in the plan.")
        job = matches[0]
        _validate_worker_cpus(job["cpus"])
        if job["kind"] == "assignment":
            function = _run_cached_assignment_task
        elif job["kind"] == "metrics-finalize":
            if _run_metrics_job(plan, absolute_plan_path, job):
                return 1
            run_metrics_finalizer(absolute_plan_path)
            return 0
        else:
            raise ValueError(f"Unknown Slurm job kind {job['kind']!r}.")

        failed = False
        with ProcessPoolExecutor(
            max_workers=job["cpus"],
            initializer=_initialize_worker,
            initargs=(absolute_plan_path,),
        ) as executor:
            futures = {
                executor.submit(function, task_index): task_index
                for task_index in job["task_indices"]
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(
                        f"Assignment task {futures[future]} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    failed = True
        return 1 if failed else 0
    except Exception as exc:
        print(f"Slurm job failed: {exc}", file=sys.stderr, flush=True)
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
        f"{len(plan['jobs'])} Slurm jobs."
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
    submission_path = submit_slurm_plan(plan_path, script_dir=submit_path.parent)
    click.echo(f"Submission: {submission_path}")


@cli.command("submit-plan", hidden=True)
@click.option(
    "--script-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
def submit_plan_command(plan_path, script_dir) -> None:
    """Submit an existing generated Slurm plan."""
    submission_path = submit_slurm_plan(plan_path, script_dir=script_dir)
    click.echo(f"Submission: {submission_path}")


@cli.command("job-worker", hidden=True)
@click.option("--job-id", required=True)
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
def job_worker_command(plan_path, job_id) -> None:
    """Run one batch of assignment or metrics tasks from a resolved plan."""
    raise SystemExit(run_job_worker(plan_path, job_id))


if __name__ == "__main__":
    cli()

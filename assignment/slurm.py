"""Plan, submit, and execute one-core Slurm assignment jobs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone

import click
from loaders import load_scenario

from .run_custom_config import _write_provenance_config, load_custom_config
from .student_assignment.configerator import Configerator


WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN_SCHEMA_VERSION = 1
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


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

    output_path = pathlib.Path(
        configurator.config["paths"]["assignment-folder"]
    ).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    plan_directory = (
        pathlib.Path(plan_dir).expanduser().resolve()
        if plan_dir is not None
        else output_path / "slurm"
    )
    plan_directory.mkdir(parents=True, exist_ok=True)
    plan_path = (plan_directory / "assignment_plan.json").resolve()
    provenance_path = (plan_directory / "provenance.json").resolve()

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
        {"subconfig": entry["name"]}
        for entry in resolved_subconfigs
        if entry["config"].get("export-aggregate-metrics", False)
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    provenance = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "workspace_root": str(WORKSPACE_ROOT),
        "source_config": str(source_path),
        "source_config_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "resolved_config": config,
        "data": _data_provenance(config),
    }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "workspace_root": str(WORKSPACE_ROOT),
        "source_config": str(source_path),
        "assignment_folder": str(output_path),
        "plan_path": str(plan_path),
        "provenance_path": str(provenance_path),
        "utility_output_owner": utility_owner,
        "subconfigs": resolved_subconfigs,
        "assignment_tasks": assignment_tasks,
        "metrics_tasks": metrics_tasks,
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
    return plan, path


def _slug(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-") or "subconfig"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{token[:40]}-{digest}"


def _job_script(
    *,
    job_name: str,
    log_path: pathlib.Path,
    command: list[str],
    workspace_root: pathlib.Path,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "#SBATCH -A soal",
        "#SBATCH -p soal",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
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

    assignment_scripts = {}
    for task in plan["assignment_tasks"]:
        name = task["subconfig"]
        iteration = task["iteration"]
        token = _slug(name)
        script_path = directory / f"assignment-{token}-iteration{iteration}.sh"
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "assignment.slurm",
            "assignment-worker",
            "--plan",
            str(absolute_plan_path),
            "--subconfig",
            name,
            "--iteration",
            str(iteration),
        ]
        _write_text_atomic(
            script_path,
            _job_script(
                job_name=f"asg-{token[:24]}-{iteration}",
                log_path=logs_dir / f"assignment-{token}-{iteration}-%j.log",
                command=command,
                workspace_root=workspace_root,
            ),
            executable=True,
        )
        assignment_scripts[(name, iteration)] = script_path.resolve()

    metrics_scripts = {}
    for task in plan["metrics_tasks"]:
        name = task["subconfig"]
        token = _slug(name)
        script_path = directory / f"metrics-{token}.sh"
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "assignment.slurm",
            "metrics-worker",
            "--plan",
            str(absolute_plan_path),
            "--subconfig",
            name,
        ]
        _write_text_atomic(
            script_path,
            _job_script(
                job_name=f"metrics-{token[:24]}",
                log_path=logs_dir / f"metrics-{token}-%j.log",
                command=command,
                workspace_root=workspace_root,
            ),
            executable=True,
        )
        metrics_scripts[name] = script_path.resolve()

    submit_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
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
    tasks_by_subconfig = {entry["name"]: [] for entry in plan["subconfigs"]}
    for task in plan["assignment_tasks"]:
        tasks_by_subconfig[task["subconfig"]].append(task)

    for index, entry in enumerate(plan["subconfigs"]):
        name = entry["name"]
        variable = f"assignment_ids_{index}"
        submit_lines.append(f"{variable}=()")
        for task in tasks_by_subconfig[name]:
            static_args = shlex.join(
                [
                    "--ntasks=1",
                    "--cpus-per-task=1",
                    "--chdir",
                    str(workspace_root),
                    str(assignment_scripts[(name, task["iteration"])]),
                ]
            )
            submit_lines.append(f'{variable}+=("$(submit_job {static_args})")')
        if name in metrics_scripts:
            dependency = f"dependency_{index}"
            submit_lines.append(
                f"{dependency}=$(IFS=:; printf '%s' \"${{{variable}[*]}}\")"
            )
            static_args = shlex.join(
                [
                    "--ntasks=1",
                    "--cpus-per-task=1",
                    "--chdir",
                    str(workspace_root),
                ]
            )
            metric_script = shlex.quote(str(metrics_scripts[name]))
            submit_lines.append(
                f"metrics_id_{index}=$(submit_job {static_args} "
                f'--dependency="afterok:${{{dependency}}}" {metric_script})'
            )

    submit_lines.append(
        f"printf 'Submitted {len(plan['assignment_tasks'])} assignment and "
        f"{len(plan['metrics_tasks'])} metrics jobs.\\n'"
    )
    submit_path = (directory / "submit.sh").resolve()
    _write_text_atomic(submit_path, "\n".join(submit_lines) + "\n", executable=True)
    return submit_path


def _subconfig_entry(plan: dict, subconfig_name: str) -> dict:
    matches = [entry for entry in plan["subconfigs"] if entry["name"] == subconfig_name]
    if len(matches) != 1:
        raise ValueError(f"Subconfig {subconfig_name!r} is not unique in the plan.")
    return matches[0]


def run_assignment_worker(
    plan_path: str | pathlib.Path, subconfig_name: str, iteration: int
) -> None:
    """Execute one planned subconfig/iteration assignment task."""
    from .student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

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
    entry = _subconfig_entry(plan, subconfig_name)
    market = MarketGenerator(
        config=entry["config"],
        assignment_path=plan["assignment_folder"],
        write_config=False,
        write_aggregate_metrics=False,
    )
    market.simulate_target(
        subconfig_name,
        iteration,
        write_utility_output=task_matches[0]["write_utility_output"],
    )


def run_metrics_worker(plan_path: str | pathlib.Path, subconfig_name: str) -> None:
    """Evaluate and merge all saved iterations for one planned subconfig."""
    from .student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    plan, _ = load_plan(plan_path)
    if not any(task["subconfig"] == subconfig_name for task in plan["metrics_tasks"]):
        raise ValueError(f"No metrics task is planned for {subconfig_name!r}.")
    entry = _subconfig_entry(plan, subconfig_name)
    market = MarketGenerator(
        config=entry["config"],
        assignment_path=plan["assignment_folder"],
        write_config=False,
        write_aggregate_metrics=False,
    )
    reports = market.evaluate_saved_subconfig(subconfig_name)
    MarketGenerator.merge_aggregate_metric_reports(
        plan["assignment_folder"], subconfig_name, reports
    )


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
        f"Planned {len(plan['assignment_tasks'])} assignment and "
        f"{len(plan['metrics_tasks'])} metrics jobs."
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


@cli.command("assignment-worker", hidden=True)
@click.option("--iteration", required=True, type=int)
@click.option("--subconfig", required=True)
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
def assignment_worker_command(plan_path, subconfig, iteration) -> None:
    """Run one assignment task from a resolved plan."""
    run_assignment_worker(plan_path, subconfig, iteration)


@cli.command("metrics-worker", hidden=True)
@click.option("--subconfig", required=True)
@click.option(
    "--plan",
    "plan_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
def metrics_worker_command(plan_path, subconfig) -> None:
    """Run one metrics task from a resolved plan."""
    run_metrics_worker(plan_path, subconfig)


if __name__ == "__main__":
    cli()

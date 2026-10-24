#!/usr/bin/env python3
"""Warm a sweep's feasible hints in Slurm jobs, before the sweep runs.

A task with ``hints: feasible`` solves its CP-SAT hint inside its own solve
budget, with whatever ``workers`` its step has, and every task that starts
before the hint is cached solves it again. Warming the hints first, in a job
with enough CPUs, means every task reads the hint back instead (the cache
key excludes ``workers``, ``seed`` and the time limit; see
``analysis/warm_feasible_hints.py``).

One job per centroid configuration in the sweep, each running
``warm_feasible_hints.py`` from this checkout, so the jobs warm the same cache
the sweep will read (cache keys depend on the checkout's absolute paths).

    uv run python analysis/submit_warm_feasible_hints.py SWEEP.yaml
    uv run python analysis/submit_warm_feasible_hints.py SWEEP.yaml --dry-run

Submit the sweep once the jobs finish (``squeue -u $USER``), or chain it with
``--dependency=afterok:<job ids>`` on the sweep's own submission. Each job's
log ends with ``OK in ...`` or ``FAILED ...`` per distinct feasible set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from benchmark.config import SimulationSweep
from benchmark.slurm import (
    GUROBI_LICENSE_SETUP,
    SBATCH_ACCOUNT,
    SBATCH_PARTITION,
    THREAD_ENVIRONMENT,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", help="Path to the sweep YAML")
    parser.add_argument(
        "--centroid",
        action="append",
        help="Centroid configuration to warm (repeatable; default: every one in the sweep)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="CPUs and CP-SAT workers per job (default: 8)"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=600.0,
        help="Seconds for each CP-SAT hint search (default: 600)",
    )
    parser.add_argument(
        "--log-dir",
        help="Directory for job logs (default: <execution.output_dir>/.slurm/feasible_hints)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the sbatch commands without submitting"
    )
    return parser.parse_args()


def sbatch_command(sweep_path: Path, centroid: str, log_dir: Path, args) -> list[str]:
    warm = shlex.join(
        [
            "uv",
            "run",
            "python",
            "analysis/warm_feasible_hints.py",
            str(sweep_path),
            centroid,
            "--workers",
            str(args.workers),
            "--time-limit",
            str(args.time_limit),
        ]
    )
    log_prefix = log_dir / f"{_safe(centroid)}-%j"
    return [
        "sbatch",
        "--parsable",
        "-A",
        SBATCH_ACCOUNT,
        "-p",
        SBATCH_PARTITION,
        f"--job-name=hint-{_safe(centroid)}",
        "--ntasks=1",
        f"--cpus-per-task={args.workers}",
        f"--chdir={PROJECT_ROOT}",
        f"--output={log_prefix}.out",
        f"--error={log_prefix}.err",
        f"--export=ALL,{','.join(THREAD_ENVIRONMENT)}",
        "--wrap",
        f"{GUROBI_LICENSE_SETUP} && {warm}",
    ]


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def main() -> int:
    args = parse_args()
    sweep_path = Path(args.sweep).resolve()
    sweep = SimulationSweep.from_yaml(sweep_path)
    tasks = sweep.generate_tasks()
    found = sorted({str(task.config.get("centroids_type")) for task in tasks})
    centroids = args.centroid or found
    missing = sorted(set(centroids) - set(found))
    if missing:
        raise SystemExit(f"No task in {args.sweep} uses centroid(s) {missing}.")
    hinted = [
        task
        for task in tasks
        if "feasible"
        in {
            str(task.config.get("hints")),
            *(
                str((step.get("options") or {}).get("hints"))
                for step in task.config.get("steps") or []
            ),
        }
    ]
    if not hinted:
        print(f"note: no task in {args.sweep} asks for hints: feasible", file=sys.stderr)

    log_dir = Path(
        args.log_dir
        or Path(sweep.execution.output_dir) / ".slurm" / "feasible_hints"
    )
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    job_ids = []
    for centroid in centroids:
        command = sbatch_command(sweep_path, centroid, log_dir, args)
        if args.dry_run:
            print(shlex.join(command))
            continue
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split(";")[0]
        job_ids.append(job_id)
        print(f"centroid={centroid} job={job_id} log={log_dir}/{_safe(centroid)}-{job_id}.out")
    if job_ids:
        print(f"afterok dependency: --dependency=afterok:{':'.join(job_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

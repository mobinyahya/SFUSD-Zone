#!/usr/bin/env python3
"""Warm every data cache a sweep reads by running a cut-down copy of it.

Graphs (with their KaHIP partitions), closer-neighbor supports, student and
distance tables, map and heatmap geometry are content-addressed by the data
and the geography, not by search settings. So one short run per data cell,
through the sweep's own task and matching code, builds exactly the entries
the full sweep will read, instead of hand-assembling them and risking a key
that differs.

The copy keeps every data cell and every task entry's steps and levels, and
cuts the rest: one matching policy, the first seed and objective weight of
each task, step time limits capped at ``--time-limit``, and matching
iterations cut to ``--iterations``. Results go to ``--output-dir`` and are
throwaway.

Run it from the checkout the sweep will run from, since source hashes are
taken relative to it:

    uv run python analysis/warm_sweep_caches.py SWEEP.yaml --output-dir DIR

``--direct`` skips the cut-down run and builds only the shared data caches
(dataset tables, graph levels, and the policy market per data block and
``matching_policy``) through :mod:`benchmark.warmup`, the same warm-up each
Slurm allocation runs before its fan-out. It writes nothing under
``--output-dir`` and is much cheaper, but leaves the per-run caches (closer-
neighbor supports, map geometry) to the sweep.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from benchmark import run as benchmark_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", help="Path to the sweep YAML")
    parser.add_argument("--output-dir", required=True, help="Throwaway output root")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=20.0,
        help="Cap on each step's time limit (s)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Matching lotteries per run (default: 1)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None,
        help="Parallel tasks (default: the Slurm allocation's CPUs)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Warm only datasets, graphs, and policy markets (benchmark.warmup)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the cut-down configs and list tasks",
    )
    return parser.parse_args()


def _first(value):
    return value[0] if isinstance(value, list) else value


def _cut_task(task: dict, time_limit: float) -> dict:
    task = dict(task)
    for key in ("seed", "objective_weights", "matching_policy"):
        if key in task:
            task[key] = _first(task[key])
    steps = []
    for step in task.get("steps") or []:
        step = dict(step)
        if "time_limit" in step:
            step["time_limit"] = min(float(step["time_limit"]), time_limit)
        steps.append(step)
    if steps:
        task["steps"] = steps
    return task


def warm_direct(sweep_path: Path, max_workers: int | None, dry_run: bool) -> int:
    """Warm each distinct dataset of the sweep once, in parallel processes."""
    from concurrent.futures import ProcessPoolExecutor

    from benchmark.config import SimulationSweep
    from benchmark.warmup import format_warm_summary, warm_dataset, warm_specs_for_tasks

    specs = warm_specs_for_tasks(SimulationSweep.from_yaml(str(sweep_path)).generate_tasks())
    print(f"{len(specs)} dataset(s) to warm", flush=True)
    if dry_run:
        for spec in specs:
            print(f"  {spec.key[:12]} levels={spec.levels} policies={spec.policies}")
        return 0
    failures = 0
    with ProcessPoolExecutor(max_workers=max_workers or 1, max_tasks_per_child=1) as pool:
        for summary in pool.map(warm_dataset, specs):
            print(format_warm_summary(summary), flush=True)
            failures += bool(summary.get("errors"))
    return int(failures > 0)


def main() -> int:
    args = parse_args()
    sweep_path = Path(args.sweep).resolve()
    if args.direct:
        return warm_direct(sweep_path, args.max_workers, args.dry_run)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(sweep_path.read_text())

    raw["name"] = f"{raw.get('name', 'sweep')}_cache_warm"
    raw["mode"] = "run"
    raw["tasks"] = [_cut_task(task, args.time_limit) for task in raw["tasks"]]
    execution = dict(raw.get("execution") or {})
    execution.update(output_dir=str(out), skip_existing=True, rerun_failed=True)
    if args.max_workers:
        execution["max_workers"] = args.max_workers
        execution["capacity"] = args.max_workers
    raw["execution"] = execution

    matching = dict(raw.get("matching") or {})
    if matching.get("enabled"):
        source = (sweep_path.parent / matching["config"]).resolve()
        cut = yaml.safe_load(source.read_text())
        start = int((cut.get("iterations") or {}).get("start", 0))
        cut["iterations"] = {"start": start, "end": start + args.iterations}
        cut["paths"] = {
            **(cut.get("paths") or {}),
            "assignment-folder": str(out / "assignments_raw"),
        }
        cut["output_dir"] = str(out / "matching_root")
        matching_path = out / "matching.yaml"
        matching_path.write_text(yaml.safe_dump(cut, sort_keys=False))
        matching["config"] = str(matching_path)
        raw["matching"] = matching

    derived = out / "sweep.yaml"
    derived.write_text(yaml.safe_dump(raw, sort_keys=False))
    if args.dry_run:
        from benchmark.config import SimulationSweep

        tasks = SimulationSweep.from_yaml(derived).generate_tasks()
        print(f"{len(tasks)} task(s) in {derived}")
        return 0
    print(f"Warming caches with {derived}", flush=True)
    benchmark_run.main([str(derived)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

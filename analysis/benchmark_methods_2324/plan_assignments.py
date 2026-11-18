#!/usr/bin/env python3
"""Plan assignment jobs for every saved run of a finished benchmark sweep.

``benchmark.slurm`` plans matching from a sweep YAML, regenerating each task
and skipping the ones whose saved result matches its config hash. That only
works while the code still reproduces the saved hashes, and
benchmark_methods_2324 predates the ``objective`` rename (#27): every task
hashes differently today, so the sweep planner would rerun all 2,100
optimizations. This script skips the YAML and reads the runs themselves.

For each run folder under ``--root`` it writes the root assignment target from
the saved zonings (``assignment_zones.csv``, or the skip marker for an
infeasible or partial final stage; see
``benchmark.assignment.prepare_completed_run_target``). Each eligible zoning
is matched in its own run's data, with ``--assignment-filter`` values added to
the assignment group. The targets go to the generated-zone Slurm planner, or
are run in process with ``--local``.

    # Plan and write scripts (on scdt), then submit from sc:
    uv run python analysis/benchmark_methods_2324/plan_assignments.py \\
        --root /soalnas/share/data/school_choice/local_runs/benchmark_methods_2324_combined \\
        --config assignment/configs/benchmark_methods_2324_assignment.yaml
    # Smoke test one zoning locally:
    uv run python analysis/benchmark_methods_2324/plan_assignments.py ... --local --limit 1
"""

from __future__ import annotations

import argparse
import collections
import copy
from pathlib import Path

from assignment.generated_zones import (
    GENERATED_ZONE_FILENAME,
    SKIP_MARKER_FILENAME,
    run_generated_zone_assignments,
)
from assignment.slurm import build_generated_zone_slurm_plan, write_slurm_scripts
from benchmark.assignment import (
    _final_level,
    prepare_completed_run_target,
    zone_building_blocks,
)
from benchmark.results import discover_run_dirs
from benchmark.runner import load_manifest
from benchmark.slurm import SLURM_DIRNAME
from loaders import load_scenario
from optimization.levels import LevelSpec


def _filters(values: list[str]) -> dict[str, str]:
    filters = {}
    for value in values:
        key, sep, setting = value.partition("=")
        if not sep or not key:
            raise SystemExit(f"--assignment-filter expects key=value, got {value!r}.")
        filters[key] = setting
    return filters


def run_targets(
    root: Path, assignment_filters: dict[str, str], *, limit: int | None = None
) -> tuple[list[dict], collections.Counter]:
    """Write every run's root target and return the eligible ones."""
    targets = []
    counts = collections.Counter()
    vintages = {}
    for run_dir in discover_run_dirs(str(root)):
        manifest = load_manifest(run_dir)
        if manifest.get("status") == "ERROR":
            counts["error"] += 1
            continue
        config = manifest["config"]
        level = _final_level(config)
        if not prepare_completed_run_target(run_dir, level):
            counts["ineligible"] += 1
            continue
        data = copy.deepcopy(config["data"])
        data.setdefault("overrides", {}).setdefault("filters", {}).setdefault(
            "assignment", {}
        ).update(assignment_filters)
        key = repr(data)
        if key not in vintages:
            vintages[key] = load_scenario(data).filter(
                "optimization", "geography_vintage"
            )
        folder = Path(run_dir)
        targets.append(
            {
                "id": f"{manifest['task_id']}-root",
                "zone_file": str(folder / GENERATED_ZONE_FILENAME),
                "skip_marker": str(folder / SKIP_MARKER_FILENAME),
                "zone_building_blocks": zone_building_blocks(
                    LevelSpec.parse(level).unit
                ),
                "geography_vintage": vintages[key],
                "data": data,
            }
        )
        counts["eligible"] += 1
        if limit is not None and len(targets) >= limit:
            break
    return targets, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assignment-dir", default="assignments")
    parser.add_argument(
        "--assignment-filter",
        action="append",
        default=["special_programs=exclude_only_special"],
        help="key=value added to each run's assignment filters (repeatable).",
    )
    # The soal QOS allows 20 submitted jobs; leave room for the baseline.
    parser.add_argument("--max-assignment-jobs", type=int, default=8)
    parser.add_argument("--max-metrics-jobs", type=int, default=6)
    parser.add_argument("--limit", type=int, help="Use only the first N zonings.")
    parser.add_argument("--local", action="store_true", help="Run in process.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    targets, counts = run_targets(
        root, _filters(args.assignment_filter), limit=args.limit
    )
    print(f"Targets: {dict(counts)}")
    if not targets:
        raise SystemExit("No eligible zonings.")
    assignment_folder = root / args.assignment_dir
    if args.local:
        run_generated_zone_assignments(
            args.config,
            targets,
            assignment_folder=assignment_folder,
            workers=args.workers,
        )
        print(f"Assignments under {assignment_folder}")
        return
    plan, plan_path = build_generated_zone_slurm_plan(
        args.config,
        targets,
        assignment_folder=assignment_folder,
        plan_dir=root / SLURM_DIRNAME / "assignment",
        max_assignment_jobs=args.max_assignment_jobs,
        max_metrics_jobs=args.max_metrics_jobs,
    )
    submit_path = write_slurm_scripts(plan_path)
    print(
        f"Planned {len(plan['assignment_tasks'])} assignment tasks, "
        f"{len(plan['metrics_tasks'])} metrics tasks, {len(plan['jobs'])} jobs."
    )
    print(f"Plan: {plan_path}")
    print(f"Submit (on sc): {submit_path}")


if __name__ == "__main__":
    main()

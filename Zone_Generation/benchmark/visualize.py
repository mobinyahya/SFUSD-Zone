"""Render visualizations for saved benchmark sweep outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from Zone_Generation.benchmark.config import SimulationSweep
from Zone_Generation.benchmark.runner import load_solutions
from Zone_Generation.optimization.visualization import visualize_solutions


@dataclass
class SweepVisualizationSummary:
    """Aggregate counts from rendering one saved benchmark sweep."""

    total_runs: int = 0
    rendered_runs: int = 0
    rendered_figures: int = 0
    skipped_runs: int = 0
    skipped_stages: int = 0
    failed_runs: int = 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render PNG visualizations for saved benchmark sweep runs."
    )
    parser.add_argument("config", help="Path to simulation sweep YAML.")
    parser.add_argument(
        "--viz-stages",
        choices=["final", "all"],
        default="final",
        help="Render only the final saved stage or every saved stage. Default: final.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional directory for cached geometry artifacts.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first run that cannot be loaded or rendered.",
    )
    args = parser.parse_args(argv)

    summary = visualize_sweep(
        args.config,
        stages=args.viz_stages,
        artifact_dir=args.artifact_dir,
        fail_fast=args.fail_fast,
    )
    print(
        "Visualization complete: "
        f"runs={summary.total_runs}, "
        f"rendered_runs={summary.rendered_runs}, "
        f"figures={summary.rendered_figures}, "
        f"skipped_runs={summary.skipped_runs}, "
        f"skipped_stages={summary.skipped_stages}, "
        f"failed_runs={summary.failed_runs}"
    )


def visualize_sweep(
    config_path: str | Path,
    *,
    stages: str = "final",
    artifact_dir: str | Path | None = None,
    fail_fast: bool = False,
) -> SweepVisualizationSummary:
    """Render maps for every task output directory declared by a sweep YAML."""

    sweep = SimulationSweep.from_yaml(str(config_path))
    tasks = sweep.generate_tasks()
    summary = SweepVisualizationSummary(total_runs=len(tasks))

    for task in tasks:
        run_dir = Path(task.output_dir).expanduser()
        try:
            solutions, _, _ = load_solutions(str(run_dir))
            if not solutions:
                summary.skipped_runs += 1
                print(f"SKIP {run_dir}: no saved stages")
                continue

            results = visualize_solutions(
                solutions,
                output_dir=run_dir,
                stages=stages,
                artifact_dir=artifact_dir,
            )
            figure_count = sum(len(result.figure_paths) for result in results)
            skipped_count = sum(1 for result in results if result.skipped)

            summary.rendered_figures += figure_count
            summary.skipped_stages += skipped_count
            if figure_count:
                summary.rendered_runs += 1
                saved = ", ".join(
                    str(path)
                    for result in results
                    for path in result.figure_paths
                )
                print(f"RENDER {run_dir}: {saved}")
            else:
                summary.skipped_runs += 1
                reasons = ", ".join(
                    f"{result.stage}: {result.skipped}"
                    for result in results
                    if result.skipped
                )
                print(f"SKIP {run_dir}: {reasons or 'no figures produced'}")
        except Exception as exc:
            summary.failed_runs += 1
            print(f"ERROR {run_dir}: {exc}")
            if fail_fast:
                raise

    return summary


if __name__ == "__main__":
    main()

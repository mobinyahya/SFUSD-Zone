"""Render visualizations for saved benchmark sweep outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

from benchmark.config import (
    BenchmarkTask,
    SimulationSweep,
    VisualizationRunConfig,
)
from benchmark.runner import (
    MANIFEST_FILENAME,
    load_manifest,
    load_solutions,
    write_json,
)
from optimization.visualization import RenderResult, visualize_solutions


VISUALIZATION_MANIFEST_SCHEMA_VERSION = 1


@dataclass
class SweepVisualizationSummary:
    """Aggregate counts from rendering one saved benchmark sweep."""

    total_runs: int = 0
    rendered_runs: int = 0
    rendered_figures: int = 0
    cached_runs: int = 0
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
        default=None,
        help="Override visualization.stages from the sweep config.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional cache-root override for content-addressed geometry artifacts.",
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
        f"cached_runs={summary.cached_runs}, "
        f"skipped_runs={summary.skipped_runs}, "
        f"skipped_stages={summary.skipped_stages}, "
        f"failed_runs={summary.failed_runs}"
    )


def visualize_sweep(
    config_path: str | Path,
    *,
    stages: str | None = None,
    artifact_dir: str | Path | None = None,
    fail_fast: bool = False,
) -> SweepVisualizationSummary:
    """Render maps for every task output directory declared by a sweep YAML."""

    sweep = SimulationSweep.from_yaml(str(config_path))
    tasks = sweep.generate_tasks()
    summary = SweepVisualizationSummary(total_runs=len(tasks))
    settings = VisualizationRunConfig(
        enabled=True,
        stages=stages or sweep.visualization.stages,
        artifact_dir=(
            str(Path(artifact_dir).expanduser().resolve())
            if artifact_dir is not None
            else sweep.visualization.artifact_dir
        ),
    )

    for task in tasks:
        run_dir = Path(task.output_dir).expanduser()
        try:
            results, cached = ensure_task_visualizations(task, settings)
            if cached:
                summary.cached_runs += 1
                print(f"CACHE {run_dir}: existing visualizations")
                continue
            if results is None:
                summary.skipped_runs += 1
                print(f"SKIP {run_dir}: no saved stages")
                continue
            figure_count = sum(len(result.figure_paths) for result in results)
            skipped_count = sum(1 for result in results if result.skipped)

            summary.rendered_figures += figure_count
            summary.skipped_stages += skipped_count
            if figure_count:
                summary.rendered_runs += 1
                saved = ", ".join(
                    str(path) for result in results for path in result.figure_paths
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


def ensure_task_visualizations(
    task: BenchmarkTask,
    settings: VisualizationRunConfig,
) -> tuple[list[RenderResult] | None, bool]:
    """Render one task unless its versioned output artifacts are complete."""

    run_dir = Path(task.output_dir).expanduser()
    manifest = load_manifest(str(run_dir))
    if manifest.get("config_hash") != task.config_hash:
        raise ValueError(f"Manifest config hash does not match task {task.task_id}.")
    if visualization_is_current(manifest, run_dir, settings):
        return [], True

    solutions, optimization_config, manifest = load_solutions(str(run_dir))
    if not solutions:
        return None, False
    results = render_task_visualizations(
        solutions,
        optimization_config,
        run_dir,
        settings,
        manifest,
    )
    write_json(str(run_dir / MANIFEST_FILENAME), manifest)
    return results, False


def render_task_visualizations(
    solutions,
    optimization_config,
    run_dir: str | Path,
    settings: VisualizationRunConfig,
    manifest: dict,
) -> list[RenderResult]:
    """Render maps and attach a reusable artifact record to a run manifest."""

    output_dir = Path(run_dir).expanduser()
    results = visualize_solutions(
        solutions,
        output_dir=output_dir,
        stages=settings.stages,
        config=optimization_config,
        artifact_dir=settings.artifact_dir,
    )
    manifest["visualization"] = {
        "schema_version": VISUALIZATION_MANIFEST_SCHEMA_VERSION,
        "stages": settings.stages,
        "artifacts": [
            {
                "stage": result.stage,
                "figures": [
                    os.path.relpath(path, output_dir) for path in result.figure_paths
                ],
                "geometry_artifact": (
                    str(result.geometry_artifact)
                    if result.geometry_artifact is not None
                    else None
                ),
                "skipped": result.skipped,
            }
            for result in results
        ],
    }
    return results


def visualization_is_current(
    manifest: dict,
    run_dir: str | Path,
    settings: VisualizationRunConfig,
) -> bool:
    """Return whether a manifest references complete current visualization output."""

    record = manifest.get("visualization")
    if not isinstance(record, dict):
        return False
    if record.get("schema_version") != VISUALIZATION_MANIFEST_SCHEMA_VERSION:
        return False
    if record.get("stages") != settings.stages:
        return False
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False

    root = Path(run_dir).expanduser().resolve()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return False
        figures = artifact.get("figures")
        if not isinstance(figures, list):
            return False
        if not figures and not artifact.get("skipped"):
            return False
        for figure in figures:
            path = (root / str(figure)).resolve()
            if (
                not path.is_relative_to(root)
                or not path.is_file()
                or path.stat().st_size == 0
            ):
                return False
    return True


if __name__ == "__main__":
    main()

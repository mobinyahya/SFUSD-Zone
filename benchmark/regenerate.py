"""Metrics-only regeneration for saved benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os

from benchmark.config import (
    BenchmarkTask,
    VisualizationRunConfig,
    optimization_config_hash,
)
from benchmark.results import discover_run_dirs
from benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    load_solutions,
    result_payload_for,
    write_json,
)
from metrics import MetricsCalculator


@dataclass
class RegenerationResult:
    total: int = 0
    regenerated: int = 0
    skipped: int = 0
    failed: int = 0


def regenerate_metrics(
    root_folder: str,
    *,
    strict: bool = True,
    compute_stage_metrics: bool = False,
    visualization: VisualizationRunConfig | None = None,
    fail_fast: bool = False,
    dataset_factory=None,
) -> RegenerationResult:
    """Recompute metrics for every manifest under ``root_folder``."""

    result = RegenerationResult()
    for run_dir in discover_run_dirs(root_folder):
        result.total += 1
        try:
            dataset = None
            if dataset_factory is not None:
                from benchmark.runner import load_manifest
                from benchmark.config import (
                    optimization_config_from_dict,
                )

                manifest_for_dataset = load_manifest(run_dir)
                config_for_dataset = optimization_config_from_dict(
                    manifest_for_dataset["config"]
                )
                dataset = dataset_factory(config_for_dataset, manifest_for_dataset)
            solutions, config, manifest = load_solutions(run_dir, dataset=dataset)
            if not solutions:
                result.skipped += 1
                continue
            task = BenchmarkTask(
                task_id=str(manifest["task_id"]),
                config_hash=str(
                    manifest.get("config_hash")
                    or optimization_config_hash(manifest["config"])
                ),
                config={k: v for k, v in manifest["config"].items() if k != "unit"},
                output_dir=run_dir,
                capacity_slots=int(manifest.get("capacity_slots") or config.workers),
            )
            calculator = MetricsCalculator(
                solutions,
                config=config,
                strict=strict,
                compute_stage_metrics=compute_stage_metrics,
            )
            metrics = calculator.compute()
            if visualization and visualization.enabled:
                from benchmark.visualize import (
                    render_task_visualizations,
                    visualization_is_current,
                )

                if not visualization_is_current(manifest, run_dir, visualization):
                    render_task_visualizations(
                        solutions,
                        config,
                        run_dir,
                        visualization,
                        manifest,
                    )
            payload = result_payload_for(
                metrics=metrics,
                config=config,
                solutions=solutions,
                task=task,
            )
            write_json(os.path.join(run_dir, RESULT_FILENAME), payload)
            manifest["status"] = payload.get("status") or manifest.get("status")
            manifest["final_stage"] = metrics.run.get("final_stage")
            manifest["total_wall_time"] = payload.get(
                "total_wall_time", manifest.get("total_wall_time")
            )
            manifest["metrics_regenerated_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            write_json(os.path.join(run_dir, MANIFEST_FILENAME), manifest)
            result.regenerated += 1
        except Exception:
            result.failed += 1
            if fail_fast:
                raise
    return result

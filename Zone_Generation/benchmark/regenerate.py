"""Metrics-only regeneration for saved benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os

from Zone_Generation.benchmark.config import BenchmarkTask, stable_hash
from Zone_Generation.benchmark.choice_metrics import preserve_choice_metrics_payload
from Zone_Generation.benchmark.matching import preserve_matching_payload
from Zone_Generation.benchmark.results import discover_run_dirs
from Zone_Generation.benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    load_solutions,
    result_payload_for,
    write_json,
)
from Zone_Generation.metrics import MetricsCalculator


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
                from Zone_Generation.benchmark.runner import load_manifest
                from Zone_Generation.benchmark.config import (
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
                    manifest.get("config_hash") or stable_hash(manifest["config"])
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
            payload = result_payload_for(
                metrics=metrics,
                config=config,
                solutions=solutions,
                task=task,
            )
            previous_payload = _load_previous_result(
                os.path.join(run_dir, RESULT_FILENAME)
            )
            if calculator.context.solution.feasible:
                preserve_matching_payload(payload, previous_payload)
                preserve_choice_metrics_payload(payload, previous_payload)
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


def _load_previous_result(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

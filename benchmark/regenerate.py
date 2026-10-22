"""Metrics-only regeneration for saved benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmark.config import BenchmarkTask, VisualizationRunConfig
from benchmark.results import discover_run_dirs
from benchmark.runner import evaluate_optimization_task, load_manifest


@dataclass
class RegenerationResult:
    total: int = 0
    regenerated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: dict[str, str] = field(default_factory=dict)


def regenerate_metrics(
    root_folder: str,
    *,
    strict: bool = True,
    compute_stage_metrics: bool = False,
    visualization: VisualizationRunConfig | None = None,
    fail_fast: bool = False,
    dataset_factory=None,
) -> RegenerationResult:
    """Re-evaluate every manifest under ``root_folder`` from its saved stages.

    Runs the same evaluation as a sweep, so a regenerated run ends exactly as a
    freshly evaluated one would. Runs whose optimization failed or saved no
    stage are skipped. A run that fails to regenerate keeps its saved files
    and is reported in ``errors``. ``dataset_factory(config, manifest)``
    supplies the dataset the stages are rebuilt on, in place of the
    configured data.
    """

    result = RegenerationResult()
    for run_dir in discover_run_dirs(root_folder):
        result.total += 1
        try:
            manifest = load_manifest(run_dir)
            if manifest.get("phase") == "optimization_error" or not manifest.get(
                "stages"
            ):
                result.skipped += 1
                continue
            task = BenchmarkTask.from_manifest(manifest, run_dir)
            dataset = (
                dataset_factory(task.optimization_config(), manifest)
                if dataset_factory is not None
                else None
            )
            evaluate_optimization_task(
                task,
                strict_metrics=strict,
                compute_stage_metrics=compute_stage_metrics,
                visualization=visualization,
                manifest=manifest,
                dataset=dataset,
                regenerating=True,
            )
        except Exception as exc:
            result.failed += 1
            result.errors[str(run_dir)] = str(exc) or exc.__class__.__name__
            if fail_fast:
                raise
            continue
        result.regenerated += 1
    return result

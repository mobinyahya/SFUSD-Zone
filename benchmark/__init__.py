"""Optimization-native benchmark sweeps."""

from benchmark.config import (
    BenchmarkTask,
    ChoiceMetricsRunConfig,
    ExecutionConfig,
    MatchingConfigSpec,
    MatchingRunConfig,
    MetricsRunConfig,
    SimulationSweep,
)
from benchmark.parallel import BatchResult, run_sweep, run_tasks
from benchmark.regenerate import regenerate_metrics
from benchmark.results import discover_run_dirs
from benchmark.runner import TaskResult, run_optimization_task
from benchmark.choice_metrics import (
    ChoiceMetricsBatchResult,
    ChoiceMetricsResult,
    run_choice_metrics_for_existing_runs,
)

__all__ = [
    "BenchmarkTask",
    "ChoiceMetricsRunConfig",
    "ExecutionConfig",
    "MatchingConfigSpec",
    "MatchingRunConfig",
    "MetricsRunConfig",
    "SimulationSweep",
    "BatchResult",
    "TaskResult",
    "ChoiceMetricsBatchResult",
    "ChoiceMetricsResult",
    "run_sweep",
    "run_tasks",
    "run_optimization_task",
    "run_choice_metrics_for_existing_runs",
    "regenerate_metrics",
    "discover_run_dirs",
]

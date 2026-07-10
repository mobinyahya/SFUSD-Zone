"""Contiguity helpers for optimization metrics."""

from __future__ import annotations

from Config.metrics_config import MetricColumns
from metrics.base import MetricOutput, MetricsContext


def compute(context: MetricsContext) -> MetricOutput:
    return MetricOutput(metrics={MetricColumns.CONTIGUOUS: int(is_contiguous(context))})


def is_contiguous(context: MetricsContext) -> bool:
    if not context.assignment:
        return False
    return context.solution.is_contiguous()

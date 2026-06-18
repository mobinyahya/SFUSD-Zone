"""Choice-utility metrics from optimization strategy metadata.

The new optimization does not call the legacy utility evaluator from metrics. Choice
models attach their outputs to ``ZoneSolution.metadata``; metrics simply expose
that information when present.
"""

from __future__ import annotations

from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.metrics.base import MetricOutput, MetricsContext


def compute(context: MetricsContext) -> MetricOutput:
    utility = context.solution.metadata.get("choice_utility")
    if utility is None:
        return MetricOutput()
    return MetricOutput(metrics={MetricColumns.FINAL_CHOICE_UTILITY: utility})

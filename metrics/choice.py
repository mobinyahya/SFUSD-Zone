"""Preassignment school-choice utility metrics for zoning solutions."""

from __future__ import annotations

from choice.models import build_mnl_choice_model
from Config.metrics_config import MetricColumns
from loaders import load_scenario
from metrics.base import MetricOutput, MetricsContext


def compute(context: MetricsContext) -> MetricOutput:
    if not context.assignment:
        return MetricOutput()

    data_config = context.config.get("data")
    if data_config is None:
        return MetricOutput()

    method = str(context.config.get("choice_model_method", "logsum"))
    model = build_mnl_choice_model(
        load_scenario(data_config),
        method=method,
    )
    utility = model.preassignment_utility(context.problem, context.assignment)
    run = {
        "choice_preassignment_utility": {
            "model": "mnl",
            "method": method,
            "utility": utility,
        }
    }
    return MetricOutput(
        metrics={MetricColumns.CHOICE_TOTAL_PREASSIGNMENT_UTILITY: utility},
        run=run,
    )

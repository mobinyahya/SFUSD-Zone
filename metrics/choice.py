"""Preassignment school-choice utility metrics for zoning solutions."""

from __future__ import annotations

from choice.models import get_configured_choice_model
from Config.metrics_config import MetricColumns
from metrics.base import MetricOutput, MetricsContext


def compute(context: MetricsContext) -> MetricOutput:
    if not context.assignment:
        return MetricOutput()

    model = get_configured_choice_model(context.config)
    utility = model.preassignment_utility(context.problem, context.assignment)
    run = {
        "choice_preassignment_utility": {
            "model": str(context.config.get("choice_model", "mnl")),
            "method": (
                str(context.config.get("choice_model_method", "logsum"))
                if str(context.config.get("choice_model", "mnl")) == "mnl"
                else None
            ),
            "utility": utility,
        }
    }
    return MetricOutput(
        metrics={MetricColumns.CHOICE_TOTAL_PREASSIGNMENT_UTILITY: utility},
        run=run,
    )

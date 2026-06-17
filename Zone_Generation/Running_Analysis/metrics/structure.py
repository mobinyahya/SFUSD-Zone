"""Final zoning structure metrics."""

from __future__ import annotations

from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.pipeline.data.contiguity import boundary_edges
from Zone_Generation.Running_Analysis.metrics.base import MetricOutput, MetricsContext
from Zone_Generation.Running_Analysis.metrics.contiguity import is_contiguous
from Zone_Generation.Running_Analysis.metrics.solution_code import compute_solution_code


def compute(context: MetricsContext) -> MetricOutput:
    num_zones = len(context.zone_nodes)
    boundary_cost = boundary_edges(context.G, context.assignment) if context.assignment else 0
    metrics = {
        MetricColumns.NUM_ZONES: num_zones,
        MetricColumns.BOUNDARY_COST: boundary_cost,
        MetricColumns.FINAL_BOUNDARY_COST: boundary_cost,
        MetricColumns.COMPACTNESS: boundary_cost / num_zones if num_zones else 0.0,
        MetricColumns.CONTIGUOUS: int(is_contiguous(context)),
        MetricColumns.SOLUTION_CODE: compute_solution_code(context.area_assignment),
    }
    return MetricOutput(metrics=metrics)

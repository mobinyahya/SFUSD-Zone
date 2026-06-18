"""Final zoning structure metrics."""

from __future__ import annotations

from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.Running_Analysis.metrics.base import MetricOutput, MetricsContext
from Zone_Generation.Running_Analysis.metrics.contiguity import is_contiguous
from Zone_Generation.Running_Analysis.metrics.solution_code import compute_solution_code
from Zone_Generation.Running_Analysis.metrics.spatial import compute_spatial_metrics


def compute(context: MetricsContext) -> MetricOutput:
    num_zones = len(context.zone_nodes)
    spatial = compute_spatial_metrics(context.solution, context.config)
    metrics = {
        MetricColumns.NUM_ZONES: num_zones,
        MetricColumns.CUT_EDGES: spatial.cut_edges,
        MetricColumns.FINAL_CUT_EDGES: spatial.cut_edges,
        MetricColumns.FRACTIONAL_CUT_EDGES: spatial.fractional_cut_edges,
        MetricColumns.AVG_REOCK_SCORE: spatial.avg_reock_score,
        MetricColumns.AVG_POLSBY_POPPER_SCORE: spatial.avg_polsby_popper_score,
        MetricColumns.CONTIGUOUS: int(is_contiguous(context)),
        MetricColumns.SOLUTION_CODE: compute_solution_code(context.area_assignment),
    }
    return MetricOutput(metrics=metrics)

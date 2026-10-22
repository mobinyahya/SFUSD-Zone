"""Final zoning structure metrics."""

from __future__ import annotations

from dataclasses import asdict

from metrics.base import MetricOutput, MetricsContext
from metrics.columns import MetricColumns
from metrics.solution_code import compute_solution_code


def compute(context: MetricsContext) -> MetricOutput:
    num_zones = len(context.zone_nodes)
    spatial = context.spatial_metrics(context.solution)
    metrics = {
        MetricColumns.NUM_ZONES: num_zones,
        **asdict(spatial),
        MetricColumns.FINAL_CUT_EDGES: spatial.cut_edges,
        MetricColumns.CONTIGUOUS: int(
            bool(context.assignment) and context.solution.is_contiguous()
        ),
        MetricColumns.SOLUTION_CODE: compute_solution_code(context.area_assignment),
    }
    return MetricOutput(metrics=metrics, run={"final_cut_edges": spatial.cut_edges})

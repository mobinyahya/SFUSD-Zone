"""Final zoning structure metrics."""

from __future__ import annotations

from metrics.base import MetricOutput, MetricsContext
from metrics.columns import MetricColumns
from metrics.solution_code import compute_solution_code
from metrics.spatial import SPATIAL_FIELDS

#: Per-zone shape fields; a zone without positive-area geometry gets ``None``.
ZONE_SHAPE_FIELDS = ("polsby_popper_score", "reock_score", "area_sq_m", "perimeter_m")


def compute(context: MetricsContext) -> MetricOutput:
    num_zones = len(context.zone_nodes)
    spatial = context.spatial_metrics(context.solution)
    metrics = {
        MetricColumns.NUM_ZONES: num_zones,
        **{field: getattr(spatial, field) for field in SPATIAL_FIELDS},
        MetricColumns.FINAL_CUT_EDGES: spatial.cut_edges,
        MetricColumns.CONTIGUOUS: int(
            bool(context.assignment) and context.solution.is_contiguous()
        ),
        MetricColumns.SOLUTION_CODE: compute_solution_code(context.area_assignment),
    }
    zone_data = {
        zone_id: {
            field: spatial.zone_scores.get(zone_id, {}).get(field)
            for field in ZONE_SHAPE_FIELDS
        }
        for zone_id in context.zone_nodes
    }
    return MetricOutput(
        metrics=metrics,
        zone_data=zone_data,
        run={"final_cut_edges": spatial.cut_edges},
    )

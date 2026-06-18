"""Demographic balance and seat-disparity metrics."""

from __future__ import annotations

from Zone_Generation.Config.Constants import AALPI_ETHNICITIES, AREA_ETHNICITIES
from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.Running_Analysis.metrics.base import MetricOutput, MetricsContext

_ETH_MAD_COLUMNS = {
    "Ethnicity_Black_or_African_American": MetricColumns.BLACK_MAD,
    "Ethnicity_Hispanic/Latinx": MetricColumns.HISPANIC_MAD,
    "Ethnicity_White": MetricColumns.WHITE_MAD,
    "Ethnicity_Asian": MetricColumns.ASIAN_MAD,
    "Ethnicity_PacificIslander": MetricColumns.PACIFIC_ISLANDER_MAD,
}

_ETH_RANGE_COLUMNS = {
    "Ethnicity_Black_or_African_American": MetricColumns.BLACK_RANGE,
    "Ethnicity_Hispanic/Latinx": MetricColumns.HISPANIC_RANGE,
    "Ethnicity_White": MetricColumns.WHITE_RANGE,
    "Ethnicity_Asian": MetricColumns.ASIAN_RANGE,
    "Ethnicity_PacificIslander": MetricColumns.PACIFIC_ISLANDER_RANGE,
}


def compute(context: MetricsContext) -> MetricOutput:
    per_zone_counts: dict[int, dict[str, float]] = {}
    zone_data: dict[int, dict] = {}

    for zone_id, nodes in context.zone_nodes.items():
        counts = _empty_counts()
        for node in nodes:
            attrs = context.G.nodes[node]
            counts["ge_students"] += float(attrs.get("ge_students", 0.0))
            counts["ge_capacity"] += float(attrs.get("ge_capacity", 0.0))
            counts["FRL"] += float(attrs.get("FRL", 0.0))
            for ethnicity in AREA_ETHNICITIES:
                counts[ethnicity] += float(attrs.get(ethnicity, 0.0))

        counts["AALPI"] = sum(counts[ethnicity] for ethnicity in AALPI_ETHNICITIES)
        per_zone_counts[zone_id] = counts
        zone_data[zone_id] = _zone_demographics(counts)
        zone_data[zone_id]["ge_capacity"] = counts["ge_capacity"]
        zone_data[zone_id]["seat_disparity"] = _seat_disparity(counts)

    nonempty = [c for c in per_zone_counts.values() if c["ge_students"] > 0]
    metrics: dict[str, float] = {}

    district_students = sum(c["ge_students"] for c in per_zone_counts.values())
    district_frl = (
        sum(c["FRL"] for c in per_zone_counts.values()) / district_students
        if district_students
        else 0.0
    )
    district_eth = context.G.graph.get("R", {})
    district_aalpi = sum(float(district_eth.get(e, 0.0)) for e in AALPI_ETHNICITIES)

    frl_props = [_share(c, "FRL") for c in nonempty]
    metrics[MetricColumns.FRL_MAD] = _mad(frl_props, district_frl)
    metrics[MetricColumns.FRL_RANGE] = _range(frl_props)

    for ethnicity, column in _ETH_MAD_COLUMNS.items():
        props = [_share(c, ethnicity) for c in nonempty]
        metrics[column] = _mad(props, float(district_eth.get(ethnicity, 0.0)))
        metrics[_ETH_RANGE_COLUMNS[ethnicity]] = _range(props)

    aalpi_props = [_share(c, "AALPI") for c in nonempty]
    metrics[MetricColumns.AALPI_MAD] = _mad(aalpi_props, district_aalpi)
    metrics[MetricColumns.AALPI_RANGE] = _range(aalpi_props)

    seat_values = [
        abs(value["seat_disparity"])
        for value in zone_data.values()
        if value["seat_disparity"] is not None
    ]
    metrics[MetricColumns.SEAT_DISPARITY] = _mean(seat_values)

    return MetricOutput(metrics=metrics, zone_data=zone_data)


def _empty_counts() -> dict[str, float]:
    counts = {ethnicity: 0.0 for ethnicity in AREA_ETHNICITIES}
    counts.update({"ge_students": 0.0, "ge_capacity": 0.0, "FRL": 0.0})
    return counts


def _zone_demographics(counts: dict[str, float]) -> dict:
    students = counts["ge_students"]
    ethnicity_pcts = {
        ethnicity: _share(counts, ethnicity) for ethnicity in AREA_ETHNICITIES
    }
    return {
        "ge_students": students,
        "frl_pct": _share(counts, "FRL"),
        "aalpi_pct": sum(ethnicity_pcts[e] for e in AALPI_ETHNICITIES),
        "ethnicity_pcts": ethnicity_pcts,
    }


def _seat_disparity(counts: dict[str, float]) -> float | None:
    students = counts["ge_students"]
    if students <= 0:
        return None
    return (counts["ge_capacity"] - students) / students


def _share(counts: dict[str, float], key: str) -> float:
    students = counts["ge_students"]
    return counts.get(key, 0.0) / students if students > 0 else 0.0


def _mad(values: list[float], reference: float) -> float:
    return _mean([abs(value - reference) for value in values])


def _range(values: list[float]) -> float:
    return max(values) - min(values) if len(values) >= 2 else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

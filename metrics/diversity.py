"""Demographic balance and seat-disparity metrics.

Shares, ``seat_disparity`` and the balance columns are taken over the problem's
own population, ``students``/``capacity`` (``problem.student_attribute`` and
``capacity_attribute``: all-program counts under ``program_population: All``).
``ge_*`` and ``all_prog_*`` report both populations under their own names.
"""

from __future__ import annotations

from Config.Constants import (
    AALPI_ETHNICITIES,
    AREA_ETHNICITIES,
    SUPPLEMENTARY_ETHNICITIES,
    UNREPORTED_ETHNICITIES,
)
from metrics.base import MetricOutput, MetricsContext
from metrics.columns import MetricColumns

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
    student_attribute = context.problem.student_attribute
    capacity_attribute = context.problem.capacity_attribute
    supplementary = context.supplementary_ethnicity_counts

    for zone_id, nodes in context.zone_nodes.items():
        counts = _empty_counts()
        for node in nodes:
            attrs = context.G.nodes[node]
            counts["students"] += float(attrs.get(student_attribute, 0.0))
            counts["capacity"] += float(attrs.get(capacity_attribute, 0.0))
            for key in _POPULATION_COUNTS:
                counts[key] += float(attrs.get(key, 0.0))
            counts["FRL"] += float(attrs.get("FRL", 0.0))
            for ethnicity in AREA_ETHNICITIES:
                counts[ethnicity] += float(attrs.get(ethnicity, 0.0))
            if supplementary is not None:
                for ethnicity, value in supplementary[int(node)].items():
                    counts[ethnicity] += value

        counts["AALPI"] = sum(counts[ethnicity] for ethnicity in AALPI_ETHNICITIES)
        counts["Ethnicity_Unreported"] = sum(
            counts[ethnicity] for ethnicity in UNREPORTED_ETHNICITIES
        )
        per_zone_counts[zone_id] = counts
        zone_data[zone_id] = _zone_demographics(counts, supplementary is not None)
        zone_data[zone_id]["seat_disparity"] = _seat_disparity(counts)

    nonempty = [c for c in per_zone_counts.values() if c["students"] > 0]
    metrics: dict[str, float] = {}

    district_students = sum(c["students"] for c in per_zone_counts.values())
    if context.solution.metadata.get("partial_assignment"):
        district_frl = float(context.G.graph.get("F", 0.0))
    else:
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

    run = {
        "zone_population": {
            "students": student_attribute,
            "capacity": capacity_attribute,
        }
    }
    return MetricOutput(metrics=metrics, zone_data=zone_data, run=run)


# Node counts reported per zone under their own names, beside the problem's
# own ``students``/``capacity``.
_POPULATION_COUNTS = (
    "ge_students",
    "ge_capacity",
    "all_prog_students",
    "all_prog_capacity",
)


def _empty_counts() -> dict[str, float]:
    counts = {
        ethnicity: 0.0 for ethnicity in AREA_ETHNICITIES + SUPPLEMENTARY_ETHNICITIES
    }
    counts.update(dict.fromkeys(_POPULATION_COUNTS, 0.0))
    counts.update({"students": 0.0, "capacity": 0.0, "FRL": 0.0})
    return counts


def _zone_demographics(counts: dict[str, float], has_supplementary: bool) -> dict:
    ethnicity_pcts = {
        ethnicity: _share(counts, ethnicity) for ethnicity in AREA_ETHNICITIES
    }
    supplementary_pcts = (
        {
            ethnicity: _share(counts, ethnicity)
            for ethnicity in SUPPLEMENTARY_ETHNICITIES + ["Ethnicity_Unreported"]
        }
        if has_supplementary
        else None
    )
    return {
        "students": counts["students"],
        "capacity": counts["capacity"],
        **{key: counts[key] for key in _POPULATION_COUNTS},
        "frl_pct": _share(counts, "FRL"),
        "aalpi_pct": sum(ethnicity_pcts[e] for e in AALPI_ETHNICITIES),
        "ethnicity_pcts": ethnicity_pcts,
        "supplementary_ethnicity_pcts": supplementary_pcts,
    }


def _seat_disparity(counts: dict[str, float]) -> float | None:
    students = counts["students"]
    if students <= 0:
        return None
    return (counts["capacity"] - students) / students


def _share(counts: dict[str, float], key: str) -> float:
    students = counts["students"]
    return counts.get(key, 0.0) / students if students > 0 else 0.0


def _mad(values: list[float], reference: float) -> float:
    return _mean([abs(value - reference) for value in values])


def _range(values: list[float]) -> float:
    return max(values) - min(values) if len(values) >= 2 else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

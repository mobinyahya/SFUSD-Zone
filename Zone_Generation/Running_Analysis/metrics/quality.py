"""School-quality balance metrics."""

from __future__ import annotations

from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.Running_Analysis.metrics.base import MetricOutput, MetricsContext


def compute(context: MetricsContext) -> MetricOutput:
    school_data = context.G.graph.get("school_data", {})
    zone_data: dict[int, dict] = {}
    math_values: list[float] = []
    english_values: list[float] = []

    for zone_id, schools in context.zone_schools.items():
        avg_math = _weighted_average_score(schools, school_data, "math_score")
        avg_english = _weighted_average_score(schools, school_data, "english_score")
        zone_data[zone_id] = {
            "avg_math_score": avg_math,
            "avg_eng_score": avg_english,
        }
        if avg_math is not None:
            math_values.append(avg_math)
        if avg_english is not None:
            english_values.append(avg_english)

    metrics = {
        MetricColumns.MAD_MATH_SCORE: _mad(math_values),
        MetricColumns.MAD_ENG_SCORE: _mad(english_values),
        MetricColumns.MATH_SCORE_RANGE: _range(math_values),
        MetricColumns.ENG_SCORE_RANGE: _range(english_values),
    }
    return MetricOutput(metrics=metrics, zone_data=zone_data)


def _weighted_average_score(
    schools: list[int], school_data: dict, field: str
) -> float | None:
    total = 0.0
    weight = 0.0
    for school_id in schools:
        data = school_data.get(school_id, {})
        score = _score_value(data, field)
        if score is None or score <= 0:
            continue
        capacity = float(data.get("ge_capacity", data.get("all_prog_capacity", 1)) or 1)
        if capacity <= 0:
            capacity = 1.0
        total += score * capacity
        weight += capacity
    return total / weight if weight > 0 else None


def _score_value(data: dict, field: str) -> float | None:
    aliases = {
        "math_score": ("math_score", "math_scores_1819"),
        "english_score": ("english_score", "eng_scores_1819"),
    }[field]
    for key in aliases:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum(abs(value - mean) for value in values) / len(values)


def _range(values: list[float]) -> float:
    return max(values) - min(values) if len(values) >= 2 else 0.0

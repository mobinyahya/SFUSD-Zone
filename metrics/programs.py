"""Program-access metrics for optimization solutions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from Config.Constants import PROGRAM_CATEGORIES
from metrics.columns import MetricColumns
from loaders.tables import load_program_records
from metrics.base import MetricOutput, MetricsContext
from optimization.data.loaders import PROGRAM_ROLE

LANGUAGE_PROGRAMS = set(PROGRAM_CATEGORIES["Language Programs"])
SPECIAL_EDUCATION = set(PROGRAM_CATEGORIES["Special Education"])


def compute(context: MetricsContext) -> MetricOutput:
    programs_by_school = context.school_programs
    zone_data: dict[int, dict[str, Any]] = {}
    total_counts: list[int] = []
    language_counts: list[int] = []
    special_ed_counts: list[int] = []
    per_type_counts: dict[str, list[int]] = {}

    for zone_id, schools in context.zone_schools.items():
        zone_programs: dict[str, int] = {}
        language_count = 0
        special_ed_count = 0
        for school_id in schools:
            for program in programs_by_school.get(school_id, []):
                zone_programs[program] = zone_programs.get(program, 0) + 1
                if program in LANGUAGE_PROGRAMS:
                    language_count += 1
                if program in SPECIAL_EDUCATION:
                    special_ed_count += 1

        total = sum(zone_programs.values())
        zone_data[zone_id] = {
            "programs": zone_programs,
            "total_programs": total,
            "language_immersion_count": language_count,
            "special_ed_count": special_ed_count,
        }
        total_counts.append(total)
        language_counts.append(language_count)
        special_ed_counts.append(special_ed_count)
        for program, count in zone_programs.items():
            per_type_counts.setdefault(program, []).append(count)

    num_zones = len(context.zone_nodes)
    metrics = {
        MetricColumns.AVG_TOTAL_PROGRAMS: _zone_average(total_counts, num_zones),
        MetricColumns.AVG_LANGUAGE_IMMERSION: _zone_average(language_counts, num_zones),
        MetricColumns.AVG_SPECIAL_ED: _zone_average(special_ed_counts, num_zones),
    }
    for program, counts in per_type_counts.items():
        metrics[MetricColumns.program_column(program)] = _zone_average(
            counts, num_zones
        )

    return MetricOutput(metrics=metrics, zone_data=zone_data)


def ge_schools(context: MetricsContext) -> set[int]:
    programs = context.school_programs
    if programs:
        return {sid for sid, values in programs.items() if "GE" in values}

    school_data = context.G.graph.get("school_data", {})
    inferred = {
        sid
        for sid, data in school_data.items()
        if float(data.get("ge_capacity", data.get("capacity", 0)) or 0) > 0
    }
    if inferred:
        return inferred
    return set(context.school_to_node)


def school_programs(context: MetricsContext) -> dict[int, list[str]]:
    """Program types per school, from the graph or the run's program table.

    Real graphs carry no program columns, so the run's own
    ``optimization.programs`` table is read with the optimization selectors --
    the same table that sized the graph's capacities.
    """
    from_graph = _programs_from_graph(context.G.graph.get("school_data", {}))
    if from_graph or context.data_scenario is None:
        return from_graph

    programs = load_program_records(
        context.data_scenario,
        PROGRAM_ROLE,
        filter_group="optimization",
        low_memory=False,
    )
    programs = programs.dropna(subset=["school_id", "program_type"])
    return {
        int(school_id): sorted(set(group.astype(str)))
        for school_id, group in programs.groupby("school_id")["program_type"]
    }


def _programs_from_graph(school_data: dict) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for school_id, data in school_data.items():
        programs: list[str] = []
        for key in ("program_types", "programs", "program_type"):
            if key in data:
                programs.extend(_normalize_programs(data[key]))
        if programs:
            out[int(school_id)] = sorted(set(programs))
    return out


def _normalize_programs(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _zone_average(values: list[int], num_zones: int) -> float:
    return sum(values) / num_zones if num_zones else 0.0

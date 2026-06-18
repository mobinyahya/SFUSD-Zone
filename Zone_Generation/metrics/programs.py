"""Program-access metrics for optimization solutions."""

from __future__ import annotations

import functools
import os
from collections.abc import Iterable
from typing import Any

import pandas as pd

from Zone_Generation.Config.Constants import PROGRAM_CATEGORIES, get_sfusd_path
from Zone_Generation.Config.metrics_config import MetricColumns
from Zone_Generation.metrics.base import MetricOutput, MetricsContext

LANGUAGE_PROGRAMS = set(PROGRAM_CATEGORIES["Language Programs"])
SPECIAL_EDUCATION = set(PROGRAM_CATEGORIES["Special Education"])


def compute(context: MetricsContext) -> MetricOutput:
    programs_by_school = school_programs(context)
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
        metrics[MetricColumns.program_column(program)] = _zone_average(counts, num_zones)

    return MetricOutput(metrics=metrics, zone_data=zone_data)


def ge_schools(context: MetricsContext) -> set[int]:
    programs = school_programs(context)
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
    from_graph = _programs_from_graph(context.G.graph.get("school_data", {}))
    if from_graph:
        return from_graph

    path = context.config.get("programs_path") or _default_programs_path(context)
    if not path:
        return {}
    return _load_programs_csv(os.path.expanduser(path))


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


@functools.lru_cache(maxsize=8)
def _load_programs_csv(path: str) -> dict[int, list[str]]:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, OSError):
        return {}
    if "school_id" not in df.columns or "program_type" not in df.columns:
        return {}

    out: dict[int, list[str]] = {}
    for _, row in df.iterrows():
        try:
            school_id = int(row["school_id"])
        except (TypeError, ValueError):
            continue
        program = row["program_type"]
        if pd.isna(program):
            continue
        out.setdefault(school_id, []).append(str(program))
    return {sid: sorted(set(programs)) for sid, programs in out.items()}


def _default_programs_path(context: MetricsContext) -> str:
    is_local = bool(context.config.get("is_local", False))
    return f"{get_sfusd_path(is_local)}/Data/Cleaned/programs_withMissionBay_2324.csv"


def _zone_average(values: list[int], num_zones: int) -> float:
    return sum(values) / num_zones if num_zones else 0.0

"""The fitted ``choice.estimate`` utility table and its keys onto a zoning graph.

Both consumers of the estimate -- the MNL zoning utility in :mod:`choice.mnl`
and the matching-welfare markets in :mod:`optimization.data.mid` -- read it
through :func:`load_estimate`, so a student or program the estimate names is
matched the same way in both, and a malformed table fails the same way.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from loaders import DataScenario, read_csv_source

from assignment.student_assignment.market_generator.utility_model import UtilityModel
from optimization.problem import ZoneProblem


identity_text = UtilityModel._identity_text


def load_estimate(
    data: DataScenario,
    student_ids: Iterable[Any],
    program_ids: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """The estimate's rows for ``student_ids``, indexed by :func:`identity_text`.

    Columns are restricted to ``program_ids`` when given, and are every program
    the estimate names otherwise. A row label matches a student either exactly
    or after dropping a two- or four-digit year prefix (``2324-123`` names
    student ``123``); students the estimate does not name get no row. Duplicate
    rows or columns, and NaN or positive-infinite utilities, are rejected; an
    unranked program is ``-inf``.
    """
    source = data.source("choice.estimate")
    UtilityModel._validate_csv_header(source.path)
    frame = read_csv_source(source, dtype={"studentno": "string"})
    student_keys = [identity_text(value) for value in student_ids]
    if len(student_keys) != len(set(student_keys)):
        raise ValueError("Student identities are ambiguous after normalization.")
    required_students = set(student_keys)

    student_labels = [
        UtilityModel._normalize_labeled_identity(value, required_students)
        for value in frame.pop("studentno")
    ]
    if len(student_labels) != len(set(student_labels)):
        raise ValueError("Choice utility CSV has duplicate student rows.")
    frame.index = pd.Index(student_labels, name="studentno")

    if program_ids is None:
        program_labels = [
            UtilityModel._normalize_program_label(value) for value in frame.columns
        ]
        required_programs = set(program_labels)
    else:
        program_keys = [identity_text(value) for value in program_ids]
        if len(program_keys) != len(set(program_keys)):
            raise ValueError("Program identities are ambiguous after normalization.")
        required_programs = set(program_keys)
        program_labels = [
            UtilityModel._normalize_program_label(value, required_programs)
            for value in frame.columns
        ]
    if len(program_labels) != len(set(program_labels)):
        raise ValueError("Choice utility CSV has duplicate program columns.")
    frame.columns = program_labels
    selected_programs = sorted(required_programs & set(program_labels))
    selected_students = sorted(required_students & set(student_labels))
    numeric = frame.loc[selected_students, selected_programs].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric.to_numpy(dtype=float)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Choice utility CSV contains NaN or positive infinity.")
    return numeric


def student_area_column(problem: ZoneProblem) -> str:
    """The student-table column naming each student's ``problem`` base unit."""
    return {
        "Block": "census_block",
        "BlockGroup": "census_blockgroup",
        "Tract": "census_tract",
        "attendance_area": "attendance_area",
    }[problem.level.unit]


def area_key(value: Any) -> str | None:
    """Canonical text for an area id, so ``1001``, ``1001.0`` and ``"1001"`` agree."""
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        value = str(value).strip()
        return value or None


def node_area_ids(attrs: dict[str, Any]) -> list[str]:
    """The base-unit area keys a graph node covers."""
    values = attrs["block_ids"] if "block_ids" in attrs else [attrs.get("area_id")]
    return [key for key in map(area_key, values) if key is not None]


def area_to_node(problem: ZoneProblem) -> dict[str, int]:
    return {
        key: int(node)
        for node, attrs in problem.G.nodes(data=True)
        for key in node_area_ids(attrs)
    }


def school_to_node(problem: ZoneProblem) -> dict[int, int]:
    """The node each school sits on; a school on no node is absent."""
    return {
        int(school_id): int(node)
        for node, attrs in problem.G.nodes(data=True)
        for school_id in attrs.get("school_ids", [])
    }


def program_school(program_id: str) -> int:
    """The school of an estimate column such as ``"413-GE-KG"``."""
    return int(str(program_id).split("-", 1)[0])

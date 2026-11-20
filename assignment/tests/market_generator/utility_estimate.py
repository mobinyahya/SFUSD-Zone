"""A utility estimate covering every 2026-27 student and program, for tests.

The 2026-27 estimate (``utility.2627.from_2526_outside``, net of each
student's outside option) names the applicants but not every enrollee of
``applied_or_enrolled``, nor the special-education programs. The equivalence
tests need an estimate the simulator accepts, so this one takes the real rows
and columns where they exist and fills the rest deterministically: a missing
program column gets the estimate's lowest utility less one, and a missing
student the mean row. It is written once per content, so the cache key of
every market built on it stays put.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ESTIMATE_ROLE = "assignment.estimate"
SOURCE_ESTIMATE = "utility.2627.from_2526_outside"


def write_covering_estimate(scenario, programs, students, path: Path) -> Path:
    """Write ``path``: ``SOURCE_ESTIMATE`` extended to ``students`` x ``programs``."""
    from loaders.config import load_scenario

    base = load_scenario(
        {
            "scenario": scenario.id,
            "overrides": {"sources": {ESTIMATE_ROLE: SOURCE_ESTIMATE}},
        }
    )
    source = base.source(ESTIMATE_ROLE).path
    frame = pd.read_csv(source, dtype={"studentno": "string"}).set_index("studentno")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    program_ids = [str(p) for p in programs.program_df["program_id"]]
    student_ids = [str(s) for s in students.student_data.index]
    values = frame.to_numpy(dtype=float)
    floor = float(values[np.isfinite(values)].min()) - 1.0
    frame = frame.reindex(columns=program_ids)
    frame = frame.fillna(floor)
    mean = frame.mean(axis=0)
    frame = frame.reindex(index=student_ids)
    frame = frame.fillna(mean)
    frame.index.name = "studentno"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, float_format="%.12g")
    return path


def estimate_source(path: Path) -> dict:
    return {"path": str(path), "classification": "restricted", "identity": "checksum"}

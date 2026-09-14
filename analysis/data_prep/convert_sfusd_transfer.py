#!/usr/bin/env python
"""Convert an SFUSD pre-run/post-run/demographics transfer into cleaned tables.

The district ships one folder per school year holding three CSVs. This script
turns those into the student, program, and school tables the ``loaders``
package already knows how to read, so that a new school year becomes a
registry entry rather than a new code path.

Usage
-----

.. code-block:: bash

    uv run python analysis/data_prep/convert_sfusd_transfer.py \
        --transfer "/soalnas/share/data/school_choice/Data/raw_SFUSD_data_downloads/Sep 14 2026 data transfer" \
        --year 2425 --year 2526 --year 2627 \
        --gaps fill-and-report

Outputs, per year, below ``--out`` (default ``<data root>/Data/Cleaned``):

* ``student_<year>.csv``   -- every applicant, every grade, one round.
* ``enrolled_<year>.csv``  -- the kindergarten subset, matching the existing
  ``enrolled_*`` convention (verified: for 2021-22 through 2023-24 the
  checked-in ``enrolled_*`` file is exactly the KG rows of ``student_*``).
* ``programs_<year>.csv`` / ``programs_withMissionBay_<year>.csv``,
  ``programs_06_<year>.csv``, ``programs_09_<year>.csv``.
* ``sfusd_transfer_report_<year>.json`` and ``.md`` -- the gap report.

Missing data policy
-------------------

Nothing is quietly filled or coerced. ``--gaps fail`` (the default) aborts on
the first gap that would otherwise need a substituted value. ``--gaps
fill-and-report`` proceeds, records every substitution in the report, and
marks it in the data itself where a column exists for that (``capacity_source``
on program rows). Fields the transfer simply does not contain are written
blank, never zero, and are listed in the report's ``absent_columns``.

Derivations that are not a direct copy of a transfer column are listed in the
report's ``derived_fields`` so that no reader has to guess.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.data_prep.sfusd_transfer_schema import (  # noqa: E402
    BLOCK_INDEX_COLUMNS,
    BLOCK_INDEX_SOURCE_YEARS,
    DEMOGRAPHICS_OPTIONAL,
    DEMOGRAPHICS_REQUIRED,
    GRADE_BUNDLES,
    LOWELL_SCHOOL_ID,
    POSTRUN_OPTIONAL,
    POSTRUN_REQUIRED,
    PRERUN_OPTIONAL,
    PRERUN_REQUIRED,
    PRIORITY_FLAGS,
    PRIORITY_LISTS,
    PROGRAM_COLUMNS,
    RAW_SCHOOL_ID_ALIASES,
    SOTA_SCHOOL_ID,
    STUDENT_COLUMNS,
    TRANSFER_FILE_FRAGMENTS,
    TRANSFER_YEAR_FOLDERS,
    GradeBundle,
)
from loaders import load_scenario  # noqa: E402
from loaders.geography import match_points_to_census  # noqa: E402
from loaders.tables import normalize_grade  # noqa: E402

_NULL_TOKENS = frozenset({"NULL", "null", "n/a", "N/A", "", "NaN", "nan", "None"})

#: Darwin ``SF_DATALESS``. A file carrying this flag is a placeholder with no
#: local content: it has a size but zero allocated blocks, and reading it
#: blocks for minutes before failing with ``OSError: [Errno 89]``. The shared
#: data root is reached through a partial local copy on some machines, so this
#: is checked before every borrowed input rather than discovered by hanging.
_SF_DATALESS = 0x40000000


class TransferGapError(RuntimeError):
    """A gap that would require substituting a value the transfer lacks."""


def dataless(path: Path) -> bool:
    """Return True when a file exists locally as a contentless placeholder."""
    try:
        info = path.stat()
    except OSError:
        return False
    if getattr(info, "st_flags", 0) & _SF_DATALESS:
        return True
    return info.st_size > 0 and getattr(info, "st_blocks", 1) == 0


def require_readable(path: Path, label: str) -> None:
    """Fail with a precise message when a borrowed input is not really here."""
    if not path.exists():
        raise TransferGapError(f"{label} {path} does not exist.")
    if dataless(path):
        raise TransferGapError(
            f"{label} {path} exists but has no local content (it is a dataless "
            "placeholder from a partial copy of the shared data root). Fetch it "
            "onto this machine, or point --cleaned-dir at a complete copy."
        )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    """Everything the transfer did not provide, plus every derivation."""

    year: str
    transfer: str
    inputs: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    absent_columns: dict[str, list[str]] = field(default_factory=dict)
    blank_output_columns: dict[str, str] = field(default_factory=dict)
    missing_values: dict[str, dict[str, Any]] = field(default_factory=dict)
    derived_fields: dict[str, str] = field(default_factory=dict)
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def absent(self, source: str, columns: Iterable[str]) -> None:
        values = sorted(columns)
        if values:
            self.absent_columns.setdefault(source, []).extend(values)

    def blank(self, column: str, reason: str) -> None:
        self.blank_output_columns[column] = reason

    def derived(self, column: str, rule: str) -> None:
        self.derived_fields[column] = rule

    def missing(self, table: str, counts: dict[str, Any]) -> None:
        self.missing_values[table] = counts

    def substitute(self, **payload: Any) -> None:
        self.substitutions.append(payload)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def to_markdown(self) -> str:
        lines = [
            f"# SFUSD transfer conversion report -- {self.year}",
            "",
            f"Transfer: `{self.transfer}`",
            "",
            "## Inputs",
            "",
        ]
        for role, path in sorted(self.inputs.items()):
            lines.append(f"- **{role}**: `{path}`")
        lines += ["", "## Row counts", ""]
        for key, value in self.row_counts.items():
            lines.append(f"- {key}: {value:,}")

        lines += ["", "## Columns the transfer does not contain", ""]
        if self.absent_columns:
            for source, columns in sorted(self.absent_columns.items()):
                lines.append(f"- **{source}**: {', '.join(sorted(set(columns)))}")
        else:
            lines.append("- none")

        lines += ["", "## Output columns written blank", ""]
        if self.blank_output_columns:
            for column, reason in sorted(self.blank_output_columns.items()):
                lines.append(f"- `{column}`: {reason}")
        else:
            lines.append("- none")

        lines += ["", "## Missing values in emitted tables", ""]
        for table, counts in sorted(self.missing_values.items()):
            lines.append(f"### {table}")
            lines.append("")
            for column, payload in counts.items():
                lines.append(f"- `{column}`: {payload}")
            lines.append("")

        lines += ["## Substituted values", ""]
        if self.substitutions:
            for item in self.substitutions:
                lines.append(f"- {item}")
        else:
            lines.append("- none")

        lines += ["", "## Derived fields", ""]
        for column, rule in sorted(self.derived_fields.items()):
            lines.append(f"- `{column}`: {rule}")

        lines += ["", "## Notes", ""]
        for note in self.notes:
            lines.append(f"- {note}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _clean_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the transfer's textual null tokens with real missing values."""
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object or isinstance(
            result[column].dtype, pd.StringDtype
        ):
            result[column] = result[column].replace(list(_NULL_TOKENS), np.nan)
    return result


def _student_identity(values: pd.Series, source: str) -> pd.Series:
    """Convert ``S0100…`` identities to the repository's integer studentno."""
    text = values.astype("string").str.strip()
    if text.isna().any():
        raise TransferGapError(f"{source} contains blank scrambledstudentno values.")
    matches = text.str.fullmatch(r"S\d+")
    if not matches.fillna(False).all():
        bad = sorted(set(text.loc[~matches.fillna(False)].tolist()))[:10]
        raise TransferGapError(
            f"{source} has scrambledstudentno values outside the expected "
            f"'S<digits>' form: {bad}."
        )
    return text.str.slice(1).astype("int64")


def _school_ids(values: pd.Series, source: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any():
        bad = sorted(set(values.loc[invalid].astype(str).tolist()))[:10]
        raise TransferGapError(f"{source} has non-numeric school IDs: {bad}.")
    non_integral = numeric.notna() & (numeric % 1 != 0)
    if non_integral.any():
        bad = sorted(set(numeric.loc[non_integral].tolist()))[:10]
        raise TransferGapError(f"{source} has non-integer school IDs: {bad}.")
    return numeric.astype("Int64")


def _check_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    optional: Sequence[str],
    source: str,
    report: Report,
) -> None:
    """Fail on an absent required column; report absent optional ones."""
    missing_required = [column for column in required if column not in frame.columns]
    if missing_required:
        raise TransferGapError(
            f"{source} is missing required columns {missing_required}. The "
            "transfer cannot be converted without them."
        )
    report.absent(
        source, (column for column in optional if column not in frame.columns)
    )


def _as_list_literal(values: Iterable[Any]) -> str:
    """Render a Python list the way the existing cleaned tables store one."""
    return repr(list(values))


def _literal(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return []
    return list(ast.literal_eval(str(value)))


# --------------------------------------------------------------------------- #
# Transfer discovery
# --------------------------------------------------------------------------- #
def discover_transfer_files(transfer: Path, year: str) -> dict[str, Path]:
    """Locate the three CSVs for one school year inside a transfer folder."""
    folder_name = TRANSFER_YEAR_FOLDERS.get(year)
    if folder_name is None:
        raise TransferGapError(
            f"Year {year!r} is not a known transfer folder; known years are "
            f"{sorted(TRANSFER_YEAR_FOLDERS)}."
        )
    folder = transfer / folder_name
    if not folder.is_dir():
        raise TransferGapError(f"Transfer folder {folder} does not exist.")

    candidates = sorted(path for path in folder.glob("*.csv") if path.is_file())
    resolved: dict[str, Path] = {}
    for role, fragment in TRANSFER_FILE_FRAGMENTS.items():
        matches = [
            path
            for path in candidates
            if fragment in path.name.lower().replace(" ", "").replace("_", "")
        ]
        if not matches:
            raise TransferGapError(
                f"Transfer folder {folder} has no {role} CSV (looked for "
                f"{fragment!r} in {[path.name for path in candidates]})."
            )
        if len(matches) > 1:
            raise TransferGapError(
                f"Transfer folder {folder} has several {role} CSVs: "
                f"{[path.name for path in matches]}."
            )
        resolved[role] = matches[0]
    return resolved


# --------------------------------------------------------------------------- #
# Block-level equity indices
# --------------------------------------------------------------------------- #
def load_block_indices(cleaned_dir: Path, report: Report) -> pd.DataFrame:
    """Build the 2010-Block equity index lookup from the checked-in years.

    ``FRL Score``, ``AALPI Score``, ``HOCidx1`` and friends are stored per
    student but are properties of the student's 2010 Census Block. They were
    recomputed at some point in the district's history: 2021-22 through 2023-24
    agree with each other exactly, while 2014-15 through 2018-19 carry a
    different, older vintage for the same Blocks.

    So the lookup is not a blind union. It starts at the newest readable year
    and walks backwards, absorbing an older year only while that year agrees
    with the accumulated values on every Block they share. The first
    disagreement stops the walk and is recorded in the report, which keeps the
    lookup one self-consistent vintage and makes the cut-off visible instead of
    silently averaging two vintages together.
    """
    accumulated: pd.DataFrame | None = None
    used: list[str] = []
    unavailable: dict[str, str] = {}
    stopped_at: dict[str, Any] | None = None

    for year in reversed(BLOCK_INDEX_SOURCE_YEARS):
        path = cleaned_dir / f"student_{year}.csv"
        if not path.exists():
            unavailable[year] = f"{path} does not exist"
            continue
        if dataless(path):
            unavailable[year] = (
                f"{path} is a dataless placeholder with no local content"
            )
            continue
        header = pd.read_csv(path, nrows=0)
        available = [
            column for column in BLOCK_INDEX_COLUMNS if column in header.columns
        ]
        if "census_block" not in header.columns:
            unavailable[year] = f"{path} has no census_block column"
            continue
        if not available:
            unavailable[year] = f"{path} carries none of the Block index columns"
            continue

        frame = pd.read_csv(
            path, usecols=["census_block", *available], low_memory=False
        ).dropna(subset=["census_block"])
        frame["census_block"] = frame["census_block"].astype("int64")
        spread = frame.groupby("census_block")[available].nunique(dropna=True)
        inconsistent = spread.gt(1).any(axis=1)
        if inconsistent.any():
            raise TransferGapError(
                f"Cleaned student file {path} gives more than one value for a "
                "Block index within a single Block "
                f"({spread.index[inconsistent][:5].tolist()}); the index is not "
                "a Block property in that file and no value is substituted."
            )
        candidate = frame.groupby("census_block", sort=True)[available].first()

        if accumulated is None:
            accumulated = candidate
            used.append(year)
            continue

        shared_columns = [
            column for column in available if column in accumulated.columns
        ]
        shared_blocks = accumulated.index.intersection(candidate.index)
        conflicts = {}
        for column in shared_columns:
            left = accumulated.loc[shared_blocks, column]
            right = candidate.loc[shared_blocks, column]
            differs = ~(
                (left.isna() & right.isna())
                | np.isclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    equal_nan=True,
                )
            )
            if differs.any():
                conflicts[column] = int(differs.sum())
        if conflicts:
            stopped_at = {
                "year": year,
                "shared_blocks": int(len(shared_blocks)),
                "conflicting_blocks_per_column": conflicts,
                "reason": (
                    "this year carries a different vintage of the Block indices "
                    "than the newer years; the walk stops here rather than mix "
                    "two vintages"
                ),
                "years_not_examined": [
                    older
                    for older in reversed(BLOCK_INDEX_SOURCE_YEARS)
                    if older < year
                ],
            }
            break

        accumulated = accumulated.combine_first(candidate)
        used.append(year)

    if unavailable:
        report.missing(
            "block_index_sources_unavailable",
            dict(unavailable)
            | {
                "note": (
                    "these checked-in years could not be read, so they "
                    "contributed no Block coverage. See block_indices for the "
                    "Blocks that ended up uncovered."
                )
            },
        )
    if stopped_at is not None:
        report.missing("block_index_vintage_cutoff", stopped_at)

    if accumulated is None:
        raise TransferGapError(
            f"No readable cleaned student file under {cleaned_dir} carries the "
            f"Block equity indices {list(BLOCK_INDEX_COLUMNS)}; they cannot be "
            "reconstructed from the transfer, which has no Block-level fields. "
            f"Per-year reasons: {unavailable}."
        )

    for column in BLOCK_INDEX_COLUMNS:
        if column not in accumulated.columns:
            report.blank(
                column,
                "no readable cleaned student file carries this Block index",
            )
    report.note(
        "Block equity indices joined on the 2010 Census Block from the "
        f"checked-in cleaned student files for years {used} "
        f"({len(accumulated):,} Blocks). The transfer contains no Block-level "
        "fields of its own."
    )
    return accumulated


# --------------------------------------------------------------------------- #
# Student table
# --------------------------------------------------------------------------- #
def build_student_table(
    prerun: pd.DataFrame,
    postrun: pd.DataFrame,
    demographics: pd.DataFrame,
    *,
    report: Report,
    gaps: str,
) -> pd.DataFrame:
    """Pivot one round of requests into the repository's wide student layout."""
    requests = prerun.copy()
    requests["studentno"] = _student_identity(requests["scrambledstudentno"], "Pre-run")
    requests["school_id"] = _school_ids(requests["idSchool"], "Pre-run")
    if requests["school_id"].isna().any():
        count = int(requests["school_id"].isna().sum())
        raise TransferGapError(
            f"Pre-run has {count} request rows with no idSchool; a ranked choice "
            "without a school cannot be represented."
        )
    aliased = requests["school_id"].isin(RAW_SCHOOL_ID_ALIASES)
    if aliased.any():
        report.substitute(
            kind="school_id_alias",
            mapping={str(k): v for k, v in RAW_SCHOOL_ID_ALIASES.items()},
            rows=int(aliased.sum()),
            reason=(
                "the shared table loader keys Mission Bay policy on {909, 999}; "
                "the transfer issues Mission Bay ES a new ID"
            ),
        )
    requests["school_id"] = (
        requests["school_id"].replace(RAW_SCHOOL_ID_ALIASES).astype("Int64")
    )
    requests["grade"] = requests["Grade"].map(normalize_grade)
    blank_grade = requests["grade"].eq("")
    if blank_grade.any():
        bad = sorted(set(requests.loc[blank_grade, "Grade"].astype(str)))[:10]
        raise TransferGapError(f"Pre-run has unnormalizable Grade values: {bad}.")

    rank = pd.to_numeric(requests["Rank"], errors="coerce")
    if rank.isna().any():
        count = int(rank.isna().sum())
        raise TransferGapError(
            f"Pre-run has {count} request rows with no numeric Rank; the ranked "
            "order cannot be reconstructed."
        )
    requests["listed_rank"] = rank.astype("int64")
    requests["program_type"] = requests["ProgramCode"].astype("string").str.strip()
    if requests["program_type"].isna().any():
        count = int(requests["program_type"].isna().sum())
        raise TransferGapError(f"Pre-run has {count} request rows with no ProgramCode.")

    multi_grade = requests.groupby("studentno")["grade"].nunique()
    if (multi_grade > 1).any():
        bad = multi_grade[multi_grade > 1].index[:10].tolist()
        raise TransferGapError(
            f"Pre-run students apply for several grades, which the one-row-per-"
            f"student layout cannot hold: {bad}."
        )
    duplicate = requests.duplicated(subset=["studentno", "listed_rank"], keep=False)
    if duplicate.any():
        bad = sorted(set(requests.loc[duplicate, "studentno"].tolist()))[:10]
        raise TransferGapError(f"Pre-run has repeated ranks within one student: {bad}.")

    requests = requests.sort_values(["studentno", "listed_rank"], kind="stable")
    grouped = requests.groupby("studentno", sort=True)

    rows: dict[str, Any] = {
        "grade": grouped["grade"].first(),
        "r1_ranked_idschool": grouped["school_id"].apply(
            lambda values: _as_list_literal(int(value) for value in values)
        ),
        "r1_listed_ranks": grouped["listed_rank"].apply(
            lambda values: _as_list_literal(int(value) for value in values)
        ),
        "r1_programs": grouped["program_type"].apply(
            lambda values: _as_list_literal(str(value) for value in values)
        ),
    }
    report.derived(
        "r1_ranked_idschool",
        "pre-run requests for one student ordered by Rank; the transfer holds a "
        "single round, so no r2/r4 columns are emitted",
    )
    report.derived(
        "r1_listed_ranks",
        "pre-run Rank values as listed, which are not always contiguous when a "
        "choice was withdrawn",
    )

    if "RandomNumber" in requests.columns:
        rows["r1_randomnumber"] = grouped["RandomNumber"].apply(
            lambda values: _as_list_literal(
                float(value) for value in pd.to_numeric(values, errors="coerce")
            )
        )
    else:
        report.blank("r1_randomnumber", "pre-run has no RandomNumber column")

    if "CohortString" in requests.columns:
        rows["r1_cohortstring"] = grouped["CohortString"].apply(
            lambda values: _as_list_literal(
                "" if pd.isna(value) else str(value) for value in values
            )
        )
    else:
        report.blank("r1_cohortstring", "pre-run has no CohortString column")

    students = pd.DataFrame(rows)
    students.index.name = "studentno"

    # Per-request priority flags collected into per-student lists.
    for spec in PRIORITY_LISTS:
        if spec.raw_column not in requests.columns:
            report.blank(
                spec.output_column,
                f"pre-run has no {spec.raw_column} column",
            )
            continue
        flagged = requests.loc[requests[spec.raw_column].notna()]
        if spec.kind == "program":
            values = (
                flagged["school_id"].astype("int64").astype(str)
                + "-"
                + flagged["program_type"].astype(str)
                + "-"
                + flagged["grade"].astype(str)
            )
            collected = values.groupby(flagged["studentno"]).apply(
                lambda items: _as_list_literal(dict.fromkeys(items))
            )
        else:
            collected = flagged.groupby("studentno")["school_id"].apply(
                lambda items: _as_list_literal(dict.fromkeys(items.astype(int)))
            )
        students[spec.output_column] = collected.reindex(students.index).fillna("[]")
        report.derived(
            spec.output_column,
            f"schools whose pre-run request row carries {spec.raw_column}",
        )

    for spec in PRIORITY_FLAGS:
        if spec.raw_column not in requests.columns:
            report.blank(spec.output_column, f"pre-run has no {spec.raw_column} column")
            continue
        students[spec.output_column] = (
            requests.loc[requests[spec.raw_column].notna()]
            .groupby("studentno")
            .size()
            .reindex(students.index)
            .notna()
            .astype(int)
        )
        report.derived(
            spec.output_column,
            f"1 when any pre-run request row carries {spec.raw_column}",
        )

    if "CTIP1" in requests.columns:
        students["ctip1"] = (
            requests.loc[requests["CTIP1"].notna()]
            .groupby("studentno")
            .size()
            .reindex(students.index)
            .notna()
            .astype(int)
        )
        report.derived("ctip1", "1 when any pre-run request row carries CTIP1")
    else:
        report.blank("ctip1", "pre-run has no CTIP1 column")

    if "MSF" in requests.columns:
        # ``msf`` is a scalar school column in the repository's schema
        # (loaders.tables._SCALAR_SCHOOL_COLUMNS), and Students.msf grants
        # priority at exactly one school. A transfer may list two.
        msf = requests.loc[requests["MSF"].notna()].sort_values(
            ["studentno", "listed_rank"], kind="stable"
        )
        collapsed = msf.groupby("studentno")["school_id"].nunique()
        multiple = collapsed[collapsed > 1]
        if len(multiple):
            extra = (
                msf.loc[msf["studentno"].isin(multiple.index)]
                .groupby("studentno")["school_id"]
                .apply(lambda values: sorted({int(value) for value in values}))
                .to_dict()
            )
            if gaps == "fail":
                raise TransferGapError(
                    "Pre-run gives several MSF schools for "
                    f"{len(multiple)} students, but the repository stores a "
                    "single msf school and grants priority at exactly one: "
                    f"{dict(list(extra.items())[:10])}. Re-run with --gaps "
                    "fill-and-report to keep each student's highest-ranked MSF "
                    "school."
                )
            report.substitute(
                kind="msf_multiple_schools",
                students=int(len(multiple)),
                schools_per_student={str(key): value for key, value in extra.items()},
                rule="kept the MSF school at the student's lowest listed rank",
                reason=(
                    "msf is a scalar school column in the repository schema and "
                    "Students.msf grants priority at one school only, so the "
                    "remaining MSF schools cannot be represented"
                ),
            )
        students["msf"] = (
            msf.groupby("studentno")["school_id"].first().reindex(students.index)
        )
        report.derived(
            "msf",
            "school whose pre-run request row carries MSF, taking the lowest "
            "listed rank when the transfer lists more than one",
        )
    else:
        report.blank("msf", "pre-run has no MSF column")

    students["num_ranked"] = students["r1_ranked_idschool"].map(
        lambda value: len(_literal(value))
    )
    report.derived(
        "num_ranked",
        "length of the student's ranked list in the single available round",
    )
    for column, school_id, label in (
        ("lowell_ranked", LOWELL_SCHOOL_ID, "Lowell HS"),
        ("sota_ranked", SOTA_SCHOOL_ID, "Ruth Asawa SOTA"),
    ):
        students[column] = students["r1_ranked_idschool"].map(
            lambda value, target=school_id: int(target in _literal(value))
        )
        report.derived(
            column, f"1 when the ranked list contains school {school_id} ({label})"
        )

    # ------------------------------------------------------------------ #
    # Post-run: coordinates and assignment outcome
    # ------------------------------------------------------------------ #
    outcomes = postrun.copy()
    outcomes["studentno"] = _student_identity(
        outcomes["scrambledstudentno"], "Post-run"
    )
    duplicated = outcomes["studentno"].duplicated(keep=False)
    if duplicated.any():
        bad = sorted(set(outcomes.loc[duplicated, "studentno"].tolist()))[:10]
        raise TransferGapError(
            f"Post-run has several rows for one student: {bad}. The assignment "
            "outcome is ambiguous."
        )
    outcomes = outcomes.set_index("studentno")

    students["latitude"] = pd.to_numeric(outcomes["Latitude"], errors="coerce").reindex(
        students.index
    )
    students["longitude"] = pd.to_numeric(
        outcomes["Longitude"], errors="coerce"
    ).reindex(students.index)

    postrun_copies = {
        "idschoolattendance": "idSchoolAttendance",
        "enrolled_idschool": "idNextSchool",
        "r1_idschool": "idNextSchool",
        "r1_programcode": "NextProgramCode",
        "r1_rank": "Rank",
        "r1_distance": "Distance",
        "r1_isdesignation": "byDesignation",
        "requestprogramdesignation": "RequestProgramDesignation",
        "r1_designation_randomnumber": "studentRandomNumber",
        "previous_pathway": "CurrentProgramCode",
    }
    for output_column, raw_column in postrun_copies.items():
        if raw_column not in outcomes.columns:
            report.blank(output_column, f"post-run has no {raw_column} column")
            continue
        series = outcomes[raw_column]
        if output_column in {
            "idschoolattendance",
            "enrolled_idschool",
            "r1_idschool",
        }:
            series = _school_ids(series, f"Post-run {raw_column}")
            series = series.replace(RAW_SCHOOL_ID_ALIASES).astype("Int64")
        elif output_column in {
            "r1_rank",
            "r1_distance",
            "r1_isdesignation",
            "requestprogramdesignation",
            "r1_designation_randomnumber",
        }:
            series = pd.to_numeric(series, errors="coerce")
        students[output_column] = series.reindex(students.index)
        report.derived(output_column, f"post-run {raw_column}")

    if "enrolled_idschool" in students.columns:
        students["final_school"] = (
            students["enrolled_idschool"].fillna(0).astype("int64")
        )
        report.derived(
            "final_school",
            "post-run idNextSchool, or 0 when the student has no assignment; "
            "this matches the checked-in convention",
        )
    else:
        report.blank(
            "final_school", "post-run has no idNextSchool column to derive it from"
        )

    if "Sped_Pathway" in outcomes.columns:
        students["sped"] = (
            outcomes["Sped_Pathway"].notna().astype(int).reindex(students.index)
        )
        report.derived("sped", "1 when post-run Sped_Pathway is present")
    elif "IEP_Code" in outcomes.columns:
        students["sped"] = (
            outcomes["IEP_Code"].notna().astype(int).reindex(students.index)
        )
        report.derived("sped", "1 when post-run IEP_Code is present")
    else:
        report.blank("sped", "post-run has neither Sped_Pathway nor IEP_Code")

    # ------------------------------------------------------------------ #
    # Demographics: ethnicity, home language, ZIP
    # ------------------------------------------------------------------ #
    demo = _collapse_demographics(demographics, report)
    if "Race_Ethnicity" in demo.columns:
        ethnicity = demo["Race_Ethnicity"].astype("string")
        if "HISPANIC_INDICATOR" in demo.columns:
            hispanic = demo["HISPANIC_INDICATOR"].astype("string").str.upper().eq("Y")
            ethnicity = ethnicity.mask(hispanic & ethnicity.isna(), "Hispanic")
            report.derived(
                "resolved_ethnicity",
                "demographics Race_Ethnicity, falling back to 'Hispanic' when "
                "Race_Ethnicity is blank and HISPANIC_INDICATOR is Y; the "
                "repository maps these labels through Config.Constants."
                "ETHNICITY_DICT",
            )
        else:
            report.derived("resolved_ethnicity", "demographics Race_Ethnicity")
        students["resolved_ethnicity"] = ethnicity.reindex(students.index)
    else:
        report.blank("resolved_ethnicity", "demographics has no Race_Ethnicity column")

    if "HLS1__Language_First_Learn" in demo.columns:
        students["homelang"] = demo["HLS1__Language_First_Learn"].reindex(
            students.index
        )
        report.derived(
            "homelang",
            "demographics HLS1__Language_First_Learn, kept as the full language "
            "name. The transfer does not carry SFUSD's two-letter home-language "
            "codes used by the checked-in years; the only consumer "
            "(Students.get_qualified_programs_dict) accepts both encodings, and "
            "HLS1 is the HLS response whose distribution matches the "
            "checked-in homelang column most closely",
        )
    else:
        report.blank(
            "homelang", "demographics has no HLS1__Language_First_Learn column"
        )

    if "Home_Zip" in demo.columns:
        students["zipcode"] = pd.to_numeric(demo["Home_Zip"], errors="coerce").reindex(
            students.index
        )
        report.derived("zipcode", "demographics Home_Zip")
    else:
        report.blank("zipcode", "demographics has no Home_Zip column")

    report.blank(
        "englprof",
        "no transfer file carries an English-proficiency field. The column is "
        "emitted empty; no live pipeline reads it (only "
        "assignment/scripts/generators/generate_fake_dataset.py names it)",
    )
    report.blank(
        "Academic Score",
        "not present in the transfer and not carried by the Block index lookup "
        "for every Block; see the Block index notes",
    )

    students = students.reset_index()
    return students


def _collapse_demographics(demographics: pd.DataFrame, report: Report) -> pd.DataFrame:
    """Reduce the demographics extract to one row per student, deterministically."""
    frame = demographics.copy()
    identity = frame["scrambledstudentno"].astype("string").str.strip()
    unattributable = identity.isna() | identity.eq("")
    if unattributable.any():
        report.missing(
            "demographics_unattributable_rows",
            {
                "rows_dropped": int(unattributable.sum()),
                "reason": (
                    "these demographics rows carry no scrambledstudentno, so "
                    "they cannot be attached to any applicant. They are dropped "
                    "rather than matched by any other key"
                ),
            },
        )
        frame = frame.loc[~unattributable].copy()
    frame["studentno"] = _student_identity(frame["scrambledstudentno"], "Demographics")
    frame = frame.drop_duplicates()
    duplicated = frame["studentno"].duplicated(keep=False)
    if duplicated.any():
        # Keep the most complete record; report how many students needed it.
        frame = frame.assign(_completeness=frame.notna().sum(axis=1))
        frame = frame.sort_values(
            ["studentno", "_completeness"], ascending=[True, False], kind="stable"
        )
        collapsed = frame.drop_duplicates("studentno", keep="first").drop(
            columns="_completeness"
        )
        report.substitute(
            kind="demographics_duplicate_collapse",
            students=int(frame["studentno"].duplicated(keep=False).sum()),
            rule="kept the row with the most non-null fields",
            reason=(
                "the demographics extract has one row per enrolment record, so a "
                "student who changed school mid-year appears more than once"
            ),
        )
        frame = collapsed
    return frame.set_index("studentno")


# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
def attach_geography(
    students: pd.DataFrame, report: Report, *, scenario_name: str = "legacy"
) -> pd.DataFrame:
    """Map student coordinates to 2010 Census Block, BlockGroup, and Tract.

    The emitted tables are tagged ``geography_vintage: "2010"`` in the catalog,
    matching every checked-in student source, so a 2010 run keeps these IDs and
    a 2020 run re-derives them from the same coordinates.
    """
    scenario = load_scenario({"scenario": scenario_name, "overrides": {}})
    matched = match_points_to_census(
        students,
        scenario,
        "optimization",
        latitude_column="latitude",
        longitude_column="longitude",
    )
    result = students.copy()
    result["census_block"] = matched["Block"]
    result["census_blockgroup"] = matched["BlockGroup"]
    result["census_tract"] = matched["Tract"]
    report.derived(
        "census_block",
        "2010 Census Block containing the post-run coordinates, resolved with "
        "loaders.geography.match_points_to_census against the same Block "
        "geometry and crosswalk the optimization graph uses",
    )
    report.derived("census_blockgroup", "2010 crosswalk parent of census_block")
    report.derived("census_tract", "2010 crosswalk parent of census_block")

    no_coordinates = int((result["latitude"].isna() | result["longitude"].isna()).sum())
    outside = int(
        (
            result["census_block"].isna()
            & result["latitude"].notna()
            & result["longitude"].notna()
        ).sum()
    )
    report.missing(
        "geography",
        {
            "students_without_coordinates": no_coordinates,
            "students_with_coordinates_outside_district_blocks": outside,
            "note": (
                "students with a blank census_block are removed by the default "
                "outside_district_students: ignore selector and rejected by "
                "optimization graph construction if that selector is set to "
                "include"
            ),
        },
    )
    return result


def attach_block_indices(
    students: pd.DataFrame, block_indices: pd.DataFrame, report: Report
) -> pd.DataFrame:
    """Join the Block equity indices, reporting Blocks the lookup does not cover."""
    result = students.copy()
    available = [
        column for column in BLOCK_INDEX_COLUMNS if column in block_indices.columns
    ]
    joined = result[["census_block"]].merge(
        block_indices[available],
        how="left",
        left_on="census_block",
        right_index=True,
    )
    for column in available:
        result[column] = joined[column].to_numpy()
        report.derived(
            column,
            "2010 Census Block lookup built from the checked-in cleaned student "
            "files; constant within a Block and identical across those years",
        )

    known = result["census_block"].notna()
    uncovered = result.loc[
        known & ~result["census_block"].isin(block_indices.index), "census_block"
    ]
    report.missing(
        "block_indices",
        {
            "students_in_blocks_absent_from_the_lookup": int(len(uncovered)),
            "distinct_blocks_absent_from_the_lookup": sorted(
                int(value) for value in uncovered.unique()
            )[:50],
            "per_column_missing": {
                column: int(result[column].isna().sum()) for column in available
            },
            "note": (
                "these students keep blank equity indices. Optimization fills "
                "FRL and AALPI Score with the column mean in "
                "optimization.data.loaders._load_students_for_year, so a run "
                "over this year is not free of that imputation -- it is the "
                "same behaviour the checked-in years already have"
            ),
        },
    )
    return result


# --------------------------------------------------------------------------- #
# Program tables
# --------------------------------------------------------------------------- #
def build_program_table(
    prerun: pd.DataFrame,
    postrun: pd.DataFrame,
    bundle: GradeBundle,
    *,
    cleaned_dir: Path,
    include_mission_bay: bool,
    gaps: str,
    report: Report,
) -> pd.DataFrame | None:
    """Derive one grade's program table from observed offers and old capacities.

    The transfer contains no capacity file. Program *existence* is observable
    (a program a student could rank existed), so the row set is derived from the
    requests; the capacity is taken from the most recent district capacity
    table that has the program, and the ``capacity_source`` column records
    which. A program with no district capacity is either an error
    (``--gaps fail``) or falls back to the count assigned to it in this year's
    post-run, which is a lower bound on its true capacity.
    """
    grade = bundle.grade
    requests = prerun.copy()
    requests["school_id"] = (
        _school_ids(requests["idSchool"], "Pre-run")
        .replace(RAW_SCHOOL_ID_ALIASES)
        .astype("Int64")
    )
    requests["grade"] = requests["Grade"].map(normalize_grade)
    requests["program_type"] = requests["ProgramCode"].astype("string").str.strip()
    selected = requests.loc[requests["grade"] == grade].copy()
    if selected.empty:
        report.note(
            f"grade {grade}: the transfer has no requests for this grade, so no "
            "program table is emitted"
        )
        return None

    if not include_mission_bay:
        selected = selected.loc[~selected["school_id"].isin([909, 999])]

    offered = (
        selected.groupby(["school_id", "program_type"], as_index=False)
        .size()
        .rename(columns={"size": "_requests"})
    )
    offered["program_id"] = (
        offered["school_id"].astype("int64").astype(str)
        + "-"
        + offered["program_type"].astype(str)
        + "-"
        + grade
    )

    # First-choice and assigned counts, which grade 6 needs for K-8 capacities.
    first_choice = (
        selected.loc[pd.to_numeric(selected["Rank"], errors="coerce") == 1]
        .groupby(["school_id", "program_type"])
        .size()
    )
    assigned = _assigned_counts(postrun, grade)

    offered["r1_first_choice"] = (
        offered.set_index(["school_id", "program_type"]).index.map(first_choice).values
    )
    offered["r1_assigned"] = (
        offered.set_index(["school_id", "program_type"]).index.map(assigned).values
    )
    report.derived(
        f"programs[{grade}].r1_first_choice",
        "pre-run requests at Rank 1 for the program",
    )
    report.derived(
        f"programs[{grade}].r1_assigned",
        "post-run students whose assignment is this program",
    )

    reference_path = cleaned_dir / bundle.capacity_reference
    require_readable(
        reference_path,
        f"grade {grade}: capacity reference (the transfer carries no capacity "
        "of its own)",
    )
    reference = pd.read_csv(reference_path)
    missing_reference = {"program_id", "capacity"} - set(reference.columns)
    if missing_reference:
        raise TransferGapError(
            f"Capacity reference {reference_path} is missing "
            f"{sorted(missing_reference)}."
        )
    reference_capacity = (
        pd.to_numeric(reference["capacity"], errors="coerce")
        .groupby(reference["program_id"].astype(str))
        .first()
    )

    offered["capacity"] = offered["program_id"].map(reference_capacity)
    offered["capacity_source"] = np.where(
        offered["capacity"].notna(), bundle.capacity_reference, pd.NA
    )

    gap = offered["capacity"].isna()
    if gap.any():
        rows = offered.loc[gap, ["program_id", "r1_assigned"]].to_dict("records")
        if gaps == "fail":
            raise TransferGapError(
                f"grade {grade}: no district capacity exists for "
                f"{[row['program_id'] for row in rows]}. The transfer has no "
                "capacity file. Re-run with --gaps fill-and-report to substitute "
                "each program's observed post-run assignment count, which is a "
                "lower bound, or supply a capacity table."
            )
        fallback = offered.loc[gap, "r1_assigned"]
        never_assigned = fallback.isna()
        offered.loc[gap, "capacity"] = fallback.fillna(0.0).to_numpy()
        offered.loc[gap, "capacity_source"] = np.where(
            never_assigned.to_numpy(),
            "NONE -- no district capacity and never assigned; written as 0",
            "observed post-run assignment count (lower bound)",
        )
        report.substitute(
            kind="program_capacity",
            grade=grade,
            include_mission_bay=include_mission_bay,
            programs=rows,
            rule=(
                "capacity set to the program's post-run assignment count, or 0 "
                "where the program was never assigned"
            ),
            reason=f"{bundle.capacity_reference} has no row for these programs",
        )

    offered = offered.sort_values("program_id", kind="stable").reset_index(drop=True)
    offered["programno"] = np.arange(1, len(offered) + 1)
    for column in bundle.extra_columns:
        if column == "r2_capacity":
            offered[column] = offered["capacity"]
            report.derived(
                f"programs[{grade}].r2_capacity",
                "copied from capacity; Programs.fix_k8_capacities overwrites it "
                "for K-8 schools at grade 6",
            )
        else:
            offered[column] = pd.NA

    columns = [*PROGRAM_COLUMNS, *bundle.extra_columns]
    return offered[columns]


def _assigned_counts(postrun: pd.DataFrame, grade: str) -> pd.Series:
    """Count post-run assignments per (school, program) for one grade."""
    if not {"idNextSchool", "NextProgramCode"} <= set(postrun.columns):
        return pd.Series(dtype="int64")
    frame = postrun.copy()
    frame["school_id"] = (
        _school_ids(frame["idNextSchool"], "Post-run idNextSchool")
        .replace(RAW_SCHOOL_ID_ALIASES)
        .astype("Int64")
    )
    frame["program_type"] = frame["NextProgramCode"].astype("string").str.strip()
    if "NextGrade" in frame.columns:
        frame["grade"] = frame["NextGrade"].map(normalize_grade)
        frame = frame.loc[frame["grade"] == grade]
    frame = frame.dropna(subset=["school_id", "program_type"])
    return frame.groupby(["school_id", "program_type"]).size()


def check_school_coverage(
    programs: pd.DataFrame,
    bundle: GradeBundle,
    *,
    cleaned_dir: Path,
    include_mission_bay: bool,
    gaps: str,
    report: Report,
) -> pd.DataFrame:
    """Verify every emitted program has a locatable school, or drop and report.

    Assignment builds student-program distances from school coordinates and
    rejects a non-finite distance, so a program at a school the school table
    does not contain would fail the run rather than degrade it.
    """
    name = bundle.schools_mission_bay if include_mission_bay else None
    name = name or bundle.schools_standard
    path = cleaned_dir / name
    require_readable(
        path,
        f"grade {bundle.grade}: school table (program locations cannot be "
        "verified without it)",
    )
    schools = pd.read_csv(path)
    required = {"school_id", "lat", "lon"}
    missing = required - set(schools.columns)
    if missing:
        raise TransferGapError(f"School table {path} is missing {sorted(missing)}.")
    locatable = set(
        schools.loc[schools["lat"].notna() & schools["lon"].notna(), "school_id"]
        .astype(int)
        .tolist()
    )
    unlocatable = sorted(set(programs["school_id"].astype(int)) - locatable)
    if not unlocatable:
        return programs

    affected = programs.loc[programs["school_id"].astype(int).isin(unlocatable)]
    if gaps == "fail":
        raise TransferGapError(
            f"grade {bundle.grade}: schools {unlocatable} appear in the transfer "
            f"but have no coordinates in {name}, so the affected programs "
            f"{affected['program_id'].tolist()} cannot be located. The transfer "
            "carries no school table. Re-run with --gaps fill-and-report to drop "
            "them, or add the schools to the school table."
        )
    report.substitute(
        kind="unlocatable_school",
        grade=bundle.grade,
        include_mission_bay=include_mission_bay,
        schools=unlocatable,
        dropped_programs=affected["program_id"].tolist(),
        requests_affected=None,
        rule="program rows dropped from the emitted program table",
        reason=(
            f"{name} has no coordinates for these schools, and assignment "
            "rejects a non-finite student-program distance"
        ),
    )
    kept = programs.loc[~programs["school_id"].astype(int).isin(unlocatable)].copy()
    kept = kept.sort_values("program_id", kind="stable").reset_index(drop=True)
    kept["programno"] = np.arange(1, len(kept) + 1)
    return kept


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def convert_year(
    transfer: Path,
    year: str,
    *,
    out_dir: Path,
    cleaned_dir: Path,
    gaps: str,
    scenario_name: str,
    dry_run: bool,
) -> Report:
    """Convert one school year and write its outputs and gap report."""
    paths = discover_transfer_files(transfer, year)
    report = Report(year=year, transfer=str(transfer))
    report.inputs = {role: str(path) for role, path in paths.items()}

    for role, path in paths.items():
        require_readable(path, f"transfer {role} file")
    prerun = _clean_nulls(pd.read_csv(paths["prerun"], low_memory=False))
    postrun = _clean_nulls(pd.read_csv(paths["postrun"], low_memory=False))
    demographics = _clean_nulls(pd.read_csv(paths["demographics"], low_memory=False))
    report.row_counts = {
        "prerun_rows": len(prerun),
        "postrun_rows": len(postrun),
        "demographics_rows": len(demographics),
    }

    _check_columns(prerun, PRERUN_REQUIRED, PRERUN_OPTIONAL, "Pre-run", report)
    _check_columns(postrun, POSTRUN_REQUIRED, POSTRUN_OPTIONAL, "Post-run", report)
    _check_columns(
        demographics,
        DEMOGRAPHICS_REQUIRED,
        DEMOGRAPHICS_OPTIONAL,
        "Demographics",
        report,
    )

    block_indices = load_block_indices(cleaned_dir, report)
    students = build_student_table(
        prerun, postrun, demographics, report=report, gaps=gaps
    )
    students = attach_geography(students, report, scenario_name=scenario_name)
    students = attach_block_indices(students, block_indices, report)

    for column in STUDENT_COLUMNS:
        if column not in students.columns:
            students[column] = pd.NA
    students = students[list(STUDENT_COLUMNS)]

    report.row_counts["students"] = len(students)
    report.missing(
        "student_table",
        {
            column: int(students[column].isna().sum())
            for column in STUDENT_COLUMNS
            if students[column].isna().any()
        },
    )

    kindergarten = students.loc[students["grade"] == "KG"].reset_index(drop=True)
    report.row_counts["kindergarten_students"] = len(kindergarten)
    report.note(
        "enrolled_<year>.csv holds the kindergarten rows of student_<year>.csv. "
        "That reproduces the checked-in convention: for 2021-22 through 2023-24 "
        "the enrolled_* file is exactly the KG subset of student_*, not a "
        "different population."
    )

    outputs: dict[str, pd.DataFrame] = {
        f"student_{year}.csv": students,
        f"enrolled_{year}.csv": kindergarten,
    }

    for grade, bundle in GRADE_BUNDLES.items():
        for include_mission_bay, template in (
            (False, bundle.programs_template),
            (True, bundle.programs_mission_bay_template),
        ):
            if template is None:
                continue
            programs = build_program_table(
                prerun,
                postrun,
                bundle,
                cleaned_dir=cleaned_dir,
                include_mission_bay=include_mission_bay,
                gaps=gaps,
                report=report,
            )
            if programs is None:
                continue
            programs = check_school_coverage(
                programs,
                bundle,
                cleaned_dir=cleaned_dir,
                include_mission_bay=include_mission_bay,
                gaps=gaps,
                report=report,
            )
            outputs[template.format(year=year)] = programs
            report.row_counts[f"programs_{grade}_mb{int(include_mission_bay)}"] = len(
                programs
            )

    report.note(
        "No school table is emitted. The transfer carries no school "
        "coordinates, category, or rating, so the registry reuses the "
        "checked-in 2023-24 school tables; school attributes are therefore "
        "2023-24 vintage while students and programs are this year's."
    )
    report.note(
        "Each transfer holds exactly one round of requests: the pre-run has no "
        "round column, no student has a repeated rank, and the demographics "
        "'rounds_applied' field is blank for all but a few hundred students. "
        "Only r1_* preference columns are emitted, so `rounds: all` resolves to "
        "[1] for these years."
    )

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in outputs.items():
            frame.to_csv(out_dir / name, index=False)
        (out_dir / f"sfusd_transfer_report_{year}.json").write_text(
            json.dumps(report.__dict__, indent=2, default=str) + "\n"
        )
        (out_dir / f"sfusd_transfer_report_{year}.md").write_text(report.to_markdown())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--transfer",
        required=True,
        type=Path,
        help="Transfer folder containing one subfolder per school year.",
    )
    parser.add_argument(
        "--year",
        action="append",
        default=None,
        help="Canonical school year, repeatable. Defaults to every known year.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to the cleaned-data directory.",
    )
    parser.add_argument(
        "--cleaned-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the checked-in cleaned tables the conversion "
            "borrows Block indices, capacities, and school locations from. "
            "Defaults to <data root>/Data/Cleaned."
        ),
    )
    parser.add_argument(
        "--gaps",
        choices=("fail", "fill-and-report"),
        default="fail",
        help=(
            "What to do when a value the transfer lacks would have to be "
            "substituted. 'fail' (default) aborts and names the gap."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="legacy",
        help="Loader scenario supplying the 2010 Census geometry and crosswalk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report gaps without writing any file.",
    )
    return parser


def _default_cleaned_dir() -> Path:
    scenario = load_scenario({"scenario": "legacy", "overrides": {}})
    return Path(scenario.roots["data"]) / "Data" / "Cleaned"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cleaned_dir = args.cleaned_dir or _default_cleaned_dir()
    out_dir = args.out or cleaned_dir
    years = args.year or list(TRANSFER_YEAR_FOLDERS)

    failures: list[str] = []
    for year in years:
        try:
            report = convert_year(
                args.transfer,
                year,
                out_dir=out_dir,
                cleaned_dir=cleaned_dir,
                gaps=args.gaps,
                scenario_name=args.scenario,
                dry_run=args.dry_run,
            )
        except TransferGapError as exc:
            failures.append(f"{year}: {exc}")
            print(f"[{year}] GAP: {exc}", file=sys.stderr)
            continue
        print(report.to_markdown())

    if failures:
        print(
            "\n".join(["", "Conversion stopped for these years:", *failures]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

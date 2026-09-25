#!/usr/bin/env python
"""Convert an SFUSD pre-run/post-run/demographics transfer into cleaned tables.

The district ships one folder per school year holding three CSVs, plus a
shared folder of auxiliary files: the Main Round capacities and the TK-to-K
auto-promotion lists. This script turns those into the student and program
tables the ``loaders`` package already knows how to read, so that a new school
year becomes a registry entry rather than a new code path.

Usage
-----

.. code-block:: bash

    uv run python analysis/data_prep/convert_sfusd_transfer.py \
        --transfer "/soalnas/share/data/school_choice/Data/raw_SFUSD_data_downloads/Sep 14 2026 data transfer" \
        --year 2425 --year 2526 --year 2627 \
        --gaps fill-and-report

Outputs, per year, below ``--out`` (default ``<data root>/Data/Cleaned``):

* ``student_<year>.csv``   -- every applicant, every grade, one
  preference list each.
* ``enrolled_<year>.csv``  -- the post-run's seated kindergarten cohort: the
  main-round applicants the run placed, plus the students it seated without a
  request (auto-promoted TK students), marked ``mr_applicant = 0``. Through
  2023-24 the checked-in ``enrolled_*`` file was exactly the KG rows of
  ``student_*`` because everyone who enrolled had applied; auto-promotion ends
  that from 2024-25.
* ``programs_<year>.csv`` / ``programs_withMissionBay_<year>.csv``,
  ``programs_06_<year>.csv``, ``programs_09_<year>.csv``.
* ``sfusd_transfer_report_<year>.json`` and ``.md`` -- the gap report.

Auto-promotion
--------------

From 2024-25 a kindergarten cohort is not a kindergarten applicant pool: SFUSD
promotes TK students into K without an application. The kindergarten rows of
both student tables therefore carry ``promote_eligible``, ``feeder_school``,
``feeder_program`` and ``pref_source``, and a promoted student's preference
list is built rather than transcribed. ``tk_promotion.py`` holds the rules and
the reasoning; nothing about it reserves a seat, because the run releases a
promote's held seat back into the same market when they win elsewhere.

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
    AUXILIARY_DIRECTORY,
    EES_FEEDER_FIRST_YEAR,
    BLOCK_INDEX_COLUMNS,
    BLOCK_INDEX_SOURCE_YEARS,
    DEMOGRAPHICS_OPTIONAL,
    DEMOGRAPHICS_REQUIRED,
    GRADE_BUNDLES,
    LOWELL_SCHOOL_ID,
    MARKET_STUDENT_COLUMNS,
    POSTRUN_OPTIONAL,
    POSTRUN_REQUIRED,
    PRERUN_OPTIONAL,
    PREF_SOURCES,
    PRERUN_REQUIRED,
    PRIORITY_FLAGS,
    PRIORITY_LISTS,
    PROGRAM_COLUMNS,
    PROMOTION_STUDENT_COLUMNS,
    RAW_SCHOOL_ID_ALIASES,
    SOTA_SCHOOL_ID,
    STUDENT_COLUMNS,
    TRANSFER_FILE_FRAGMENTS,
    TRANSFER_YEAR_FOLDERS,
    GradeBundle,
)
from analysis.data_prep.tk_promotion import (  # noqa: E402
    PreferenceInputs,
    PreferenceResult,
    PromotionMap,
    build_market_preferences,
    build_promotion_map,
    compare_promote_counts,
    discover_auxiliary_files,
    load_mr_capacities,
    prior_tk_source,
    program_keys,
    promotion_entitlements,
    tk_lists_from_cleaned_table,
    tk_lists_from_prerun,
)
from loaders import load_scenario  # noqa: E402
from loaders.geography import match_points_to_census  # noqa: E402
from loaders.tables import NOT_ENROLLED_SCHOOL_ID, normalize_grade  # noqa: E402

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
    outcomes = _postrun_outcomes(postrun)
    _attach_postrun_columns(students, outcomes, report)

    # ------------------------------------------------------------------ #
    # Demographics: ethnicity, home language, ZIP
    # ------------------------------------------------------------------ #
    demo = _collapse_demographics(demographics, report)
    _attach_demographics_columns(students, demo, report)
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


def _postrun_outcomes(postrun: pd.DataFrame) -> pd.DataFrame:
    """Index the post-run by student, rejecting an ambiguous outcome."""
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
    return outcomes.set_index("studentno")


def _attach_postrun_columns(
    students: pd.DataFrame, outcomes: pd.DataFrame, report: Report
) -> None:
    """Copy coordinates and the assignment outcome onto an indexed student frame.

    ``students`` is indexed by ``studentno`` and mutated in place. The rules are
    identical for the applicant table and the enrolled table, and the report's
    per-column entries are keyed by column, so calling this twice records each
    derivation once.
    """
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


def _attach_demographics_columns(
    students: pd.DataFrame, demo: pd.DataFrame, report: Report
) -> None:
    """Copy ethnicity, home language, and ZIP onto an indexed student frame."""
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


def _collapse_demographics(
    demographics: pd.DataFrame, report: Report, *, record: bool = True
) -> pd.DataFrame:
    """Reduce the demographics extract to one row per student, deterministically.

    ``record`` is false for the second pass over the same extract (the enrolled
    table), whose substitutions would otherwise be appended to the report twice.
    """
    frame = demographics.copy()
    identity = frame["scrambledstudentno"].astype("string").str.strip()
    unattributable = identity.isna() | identity.eq("")
    if unattributable.any():
        if record:
            report.missing(
                "demographics_unattributable_rows",
                {
                    "rows_dropped": int(unattributable.sum()),
                    "reason": (
                        "these demographics rows carry no scrambledstudentno, "
                        "so they cannot be attached to any applicant. They are "
                        "dropped rather than matched by any other key"
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
        if record:
            report.substitute(
                kind="demographics_duplicate_collapse",
                students=int(frame["studentno"].duplicated(keep=False).sum()),
                rule="kept the row with the most non-null fields",
                reason=(
                    "the demographics extract has one row per enrolment record, "
                    "so a student who changed school mid-year appears more than "
                    "once"
                ),
            )
        frame = collapsed
    return frame.set_index("studentno")


# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
def attach_geography(
    students: pd.DataFrame,
    report: Report,
    *,
    scenario_name: str = "legacy",
    report_label: str = "geography",
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
        report_label,
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
    students: pd.DataFrame,
    block_indices: pd.DataFrame,
    report: Report,
    *,
    report_label: str = "block_indices",
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
        report_label,
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
# Seats the run fills without a request
# --------------------------------------------------------------------------- #
def _requested_at_grade(prerun: pd.DataFrame, grade: str) -> set[int]:
    """Students holding at least one pre-run request for one grade."""
    identity = _student_identity(prerun["scrambledstudentno"], "Pre-run")
    grades = prerun["Grade"].map(normalize_grade)
    return set(identity.loc[grades == grade].dropna())


def seated_at_grade(outcomes: pd.DataFrame, grade: str) -> pd.DataFrame | None:
    """Post-run rows the run seats at one grade, indexed by student.

    ``None`` when the post-run has no ``NextGrade`` column, which no transfer
    year so far is missing but the SY26-27 truncation shows is possible.
    """
    if "NextGrade" not in outcomes.columns:
        return None
    grades = outcomes["NextGrade"].map(normalize_grade)
    return outcomes.loc[grades == grade]


def seated_without_request(
    prerun: pd.DataFrame, outcomes: pd.DataFrame, grade: str
) -> pd.DataFrame | None:
    """Students the run seats at one grade who filed no request for that grade.

    From 2024-25 SFUSD auto-promotes TK students into kindergarten as the first
    step of the main run: they take a seat before any applicant is placed, and
    they are absent from the pre-run because they filed no request (847 of them
    in SY26-27, 580 in SY25-26, 40 in SY24-25). The same shape appears at grades
    6 and 9 in SY24-25.

    Identified as a post-run seat at the grade with no pre-run request for it,
    which deliberately does not use ``byPromote``: SY24-25 carries that column
    but leaves it 0 for every student, and in the later years it is also set for
    applicants who were promoted after an unsuccessful application -- students
    who *are* in the applicant pool, and whose seat must not be counted twice.
    """
    seated = seated_at_grade(outcomes, grade)
    if seated is None:
        return None
    requested = _requested_at_grade(prerun, grade)
    return seated.loc[~seated.index.isin(requested)]


# --------------------------------------------------------------------------- #
# The market at one grade
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Market:
    """Who is in one grade's market, and which program each of them may claim.

    Decided once, at the top of the conversion, because three later steps have
    to agree about it: the student table gains a row for every member, the
    preference construction gives every member a list, and the enrolled table
    is the subset the run seated. Recomputing any part of it separately is how
    those three drift apart.
    """

    grade: str
    #: Every student in the market, sorted.
    population: list[int]
    #: The students who filed a Main Round request at the grade.
    applicants: set[int]
    #: Member -> the program they may claim. Restricted to programs a school
    #: table can place, because a claim that cannot enter a preference list is
    #: not usable; ``entitlements`` holds the unrestricted identification.
    feeders: dict[int, tuple[int, str]]
    #: Every placeable program at the grade.
    programs: set[tuple[int, str]]
    #: The identification before that restriction, indexed by ``studentno``
    #: with ``feeder_school`` and ``feeder_program``. This is the set the
    #: district's own promote counts are measured over, so the report compares
    #: against it rather than against ``feeders``.
    entitlements: pd.DataFrame


def define_market(
    students: pd.DataFrame,
    prerun: pd.DataFrame,
    postrun: pd.DataFrame,
    *,
    grade: str,
    capacities: pd.DataFrame,
    cleaned_dir: Path,
    report: Report,
) -> Market:
    """Decide one grade's market: its members and their promotion claims.

    Three groups, and the union of them is the applicant pool that
    ``student_<year>.csv`` holds at this grade:

    * the Main Round applicants, straight from the pre-run;
    * the promotion-eligible students, who hold a claim on a program whether
      or not they applied and whether or not they took a seat;
    * the students the post-run seats at the grade without a request. Most of
      those are promotion-eligible, but not all: 8 of 2024-25's 40, 4 of
      2025-26's 580 and 10 of 2026-27's 847 sit in a program that is not a
      kindergarten program, an Early Education School in the main. They took a
      kindergarten seat all the same, so the market has to hold them.

    A promotion-eligible student who took no seat anywhere belongs here too --
    they had a claim the run did not convert -- and none exists in the three
    transfer years, which is worth recording rather than assuming.
    """
    outcomes = _postrun_outcomes(postrun)
    indexed = students.set_index("studentno")
    applicants = set(indexed.index[indexed["grade"] == grade])

    entitlements = promotion_entitlements(outcomes, program_keys(capacities, grade))
    eligible = set(entitlements.index)
    programs = _placeable_programs(capacities, cleaned_dir=cleaned_dir, grade=grade)
    feeders = {
        int(studentno): (int(school), str(program))
        for studentno, school, program in zip(
            entitlements.index,
            entitlements["feeder_school"],
            entitlements["feeder_program"],
            strict=True,
        )
        if (int(school), str(program)) in programs
    }
    unplaceable = len(entitlements) - len(feeders)
    if unplaceable:
        report.missing(
            f"promotion_feeders_at_unplaceable_schools_{grade}",
            {
                "students": unplaceable,
                "reason": (
                    "their feeder program is at a school no school table can "
                    "place, so it cannot enter a preference list. They are not "
                    "marked promote_eligible, though the district's promote "
                    "counts still include them"
                ),
            },
        )

    seated = seated_without_request(prerun, outcomes, grade)
    without_request = set() if seated is None else set(seated.index)
    seated_at = seated_at_grade(outcomes, grade)
    unseated = eligible - (set() if seated_at is None else set(seated_at.index))
    if unseated:
        report.missing(
            f"promotion_eligible_who_took_no_{grade}_seat",
            {
                "students": len(unseated),
                "reason": (
                    "identified as promotion-eligible but not seated at this "
                    "grade by the run. They are in the market, because the "
                    "claim was theirs whether or not it was converted"
                ),
            },
        )

    report.row_counts[f"market_{grade}_main_round_applicants"] = len(applicants)
    report.row_counts[f"market_{grade}_promotion_eligible_non_applicants"] = len(
        eligible - applicants
    )
    report.row_counts[f"market_{grade}_seated_without_a_request"] = len(without_request)
    report.row_counts[f"market_{grade}_seated_without_a_claim"] = len(
        without_request - eligible
    )
    return Market(
        grade=grade,
        population=sorted(applicants | eligible | without_request),
        applicants=applicants,
        feeders=feeders,
        programs=programs,
        entitlements=entitlements,
    )


def _placeable_programs(
    capacities: pd.DataFrame, *, cleaned_dir: Path, grade: str
) -> set[tuple[int, str]]:
    """Every program at one grade a student may be given this year.

    The capacity file's programs, less the ones at a school no borrowed school
    table can place: assignment builds student-program distances from school
    coordinates and rejects a non-finite one, so putting an unplaceable
    program into a preference list would fail a run rather than degrade it.
    The test is the union of both school tables, matching
    ``drop_unlocatable_requests`` -- Mission Bay is placeable in one of them,
    and which variant a run selects is the loader's business, not this one's.
    """
    bundle = GRADE_BUNDLES[grade]
    locatable: set[int] = set()
    for include_mission_bay in (False, True):
        if include_mission_bay and bundle.schools_mission_bay is None:
            continue
        locatable |= locatable_schools(
            bundle, cleaned_dir=cleaned_dir, include_mission_bay=include_mission_bay
        )
    return {
        program
        for program in program_keys(capacities, grade)
        if program[0] in locatable
    }


def add_market_students(
    students: pd.DataFrame,
    postrun: pd.DataFrame,
    demographics: pd.DataFrame,
    block_indices: pd.DataFrame,
    *,
    market: Market,
    report: Report,
    scenario_name: str,
) -> pd.DataFrame:
    """Add the market's non-applicants to the student table as grade rows.

    ``student_<year>.csv`` is the applicant pool, and from 2024-25 the
    kindergarten applicant pool is not the set of people who applied: a
    promoted TK student holds a claim on a seat without filing anything. They
    belong in the table, so they are added here with their outcome,
    demographics, geography and Block indices resolved the same way an
    applicant's are, and with ``mr_applicant = 0``.

    Their preference list is written later, by
    :func:`build_market_table`. The empty lists set here are
    placeholders that every one of them overwrites.
    """
    grade = market.grade
    outcomes = _postrun_outcomes(postrun)
    missing = pd.Index(sorted(set(market.population) - set(students["studentno"])))
    students = students.copy()
    students["mr_applicant"] = 1
    if missing.empty:
        report.row_counts[f"market_{grade}_students_added_to_the_table"] = 0
        return students

    extra = pd.DataFrame(index=pd.Index(missing, name="studentno"))
    extra["grade"] = grade
    for column in (
        "r1_ranked_idschool",
        "r1_listed_ranks",
        "r1_programs",
        "r1_randomnumber",
        "r1_cohortstring",
        "sibling",
        "currentlpsibling",
        "currentlp",
        "aaprek",
        "prek",
        "aa",
    ):
        extra[column] = "[]"
    for column in (
        "num_ranked",
        "lowell_ranked",
        "sota_ranked",
        "bayview_to_all_ms",
        "bayview_to_brown_ms",
        "brown_ms_to_hs",
    ):
        extra[column] = 0
    extra["ctip1"] = pd.NA
    extra["msf"] = pd.NA
    report.blank(
        "ctip1 (promoted students)",
        "CTIP1 is a per-request pre-run flag, so a student who filed no "
        "request has none. The post-run's own CTIP1 column is empty for every "
        "one of them (checked: 0 of 580 in SY25-26), so it cannot stand in",
    )

    _attach_postrun_columns(extra, outcomes, report)
    _attach_demographics_columns(
        extra, _collapse_demographics(demographics, report, record=False), report
    )
    extra = extra.reset_index()
    extra = attach_geography(
        extra, report, scenario_name=scenario_name, report_label="geography_promoted"
    )
    extra = attach_block_indices(
        extra, block_indices, report, report_label="block_indices_promoted"
    )
    extra["mr_applicant"] = 0
    for column in students.columns:
        if column not in extra.columns:
            extra[column] = pd.NA

    combined = pd.concat([students, extra[list(students.columns)]], ignore_index=True)
    combined = combined.sort_values("studentno", kind="stable").reset_index(drop=True)
    report.row_counts[f"market_{grade}_students_added_to_the_table"] = len(extra)
    report.derived(
        "mr_applicant",
        "1 when the student filed a Main Round request, 0 when they are in "
        "the market without one -- an auto-promoted TK student, in the main",
    )
    report.note(
        f"student_<year>.csv holds {len(extra):,} grade-{grade} students who "
        "filed no Main Round request. From 2024-25 the kindergarten applicant "
        "pool is the Main Round applicants plus the students SFUSD promotes "
        "into kindergarten from TK, who hold a claim on a seat without "
        "applying; leaving them out would make the pool smaller than the "
        "cohort the run actually seated. They carry mr_applicant = 0. No "
        "other grade is treated this way: grades 6 and 9 are out of scope, so "
        "their rows are Main Round applicants only."
    )
    return combined


#: Where an enrolled row's ``enrolled_idschool`` and ``enrolled_programcode``
#: came from, in ``enrolled_<year>.csv``'s ``enrollment_source`` column.
#: ``fall_record`` is the demographics extract's enrolment record at the
#: grade; the other two fall back to the post-run seat because there is no
#: such record -- no demographics row at all, or a record at another grade.
ENROLLMENT_SOURCES: tuple[str, ...] = (
    "fall_record",
    "postrun_no_record",
    "postrun_other_grade",
)


def _fall_enrollment(
    demographics: pd.DataFrame | None, report: Report
) -> pd.DataFrame | None:
    """Each student's enrolment record, indexed by student, or ``None``.

    The demographics extract's ``SCHOOL_CODE``, ``GRADE`` and ``ENR_PATHWAY``
    are the district's record of where the student is enrolled for the fall
    the transfer year starts (``nextEnterDate``), which is not the post-run's
    ``idNextSchool``: a student can take a different seat after the Main Round,
    or none. ``SCHOOL_CODE`` 899 ("Central Enrollment") is the district's
    placeholder for a student with no school.
    """
    if demographics is None or not {"SCHOOL_CODE", "GRADE"} <= set(
        demographics.columns
    ):
        return None
    frame = _collapse_demographics(demographics, report, record=False)
    school = _school_ids(frame["SCHOOL_CODE"], "Demographics SCHOOL_CODE")
    return pd.DataFrame(
        {
            "school": school.replace(RAW_SCHOOL_ID_ALIASES).astype("Int64"),
            "grade": frame["GRADE"].map(normalize_grade),
            "program": (
                frame["ENR_PATHWAY"]
                if "ENR_PATHWAY" in frame.columns
                else pd.Series(pd.NA, index=frame.index, dtype=object)
            ),
        },
        index=frame.index,
    )


def not_enrolled_students(
    demographics: pd.DataFrame | None, report: Report
) -> set[int]:
    """Students whose enrolment record puts them at no school (code 899)."""
    record = _fall_enrollment(demographics, report)
    if record is None:
        return set()
    return set(record.index[record["school"].eq(NOT_ENROLLED_SCHOOL_ID).fillna(False)])


def build_enrolled_table(
    students: pd.DataFrame,
    postrun: pd.DataFrame,
    demographics: pd.DataFrame | None = None,
    *,
    market: Market,
    report: Report,
) -> pd.DataFrame:
    """Select the enrolled cohort out of the student table.

    ``enrolled_<year>.csv`` is the market students the post-run seats at
    ``grade``, less those the district's enrolment record puts at school 899,
    which means enrolled nowhere. Since :func:`add_market_students` puts the
    whole market in the student table, this is a filter of it rather than a
    second construction, which is what makes the enrolled population a subset
    of the applicant one.

    The enrolled table's ``enrolled_idschool`` and ``enrolled_programcode``
    are where the student actually enrolled, from the demographics extract
    whenever it holds a record at the grade, and the post-run seat otherwise
    (``enrollment_source`` says which). The student table's
    ``enrolled_idschool`` stays the post-run seat, and ``final_school`` and
    the ``r1_*`` outcome columns stay the Main Round assignment in both, so
    the two tables disagree on ``enrolled_idschool`` for anyone who moved
    after the Main Round. Through 2023-24 the checked-in ``enrolled_*`` file
    was exactly the KG rows of ``student_*``, not-enrolled students included.
    """
    grade = market.grade
    outcomes = _postrun_outcomes(postrun)
    at_grade = students.loc[students["grade"] == grade]
    seated = seated_at_grade(outcomes, grade)
    if seated is None:
        report.note(
            f"enrolled_<year>.csv falls back to every grade-{grade} row of "
            "student_<year>.csv: the post-run has no NextGrade column, so "
            "which of them took a seat cannot be determined."
        )
        enrolled = at_grade.reset_index(drop=True)
    else:
        enrolled = at_grade.loc[at_grade["studentno"].isin(seated.index)]
        enrolled = enrolled.sort_values("studentno", kind="stable").reset_index(
            drop=True
        )
    seated_count = len(enrolled)
    enrolled = _apply_fall_enrollment(enrolled, demographics, grade, report)
    if seated is None:
        return enrolled
    unseated = len(at_grade) - seated_count

    report.row_counts[f"enrolled_{grade}_total"] = len(enrolled)
    report.row_counts[f"enrolled_{grade}_main_round_applicants"] = int(
        enrolled["mr_applicant"].sum()
    )
    report.row_counts[f"enrolled_{grade}_market_students_with_no_seat"] = unseated
    report.note(
        f"enrolled_<year>.csv is the grade-{grade} subset of "
        f"student_<year>.csv that the post-run seats and that is not recorded "
        f"at school {NOT_ENROLLED_SCHOOL_ID}: {len(enrolled):,} of "
        f"{len(at_grade):,} market students, "
        f"{int(enrolled['mr_applicant'].sum()):,} of whom filed a Main Round "
        f"request. {unseated:,} market students took no seat at the grade, and "
        f"{seated_count - len(enrolled):,} seated ones are recorded at "
        f"{NOT_ENROLLED_SCHOOL_ID}."
    )
    return enrolled


def _apply_fall_enrollment(
    enrolled: pd.DataFrame,
    demographics: pd.DataFrame | None,
    grade: str,
    report: Report,
) -> pd.DataFrame:
    """Point the enrolled rows at where each student actually enrolled.

    Drops every student recorded at school 899 and sets ``enrolled_idschool``,
    ``enrolled_programcode`` and ``enrollment_source``; see
    :func:`build_enrolled_table`.
    """
    enrolled = enrolled.copy()
    postrun_school = enrolled["enrolled_idschool"].astype("Int64")
    postrun_program = enrolled["r1_programcode"].astype(object)
    record = _fall_enrollment(demographics, report)
    if record is None:
        report.note(
            "The demographics extract has no SCHOOL_CODE/GRADE enrolment "
            "record, so enrolled_<year>.csv takes every student's enrolled "
            "school from the post-run seat and drops nobody as unenrolled."
        )
        enrolled["enrolled_programcode"] = postrun_program
        enrolled["enrollment_source"] = "postrun_no_record"
        return enrolled

    record = record.reindex(enrolled["studentno"].to_numpy())
    school = pd.Series(record["school"].to_numpy(), index=enrolled.index, dtype="Int64")
    record_grade = pd.Series(record["grade"].to_numpy(), index=enrolled.index)
    program = pd.Series(
        record["program"].to_numpy(), index=enrolled.index, dtype=object
    )
    not_enrolled = school.eq(NOT_ENROLLED_SCHOOL_ID).fillna(False).astype(bool)
    has_record = school.notna() & ~not_enrolled
    at_grade = has_record & record_grade.eq(grade)
    # A record without a pathway at the post-run school keeps the post-run
    # program; at any other school the program is unknown.
    program = program.where(
        program.notna() | school.ne(postrun_school).fillna(True),
        postrun_program,
    )

    enrolled["enrolled_idschool"] = school.where(at_grade, postrun_school)
    enrolled["enrolled_programcode"] = program.where(at_grade, postrun_program)
    enrolled["enrollment_source"] = np.select(
        [at_grade, has_record],
        ["fall_record", "postrun_other_grade"],
        default="postrun_no_record",
    )
    moved = at_grade & school.ne(postrun_school).fillna(True)
    counts = {
        f"enrolled_{grade}_seated_not_enrolled_{NOT_ENROLLED_SCHOOL_ID}": int(
            not_enrolled.sum()
        ),
        f"enrolled_{grade}_fall_record": int(at_grade.sum()),
        f"enrolled_{grade}_fall_record_moved_from_postrun_seat": int(moved.sum()),
        f"enrolled_{grade}_postrun_other_grade": int((has_record & ~at_grade).sum()),
        f"enrolled_{grade}_postrun_no_record": int(school.isna().sum()),
    }
    report.row_counts.update(counts)
    report.note(
        f"Of the students the post-run seats at grade {grade}, "
        f"{counts[f'enrolled_{grade}_seated_not_enrolled_{NOT_ENROLLED_SCHOOL_ID}']:,} "
        f"are recorded in the demographics extract at school "
        f"{NOT_ENROLLED_SCHOOL_ID} (Central Enrollment, i.e. enrolled nowhere) "
        "and are dropped from enrolled_<year>.csv. Of the rest, "
        f"{counts[f'enrolled_{grade}_fall_record']:,} take enrolled_idschool and "
        "enrolled_programcode from their enrolment record "
        f"({counts[f'enrolled_{grade}_fall_record_moved_from_postrun_seat']:,} "
        "at a different school from their post-run seat), and "
        f"{counts[f'enrolled_{grade}_postrun_no_record']:,} with no record and "
        f"{counts[f'enrolled_{grade}_postrun_other_grade']:,} with a record at "
        "another grade keep the post-run seat."
    )
    return enrolled.loc[~not_enrolled].reset_index(drop=True)


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
    capacities: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Build one grade's program table.

    Kindergarten reads the transfer's own Main Round capacity file: it lists
    every program the run opened, and ``TotalSeats`` is its capacity. That
    capacity is *gross*. The file also publishes ``OpenSeatsPreRun``, which is
    ``TotalSeats`` less the seats held for auto-promoted TK students, and that
    is deliberately not used: a promote who wins a school elsewhere releases
    the held seat back into the same run, so the seats never leave the market.
    413-GE had 52 open seats before the SY26-27 run and made 53 choice
    assignments. The promotion claim is a run-time priority at the feeder
    program, not a seat withheld here. A program with ``TotalSeats <= 0`` is a
    closed program and is emitted as such rather than dropped, because a TK
    student can still sit in one.

    Grades 6 and 9 have no reconciled capacity in the transfer, so they keep
    the earlier derivation: program *existence* comes from the requests (a
    program a student could rank existed), the capacity from the most recent
    district capacity table that has the program, and ``capacity_source``
    records which. A program with no district capacity is either an error
    (``--gaps fail``) or falls back to the count assigned to it in this year's
    post-run, which is a lower bound.

    ``promotes_seated`` counts the students the post-run seats in the program
    having filed no main-round request for the grade (see
    ``seated_without_request``). It is provenance only; no capacity is netted
    against it.
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
    if not include_mission_bay:
        selected = selected.loc[~selected["school_id"].isin([909, 999])]

    if bundle.capacity_grade is not None:
        offered = _programs_from_capacity_file(
            bundle,
            capacities,
            include_mission_bay=include_mission_bay,
            cleaned_dir=cleaned_dir,
            report=report,
        )
        if offered is None:
            return None
    else:
        if selected.empty:
            report.note(
                f"grade {grade}: the transfer has no requests for this grade, so "
                "no program table is emitted"
            )
            return None
        offered = _programs_from_requests(selected, grade)

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

    if bundle.capacity_grade is None:
        _borrow_capacities(
            offered, bundle, cleaned_dir=cleaned_dir, gaps=gaps, report=report
        )

    promotes = seated_without_request(prerun, _postrun_outcomes(postrun), grade)
    if promotes is None:
        offered["promotes_seated"] = pd.NA
        report.blank(
            f"programs[{grade}].promotes_seated",
            "post-run has no NextGrade column, so seats taken without a "
            "request cannot be attributed to a program",
        )
    else:
        counts = _program_counts(promotes)
        keys = offered.set_index(["school_id", "program_type"]).index
        offered["promotes_seated"] = (
            pd.Series(keys.map(counts), index=offered.index).fillna(0).astype("int64")
        )
        report.derived(
            f"programs[{grade}].promotes_seated",
            "post-run students seated in this program who filed no pre-run "
            "request for the grade -- auto-promoted TK students, in the main. "
            "Provenance only: no capacity is netted against it",
        )
        unattributed = int(len(promotes) - int(offered["promotes_seated"].sum()))
        if unattributed:
            report.missing(
                f"promotes_unattributed_{grade}_mb{int(include_mission_bay)}",
                {
                    "students": unattributed,
                    "reason": (
                        "counted at this grade with no request, but their seat "
                        "is not in a program this table holds: they took no "
                        "seat at all, no applicant ranked the program, or it "
                        "belongs to a school this variant excludes"
                    ),
                },
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

    for column in PROGRAM_COLUMNS:
        if column not in offered.columns:
            offered[column] = pd.NA
    columns = [*PROGRAM_COLUMNS, *bundle.extra_columns]
    return offered[columns]


def _programs_from_requests(selected: pd.DataFrame, grade: str) -> pd.DataFrame:
    """One row per (school, program) any student ranked at this grade."""
    offered = (
        selected.groupby(["school_id", "program_type"], as_index=False)
        .size()
        .drop(columns="size")
    )
    offered["program_id"] = (
        offered["school_id"].astype("int64").astype(str)
        + "-"
        + offered["program_type"].astype(str)
        + "-"
        + grade
    )
    return offered


def _programs_from_capacity_file(
    bundle: GradeBundle,
    capacities: pd.DataFrame | None,
    *,
    include_mission_bay: bool,
    cleaned_dir: Path,
    report: Report,
) -> pd.DataFrame | None:
    """One row per capacity-file program at this grade, at gross capacity."""
    grade = bundle.grade
    if capacities is None:
        raise TransferGapError(
            f"grade {grade}: no Main Round capacity file was supplied, but the "
            "grade takes its capacity from one. Point --transfer at a transfer "
            f"whose '{AUXILIARY_DIRECTORY}' folder holds it."
        )
    rows = capacities.loc[capacities["grade"] == grade].copy()
    if rows.empty:
        report.note(
            f"grade {grade}: the capacity file has no rows for this grade, so "
            "no program table is emitted"
        )
        return None

    if not include_mission_bay:
        rows = rows.loc[~rows["school_id"].isin([909, 999])]

    locatable = locatable_schools(
        bundle, cleaned_dir=cleaned_dir, include_mission_bay=include_mission_bay
    )
    unlocatable = sorted(set(rows["school_id"].dropna().astype(int)) - locatable)
    if unlocatable:
        dropped = rows.loc[rows["school_id"].isin(unlocatable)]
        report.substitute(
            kind="capacity_program_unlocatable_school",
            grade=grade,
            include_mission_bay=include_mission_bay,
            schools=unlocatable,
            programs=int(len(dropped)),
            seats=int(pd.to_numeric(dropped["capacity"], errors="coerce").sum()),
            rule="the affected programs are not emitted",
            reason=(
                "the borrowed school table has no coordinates for these "
                "schools, and assignment rejects a non-finite student-program "
                "distance"
            ),
        )
        rows = rows.loc[~rows["school_id"].isin(unlocatable)]

    offered = rows.reset_index(drop=True)
    offered["program_id"] = (
        offered["school_id"].astype("int64").astype(str)
        + "-"
        + offered["program_type"].astype(str)
        + "-"
        + grade
    )
    offered["capacity_source"] = (
        f"transfer Main Round capacity file, TotalSeats at grade "
        f"{bundle.capacity_grade}"
    )
    report.derived(
        f"programs[{grade}].capacity",
        "TotalSeats from the transfer's Main Round capacity file. Gross: the "
        "seats the file holds for auto-promoted TK students are not removed, "
        "because promotes who win elsewhere release them back into the same "
        "run",
    )
    closed = int((pd.to_numeric(offered["capacity"], errors="coerce") <= 0).sum())
    report.missing(
        f"closed_programs_{grade}_mb{int(include_mission_bay)}",
        {
            "programs": closed,
            "reason": (
                "the capacity file opens these programs with no seats. They are "
                "emitted rather than dropped: a program can exist, hold TK "
                "students, and offer nobody a kindergarten seat"
            ),
        },
    )
    return offered


def _borrow_capacities(
    offered: pd.DataFrame,
    bundle: GradeBundle,
    *,
    cleaned_dir: Path,
    gaps: str,
    report: Report,
) -> None:
    """Fill a grade's capacity from the most recent checked-in district table."""
    grade = bundle.grade
    reference_path = cleaned_dir / bundle.capacity_reference
    require_readable(
        reference_path,
        f"grade {grade}: capacity reference (the transfer's capacity file has "
        "not been reconciled for this grade)",
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
    if not gap.any():
        return
    rows = offered.loc[gap, ["program_id", "r1_assigned"]].to_dict("records")
    if gaps == "fail":
        raise TransferGapError(
            f"grade {grade}: no district capacity exists for "
            f"{[row['program_id'] for row in rows]}. Re-run with --gaps "
            "fill-and-report to substitute each program's observed post-run "
            "assignment count, which is a lower bound, or supply a capacity "
            "table."
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
        programs=rows,
        rule=(
            "capacity set to the program's post-run assignment count, or 0 "
            "where the program was never assigned"
        ),
        reason=f"{bundle.capacity_reference} has no row for these programs",
    )


def _program_counts(frame: pd.DataFrame) -> pd.Series:
    """Count post-run rows per (school, program), keyed like the program table."""
    if not {"idNextSchool", "NextProgramCode"} <= set(frame.columns):
        return pd.Series(dtype="int64")
    frame = frame.copy()
    frame["school_id"] = (
        _school_ids(frame["idNextSchool"], "Post-run idNextSchool")
        .replace(RAW_SCHOOL_ID_ALIASES)
        .astype("Int64")
    )
    frame["program_type"] = frame["NextProgramCode"].astype("string").str.strip()
    frame = frame.dropna(subset=["school_id", "program_type"])
    return frame.groupby(["school_id", "program_type"]).size()


def _assigned_counts(postrun: pd.DataFrame, grade: str) -> pd.Series:
    """Count post-run assignments per (school, program) for one grade."""
    frame = postrun
    if "NextGrade" in frame.columns:
        frame = frame.loc[frame["NextGrade"].map(normalize_grade) == grade]
    return _program_counts(frame)


def locatable_schools(
    bundle: GradeBundle, *, cleaned_dir: Path, include_mission_bay: bool
) -> set[int]:
    """Return the school IDs one grade's school table can place on a map."""
    name = bundle.schools_mission_bay if include_mission_bay else None
    name = name or bundle.schools_standard
    path = cleaned_dir / name
    require_readable(
        path,
        f"grade {bundle.grade}: school table (program locations cannot be "
        "verified without it)",
    )
    schools = pd.read_csv(path)
    missing = {"school_id", "lat", "lon"} - set(schools.columns)
    if missing:
        raise TransferGapError(f"School table {path} is missing {sorted(missing)}.")
    return set(
        schools.loc[schools["lat"].notna() & schools["lon"].notna(), "school_id"]
        .astype(int)
        .tolist()
    )


def drop_unlocatable_requests(
    prerun: pd.DataFrame,
    *,
    cleaned_dir: Path,
    gaps: str,
    report: Report,
) -> pd.DataFrame:
    """Remove requests for schools no borrowed school table can place.

    Assignment builds student-program distances from school coordinates and
    rejects a non-finite distance, so a program at a school the school table
    does not contain would fail a run rather than degrade it. Dropping the
    program but keeping the request that ranks it is worse still: the student
    then references a program ID the market does not know, which
    ``Students._validate_ranked_programs`` rejects. So the request and the
    program go together, before anything else is built.

    Only the grades with a registered school table are checked. Requests for
    other grades are left alone: nothing borrows a school table for them, so
    there is nothing to be inconsistent with.
    """
    requests = prerun.copy()
    requests["_grade"] = requests["Grade"].map(normalize_grade)
    requests["_school_id"] = (
        _school_ids(requests["idSchool"], "Pre-run")
        .replace(RAW_SCHOOL_ID_ALIASES)
        .astype("Int64")
    )

    drop = pd.Series(False, index=requests.index)
    findings: list[dict[str, Any]] = []
    for grade, bundle in GRADE_BUNDLES.items():
        in_grade = requests["_grade"] == grade
        if not in_grade.any():
            continue
        # A school must be locatable in every variant that could select it, so
        # the union of both school tables is the fair test.
        locatable: set[int] = set()
        for include_mission_bay in (False, True):
            if include_mission_bay and bundle.schools_mission_bay is None:
                continue
            locatable |= locatable_schools(
                bundle,
                cleaned_dir=cleaned_dir,
                include_mission_bay=include_mission_bay,
            )
        unlocatable = sorted(
            set(requests.loc[in_grade, "_school_id"].dropna().astype(int)) - locatable
        )
        if not unlocatable:
            continue
        affected = in_grade & requests["_school_id"].isin(unlocatable)
        drop |= affected
        findings.append(
            {
                "grade": grade,
                "schools": unlocatable,
                "requests": int(affected.sum()),
                "school_names": sorted(
                    set(requests.loc[affected, "SchoolName"].dropna().astype(str))
                )
                if "SchoolName" in requests.columns
                else [],
            }
        )

    if not findings:
        return prerun

    if gaps == "fail":
        raise TransferGapError(
            "These schools appear in the transfer but have no coordinates in "
            f"the borrowed school tables: {findings}. The transfer carries no "
            "school table of its own. Re-run with --gaps fill-and-report to "
            "drop the affected requests and programs, or add the schools to "
            "the school table."
        )
    report.substitute(
        kind="unlocatable_school",
        findings=findings,
        rule=(
            "the affected requests are removed from the student preference "
            "lists and the affected programs are not emitted"
        ),
        reason=(
            "the borrowed school table has no coordinates for these schools, "
            "and assignment rejects a non-finite student-program distance"
        ),
    )
    return prerun.loc[~drop.to_numpy()].reset_index(drop=True)


def _report_round_participation(
    prerun: pd.DataFrame, demographics: pd.DataFrame, report: Report
) -> None:
    """Record how many applicants the demographics extract ties to a later round.

    The pre-run carries one list per student and nothing that says which round
    it came from, so the converter labels every list ``r1_``. The demographics
    extract has a per-student ``rounds_applied`` field naming the later rounds
    that student engaged with (``4`` is the amendment round; ``50``, ``666``,
    ``888``, ``902``, ``905`` and ``999`` are administrative codes). It is
    blank for the majority but not for a small minority, so the ``r1_`` label
    is an assumption worth quantifying rather than a fact.

    Two checks say the pre-run is nonetheless a single coherent snapshot rather
    than several rounds stacked together: no student has a repeated rank, and
    ``idRequest`` spans the same range for tagged and untagged students, so
    tagged rows are not a distinguishable later batch. Whatever round each list
    came from, there is exactly one per student and no column separates them.
    """
    if "rounds_applied" not in demographics.columns:
        report.missing(
            "round_participation",
            {
                "note": (
                    "the demographics extract has no rounds_applied column, so "
                    "nothing corroborates or contradicts the r1_ label"
                )
            },
        )
        return

    identity = demographics["scrambledstudentno"].astype("string").str.strip()
    tags = (
        demographics.loc[identity.notna() & identity.ne(""), "rounds_applied"]
        .groupby(identity.loc[identity.notna() & identity.ne("")])
        .first()
    )

    def normalize(value: Any) -> str:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return "blank"
        parts = (part.strip() for part in str(value).replace("\n", ",").split(","))
        return ",".join(sorted(part for part in parts if part)) or "blank"

    applicants = prerun["scrambledstudentno"].astype("string").str.strip()
    grades = prerun["Grade"].map(normalize_grade)
    payload: dict[str, Any] = {}
    for label, mask in (
        ("all_applicants", pd.Series(True, index=prerun.index)),
        ("kindergarten_applicants", grades.eq("KG")),
    ):
        selected = applicants.loc[mask].drop_duplicates()
        if selected.empty:
            continue
        counts = selected.map(tags).map(normalize).value_counts()
        total = int(counts.sum())
        blank = int(counts.get("blank", 0))
        payload[label] = {
            "total": total,
            "no_later_round_recorded": blank,
            "some_later_round_recorded": total - blank,
            "by_rounds_applied": {
                str(key): int(value) for key, value in counts.items() if key != "blank"
            },
        }
    payload["note"] = (
        "rounds_applied is a per-student field in the demographics extract, "
        "not a per-request one, and idRequest does not separate tagged from "
        "untagged rows, so the pre-run cannot be split by round. Students "
        "counted under some_later_round_recorded still contribute exactly one "
        "list, emitted as r1_."
    )
    report.missing("round_participation", payload)


# --------------------------------------------------------------------------- #
# TK-to-K promotion: the preference construction
# --------------------------------------------------------------------------- #
def load_prior_tk_lists(
    year: str,
    *,
    transfer: Path,
    cleaned_dir: Path,
    report: Report,
) -> dict[int, list[tuple[int, str]]]:
    """Load the TK requests the market's students filed the year before."""
    source = prior_tk_source(year)
    if source.cleaned_file is not None:
        path = cleaned_dir / source.cleaned_file
        require_readable(
            path,
            f"prior-year TK preference lists for {year} (a student promoted "
            "into kindergarten filed no kindergarten application, so this is "
            "the only list they have)",
        )
        frame = pd.read_csv(path, low_memory=False)
        lists = tk_lists_from_cleaned_table(frame)
        report.derived(
            "pref_source=tk_imputed",
            f"TK rows of {source.cleaned_file}, mapped onto this year's "
            "kindergarten programs",
        )
    else:
        paths = discover_transfer_files(transfer, source.transfer_year)
        require_readable(
            paths["prerun"],
            f"prior-year TK preference lists for {year}",
        )
        prior = _clean_nulls(pd.read_csv(paths["prerun"], low_memory=False))
        identity = _student_identity(
            prior["scrambledstudentno"], f"Pre-run {source.transfer_year}"
        )
        lists = tk_lists_from_prerun(prior, identity)
        report.derived(
            "pref_source=tk_imputed",
            f"Grade == TK requests in the {source.transfer_year} pre-run, "
            "mapped onto this year's kindergarten programs",
        )
    report.row_counts["prior_year_tk_students"] = len(lists)
    return lists


def build_market_table(
    students: pd.DataFrame,
    postrun: pd.DataFrame,
    *,
    market: Market,
    year: str,
    capacities: pd.DataFrame,
    promotion: PromotionMap,
    prior_tk: dict[int, list[tuple[int, str]]],
    report: Report,
) -> pd.DataFrame:
    """Build every market student's promotion columns and preference list.

    ``students`` must already hold the whole market at the grade -- see
    :func:`define_market` and :func:`add_market_students` -- so this reads its
    grade rows rather than deciding again who is in the market.

    Returns one row per market student, indexed by ``studentno``, holding the
    promotion columns, the reconstructed preference lists, and the derived
    counts that read them.
    """
    grade = market.grade
    outcomes = _postrun_outcomes(postrun)
    indexed = students.set_index("studentno")
    at_grade = indexed.loc[indexed["grade"] == grade]
    # The applicant set is the pre-run's, carried on the market: the table's
    # grade rows are the whole market by this point, promoted non-applicants
    # included, and treating one of those as an applicant would label an empty
    # submitted list ``k_list``.
    applicants = at_grade.loc[at_grade.index.isin(market.applicants)]
    population = sorted(at_grade.index)

    inputs = PreferenceInputs(
        submitted={
            int(studentno): list(
                zip(
                    (int(value) for value in _literal(schools)),
                    (str(value) for value in _literal(programs)),
                    strict=True,
                )
            )
            for studentno, schools, programs in zip(
                applicants.index,
                applicants["r1_ranked_idschool"],
                applicants["r1_programs"],
                strict=True,
            )
        },
        submitted_ranks={
            int(studentno): [int(value) for value in _literal(ranks)]
            for studentno, ranks in zip(
                applicants.index, applicants["r1_listed_ranks"], strict=True
            )
        },
        submitted_lottery=_aligned_list_column(applicants, "r1_randomnumber", float),
        submitted_cohort=_aligned_list_column(applicants, "r1_cohortstring", str),
        prior_tk=prior_tk,
        tk_to_k=promotion.active_feeders(),
        feeders=market.feeders,
        attendance_area=_attendance_area_map(outcomes),
        student_lottery=_student_lottery_map(outcomes),
        kindergarten_programs=market.programs,
    )
    results = build_market_preferences(population, inputs)

    frame = pd.DataFrame(
        {
            "r1_ranked_idschool": [_as_list_literal(r.schools) for r in results],
            "r1_listed_ranks": [_as_list_literal(r.ranks) for r in results],
            "r1_programs": [_as_list_literal(r.programs) for r in results],
            "r1_randomnumber": [_as_list_literal(r.lottery) for r in results],
            "r1_cohortstring": [_as_list_literal(r.cohort) for r in results],
            "num_ranked": [len(r.schools) for r in results],
            "lowell_ranked": [int(LOWELL_SCHOOL_ID in r.schools) for r in results],
            "sota_ranked": [int(SOTA_SCHOOL_ID in r.schools) for r in results],
            "promote_eligible": [r.promote_eligible for r in results],
            "promote": [
                _as_list_literal(
                    []
                    if r.feeder_school is None
                    else [f"{r.feeder_school}-{r.feeder_program}-{grade}"]
                )
                for r in results
            ],
            "feeder_school": pd.array(
                [r.feeder_school for r in results], dtype="Int64"
            ),
            "feeder_program": pd.array(
                [r.feeder_program for r in results], dtype="string"
            ),
            "pref_source": [r.pref_source for r in results],
        },
        index=pd.Index([r.studentno for r in results], name="studentno"),
    )
    _report_market(
        frame,
        results,
        market=market,
        capacities=capacities,
        promotion=promotion,
        year=year,
        report=report,
    )
    return frame


def _aligned_list_column(
    applicants: pd.DataFrame, column: str, cast: Any
) -> dict[int, list[Any]]:
    """Parse one of the per-request list columns, keyed by student."""
    if column not in applicants.columns:
        return {}
    return {
        int(studentno): [cast(value) for value in _literal(values)]
        for studentno, values in zip(applicants.index, applicants[column], strict=True)
    }


def _attendance_area_map(outcomes: pd.DataFrame) -> dict[int, int]:
    if "idSchoolAttendance" not in outcomes.columns:
        return {}
    schools = (
        _school_ids(outcomes["idSchoolAttendance"], "Post-run idSchoolAttendance")
        .replace(RAW_SCHOOL_ID_ALIASES)
        .astype("Int64")
    )
    return {
        int(studentno): int(school)
        for studentno, school in zip(outcomes.index, schools, strict=True)
        if pd.notna(school)
    }


def _student_lottery_map(outcomes: pd.DataFrame) -> dict[int, float]:
    if "studentRandomNumber" not in outcomes.columns:
        return {}
    numbers = pd.to_numeric(outcomes["studentRandomNumber"], errors="coerce")
    return {
        int(studentno): float(value)
        for studentno, value in zip(outcomes.index, numbers, strict=True)
        if pd.notna(value)
    }


def _report_market(
    frame: pd.DataFrame,
    results: list[PreferenceResult],
    *,
    market: Market,
    capacities: pd.DataFrame,
    promotion: PromotionMap,
    year: str,
    report: Report,
) -> None:
    """Record what the market construction did, and how it compares to the run."""
    grade = market.grade
    applicants = market.applicants
    counts = frame["pref_source"].value_counts()
    report.row_counts[f"market_{grade}_total"] = len(frame)
    for source in PREF_SOURCES:
        report.row_counts[f"market_{grade}_{source}"] = int(counts.get(source, 0))
    report.row_counts[f"market_{grade}_promote_eligible"] = int(
        frame["promote_eligible"].sum()
    )
    report.row_counts[f"market_{grade}_promote_eligible_with_application"] = int(
        frame.loc[frame.index.isin(applicants), "promote_eligible"].sum()
    )
    report.row_counts[f"market_{grade}_feeder_appended"] = sum(
        1 for result in results if result.feeder_appended
    )
    report.row_counts[f"market_{grade}_attendance_area_appended"] = sum(
        1 for result in results if result.attendance_area_appended
    )

    empty = [result.studentno for result in results if not result.schools]
    if empty:
        report.missing(
            f"market_{grade}_students_with_no_list",
            {
                "students": len(empty),
                "reason": (
                    "no kindergarten application, no prior-year TK list with a "
                    "kindergarten counterpart, no feeder, and either no "
                    "attendance-area school on record or one that runs no "
                    "general education kindergarten. An empty list is emitted "
                    "rather than a placement nothing in the transfer states"
                ),
            },
        )

    for column, rule in (
        (
            "promote_eligible",
            "1 when the post-run puts the student in TK in a program that is "
            "also a kindergarten program in this year's capacity file. Not "
            "byPromote, which is 0 for every 2024-25 student despite the "
            "capacity file holding seats, and which marks Lowell and SOTA "
            "admissions at grade 9",
        ),
        ("feeder_school", "idCurrentSchool of a promotion-eligible student"),
        ("feeder_program", "CurrentProgramCode of a promotion-eligible student"),
        (
            "promote",
            "the feeder as a one-element list of program IDs, the shape the "
            "priority layer reads -- the same idiom as currentlpsibling, and "
            "filtered for include_mission_bay by the shared loader where the "
            "raw feeder columns are not",
        ),
        (
            "pref_source",
            "which source supplied the student's list: their own kindergarten "
            "application (k_list), the prior year's TK requests (tk_imputed), "
            "or neither, in which case the list is built from the feeder and "
            "the attendance-area program (feeder_only) or from the "
            "attendance-area program alone (aa_only)",
        ),
        (
            f"r1_ranked_idschool[{grade}]",
            "the submitted list with the feeder appended where the student is "
            "promotion-eligible and does not already rank it. The "
            "attendance-area program is appended only for a student with no "
            "list of their own; for everyone else that is the policy config's "
            "job (add_aa_schools)",
        ),
    ):
        report.derived(column, rule)

    comparison = compare_promote_counts(
        market.entitlements, applicants, capacities, grade
    )
    report.missing(f"promote_counts_vs_district_{grade}", comparison)

    if promotion.empty:
        report.note(
            f"{year}: the auto-promotion list for this year carries no map -- "
            "it is the single sentence that all TK students had to reapply for "
            "kindergarten. The identification rule is applied all the same, "
            "because the post-run and the capacity file both contradict the "
            "sentence, and the three sources disagree three ways: the capacity "
            "file holds "
            f"{int(comparison['eligible']['district']):,} seats for promotion, "
            "the post-run flags byPromote for none of them, and it seats "
            f"{report.row_counts.get(f'market_{grade}_seated_without_a_request', 0):,} "
            "students at kindergarten with no application at all. See the "
            "promote_counts_vs_district section for the per-program difference."
        )
    if promotion.ees_feeder and not promotion.ees_active:
        report.note(
            f"{year}: the auto-promotion list carries "
            f"{len(promotion.ees_feeder)} Early Education School feeder rows. "
            "They are parsed and kept but not applied: the feeder rule covers "
            "the 2026-27 TK cohort onward, whose first kindergarten class "
            f"enters in SY{EES_FEEDER_FIRST_YEAR[:2]}-{EES_FEEDER_FIRST_YEAR[2:]}. "
            "A TK student at an EES in these years had to apply."
        )


def validate_market(
    table: pd.DataFrame,
    students: pd.DataFrame,
    enrolled: pd.DataFrame,
    postrun: pd.DataFrame,
    *,
    market: Market,
    report: Report,
    not_enrolled: Iterable[int] = (),
) -> None:
    """Fail the build when the emitted tables do not reconcile with the market.

    These gates are checkable against the transfer itself, so they hold for
    any year rather than only for the three converted so far:

    * every market student carries exactly one ``pref_source``, drawn from the
      four the schema names;
    * the market is exactly the student table's rows at the grade, which is
      what makes ``student_<year>.csv`` the applicant pool rather than the
      subset of it that filed something;
    * the ``k_list`` students are exactly the Main Round applicants, and
      exactly the ``mr_applicant`` rows;
    * the enrolled table is a subset of the market and holds every market
      student the post-run seats, except those in ``not_enrolled`` -- the
      students the enrolment record puts at school 899.

    The numbers those gates come out at are pinned in the converter's tests
    against the real transfer, not asserted here: this checks that the parts
    agree with each other, the tests check that they agree with the district.
    """
    grade = market.grade
    sources = table["pref_source"]
    unknown = sorted(set(sources.dropna()) - set(PREF_SOURCES))
    if unknown or sources.isna().any():
        raise TransferGapError(
            f"grade {grade}: {int(sources.isna().sum())} market students have "
            f"no pref_source and {unknown} are not among {list(PREF_SOURCES)}."
        )
    buckets = int(sources.value_counts().sum())
    if buckets != len(table):
        raise TransferGapError(
            f"grade {grade}: the pref_source buckets hold {buckets:,} students "
            f"but the market has {len(table):,}."
        )

    at_grade = students.loc[students["grade"].eq(grade)]
    if set(at_grade["studentno"]) != set(table.index):
        raise TransferGapError(
            f"grade {grade}: the student table holds {len(at_grade):,} rows at "
            f"the grade but the market has {len(table):,}. The table is the "
            "applicant pool, so the two are the same set by construction; a "
            "difference means a market student has no row, or a row has no "
            "constructed preference list."
        )

    listed = set(table.index[sources.eq("k_list")])
    if listed != market.applicants & set(table.index):
        raise TransferGapError(
            f"grade {grade}: {len(listed):,} students are labelled k_list but "
            f"the pre-run holds a request from {len(market.applicants):,} at "
            "the grade."
        )
    flagged = set(at_grade.loc[at_grade["mr_applicant"].eq(1), "studentno"])
    if flagged != listed:
        raise TransferGapError(
            f"grade {grade}: {len(flagged):,} rows are flagged mr_applicant "
            f"but {len(listed):,} are labelled k_list. The two say the same "
            "thing and must agree."
        )

    outside = sorted(set(enrolled["studentno"]) - set(table.index))
    if outside:
        raise TransferGapError(
            f"grade {grade}: {len(outside)} enrolled students are outside the "
            "market, so the enrolled population is not a subset of the "
            "applicant one."
        )
    seated = seated_at_grade(_postrun_outcomes(postrun), grade)
    if seated is not None:
        unclaimed = sorted(
            set(seated.index)
            & set(table.index) - set(enrolled["studentno"]) - set(not_enrolled)
        )
        if unclaimed:
            raise TransferGapError(
                f"grade {grade}: the post-run seats {len(unclaimed)} market "
                "students the enrolled table does not hold."
            )

    no_source = int(sources.eq("feeder_only").sum()) + int(sources.eq("aa_only").sum())
    report.note(
        f"The grade-{grade} market is {len(table):,} students: "
        f"{int(sources.eq('k_list').sum()):,} with their own Main Round list, "
        f"{int(sources.eq('tk_imputed').sum()):,} whose list is imputed from "
        f"the TK requests they filed the year before, and {no_source:,} with "
        "neither, who get their feeder and their attendance-area program. "
        f"{int(table['promote_eligible'].sum()):,} of them hold an "
        "auto-promotion claim on one program."
    )


def apply_market_columns(
    frame: pd.DataFrame, table: pd.DataFrame, *, grade: str
) -> pd.DataFrame:
    """Overwrite one table's kindergarten rows with the market construction.

    Rows outside the market and rows at another grade are untouched, so the
    student table keeps every other grade exactly as the pre-run filed it.
    """
    result = frame.copy()
    for column in table.columns:
        if column not in result.columns:
            result[column] = pd.NA
    selected = result["studentno"].isin(table.index) & result["grade"].eq(grade)
    if not selected.any():
        return result
    aligned = table.reindex(result.loc[selected, "studentno"])
    for column in table.columns:
        result.loc[selected, column] = aligned[column].to_numpy()
        # Writing into a column of pd.NA widens a school ID to float, so
        # feeder_school would be emitted as "664.0" where every other school
        # column in the table is "664". Restore the nullable integer.
        if table[column].dtype == "Int64":
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(
                "Int64"
            )
    return result


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
    auxiliary = discover_auxiliary_files(transfer, year)
    report = Report(year=year, transfer=str(transfer))
    report.inputs = {role: str(path) for role, path in paths.items()}
    report.inputs |= {role: str(path) for role, path in auxiliary.items()}

    for role, path in paths.items():
        require_readable(path, f"transfer {role} file")
    require_readable(
        auxiliary["capacities"],
        "transfer Main Round capacity file (kindergarten capacity comes from "
        "it, and its promote counts are the validation gates)",
    )
    require_readable(
        auxiliary["autopromotion"],
        "transfer TK-to-K auto-promotion list",
    )
    prerun = _clean_nulls(pd.read_csv(paths["prerun"], low_memory=False))
    postrun = _clean_nulls(pd.read_csv(paths["postrun"], low_memory=False))
    demographics = _clean_nulls(pd.read_csv(paths["demographics"], low_memory=False))
    capacities = load_mr_capacities(auxiliary["capacities"])
    promotion = build_promotion_map(auxiliary["autopromotion"], year)
    report.row_counts = {
        "prerun_rows": len(prerun),
        "postrun_rows": len(postrun),
        "demographics_rows": len(demographics),
        "capacity_rows": len(capacities),
        "autopromotion_same_site_rows": len(promotion.same_site),
        "autopromotion_ees_feeder_rows": len(promotion.ees_feeder),
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

    prerun = drop_unlocatable_requests(
        prerun, cleaned_dir=cleaned_dir, gaps=gaps, report=report
    )
    report.row_counts["prerun_rows_used"] = len(prerun)

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

    report.row_counts["main_round_applicants"] = len(students)
    report.row_counts["kindergarten_main_round_applicants"] = int(
        (students["grade"] == "KG").sum()
    )

    # From 2024-25 the kindergarten applicant pool is not the set of people
    # who applied: SFUSD promotes TK students into kindergarten, and they hold
    # a claim on a seat without filing anything. They join the student table
    # first, so that everything below -- the preference construction, the
    # enrolled table, the emitted columns -- sees one population.
    market = define_market(
        students,
        prerun,
        postrun,
        grade="KG",
        capacities=capacities,
        cleaned_dir=cleaned_dir,
        report=report,
    )
    students = add_market_students(
        students,
        postrun,
        demographics,
        block_indices,
        market=market,
        report=report,
        scenario_name=scenario_name,
    )
    table = build_market_table(
        students,
        postrun,
        market=market,
        year=year,
        capacities=capacities,
        promotion=promotion,
        prior_tk=load_prior_tk_lists(
            year, transfer=transfer, cleaned_dir=cleaned_dir, report=report
        ),
        report=report,
    )
    students = apply_market_columns(students, table, grade=market.grade)

    student_columns = [
        *STUDENT_COLUMNS,
        *PROMOTION_STUDENT_COLUMNS,
        *MARKET_STUDENT_COLUMNS,
    ]
    for column in student_columns:
        if column not in students.columns:
            students[column] = pd.NA
    students = students[student_columns]
    report.row_counts["students"] = len(students)

    # The enrolled table is a filter of the student table, so the enrolled
    # population is a subset of the applicant one by construction.
    enrolled = build_enrolled_table(
        students, postrun, demographics, market=market, report=report
    )
    validate_market(
        table,
        students,
        enrolled,
        postrun,
        market=market,
        report=report,
        not_enrolled=not_enrolled_students(demographics, report),
    )

    # Counted after the market is written on, so the promotion columns are
    # included and the kindergarten preference columns are the emitted ones.
    report.missing(
        "student_table",
        {
            column: int(students[column].isna().sum())
            for column in student_columns
            if students[column].isna().any()
        },
    )

    outputs: dict[str, pd.DataFrame] = {
        f"student_{year}.csv": students,
        f"enrolled_{year}.csv": enrolled,
    }

    for grade, bundle in GRADE_BUNDLES.items():
        variants = (
            (False, bundle.programs_template),
            (True, bundle.programs_mission_bay_template),
        )
        for include_mission_bay, template in variants:
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
                capacities=capacities,
            )
            if programs is None:
                continue
            outputs[template.format(year=year)] = programs
            report.row_counts[f"programs_{grade}_mb{int(include_mission_bay)}"] = len(
                programs
            )
            report.row_counts[
                f"programs_{grade}_mb{int(include_mission_bay)}_seats"
            ] = int(pd.to_numeric(programs["capacity"], errors="coerce").sum())

    report.note(
        "No school table is emitted. The transfer carries no school "
        "coordinates, category, or rating, so the registry reuses the "
        "checked-in 2023-24 school tables; school attributes are therefore "
        "2023-24 vintage while students and programs are this year's."
    )
    report.note(
        "Grades 6 and 9 still borrow their capacities from the checked-in "
        "district tables. The transfer's capacity file covers them, but their "
        "held seats are a different phenomenon -- invisible K-8 continuers at "
        "grade 6, Lowell and SOTA admissions at grade 9, both of which "
        "byPromote also marks -- and neither has been reconciled against the "
        "district's counts the way kindergarten has. Neither grade has an "
        "attendance-area school in the transfer at all."
    )
    report.note(
        "The pre-run holds exactly one preference list per student: it has no "
        "round column and no student has a repeated rank. Only r1_* preference "
        "columns are emitted, so `rounds: all` resolves to [1] for these "
        "years. That list is the main round -- the district confirmed the "
        "extract is the main-round request file (Levitt, 15 Sep 2026), and the "
        "post-run corroborates it: every student it seats at a grade either "
        "holds a pre-run request for that grade or is flagged byPromote. What "
        "the transfer omits is the later-round requests themselves: see "
        "rounds_applied below for how many applicants also engaged with a "
        "later round elsewhere."
    )
    report.note(
        "Kindergarten capacity is the Main Round capacity file's TotalSeats, "
        "gross. No seat is held back for auto-promotion anywhere in the data. "
        "The file's OpenSeatsPreRun does net the held seats out, and is "
        "deliberately unused: a promote who wins a school elsewhere releases "
        "the held seat back into the same run, so a netted table models a "
        "market the district never ran. The promotion claim is a run-time "
        "priority at the feeder program instead, and the data side of it is "
        "the feeder columns on the student and enrolled tables."
    )
    _report_round_participation(prerun, demographics, report)

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

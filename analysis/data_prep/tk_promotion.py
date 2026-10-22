"""SFUSD's TK-to-K auto-promotion, as a data layer.

Kindergarten has two entry paths. A student either applies in the Main Round,
or is already in transitional kindergarten and is promoted into K without
applying. The district adopted auto-promotion in April 2025:

* TK at an elementary or K-8 that also runs kindergarten in the building ->
  the same school and the same pathway, no application.
* TK at an Early Education School or the Mission Education Center -> a
  designated feeder elementary in the same pathway, no application, but only
  from the 2026-27 TK cohort, whose first kindergarten class enters in
  SY2027-28. Every earlier EES cohort had to apply.

A family that wants a different school files an ordinary Main Round
application and keeps its current or feeder seat if nothing it asked for comes
through. That makes the kindergarten market the main-round applicants *plus*
the promoted students, with some students in both groups, and it makes the
promotion claim a *priority* rather than a reserved seat: a promote who wins a
school elsewhere releases the held seat back into the same run. At 413-GE in
SY26-27 there were 52 open seats before the run and 53 choice assignments.

This module builds the three artifacts that fact needs:

#. :func:`build_promotion_map` -- the ``(TK school, TK pathway) -> (K school,
   K pathway)`` map the district publishes each year.
#. :func:`load_mr_capacities` -- the Main Round capacity file, whose
   ``TotalSeats`` is the gross kindergarten capacity and whose promote counts
   are the validation gates.
#. :func:`promotion_entitlements` and :func:`build_market_preferences` -- who
   is entitled to which program, and the preference list each student in the
   market carries.

Nothing here reserves a seat or ranks a student. The priority boost that makes
the claim bite is a run-time concern; see ``PriorityGenerator``.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.data_prep.sfusd_transfer_schema import (  # noqa: E402
    AUTOPROMOTION_COLUMNS,
    AUTOPROMOTION_FILENAME,
    AUTOPROMOTION_HEADER_ROWS,
    AUXILIARY_DIRECTORY,
    EES_FEEDER_FIRST_YEAR,
    MR_CAPACITY_FILENAME,
    MR_CAPACITY_PROVENANCE,
    MR_CAPACITY_REQUIRED,
    PRIOR_TK_SOURCES,
    RAW_SCHOOL_ID_ALIASES,
    TRANSFER_YEAR_LABELS,
)
from loaders.tables import normalize_grade  # noqa: E402

#: A program, as this module keys one: ``(school id, pathway code)`` at a
#: single grade. The grade is implicit -- every map in here is kindergarten.
ProgramKey = tuple[int, str]


# --------------------------------------------------------------------------- #
# Auxiliary file discovery
# --------------------------------------------------------------------------- #
def auxiliary_directory(transfer: Path) -> Path:
    """Return the transfer's auxiliary-data folder."""
    return transfer / AUXILIARY_DIRECTORY


def discover_auxiliary_files(transfer: Path, year: str) -> dict[str, Path]:
    """Locate the capacity file and auto-promotion list for one school year.

    Both live in one folder shared by every year, named by a ``YY-YY`` label
    rather than by the ``SY``-prefixed subfolder name the three per-year CSVs
    use.
    """
    label = TRANSFER_YEAR_LABELS.get(year)
    if label is None:
        raise KeyError(
            f"Year {year!r} has no auxiliary-file label; known years are "
            f"{sorted(TRANSFER_YEAR_LABELS)}."
        )
    folder = auxiliary_directory(transfer)
    return {
        "capacities": folder / MR_CAPACITY_FILENAME.format(label=label),
        "autopromotion": folder / AUTOPROMOTION_FILENAME.format(label=label),
    }


# --------------------------------------------------------------------------- #
# Artifact 2 -- Main Round capacities
# --------------------------------------------------------------------------- #
def load_mr_capacities(path: Path) -> pd.DataFrame:
    """Read one year's Main Round capacity file into normalized columns.

    The file is one row per school, grade, and program, covering every grade
    from PK to 13. School IDs are put through the Mission Bay alias and grades
    through :func:`normalize_grade`, so ``K`` arrives as ``KG`` and matches the
    rest of the repository.
    """
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in MR_CAPACITY_REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"Capacity file {path} is missing columns {missing}.")

    result = pd.DataFrame(index=frame.index)
    result["school_id"] = (
        pd.to_numeric(frame["idSchool"], errors="coerce")
        .astype("Int64")
        .replace(RAW_SCHOOL_ID_ALIASES)
        .astype("Int64")
    )
    result["school_name"] = frame.get("SchoolName", pd.Series(pd.NA, index=frame.index))
    result["grade"] = frame["Grade"].map(normalize_grade)
    result["program_type"] = frame["ProgramCode"].astype("string").str.strip()
    result["capacity"] = pd.to_numeric(frame["TotalSeats"], errors="coerce")
    for raw_column, output_column in MR_CAPACITY_PROVENANCE.items():
        result[output_column] = (
            pd.to_numeric(frame[raw_column], errors="coerce")
            if raw_column in frame.columns
            else pd.NA
        )

    unusable = result["school_id"].isna() | result["program_type"].isna()
    if unusable.any():
        result = result.loc[~unusable]
    duplicated = result.duplicated(["school_id", "grade", "program_type"], keep=False)
    if duplicated.any():
        bad = (
            result.loc[duplicated, ["school_id", "grade", "program_type"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            f"Capacity file {path} has several rows for one school, grade and "
            f"program: {bad[:10]}. The capacity is ambiguous."
        )
    return result.reset_index(drop=True)


def program_keys(capacities: pd.DataFrame, grade: str) -> set[ProgramKey]:
    """Return the ``(school, pathway)`` programs the capacity file has at one grade.

    Every program the file lists is a program that exists, including the ones
    it opens with no seats: ``TotalSeats <= 0`` marks a closed program, not an
    absent one, and a TK student sitting in a closed program is still a TK
    student in a kindergarten program.
    """
    selected = capacities.loc[capacities["grade"] == grade]
    return {
        (int(school), str(program))
        for school, program in zip(
            selected["school_id"], selected["program_type"], strict=True
        )
        if pd.notna(school) and pd.notna(program)
    }


# --------------------------------------------------------------------------- #
# Artifact 1 -- the TK-to-K map
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PromotionMap:
    """One year's published ``(TK school, TK pathway) -> (K program)`` map.

    Split in two because the two halves arrived under different policies and
    take effect in different years:

    ``same_site``
        TK at an elementary or K-8 that also runs kindergarten. The TK pathway
        code is an ordinary one (``GE``, ``SE``, ``CE``) and the destination is
        the same school and the same pathway.

    ``ees_feeder``
        TK at an Early Education School or the Mission Education Center. The TK
        pathway code is compound and encodes the destination -- ``GE750`` is
        general education feeding Sunset, ``SE420`` Spanish immersion feeding
        Alvarado -- because one EES feeds several elementaries. These rows
        appear only in the SY26-27 list and apply only from
        ``EES_FEEDER_FIRST_YEAR``; see :meth:`active_feeders`.
    """

    year: str
    same_site: dict[ProgramKey, ProgramKey] = field(default_factory=dict)
    ees_feeder: dict[ProgramKey, ProgramKey] = field(default_factory=dict)
    #: True when the file held no data rows at all.
    empty: bool = False

    @property
    def ees_active(self) -> bool:
        """Whether this year's kindergarten class is covered by the EES rule."""
        return self.year >= EES_FEEDER_FIRST_YEAR

    def active_feeders(self) -> dict[ProgramKey, ProgramKey]:
        """The map as it applies to this year, EES rows included only if due."""
        resolved = dict(self.same_site)
        if self.ees_active:
            resolved.update(self.ees_feeder)
        return resolved


def build_promotion_map(path: Path, year: str) -> PromotionMap:
    """Parse one year's auto-promotion list into a TK-to-K program map.

    The lists are spreadsheets exported with their prose intact: a paragraph of
    policy, sometimes a table of excluded schools, then a header row and the
    map. ``AUTOPROMOTION_HEADER_ROWS`` says how far down the header sits, and
    ``None`` marks a year with no map at all -- SY24-25's file is the single
    sentence "NONE - All TK students had to reapply for K", which parses to an
    empty map rather than to an error.
    """
    header_rows = AUTOPROMOTION_HEADER_ROWS.get(year)
    if header_rows is None:
        return PromotionMap(year=year, empty=True)

    frame = pd.read_csv(path, skiprows=header_rows, header=0, dtype=str)
    if frame.shape[1] < len(AUTOPROMOTION_COLUMNS):
        raise ValueError(
            f"Auto-promotion list {path} has {frame.shape[1]} columns below "
            f"its header, but the map needs {len(AUTOPROMOTION_COLUMNS)}: "
            f"{list(AUTOPROMOTION_COLUMNS)}. Check "
            "AUTOPROMOTION_HEADER_ROWS for this year."
        )
    # The district pads every row out to the width of the prose line above the
    # header, so there are trailing empty columns to discard.
    frame = frame.iloc[:, : len(AUTOPROMOTION_COLUMNS)]
    frame.columns = list(AUTOPROMOTION_COLUMNS)

    tk_school = pd.to_numeric(frame["tk_school"], errors="coerce")
    k_school = pd.to_numeric(frame["k_school"], errors="coerce")
    tk_pathway = frame["tk_pathway"].astype("string").str.strip()
    k_pathway = frame["k_pathway"].astype("string").str.strip()
    usable = tk_school.notna() & k_school.notna() & tk_pathway.notna()
    usable &= k_pathway.notna()
    if not usable.any():
        return PromotionMap(year=year, empty=True)

    same_site: dict[ProgramKey, ProgramKey] = {}
    ees_feeder: dict[ProgramKey, ProgramKey] = {}
    for tk_id, tk_path, k_id, k_path in zip(
        tk_school.loc[usable],
        tk_pathway.loc[usable],
        k_school.loc[usable],
        k_pathway.loc[usable],
        strict=True,
    ):
        source = (
            int(RAW_SCHOOL_ID_ALIASES.get(int(tk_id), int(tk_id))),
            str(tk_path),
        )
        target = (
            int(RAW_SCHOOL_ID_ALIASES.get(int(k_id), int(k_id))),
            str(k_path),
        )
        if source == target:
            same_site[source] = target
        else:
            ees_feeder[source] = target
    return PromotionMap(year=year, same_site=same_site, ees_feeder=ees_feeder)


# --------------------------------------------------------------------------- #
# Prior-year TK preference lists
# --------------------------------------------------------------------------- #
def prior_tk_source(year: str) -> Any:
    """Return where one run year's prior-year TK requests come from."""
    source = PRIOR_TK_SOURCES.get(year)
    if source is None:
        raise KeyError(
            f"Year {year!r} has no prior-year TK source; known years are "
            f"{sorted(PRIOR_TK_SOURCES)}."
        )
    return source


def tk_lists_from_cleaned_table(frame: pd.DataFrame) -> dict[int, list[ProgramKey]]:
    """Read TK preference lists out of a checked-in cleaned student table.

    A student on the TK roll with no list at all still gets an entry, holding
    an empty list. Being on the roll is the fact that matters: it is why the
    converter looked here, and it is what ``pref_source = tk_imputed``
    records.
    """
    rows = frame.loc[frame["grade"].astype("string").str.strip().eq("TK")]
    lists: dict[int, list[ProgramKey]] = {}
    for studentno, schools, programs in zip(
        rows["studentno"],
        rows.get("r1_ranked_idschool", pd.Series(pd.NA, index=rows.index)),
        rows.get("r1_programs", pd.Series(pd.NA, index=rows.index)),
        strict=True,
    ):
        lists[int(studentno)] = _paired_choices(schools, programs)
    return lists


def tk_lists_from_prerun(
    prerun: pd.DataFrame, identity: pd.Series
) -> dict[int, list[ProgramKey]]:
    """Read TK preference lists out of an earlier year's pre-run extract.

    Args:
        prerun: The earlier year's pre-run, one row per ranked choice.
        identity: That pre-run's ``scrambledstudentno`` already converted to
            the repository's integer ``studentno``, aligned to its index.
    """
    frame = prerun.copy()
    frame["studentno"] = identity
    frame["grade"] = frame["Grade"].map(normalize_grade)
    frame["school_id"] = (
        pd.to_numeric(frame["idSchool"], errors="coerce")
        .astype("Int64")
        .replace(RAW_SCHOOL_ID_ALIASES)
        .astype("Int64")
    )
    frame["program_type"] = frame["ProgramCode"].astype("string").str.strip()
    frame["rank"] = pd.to_numeric(frame["Rank"], errors="coerce")
    frame = frame.loc[frame["grade"].eq("TK")]
    frame = frame.sort_values(["studentno", "rank"], kind="stable")

    lists: dict[int, list[ProgramKey]] = {}
    for studentno, group in frame.groupby("studentno", sort=True):
        lists[int(studentno)] = [
            (int(school), str(program))
            for school, program in zip(
                group["school_id"], group["program_type"], strict=True
            )
            if pd.notna(school) and pd.notna(program)
        ]
    return lists


def _paired_choices(schools: Any, programs: Any) -> list[ProgramKey]:
    """Zip a cleaned table's parallel school and pathway list literals."""
    school_values = _literal_list(schools)
    program_values = _literal_list(programs)
    if len(school_values) != len(program_values):
        return []
    return [
        (int(school), str(program).strip())
        for school, program in zip(school_values, program_values, strict=True)
        if school is not None and program is not None and not pd.isna(school)
    ]


def _literal_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    return list(parsed) if isinstance(parsed, (list, tuple)) else []


# --------------------------------------------------------------------------- #
# Artifact 3a -- who is entitled to what
# --------------------------------------------------------------------------- #
def promotion_entitlements(
    outcomes: pd.DataFrame, kindergarten_programs: set[ProgramKey]
) -> pd.DataFrame:
    """Identify the promotion-eligible students and name each one's feeder.

    A student is eligible when the post-run puts them in TK
    (``CurrentGrade == TK``) in a program that also exists at kindergarten in
    the same year's capacity file. That program is their feeder.

    This deliberately does not use ``byPromote``. The flag is overloaded --
    kindergarten promotion at K, Lowell and SOTA admission at grade 9, K-8
    continuation at grade 6 -- and it is 0 for every SY24-25 student despite
    the capacity file holding seats. The rule above is checkable against the
    district's own counts instead, and reproduces
    ``TotalPromoteWithReqBeforeRun`` program by program for SY26-27.

    Args:
        outcomes: The post-run, indexed by ``studentno``.
        kindergarten_programs: Every ``(school, pathway)`` the capacity file
            lists at kindergarten, closed programs included.

    Returns:
        One row per eligible student, indexed by ``studentno``, with
        ``feeder_school`` and ``feeder_program``.
    """
    empty = pd.DataFrame(
        {
            "feeder_school": pd.Series(dtype="Int64"),
            "feeder_program": pd.Series(dtype="string"),
        },
        index=pd.Index([], name=outcomes.index.name or "studentno"),
    )
    required = {"CurrentGrade", "idCurrentSchool", "CurrentProgramCode"}
    if not required <= set(outcomes.columns):
        return empty

    school = (
        pd.to_numeric(outcomes["idCurrentSchool"], errors="coerce")
        .astype("Int64")
        .replace(RAW_SCHOOL_ID_ALIASES)
        .astype("Int64")
    )
    program = outcomes["CurrentProgramCode"].astype("string").str.strip()
    in_tk = outcomes["CurrentGrade"].map(normalize_grade).eq("TK")
    is_kindergarten_program = pd.Series(
        [
            pd.notna(one)
            and pd.notna(other)
            and (int(one), str(other)) in kindergarten_programs
            for one, other in zip(school, program, strict=True)
        ],
        index=outcomes.index,
    )
    eligible = in_tk & is_kindergarten_program
    if not eligible.any():
        return empty
    return pd.DataFrame(
        {
            "feeder_school": school.loc[eligible],
            "feeder_program": program.loc[eligible],
        }
    )


# --------------------------------------------------------------------------- #
# Artifact 3b -- preference construction
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PreferenceInputs:
    """Everything :func:`build_market_preferences` reads, already normalized.

    Every mapping is keyed by ``studentno``. ``submitted`` and ``prior_tk``
    hold ordered ``(school, pathway)`` choices; ``prior_tk`` may hold an empty
    list for a student who is on the prior year's TK roll without a usable
    request, which is still a record and still resolves to ``tk_imputed``.
    """

    #: The student's own kindergarten Main Round list.
    submitted: Mapping[int, Sequence[ProgramKey]]
    #: The rank the district listed against each submitted choice. Non-
    #: contiguous where a choice was withdrawn, so it is carried rather than
    #: recomputed.
    submitted_ranks: Mapping[int, Sequence[int]]
    #: Per-request lottery numbers and cohort strings, aligned to ``submitted``.
    submitted_lottery: Mapping[int, Sequence[float]]
    submitted_cohort: Mapping[int, Sequence[str]]
    #: The prior year's TK requests, before mapping to kindergarten.
    prior_tk: Mapping[int, Sequence[ProgramKey]]
    #: ``(school, pathway) -> (school, pathway)``, this year's active map.
    tk_to_k: Mapping[ProgramKey, ProgramKey]
    #: The student's feeder, for the promotion-eligible only.
    feeders: Mapping[int, ProgramKey]
    #: The student's attendance-area school.
    attendance_area: Mapping[int, int]
    #: The student's own lottery draw in this run, used for a choice the
    #: student did not file and so has no per-request number for.
    student_lottery: Mapping[int, float]
    #: Every kindergarten program in the year's capacity file.
    kindergarten_programs: set[ProgramKey]


@dataclass(slots=True)
class PreferenceResult:
    """One student's constructed kindergarten preference list."""

    studentno: int
    schools: list[int]
    programs: list[str]
    ranks: list[int]
    lottery: list[float]
    cohort: list[str]
    pref_source: str
    promote_eligible: int
    feeder_school: int | None
    feeder_program: str | None
    #: True when the feeder was added to a list that did not already hold it.
    feeder_appended: bool
    #: True when the attendance-area program was added to the list, which
    #: happens only for a student with no list of their own.
    attendance_area_appended: bool


#: The attendance-area pathway. A student with no list of their own falls back
#: to their attendance-area school's general education program; the district
#: runs no attendance-area claim on a language pathway.
ATTENDANCE_AREA_PATHWAY = "GE"


def build_market_preferences(
    population: Iterable[int], inputs: PreferenceInputs
) -> list[PreferenceResult]:
    """Build one kindergarten preference list per student in the market.

    The list resolves in source order -- the student's own kindergarten
    application, else the prior year's TK list mapped through this year's
    TK-to-K map, else nothing -- and ``pref_source`` names the source that
    answered. A promotion-eligible student then gets their feeder appended if
    it is not already on the list, and a student left with nothing gets their
    feeder and their attendance-area program.

    ``pref_source`` reports the source, not the outcome: a student on the
    prior year's TK roll whose requests have no kindergarten counterpart is
    still ``tk_imputed``, because that is where the converter looked and what
    it found. ``feeder_only`` and ``aa_only`` are reserved for a student no
    source held at all.
    """
    results: list[PreferenceResult] = []
    for studentno in population:
        feeder = inputs.feeders.get(studentno)
        if studentno in inputs.submitted:
            source = "k_list"
            choices = list(inputs.submitted[studentno])
            ranks = list(inputs.submitted_ranks.get(studentno, ()))
            lottery = list(inputs.submitted_lottery.get(studentno, ()))
            cohort = list(inputs.submitted_cohort.get(studentno, ()))
        elif studentno in inputs.prior_tk:
            source = "tk_imputed"
            choices = _map_tk_choices(
                inputs.prior_tk[studentno],
                inputs.tk_to_k,
                inputs.kindergarten_programs,
            )
            # The prior year's ranks belong to a TK market with a different
            # program set, and mapping drops entries out of the middle of the
            # list, so the surviving order is renumbered from 1.
            ranks = list(range(1, len(choices) + 1))
            lottery = [_lottery_for(studentno, inputs)] * len(choices)
            cohort = [""] * len(choices)
        else:
            source = None
            choices = []
            ranks = []
            lottery = []
            cohort = []

        # A parallel column the pre-run does not carry arrives as an empty
        # list for every student, and the loader accepts that: a round-aligned
        # list is either empty or exactly as long as the ranked list. What it
        # rejects is a *partial* one, which is what appending to an absent
        # column would produce. So a column that does not already line up with
        # the base stays empty through every append below.
        ranks = ranks if len(ranks) == len(choices) else None
        lottery = lottery if len(lottery) == len(choices) else None
        cohort = cohort if len(cohort) == len(choices) else None

        base_was_empty = not choices
        feeder_appended = False
        if feeder is not None and feeder not in choices:
            ranks = _extend(ranks, _next_rank(ranks or []))
            lottery = _extend(lottery, _lottery_for(studentno, inputs))
            cohort = _extend(cohort, "")
            choices = [*choices, feeder]
            feeder_appended = True

        # A student whose base list came out empty expressed no preference the
        # transfer records, so the list is built for them: their feeder, above,
        # and their attendance-area program. That is the only case in which
        # the attendance area enters the *data* at all. For everyone else
        # appending it is the policy config's job (``add_aa_schools``), which
        # already exists and applies it under each policy's own rules.
        #
        # The fallback and the label are separate. ``pref_source`` names the
        # source that was consulted, so it becomes feeder_only or aa_only only
        # when no source held the student at all; a student on last year's TK
        # roll whose requests have no kindergarten counterpart stays
        # tk_imputed and still gets the fallback list.
        attendance_area_appended = False
        if base_was_empty:
            area = _attendance_area_program(studentno, inputs)
            if area is not None and area not in choices:
                ranks = _extend(ranks, _next_rank(ranks or []))
                lottery = _extend(lottery, _lottery_for(studentno, inputs))
                cohort = _extend(cohort, "")
                choices = [*choices, area]
                attendance_area_appended = True
        if source is None:
            source = "feeder_only" if feeder is not None else "aa_only"

        results.append(
            PreferenceResult(
                studentno=studentno,
                schools=[school for school, _ in choices],
                programs=[program for _, program in choices],
                ranks=ranks or [],
                lottery=lottery or [],
                cohort=cohort or [],
                pref_source=source,
                promote_eligible=int(feeder is not None),
                feeder_school=None if feeder is None else feeder[0],
                feeder_program=None if feeder is None else feeder[1],
                feeder_appended=feeder_appended,
                attendance_area_appended=attendance_area_appended,
            )
        )
    return results


def _map_tk_choices(
    choices: Sequence[ProgramKey],
    tk_to_k: Mapping[ProgramKey, ProgramKey],
    kindergarten_programs: set[ProgramKey],
) -> list[ProgramKey]:
    """Map a TK preference list onto kindergarten programs, dropping the rest.

    An ordinary school keeps its school and pathway, so the published map and
    the plain identity agree; the map's value is the EES rows, where the TK
    pathway code is compound and only the map knows the destination. A TK
    choice with no kindergarten counterpart -- an EES in a year the feeder rule
    does not yet cover, or a program the school no longer runs -- is dropped
    rather than guessed at.
    """
    mapped: list[ProgramKey] = []
    for choice in choices:
        target = tk_to_k.get(choice, choice)
        if target in kindergarten_programs and target not in mapped:
            mapped.append(target)
    return mapped


def _next_rank(ranks: Sequence[int]) -> int:
    """The rank an appended choice takes: one past the lowest already listed."""
    return max(ranks) + 1 if ranks else 1


def _extend(values: list[Any] | None, addition: Any) -> list[Any] | None:
    """Append to a round-aligned list, leaving an unavailable one unavailable.

    ``None`` means the pre-run does not carry this column, in which case the
    output is an empty list for every student rather than a list that is one
    entry long for the students who gained an appended choice.
    """
    return None if values is None else [*values, addition]


def _lottery_for(studentno: int, inputs: PreferenceInputs) -> float:
    """The lottery number to give a choice the student did not file.

    ``r1_randomnumber`` is a per-request draw -- SFUSD runs multiple
    tiebreaking -- so a choice the student never filed has none, and
    ``PriorityGenerator._mtb_real`` rejects a list whose lottery numbers do not
    align with it. The student's own post-run draw is a real number from the
    same run, and it is already what ``r1_designation_randomnumber`` carries.
    It decides nothing at a feeder in any case: the promotion boost outranks
    every tiebreaker there.

    Falls back to 0, which is the worst draw -- priorities and lottery numbers
    are added and the larger wins, and ``_mtb_real`` already leaves 0 in every
    cell it has no historical number for. Every transfer year so far carries
    ``studentRandomNumber`` for every student, so the fallback is a guard
    rather than a path; the converter reports it when it fires.
    """
    value = inputs.student_lottery.get(studentno)
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def _attendance_area_program(
    studentno: int, inputs: PreferenceInputs
) -> ProgramKey | None:
    """The student's attendance-area GE program, when one exists this year.

    Returns None for a student with no attendance-area school on record, and
    for one whose attendance-area school runs no general education
    kindergarten. Both are real: SY26-27 has 70 such students in the market.
    Inventing a placement for them would be worse than an empty list, which is
    recorded in the conversion report instead.
    """
    school = inputs.attendance_area.get(studentno)
    if school is None:
        return None
    candidate = (int(school), ATTENDANCE_AREA_PATHWAY)
    return candidate if candidate in inputs.kindergarten_programs else None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def compare_promote_counts(
    entitlements: pd.DataFrame,
    applicants: set[int],
    capacities: pd.DataFrame,
    grade: str,
) -> dict[str, Any]:
    """Compare the identified promotes with the district's own counts.

    The capacity file publishes, per program, how many students the run held a
    seat for (``TotalPromoteBeforeRun``) and how many of those also filed an
    application (``TotalPromoteWithReqBeforeRun``). The identification rule
    should reproduce both. Where it does not, the difference is a fact about
    the transfer worth recording program by program rather than a reason to
    stop: SY26-27 identifies 1,178 against 1,188, and SY24-25 identifies 607
    against a file that promoted nobody.
    """
    district = capacities.loc[capacities["grade"] == grade].set_index(
        ["school_id", "program_type"]
    )
    identified = entitlements.groupby(["feeder_school", "feeder_program"]).size()
    with_request = (
        entitlements.loc[entitlements.index.isin(applicants)]
        .groupby(["feeder_school", "feeder_program"])
        .size()
    )

    def _per_program(mine: pd.Series, column: str) -> dict[str, Any]:
        theirs = district[column] if column in district.columns else None
        theirs = pd.Series(dtype="float64") if theirs is None else theirs.dropna()
        table = pd.concat(
            [mine.rename("identified"), theirs.rename("district")], axis=1
        ).fillna(0)
        differing = table.loc[table["identified"] != table["district"]]
        return {
            "identified": int(table["identified"].sum()),
            "district": int(table["district"].sum()),
            "programs_differing": int(len(differing)),
            "differences": [
                {
                    "school_id": int(school),
                    "program_type": str(program),
                    "identified": int(row["identified"]),
                    "district": int(row["district"]),
                }
                for (school, program), row in differing.iterrows()
            ],
        }

    return {
        "eligible": _per_program(identified, "total_promote_before_run"),
        "eligible_with_application": _per_program(
            with_request, "total_promote_with_request_before_run"
        ),
    }


__all__ = [
    "ATTENDANCE_AREA_PATHWAY",
    "PreferenceInputs",
    "PreferenceResult",
    "ProgramKey",
    "PromotionMap",
    "auxiliary_directory",
    "build_market_preferences",
    "build_promotion_map",
    "compare_promote_counts",
    "discover_auxiliary_files",
    "load_mr_capacities",
    "prior_tk_source",
    "program_keys",
    "promotion_entitlements",
    "tk_lists_from_cleaned_table",
    "tk_lists_from_prerun",
]

"""Declarative schema for the SFUSD pre-run/post-run/demographics transfers.

The district ships one folder per school year containing three CSVs: a pre-run
request extract (one row per ranked choice), a post-run assignment extract (one
row per student), and a student demographics extract (one row per enrolment
record). This module records what the repository's normalized student, program,
and school tables need from those files so that
``convert_sfusd_transfer.py`` can validate a transfer before converting it and
report every field that the transfer does not carry.

Nothing here fills a gap. A column listed as required must exist; a column
listed as optional produces an explicit entry in the conversion report when it
is absent, and the dependent output column is written blank rather than zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Raw transfer layout
# --------------------------------------------------------------------------- #

#: Canonical school year -> transfer subfolder name.
TRANSFER_YEAR_FOLDERS: dict[str, str] = {
    "2425": "SY24-25",
    "2526": "SY25-26",
    "2627": "SY26-27",
}

#: Canonical school year -> the ``YY-YY`` label the district uses inside the
#: auxiliary filenames, which is spaced and punctuated differently from the
#: subfolder name.
TRANSFER_YEAR_LABELS: dict[str, str] = {
    "2425": "24-25",
    "2526": "25-26",
    "2627": "26-27",
}

#: Subfolder of the transfer holding the district's capacity and
#: auto-promotion files. The district's own spelling, kept verbatim so the
#: path matches the shipped folder.
AUXILIARY_DIRECTORY = "auxillary data"

#: Main Round capacity file, one per year, all three in the same folder.
MR_CAPACITY_FILENAME = (
    "Main Round capacities (including inflation) for SY24-25, SY25-26, "
    "SY26-27 - {label} MR Capacities.csv"
)

#: TK-to-K auto-promotion list, one per year.
AUTOPROMOTION_FILENAME = "TK-to-K autopromotion lists - SY {label}.csv"

#: Rows of prose above each auto-promotion list's header. The district leads
#: every list with a paragraph describing the policy and, in SY25-26, with the
#: excluded-school table as well, so the header row sits at a different line in
#: each file. ``None`` marks a year whose file carries no data rows at all: the
#: SY24-25 "list" is the single sentence "NONE - All TK students had to reapply
#: for K".
AUTOPROMOTION_HEADER_ROWS: dict[str, int | None] = {
    "2425": None,
    "2526": 13,
    "2627": 2,
}

#: Column names of an auto-promotion list, in file order. The files carry no
#: usable header beyond these eight fields; trailing empty columns pad each row
#: out to the width of the prose line above it.
AUTOPROMOTION_COLUMNS: tuple[str, ...] = (
    "tk_school",
    "tk_school_name",
    "tk_grade",
    "tk_pathway",
    "k_school",
    "k_school_name",
    "k_grade",
    "k_pathway",
)

#: Required columns of a Main Round capacity file.
MR_CAPACITY_REQUIRED: tuple[str, ...] = (
    "idSchool",
    "Grade",
    "ProgramCode",
    "TotalSeats",
)

#: Capacity-file columns carried onto the program table as provenance. They are
#: inputs to the validation gates and to the conversion report, never to
#: capacity: see ``PROGRAM_COLUMNS``.
MR_CAPACITY_PROVENANCE: dict[str, str] = {
    "TotalPromoteBeforeRun": "total_promote_before_run",
    "TotalPromoteWithReqBeforeRun": "total_promote_with_request_before_run",
    "FreeSeats": "free_seats",
}

#: The first school year in which a TK student at an Early Education School or
#: the Mission Education Center is auto-promoted to a feeder elementary. The
#: rule was adopted for the 2026-27 TK cohort, so the first class it places in
#: kindergarten enters in SY2027-28. Every year before it had to apply: the
#: 2025-26 EES cohort was not covered, and 259 of them filed a SY26-27
#: application of whom none was promoted. The 24 EES rows in the SY26-27
#: auto-promotion list are therefore parsed and carried but not applied.
EES_FEEDER_FIRST_YEAR = "2728"


@dataclass(frozen=True, slots=True)
class PriorTkSource:
    """Where one run year's prior-year TK preference lists come from.

    A student promoted into kindergarten filed no kindergarten application, so
    the only preferences they ever expressed are the ones they filed for TK a
    year earlier. Those come either from a checked-in cleaned table or from an
    earlier year's pre-run inside the same transfer.
    """

    #: ``Data/Cleaned`` filename whose ``grade == "TK"`` rows hold the lists.
    cleaned_file: str | None = None
    #: Canonical year of the transfer pre-run whose ``Grade == "TK"`` requests
    #: hold them.
    transfer_year: str | None = None


#: Run year -> the TK requests filed the year before.
#:
#: The 2024-25 run reaches back into the checked-in 2023-24 student table. That
#: is deliberate and is not the forward-fill the rest of this conversion
#: avoids: the prohibition is on taking 2023-24 *capacities*, school attributes
#: and the choice estimate into a later year, none of which is a preference
#: list a real family filed in 2023-24 for a TK seat.
PRIOR_TK_SOURCES: dict[str, PriorTkSource] = {
    "2425": PriorTkSource(cleaned_file="student_2324.csv"),
    "2526": PriorTkSource(transfer_year="2425"),
    "2627": PriorTkSource(transfer_year="2526"),
}

#: Transfer file roles and the filename fragments that identify them. The
#: district is inconsistent about capitalisation and separators
#: ("out_SY24-25 student demographics.csv" versus
#: "out_SY25-26_Student Demographics.csv"), so files are matched case-
#: insensitively on a fragment rather than on an exact name.
TRANSFER_FILE_FRAGMENTS: dict[str, str] = {
    "prerun": "prerun",
    "postrun": "postrun",
    "demographics": "demographics",
}

# --------------------------------------------------------------------------- #
# Raw columns
# --------------------------------------------------------------------------- #

#: Pre-run columns without which a transfer cannot be converted at all.
PRERUN_REQUIRED: tuple[str, ...] = (
    "scrambledstudentno",
    "Rank",
    "idSchool",
    "Grade",
    "ProgramCode",
)

#: Pre-run columns that carry per-request metadata. Each maps to one output
#: column or list; a missing one is reported and its output left blank.
PRERUN_OPTIONAL: tuple[str, ...] = (
    "RandomNumber",
    "CohortString",
    "Sibling",
    "CurrentLP",
    "CurrentLPsibling",
    "AAPreK",
    "PreK",
    "AA",
    "MSF",
    "CTIP1",
    "BAY",
    "BRN",
    "B2A",
)

#: Post-run columns without which no assignment outcome can be reconstructed.
POSTRUN_REQUIRED: tuple[str, ...] = (
    "scrambledstudentno",
    "Latitude",
    "Longitude",
)

POSTRUN_OPTIONAL: tuple[str, ...] = (
    "idSchoolAttendance",
    "idNextSchool",
    "NextGrade",
    "NextProgramCode",
    "byPromote",
    "CurrentProgramCode",
    "Rank",
    "Distance",
    "byDesignation",
    "RequestProgramDesignation",
    "studentRandomNumber",
    "IEP_Code",
    "Sped_Pathway",
)

DEMOGRAPHICS_REQUIRED: tuple[str, ...] = ("scrambledstudentno",)

DEMOGRAPHICS_OPTIONAL: tuple[str, ...] = (
    "Race_Ethnicity",
    "HISPANIC_INDICATOR",
    "HLS1__Language_First_Learn",
    "Home_Zip",
    "GRADE",
    "ENR_PATHWAY",
)

# --------------------------------------------------------------------------- #
# Fixed school identities
# --------------------------------------------------------------------------- #

#: ``lowell_ranked`` and ``sota_ranked`` are indicator columns for two specific
#: selective high schools. Both IDs reproduce the checked-in 2023-24 columns
#: exactly (see the validation notes in the module docstring of the converter).
LOWELL_SCHOOL_ID = 697
SOTA_SCHOOL_ID = 815

#: SY26-27 is the first year in which Mission Bay ES appears in real requests,
#: and the district issued it a new identity (1731) rather than reusing the
#: placeholder IDs the repository already knows (909 and 999). The shared table
#: loader keys every Mission Bay policy on {909, 999}
#: (``loaders.tables._MISSION_BAY_SCHOOL_IDS``), so the converter rewrites 1731
#: to the canonical 999 and records the rewrite in the report. Without the
#: rewrite ``include_mission_bay: false`` would silently retain the school.
RAW_SCHOOL_ID_ALIASES: dict[int, int] = {1731: 999}

# --------------------------------------------------------------------------- #
# Per-request priority columns
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PriorityList:
    """A pre-run flag column collected into a per-student output list."""

    output_column: str
    raw_column: str
    #: ``school`` collects ``idSchool``; ``program`` collects the full
    #: ``<school>-<program>-<grade>`` program ID.
    kind: str = "school"


#: Per-request flags the repository stores as per-student lists of school IDs.
#: The raw files carry one sentinel string per flag ("S", "AA", "AAP", ...) on
#: the request rows the flag applies to.
PRIORITY_LISTS: tuple[PriorityList, ...] = (
    PriorityList("sibling", "Sibling"),
    PriorityList("currentlp", "CurrentLP"),
    PriorityList("currentlpsibling", "CurrentLPsibling", kind="program"),
    PriorityList("aaprek", "AAPreK"),
    PriorityList("prek", "PreK"),
    PriorityList("aa", "AA"),
)


@dataclass(frozen=True, slots=True)
class PriorityFlag:
    """A pre-run flag column collected into a per-student 0/1 indicator.

    The three transfer codes were disambiguated empirically rather than from
    district documentation, because the names do not line up one-to-one with
    the repository's column names. In every transfer year:

    * ``BAY`` appears only on grade-6 requests and only for school 858
      (Willie Brown MS), so it is the Bayview-to-Brown priority;
    * ``B2A`` appears only on grade-6 requests and spans every middle school,
      so it is the Bayview-to-all-middle-schools priority;
    * ``BRN`` appears only on grade-9 requests and spans every high school, so
      it is the Brown-middle-school-to-high-school priority.
    """

    output_column: str
    raw_column: str


PRIORITY_FLAGS: tuple[PriorityFlag, ...] = (
    PriorityFlag("bayview_to_brown_ms", "BAY"),
    PriorityFlag("bayview_to_all_ms", "B2A"),
    PriorityFlag("brown_ms_to_hs", "BRN"),
)

# --------------------------------------------------------------------------- #
# Block-level equity indices
# --------------------------------------------------------------------------- #

#: Columns the repository stores per student but that are in fact properties of
#: the student's 2010 Census Block. The transfer carries none of them, so the
#: converter joins them from the existing cleaned student files instead of
#: inventing them. They are *not* identical across every checked-in year: the
#: district recomputed them between 2018-19 and 2021-22, so
#: ``load_block_indices`` walks from the newest year backwards and stops at the
#: first disagreement. Blocks with no index are reported and left blank.
BLOCK_INDEX_COLUMNS: tuple[str, ...] = (
    "freelunch_prob",
    "reducedlunch_prob",
    "FRL Score",
    "N'hood SES Score",
    "Academic Score",
    "AALPI Score",
    "HOCidx1",
    "median_hh_income",
)

#: Cleaned student files the Block index lookup may draw on, oldest first. The
#: lookup is built newest-first and stops at the first vintage disagreement, so
#: adding an older year here extends coverage only if it agrees.
BLOCK_INDEX_SOURCE_YEARS: tuple[str, ...] = (
    "1415",
    "1516",
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
)

# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #

#: Column order of the emitted student tables. It is the 2023-24 layout minus
#: the ``r2_*``/``r4_*`` preference blocks, which these transfers do not carry:
#: each transfer contains exactly one round of requests.
STUDENT_COLUMNS: tuple[str, ...] = (
    "studentno",
    "grade",
    "r1_ranked_idschool",
    "r1_listed_ranks",
    "r1_programs",
    "r1_randomnumber",
    "r1_cohortstring",
    "r1_designation_randomnumber",
    "bayview_to_all_ms",
    "brown_ms_to_hs",
    "bayview_to_brown_ms",
    "requestprogramdesignation",
    "latitude",
    "longitude",
    "previous_pathway",
    "msf",
    "r1_idschool",
    "r1_programcode",
    "r1_rank",
    "r1_isdesignation",
    "r1_distance",
    "idschoolattendance",
    "ctip1",
    "enrolled_idschool",
    "homelang",
    "englprof",
    "sped",
    "resolved_ethnicity",
    "final_school",
    "num_ranked",
    "census_block",
    "freelunch_prob",
    "reducedlunch_prob",
    "census_blockgroup",
    "census_tract",
    "FRL Score",
    "N'hood SES Score",
    "Academic Score",
    "AALPI Score",
    "HOCidx1",
    "sibling",
    "currentlpsibling",
    "currentlp",
    "aaprek",
    "prek",
    "aa",
    "zipcode",
    "median_hh_income",
    "lowell_ranked",
    "sota_ranked",
)

#: Provenance column both emitted student tables carry beyond
#: ``STUDENT_COLUMNS``. From 2024-25 the kindergarten applicant pool is not
#: the set of people who applied: TK students are auto-promoted into K, so
#: they hold a claim on a seat without filing a request. They are in
#: ``student_<year>.csv`` because the market includes them, and this column
#: says which rows came with a Main Round request (1) and which did not (0).
#: The repository's readers ignore it.
MARKET_STUDENT_COLUMNS: tuple[str, ...] = ("mr_applicant",)

#: Per-student columns describing SFUSD's TK-to-K auto-promotion, emitted on
#: both the student and the enrolled table. A student in TK at a school that
#: also runs their pathway at kindergarten is promoted into that program
#: without applying, so the kindergarten market is the main-round applicants
#: *plus* the promoted students, and a promoted applicant holds a claim on one
#: program that no preference list records.
#:
#: ``promote_eligible`` is 1 when the post-run puts the student in TK in a
#: program that is a kindergarten program in the same year's capacity file;
#: ``feeder_school`` and ``feeder_program`` name that program. ``pref_source``
#: says where the emitted preference list came from -- see ``PREF_SOURCES``.
#:
#: ``promote`` says the same thing as the two feeder columns in the shape the
#: priority layer already reads: a list of program IDs the student holds a
#: claim on, exactly like ``currentlpsibling``. It is what a promotion
#: priority weight keys on, and it is not redundant with ``feeder_school``
#: after loading -- the shared loader filters and aliases program lists for
#: ``include_mission_bay``, so a feeder at Mission Bay drops out of ``promote``
#: in a run that excludes the school while the raw feeder columns still name
#: it.
PROMOTION_STUDENT_COLUMNS: tuple[str, ...] = (
    "promote_eligible",
    "promote",
    "feeder_school",
    "feeder_program",
    "pref_source",
)

#: The four values ``pref_source`` can take, in resolution order. It names the
#: *source* consulted for the student's list, not the list's contents: a
#: student found in the prior year's TK requests is ``tk_imputed`` even when
#: none of those requests has a kindergarten counterpart, because that is still
#: where the converter looked. ``feeder_only`` and ``aa_only`` mean no source
#: held the student at all, split by whether they have a feeder to fall back
#: on.
PREF_SOURCES: tuple[str, ...] = (
    "k_list",
    "tk_imputed",
    "feeder_only",
    "aa_only",
)

#: Column order of the emitted program tables. Everything after ``capacity`` up
#: to ``programno`` is provenance the repository's readers ignore.
#:
#: ``capacity`` is gross: at kindergarten it is the capacity file's
#: ``TotalSeats``, and no seat is held back anywhere. The district does hold
#: seats for auto-promoted TK students, but the promotes who win a school
#: elsewhere release theirs back into the same run -- 413-GE had 52 open seats
#: before the SY26-27 run and made 53 choice assignments -- so a table with
#: those seats removed models a market the district never ran. The promotion
#: claim belongs at run time, as a priority boost at the feeder program.
#:
#: ``capacity_source`` names where the number came from, so a capacity that did
#: not come from a district capacity file is visible in the data and not only
#: in the conversion report. ``total_promote_before_run``,
#: ``total_promote_with_request_before_run`` and ``free_seats`` are the
#: district's own counts, carried through as the inputs to the promotion
#: validation gates. ``promotes_seated`` is the converter's own count of
#: students the post-run seats in the program having filed no main-round
#: request for the grade, which is the same quantity observed after the fact.
PROGRAM_COLUMNS: tuple[str, ...] = (
    "program_id",
    "school_id",
    "program_type",
    "capacity",
    "capacity_source",
    "total_promote_before_run",
    "total_promote_with_request_before_run",
    "free_seats",
    "promotes_seated",
    "programno",
    "r1_assigned",
    "r1_first_choice",
)


#: Grades the converter can emit assignment program tables for, and the
#: checked-in capacity/school tables each one borrows from. The transfers carry
#: requests for every grade, but only these three have a district capacity
#: table and a school-coordinate table in the shared data root, which the
#: assignment market requires.
@dataclass(frozen=True, slots=True)
class GradeBundle:
    """Existing capacity and school tables a converted grade builds on."""

    grade: str
    #: ``Data/Cleaned`` filename holding the most recent district capacities.
    #: Used only when ``capacity_grade`` is None: a grade whose capacity comes
    #: from the transfer's own capacity file borrows nothing.
    capacity_reference: str
    #: ``Data/Cleaned`` filenames of the school tables, without and with
    #: Mission Bay.
    schools_standard: str
    schools_mission_bay: str | None = None
    #: Output filename template, formatted with the canonical year.
    programs_template: str = "programs_{year}.csv"
    programs_mission_bay_template: str | None = None
    #: The grade code this bundle takes from the transfer's Main Round capacity
    #: file, or None to fall back to ``capacity_reference``. The capacity file
    #: covers every grade, but only kindergarten has been reconciled against
    #: the district's own promotion counts; grades 6 and 9 hold seats for a
    #: different reason (invisible K-8 continuers at 6, Lowell and SOTA
    #: admissions at 9) and stay on the borrowed table until that is checked.
    capacity_grade: str | None = None
    extra_columns: tuple[str, ...] = field(default_factory=tuple)


GRADE_BUNDLES: dict[str, GradeBundle] = {
    "KG": GradeBundle(
        grade="KG",
        capacity_reference="programs_statusQuo_2324.csv",
        schools_standard="schools_rehauled_2324.csv",
        schools_mission_bay="schools_rehauled_withMissionBay_2324.csv",
        programs_template="programs_{year}.csv",
        programs_mission_bay_template="programs_withMissionBay_{year}.csv",
        capacity_grade="K",
    ),
    "06": GradeBundle(
        grade="06",
        capacity_reference="programs_06_2223.csv",
        schools_standard="schools_rehauled_06_2324.csv",
        programs_template="programs_06_{year}.csv",
        # Grade 6 drives Programs.fix_k8_capacities, which writes r2_capacity.
        extra_columns=("r2_capacity",),
    ),
    "09": GradeBundle(
        grade="09",
        capacity_reference="programs_09_2223.csv",
        schools_standard="schools_rehauled_09_2324.csv",
        programs_template="programs_09_{year}.csv",
    ),
}


__all__ = [
    "AUTOPROMOTION_COLUMNS",
    "AUTOPROMOTION_FILENAME",
    "AUTOPROMOTION_HEADER_ROWS",
    "AUXILIARY_DIRECTORY",
    "BLOCK_INDEX_COLUMNS",
    "BLOCK_INDEX_SOURCE_YEARS",
    "DEMOGRAPHICS_OPTIONAL",
    "DEMOGRAPHICS_REQUIRED",
    "EES_FEEDER_FIRST_YEAR",
    "GRADE_BUNDLES",
    "GradeBundle",
    "LOWELL_SCHOOL_ID",
    "MARKET_STUDENT_COLUMNS",
    "MR_CAPACITY_FILENAME",
    "MR_CAPACITY_PROVENANCE",
    "MR_CAPACITY_REQUIRED",
    "PREF_SOURCES",
    "PRIOR_TK_SOURCES",
    "POSTRUN_OPTIONAL",
    "POSTRUN_REQUIRED",
    "PRERUN_OPTIONAL",
    "PRERUN_REQUIRED",
    "PRIORITY_FLAGS",
    "PRIORITY_LISTS",
    "PROGRAM_COLUMNS",
    "PROMOTION_STUDENT_COLUMNS",
    "PriorTkSource",
    "PriorityFlag",
    "PriorityList",
    "RAW_SCHOOL_ID_ALIASES",
    "SOTA_SCHOOL_ID",
    "STUDENT_COLUMNS",
    "TRANSFER_FILE_FRAGMENTS",
    "TRANSFER_YEAR_FOLDERS",
    "TRANSFER_YEAR_LABELS",
]

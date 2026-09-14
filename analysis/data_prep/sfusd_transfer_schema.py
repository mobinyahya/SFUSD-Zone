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
    "NextProgramCode",
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

#: Column order of the emitted program tables. ``capacity_source`` is an extra
#: provenance column the repository's readers ignore; it exists so a capacity
#: that did not come from a district capacity file is visible in the data and
#: not only in the conversion report.
PROGRAM_COLUMNS: tuple[str, ...] = (
    "program_id",
    "school_id",
    "program_type",
    "capacity",
    "capacity_source",
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
    capacity_reference: str
    #: ``Data/Cleaned`` filenames of the school tables, without and with
    #: Mission Bay.
    schools_standard: str
    schools_mission_bay: str | None = None
    #: Output filename template, formatted with the canonical year.
    programs_template: str = "programs_{year}.csv"
    programs_mission_bay_template: str | None = None
    extra_columns: tuple[str, ...] = field(default_factory=tuple)


GRADE_BUNDLES: dict[str, GradeBundle] = {
    "KG": GradeBundle(
        grade="KG",
        capacity_reference="programs_statusQuo_2324.csv",
        schools_standard="schools_rehauled_2324.csv",
        schools_mission_bay="schools_rehauled_withMissionBay_2324.csv",
        programs_template="programs_{year}.csv",
        programs_mission_bay_template="programs_withMissionBay_{year}.csv",
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
    "BLOCK_INDEX_COLUMNS",
    "BLOCK_INDEX_SOURCE_YEARS",
    "DEMOGRAPHICS_OPTIONAL",
    "DEMOGRAPHICS_REQUIRED",
    "GRADE_BUNDLES",
    "GradeBundle",
    "LOWELL_SCHOOL_ID",
    "POSTRUN_OPTIONAL",
    "POSTRUN_REQUIRED",
    "PRERUN_OPTIONAL",
    "PRERUN_REQUIRED",
    "PRIORITY_FLAGS",
    "PRIORITY_LISTS",
    "PROGRAM_COLUMNS",
    "PriorityFlag",
    "PriorityList",
    "RAW_SCHOOL_ID_ALIASES",
    "SOTA_SCHOOL_ID",
    "STUDENT_COLUMNS",
    "TRANSFER_FILE_FRAGMENTS",
    "TRANSFER_YEAR_FOLDERS",
]

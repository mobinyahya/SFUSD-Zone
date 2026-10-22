"""Tests for the SFUSD transfer converter.

These run on a synthetic three-file transfer, so they need no access to the
shared data root. They pin the behaviour that is easy to get quietly wrong:
the ranked-list pivot, the per-request priority flags, the Mission Bay school
alias, and above all that a gap fails loudly under the default policy instead
of being filled.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from analysis.data_prep import convert_sfusd_transfer as converter
from analysis.data_prep.convert_sfusd_transfer import (
    Report,
    TransferGapError,
    build_program_table,
    build_student_table,
    discover_transfer_files,
)
from analysis.data_prep.sfusd_transfer_schema import (
    AUXILIARY_DIRECTORY,
    GRADE_BUNDLES,
    PROMOTION_STUDENT_COLUMNS,
)
from analysis.data_prep.tk_promotion import (
    build_promotion_map,
    discover_auxiliary_files,
    load_mr_capacities,
    program_keys,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
PRERUN_ROWS = [
    # Student 1: three KG choices with a gap in the listed ranks (rank 3
    # withdrawn), sibling priority at the second, CTIP1 throughout. Currently
    # in TK at 413-GE, which is the first choice, so the promotion claim is
    # already on the list and appending it would duplicate a choice.
    ("S000000001", 1, 413, "K", "GE", 0.10, "CT1;", None, "CT1"),
    ("S000000001", 2, 420, "K", "GE", 0.20, "CT1;S;", "S", "CT1"),
    ("S000000001", 4, 999, "K", "GE", 0.30, "CT1;", None, "CT1"),
    # Student 2: one KG choice at Mission Bay's new transfer ID, but currently
    # in TK at 420-GE, so the feeder has to be appended.
    ("S000000002", 1, 1731, "K", "GE", 0.40, None, None, None),
    # Student 3: grade 6, so it exercises a second grade bundle.
    ("S000000003", 1, 404, "6", "GE", 0.50, None, None, None),
    # Student 5: a KG applicant who is not in TK at all, so no promotion.
    ("S000000005", 1, 413, "K", "GE", 0.60, None, None, None),
]

PRERUN_COLUMNS = [
    "scrambledstudentno",
    "Rank",
    "idSchool",
    "Grade",
    "ProgramCode",
    "RandomNumber",
    "CohortString",
    "Sibling",
    "CTIP1",
]

POSTRUN_ROWS = [
    (
        "S000000001",
        413,
        413,
        "GE",
        "K",
        37.7830,
        -122.4823,
        1,
        0.9,
        413,
        "TK",
        "GE",
        1.2,
        0,
    ),
    (
        "S000000002",
        420,
        999,
        "GE",
        "K",
        37.7698,
        -122.3945,
        1,
        0.8,
        420,
        "TK",
        "GE",
        0.4,
        0,
    ),
    (
        "S000000003",
        None,
        404,
        "GE",
        "06",
        37.7500,
        -122.4000,
        1,
        0.7,
        404,
        "05",
        "GE",
        2.1,
        0,
    ),
    # Student 4 is a TK student auto-promoted into KG: seated at 413 with no
    # pre-run request anywhere, which is the 2025-26 and 2026-27 shape. They
    # filed TK requests the year before, so their list is imputed from those.
    (
        "S000000004",
        420,
        413,
        "GE",
        "K",
        37.7841,
        -122.4810,
        None,
        0.6,
        413,
        "TK",
        "GE",
        0.9,
        1,
    ),
    # Student 5 applied, lost, and was seated at their second choice. They were
    # in PK, not TK, so they hold no promotion claim.
    (
        "S000000005",
        420,
        420,
        "GE",
        "K",
        37.7530,
        -122.4380,
        None,
        0.5,
        None,
        "PK",
        None,
        0.7,
        0,
    ),
    # Student 6 was in TK at an Early Education School, which runs no
    # kindergarten: no feeder, and no prior-year TK list either, so their only
    # option is their attendance area.
    (
        "S000000006",
        420,
        420,
        "GE",
        "K",
        37.7521,
        -122.4371,
        None,
        0.4,
        903,
        "TK",
        "GE",
        1.1,
        0,
    ),
    # Student 7 was in TK at 420-GE and filed nothing anywhere, so the list is
    # built from the feeder and the attendance area.
    (
        "S000000007",
        413,
        420,
        "GE",
        "K",
        37.7802,
        -122.4799,
        None,
        0.3,
        420,
        "TK",
        "GE",
        1.4,
        1,
    ),
]

POSTRUN_COLUMNS = [
    "scrambledstudentno",
    "idSchoolAttendance",
    "idNextSchool",
    "NextProgramCode",
    "NextGrade",
    "Latitude",
    "Longitude",
    "Rank",
    "studentRandomNumber",
    "idCurrentSchool",
    "CurrentGrade",
    "CurrentProgramCode",
    "Distance",
    "byPromote",
]

DEMOGRAPHICS_ROWS = [
    ("S000000001", "Chinese", "N", "Cantonese", 94121),
    ("S000000002", None, "Y", "Spanish", 94158),
    # A row that cannot be attributed to anyone.
    (None, "White", "N", "English", 94110),
]

DEMOGRAPHICS_COLUMNS = [
    "scrambledstudentno",
    "Race_Ethnicity",
    "HISPANIC_INDICATOR",
    "HLS1__Language_First_Learn",
    "Home_Zip",
]

# The prior year's transfer, which SY26-27 reads for the TK requests its
# promoted students filed. Student 4 ranked two TK programs; one of them
# (903-GE, an Early Education School) has no kindergarten counterpart and is
# dropped, and the other does.
PRIOR_PRERUN_ROWS = [
    ("S000000004", 1, 903, "TK", "GE", 0.55),
    ("S000000004", 2, 420, "TK", "GE", 0.65),
]
PRIOR_PRERUN_COLUMNS = [
    "scrambledstudentno",
    "Rank",
    "idSchool",
    "Grade",
    "ProgramCode",
    "RandomNumber",
]

#: One Main Round capacity row per school, grade, and program. 420-SE opens
#: with no seats, which is a closed program rather than an absent one; 903-GE
#: exists at TK only, which is what makes an Early Education School an EES.
CAPACITY_ROWS = [
    (413, "Alamo ES", "K", "GE", 64, 40, 2, 3, 1),
    (420, "Alvarado ES", "K", "GE", 88, 70, 5, 2, 1),
    (420, "Alvarado ES", "K", "SE", 0, 0, 0, 0, 0),
    (999, "Mission Bay ES", "K", "GE", 20, 20, 0, 0, 0),
    (413, "Alamo ES", "TK", "GE", 48, 48, 0, 0, 0),
    (903, "Argonne EES", "TK", "GE", 24, 24, 0, 0, 0),
    (404, "Aptos MS", "6", "GE", 271, 271, 0, 0, 0),
]
CAPACITY_COLUMNS = [
    "idSchool",
    "SchoolName",
    "Grade",
    "ProgramCode",
    "TotalSeats",
    "OpenSeatsPreRun",
    "FreeSeats",
    "TotalPromoteBeforeRun",
    "TotalPromoteWithReqBeforeRun",
]

#: The auto-promotion list, prose and all. The header sits three lines down,
#: which is where ``AUTOPROMOTION_HEADER_ROWS["2627"] == 2`` puts it, and the
#: first data row is an Early Education School feeder that must not be applied.
AUTOPROMOTION_LINES = [
    "TK students at all elementary and K-8 schools are eligible...,,,,,,,",
    "These TK school-grade-programs feed into...,,,,these K school-grade-programs,,,",
    "TK school #,TK school name,TK grade,TK pathway,K school #,K school "
    "name,K grade,K pathway",
    "903,Argonne EES,TK,GE644,644,Jefferson ES,K,GE",
    "413,Alamo ES,TK,GE,413,Alamo ES,K,GE",
    "420,Alvarado ES,TK,GE,420,Alvarado ES,K,GE",
]


def _write_year(folder: Path, label: str, prerun, postrun, demographics) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    prerun.to_csv(folder / f"out_{label}_PreRun.csv", index=False)
    postrun.to_csv(folder / f"out_{label}_PostRun.csv", index=False)
    demographics.to_csv(folder / f"out_{label}_Student Demographics.csv", index=False)


@pytest.fixture
def transfer(tmp_path: Path) -> Path:
    _write_year(
        tmp_path / "SY26-27",
        "SY26-27",
        pd.DataFrame(PRERUN_ROWS, columns=PRERUN_COLUMNS),
        pd.DataFrame(POSTRUN_ROWS, columns=POSTRUN_COLUMNS),
        pd.DataFrame(DEMOGRAPHICS_ROWS, columns=DEMOGRAPHICS_COLUMNS),
    )
    # SY26-27 imputes preferences from the SY25-26 TK requests, so the prior
    # year has to be present in the same transfer.
    _write_year(
        tmp_path / "SY25-26",
        "SY25-26",
        pd.DataFrame(PRIOR_PRERUN_ROWS, columns=PRIOR_PRERUN_COLUMNS),
        pd.DataFrame(
            [("S000000004", 37.78, -122.48)],
            columns=["scrambledstudentno", "Latitude", "Longitude"],
        ),
        pd.DataFrame([("S000000004",)], columns=["scrambledstudentno"]),
    )

    auxiliary = tmp_path / AUXILIARY_DIRECTORY
    auxiliary.mkdir()
    paths = discover_auxiliary_files(tmp_path, "2627")
    pd.DataFrame(CAPACITY_ROWS, columns=CAPACITY_COLUMNS).to_csv(
        paths["capacities"], index=False
    )
    paths["autopromotion"].write_text(
        "\n".join(AUTOPROMOTION_LINES) + "\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def capacities(transfer: Path) -> pd.DataFrame:
    return load_mr_capacities(discover_auxiliary_files(transfer, "2627")["capacities"])


@pytest.fixture
def promotion(transfer: Path):
    return build_promotion_map(
        discover_auxiliary_files(transfer, "2627")["autopromotion"], "2627"
    )


@pytest.fixture
def frames(transfer: Path) -> dict[str, pd.DataFrame]:
    paths = discover_transfer_files(transfer, "2627")
    return {
        role: converter._clean_nulls(pd.read_csv(path, low_memory=False))
        for role, path in paths.items()
    }


def _students(frames, *, gaps: str = "fill-and-report") -> pd.DataFrame:
    report = Report(year="2627", transfer="synthetic")
    frame = build_student_table(
        frames["prerun"],
        frames["postrun"],
        frames["demographics"],
        report=report,
        gaps=gaps,
    )
    return frame.set_index("studentno")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_discovery_matches_the_district_inconsistent_filenames(transfer):
    paths = discover_transfer_files(transfer, "2627")
    assert paths["prerun"].name == "out_SY26-27_PreRun.csv"
    assert paths["postrun"].name == "out_SY26-27_PostRun.csv"
    # The demographics file uses a space, not an underscore, in some years.
    assert paths["demographics"].name == "out_SY26-27_Student Demographics.csv"


def test_discovery_names_the_missing_year_rather_than_guessing(transfer):
    with pytest.raises(TransferGapError, match="not a known transfer folder"):
        discover_transfer_files(transfer, "2728")


def test_discovery_fails_when_a_role_is_absent(transfer):
    (transfer / "SY26-27" / "out_SY26-27_PostRun.csv").unlink()
    with pytest.raises(TransferGapError, match="no postrun CSV"):
        discover_transfer_files(transfer, "2627")


# --------------------------------------------------------------------------- #
# Student pivot
# --------------------------------------------------------------------------- #
def test_ranked_lists_preserve_order_and_non_contiguous_listed_ranks(frames):
    students = _students(frames)
    assert students.loc[1, "r1_ranked_idschool"] == "[413, 420, 999]"
    assert students.loc[1, "r1_programs"] == "['GE', 'GE', 'GE']"
    # Rank 3 was withdrawn: the gap is kept rather than renumbered, because the
    # loader aligns listed ranks to the school list.
    assert students.loc[1, "r1_listed_ranks"] == "[1, 2, 4]"
    assert students.loc[1, "num_ranked"] == 3


def test_student_identity_drops_the_S_prefix(frames):
    students = _students(frames)
    assert sorted(students.index) == [1, 2, 3, 5]


def test_malformed_student_identity_is_an_error_not_a_dropped_row(frames):
    frames["prerun"].loc[0, "scrambledstudentno"] = "X12345"
    with pytest.raises(TransferGapError, match="outside the expected"):
        _students(frames)


def test_mission_bay_new_transfer_id_is_rewritten_to_the_canonical_id(frames):
    students = _students(frames)
    # 1731 is Mission Bay ES in the SY26-27 transfer; the shared table loader
    # keys every Mission Bay policy on {909, 999}, so it must arrive as 999.
    assert students.loc[2, "r1_ranked_idschool"] == "[999]"
    assert int(students.loc[2, "enrolled_idschool"]) == 999


def test_per_request_priority_flags_become_per_student_school_lists(frames):
    students = _students(frames)
    assert students.loc[1, "sibling"] == "[420]"
    assert students.loc[2, "sibling"] == "[]"
    assert students.loc[1, "ctip1"] == 1
    assert students.loc[2, "ctip1"] == 0


def test_cohort_strings_stay_aligned_with_the_ranked_list(frames):
    students = _students(frames)
    assert students.loc[1, "r1_cohortstring"] == "['CT1;', 'CT1;S;', 'CT1;']"
    assert students.loc[2, "r1_cohortstring"] == "['']"


def test_selective_school_indicators_read_the_ranked_list(frames):
    students = _students(frames)
    assert set(students["lowell_ranked"]) == {0}
    assert set(students["sota_ranked"]) == {0}


def test_hispanic_indicator_fills_only_a_blank_ethnicity(frames):
    students = _students(frames)
    assert students.loc[1, "resolved_ethnicity"] == "Chinese"
    assert students.loc[2, "resolved_ethnicity"] == "Hispanic"


def test_unattributable_demographics_rows_are_reported_not_matched(frames):
    report = Report(year="2627", transfer="synthetic")
    build_student_table(
        frames["prerun"],
        frames["postrun"],
        frames["demographics"],
        report=report,
        gaps="fill-and-report",
    )
    dropped = report.missing_values["demographics_unattributable_rows"]
    assert dropped["rows_dropped"] == 1


def test_absent_postrun_column_leaves_the_output_blank_and_is_reported(frames):
    frames["postrun"] = frames["postrun"].drop(columns=["Distance"])
    report = Report(year="2627", transfer="synthetic")
    frame = build_student_table(
        frames["prerun"],
        frames["postrun"],
        frames["demographics"],
        report=report,
        gaps="fill-and-report",
    )
    assert "r1_distance" in report.blank_output_columns
    assert "r1_distance" not in frame.columns


def test_a_student_applying_for_two_grades_is_an_error(frames):
    frames["prerun"].loc[1, "Grade"] = "6"
    with pytest.raises(TransferGapError, match="several grades"):
        _students(frames)


def test_repeated_rank_within_a_student_is_an_error(frames):
    frames["prerun"].loc[1, "Rank"] = 1
    with pytest.raises(TransferGapError, match="repeated ranks"):
        _students(frames)


def test_multiple_msf_schools_fail_by_default_and_report_when_allowed(frames):
    frames["prerun"]["MSF"] = [None] * len(frames["prerun"])
    frames["prerun"].loc[0, "MSF"] = "MSF"
    frames["prerun"].loc[1, "MSF"] = "MSF"
    with pytest.raises(TransferGapError, match="several MSF schools"):
        _students(frames, gaps="fail")

    report = Report(year="2627", transfer="synthetic")
    frame = build_student_table(
        frames["prerun"],
        frames["postrun"],
        frames["demographics"],
        report=report,
        gaps="fill-and-report",
    )
    kinds = {item["kind"] for item in report.substitutions}
    assert "msf_multiple_schools" in kinds
    # The highest-ranked MSF school is the one kept.
    assert int(frame.set_index("studentno").loc[1, "msf"]) == 413


# --------------------------------------------------------------------------- #
# Program tables
# --------------------------------------------------------------------------- #
@pytest.fixture
def cleaned_dir(tmp_path: Path) -> Path:
    """The checked-in tables the conversion borrows from.

    Kindergarten no longer borrows a capacity table -- it reads the transfer's
    own capacity file -- but it still borrows the school coordinates, because
    the transfer carries none. Mission Bay is placeable in one school table
    only, which is what ``include_mission_bay`` selects between.
    """
    folder = tmp_path / "cleaned"
    folder.mkdir()
    bundle = GRADE_BUNDLES["KG"]
    pd.DataFrame(
        {
            "school_id": [413, 420],
            "lat": [37.783, 37.7537],
            "lon": [-122.4823, -122.4382],
        }
    ).to_csv(folder / bundle.schools_standard, index=False)
    pd.DataFrame(
        {
            "school_id": [413, 420, 999],
            "lat": [37.783, 37.7537, 37.7698],
            "lon": [-122.4823, -122.4382, -122.3945],
        }
    ).to_csv(folder / bundle.schools_mission_bay, index=False)
    return folder


def _programs(
    frames,
    cleaned_dir,
    capacities,
    *,
    gaps: str = "fill-and-report",
    report: Report | None = None,
    include_mission_bay: bool = True,
):
    return build_program_table(
        frames["prerun"],
        frames["postrun"],
        GRADE_BUNDLES["KG"],
        cleaned_dir=cleaned_dir,
        include_mission_bay=include_mission_bay,
        gaps=gaps,
        report=report or Report(year="2627", transfer="synthetic"),
        capacities=capacities,
    )


def test_kindergarten_capacity_is_the_capacity_file_total_not_the_open_seats(
    frames, cleaned_dir, capacities
):
    programs = _programs(frames, cleaned_dir, capacities).set_index("program_id")
    # TotalSeats, not OpenSeatsPreRun (40): the seats the file holds back for
    # auto-promotion re-enter the same run when a promote wins elsewhere.
    assert programs.loc["413-GE-KG", "capacity"] == 64
    assert "TotalSeats" in programs.loc["413-GE-KG", "capacity_source"]
    assert programs.loc["999-GE-KG", "capacity"] == 20


def test_district_promote_counts_ride_along_as_provenance(
    frames, cleaned_dir, capacities
):
    programs = _programs(frames, cleaned_dir, capacities).set_index("program_id")
    assert programs.loc["413-GE-KG", "total_promote_before_run"] == 3
    assert programs.loc["413-GE-KG", "total_promote_with_request_before_run"] == 1
    assert programs.loc["413-GE-KG", "free_seats"] == 2


def test_a_program_the_file_opens_with_no_seats_is_kept(
    frames, cleaned_dir, capacities
):
    # A closed program is not an absent one, and a TK student can sit in it.
    programs = _programs(frames, cleaned_dir, capacities).set_index("program_id")
    assert programs.loc["420-SE-KG", "capacity"] == 0


def test_the_row_set_is_the_capacity_file_not_the_requests(
    frames, cleaned_dir, capacities
):
    programs = _programs(frames, cleaned_dir, capacities)
    # 420-SE is in the capacity file and nobody ranked it; both TK rows and the
    # grade-6 row belong to other grades.
    assert sorted(programs["program_id"]) == [
        "413-GE-KG",
        "420-GE-KG",
        "420-SE-KG",
        "999-GE-KG",
    ]


def test_mission_bay_is_dropped_from_the_standard_variant(
    frames, cleaned_dir, capacities
):
    # 999 has coordinates only in the Mission Bay school table, so the standard
    # variant must not carry it -- a program assignment cannot place.
    programs = _programs(frames, cleaned_dir, capacities, include_mission_bay=False)
    assert "999-GE-KG" not in set(programs["program_id"])


def test_a_capacity_program_at_an_unplaceable_school_is_dropped_and_reported(
    frames, cleaned_dir, capacities
):
    extra = pd.DataFrame(
        [(466, "Independence HS", "KG", "GE", 30, pd.NA, pd.NA, pd.NA)],
        columns=[
            "school_id",
            "school_name",
            "grade",
            "program_type",
            "capacity",
            "total_promote_before_run",
            "total_promote_with_request_before_run",
            "free_seats",
        ],
    )
    report = Report(year="2627", transfer="synthetic")
    programs = _programs(
        frames,
        cleaned_dir,
        pd.concat([capacities, extra], ignore_index=True),
        report=report,
    )
    assert "466-GE-KG" not in set(programs["program_id"])
    finding = next(
        item
        for item in report.substitutions
        if item["kind"] == "capacity_program_unlocatable_school"
    )
    assert finding["schools"] == [466]
    assert finding["seats"] == 30


def test_program_numbers_are_unique_and_contiguous(frames, cleaned_dir, capacities):
    programs = _programs(frames, cleaned_dir, capacities)
    assert sorted(programs["programno"]) == list(range(1, len(programs) + 1))
    assert not programs["program_id"].duplicated().any()


def test_promotes_seated_is_counted_but_never_netted_out(
    frames, cleaned_dir, capacities
):
    programs = _programs(frames, cleaned_dir, capacities).set_index("program_id")
    # Student 4 is seated at 413 without a request; students 6 and 7 at 420.
    assert programs.loc["413-GE-KG", "promotes_seated"] == 1
    assert programs.loc["420-GE-KG", "promotes_seated"] == 2
    # The capacity is untouched by either count.
    assert programs.loc["413-GE-KG", "capacity"] == 64
    assert programs.loc["420-GE-KG", "capacity"] == 88


def test_a_grade_the_capacity_file_lacks_emits_no_table(
    frames, cleaned_dir, capacities
):
    report = Report(year="2627", transfer="synthetic")
    result = build_program_table(
        frames["prerun"],
        frames["postrun"],
        GRADE_BUNDLES["09"],
        cleaned_dir=cleaned_dir,
        include_mission_bay=False,
        gaps="fill-and-report",
        report=report,
        capacities=capacities,
    )
    assert result is None
    assert any("grade 09" in note for note in report.notes)


def test_grade_six_still_borrows_its_capacity_and_fails_loudly_without_one(
    frames, cleaned_dir, capacities
):
    bundle = GRADE_BUNDLES["06"]
    assert bundle.capacity_grade is None
    pd.DataFrame({"school_id": [404], "lat": [37.75], "lon": [-122.40]}).to_csv(
        cleaned_dir / bundle.schools_standard, index=False
    )
    pd.DataFrame(
        {
            "program_id": ["404-GE-06"],
            "school_id": [404],
            "program_type": ["GE"],
            "capacity": [271],
        }
    ).to_csv(cleaned_dir / bundle.capacity_reference, index=False)

    programs = build_program_table(
        frames["prerun"],
        frames["postrun"],
        bundle,
        cleaned_dir=cleaned_dir,
        include_mission_bay=False,
        gaps="fail",
        report=Report(year="2627", transfer="synthetic"),
        capacities=capacities,
    ).set_index("program_id")
    assert programs.loc["404-GE-06", "capacity"] == 271
    assert programs.loc["404-GE-06", "capacity_source"] == bundle.capacity_reference
    # The capacity file does carry grade 6, but it has not been reconciled, so
    # the borrowed table is still the source and its gaps still fail.
    assert pd.isna(programs.loc["404-GE-06", "total_promote_before_run"])


def test_unlocatable_school_requests_and_programs_are_dropped_together(
    frames, cleaned_dir
):
    # Add a grade-6 request for a school the grade-6 school table cannot place.
    extra = frames["prerun"].iloc[[4]].copy()
    extra["Rank"] = 2
    extra["idSchool"] = 466
    extra["SchoolName"] = "Independence HS"
    frames["prerun"] = pd.concat([frames["prerun"], extra], ignore_index=True)

    bundle = GRADE_BUNDLES["06"]
    pd.DataFrame(
        {
            "program_id": ["404-GE-06"],
            "school_id": [404],
            "program_type": ["GE"],
            "capacity": [271],
        }
    ).to_csv(cleaned_dir / bundle.capacity_reference, index=False)
    pd.DataFrame({"school_id": [404], "lat": [37.75], "lon": [-122.40]}).to_csv(
        cleaned_dir / bundle.schools_standard, index=False
    )

    with pytest.raises(TransferGapError, match="no coordinates"):
        converter.drop_unlocatable_requests(
            frames["prerun"],
            cleaned_dir=cleaned_dir,
            gaps="fail",
            report=Report(year="2627", transfer="synthetic"),
        )

    report = Report(year="2627", transfer="synthetic")
    filtered = converter.drop_unlocatable_requests(
        frames["prerun"],
        cleaned_dir=cleaned_dir,
        gaps="fill-and-report",
        report=report,
    )
    # The request is gone, so no student list can reference the dropped program.
    assert 466 not in set(filtered["idSchool"])
    programs = build_program_table(
        filtered,
        frames["postrun"],
        bundle,
        cleaned_dir=cleaned_dir,
        include_mission_bay=False,
        gaps="fill-and-report",
        report=report,
    )
    assert "466-GE-06" not in set(programs["program_id"])
    finding = next(
        item for item in report.substitutions if item["kind"] == "unlocatable_school"
    )
    assert finding["findings"][0]["schools"] == [466]
    assert finding["findings"][0]["requests"] == 1


def test_a_grade_absent_from_the_transfer_emits_no_table(frames, cleaned_dir):
    report = Report(year="2627", transfer="synthetic")
    result = build_program_table(
        frames["prerun"],
        frames["postrun"],
        GRADE_BUNDLES["09"],
        cleaned_dir=cleaned_dir,
        include_mission_bay=False,
        gaps="fill-and-report",
        report=report,
    )
    assert result is None
    assert any("grade 09" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# Enrolled table
# --------------------------------------------------------------------------- #
@pytest.fixture
def no_geography(monkeypatch):
    """Skip the Census join, which needs the shared data root.

    ``add_market_students`` derives geography for the promoted students it
    adds, and that reads the district's Block geometry. These tests are about
    which rows and which columns the tables hold, so the join is stubbed out.
    """

    def stub(students, report, *, scenario_name="legacy", report_label="geography"):
        frame = students.copy()
        for column in ("census_block", "census_blockgroup", "census_tract"):
            frame[column] = pd.NA
        return frame

    monkeypatch.setattr(converter, "attach_geography", stub)


def _tables(frames, cleaned_dir, capacities, promotion, transfer, report=None):
    """Run the whole kindergarten pipeline the way ``convert_year`` does.

    Returns a namespace with ``.students``, ``.enrolled``, ``.table`` (the
    per-student promotion columns and preference lists), ``.market`` (who is
    in it and what they may claim) and ``.report``. The order the pipeline
    runs in is the point of several tests below: the market's non-applicants
    join the student table *before* their preference lists are built, and the
    enrolled table is then a filter of the student table rather than a second
    construction.
    """
    report = report or Report(year="2627", transfer="synthetic")
    students = build_student_table(
        frames["prerun"],
        frames["postrun"],
        frames["demographics"],
        report=report,
        gaps="fill-and-report",
    )
    market = converter.define_market(
        students,
        frames["prerun"],
        frames["postrun"],
        grade="KG",
        capacities=capacities,
        cleaned_dir=cleaned_dir,
        report=report,
    )
    students = converter.add_market_students(
        students,
        frames["postrun"],
        frames["demographics"],
        pd.DataFrame(index=pd.Index([], name="census_block")),
        market=market,
        report=report,
        scenario_name="legacy",
    )
    table = converter.build_market_table(
        students,
        frames["postrun"],
        market=market,
        year="2627",
        capacities=capacities,
        promotion=promotion,
        prior_tk=converter.load_prior_tk_lists(
            "2627", transfer=transfer, cleaned_dir=cleaned_dir, report=report
        ),
        report=report,
    )
    students = converter.apply_market_columns(students, table, grade="KG")
    enrolled = converter.build_enrolled_table(
        students,
        frames["postrun"],
        frames["demographics"],
        market=market,
        report=report,
    )
    return SimpleNamespace(
        students=students,
        enrolled=enrolled,
        table=table,
        market=market,
        report=report,
    )


def test_the_student_table_holds_the_whole_market_not_just_the_applicants(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    by_id = students.set_index("studentno")
    # 1, 2 and 5 applied for KG; 4, 6 and 7 are in the market without applying.
    # 3 is a grade-6 applicant and is untouched.
    assert sorted(by_id.loc[by_id["grade"] == "KG"].index) == [1, 2, 4, 5, 6, 7]
    assert by_id.loc[[1, 2, 5], "mr_applicant"].tolist() == [1, 1, 1]
    assert by_id.loc[[4, 6, 7], "mr_applicant"].tolist() == [0, 0, 0]
    # A grade with no promotion handling is applicants only.
    assert by_id.loc[3, "mr_applicant"] == 1


def test_the_enrolled_table_is_a_subset_of_the_student_table(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    enrolled = built.enrolled
    assert set(enrolled["studentno"]) <= set(students["studentno"])
    assert len(enrolled) <= int((students["grade"] == "KG").sum())
    # Every market student here takes a seat, so the two coincide.
    assert sorted(enrolled["studentno"]) == [1, 2, 4, 5, 6, 7]


def test_the_enrolled_table_keeps_only_students_with_an_enrolment_record(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    # The demographics extract records where each student enrolled for the
    # fall. Only a record at kindergarten, at a real school, counts: student 1
    # is at 899 ("Central Enrollment", enrolled nowhere), student 4 is recorded
    # at TK, and 6 and 7 have no record, so the post-run seated all four and
    # none of them enrolled. Student 2 moved after the Main Round from their
    # 999-GE seat to 420-SE; student 5 is recorded at Mission Bay under the
    # district's new id 1731.
    frames["demographics"] = pd.DataFrame(
        {
            "scrambledstudentno": [f"S00000000{n}" for n in (1, 2, 4, 5)],
            "Race_Ethnicity": ["Chinese", None, None, None],
            "HISPANIC_INDICATOR": ["N", "Y", "N", "N"],
            "HLS1__Language_First_Learn": ["Cantonese", "Spanish", None, None],
            "Home_Zip": [94121, 94158, None, None],
            "SCHOOL_CODE": [899, 420, 413, 1731],
            "GRADE": ["K", "K", "TK", "K"],
            "ENR_PATHWAY": ["GE", "SE", "GE", "GE"],
        }
    )
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students.set_index("studentno")
    enrolled = built.enrolled.set_index("studentno")
    report = built.report

    assert sorted(enrolled.index) == [2, 5]
    assert {1, 4, 6, 7} <= set(students.index)
    assert enrolled.loc[2, ["enrolled_idschool", "enrolled_programcode"]].tolist() == [
        420,
        "SE",
    ]
    assert enrolled.loc[5, "enrolled_idschool"] == 999
    # The student table, and the Main Round outcome columns of both tables,
    # stay the post-run seat.
    assert students.loc[2, "enrolled_idschool"] == 999
    assert enrolled.loc[2, "final_school"] == students.loc[2, "final_school"]
    assert report.row_counts["enrolled_KG_seated_at_899"] == 1
    assert report.row_counts["enrolled_KG_seated_no_record"] == 2
    assert report.row_counts["enrolled_KG_seated_record_at_other_grade"] == 1
    assert report.row_counts["enrolled_KG_enrolled"] == 2

    recorded = converter.enrolled_at_grade_students(
        frames["demographics"], "KG", report
    )
    assert recorded == {2, 5}
    converter.validate_market(
        built.table,
        built.students,
        built.enrolled,
        frames["postrun"],
        market=built.market,
        report=report,
        recorded=recorded,
    )
    # Without the record every seated student is expected.
    with pytest.raises(TransferGapError, match="does not hold"):
        converter.validate_market(
            built.table,
            built.students,
            built.enrolled,
            frames["postrun"],
            market=built.market,
            report=report,
        )


def test_a_market_student_with_no_seat_is_in_the_applicant_pool_only(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    # Drop student 2's post-run seat by moving them to another grade: they
    # applied for kindergarten, so they stay an applicant, but the run seated
    # them nowhere at kindergarten.
    frames["postrun"].loc[1, "NextGrade"] = "01"
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    enrolled = built.enrolled
    report = built.report
    assert 2 in set(students["studentno"])
    assert 2 not in set(enrolled["studentno"])
    assert report.row_counts["enrolled_KG_market_students_with_no_seat"] == 1


def test_a_promoted_student_gets_the_post_run_columns_of_any_other_student(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    promoted = students.set_index("studentno").loc[4]
    assert promoted["grade"] == "KG"
    assert promoted["mr_applicant"] == 0
    # The outcome columns come from the post-run, so the seat is visible.
    assert int(promoted["enrolled_idschool"]) == 413
    assert int(promoted["idschoolattendance"]) == 420
    # ctip1 is a per-request pre-run flag, so it cannot be known for them.
    assert pd.isna(promoted["ctip1"])


def test_without_nextgrade_only_the_students_with_a_claim_stay_in_the_market(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    """A truncated post-run loses the seat, not the entitlement.

    Promotion eligibility reads ``CurrentGrade`` and the current program, so
    it survives a post-run with no ``NextGrade``: students 4 and 7 are still
    identified. What is lost is the other route into the market -- a seat at
    the grade with no request -- so student 6, who was in TK at an Early
    Education School and holds no claim, drops out. And with nobody
    identifiable as seated, the enrolled table cannot be a filter of the
    student table and falls back to all of it.
    """
    frames["postrun"] = frames["postrun"].drop(columns="NextGrade")
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    enrolled = built.enrolled
    report = built.report
    by_id = students.set_index("studentno")
    assert sorted(by_id.loc[by_id["grade"] == "KG"].index) == [1, 2, 4, 5, 7]
    assert 6 not in set(students["studentno"])
    assert sorted(enrolled["studentno"]) == [1, 2, 4, 5, 7]
    assert any("no NextGrade column" in note for note in report.notes)


# --------------------------------------------------------------------------- #
# TK-to-K promotion
# --------------------------------------------------------------------------- #
def test_the_promotion_map_separates_same_site_rows_from_ees_feeders(promotion):
    assert promotion.same_site == {(413, "GE"): (413, "GE"), (420, "GE"): (420, "GE")}
    # The EES row's TK pathway code is compound and encodes the destination.
    assert promotion.ees_feeder == {(903, "GE644"): (644, "GE")}


def test_the_ees_feeder_rule_is_parsed_but_not_yet_applied(promotion):
    # The rule covers the 2026-27 TK cohort, whose first kindergarten class
    # enters in SY2027-28, so a SY26-27 run must not use it.
    assert not promotion.ees_active
    assert (903, "GE644") not in promotion.active_feeders()
    assert promotion.active_feeders() == promotion.same_site


def test_a_year_with_no_map_parses_to_an_empty_one(tmp_path):
    # SY24-25's list is the one sentence "NONE - All TK students had to reapply
    # for K". That is a fact about the year, not a malformed file.
    path = tmp_path / "none.csv"
    path.write_text("NONE - All TK students had to reapply for K\n", encoding="utf-8")
    empty = build_promotion_map(path, "2425")
    assert empty.empty
    assert empty.active_feeders() == {}


def test_closed_programs_still_count_as_kindergarten_programs(capacities):
    # 420-SE opens with no seats. A TK student sitting in it is still a TK
    # student in a kindergarten program, so eligibility must see it.
    assert (420, "SE") in program_keys(capacities, "KG")
    assert (903, "GE") not in program_keys(capacities, "KG")


def test_the_market_is_the_applicants_plus_the_students_seated_without_one(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    # 1, 2 and 5 applied for KG; 4, 6 and 7 were seated without applying.
    # 3 is a grade-6 applicant and is not in the kindergarten market.
    assert sorted(market.index) == [1, 2, 4, 5, 6, 7]


def test_pref_source_names_the_source_that_answered(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    assert market.loc[1, "pref_source"] == "k_list"
    assert market.loc[2, "pref_source"] == "k_list"
    assert market.loc[5, "pref_source"] == "k_list"
    # Student 4 filed TK requests last year.
    assert market.loc[4, "pref_source"] == "tk_imputed"
    # Student 6 has no list and no feeder: an EES runs no kindergarten.
    assert market.loc[6, "pref_source"] == "aa_only"
    # Student 7 has no list but does have a feeder.
    assert market.loc[7, "pref_source"] == "feeder_only"


def test_eligibility_reads_the_current_tk_program_not_bypromote(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    assert market["promote_eligible"].to_dict() == {1: 1, 2: 1, 4: 1, 5: 0, 6: 0, 7: 1}
    assert market.loc[2, "feeder_school"] == 420
    assert market.loc[2, "feeder_program"] == "GE"
    # Student 5 was in PK, and student 6 in TK at a school with no
    # kindergarten, so neither holds a claim on anything.
    assert pd.isna(market.loc[5, "feeder_school"])
    assert pd.isna(market.loc[6, "feeder_school"])
    # byPromote would have said otherwise: it is 0 for student 2, who does
    # hold a claim, and 1 for student 7, whose seat it marks after the fact.
    assert market.loc[2, "promote_eligible"] == 1


def test_a_feeder_already_ranked_is_not_appended_twice(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    assert market.loc[1, "r1_ranked_idschool"] == "[413, 420, 999]"
    assert market.loc[1, "r1_programs"] == "['GE', 'GE', 'GE']"
    # The withdrawn rank 3 is still missing, because the list did not change.
    assert market.loc[1, "r1_listed_ranks"] == "[1, 2, 4]"


def test_a_feeder_not_ranked_is_appended_after_the_submitted_choices(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    assert market.loc[2, "r1_ranked_idschool"] == "[999, 420]"
    assert market.loc[2, "r1_programs"] == "['GE', 'GE']"
    assert market.loc[2, "r1_listed_ranks"] == "[1, 2]"
    assert market.loc[2, "num_ranked"] == 2


def test_an_appended_choice_carries_a_lottery_number_that_aligns(
    frames, cleaned_dir, capacities, promotion, transfer
):
    # PriorityGenerator._mtb_real rejects a list whose lottery numbers do not
    # align with it, so an appended choice cannot be left without one. The
    # student's own post-run draw is the number used.
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    assert market.loc[2, "r1_randomnumber"] == "[0.4, 0.8]"
    assert market.loc[2, "r1_cohortstring"] == "['', '']"


def test_an_absent_round_aligned_column_stays_empty_rather_than_partial(
    frames, cleaned_dir, capacities, promotion, transfer
):
    # The loader accepts a round-aligned list that is empty or exactly as long
    # as the ranked list, and rejects a partial one. If the pre-run carries no
    # RandomNumber at all, appending a feeder must not produce a one-entry
    # lottery list against a two-entry ranked list.
    frames["prerun"] = frames["prerun"].drop(columns=["RandomNumber"])
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    assert market.loc[2, "r1_ranked_idschool"] == "[999, 420]"
    assert market.loc[2, "r1_randomnumber"] == "[]"
    # The columns the pre-run does carry still line up.
    assert market.loc[2, "r1_listed_ranks"] == "[1, 2]"
    assert market.loc[2, "r1_cohortstring"] == "['', '']"


def test_the_feeder_school_keeps_its_integer_form_in_the_written_table(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    market = built.table
    applied = converter.apply_market_columns(students, market, grade="KG")
    assert applied["feeder_school"].dtype == "Int64"
    # Not "420.0": every other school column in the table is a bare integer.
    assert str(applied.set_index("studentno").loc[2, "feeder_school"]) == "420"


def test_an_imputed_list_maps_through_the_year_kindergarten_programs(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    # Student 4 ranked 903-GE and 420-GE in TK. 903 is an Early Education
    # School with no kindergarten, so it is dropped rather than guessed at;
    # 420-GE survives, and the feeder 413-GE is appended after it.
    assert market.loc[4, "r1_ranked_idschool"] == "[420, 413]"
    assert market.loc[4, "r1_listed_ranks"] == "[1, 2]"


def test_a_student_with_no_list_gets_their_feeder_and_attendance_area(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    # Student 7: feeder 420-GE first, then the attendance area 413-GE.
    assert market.loc[7, "r1_ranked_idschool"] == "[420, 413]"
    assert market.loc[7, "r1_programs"] == "['GE', 'GE']"
    # Student 6 has no feeder, so only the attendance area.
    assert market.loc[6, "r1_ranked_idschool"] == "[420]"


def test_the_attendance_area_is_not_appended_to_a_student_who_has_a_list(
    frames, cleaned_dir, capacities, promotion, transfer
):
    # For everyone else that is the policy config's job (add_aa_schools).
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    market = built.table
    assert market.loc[2, "r1_ranked_idschool"] == "[999, 420]"
    assert 413 not in [999, 420]


def test_the_report_compares_the_identified_promotes_with_the_district(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    report = built.report
    comparison = report.missing_values["promote_counts_vs_district_KG"]
    # Four students are identified as eligible against the file's 3 + 2 = 5.
    assert comparison["eligible"]["identified"] == 4
    assert comparison["eligible"]["district"] == 5
    # Students 1 and 2 hold a claim and also applied, which is what the file's
    # TotalPromoteWithReqBeforeRun counts.
    assert comparison["eligible_with_application"]["identified"] == 2


def test_the_market_is_written_onto_both_tables_and_no_other_grade(
    frames, cleaned_dir, capacities, promotion, transfer
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    market = built.table
    applied = converter.apply_market_columns(students, market, grade="KG")
    by_id = applied.set_index("studentno")
    assert by_id.loc[2, "r1_ranked_idschool"] == "[999, 420]"
    assert by_id.loc[2, "pref_source"] == "k_list"
    # The grade-6 applicant keeps the list they filed and holds no promotion
    # columns at all.
    assert by_id.loc[3, "r1_ranked_idschool"] == "[404]"
    assert pd.isna(by_id.loc[3, "pref_source"])
    assert set(PROMOTION_STUDENT_COLUMNS) <= set(applied.columns)


def test_the_validator_accepts_the_tables_the_converter_builds(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    enrolled = built.enrolled
    market = built.table
    report = built.report
    converter.validate_market(
        market,
        students,
        enrolled,
        frames["postrun"],
        market=built.market,
        report=report,
    )


def test_the_validator_rejects_a_market_smaller_than_the_student_table(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    # The student table is the applicant pool, so a grade row without a
    # constructed preference list would ship an empty list as if it were one.
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    enrolled = built.enrolled
    market = built.table
    report = built.report
    with pytest.raises(TransferGapError, match="applicant pool"):
        converter.validate_market(
            market.drop(index=6),
            students,
            enrolled,
            frames["postrun"],
            market=built.market,
            report=report,
        )


def test_the_validator_rejects_an_enrolled_row_outside_the_market(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    enrolled = built.enrolled
    market = built.table
    report = built.report
    stranger = pd.concat(
        [enrolled, enrolled.head(1).assign(studentno=9999)], ignore_index=True
    )
    with pytest.raises(TransferGapError, match="not a subset"):
        converter.validate_market(
            market,
            students,
            stranger,
            frames["postrun"],
            market=built.market,
            report=report,
        )


def test_the_validator_rejects_a_k_list_label_without_a_request(
    frames, cleaned_dir, capacities, promotion, transfer, no_geography
):
    # pref_source and mr_applicant say the same thing, and a market whose
    # promoted students were labelled k_list would claim the district has
    # preference lists it does not.
    built = _tables(frames, cleaned_dir, capacities, promotion, transfer)
    students = built.students
    enrolled = built.enrolled
    market = built.table
    report = built.report
    market.loc[4, "pref_source"] = "k_list"
    with pytest.raises(TransferGapError, match="labelled k_list"):
        converter.validate_market(
            market,
            students,
            enrolled,
            frames["postrun"],
            market=built.market,
            report=report,
        )


# --------------------------------------------------------------------------- #
# Dataless placeholder detection
# --------------------------------------------------------------------------- #
def test_require_readable_rejects_an_absent_file(tmp_path):
    with pytest.raises(TransferGapError, match="does not exist"):
        converter.require_readable(tmp_path / "nope.csv", "test input")


def test_require_readable_accepts_a_real_file(tmp_path):
    path = tmp_path / "real.csv"
    path.write_text("a\n1\n", encoding="utf-8")
    converter.require_readable(path, "test input")


def test_dataless_detection_uses_the_size_and_block_count(tmp_path, monkeypatch):
    path = tmp_path / "stub.csv"
    path.write_text("a\n1\n", encoding="utf-8")
    assert not converter.dataless(path)

    real_stat = Path.stat

    class Stub:
        st_flags = converter._SF_DATALESS
        st_size = 1024
        st_blocks = 0

    monkeypatch.setattr(
        Path, "stat", lambda self, *a, **k: Stub() if self == path else real_stat(self)
    )
    assert converter.dataless(path)
    with pytest.raises(TransferGapError, match="no local content"):
        converter.require_readable(path, "test input")

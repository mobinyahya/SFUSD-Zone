"""Tests for the SFUSD transfer converter.

These run on a synthetic three-file transfer, so they need no access to the
shared data root. They pin the behaviour that is easy to get quietly wrong:
the ranked-list pivot, the per-request priority flags, the Mission Bay school
alias, and above all that a gap fails loudly under the default policy instead
of being filled.
"""

from __future__ import annotations

from pathlib import Path

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
from analysis.data_prep.sfusd_transfer_schema import GRADE_BUNDLES

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
PRERUN_ROWS = [
    # Student 1: three KG choices with a gap in the listed ranks (rank 3
    # withdrawn), sibling priority at the second, CTIP1 throughout.
    ("S000000001", 1, 413, "K", "GE", 0.10, "CT1;", None, "CT1"),
    ("S000000001", 2, 420, "K", "GE", 0.20, "CT1;S;", "S", "CT1"),
    ("S000000001", 4, 999, "K", "GE", 0.30, "CT1;", None, "CT1"),
    # Student 2: one KG choice at Mission Bay's new transfer ID.
    ("S000000002", 1, 1731, "K", "GE", 0.40, None, None, None),
    # Student 3: grade 6, so it exercises a second grade bundle.
    ("S000000003", 1, 404, "6", "GE", 0.50, None, None, None),
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
    ("S000000001", 413, 413, "GE", "K", 37.7830, -122.4823, 1, 0.9, "GE", 1.2),
    ("S000000002", None, 999, "GE", "K", 37.7698, -122.3945, 1, 0.8, "GE", 0.4),
    ("S000000003", None, 404, "GE", "06", 37.7500, -122.4000, 1, 0.7, "GE", 2.1),
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
    "CurrentProgramCode",
    "Distance",
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


@pytest.fixture
def transfer(tmp_path: Path) -> Path:
    folder = tmp_path / "SY26-27"
    folder.mkdir()
    pd.DataFrame(PRERUN_ROWS, columns=PRERUN_COLUMNS).to_csv(
        folder / "out_SY26-27_PreRun.csv", index=False
    )
    pd.DataFrame(POSTRUN_ROWS, columns=POSTRUN_COLUMNS).to_csv(
        folder / "out_SY26-27_PostRun.csv", index=False
    )
    pd.DataFrame(DEMOGRAPHICS_ROWS, columns=DEMOGRAPHICS_COLUMNS).to_csv(
        folder / "out_SY26-27_Student Demographics.csv", index=False
    )
    return tmp_path


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
    assert sorted(students.index) == [1, 2, 3]


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
    frames["prerun"]["MSF"] = [None, None, None, None, None]
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
    folder = tmp_path / "cleaned"
    folder.mkdir()
    bundle = GRADE_BUNDLES["KG"]
    pd.DataFrame(
        {
            "program_id": ["413-GE-KG", "420-GE-KG"],
            "school_id": [413, 420],
            "program_type": ["GE", "GE"],
            "capacity": [64, 88],
        }
    ).to_csv(folder / bundle.capacity_reference, index=False)
    for name in (bundle.schools_standard, bundle.schools_mission_bay):
        pd.DataFrame(
            {
                "school_id": [413, 420, 999],
                "lat": [37.783, 37.7537, 37.7698],
                "lon": [-122.4823, -122.4382, -122.3945],
            }
        ).to_csv(folder / name, index=False)
    return folder


def _programs(frames, cleaned_dir, *, gaps: str, report: Report | None = None):
    return build_program_table(
        frames["prerun"],
        frames["postrun"],
        GRADE_BUNDLES["KG"],
        cleaned_dir=cleaned_dir,
        include_mission_bay=True,
        gaps=gaps,
        report=report or Report(year="2627", transfer="synthetic"),
    )


def test_missing_capacity_fails_under_the_default_policy(frames, cleaned_dir):
    # Mission Bay's KG program is in the requests but not in the capacity
    # reference, which is exactly the real 2026-27 situation.
    with pytest.raises(TransferGapError, match="no district capacity exists"):
        _programs(frames, cleaned_dir, gaps="fail")


def test_missing_capacity_falls_back_to_the_observed_count_and_is_labelled(
    frames, cleaned_dir
):
    report = Report(year="2627", transfer="synthetic")
    programs = _programs(frames, cleaned_dir, gaps="fill-and-report", report=report)
    by_id = programs.set_index("program_id")
    assert by_id.loc["413-GE-KG", "capacity"] == 64
    assert by_id.loc["413-GE-KG", "capacity_source"] == (
        GRADE_BUNDLES["KG"].capacity_reference
    )
    # One student was assigned to Mission Bay in the synthetic post-run.
    assert by_id.loc["999-GE-KG", "capacity"] == 1
    assert "observed" in by_id.loc["999-GE-KG", "capacity_source"]
    kinds = {item["kind"] for item in report.substitutions}
    assert "program_capacity" in kinds


def test_program_numbers_are_unique_and_contiguous(frames, cleaned_dir):
    programs = _programs(frames, cleaned_dir, gaps="fill-and-report")
    assert sorted(programs["programno"]) == list(range(1, len(programs) + 1))
    assert not programs["program_id"].duplicated().any()


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
